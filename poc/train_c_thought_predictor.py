import argparse
import json
import os
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut


def load_coconut_model(checkpoint_path: str, model_id: str, device: str):
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_model.resize_token_embeddings(len(tokenizer))
    embeddings = base_model.get_input_embeddings()
    output_embeddings = base_model.get_output_embeddings()

    target_id = tokenizer.convert_tokens_to_ids("<<")
    target_embedding = embeddings.weight.data[target_id]
    target_output = output_embeddings.weight.data[target_id]

    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = target_embedding
        output_embeddings.weight.data[token_id] = target_output

    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    saved_weights = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(saved_weights, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"[warning] load_result = {load_result}")

    model = model.to(device)
    model.eval()
    return model, tokenizer, {"start_id": start_id, "latent_id": latent_id, "end_id": end_id}


def get_prefix_hidden_state(
    model,
    tokenizer,
    question_tokens: list,
    special_tokens: Dict[str, int],
    device: str,
    n_latents_prefix: int,
) -> torch.Tensor:
    """
    Get last hidden state for prefix that includes `n_latents_prefix` latent tokens.
    This corresponds to the position right before the (n_latents_prefix+1)-th latent.
    """
    input_ids = (
        question_tokens
        + [special_tokens["start_id"]]
        + [special_tokens["latent_id"]] * n_latents_prefix
    )
    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    with torch.no_grad():
        outputs = model.base_causallm(
            input_ids=torch.tensor([input_ids], device=device),
            attention_mask=torch.tensor([attention_mask], device=device),
            position_ids=torch.tensor([position_ids], device=device),
            output_hidden_states=True,
        )
    return outputs.hidden_states[-1][:, -1, :].squeeze(0).cpu()


class MLPPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int = 1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # repeat to seq_len to keep positional logic extensible
        seq_len = self.max_seq_len
        h = x.unsqueeze(1).repeat(1, seq_len, 1)
        h = self.input_proj(h)
        positions = torch.arange(seq_len, device=h.device)
        h = h + self.pos_emb(positions)[None, :, :]
        h = self.encoder(h)
        pooled = h[:, 0, :]
        return self.head(pooled).squeeze(-1)


def collect_hidden_states_and_labels(
    checkpoint_path: str,
    model_id: str,
    poc_results_path: str,
    max_questions: int,
    device: str,
    output_path: str,
):
    """
    Use existing oracle results (poc_experiment_results.json) to avoid re-running
    generation. For each question with valid oracle info, extract multiple
    (hidden_state, c_thoughts_left) pairs along the latent sequence.

    Strategy:
    - Define optimal_total = average of correct tokens (if multiple).
    - For each integer step k in [0, floor(optimal_total)-1], get hidden state after k latents
      (i.e., before latent k+1) and pair with label = optimal_total - k.
    """
    model, tokenizer, special_tokens = load_coconut_model(checkpoint_path, model_id, device)

    with open(poc_results_path, "r") as f:
        poc_data = json.load(f)

    token_choices = [int(t) for t in poc_data["experiment_config"]["token_choices"]]
    question_results = poc_data["checkpoint_12_results"]["question_results"]
    if max_questions:
        question_results = question_results[:max_questions]

    hidden_states = []
    all_targets = []  # Will store lists of targets for each sample

    for item in tqdm(question_results, desc="Collecting hidden states"):
        token_results = item.get("token_results", {})
        correct_tokens = sorted(
            int(k) for k, v in token_results.items() if isinstance(v, dict) and v.get("correct", False)
        )
        if len(correct_tokens) == 0:
            continue  # skip if no oracle label

        optimal_total = sum(correct_tokens) / len(correct_tokens)  # average of valid tokens
        question_tokens = tokenizer.encode(item["question"] + "\n", add_special_tokens=True)

        max_int_step = int(optimal_total)
        for k in range(max_int_step):
            # Store all correct lengths shifted by k
            targets_at_k = [float(c - k) for c in correct_tokens]
            hidden_state = get_prefix_hidden_state(
                model, tokenizer, question_tokens, special_tokens, device, n_latents_prefix=k
            )
            hidden_states.append(hidden_state)
            all_targets.append(targets_at_k)

    if len(hidden_states) == 0:
        raise RuntimeError("No training samples collected (no valid oracle labels).")

    # Pad all_targets to the same length
    max_num_targets = max(len(targets) for targets in all_targets)
    padded_targets = []
    masks = []
    for targets in all_targets:
        num_valid = len(targets)
        padded = targets + [0.0] * (max_num_targets - num_valid)  # pad with 0
        mask = [1.0] * num_valid + [0.0] * (max_num_targets - num_valid)
        padded_targets.append(padded)
        masks.append(mask)

    torch.save(
        {
            "hidden_states": torch.stack(hidden_states),
            "targets": torch.tensor(padded_targets, dtype=torch.float32),
            "masks": torch.tensor(masks, dtype=torch.float32),
            "token_choices": sorted(set(token_choices)),
        },
        output_path,
    )
    print(f"Saved {len(hidden_states)} samples to {output_path}")
    return output_path


def multi_target_mse_loss(preds, targets, masks):
    """
    Compute MSE loss as mean squared distance to all correct targets.

    For example, if correct lengths are [2, 3], the loss for prediction x is:
    0.5 * ((x-2)^2 + (x-3)^2)

    Args:
        preds: (batch_size,) - predicted values
        targets: (batch_size, num_targets) - multiple target values per sample (padded)
        masks: (batch_size, num_targets) - 1.0 for valid targets, 0.0 for padding

    Returns:
        Scalar loss value
    """
    # Expand preds to match targets shape: (batch_size, num_targets)
    preds_expanded = preds.unsqueeze(1).expand_as(targets)

    # Compute squared distances: (batch_size, num_targets)
    squared_dists = (preds_expanded - targets) ** 2

    # Apply mask and compute mean over valid targets for each sample
    masked_squared_dists = squared_dists * masks
    num_valid_per_sample = masks.sum(dim=1)  # (batch_size,)
    mean_per_sample = masked_squared_dists.sum(dim=1) / num_valid_per_sample  # (batch_size,)

    # Return mean over all samples
    return mean_per_sample.mean()


def train_predictor(
    data_path: str,
    batch_size: int,
    lr: float,
    num_epochs: int,
    output_path: str,
    val_ratio: float,
    config_path: str,
):
    data = torch.load(data_path)
    hidden_states = data["hidden_states"]
    targets = data.get("targets")
    masks = data.get("masks")
    if targets is None or masks is None:
        raise RuntimeError("Expected 'targets' and 'masks' in dataset.")
    token_choices = data["token_choices"]

    # split train/val
    total_n = len(hidden_states)
    n_val = max(1, int(total_n * val_ratio))
    n_train = total_n - n_val
    if n_train <= 0:
        raise RuntimeError("Not enough samples to create a validation split.")

    perm = torch.randperm(total_n)
    hidden_states = hidden_states[perm]
    targets = targets[perm]
    masks = masks[perm]

    train_states, val_states = torch.split(hidden_states, [n_train, n_val])
    train_targets, val_targets = torch.split(targets, [n_train, n_val])
    train_masks, val_masks = torch.split(masks, [n_train, n_val])

    train_loader = DataLoader(
        TensorDataset(train_states, train_targets, train_masks), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_states, val_targets, val_masks), batch_size=batch_size, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_cfg = load_predictor_config(config_path)
    model_type = predictor_cfg.get("model_type", "mlp")
    if model_type == "mlp":
        model = MLPPredictor(
            input_dim=hidden_states.shape[1],
            hidden_layers=predictor_cfg.get("hidden_layers", [512, 512]),
            dropout=predictor_cfg.get("dropout", 0.1),
        ).to(device)
    elif model_type == "transformer":
        model = TransformerPredictor(
            input_dim=hidden_states.shape[1],
            d_model=predictor_cfg.get("d_model", 256),
            num_layers=predictor_cfg.get("num_layers", 2),
            num_heads=predictor_cfg.get("num_heads", 4),
            dim_feedforward=predictor_cfg.get("dim_feedforward", 512),
            dropout=predictor_cfg.get("dropout", 0.1),
            max_seq_len=predictor_cfg.get("max_seq_len", 1),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_states, batch_targets, batch_masks in train_loader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)
            batch_masks = batch_masks.to(device)

            preds = model(batch_states)
            loss = multi_target_mse_loss(preds, batch_targets, batch_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_states, batch_targets, batch_masks in val_loader:
                batch_states = batch_states.to(device)
                batch_targets = batch_targets.to(device)
                batch_masks = batch_masks.to(device)
                preds = model(batch_states)
                loss = multi_target_mse_loss(preds, batch_targets, batch_masks)
                val_loss += loss.item()

                # For validation accuracy, compare against mean of targets
                target_means = (batch_targets * batch_masks).sum(dim=1) / batch_masks.sum(dim=1)
                preds_rounded = preds.round()
                correct += (preds_rounded == target_means.round()).sum().item()
                total += batch_targets.size(0)
        val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - train_loss: {avg_loss:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )
        model.train()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "token_choices": token_choices,
            "input_dim": hidden_states.shape[1],
            "predictor_config": predictor_cfg,
        },
        output_path,
    )
    print(f"Saved predictor to {output_path}")
    return output_path


def load_predictor_config(config_path: str) -> Dict:
    default_cfg = {
        "model_type": "mlp",
        "hidden_layers": [512, 512],
        "dropout": 0.1,
        # transformer-specific defaults
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 4,
        "dim_feedforward": 512,
        "max_seq_len": 1,
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return {**default_cfg, **cfg}
    return default_cfg


def main():
    parser = argparse.ArgumentParser(description="Train transformer predictor for adaptive c_thoughts")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
    )
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--poc_results", type=str, default="poc_experiment_results.json")
    parser.add_argument("--max_questions", type=int, default=10000)
    parser.add_argument("--collection_output", type=str, default="c_thought_dataset.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--predictor_output", type=str, default="c_thought_predictor.pt")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--predictor_config", type=str, default="predictor_config.json")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load existing dataset if present
    if os.path.exists(args.collection_output):
        print(f"Found existing collection file at {args.collection_output}, skipping recollection.")
        dataset_path = args.collection_output
    else:
        dataset_path = collect_hidden_states_and_labels(
            checkpoint_path=args.checkpoint_path,
            model_id=args.model_id,
            poc_results_path=args.poc_results,
            max_questions=args.max_questions,
            device=device,
            output_path=args.collection_output,
        )

    train_predictor(
        data_path=dataset_path,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        output_path=args.predictor_output,
        val_ratio=args.val_ratio,
        config_path=args.predictor_config,
    )


if __name__ == "__main__":
    main()
