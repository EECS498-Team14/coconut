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
    question: str,
    special_tokens: Dict[str, int],
    device: str,
) -> torch.Tensor:
    question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)
    input_ids = question_tokens + [special_tokens["start_id"]]
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


class SimplePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
    generation. For each question with a valid optimal_tokens >= 0, extract the
    prefix hidden state and pair it with the oracle label.
    """
    model, tokenizer, special_tokens = load_coconut_model(checkpoint_path, model_id, device)

    with open(poc_results_path, "r") as f:
        poc_data = json.load(f)

    token_choices = [int(t) for t in poc_data["experiment_config"]["token_choices"]]
    question_results = poc_data["checkpoint_12_results"]["question_results"]
    if max_questions:
        question_results = question_results[:max_questions]

    hidden_states = []
    labels = []

    for item in tqdm(question_results, desc="Collecting hidden states"):
        optimal_tokens = item.get("optimal_tokens", -1)
        if optimal_tokens is None or optimal_tokens < 0:
            continue  # skip if no oracle label
        if optimal_tokens not in token_choices:
            continue  # skip tokens that are outside known choices

        question = item["question"]
        hidden_state = get_prefix_hidden_state(model, tokenizer, question, special_tokens, device)
        hidden_states.append(hidden_state)
        labels.append(optimal_tokens)

    if len(hidden_states) == 0:
        raise RuntimeError("No training samples collected (no valid oracle labels).")

    token_to_idx = {t: i for i, t in enumerate(sorted(set(token_choices)))}
    label_indices = [token_to_idx[l] for l in labels]

    torch.save(
        {
            "hidden_states": torch.stack(hidden_states),
            "label_indices": torch.tensor(label_indices, dtype=torch.long),
            "token_mapping": token_to_idx,
            "token_choices": sorted(set(token_choices)),
        },
        output_path,
    )
    print(f"Saved {len(hidden_states)} samples to {output_path}")
    return output_path


def train_predictor(
    data_path: str,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    num_epochs: int,
    output_path: str,
    val_ratio: float,
):
    data = torch.load(data_path)
    hidden_states = data["hidden_states"]
    labels = data["label_indices"]
    token_choices = data["token_choices"]

    # split train/val
    total_n = len(hidden_states)
    n_val = max(1, int(total_n * val_ratio))
    n_train = total_n - n_val
    if n_train <= 0:
        raise RuntimeError("Not enough samples to create a validation split.")

    train_states, val_states = torch.split(hidden_states, [n_train, n_val])
    train_labels, val_labels = torch.split(labels, [n_train, n_val])

    train_loader = DataLoader(
        TensorDataset(train_states, train_labels), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_states, val_labels), batch_size=batch_size, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimplePredictor(
        input_dim=hidden_states.shape[1],
        hidden_dim=hidden_dim,
        num_classes=len(token_choices),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_states, batch_labels in train_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_states)
            loss = criterion(logits, batch_labels)

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
            for batch_states, batch_labels in val_loader:
                batch_states = batch_states.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_states)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.numel()
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
            "hidden_dim": hidden_dim,
        },
        output_path,
    )
    print(f"Saved predictor to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train predictor for adaptive c_thoughts")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
    )
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--poc_results", type=str, default="poc_experiment_results.json")
    parser.add_argument("--max_questions", type=int, default=10000)
    parser.add_argument("--collection_output", type=str, default="c_thought_dataset.pt")
    parser.add_argument("--predictor_hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--predictor_output", type=str, default="c_thought_predictor.pt")
    parser.add_argument("--val_ratio", type=float, default=0.1)

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
        hidden_dim=args.predictor_hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        output_path=args.predictor_output,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
