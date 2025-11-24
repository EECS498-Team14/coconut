import argparse
import json
from collections import Counter
from typing import Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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
    model.load_state_dict(saved_weights, strict=False)

    model = model.to(device)
    model.eval()
    return model, tokenizer, {"start_id": start_id, "latent_id": latent_id, "end_id": end_id}


def get_prefix_hidden_state(
    model,
    tokenizer,
    question_tokens: list,
    special_tokens: Dict[str, int],
    device: str,
    n_latents_prefix: int = 0,
) -> torch.Tensor:
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
    return outputs.hidden_states[-1][:, -1, :].squeeze(0)


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
        layers.append(nn.Linear(prev_dim, 1))
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
        seq_len = self.max_seq_len
        h = x.unsqueeze(1).repeat(1, seq_len, 1)
        h = self.input_proj(h)
        positions = torch.arange(seq_len, device=h.device)
        h = h + self.pos_emb(positions)[None, :, :]
        h = self.encoder(h)
        pooled = h[:, 0, :]
        return self.head(pooled).squeeze(-1)


def load_predictor(predictor_path: str, device: str):
    ckpt = torch.load(predictor_path, map_location=device)
    token_choices = ckpt["token_choices"]
    cfg = ckpt.get(
        "predictor_config",
        {
            "model_type": "mlp",
            "hidden_layers": [512, 512],
            "dropout": 0.1,
            "d_model": 256,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 512,
            "max_seq_len": 1,
        },
    )
    model_type = cfg.get("model_type", "mlp")
    if model_type == "mlp":
        model = MLPPredictor(
            input_dim=ckpt["input_dim"],
            hidden_layers=cfg.get("hidden_layers", [512, 512]),
            dropout=cfg.get("dropout", 0.1),
        ).to(device)
    elif model_type == "transformer":
        model = TransformerPredictor(
            input_dim=ckpt["input_dim"],
            d_model=cfg.get("d_model", 256),
            num_layers=cfg.get("num_layers", 2),
            num_heads=cfg.get("num_heads", 4),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.1),
            max_seq_len=cfg.get("max_seq_len", 1),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, token_choices


def analyze_predictions(
    data_path: str,
    coconut_ckpt: str,
    model_id: str,
    predictor_path: str,
    max_questions: int,
    device: str,
    output_path: str = None,
    max_c_thoughts: int = 10,
):
    coconut_model, tokenizer, special_tokens = load_coconut_model(coconut_ckpt, model_id, device)
    predictor, token_choices = load_predictor(predictor_path, device)
    token_choices = list(token_choices)

    with open(data_path, "r") as f:
        dataset = json.load(f)
    if max_questions:
        dataset = dataset[:max_questions]

    pred_values = []
    nearest_tokens = []
    actual_steps_used = []
    for item in tqdm(dataset, desc="Collecting predictor outputs"):
        question = item["question"]
        question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)

        # iterative prediction of remaining steps
        k_latents = 0
        step_pred_values = []
        while k_latents < max_c_thoughts:
            with torch.no_grad():
                hs = get_prefix_hidden_state(
                    coconut_model,
                    tokenizer,
                    question_tokens,
                    special_tokens,
                    device,
                    n_latents_prefix=k_latents,
                )
                val = predictor(hs.unsqueeze(0)).item()
            step_pred_values.append(val)
            if val <= 0:
                break
            k_latents += 1

        # record last prediction and nearest token for the first step
        if step_pred_values:
            pred_values.append(step_pred_values[0])
            nearest_tokens.append(min(token_choices, key=lambda t: abs(t - step_pred_values[0])))
        actual_steps_used.append(k_latents)

    token_counter = Counter(nearest_tokens)
    mean_pred = sum(pred_values) / len(pred_values) if pred_values else 0.0
    min_pred = min(pred_values) if pred_values else 0.0
    max_pred = max(pred_values) if pred_values else 0.0
    steps_counter = Counter(actual_steps_used)
    mean_steps = sum(actual_steps_used) / len(actual_steps_used) if actual_steps_used else 0.0

    print(f"Total samples: {len(pred_values)}")
    print(f"Pred value mean: {mean_pred:.3f}, min: {min_pred:.3f}, max: {max_pred:.3f}")
    print("Nearest-token distribution:")
    for k in sorted(token_counter.keys()):
        cnt = token_counter[k]
        pct = cnt / max(1, len(nearest_tokens)) * 100
        print(f"  {k}: {cnt} ({pct:.2f}%)")
    print("Actual steps used distribution:")
    for k in sorted(steps_counter.keys()):
        cnt = steps_counter[k]
        pct = cnt / max(1, len(actual_steps_used)) * 100
        print(f"  {k}: {cnt} ({pct:.2f}%)")
    print(f"Actual steps mean: {mean_steps:.3f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(
                {
                    "pred_values": pred_values,
                    "nearest_token_counts": dict(token_counter),
                    "token_choices": token_choices,
                    "stats": {"mean": mean_pred, "min": min_pred, "max": max_pred},
                    "steps_used": actual_steps_used,
                    "steps_counts": dict(steps_counter),
                    "steps_mean": mean_steps,
                },
                f,
                indent=2,
            )
        print(f"Saved analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze predictor output distribution")
    parser.add_argument("--data_path", type=str, default="data/gsm_valid.json")
    parser.add_argument(
        "--coconut_ckpt",
        type=str,
        default="pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
    )
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--predictor_path", type=str, default="c_thought_predictor.pt")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="predictor_bias_report.json")
    parser.add_argument("--max_c_thoughts", type=int, default=10, help="Ceiling for iterative latent generation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyze_predictions(
        data_path=args.data_path,
        coconut_ckpt=args.coconut_ckpt,
        model_id=args.model_id,
        predictor_path=args.predictor_path,
        max_questions=args.max_questions,
        device=device,
        output_path=args.output_path,
        max_c_thoughts=args.max_c_thoughts,
    )


if __name__ == "__main__":
    main()
