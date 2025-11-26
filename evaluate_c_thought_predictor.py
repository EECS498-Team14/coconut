import argparse
import json
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
    n_latents_prefix: int,
) -> torch.Tensor:
    """
    Get the last hidden state of the prefix that includes n_latents_prefix latent tokens.
    This corresponds to the state right before the next latent token.
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
    return outputs.hidden_states[-1][:, -1, :].squeeze(0)


def build_input(question: str, num_latent_tokens: int, tokenizer, special_tokens: Dict[str, int]):
    question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)
    if num_latent_tokens > 0:
        input_ids = (
            question_tokens
            + [special_tokens["start_id"]]
            + [special_tokens["latent_id"]] * num_latent_tokens
            + [special_tokens["end_id"]]
        )
    else:
        input_ids = question_tokens

    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
        "position_ids": torch.tensor([position_ids]),
    }


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


def extract_answer(generated_text: str) -> str:
    if "#" in generated_text:
        answer = generated_text.split("#")[-1].strip()
    else:
        answer = generated_text.strip()
    return answer.replace(",", "").strip()


def evaluate(
    data_path: str,
    coconut_ckpt: str,
    model_id: str,
    predictor_path: str,
    max_new_tokens: int,
    max_questions: int,
    device: str,
    baseline_steps: int,
    max_c_thoughts: int,
):
    coconut_model, tokenizer, special_tokens = load_coconut_model(coconut_ckpt, model_id, device)
    predictor, token_choices = load_predictor(predictor_path, device)
    token_choices = list(token_choices)

    with open(data_path, "r") as f:
        dataset = json.load(f)
    if max_questions:
        dataset = dataset[:max_questions]

    correct_pred = 0
    correct_base = 0
    total = 0

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        answer = item["answer"].replace(",", "").strip()
        question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)

        # iterative prediction of remaining steps
        k_latents = 0
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
                pred_remaining = predictor(hs.unsqueeze(0)).item()
                pred_remaining_int = max(0, int(round(pred_remaining)))
            if pred_remaining_int <= 1:
                break
            k_latents += 1

        num_tokens = k_latents

        # run coconut with predicted tokens
        inputs = build_input(question, num_tokens, tokenizer, special_tokens)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = coconut_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        pred_answer = extract_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))

        correct_pred += pred_answer == answer

        # baseline: fixed number of latent steps
        inputs_base = build_input(question, baseline_steps, tokenizer, special_tokens)
        inputs_base = {k: v.to(device) for k, v in inputs_base.items()}
        with torch.no_grad():
            outputs_base = coconut_model.generate(
                input_ids=inputs_base["input_ids"],
                attention_mask=inputs_base["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        base_answer = extract_answer(tokenizer.decode(outputs_base[0], skip_special_tokens=True))
        correct_base += base_answer == answer

        total += 1

    acc_pred = correct_pred / max(1, total)
    acc_base = correct_base / max(1, total)
    print(f"Predictor Accuracy: {correct_pred}/{total} = {acc_pred:.4f}")
    print(f"Baseline  Accuracy: {correct_base}/{total} = {acc_base:.4f} (fixed steps = {baseline_steps})")
    return acc_pred, acc_base


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive c_thoughts predictor")
    parser.add_argument("--data_path", type=str, default="data/gsm_valid.json")
    parser.add_argument(
        "--coconut_ckpt",
        type=str,
        default="pretrained_checkpoints/stage_1_training_ck/checkpoint_5",
    )
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--predictor_path", type=str, default="c_thought_predictor.pt")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--baseline_steps", type=int, default=2, help="Fixed latent steps for baseline comparison")
    parser.add_argument("--max_c_thoughts", type=int, default=10, help="Ceiling for iterative latent generation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate(
        data_path=args.data_path,
        coconut_ckpt=args.coconut_ckpt,
        model_id=args.model_id,
        predictor_path=args.predictor_path,
        max_new_tokens=args.max_new_tokens,
        max_questions=args.max_questions,
        device=device,
        baseline_steps=args.baseline_steps,
        max_c_thoughts=args.max_c_thoughts,
    )


if __name__ == "__main__":
    main()
