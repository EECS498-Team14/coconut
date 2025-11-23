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
    def __init__(self, input_dim: int, num_classes: int, hidden_layers: list, dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_predictor(predictor_path: str, device: str):
    ckpt = torch.load(predictor_path, map_location=device)
    token_choices = ckpt["token_choices"]
    cfg = ckpt.get(
        "predictor_config",
        {
            "model_type": "mlp",
            "hidden_layers": [512, 512],
            "dropout": 0.1,
        },
    )
    if cfg.get("model_type", "mlp") != "mlp":
        raise ValueError(f"Unsupported model_type in checkpoint: {cfg.get('model_type')}")
    model = MLPPredictor(
        input_dim=ckpt["input_dim"],
        num_classes=len(token_choices),
        hidden_layers=cfg.get("hidden_layers", [512, 512]),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
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
):
    coconut_model, tokenizer, special_tokens = load_coconut_model(coconut_ckpt, model_id, device)
    predictor, token_choices = load_predictor(predictor_path, device)
    token_choices = list(token_choices)

    with open(data_path, "r") as f:
        dataset = json.load(f)
    if max_questions:
        dataset = dataset[:max_questions]

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        answer = item["answer"].replace(",", "").strip()

        # predict optimal tokens
        with torch.no_grad():
            hs = get_prefix_hidden_state(coconut_model, tokenizer, question, special_tokens, device)
            logits = predictor(hs.unsqueeze(0))
            pred_idx = logits.argmax(dim=-1).item()
            num_tokens = token_choices[pred_idx]

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

        correct += pred_answer == answer
        total += 1

    acc = correct / max(1, total)
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive c_thoughts predictor")
    parser.add_argument("--data_path", type=str, default="data/gsm_valid.json")
    parser.add_argument(
        "--coconut_ckpt",
        type=str,
        default="pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
    )
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--predictor_path", type=str, default="c_thought_predictor.pt")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_questions", type=int, default=None)
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
    )


if __name__ == "__main__":
    main()
