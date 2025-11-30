"""
Evaluate multi-target predictor on test set.

Tests multiple inference strategies:
1. Argmax: pick most likely token count
2. Conservative: pick highest token count with prob > threshold
3. Sample: sample from distribution
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from coconut import Coconut
from proof_of_concept_experiment import (
    evaluate_single_question,
    load_coconut_model,
    build_input_with_tokens,
)

from forward_predictor import create_multi_target_predictor


def get_h0_hidden_state(
    model, tokenizer, question: str, special_tokens: dict, device: str
) -> torch.Tensor:
    """Get h0 (hidden state after question, before any latent reasoning)."""
    inputs = build_input_with_tokens(
        question=question,
        num_latent_tokens=0,
        tokenizer=tokenizer,
        **special_tokens,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.base_causallm(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        h0 = outputs.hidden_states[-1][:, -1, :].squeeze(0).cpu()

    return h0


def evaluate_multi_target_predictor(
    predictor_checkpoint: str,
    coconut_checkpoint: str,
    test_data_path: str,
    strategy: str = "argmax",
    threshold: float = 0.3,
    temperature: float = 1.0,
    max_questions: int = None,
    device: str = "cuda",
    model_id: str = "openai-community/gpt2",
):
    """
    Evaluate multi-target predictor on test set.
    """
    print("=" * 80)
    print("EVALUATING MULTI-TARGET PREDICTOR")
    print("=" * 80)

    # Load predictor
    print(f"\nLoading predictor from {predictor_checkpoint}...")
    checkpoint = torch.load(predictor_checkpoint, map_location="cpu")

    config = checkpoint["config"]
    predictor = create_multi_target_predictor(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        hidden_layers=config.get("hidden_layers", [1024, 512]),
        dropout=config.get("dropout", 0.3),
        architecture=config.get("architecture", "mlp"),
        hidden_dim=config.get("hidden_dim", 512),
        num_blocks=config.get("num_blocks", 6),
        num_tokens=config.get("num_tokens", 12),
        d_model=config.get("d_model", 64),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 256),
    ).to(device)

    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val 'any correct' accuracy: {checkpoint['metrics']['val_any_correct']:.2%}")
    print(f"  Inference strategy: {strategy}")
    if strategy == "conservative":
        print(f"  Threshold: {threshold}")
    elif strategy == "sample":
        print(f"  Temperature: {temperature}")
    elif strategy == "ensemble":
        print(f"  Confidence threshold: {threshold}")
        print(f"  Fallback: always 6 (baseline)")

    # Load coconut model
    print(f"\nLoading coconut model from {coconut_checkpoint}...")
    coconut_model = load_coconut_model(coconut_checkpoint, model_id=model_id, device=device)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    special_tokens = {
        "start_id": tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        "latent_id": tokenizer.convert_tokens_to_ids("<|latent|>"),
        "end_id": tokenizer.convert_tokens_to_ids("<|end-latent|>"),
    }

    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    if max_questions:
        test_data = test_data[:max_questions]

    print(f"Test questions: {len(test_data)}\n")

    # Evaluate
    results = {
        "correct": 0,
        "total": len(test_data),
        "predicted_steps": [],
        "step_distribution": {},
    }

    for question_data in tqdm(test_data, desc="Evaluating"):
        question = question_data["question"]
        ground_truth = question_data["answer"].replace(",", "").strip()

        # Get h0 hidden state
        h0 = get_h0_hidden_state(
            coconut_model, tokenizer, question, special_tokens, device
        )

        # Predict token count using specified strategy
        h0_batch = h0.unsqueeze(0).to(device)

        with torch.no_grad():
            if strategy == "argmax":
                predicted_step = predictor.predict_argmax(h0_batch)[0].item()
            elif strategy == "conservative":
                predicted_step = predictor.predict_conservative(h0_batch, threshold=threshold)[0].item()
            elif strategy == "sample":
                predicted_step = predictor.predict_sample(h0_batch, temperature=temperature)[0].item()
            elif strategy == "ensemble":
                predicted_step = predictor.predict_ensemble(h0_batch, confidence_threshold=threshold)[0].item()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        results["predicted_steps"].append(predicted_step)

        # Clamp to valid range [0, 6]
        num_tokens = max(0, min(6, int(predicted_step)))

        # Update distribution
        if num_tokens not in results["step_distribution"]:
            results["step_distribution"][num_tokens] = 0
        results["step_distribution"][num_tokens] += 1

        # Evaluate with coconut
        is_correct, predicted_answer = evaluate_single_question(
            question=question,
            ground_truth=ground_truth,
            num_latent_tokens=num_tokens,
            model=coconut_model,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            max_new_tokens=64,
        )

        if is_correct:
            results["correct"] += 1

    # Calculate metrics
    accuracy = results["correct"] / results["total"]

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nAccuracy: {results['correct']}/{results['total']} = {accuracy:.2%}")

    print("\nPredicted step distribution:")
    for step in sorted(results["step_distribution"].keys()):
        count = results["step_distribution"][step]
        pct = count / results["total"] * 100
        print(f"  {step} tokens: {count:4d} ({pct:5.1f}%)")

    print(f"\n{'Multi-target predictor':<20} {accuracy:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-target predictor")
    parser.add_argument(
        "--predictor_checkpoint",
        type=str,
        default="multi_target_predictor.pt",
        help="Path to predictor checkpoint",
    )
    parser.add_argument(
        "--coconut_checkpoint",
        type=str,
        default="../pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
        help="Path to coconut checkpoint",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../data/gsm_test.json",
        help="Path to test data",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="argmax",
        choices=["argmax", "conservative", "sample", "ensemble"],
        help="Inference strategy (ensemble = use predictor if confident, else fallback to 6)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for conservative/ensemble strategy (ensemble default: 0.7)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling strategy",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Max questions to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID (e.g., openai-community/gpt2, meta-llama/Llama-2-7b-hf)",
    )

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    evaluate_multi_target_predictor(
        predictor_checkpoint=args.predictor_checkpoint,
        coconut_checkpoint=args.coconut_checkpoint,
        test_data_path=args.test_data,
        strategy=args.strategy,
        threshold=args.threshold,
        temperature=args.temperature,
        max_questions=args.max_questions,
        device=device,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
