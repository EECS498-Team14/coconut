"""
Evaluate baseline curve: accuracy vs token count for fixed token counts.

Evaluates Coconut checkpoint 12 with all fixed token counts [0,1,2,3,4,5,6].
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
)


def evaluate_baseline_curve(
    coconut_checkpoint: str,
    test_data_path: str,
    output_path: str,
    token_counts: list = [0, 1, 2, 3, 4, 5, 6],
    device: str = "cuda",
    model_id: str = "openai-community/gpt2",
):
    """
    Evaluate baseline curve with fixed token counts.

    Args:
        coconut_checkpoint: Path to Coconut checkpoint
        test_data_path: Path to test data JSON file
        output_path: Path to save results JSON
        token_counts: List of fixed token counts to evaluate
        device: Device to run on
    """
    print("=" * 80)
    print("EVALUATING BASELINE CURVE (FIXED TOKEN COUNTS)")
    print("=" * 80)

    # Load model
    print(f"\nLoading Coconut model from {coconut_checkpoint}...")
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

    print(f"Test questions: {len(test_data)}")

    results = {
        "baseline_type": "fixed_tokens",
        "coconut_checkpoint": coconut_checkpoint,
        "test_data": test_data_path,
        "total_questions": len(test_data),
        "points": []
    }

    # Evaluate each fixed token count
    print(f"\n{'=' * 80}")
    print("EVALUATING EACH FIXED TOKEN COUNT")
    print(f"{'=' * 80}\n")

    for num_tokens in token_counts:
        print(f"Evaluating with {num_tokens} tokens...")
        correct = 0

        for question_data in tqdm(test_data, desc=f"Tokens={num_tokens}"):
            question = question_data["question"]
            ground_truth = question_data["answer"].replace(",", "").strip()

            is_correct, _ = evaluate_single_question(
                question=question,
                ground_truth=ground_truth,
                num_latent_tokens=num_tokens,
                model=coconut_model,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                max_new_tokens=64,
            )

            if is_correct:
                correct += 1

        accuracy = correct / len(test_data)

        results["points"].append({
            "num_tokens": num_tokens,
            "avg_tokens": float(num_tokens),  # Same as num_tokens for fixed
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_data),
        })

        print(f"  Tokens={num_tokens}: {correct}/{len(test_data)} = {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    # Save results
    print(f"{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}\n")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("BASELINE CURVE SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Token Count':<12} {'Accuracy':<10} {'Correct/Total':<15}")
    print("-" * 37)
    for point in results["points"]:
        print(
            f"{point['num_tokens']:<12} "
            f"{point['accuracy']:<10.4f} "
            f"{point['correct']}/{point['total']}"
        )

    print(f"\n{'=' * 80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline curve with fixed token counts"
    )
    parser.add_argument(
        "--coconut_checkpoint",
        type=str,
        default="../pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
        help="Path to Coconut checkpoint",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../data/gsm_test.json",
        help="Path to test data JSON",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="baseline_curve.json",
        help="Path to save results JSON",
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

    evaluate_baseline_curve(
        coconut_checkpoint=args.coconut_checkpoint,
        test_data_path=args.test_data,
        output_path=args.output_path,
        device=device,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
