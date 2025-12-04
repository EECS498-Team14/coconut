"""
Generate predictor curve using test-time logit biasing.

Trains a SINGLE predictor with token_penalty=0.0, then evaluates it
with different bias_weight values to span the accuracy-token tradeoff.

This is more efficient than training multiple predictors with different
token_penalty values, as it requires only one training run.
"""

import argparse
import json
import re
import subprocess
import sys


def train_unbiased_predictor(
    data_path: str,
    output_path: str,
):
    """Train a single unbiased predictor (token_penalty=0)."""

    print(f"\n{'='*80}")
    print("TRAINING UNBIASED PREDICTOR (token_penalty=0.0)")
    print(f"{'='*80}\n")

    train_cmd = [
        "python", "train_multi_target_predictor.py",
        "--data_path", data_path,
        "--output_path", output_path,
        "--architecture", "mlp",
        "--loss_type", "weighted",
        "--hidden_layers", "1024,512",
        "--dropout", "0.3",
        "--batch_size", "64",
        "--lr", "1e-3",
        "--num_epochs", "20",
        "--val_ratio", "0.15",
        "--token_penalty", "0.0",  # UNBIASED TRAINING
    ]

    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        print(f"\nERROR: Training failed with return code {result.returncode}")
        sys.exit(1)
    print(f"\nPredictor trained and saved to {output_path}")


def evaluate_with_bias(
    predictor_checkpoint: str,
    bias_weight: float,
    coconut_checkpoint: str,
    test_data: str,
    model_id: str = "openai-community/gpt2",
):
    """Evaluate predictor with given bias_weight."""

    print(f"\n{'='*80}")
    print(f"EVALUATING WITH BIAS_WEIGHT={bias_weight:+.3f}")
    print(f"{'='*80}\n")

    eval_cmd = [
        "python", "evaluate_multi_target_predictor.py",
        "--predictor_checkpoint", predictor_checkpoint,
        "--coconut_checkpoint", coconut_checkpoint,
        "--test_data", test_data,
        "--strategy", "biased",
        "--bias_weight", str(bias_weight),
        "--model_id", model_id,
    ]

    # Capture output to parse results
    result = subprocess.run(eval_cmd, capture_output=True, text=True)

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Check for errors
    if result.returncode != 0:
        print(f"\nERROR: Evaluation failed with return code {result.returncode}")
        sys.exit(1)

    # Parse accuracy, token distribution, avg tokens generated, and avg total tokens
    output_lines = result.stdout.split("\n")
    accuracy = None
    token_dist = {}
    avg_latent_tokens = None
    avg_tokens_generated = None
    avg_total_tokens = None
    avg_total_tokens_correct = None

    for line in output_lines:
        # Match lines like: "Accuracy: 431/1319 = 32.68%"
        if "Accuracy:" in line and "=" in line:
            match = re.search(r'=\s*([\d.]+)%', line)
            if match:
                accuracy = float(match.group(1)) / 100
        # Match lines like: "Avg latent tokens: 5.23"
        elif "Avg latent tokens:" in line:
            match = re.search(r'Avg latent tokens:\s*([\d.]+)', line)
            if match:
                avg_latent_tokens = float(match.group(1))
        # Match lines like: "Avg tokens generated: 45.23"
        elif "Avg tokens generated:" in line:
            match = re.search(r'Avg tokens generated:\s*([\d.]+)', line)
            if match:
                avg_tokens_generated = float(match.group(1))
        # Match lines like: "Avg total tokens: 50.46"
        elif "Avg total tokens (correct only):" in line:
            match = re.search(r'Avg total tokens \(correct only\):\s*([\d.]+)', line)
            if match:
                avg_total_tokens_correct = float(match.group(1))
        elif "Avg total tokens:" in line:
            match = re.search(r'Avg total tokens:\s*([\d.]+)', line)
            if match:
                avg_total_tokens = float(match.group(1))
        # Match lines like: "  6 tokens: 1305 (98.9%)"
        elif "tokens:" in line and "(" in line:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    token_count = int(parts[0])
                    count = int(parts[2])
                    token_dist[token_count] = count
                except (ValueError, IndexError):
                    pass

    # Compute average latent tokens if not already parsed
    if avg_latent_tokens is None:
        total = sum(token_dist.values())
        avg_latent_tokens = sum(tok * count for tok, count in token_dist.items()) / total if total > 0 else 0

    # Compute total tokens if not already parsed
    if avg_total_tokens is None and avg_latent_tokens is not None and avg_tokens_generated is not None:
        avg_total_tokens = avg_latent_tokens + avg_tokens_generated

    print(f"\n{'='*80}")
    print(f"RESULTS FOR BIAS_WEIGHT={bias_weight:+.3f}")
    print(f"{'='*80}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Avg Latent Tokens: {avg_latent_tokens:.2f}")
    print(f"  Avg Tokens Generated: {avg_tokens_generated:.2f}")
    print(f"  Avg Total Tokens: {avg_total_tokens:.2f}")
    print(f"  Avg Total Tokens (Correct): {avg_total_tokens_correct:.2f}")
    print(f"{'='*80}\n")

    return {
        "bias_weight": bias_weight,
        "accuracy": accuracy,
        "avg_tokens": avg_latent_tokens,
        "avg_tokens_generated": avg_tokens_generated,
        "avg_total_tokens": avg_total_tokens,
        "avg_total_tokens_correct": avg_total_tokens_correct,
        "token_distribution": token_dist,
    }


def generate_biased_curve(
    data_path: str,
    predictor_checkpoint: str,
    coconut_checkpoint: str,
    test_data: str,
    bias_weight_values: list,
    output_path: str,
    train_predictor: bool = True,
    model_id: str = "openai-community/gpt2",
):
    """Generate predictor curve using test-time logit biasing."""

    # Train unbiased predictor (unless using existing checkpoint)
    if train_predictor:
        train_unbiased_predictor(data_path, predictor_checkpoint)
    else:
        print(f"\nUsing existing predictor checkpoint: {predictor_checkpoint}")

    # Evaluate with different bias weights
    results = {
        "curve_type": "test_time_biased",
        "predictor_checkpoint": predictor_checkpoint,
        "coconut_checkpoint": coconut_checkpoint,
        "test_data": test_data,
        "training_token_penalty": 0.0,
        "points": []
    }

    for bias_weight in bias_weight_values:
        point = evaluate_with_bias(
            predictor_checkpoint=predictor_checkpoint,
            bias_weight=bias_weight,
            coconut_checkpoint=coconut_checkpoint,
            test_data=test_data,
            model_id=model_id,
        )
        results["points"].append(point)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("BIASED PREDICTOR CURVE GENERATED")
    print(f"{'='*80}\n")
    print(f"Results saved to {output_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Bias Weight':<14} {'Accuracy':<10} {'Total (All)':<12} {'Total (Correct)':<15}")
    print("-" * 51)
    for point in results["points"]:
        print(
            f"{point['bias_weight']:+.3f}          "
            f"{point['accuracy']:<10.4f} "
            f"{point['avg_total_tokens']:<12.2f} "
            f"{point['avg_total_tokens_correct']:<15.2f}"
        )

    print(f"\n{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictor curve using test-time logit biasing"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="comprehensive_dataset/multi_target_dataset.pt",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--predictor_checkpoint",
        type=str,
        default="unbiased_predictor.pt",
        help="Path to save/load unbiased predictor",
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
        default="biased_predictor_curve.json",
        help="Path to save curve results JSON",
    )
    parser.add_argument(
        "--bias_weight_values",
        type=str,
        default="-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated bias weight values",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and use existing predictor checkpoint",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID (e.g., openai-community/gpt2, meta-llama/Llama-2-7b-hf)",
    )

    args = parser.parse_args()

    bias_weight_values = [float(x) for x in args.bias_weight_values.split(",")]

    generate_biased_curve(
        data_path=args.data_path,
        predictor_checkpoint=args.predictor_checkpoint,
        coconut_checkpoint=args.coconut_checkpoint,
        test_data=args.test_data,
        bias_weight_values=bias_weight_values,
        output_path=args.output_path,
        train_predictor=not args.skip_training,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
