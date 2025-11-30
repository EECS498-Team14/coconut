"""
Generate predictor curve using test-time logit biasing.

Trains a SINGLE predictor with token_penalty=0.0, then evaluates it
with different bias_weight values to span the accuracy-token tradeoff.

This is more efficient than training multiple predictors with different
token_penalty values, as it requires only one training run.
"""

import argparse
import json
import os
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

    # Parse accuracy and token distribution (same logic as generate_predictor_curve.py)
    output_lines = result.stdout.split("\n")
    accuracy = None
    token_dist = {}

    for line in output_lines:
        # Match lines like: "Accuracy: 431/1319 = 32.68%"
        if "Accuracy:" in line and "=" in line:
            match = re.search(r'=\s*([\d.]+)%', line)
            if match:
                accuracy = float(match.group(1)) / 100
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

    # Compute average tokens
    total = sum(token_dist.values())
    avg_tokens = sum(tok * count for tok, count in token_dist.items()) / total if total > 0 else 0

    print(f"\n{'='*80}")
    print(f"RESULTS FOR BIAS_WEIGHT={bias_weight:+.3f}")
    print(f"{'='*80}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Avg Tokens: {avg_tokens:.2f}")
    print(f"{'='*80}\n")

    return {
        "bias_weight": bias_weight,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
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

    print(f"{'Bias Weight':<14} {'Avg Tokens':<12} {'Accuracy':<10}")
    print("-" * 36)
    for point in results["points"]:
        print(
            f"{point['bias_weight']:+.3f}          "
            f"{point['avg_tokens']:<12.2f} "
            f"{point['accuracy']:<10.4f}"
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
    )


if __name__ == "__main__":
    main()
