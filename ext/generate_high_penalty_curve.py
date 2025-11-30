"""
Generate predictor curve with high token penalties (for low avg token counts).

Explores larger penalty values to get better coverage in the 0-3 avg token range.
"""

import argparse
import json
import os
import re
import subprocess
import sys


def train_and_evaluate_predictor(
    data_path: str,
    token_penalty: float,
    output_dir: str,
    coconut_checkpoint: str,
    test_data: str,
    model_id: str = "openai-community/gpt2",
):
    """Train predictor with given token_penalty and evaluate."""

    # Output paths
    model_name = f"predictor_penalty_{token_penalty:+.2f}".replace(".", "p").replace("-", "m")
    checkpoint_path = os.path.join(output_dir, f"{model_name}.pt")

    # Train
    print(f"\n{'='*80}")
    print(f"TRAINING WITH TOKEN_PENALTY={token_penalty:+.2f}")
    print(f"{'='*80}\n")

    train_cmd = [
        "python", "train_multi_target_predictor.py",
        "--data_path", data_path,
        "--output_path", checkpoint_path,
        "--architecture", "mlp",
        "--loss_type", "weighted",
        "--hidden_layers", "1024,512",
        "--dropout", "0.3",
        "--batch_size", "64",
        "--lr", "1e-3",
        "--num_epochs", "20",
        "--val_ratio", "0.15",
        "--token_penalty", str(token_penalty),
    ]

    subprocess.run(train_cmd, check=True)

    # Evaluate
    print(f"\n{'='*80}")
    print(f"EVALUATING WITH TOKEN_PENALTY={token_penalty:+.2f}")
    print(f"{'='*80}\n")

    eval_cmd = [
        "python", "evaluate_multi_target_predictor.py",
        "--predictor_checkpoint", checkpoint_path,
        "--coconut_checkpoint", coconut_checkpoint,
        "--test_data", test_data,
        "--strategy", "argmax",
        "--model_id", model_id,
    ]

    # Capture output to parse results
    result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)

    # Print the captured output so user can see it
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Parse accuracy and token distribution from output
    output_lines = result.stdout.split("\n")
    accuracy = None
    token_dist = {}

    for line in output_lines:
        # Match lines like: "Accuracy: 431/1319 = 32.68%"
        if "Accuracy:" in line and "=" in line:
            # Extract accuracy percentage
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

    # Print immediate results
    print(f"\n{'='*80}")
    print(f"RESULTS FOR TOKEN_PENALTY={token_penalty:+.2f}")
    print(f"{'='*80}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Avg Tokens: {avg_tokens:.2f}")
    print(f"{'='*80}\n")

    return {
        "token_penalty": token_penalty,
        "checkpoint": checkpoint_path,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "token_distribution": token_dist,
    }


def generate_high_penalty_curve(
    data_path: str,
    output_dir: str,
    coconut_checkpoint: str,
    test_data: str,
    token_penalty_values: list,
    output_path: str,
    model_id: str = "openai-community/gpt2",
):
    """Generate predictor curve with high penalties."""

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "curve_type": "predictor_high_penalty",
        "data_path": data_path,
        "coconut_checkpoint": coconut_checkpoint,
        "test_data": test_data,
        "points": []
    }

    for token_penalty in token_penalty_values:
        point = train_and_evaluate_predictor(
            data_path=data_path,
            token_penalty=token_penalty,
            output_dir=output_dir,
            coconut_checkpoint=coconut_checkpoint,
            test_data=test_data,
            model_id=model_id,
        )
        results["points"].append(point)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("HIGH PENALTY PREDICTOR CURVE GENERATED")
    print(f"{'='*80}\n")
    print(f"Results saved to {output_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Token Penalty':<14} {'Avg Tokens':<12} {'Accuracy':<10}")
    print("-" * 36)
    for point in results["points"]:
        print(
            f"{point['token_penalty']:+.2f}            "
            f"{point['avg_tokens']:<12.2f} "
            f"{point['accuracy']:<10.4f}"
        )

    print(f"\n{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictor curve with high token penalties (for low avg tokens)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="comprehensive_dataset/multi_target_dataset.pt",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictor_curve_models",
        help="Directory to save trained models",
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
        default="predictor_curve_high_penalty.json",
        help="Path to save curve results JSON",
    )
    parser.add_argument(
        "--token_penalty_values",
        type=str,
        default="0.30,0.40,0.50,0.60,0.80,1.00",
        help="Comma-separated high token penalty values",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID (e.g., openai-community/gpt2, meta-llama/Llama-2-7b-hf)",
    )

    args = parser.parse_args()

    token_penalty_values = [float(x) for x in args.token_penalty_values.split(",")]

    generate_high_penalty_curve(
        data_path=args.data_path,
        output_dir=args.output_dir,
        coconut_checkpoint=args.coconut_checkpoint,
        test_data=args.test_data,
        token_penalty_values=token_penalty_values,
        output_path=args.output_path,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
