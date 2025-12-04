"""
Compute baseline accuracy for each fixed step count from comprehensive dataset.
Uses precomputed data (no LLM inference needed).
"""

import argparse
import json
import torch
from collections import defaultdict


def compute_baseline_accuracy(
    dataset_path: str,
    output_path: str = None
):
    """
    Compute accuracy for each fixed step count from comprehensive dataset.
    """
    print("=" * 80)
    print("COMPUTING BASELINE ACCURACY FROM COMPREHENSIVE DATASET")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    data = torch.load(dataset_path, map_location="cpu")

    labels = data["labels"]
    step_indices = data["step_indices"]
    question_indices = data["question_indices"]
    num_steps = data["num_steps"]

    # Check if we have token counts
    has_tokens = "total_tokens" in data
    if has_tokens:
        total_tokens = data["total_tokens"]
        print(f"Dataset includes token counts")
    else:
        print(f"Dataset does NOT include token counts (token metrics will be unavailable)")

    # Convert to lists
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if isinstance(step_indices, list):
        step_indices = step_indices
    else:
        step_indices = step_indices.tolist()
    if isinstance(question_indices, list):
        question_indices = question_indices
    else:
        question_indices = question_indices.tolist()
    if has_tokens:
        if isinstance(total_tokens, torch.Tensor):
            total_tokens = total_tokens.tolist()

    print(f"Total samples: {len(labels)}")
    print(f"Num steps: {num_steps}")

    # Group by question
    print("\nGrouping data by question...")
    question_data = defaultdict(dict)
    for i in range(len(labels)):
        q_idx = question_indices[i]
        step = step_indices[i]
        is_correct = labels[i] > 0.5

        if step not in question_data[q_idx]:
            question_data[q_idx][step] = {
                'is_correct': is_correct,
                'total_tokens': total_tokens[i] if has_tokens else None
            }

    unique_questions = sorted(question_data.keys())
    print(f"Unique questions: {len(unique_questions)}")

    # Compute baseline accuracy for each fixed step
    print("\n" + "=" * 80)
    print("BASELINE ACCURACY (FIXED STEP COUNTS)")
    print("=" * 80)

    results = {
        "dataset": dataset_path,
        "total_questions": len(unique_questions),
        "num_steps": num_steps,
        "has_tokens": has_tokens,
        "points": []
    }

    print(f"\n{'Step':<6} {'Accuracy':<12} {'Correct':<10} {'Total':<10}", end="")
    if has_tokens:
        print(f"{'Avg Tokens':<15}")
    else:
        print()
    print("-" * (38 if not has_tokens else 53))

    for step in range(num_steps):
        correct = 0
        total_tokens_sum = 0
        count_with_tokens = 0

        for q_idx in unique_questions:
            if step in question_data[q_idx]:
                if question_data[q_idx][step]['is_correct']:
                    correct += 1
                if has_tokens and question_data[q_idx][step]['total_tokens'] is not None:
                    total_tokens_sum += question_data[q_idx][step]['total_tokens']
                    count_with_tokens += 1

        accuracy = correct / len(unique_questions)
        avg_tokens = total_tokens_sum / count_with_tokens if count_with_tokens > 0 else 0.0

        point_data = {
            "step": step,
            "num_tokens": step,
            "avg_tokens": float(step),
            "accuracy": accuracy,
            "correct": correct,
            "total": len(unique_questions),
        }

        if has_tokens:
            point_data["avg_total_tokens"] = avg_tokens

        results["points"].append(point_data)

        print(f"{step:<6} {accuracy:<12.2%} {correct:<10} {len(unique_questions):<10}", end="")
        if has_tokens:
            print(f"{avg_tokens:<15.2f}")
        else:
            print()

    # Compute oracle (best step per question)
    print("\n" + "-" * (38 if not has_tokens else 53))
    oracle_correct = sum(
        1 for q_idx in unique_questions
        if any(question_data[q_idx][s]['is_correct'] for s in question_data[q_idx])
    )
    oracle_accuracy = oracle_correct / len(unique_questions)

    results["oracle"] = {
        "accuracy": oracle_accuracy,
        "correct": oracle_correct,
        "total": len(unique_questions),
    }

    print(f"{'Oracle':<6} {oracle_accuracy:<12.2%} {oracle_correct:<10} {len(unique_questions):<10}")
    print("=" * 80)

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute baseline accuracy from comprehensive dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="comprehensive_dataset_with_tokens_train.pt",
        help="Path to comprehensive dataset (train/test/valid)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="baseline_curve_train.json",
        help="Path to save results JSON",
    )

    args = parser.parse_args()
    compute_baseline_accuracy(args.dataset_path, args.output_path)


if __name__ == "__main__":
    main()
