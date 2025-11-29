"""
Prepare multi-target dataset for predicting ALL correct token counts from h0.

From comprehensive dataset, extract:
- h0 (step 0 hidden states)
- Multi-hot encoding of all correct steps (e.g., [0,0,1,1,1,0,0] if steps 2,3,4 are correct)
"""

import argparse
from collections import defaultdict

import torch


def is_consecutive(correct_steps):
    """
    Check if list of correct steps forms a consecutive sequence.

    Args:
        correct_steps: List of token counts that are correct (e.g., [2,3,4])

    Returns:
        True if consecutive, False otherwise

    Examples:
        is_consecutive([3]) -> True (single)
        is_consecutive([2,3,4]) -> True (consecutive)
        is_consecutive([0,1,2,3,4,5,6]) -> True (full range)
        is_consecutive([0,1,5]) -> False (gap: 1→5)
        is_consecutive([2,5,6]) -> False (gap: 2→5)
    """
    if len(correct_steps) == 0:
        return False
    if len(correct_steps) == 1:
        return True  # Single token is consecutive

    # Sort and check if each element is +1 from previous
    sorted_steps = sorted(correct_steps)
    for i in range(1, len(sorted_steps)):
        if sorted_steps[i] != sorted_steps[i-1] + 1:
            return False  # Gap found

    return True


def prepare_multi_target_dataset(
    comprehensive_data_path: str,
    output_path: str,
    consecutive_only: bool = False,
):
    """
    Extract h0 samples and label with multi-hot encoding of all correct steps.

    Args:
        comprehensive_data_path: Path to comprehensive dataset
        output_path: Path to save prepared dataset
        consecutive_only: If True, keep only samples with consecutive correct token counts
    """
    print("=" * 80)
    print("PREPARING MULTI-TARGET DATASET FROM COMPREHENSIVE DATA")
    print("=" * 80)

    # Load comprehensive dataset
    print(f"\nLoading comprehensive dataset from {comprehensive_data_path}...")
    data = torch.load(comprehensive_data_path)

    hidden_states = data["hidden_states"]
    labels = data["labels"]  # 1=correct, 0=incorrect
    step_indices = data["step_indices"]
    question_indices = data["question_indices"]
    num_steps = data["num_steps"]

    print(f"  Total samples: {len(hidden_states):,}")
    print(f"  Total questions: {len(set(question_indices)):,}")
    print(f"  Steps per question: {num_steps}")

    # Group by question
    print("\nGrouping samples by question (handling duplicates with majority vote)...")

    # First pass: collect all labels for each (question, step) pair
    pair_correctness = defaultdict(list)
    h0_candidates = defaultdict(list)

    for i in range(len(hidden_states)):
        q_idx = question_indices[i]
        step = step_indices[i]
        is_correct = labels[i].item() == 1

        pair_correctness[(q_idx, step)].append(is_correct)

        if step == 0:
            h0_candidates[q_idx].append(hidden_states[i])

    # Second pass: determine final correctness via majority vote
    question_data = defaultdict(lambda: {"steps": {}, "h0": None})

    for (q_idx, step), correct_list in pair_correctness.items():
        # Majority vote: if more than half are correct, mark as correct
        num_correct = sum(correct_list)
        is_correct = num_correct > len(correct_list) / 2
        question_data[q_idx]["steps"][step] = is_correct

    # Use first h0 for each question (they should be similar if deterministic)
    for q_idx, h0_list in h0_candidates.items():
        question_data[q_idx]["h0"] = h0_list[0]

    print(f"  Found {len(question_data):,} unique questions")
    print(f"  Resolved duplicates using majority vote")

    # Extract h0 and multi-hot encoding of all correct steps
    print("\nExtracting h0 and multi-hot labels...")

    h0_states = []
    multi_hot_labels = []

    no_h0_count = 0
    never_correct_count = 0
    skipped_non_consecutive = 0
    correct_counts_distribution = defaultdict(int)
    consecutive_lengths_distribution = defaultdict(int)

    for q_idx in sorted(question_data.keys()):
        q_data = question_data[q_idx]

        # Check if h0 exists
        if q_data["h0"] is None:
            no_h0_count += 1
            continue

        # Find all correct steps
        correct_steps = []
        for step in range(num_steps):
            if step in q_data["steps"] and q_data["steps"][step]:
                correct_steps.append(step)

        if len(correct_steps) == 0:
            never_correct_count += 1
            continue  # Skip questions that never get correct

        # Filter for consecutive-only if requested
        if consecutive_only:
            if not is_consecutive(correct_steps):
                skipped_non_consecutive += 1
                continue  # Skip non-consecutive samples

        # Create multi-hot encoding
        multi_hot = torch.zeros(num_steps, dtype=torch.float32)
        for step in correct_steps:
            multi_hot[step] = 1.0

        # Add to dataset
        h0_states.append(q_data["h0"])
        multi_hot_labels.append(multi_hot)
        correct_counts_distribution[len(correct_steps)] += 1

        # Track consecutive sequence lengths
        if consecutive_only:
            consecutive_lengths_distribution[len(correct_steps)] += 1

    print(f"\n[Summary]")
    print(f"  Questions with h0: {len(question_data) - no_h0_count:,}")
    print(f"  Questions missing h0: {no_h0_count}")
    print(f"  Questions never correct (skipped): {never_correct_count:,}")
    if consecutive_only:
        print(f"  Questions non-consecutive (skipped): {skipped_non_consecutive:,}")
    print(f"  Questions in final dataset: {len(h0_states):,}")

    print(f"\nNumber of correct steps per question:")
    for num_correct in sorted(correct_counts_distribution.keys()):
        count = correct_counts_distribution[num_correct]
        pct = 100 * count / len(h0_states)
        print(f"  {num_correct} correct steps: {count:5d} ({pct:5.1f}%)")

    # Calculate average number of correct steps
    total_correct_steps = sum(k * v for k, v in correct_counts_distribution.items())
    avg_correct_steps = total_correct_steps / len(h0_states)
    print(f"\nAverage correct steps per question: {avg_correct_steps:.2f}")

    # Show consecutive sequence lengths distribution if filtering was applied
    if consecutive_only and consecutive_lengths_distribution:
        print(f"\nDistribution of consecutive sequence lengths:")
        for seq_len in sorted(consecutive_lengths_distribution.keys()):
            count = consecutive_lengths_distribution[seq_len]
            pct = 100 * count / len(h0_states)
            print(f"  Length {seq_len}: {count:5d} ({pct:5.1f}%)")

    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    torch.save(
        {
            "hidden_states": torch.stack(h0_states),
            "multi_hot_labels": torch.stack(multi_hot_labels),
            "num_steps": num_steps,
        },
        output_path,
    )

    print(f"✓ Saved {len(h0_states):,} samples")
    print(f"  Hidden states shape: {torch.stack(h0_states).shape}")
    print(f"  Multi-hot labels shape: {torch.stack(multi_hot_labels).shape}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-target dataset from comprehensive data"
    )
    parser.add_argument(
        "--comprehensive_data",
        type=str,
        default="/data/wweiyi/coconut/comprehensive_dataset/comprehensive_dataset.pt",
        help="Path to comprehensive dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="multi_target_dataset.pt",
        help="Path to save prepared dataset",
    )
    parser.add_argument(
        "--consecutive_only",
        action="store_true",
        help="Keep only samples with consecutive correct token counts",
    )

    args = parser.parse_args()

    prepare_multi_target_dataset(
        comprehensive_data_path=args.comprehensive_data,
        output_path=args.output_path,
        consecutive_only=args.consecutive_only,
    )


if __name__ == "__main__":
    main()
