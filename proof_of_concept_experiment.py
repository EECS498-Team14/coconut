"""
Proof-of-Concept Experiment: Variable c_thoughts with Fixed Checkpoint

Goal:
    Generate oracle labels (optimal c_thoughts per question) for training a
    predictor that selects the right number of latent tokens at inference time.

Experiment Design:
    1. Load checkpoint_12 (latest, seen all stages)
    2. For each test question:
       - Try different c_thoughts values: [0, 2, 3, 4, 5, 6]
       - Pick the one that gives correct answer (smallest count wins)
    3. Record detailed per-question results for later supervised training
"""

import torch
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from coconut import Coconut


def build_input_with_tokens(
    question: str,
    num_latent_tokens: int,
    tokenizer,
    start_id: int,
    latent_id: int,
    end_id: int,
) -> Dict:
    """
    Build input sequence with specified number of latent tokens.

    This function controls c_thoughts (number of continuous thoughts) during inference
    by dynamically inserting the specified number of <|latent|> tokens.

    HOW C_THOUGHTS IS CONTROLLED:
    =============================
    c_thoughts is controlled by the NUMBER of <|latent|> tokens inserted in the input.
    The coconut model processes these tokens sequentially in a feedback loop:

    1. Standard input format:
       [question_tokens] + <|start-latent|> + [<|latent|> x num_latent_tokens] + <|end-latent|>

    2. The coconut forward pass (coconut.py) detects all <|latent|> tokens and:
       - For pass_idx = 0, 1, 2, ..., num_latent_tokens-1:
         * Run forward pass up to the current <|latent|> token
         * Extract hidden state from the PREVIOUS token
         * Replace the <|latent|> token embedding with this hidden state
         * Continue to next <|latent|> token
       - This creates a continuous thought feedback loop

    3. Examples:
       num_latent_tokens = 0:  No coconut, pure CoT-style reasoning
       num_latent_tokens = 2:  Stage 1 training (2 continuous thoughts)
       num_latent_tokens = 4:  Stage 2 training (4 continuous thoughts)
       num_latent_tokens = 6:  Stage 3 training (6 continuous thoughts)

    4. Key insight for adaptive c_thoughts:
       By varying num_latent_tokens at inference time, we can control how much
       "thinking" the model does for each question. Simple questions may only
       need 2 tokens, while complex ones may benefit from 6 tokens.

    5. Why this works with a fixed checkpoint:
       Checkpoint_12 (trained through all stages 0-3) has seen 0, 2, 4, and 6
       latent tokens during training. It should be able to handle variable
       token counts at inference time without retraining.

    Args:
        question: Input question
        num_latent_tokens: Number of <|latent|> tokens to insert (controls c_thoughts!)
        tokenizer: Tokenizer
        start_id, latent_id, end_id: Special token IDs

    Returns:
        Dictionary with input_ids, attention_mask, position_ids
    """
    # Tokenize question
    question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)

    # Build sequence with dynamic number of latent tokens
    if num_latent_tokens > 0:
        input_ids = (
            question_tokens + [start_id] + [latent_id] * num_latent_tokens + [end_id]
        )
    else:
        # No latent tokens (CoT-style)
        input_ids = question_tokens

    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
        "position_ids": torch.tensor([position_ids]),
    }


def load_coconut_model(
    checkpoint_path: str, model_id: str = "openai-community/gpt2", device: str = "cuda"
):
    """
    Load a coconut model from a checkpoint.

    This replicates the loading logic from run.py for inference.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., ./pretrained_checkpoints/stage_1_training_ck/checkpoint_12)
        model_id: Base model ID (default: gpt2)
        device: Device to load model on

    Returns:
        Loaded coconut model in eval mode
    """
    print(f"  Loading base model {model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    # Setup tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Resize embeddings for new tokens
    base_model.resize_token_embeddings(len(tokenizer))
    embeddings = base_model.get_input_embeddings()
    output_embeddings = base_model.get_output_embeddings()

    # Initialize new token embeddings (same as run.py)
    target_id = tokenizer.convert_tokens_to_ids("<<")
    target_embedding = embeddings.weight.data[target_id]
    target_output = output_embeddings.weight.data[target_id]

    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = target_embedding
        output_embeddings.weight.data[token_id] = target_output

    # Wrap with Coconut
    print(f"  Wrapping with Coconut...")
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # Load checkpoint weights
    print(f"  Loading checkpoint weights from {checkpoint_path}...")
    saved_weights = torch.load(checkpoint_path, map_location=device)

    # Handle potential key mismatches
    load_result = model.load_state_dict(saved_weights, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"  Warning: Load result = {load_result}")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully!")
    return model


def extract_answer(generated_text: str) -> str:
    """Extract final answer from generated text."""
    # Answer format: "### {answer}"
    if "#" in generated_text:
        answer = generated_text.split("#")[-1].strip()
    else:
        answer = generated_text.strip()

    # Remove commas and extra whitespace
    answer = answer.replace(",", "").strip()

    return answer


def evaluate_single_question(
    question: str,
    ground_truth: str,
    num_latent_tokens: int,
    model,
    tokenizer,
    special_tokens: Dict[str, int],
    max_new_tokens: int = 64,
) -> Tuple[bool, str]:
    """
    Evaluate a single question with specified number of latent tokens.

    Returns:
        (is_correct, predicted_answer)
    """
    # Build input
    inputs = build_input_with_tokens(
        question=question,
        num_latent_tokens=num_latent_tokens,
        tokenizer=tokenizer,
        **special_tokens,
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )

    # Extract answer
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = extract_answer(generated_text)

    # Check correctness
    is_correct = predicted_answer == ground_truth

    return is_correct, predicted_answer


def run_poc_experiment(
    test_data_path: str,
    checkpoint_12_path: str,
    token_choices: List[int] = [0, 2, 3, 4, 5, 6],
    max_questions: int = None,
    device: str = "cuda",
):
    """
    Main proof-of-concept experiment.

    Args:
        test_data_path: Path to GSM8k test data
        checkpoint_12_path: Path to checkpoint_12 (last checkpoint)
        token_choices: Different c_thoughts values to try
        max_questions: Limit number of questions (for quick testing)
        device: Device to run on
    """

    # Load data
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    if max_questions:
        test_data = test_data[:max_questions]

    print(f"Loaded {len(test_data)} questions")
    print()

    # Setup tokenizer and special tokens
    print("Setting up tokenizer and special tokens...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    special_tokens = {
        "start_id": tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        "latent_id": tokenizer.convert_tokens_to_ids("<|latent|>"),
        "end_id": tokenizer.convert_tokens_to_ids("<|end-latent|>"),
    }
    print(f"Special tokens: {special_tokens}")
    print()

    # ========================================================================
    # Checkpoint 12 with Oracle Token Selection
    # ========================================================================
    print("=" * 80)
    print("CHECKPOINT 12 + ORACLE TOKEN SELECTION")
    print("=" * 80)
    print(f"For each question, try all token choices {token_choices}")
    print(f"and pick the one that gives correct answer (if any).")
    print()

    print(f"Loading checkpoint_12 from {checkpoint_12_path}...")
    model_12 = load_coconut_model(checkpoint_12_path, device=device)
    print()

    results_12_oracle = {
        "correct_by_token_count": defaultdict(int),
        "total_by_token_count": defaultdict(int),
        "optimal_tokens_per_question": [],
        "question_results": [],
    }

    print("Evaluating checkpoint_12 with different token counts...")
    for question_data in tqdm(test_data, desc="Questions"):
        question = question_data["question"]
        ground_truth = question_data["answer"].replace(",", "").strip()

        # Try all token counts
        token_results = {}
        for num_tokens in token_choices:
            is_correct, predicted = evaluate_single_question(
                question=question,
                ground_truth=ground_truth,
                num_latent_tokens=num_tokens,
                model=model_12,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
            )

            token_results[num_tokens] = {"correct": is_correct, "answer": predicted}

            results_12_oracle["total_by_token_count"][num_tokens] += 1
            if is_correct:
                results_12_oracle["correct_by_token_count"][num_tokens] += 1

        # Find best token count for this question (oracle)
        correct_tokens = [k for k, v in token_results.items() if v["correct"]]

        if correct_tokens:
            # If multiple work, choose the smallest (most efficient)
            optimal_tokens = min(correct_tokens)
            question_correct = True
        else:
            # None worked - record as -1
            optimal_tokens = -1
            question_correct = False

        results_12_oracle["optimal_tokens_per_question"].append(optimal_tokens)
        results_12_oracle["question_results"].append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "optimal_tokens": optimal_tokens,
                "token_results": token_results,
                "oracle_correct": question_correct,
            }
        )

    # Compute accuracy with oracle
    oracle_correct = sum(
        1 for x in results_12_oracle["optimal_tokens_per_question"] if x >= 0
    )
    oracle_accuracy = oracle_correct / len(test_data)

    print()
    print("Results for Checkpoint 12 + Oracle:")
    print(
        f"  Oracle Accuracy: {oracle_correct}/{len(test_data)} = {oracle_accuracy:.1%}"
    )
    print()
    print("  Accuracy by token count:")
    for num_tokens in sorted(token_choices):
        correct = results_12_oracle["correct_by_token_count"][num_tokens]
        total = results_12_oracle["total_by_token_count"][num_tokens]
        acc = correct / total if total > 0 else 0
        print(f"    {num_tokens} tokens: {correct}/{total} = {acc:.1%}")
    print()
    print("  Optimal token distribution:")
    token_dist = defaultdict(int)
    for t in results_12_oracle["optimal_tokens_per_question"]:
        if t >= 0:
            token_dist[t] += 1
    for num_tokens in sorted(token_dist.keys()):
        count = token_dist[num_tokens]
        pct = count / oracle_correct * 100
        print(f"    {num_tokens} tokens: {count} questions ({pct:.1f}%)")
    print()

    # Save detailed results
    output_path = "poc_experiment_results.json"
    print(f"Saving detailed results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(
            {
                "experiment_config": {
                    "checkpoint_12_path": checkpoint_12_path,
                    "token_choices": token_choices,
                    "num_questions": len(test_data),
                },
                "summary": {
                    "checkpoint_12_oracle_accuracy": oracle_accuracy,
                },
                "checkpoint_12_results": results_12_oracle,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return {
        "oracle_accuracy": oracle_accuracy,
        "results_12": results_12_oracle,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Proof-of-concept experiment")
    parser.add_argument(
        "--test_data", type=str, default="data/gsm_train.json", help="Path to test data"
    )
    parser.add_argument(
        "--checkpoint_12",
        type=str,
        default="pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
        help="Path to checkpoint_12",
    )
    parser.add_argument(
        "--token_choices",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Comma-separated list of token counts to try",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=10000,
        help="Limit number of questions for quick testing",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    args = parser.parse_args()

    token_choices = [int(x) for x in args.token_choices.split(",")]

    results = run_poc_experiment(
        test_data_path=args.test_data,
        checkpoint_12_path=args.checkpoint_12,
        token_choices=token_choices,
        max_questions=args.max_questions,
        device=args.device,
    )
