"""
Collect comprehensive dataset with all information.

For each question and each step (0, 1, 2, ..., num_steps):
- Extract hidden state at that step
- Generate answer at that step
- Check if answer is correct
- Store: (hidden_state, is_correct)

This gives direct supervision: "given this hidden state, is the answer correct now?"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from coconut import Coconut


def load_coconut_model(checkpoint_path: str, model_id: str, device: str):
    """Load the coconut model and tokenizer."""
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
    load_result = model.load_state_dict(saved_weights, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"[warning] load_result = {load_result}")

    model = model.to(device)
    model.eval()
    return model, tokenizer, {"start_id": start_id, "latent_id": latent_id, "end_id": end_id}


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


def build_input_with_tokens(
    question: str,
    num_latent_tokens: int,
    tokenizer,
    start_id: int,
    latent_id: int,
    end_id: int,
) -> Dict:
    """Build input sequence with specified number of latent tokens."""
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


def evaluate_at_step(
    model,
    tokenizer,
    question: str,
    ground_truth: str,
    num_latent_tokens: int,
    special_tokens: Dict[str, int],
    device: str,
    max_new_tokens: int = 64,
) -> Tuple[torch.Tensor, bool, str]:
    """
    Evaluate at a specific step.

    Returns:
        (hidden_state, is_correct, predicted_answer)
    """
    # Build input
    inputs = build_input_with_tokens(
        question=question,
        num_latent_tokens=num_latent_tokens,
        tokenizer=tokenizer,
        **special_tokens,
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Get hidden state at current position
        if num_latent_tokens > 0:
            outputs = model.base_causallm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                output_hidden_states=True,
            )
        else:
            outputs = model.base_causallm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # Extract hidden state at last position
        hidden_state = outputs.hidden_states[-1][:, -1, :].squeeze(0).cpu()

        # Generate answer (same pattern as proof_of_concept_experiment.py)
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )

        # Decode and extract answer
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        predicted_answer = extract_answer(generated_text)

        # Check correctness
        is_correct = predicted_answer == ground_truth

    return hidden_state, is_correct, predicted_answer


def collect_comprehensive_dataset(
    checkpoint_path: str,
    model_id: str,
    data_path: str,
    num_steps: int,
    max_questions: int,
    output_path: str,
    device: str,
    gpu_id: int = None,
    num_gpus: int = 1,
):
    """
    Collect comprehensive dataset with hidden states and correctness labels.
    """
    model, tokenizer, special_tokens = load_coconut_model(checkpoint_path, model_id, device)

    # Load questions
    with open(data_path, "r") as f:
        data = json.load(f)

    if max_questions:
        data = data[:max_questions]

    # Split data across GPUs if multi-GPU processing
    # Keep track of original indices!
    if gpu_id is not None and num_gpus > 1:
        original_indices = [i for i in range(len(data)) if i % num_gpus == gpu_id]
        data = [data[i] for i in original_indices]
        print(f"GPU {gpu_id}: Processing {len(data)} questions (indices {original_indices[0]} to {original_indices[-1]})")
    else:
        original_indices = list(range(len(data)))

    hidden_states = []
    labels = []  # 1 if correct, 0 if not
    step_indices = []  # which step (0, 1, 2, ...)
    question_indices = []  # which question (GLOBAL index)

    # Statistics
    total_samples = 0
    correct_samples = 0
    correct_by_step = {i: 0 for i in range(num_steps)}
    total_by_step = {i: 0 for i in range(num_steps)}

    for local_idx, item in enumerate(tqdm(data, desc="Collecting comprehensive dataset")):
        global_q_idx = original_indices[local_idx]  # Use GLOBAL index
        question = item["question"]
        ground_truth = item["answer"].replace(",", "").strip()

        # Evaluate at each step
        for step in range(num_steps):
            hidden_state, is_correct, predicted_answer = evaluate_at_step(
                model, tokenizer, question, ground_truth, step, special_tokens, device
            )

            hidden_states.append(hidden_state)
            labels.append(1 if is_correct else 0)
            step_indices.append(step)
            question_indices.append(global_q_idx)  # Use GLOBAL index!

            total_samples += 1
            total_by_step[step] += 1
            if is_correct:
                correct_samples += 1
                correct_by_step[step] += 1

    # Save dataset
    torch.save(
        {
            "hidden_states": torch.stack(hidden_states),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "step_indices": step_indices,
            "question_indices": question_indices,
            "num_steps": num_steps,
        },
        output_path,
    )

    print(f"\n[SUMMARY]")
    print(f"  Total samples: {total_samples}")
    print(f"  Correct samples: {correct_samples} ({100*correct_samples/total_samples:.1f}%)")
    print(f"  Total questions: {len(set(question_indices))}")

    print(f"\nCorrectness by step:")
    for step in range(num_steps):
        total = total_by_step[step]
        correct = correct_by_step[step]
        acc = 100 * correct / total if total > 0 else 0
        print(f"  Step {step}: {correct}/{total} = {acc:.1f}%")

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect comprehensive dataset with hidden states and correctness"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../pretrained_checkpoints/stage_1_training_ck/checkpoint_12",
        help="Path to coconut checkpoint",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/gsm_train.json",
        help="Path to input data (JSON format)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=7,
        help="Number of steps to evaluate (0, 1, 2, ..., num_steps-1)",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="comprehensive_dataset.pt",
        help="Path to save the dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID for multi-GPU processing",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Total number of GPUs for multi-GPU processing",
    )

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    if args.gpu_id is not None:
        print(f"Multi-GPU mode: GPU {args.gpu_id} of {args.num_gpus}")

    collect_comprehensive_dataset(
        checkpoint_path=args.checkpoint_path,
        model_id=args.model_id,
        data_path=args.data_path,
        num_steps=args.num_steps,
        max_questions=args.max_questions,
        output_path=args.output_path,
        device=device,
        gpu_id=args.gpu_id,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
