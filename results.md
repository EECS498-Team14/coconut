Modified the training code for compatibility with other kinds of models and token generation counting.

1. Support for Multiple Model Architectures

The script uses HuggingFace's AutoModelForCausalLM and AutoTokenizer (lines 7, 102-103) which automatically detect and load
any causal language model:

model = AutoModelForCausalLM.from_pretrained(configs.model_id)
tokenizer = AutoTokenizer.from_pretrained(configs.model_id)

This means the same code works for:
- GPT-2: openai-community/gpt2
- Pythia: EleutherAI/pythia-14m, EleutherAI/pythia-70m, etc.
- Llama: Any Llama model ID

The FSDP wrapping policy (lines 176-182) is configured for Llama layers specifically, but for smaller models like GPT-2 and
Pythia, FSDP effectively becomes DDP since no layers match the wrap policy.

2. Token Generation Counting (My Modifications)

I added tracking for the average number of tokens generated during inference:

Line 435-440 - Added total_tokens_generated counter:
cor, cor_cot, total, total_tokens_generated = (
  torch.tensor(0, device=rank),
  torch.tensor(0, device=rank),
  torch.tensor(0, device=rank),
  torch.tensor(0, device=rank),
)

Lines 462, 469-471 - Calculate tokens generated per sample:
input_length = batch["input_ids"].shape[1]
outputs = parallel_model.module.generate(...)

tokens_generated = outputs.shape[1] - input_length
total_tokens_generated += tokens_generated

Line 501 - Aggregate across all GPUs:
dist.all_reduce(total_tokens_generated, op=dist.ReduceOp.SUM)

Lines 506-507 - Compute average:
total_tokens_generated = total_tokens_generated.item()
avg_tokens_generated = total_tokens_generated / total if total > 0 else 0

Lines 513-515 - Print results:
print(f"Avg tokens generated: {total_tokens_generated} / {total} = {avg_tokens_generated:.2f}")

Line 519 - Log to wandb:
wandb_run.log({..., "eval/avg_tokens_generated": avg_tokens_generated})

## GSM8k 

Run for 3 epochs (paper runs for 25 epochs)

<!-- CoT -->
Accuracy on validation set: 265 / 1319 = 0.20090978013646701
CoT match on validation set: 108 / 1319 = 0.08188021228203184
Avg tokens generated: 43291 / 1319 = 32.82

<!-- COCONUT -->
Accuracy on validation set: 304 / 1319 = 0.2304776345716452
CoT match on validation set: 0 / 1319 = 0.0
Avg tokens generated: 30088 / 1319 = 22.81

## ProsQA

Run for 25 epochs (paper runs for 50 epochs)

Accuracy on validation set: 472 / 500 = 0.944
CoT match on validation set: 0 / 500 = 0.0

## ProsQA, 14M

<!-- CoT -->
Accuracy on validation set: 264 / 500 = 0.528
CoT match on validation set: 117 / 500 = 0.234
Avg tokens generated: 25702 / 500 = 51.40

<!-- COCONUT -->
Accuracy on validation set: 318 / 500 = 0.636
CoT match on validation set: 0 / 500 = 0.0
Avg tokens generated: 4114 / 500 = 8.23

## ProntonQA

Run for 25 epochs (paper runs for 50 epochs)

Accuracy on validation set: 799 / 800 = 0.99875
CoT match on validation set: 0 / 800 = 0.0
Avg tokens generated: 19106 / 800 = 23.88

## ProntonQA, 14M

<!-- CoT -->
Accuracy on validation set: 450 / 800 = 0.5625
CoT match on validation set: 72 / 800 = 0.09
Avg tokens generated: 77527 / 800 = 96.91

<!-- COCONUT -->
Accuracy on validation set: 403 / 800 = 0.50375
CoT match on validation set: 0 / 800 = 0.0
Avg tokens generated: 19051 / 800 = 23.81






