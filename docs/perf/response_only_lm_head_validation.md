# Response-only LM head validation

Use `lm_head_test2` from `https://github.com/qiangyupei/verl`. It contains the functional changes rebased on verl `main` plus optional profiling, without historical experiment logs. Run from the repository root in a compatible CUDA/Megatron training environment. Install this checkout into that environment (`uv pip install -e . --no-deps`) and confirm `python -c "import verl; print(verl.__file__)"` points to this checkout.

## 1. Check values and non-zero gradients

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nproc_per_node=4 \
  -m pytest -q tests/special_distributed/test_response_only_lm_head.py
```

This runs the real fused kernel with TP=2 and CP=2. It compares full-row and selected-row results against a full-vocabulary Torch reference, with SP on/off, FP32/BF16, entropy on/off, internal masked spans, and an empty CP shard. All four ranks must finish without failures. It tests the LM-head boundary, not the full PP model. This test has not been run on the development machine, which has CPU-only PyTorch.

## 2. Run one cache-fill and four measured four-step jobs

```bash
export MODEL_PATH=/home/p00938733/Qwen3-8B
export TRAIN_FILE=/path/to/ruler-8k/train.parquet
export VAL_FILE=/path/to/ruler-8k/test.parquet

MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=8192 bash rlmh_test.sh
```

For the prompt-heavy workload, use its separately generated parquet files and run:

```bash
TRAIN_FILE=/path/to/ruler-30k/train.parquet \
VAL_FILE=/path/to/ruler-30k/test.parquet \
MAX_PROMPT_LENGTH=32768 MAX_RESPONSE_LENGTH=4096 bash rlmh_test.sh
```

Sequence limits do not generate or lengthen prompts: the parquet must already contain tokenized prompts of the intended length. The script sets `ignore_eos=true` for a fixed-length response stress test. Keep the total sequence within the model's supported context window.

Each invocation runs exactly these jobs on eight GPUs (TP=2, PP=2, CP=2, SP=true, rollout TP=4):

| Job | Response-only | Fused | Trajectories |
| --- | --- | --- | --- |
| `01-cache` | off | off | Generate/cache four steps |
| `02-unfused-baseline` | off | off | Reuse cached trajectories |
| `03-unfused-response-only` | on | off | Reuse the same trajectories |
| `04-fused-baseline` | off | on | Reuse the same trajectories |
| `05-fused-response-only` | on | on | Reuse the same trajectories |

Use a fresh default cache directory for each workload. Verify jobs 02--05 report trajectory-cache hits. Learning rate is zero to keep model weights fixed; entropy coefficient is 0.001 to produce non-zero gradients even if all NIAH rewards are 1. Confirm a non-zero actor gradient norm, matched token counts, and matching selected-token log-probabilities, entropy, loss, and gradient norms within the precision tolerance appropriate to your stack. A100 training execution remains required to validate the entire TP/SP/CP/PP path.

The four measured jobs cover the full `use_fused_kernels` x `response_only_lm_head` matrix in one invocation. Extra Hydra arguments can be appended to the command; the script keeps the two feature switches under its own control so every combination is exercised.

## 3. Read the results

The script writes logs, `summary-unfused.log`, and `summary-fused.log` under `rlmh-combination-logs/<run-id>/`. The first summary compares jobs 02 and 03; the second compares jobs 04 and 05. This isolates the incremental benefit of response-only projection both without fusion and on top of the fused kernel. The optimized arm should show fewer `projected_rows`; empty CP shards use one dummy row.

Unfused records report the materialized logits size, while both fused arms report zero materialized logits bytes. For fused runs, the useful measurements are synchronized LM-head forward latency and the measured incremental peak allocation, which includes temporary kernel workspace. These are not backward timings or whole-training peak memory. The summarizer intentionally compares only equal backends because fused and unfused records have different memory semantics. Whole-step timing can still be compared manually across jobs 02/04 and 03/05. Check individual rank records too: CP ranks can have different active-row ratios, and Ray log forwarding may omit ranks.

Treat step 1 as warmup when comparing training-stage or whole-step times. The summarizer's `--warmup 2` discards two LM-head invocations per rank/stream, not an entire training step. Profiling synchronizes the device and resets peak counters; for production throughput evidence, repeat longer reversed-order runs with `PROFILE=0`. The historical unfused A100 results do not measure the new fused combination.
