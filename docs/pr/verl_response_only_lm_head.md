<!-- Suggested title: [megatron] feat: project LM head only for response tokens -->

### What does this PR do?

Megatron currently applies the vocabulary projection to every token row in a
packed or padded micro-batch, even though PPO losses often select only assistant
tokens through `loss_mask`. Long agent trajectories may contain prompts, tool
outputs, environment observations, and padding that do not contribute to the
policy loss but still materialize vocabulary-sized logits.

This change adds an opt-in `response_only_lm_head` setting for the unfused
Megatron language-model path. It applies the same causal shift, padding, and
context-parallel layout used for labels to the loss mask; selects active hidden
rows immediately before the output projection; computes log-probability,
entropy, and sum-of-squared-probability values only for those rows; and scatters
the scalar outputs back to the existing token layout. Unselected positions are
zero-filled and remain excluded by the existing loss mask.

The option defaults to `false`. Transformer, attention, and required
sequence-parallel hidden-state communication are unchanged.

#### Why this is needed

For a processed token count `T`, padded vocabulary size `V`, tensor-parallel
size `TP`, context-parallel size `CP`, and logits element size `b`, one dense
vocabulary tensor occupies approximately:

```text
T * V * b / (TP * CP) bytes per final-pipeline-stage rank
```

If the active-row ratio is `r`, response-only projection reduces LM-head GEMM
rows and vocabulary tensor elements to approximately `r` of the dense path.
The corresponding logits saving is approximately `1 - r`. Entropy computation
currently clones logits, so the transient LM-head saving can be larger than one
logits tensor.

Example BF16 estimates for one logits tensor per rank:

| Processed shape | Dense logits | Saved at 10% active | Saved at 25% active | Saved at 50% active |
| --- | ---: | ---: | ---: | ---: |
| 32K tokens, V=152064, TP=4, CP=1 | 2.320 GiB | 2.088 GiB | 1.740 GiB | 1.160 GiB |
| 128K tokens, V=248320, TP=2, CP=8 | 3.789 GiB | 3.410 GiB | 2.842 GiB | 1.895 GiB |

These estimates cover vocabulary logits only, not whole-process peak memory.

### Checklist Before Starting

- [x] Searched open PRs for `response only LM head`, `active rows LM head`,
  `loss mask vocabulary projection`, and related Megatron/fused-kernel terms.
- [x] This does not duplicate
  [#7370](https://github.com/verl-project/verl/pull/7370), which implements the
  same row-selection principle only for the FSDP/FSDP2 Torch fused LM head and
  explicitly leaves other paths for follow-up work.
- [x] This does not duplicate
  [#5130](https://github.com/verl-project/verl/pull/5130), which integrates a
  Blackwell-specific fused linear cross-entropy path for Megatron.
- [x] This PR targets the unfused Megatron path and preserves TP/SP/CP/PP
  behavior, so its implementation and compatibility surface are different.

### Design and correctness

#### Causal alignment

`loss_mask` is response-only data. The implementation first builds a full
`[prompt zeros; response mask]` sequence and then applies the same next-token
roll as labels. A loss-bearing target token at position `t` therefore selects
the predictor hidden state at `t - 1`. Internal zero spans, such as tool output,
are preserved rather than treated as the end of the response.

The mask is passed through the same THD/BSHD preprocessing, FP8 padding,
length-bucket padding, dynamic/static CP size, and zigzag/contiguous CP layout
as the model inputs and labels.

#### Tensor and sequence parallelism

With sequence parallelism, Megatron's output layer normally gathers hidden
states over TP before vocabulary projection. The implementation performs that
gather explicitly, selects active rows, and disables the output layer's second
gather for that invocation. The explicit gather owns backward reduce-scatter;
the output layer's non-SP input-gradient all-reduce is disabled to avoid double
reduction. Without sequence parallelism, the regular ColumnParallelLinear
gradient reduction remains unchanged.

Each CP rank may have a different active-row count. A rank with no active rows
projects one dummy row and restores a zero-gradient scalar path so all ranks
retain the LM head in the distributed backward graph.

#### Output compatibility

Sparse labels and temperatures are selected in the same order as hidden rows.
After log-probability, entropy, and sum-pi-squared computation, token-level
scalars are scattered back to the original CP-local dense layout. Existing CP
postprocessing and downstream PPO loss code remain unchanged.

### Compatibility and limitations

- Supported data formats: THD and BSHD.
- Supported parallel dimensions: TP, SP, static/dynamic CP, and PP.
- The option uses the unfused Megatron forward path. If fused kernels were
  requested, they are disabled before the model forward patch is selected.
- MTP training is not supported.
- Top-k distillation is not supported.
- The optimization does not reduce transformer/attention work or the required
  SP hidden-state gather.
- Benefits become small when most token rows are active; a 100% active control
  should show approximately zero improvement.

### Test

Focused CPU correctness, configuration, profiling, and summary tests:

```bash
python -m pytest -q \
  tests/models/mcore/test_response_only_lm_head_on_cpu.py \
  tests/utils/test_response_only_lm_head_profile_summary_on_cpu.py \
  tests/workers/test_megatron_distillation_only_on_cpu.py \
  tests/workers/config/test_engine_config_on_cpu.py
```

Result: `26 passed`.

Coverage includes:

- causal next-token mask alignment for nested THD and padded BSHD inputs with
  internal tool-output spans;
- sparse/dense active-logits and hidden/LM-head gradient parity;
- SP gather invocation and output-layer state restoration;
- empty local mask through the real projection hook, including zero hidden and
  LM-head weight gradients;
- sparse input selection and output-layout restoration, including non-empty
  backward scatter;
- fused-forward initialization behavior;
- configuration defaults;
- profiler dense/sparse byte accounting and JSON output;
- per-rank warmup, aggregation, latency, and memory comparison logic, including
  rejection of mismatched dense workloads.

The merged core/profiler test file follows the repository's `*_on_cpu.py`
convention, so these tests are collected by the standard CPU CI job.

Additional checks completed:

| Check | Result |
| --- | --- |
| Ruff check and format for changed Python files | PASS |
| Mypy for changed Python files | PASS |
| Generated Megatron trainer configuration | PASS |
| `git diff --check` | PASS |
| docs-time, docstrings, license, device API, DataProto, compileall | PASS |

Real TP/SP/CP accelerator validation is still required before requesting
upstream review. CPU tests mock the SP collective boundary and cannot validate
NCCL/HCCL behavior or production kernel timing.

### Performance validation patch

Profiling used for controlled A/B validation is maintained as a separate patch
and is not intended to be included in the production PR. Enable it with:

```bash
export VERL_RESPONSE_ONLY_LM_HEAD_PROFILE=1
```

Run identical unfused Megatron jobs with:

```text
baseline:  response_only_lm_head=false
optimized: response_only_lm_head=true
```

The profiler emits `VERL_RESPONSE_ONLY_LM_HEAD_PROFILE` JSON records containing:

- active, dense, and projected row counts;
- synchronized LM-head plus log-probability/entropy latency;
- vocabulary shard size and dtype;
- dense-equivalent, actual, and saved logits bytes;
- allocated memory before the measured region;
- peak allocated memory and incremental peak above the pre-head allocation;
- role, data format, TP/CP/PP sizes, vocabulary shard/dtype, and entropy mode
  including chunking settings.

Summarize captured logs with:

```bash
python examples/profile/summarize_response_only_lm_head.py \
  --baseline baseline.log \
  --optimized response_only.log \
  --warmup 2
```

The profiler synchronizes the accelerator and resets peak-memory statistics at
the LM-head boundary. Both A/B arms must use it, and results should not be mixed
with production step timing or process-wide peak-memory metrics.

Recommended hardware matrix:

| Dimension | Values |
| --- | --- |
| Active-row ratio | 10%, 25%, 50%, 75%, 100% control |
| Entropy | disabled, enabled |
| Format | THD; BSHD where applicable |
| Parallelism | TP+SP, TP+CP, and production TP+SP+CP+PP |
| Samples | at least 5 warmups and 20 synchronized measurements |

Report per-rank maximum latency, LM-head median/p95 latency, optimizer-step
time, actual/dense logits GiB, incremental LM-head peak, and whole-process peak.
No Megatron accelerator numbers are claimed in this document until that A/B is
run. FSDP measurements in #7370 are useful as an order-of-magnitude reference
but are not evidence for this backend.

### API and usage

```yaml
actor_rollout_ref:
  actor:
    megatron:
      response_only_lm_head: true
  ref:
    megatron:
      response_only_lm_head: true
```

Actor and reference settings are independent. Enable only the roles whose
log-probability or training passes should use sparse vocabulary projection.

### Code changes

Functional patch:

- add the output-layer hidden-row selection and sparse-output restoration
  helpers;
- construct next-token-aligned CP-local masks for THD and BSHD;
- integrate sparse labels, temperatures, log-probability, entropy, and
  sum-pi-squared processing;
- add opt-in engine/Hydra configuration and generated config;
- add correctness, gradient, compatibility, and configuration tests;
- document usage and limitations.

Separate profiling patch:

- add opt-in synchronized latency and accelerator-memory records;
- collect exact dense/actual vocabulary tensor sizes;
- add an A/B log summarizer and focused CPU tests;
- document the controlled benchmark procedure.

### Checklist Before Submitting

- [x] Read the contribution guide and repository `AGENTS.md`.
- [x] Checked for materially overlapping open PRs.
- [x] Added focused CPU tests and documentation.
- [ ] Run real multi-accelerator TP/SP/CP parity and A/B measurements.
- [ ] Run full pre-commit in an environment that can fetch its remote hook
  environments. Local equivalents have passed.
- [ ] Rebase onto the requested upstream target if it has advanced beyond
  v0.9.0.
- [ ] Request CI through the verl Slack/Feishu process when ready.
- [x] Recipe-submodule update is not applicable.

### AI assistance

OpenAI Codex assisted with analysis, implementation, test design, profiling
instrumentation, and drafting. The human submitter must review every changed
line, run the distributed hardware validation, and remain responsible for the
submission.
