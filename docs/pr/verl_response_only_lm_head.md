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

#### A100 controlled A/B

The verl v0.9.0 adaptation was exercised on one node with 8 NVIDIA A100 GPUs
using Qwen3-8B BF16, THD/remove-padding, TP=2, PP=2, CP=2, sequence parallelism
enabled, and rollout TP=4. Actor/ref parameter offload was enabled. Each
workload used one prompt with two rollouts per step and ran for four steps. A
cache-fill run generated the trajectories; both measured arms injected the
same two cached trajectories at every step. The baseline and optimized arms
both enabled the same synchronized profiling instrumentation.

Step 1 was treated as warmup. Mean results for steps 2-4 were:

| Workload | Actual prompt | Response | Baseline step | Optimized step | Step reduction | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | 8046-8052 | 8192 | 47.815 s | 47.249 s | 1.18% | 1.20% |
| prompt-heavy | 16230-16232 | 4096 | 54.398 s | 50.663 s | **6.86%** | **7.37%** |

The three prompt-heavy steady-step reductions were 7.03%, 6.88%, and 6.68%.
The measured stage means were:

| Workload | Stage | Baseline | Optimized | Reduction |
| --- | --- | ---: | ---: | ---: |
| balanced | actor old-log-prob | 7.201 s | 6.546 s | 9.09% |
| balanced | ref log-prob | 7.777 s | 7.578 s | 2.56% |
| balanced | actor update | 19.267 s | 18.815 s | 2.34% |
| prompt-heavy | actor old-log-prob | 8.605 s | 7.253 s | **15.71%** |
| prompt-heavy | ref log-prob | 9.008 s | 8.421 s | **6.52%** |
| prompt-heavy | actor update | 22.918 s | 21.501 s | **6.18%** |

`actor old-log-prob` requests entropy in the v1 trainer. The measured actor
update did not request entropy because both `actor.calculate_entropy` and
`actor.entropy_coeff` were zero. The local profiler covers the forward LM-head
and vocabulary processing region; actor-update stage time above also includes
the subsequent backward and optimizer-facing work.

#### Final-pipeline-stage LM-head measurements

TP partitions vocabulary columns and CP partitions token rows. For a fixed DP
replica and final PP stage, summing local logits elements over all TP x CP ranks
therefore reconstructs the logical global `[tokens, vocabulary]` matrix. This
is an aggregate work/temporary-storage metric, not memory resident on one GPU.
Per-GPU feasibility and distributed critical-path latency are instead bounded
by the response-bearing final-stage rank.

The balanced workload had a global active-row ratio of about 50.44%, implying
about 49.56% fewer aggregate logits elements. On the observed response-bearing
rank, post-warmup medians were:

| Pass | Baseline latency | Optimized latency | Reduction | Baseline peak increment | Optimized peak increment |
| --- | ---: | ---: | ---: | ---: | ---: |
| actor, no entropy | 180.80 ms | 112.79 ms | 37.61% | 6.894 GiB | 3.444 GiB |
| actor, entropy | 513.11 ms | 183.98 ms | 64.14% | 13.850 GiB | 6.827 GiB |
| ref, no entropy | 155.15 ms | 92.86 ms | 40.15% | 6.832 GiB | 3.382 GiB |

The prompt-heavy workload had a global active-row ratio of about 20.15%, so
the logical TP x CP aggregate contains about 79.85% fewer logits elements.
With CP=2, response rows were concentrated on one CP shard; its local active
ratio was 40.30%, while the other observed CP shard had no active rows. On the
observed response-bearing rank:

| Pass | Baseline latency | Optimized latency | Reduction | Baseline peak increment | Optimized peak increment |
| --- | ---: | ---: | ---: | ---: | ---: |
| actor, no entropy | 217.92 ms | 120.59 ms | **44.66%** | 8.629 GiB | 3.462 GiB |
| actor, entropy | 635.70 ms | 165.71 ms | **73.93%** | 17.337 GiB | 6.878 GiB |
| ref, no entropy | 206.73 ms | 99.05 ms | **52.09%** | 8.552 GiB | 3.400 GiB |

This is approximately a 60% reduction in the measured LM-head peak increment
on the response-bearing rank. Empty CP ranks projected only the required dummy
row and completed forward/backward collectives without a hang.

Ray forwarded records for only three of the expected four final-stage ranks.
The tables above therefore use the observed response-bearing rank; the global
logits reductions come from exact token/vocabulary geometry, not an incomplete
sum of driver-forwarded records. Cross-rank p50 results that were dominated by
the two empty-rank streams (and suggested 94-99% latency reductions) are
intentionally not reported. A production-quality summary must fail on missing
ranks, report the maximum per-rank median for latency/peak memory, and sum
actual/dense logits bytes over the complete final-stage rank set.

#### Correctness evidence and limitation

Baseline and optimized runs injected identical cached trajectories. Prompt and
response lengths, actor entropy, rollout-correlation metrics, rewards, and
loss metrics matched step by step. The runs provide real TP=2, SP=true, CP=2,
PP=2 coverage, including CP-local empty masks and NCCL forward/backward
collective liveness.

The synthetic NIAH reward saturated: all two-rollout groups received reward 1,
so GRPO advantages, actor loss, and gradient norm were zero. These runs
therefore validate selected-token forward outputs and distributed liveness,
but do not establish non-zero-gradient multi-GPU backward parity. That remains
a submission gate and should be covered with a non-saturated reward workload
or a non-zero entropy coefficient using the same cached-token A/B procedure.

Because profiling synchronizes the accelerator and resets peak-memory stats,
the end-to-end numbers above are controlled engineering evidence rather than
production timing. A profiler-disabled, repeated/reversed-order A/B is
recommended before making a statistically rigorous throughput claim.

#### Functional-patch CPU tests

The tests shipped in the functional patch were run with:

```bash
python -m pytest -q \
  tests/models/mcore/test_response_only_lm_head_on_cpu.py \
  tests/workers/test_megatron_distillation_only_on_cpu.py \
  tests/workers/config/test_engine_config_on_cpu.py
```

Result: `24 passed`.

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
- configuration defaults.

The feature test file follows the repository's `*_on_cpu.py` convention, so it
is collected by the standard CPU CI job.

The local-only profiling patch has a separate summary test:

```bash
python -m pytest -q \
  tests/utils/test_response_only_lm_head_profile_summary_on_cpu.py
```

Result: `2 passed`. It covers dense/sparse byte accounting, JSON parsing,
per-rank warmup, latency/memory comparisons, and rejection of mismatched dense
workloads. This test and its profiling implementation are not part of the
functional PR.

Additional checks completed:

| Check | Result |
| --- | --- |
| Ruff check and format for changed Python files | PASS |
| Mypy for changed Python files | PASS |
| Generated Megatron trainer configuration | PASS |
| `git diff --check` | PASS |
| docs-time, docstrings, license, device API, DataProto, compileall | PASS |

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
  --warmup 1
```

The profiler synchronizes the accelerator and resets peak-memory statistics at
the LM-head boundary. Both A/B arms must use it, and results should not be mixed
with production step timing or process-wide peak-memory metrics.

For the four-step smoke test, `--warmup 1` leaves three samples per recurring
rank/pass stream. Before accepting an aggregate, verify that every expected
final-stage rank is present. Do not interpret a median across CP ranks as the
distributed critical path: report per-rank medians first, take the maximum for
latency and peak memory, and sum actual/dense logits bytes only over a complete
TP x CP rank set.

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
The A100 measurements above cover one real TP+SP+CP+PP topology. Additional
active ratios, BSHD, non-zero-gradient backward parity, and longer
profiler-disabled timing runs remain useful follow-ups.

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
- [x] Searched GitHub and current upstream source for materially overlapping
  work as of 2026-08-31.
- [ ] Re-run the repository-mandated duplicate checks with `gh` immediately
  before opening the PR; GitHub CLI was unavailable on the analysis host.
- [x] Added focused CPU tests and documentation.
- [x] Run real A100 TP=2/SP/CP=2/PP=2 A/B and CP-empty-rank liveness.
- [ ] Run real multi-accelerator non-zero-gradient backward parity.
- [ ] Repeat profiler-disabled timing in reversed A/B order if claiming
  statistically rigorous production throughput.
- [ ] Run full pre-commit in an environment that can fetch its remote hook
  environments. Local equivalents have passed.
- [ ] Rebase the functional commit onto current `upstream/main` and re-run
  tests; the validation branch is based on verl v0.9.0.
- [ ] Request CI through the verl Slack/Feishu process when ready.
- [x] Recipe-submodule update is not applicable.

### AI assistance

OpenAI Codex assisted with analysis, implementation, test design, profiling
instrumentation, and drafting. The human submitter must review every changed
line, run the distributed hardware validation, and remain responsible for the
submission.
