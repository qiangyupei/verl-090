<!-- Suggested title: [megatron] feat: project LM head only for response tokens -->

### What does this PR do?

Megatron currently applies the vocabulary projection to every token row in a packed or padded micro-batch, even though PPO losses often select only assistant tokens through `loss_mask`. Long agent trajectories may contain prompts, tool outputs, environment observations, and padding that do not contribute to the policy loss but still materialize vocabulary-sized logits.

This change adds an opt-in `response_only_lm_head` setting for the Megatron language-model engine, compatible with both unfused and fused output heads. It applies the same causal shift, padding, and context-parallel layout used for labels to the loss mask; selects active hidden rows before vocabulary projection or fused linear log-probability/entropy computation; and scatters the scalar outputs back to the existing token layout. Unselected positions are zero-filled and remain excluded by the existing loss mask. The unfused path also supports sum-of-squared-probability values.

The option defaults to `false`. Transformer, attention, and required sequence-parallel hidden-state communication are unchanged.

#### Why this is needed

For a processed token count `T`, padded vocabulary size `V`, tensor-parallel size `TP`, context-parallel size `CP`, and logits element size `b`, one dense vocabulary tensor occupies approximately:

```text
T * V * b / (TP * CP) bytes per final-pipeline-stage rank
```

If the active-row ratio is `r`, response-only projection reduces LM-head GEMM rows to approximately `r` of the dense path. On the unfused path, vocabulary tensor elements also fall to approximately `r`, saving approximately `1 - r` of logits storage. Unfused entropy computation clones logits, so the transient LM-head saving can be larger than one logits tensor.

Fused kernels such as [#462](https://github.com/verl-project/verl/pull/462) avoid materializing the full logits tensor, but still process every hidden row supplied to them. Fusion changes how the vocabulary computation is executed; response-only selection changes its row count from `T` to `K`. The two switches are independent: fused linear log-probability/entropy now receives only the selected rows when both are enabled. This reduces LM-head arithmetic and token-dependent workspace, although whole-step peak-memory savings depend on where the peak occurs. The unfused logits estimates below are not fused-path memory-saving estimates.

Example BF16 estimates for one logits tensor per rank:

| Processed shape | Dense logits | Saved at 10% active | Saved at 25% active | Saved at 50% active |
| --- | ---: | ---: | ---: | ---: |
| 32K tokens, V=152064, TP=4, CP=1 | 2.320 GiB | 2.088 GiB | 1.740 GiB | 1.160 GiB |
| 128K tokens, V=248320, TP=2, CP=8 | 3.789 GiB | 3.410 GiB | 2.842 GiB | 1.895 GiB |

These estimates cover vocabulary logits only, not whole-process peak memory.

### Checklist Before Starting

- [x] Searched open PRs for `response only LM head`, `active rows LM head`, `loss mask vocabulary projection`, and related Megatron/fused-kernel terms.
- [x] This does not duplicate [#7370](https://github.com/verl-project/verl/pull/7370), which implements the same row-selection principle only for the FSDP/FSDP2 Torch fused LM head and explicitly leaves other paths for follow-up work.
- [x] This does not duplicate [#5130](https://github.com/verl-project/verl/pull/5130), which integrates a Blackwell-specific fused linear cross-entropy path for Megatron.
- [x] This PR implements mask-aware row selection in the Megatron engine, including its existing Triton fused head and both native output-processor and legacy forward adapters. Its TP/SP/CP/PP integration differs from the FSDP/FSDP2 work in #7370 and the Blackwell kernel integration in #5130.

### Design and correctness

#### Causal alignment

`loss_mask` is response-only data. The implementation first builds a full `[prompt zeros; response mask]` sequence and then applies the same next-token roll as labels. A loss-bearing target token at position `t` therefore selects the predictor hidden state at `t - 1`. Internal zero spans, such as tool output, are preserved rather than treated as the end of the response.

The mask is passed through the same THD/BSHD preprocessing, FP8 padding, length-bucket padding, dynamic/static CP size, and zigzag/contiguous CP layout as the model inputs and labels.

#### Tensor and sequence parallelism

With sequence parallelism, Megatron's output layer normally gathers hidden states over TP before vocabulary projection. On the unfused path, the implementation performs that gather explicitly, selects active rows, and disables the output layer's second gather for that invocation. The explicit gather owns backward reduce-scatter; the output layer's non-SP input-gradient all-reduce is disabled to avoid double reduction. Without sequence parallelism, the regular ColumnParallelLinear gradient reduction remains unchanged.

The fused path selects rows after its SP gather and uses the same backward reduce-scatter. Without SP, `copy_to_tensor_model_parallel_region` sums the fused kernel's vocabulary-local hidden gradients over TP. Native and legacy fused adapters share this implementation and use the existing tied or untied output weight.

Each CP rank may have a different active-row count. A rank with no active rows projects one dummy row and restores a zero-gradient scalar path so all ranks retain the LM head in the distributed backward graph.

#### Output compatibility

Sparse labels are selected in the same order as hidden rows. The unfused path also selects per-token temperatures; the fused path retains its existing uniform-temperature requirement. After log-probability and entropy computation (and sum-pi-squared on the unfused path), token-level scalars are scattered back to the original CP-local dense layout. Existing CP postprocessing and downstream PPO loss code remain unchanged.

### Compatibility and limitations

- Supported data formats: THD on both paths; BSHD on the unfused path.
- Supported parallel dimensions: TP, SP, static/dynamic CP, and PP.
- `response_only_lm_head` and `use_fused_kernels` can be enabled independently. Existing fused-head requirements still apply: remove-padding, a uniform temperature, a supported CUDA/Triton stack, and no sum-pi-squared output. This change does not add an NPU fused kernel.
- MTP training is not supported.
- Top-k distillation is not supported.
- The optimization does not reduce transformer/attention work or the required SP hidden-state gather.
- Benefits become small when most token rows are active; a 100% active control should show approximately zero improvement.

### Test

#### A100 controlled A/B

The original unfused verl v0.9.0 adaptation was exercised on one node with 8 NVIDIA A100 GPUs using Qwen3-8B BF16, THD/remove-padding, TP=2, PP=2, CP=2, sequence parallelism enabled, and rollout TP=4. Actor/ref parameter offload was enabled. Each workload used one prompt with two rollouts per step and ran for four steps. A cache-fill run generated the trajectories; both measured arms injected the same two cached trajectories at every step and used identical synchronized measurement instrumentation. Both arms had fused kernels disabled. These measurements are retained as unfused evidence and do not establish the incremental benefit on top of fusion; the combined path requires a new A/B with fusion enabled in both arms.

Step 1 was treated as warmup. Mean results for steps 2-4 were:

| Workload | Actual prompt | Response | Baseline step | Optimized step | Step reduction | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | 8046-8052 | 8192 | 47.815 s | 47.249 s | 1.18% | 1.20% |
| prompt-heavy | 16230-16232 | 4096 | 54.398 s | 50.663 s | **6.86%** | **7.37%** |

The three prompt-heavy steady-step reductions were 7.03%, 6.88%, and 6.68%. The measured stage means were:

| Workload | Stage | Baseline | Optimized | Reduction |
| --- | --- | ---: | ---: | ---: |
| balanced | actor old-log-prob | 7.201 s | 6.546 s | 9.09% |
| balanced | ref log-prob | 7.777 s | 7.578 s | 2.56% |
| balanced | actor update | 19.267 s | 18.815 s | 2.34% |
| prompt-heavy | actor old-log-prob | 8.605 s | 7.253 s | **15.71%** |
| prompt-heavy | ref log-prob | 9.008 s | 8.421 s | **6.52%** |
| prompt-heavy | actor update | 22.918 s | 21.501 s | **6.18%** |

`actor old-log-prob` requests entropy in the v1 trainer. The measured actor update did not request entropy because both `actor.calculate_entropy` and `actor.entropy_coeff` were zero. Actor-update stage time also includes backward and optimizer-facing work.

#### LM-head latency and memory measurements

TP partitions vocabulary columns and CP partitions token rows. The following post-warmup medians were measured on the observed response-bearing final-pipeline-stage rank. Incremental peak memory is the peak allocated memory during LM-head and vocabulary processing minus the allocation immediately before that region; it is not whole-process or whole-device peak memory.

| Workload | Pass | Baseline latency | Optimized latency | Reduction | Baseline incremental peak | Optimized incremental peak | Memory reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | actor, no entropy | 180.80 ms | 112.79 ms | 37.61% | 6.894 GiB | 3.444 GiB | 50.04% |
| balanced | actor, entropy | 513.11 ms | 183.98 ms | 64.14% | 13.850 GiB | 6.827 GiB | 50.71% |
| balanced | ref, no entropy | 155.15 ms | 92.86 ms | 40.15% | 6.832 GiB | 3.382 GiB | 50.50% |
| prompt-heavy | actor, no entropy | 217.92 ms | 120.59 ms | **44.66%** | 8.629 GiB | 3.462 GiB | **59.88%** |
| prompt-heavy | actor, entropy | 635.70 ms | 165.71 ms | **73.93%** | 17.337 GiB | 6.878 GiB | **60.33%** |
| prompt-heavy | ref, no entropy | 206.73 ms | 99.05 ms | **52.09%** | 8.552 GiB | 3.400 GiB | **60.24%** |

The balanced workload had a global active-row ratio of about 50.44%, consistent with the approximately 50% reduction in LM-head incremental peak memory. The prompt-heavy workload had a global active-row ratio of about 20.15%; with CP=2, response rows were concentrated on one CP shard, giving the response-bearing rank a 40.30% local active-row ratio and approximately 60% lower incremental peak memory. The other observed CP shard had no active rows and projected only the required dummy row without hanging during forward or backward collectives.

Ray forwarded records for three of the expected four final-stage ranks. The per-rank measurements above therefore report the observed response-bearing rank rather than an incomplete cross-rank aggregate. The global active-row ratios come from exact token/vocabulary geometry, but these measurements do not claim a complete maximum across all final-stage ranks or a reduction in whole-process peak memory.

The synchronized instrumentation resets peak-memory statistics at the LM-head boundary. Consequently, the short end-to-end timings above are controlled engineering evidence rather than statistically rigorous production throughput measurements; a longer repeated and reversed-order A/B without synchronization is still recommended for a production throughput claim.

#### Correctness evidence and limitation

Baseline and optimized runs injected identical cached trajectories. Prompt and response lengths, actor entropy, rollout-correlation metrics, rewards, and loss metrics matched step by step. The runs provide real TP=2, SP=true, CP=2, PP=2 coverage, including CP-local empty masks and NCCL forward/backward collective liveness.

The synthetic NIAH reward saturated: all two-rollout groups received reward 1, so GRPO advantages, actor loss, and gradient norm were zero. These runs therefore validate selected-token forward outputs and distributed liveness, but do not establish non-zero-gradient multi-GPU backward parity. That remains a submission gate and should be covered with a non-saturated reward workload or a non-zero entropy coefficient using the same cached-token A/B procedure.

#### CPU tests

The functional patch keeps one focused test module:

```bash
python -m pytest -q tests/models/mcore/test_response_only_lm_head_on_cpu.py
```

It covers causal next-token mask alignment, unfused sparse/dense value and gradient parity (including an empty mask), SP gather/state restoration, and fused sparse/dense value and gradient parity for SP on/off. The fused test uses differentiable Torch math in place of the CUDA kernel; real fused-kernel execution remains in the separate validation patch below.

Before this test-only cleanup, the same implementation passed the broader CPU matrix (`72 passed`), including the native/legacy fused adapters, THD/BSHD layouts, empty/full/sparse masks, entropy on/off, and CP layout combinations. Those redundant combinations were removed from the community-facing functional patch rather than from the implementation.

#### Distributed fused-kernel correctness

The following regression runs the real Triton kernel with TP=2 and CP=2, compares full-row and response-only outputs and gradients against a full-vocabulary Torch reference, and covers SP on/off, FP32/BF16, entropy on/off, internal masked spans, and a CP rank with no active rows:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nproc_per_node=4 \
  -m pytest -q tests/special_distributed/test_response_only_lm_head.py
```

This GPU test must pass on the target training stack before claiming distributed fused backward correctness. It does not exercise an entire PP model; the cached-trajectory training A/B additionally covers TP=2, CP=2, PP=2 with SP enabled. Use a non-zero entropy coefficient to exercise non-zero gradients even when NIAH rewards saturate.

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

Actor and reference settings are independent. Enable only the roles whose log-probability or training passes should use sparse vocabulary projection.

To combine row selection with the existing fused head, also set `actor_rollout_ref.model.use_fused_kernels=true`. Keep the fused setting identical in baseline and response-only runs to measure the incremental row-selection benefit.

### Code changes

- add the output-layer hidden-row selection and sparse-output restoration helpers;
- construct next-token-aligned CP-local masks for THD and BSHD;
- integrate sparse labels, temperatures, log-probability, entropy, and sum-pi-squared processing;
- apply shared row selection before fused linear log-probability/entropy in native and legacy Megatron adapters;
- add opt-in engine/Hydra configuration and generated config;
- add focused correctness and gradient tests; keep profiling and distributed stress tests in the optional validation patch;
- document usage and limitations.

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md) and repository `AGENTS.md`.
- [ ] Run `pre-commit run --all-files --show-diff-on-failure --color=always`.
- [x] Add/update configuration documentation and usage examples.
- [x] Add focused CPU tests; retain the completed unfused A100 TP=2/SP/CP=2/PP=2 A/B.
- [ ] Run the new fused TP=2/CP=2 kernel test and fused TP=2/SP/CP=2/PP=2 training A/B on A100.
- [ ] Request CI through the verl `ci-request` channel when ready.
- [x] Recipe-submodule update is not applicable.

Codex assisted with this PR.
