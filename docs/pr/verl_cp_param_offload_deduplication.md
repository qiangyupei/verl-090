<!-- Suggested title: [megatron] feat: deduplicate host parameter-offload replicas -->

### What does this PR do?

Megatron parameter offload currently keeps one CPU copy of every parameter
shard on every data-parallel (DP) and context-parallel (CP) replica. The
parameters are identical within these replica groups, so retaining every host
copy is redundant.

This PR adds an opt-in `deduplicate_param_offload` setting. For each replica
group, only group rank zero retains the offloaded CPU parameter storage. On
reload, that rank copies the shard to its accelerator and broadcasts it to the
remaining replicas. Dense parameters use the DP x CP replica group, while
expert parameters use the expert-data-parallel group. Gradient and optimizer
offload are unchanged.

The option defaults to `false`, requires `param_offload=true`, and is not
supported with Megatron FSDP. Existing behavior is unchanged when it is
disabled.

#### Why this is needed

Host memory is increasingly a limiting resource for colocated agentic RL:

- Agentic workloads commonly use multi-turn conversations with tool calls,
  generating long trajectories. Their longer sequences substantially increase
  activation and rollout KV-cache pressure on the accelerator.
- Colocated sync/async training therefore often needs full parameter,
  gradient, and optimizer offload so that the rollout engine can reclaim
  accelerator memory between training phases.
- Long sequences also require larger CP degrees. CP partitions sequence
  activations, but it does not shard model weights: identical weight shards are
  replicated across CP ranks. Existing parameter offload moves every replica
  to host memory, so redundant host weight storage grows linearly with the
  DP x CP replica-group size.
- Actor and reference models can both be offloaded, doubling this redundant
  storage. Optimizer/gradient offload, rollout state, and deployments that also
  offload or swap KV cache compete for the same host-memory budget.

As a result, increasing CP to solve accelerator-memory pressure can move the
bottleneck to host memory. This PR removes only the redundant parameter
replicas; it does not change optimizer, gradient, or KV-cache storage. For a
replica group of size `r`, that group's offloaded parameter storage falls from
`r` copies to one, a theoretical reduction of `1 - 1/r`.

### Checklist Before Starting

- [x] Searched open PRs for
  [`deduplicate_param_offload`](https://github.com/verl-project/verl/pulls?q=is%3Apr+is%3Aopen+deduplicate_param_offload),
  [`context parallel parameter offload`](https://github.com/verl-project/verl/pulls?q=is%3Apr+is%3Aopen+%22context+parallel%22+%22parameter+offload%22), and
  [`parameter offload`](https://github.com/verl-project/verl/pulls?q=is%3Apr+is%3Aopen+%22parameter+offload%22).
- [x] No open PR found by those searches implements host parameter ownership
  per DP x CP / expert-DP replica group as of 2026-08-18.
- [x] This does not duplicate
  [#6193](https://github.com/verl-project/verl/pull/6193), which avoids an
  allocate-before-free host-memory peak, or
  [#5651](https://github.com/verl-project/verl/pull/5651), which offloads
  optimizer FP32 master weights. Neither deduplicates persistent host parameter
  replicas.

### Test

#### Four-node controlled A/B

Environment:

- Model: Qwen3.6-35B-A3B, BF16
- Hardware: 4 nodes x 8 Ascend NPUs, 32 GiB device memory per NPU
- Parallelism: TP=2, PP=4, CP=4, EP=4, ETP=1, world size=32
- Offload: actor/ref parameter offload enabled
- Workload: four training steps, 32 prompts x 5 responses per step
- Control: identical pre-generated trajectory cache, checkpoint, seeds,
  topology, and rank-to-host placement
- All measured runs reported four cache injections, zero cache misses, zero
  cache writes, and completed 4/4 steps without OOM or fatal errors.

The dense replica group size was 4 and the expert replica group size was 2.
Tracked CPU parameter tensor storage was:

| Scope | Baseline host GiB | Dedup host GiB | Saved GiB | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Actor dense parameters | 21.761 | 5.440 | 16.321 | 75.0% |
| Actor expert parameters | 120.000 | 60.000 | 60.000 | 50.0% |
| Actor total | 141.761 | 65.440 | 76.321 | 53.8% |
| Reference total | 141.761 | 65.440 | 76.321 | 53.8% |
| Actor + reference | 283.523 | 130.881 | **152.642** | **53.8%** |

Accelerator peak allocated memory was unchanged at 17.785 GiB,
as expected for a host-memory optimization.

End-to-end timing with identical profiling enabled in all measured arms was:

| Run | Mean step time | Difference from first baseline |
| --- | ---: | ---: |
| Baseline | 1232.59 s | - |
| Baseline repeat | 1211.60 s | -1.70% |
| Dedup | 1224.95 s | -0.62% |

Dedup was 1.10% slower than the repeated baseline and 0.62% faster than the
first baseline, so its result lies inside the observed 1.70% baseline
run-to-run spread. No measurable end-to-end performance regression was
observed.

Transition profiling showed the expected trade-off. The ref-model reload
median increased from 1.45 s to 1.98 s because reload now includes a
broadcast. Parameter offload medians decreased because only owners perform
D2H copies: actor offload medians decreased by about 14-16%, and ref offload
decreased from 1.56 s to 1.00 s. Two recurring rank log streams were not
forwarded by the Ray driver, so tail/max transition latency is intentionally
not reported; the end-to-end timing above includes all ranks and is the primary
performance result.

Training correctness was evaluated against both the first baseline and an
independent baseline repeat. All runs used the same cached tokens. Token
counts, policy-gradient loss, rewards, advantages, and prompt/response-length
statistics matched. Full determinism was disabled, so the baseline repeat
quantifies the NPU/MoE run-to-run variation:

| Metric over four steps | Max baseline vs repeat | Max baseline vs dedup |
| --- | ---: | ---: |
| Policy-gradient loss, absolute | 0 | 0 |
| Actor loss, absolute (relative) | 4.00e-7 (5.12%) | 3.70e-7 (4.74%) |
| KL loss, absolute (relative) | 4.01e-5 (5.14%) | 3.72e-5 (4.76%) |
| Entropy, absolute (relative) | 9.90e-5 (0.055%) | 7.83e-5 (0.043%) |
| Gradient norm, absolute (relative) | 9.21e-2 (47.06%) | 6.11e-2 (36.79%) |

The dedup differences did not exceed the run-to-run variation observed between
the two baseline runs. In particular, the largest gradient-norm deviation was
larger between the two baselines than between baseline and dedup. No training
correctness regression was observed over these four controlled steps.

#### Single-node host-OOM case

A second experiment used Qwen3.6-35B-A3B on one node with 1 TB host memory and
16 x 64 GiB Ascend NPUs, using TP=2, CP=8, EP=8, ETP=1 (world size=16, PP=1),
with actor and reference parameter offload. The existing offload path exhausted
host memory during the run. With `deduplicate_param_offload=true`, observed
host-memory peak stayed around 900 GB and the workload fit.

The four-node profile above measured approximately 5.44 GiB of unique dense
BF16 parameters and 60.00 GiB of unique expert BF16 parameters per model role.
For the single-node topology, the dense replica group has size 8 and the
expert-DP group has size 2. This gives the following parameter-storage estimate:

```text
baseline = 8 * 5.44 + 2 * 60.00 = 163.52 GiB
dedup    =     5.44 +     60.00 =  65.44 GiB

theoretical saving = 163.52 - 65.44 = 98.08 GiB
```

The 98.08 GiB value covers only redundant parameter tensor storage. The
reported <900 GB peak is whole-node memory and also includes optimizer and
gradient state, pinned/staging buffers, Ray processes, allocator overhead, and
rollout memory. The theoretical reduction is therefore not
expected to equal the observed peak difference exactly, but it explains why
removing parameter replicas changes this configuration from host OOM to fit.

#### Focused checks

| Command / check | Result |
| --- | --- |
| Apply functional patch to a clean verl v0.9.0 tree with `git apply --check --whitespace=error-all` | PASS |
| `git diff --check` and Python syntax checks for changed files | PASS |
| `ruff check` and `ruff format --check` for changed Python files | PASS |
| Generated Megatron trainer config compared with its Hydra source config | PASS |
| `pytest -q tests/workers/config/test_engine_config_on_cpu.py -k deduplicate_param_offload` | PASS; 2 passed, 11 deselected |
| Four-node baseline / baseline-repeat / dedup runs described above | PASS; all completed 4/4 steps |
| Single-node 16-NPU host-memory run | Baseline OOM; dedup peak <900 GB |

The feature patch also adds focused tests for replica-group selection,
owner/non-owner parameter offload and reload, snapshot restore, configuration
validation, and a real DP x CP Megatron engine path. The large-scale host-memory
behavior cannot be reproduced in CPU CI; the accelerator experiments above
cover that behavior.

Profiling used for these measurements is maintained as a separate validation
patch and is not included in this PR.

### API and Usage Example

```yaml
actor_rollout_ref:
  actor:
    megatron:
      param_offload: true
      deduplicate_param_offload: true
```

The reference model inherits the actor setting by default and can be
overridden independently:

```yaml
actor_rollout_ref:
  ref:
    megatron:
      param_offload: true
      deduplicate_param_offload: true
```

Enabling deduplication without parameter offload raises a configuration error.

### Design & Code Changes

- Build dense DP x CP and expert-DP replica groups after Megatron parallel-state
  initialization; exclude GTP/EGTP rematerialization axes when supported.
- Retain pinned DDP CPU parameter buffers only on group rank zero and broadcast
  flat device buffers on reload.
- Apply the same ownership/broadcast rule to frozen and forward-only non-DDP
  parameters.
- Preserve replica-group information across checkpoint save/load, weight
  export, and experimental separation snapshot/restore paths.
- Leave optimizer and gradient offload unchanged. When the feature is disabled,
  retain the existing offload/load path without additional collectives.
- Add focused configuration, utility, snapshot, and distributed engine tests.
  Production profiling code is not part of this PR.

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md)
  and repository `AGENTS.md`.
- [ ] Run `pre-commit run --all-files --show-diff-on-failure --color=always`.
- [x] Add/update configuration documentation and usage examples.
- [x] Add focused tests. Large-scale host-memory validation is hardware-only and
  is covered by the experiments above rather than CPU CI.
- [ ] Request CI through the verl `ci-request` channel when ready.
- [x] Recipe-submodule update is not applicable.

### AI assistance

Codex assisted with implementation review, log analysis, and drafting this
PR description. All changes have been reviewed line by line by the submitter.
