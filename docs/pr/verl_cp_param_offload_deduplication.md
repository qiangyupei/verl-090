# [megatron] feat: deduplicate host parameter-offload replicas

<!--
Draft upstream PR body for the functional change in patches/0001 only.
patches/0002 is a local validation aid and is not part of the upstream PR.
Replace every Pending item with results from the submitter's environment before
requesting review.
-->

### What does this PR do?

Megatron parameter offload currently retains a host copy of every model shard
on each data- and context-parallel replica. Those ranks hold identical
parameters, so the copies are redundant and can exhaust host memory for large
models.

This PR adds the opt-in `deduplicate_param_offload` setting. Rank zero of each
replica group retains the host copy. Reload copies that shard to the owner's
device and broadcasts it to the remaining replicas. Dense parameters use the
DP x CP replica group; expert parameters use the expert-data-parallel group.
Gradient and optimizer offload are unchanged.

The option defaults to `false`, requires `param_offload=true`, and is not
supported with Megatron FSDP. For a replica group of size `r`, host parameter
storage for that group falls from `r` copies to one, a theoretical reduction
of `1 - 1/r`. MoE savings depend on the dense/expert split and their respective
replica-group sizes.

The functional change is based on verl `v0.9.0`
(`483b8a009ba3a97563edee3a19887e4862b8094a`). Profiling is intentionally not
included in the PR patch. A separate local-only validation patch records
tracked CPU parameter storage, worker RSS snapshots, replica-group sizes, and
model load/offload transition latency.

### Checklist Before Starting

- [x] Searched open PRs for
  [`deduplicate_param_offload`](https://github.com/verl-project/verl/pulls?q=is%3Apr+is%3Aopen+deduplicate_param_offload),
  [`deduplicate offload`](https://github.com/verl-project/verl/pulls?q=is%3Apr+is%3Aopen+deduplicate+offload), and
  [`parameter offload`](https://github.com/verl-project/verl/pulls?q=is%3Apr+is%3Aopen+%22parameter+offload%22).
- [x] No open PR found by those searches deduplicates Megatron host-side
  parameter replicas as of 2026-08-17.
- [x] [#6193](https://github.com/verl-project/verl/pull/6193) avoids a separate
  allocate-before-free host-memory peak, while
  [#5651](https://github.com/verl-project/verl/pull/5651) handles optimizer FP32
  master-weight offload. Neither implements replica-group ownership and reload
  broadcast.
- [x] The title follows `[{modules}] {type}: {description}`.

No issue is claimed as fixed, so issue-number-specific duplicate lookup is N/A.

### Patch layout and application

`0001` is the functional patch intended for review. `0002` depends on `0001`
and is only for collecting validation data:

```bash
git checkout v0.9.0

git apply --check --whitespace=error-all \
  patches/0001-deduplicate-host-parameter-offload-replicas.patch
git apply --whitespace=error-all \
  patches/0001-deduplicate-host-parameter-offload-replicas.patch

# Optional: local profiling only; do not include this patch in the feature PR.
git apply --check --whitespace=error-all \
  patches/0002-profile-parameter-offload-deduplication.patch
git apply --whitespace=error-all \
  patches/0002-profile-parameter-offload-deduplication.patch
```

Both patches use LF line endings. Apply them in this order to an unmodified
`v0.9.0` tree.

### Test

Checks run while adapting the patches:

| Command | Result |
| --- | --- |
| Sequential `git apply --check --whitespace=error-all` for `0001`, then `0002` on clean `v0.9.0` | PASS |
| Compare the fully applied tree with the two source commits | PASS; identical tree hash |
| `git diff --check v0.9.0..HEAD` | PASS |
| `python -m py_compile` for all changed Python files | PASS (Python 3.12) |
| `ruff check` and `ruff format --check` for all changed Python files | PASS |
| Generated Megatron trainer config compared with source Hydra config | PASS |
| `pytest -q tests/workers/config/test_engine_config_on_cpu.py -k deduplicate_param_offload` | PASS; 2 passed, 11 deselected |
| `pytest -q tests/utils/test_param_offload_profile_summary_on_cpu.py` | PASS; 4 passed (`0002` only) |

The following accelerator/dependency tests remain to be run in the target
Linux Megatron environment before requesting review:

| Command | Coverage | Result |
| --- | --- | --- |
| `pytest -q tests/utils/megatron/test_param_offload_deduplication.py` | Group selection, owner/non-owner offload, snapshot restore | Pending; local Windows environment has no `megatron-core` |
| `pytest -s -x tests/models/test_engine.py -k "test_actor_engine and megatron"` | Real DP x CP collective, HF parity, train, automatic/manual reload | Pending |
| Two-node Qwen3.6-35B-A3B run | NPU/HCCL, dense and expert buffers, checkpoint/export boundaries | Pending |
| `pre-commit run --all-files --show-diff-on-failure --color=always` | Repository checks | Pending |

### Two-node validation

Use the same checkpoint, cached trajectory token tensors, seeds, topology, rank
placement, and step count for the two measured arms:

| Arm | `param_offload` | `deduplicate_param_offload` | Profile | Purpose |
| --- | ---: | ---: | ---: | --- |
| cache-fill | true | true | false | Generate the fixed rollout cache; discard timing |
| baseline | true | false | true | Existing offload behavior with identical instrumentation |
| dedup | true | true | true | Deduplicated behavior |

Apply both patches for this validation. Pass
`VERL_PARAM_OFFLOAD_PROFILE=1` through Ray's `runtime_env.env_vars` for both
measured arms; setting it only in the driver shell is not sufficient for an
already-running two-node Ray cluster. Verify every measured step injects its
cached trajectory and does not generate or overwrite one.

Summarize the two logs with:

```bash
python examples/profile/summarize_param_offload.py \
  --baseline /shared/verl-offload-ab/baseline.log \
  --dedup /shared/verl-offload-ab/dedup.log \
  --warmup 2
```

Each recurring role/action/reason group must contain more events than the
warmup count. Use at least four measured steps with `--warmup 2`; for a
two-step smoke test, use `--warmup 1` and treat the single remaining sample as
functional evidence rather than a stable latency measurement.

The primary memory metric is tracked CPU parameter tensor storage. RSS is a
post-transition worker snapshot, not node physical-memory usage or a peak;
node/cgroup peak memory should be sampled externally if that claim is needed.
Latency is model load/offload transition latency with profiling enabled, not
end-to-end step latency.

For correctness, compare the baseline and dedup training metrics step by step
from the same cached token tensors, including actor loss, policy-gradient loss,
KL, and gradient norm. Report the actual maximum absolute/relative differences
and the tolerance used. Do not claim bitwise equivalence unless the target NPU
run actually demonstrates it.

| Role | Dense group | Expert group | Baseline host GiB | Dedup host GiB | Saved | Offload overhead | Reload overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| ref | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

| Correctness metric | Maximum difference | Tolerance | Result |
| --- | ---: | ---: | --- |
| Actor / policy-gradient loss | Pending | Pending | Pending |
| KL | Pending | Pending | Pending |
| Gradient norm | Pending | Pending | Pending |

### API and Usage Example

```yaml
actor_rollout_ref:
  actor:
    megatron:
      param_offload: true
      deduplicate_param_offload: true
```

The reference model inherits the actor setting by default and can be overridden
independently. Enabling deduplication without parameter offload raises a
configuration error.

### Design & Code Changes

- Build dense DP x CP and expert-DP replica groups after Megatron parallel-state
  initialization; exclude GTP/EGTP rematerialization axes when supported.
- Retain the pinned DDP CPU buffer only on group rank zero and broadcast the
  flat device buffer on reload.
- Apply the same owner/broadcast rule to frozen and forward-only non-DDP
  parameters; gradients and optimizer state keep their existing paths.
- Preserve replica-group information through checkpoint, weight export, and
  experimental separation snapshot/restore paths.
- Add focused config, helper, and real DP x CP engine coverage. No production
  profiling code is included in the feature PR.

### Checklist Before Submitting

- [ ] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md)
  and repository `AGENTS.md`.
- [ ] Re-run the repository-mandated duplicate-work commands with `gh` and
  record their output in the PR discussion.
- [ ] Replace every Pending runtime result above with actual evidence.
- [ ] Submit only the functional `0001` source diff, not either patch artifact
  file or the local profiling `0002` changes.
- [ ] Run required pre-commit and CI checks.
- [ ] Review and understand every changed line.

### AI assistance

OpenAI Codex assisted with repository exploration, implementation review,
v0.9.0 adaptation, test and benchmark design, and drafting this PR description.
The human submitter must review every changed line, understand and be able to
defend the design and trade-offs, and personally verify every reported result
before submission.
