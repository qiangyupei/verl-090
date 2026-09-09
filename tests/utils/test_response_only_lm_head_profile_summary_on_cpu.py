# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest

from examples.profile import summarize_response_only_lm_head as summary


def _record(rank, event_id, *, enabled, elapsed, actual_bytes, peak_increment=100):
    dense_bytes = 1000
    return {
        "event_id": event_id,
        "rank": rank,
        "response_only_enabled": enabled,
        "role": "actor",
        "data_format": "thd",
        "calculate_entropy": True,
        "entropy_from_logits_with_chunking": False,
        "entropy_from_logits_chunk_size": 2048,
        "calculate_sum_pi_squared": False,
        "distillation_only": False,
        "tensor_parallel_size": 2,
        "context_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "logits_dtype": "torch.bfloat16",
        "vocab_shard": 128,
        "active_ratio": 0.25,
        "elapsed_s": elapsed,
        "dense_logits_bytes": dense_bytes,
        "actual_logits_bytes": actual_bytes,
        "logits_saved_bytes": dense_bytes - actual_bytes,
        "peak_increment_bytes": peak_increment,
        "peak_allocated_bytes": 2000,
    }


def test_log_summary_discards_warmup_per_rank_and_compares(tmp_path):
    baseline_records = []
    optimized_records = []
    for rank in range(2):
        baseline_records.extend(
            [
                _record(rank, 0, enabled=False, elapsed=9.0, actual_bytes=1000),
                _record(rank, 1, enabled=False, elapsed=2.0 + rank, actual_bytes=1000, peak_increment=500),
            ]
        )
        optimized_records.extend(
            [
                _record(rank, 0, enabled=True, elapsed=9.0, actual_bytes=250),
                _record(rank, 1, enabled=True, elapsed=0.5 + 0.25 * rank, actual_bytes=250, peak_increment=200),
            ]
        )

    paths = {"baseline": tmp_path / "baseline.log", "optimized": tmp_path / "optimized.log"}
    for name, records in (("baseline", baseline_records), ("optimized", optimized_records)):
        lines = [f"worker | {summary.PREFIX}{json.dumps(record)}" for record in records]
        paths[name].write_text("\n".join(lines), encoding="utf-8")

    baseline = summary.summarize(summary.load_records(paths["baseline"]), warmup=1)
    optimized = summary.summarize(summary.load_records(paths["optimized"]), warmup=1)
    key = summary.group_key(baseline_records[0])
    result = summary.compare(baseline, optimized)[key]

    assert baseline[key]["samples"] == optimized[key]["samples"] == 2
    assert result["forward_latency_reduction_pct"] == pytest.approx(75.0)
    assert result["logits_saved_gib"] == pytest.approx(750 / summary.GIB)
    assert result["peak_increment_saved_gib"] == pytest.approx(300 / summary.GIB)

    optimized[key]["dense_logits_gib"] *= 2
    with pytest.raises(ValueError, match="dense logits sizes differ"):
        summary.compare(baseline, optimized)


def test_validate_records_rejects_mixed_modes():
    records = [
        _record(0, 0, enabled=False, elapsed=1.0, actual_bytes=1000),
        _record(0, 1, enabled=True, elapsed=1.0, actual_bytes=250),
    ]

    with pytest.raises(ValueError, match="expected response_only_enabled=False"):
        summary.validate_records(records, response_only_enabled=False)


def test_fused_summary_measures_workspace_without_claiming_logits_savings():
    before = _record(0, 0, enabled=False, elapsed=2, actual_bytes=0, peak_increment=500)
    after = _record(0, 0, enabled=True, elapsed=1, actual_bytes=0, peak_increment=200)
    for record in (before, after):
        record.update(backend="fused", dense_logits_bytes=0, logits_saved_bytes=0)
    baseline = summary.summarize([before], warmup=0)
    optimized = summary.summarize([after], warmup=0)
    comparison = summary.compare(baseline, optimized)[summary.group_key(before)]
    assert comparison["forward_latency_reduction_pct"] == 50
    assert comparison["logits_saved_gib"] == 0
    assert comparison["peak_increment_saved_gib"] == pytest.approx(300 / summary.GIB)

    after["backend"] = "unfused"
    with pytest.raises(ValueError, match="profile groups differ"):
        summary.compare(baseline, summary.summarize([after], warmup=0))
