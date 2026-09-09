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

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PREFIX = "VERL_RESPONSE_ONLY_LM_HEAD_PROFILE "
GIB = 1024**3
GROUP_FIELDS = (
    "backend",
    "role",
    "data_format",
    "calculate_entropy",
    "entropy_from_logits_with_chunking",
    "entropy_from_logits_chunk_size",
    "calculate_sum_pi_squared",
    "distillation_only",
    "tensor_parallel_size",
    "context_parallel_size",
    "pipeline_parallel_size",
    "logits_dtype",
    "vocab_shard",
)


def load_records(path: Path) -> list[dict]:
    records = []
    decoder = json.JSONDecoder()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = line.find(PREFIX)
        if start >= 0:
            record, _ = decoder.raw_decode(line[start + len(PREFIX) :])
            records.append(record)
    if not records:
        raise ValueError(f"no {PREFIX.strip()} records found in {path}")
    return records


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def group_key(record: dict) -> tuple:
    return tuple(record.get(field, "unfused" if field == "backend" else None) for field in GROUP_FIELDS)


def validate_records(records: list[dict], *, response_only_enabled: bool) -> None:
    modes = {bool(record["response_only_enabled"]) for record in records}
    if modes != {response_only_enabled}:
        raise ValueError(f"expected response_only_enabled={response_only_enabled}, found {sorted(modes)}")


def discard_warmup(records: list[dict], warmup: int) -> list[dict]:
    by_stream = defaultdict(list)
    for record in records:
        by_stream[(group_key(record), record["rank"])].append(record)

    measured = []
    for rows in by_stream.values():
        rows.sort(key=lambda row: row["event_id"])
        measured.extend(rows[warmup:])
    if not measured:
        raise ValueError(f"no profile stream has more than {warmup} warmup samples")
    return measured


def _median_optional(records: list[dict], field: str) -> float | None:
    values = [record[field] for record in records if record.get(field) is not None]
    return statistics.median(values) if values else None


def summarize(records: list[dict], warmup: int) -> dict[tuple, dict]:
    grouped = defaultdict(list)
    for record in discard_warmup(records, warmup):
        grouped[group_key(record)].append(record)

    summary = {}
    for key, rows in grouped.items():
        elapsed = [row["elapsed_s"] for row in rows]
        summary[key] = {
            "samples": len(rows),
            "ranks": len({row["rank"] for row in rows}),
            "active_ratio": statistics.median(row["active_ratio"] for row in rows),
            "latency_p50_s": percentile(elapsed, 0.5),
            "latency_p95_s": percentile(elapsed, 0.95),
            "dense_logits_gib": statistics.median(row["dense_logits_bytes"] for row in rows) / GIB,
            "actual_logits_gib": statistics.median(row["actual_logits_bytes"] for row in rows) / GIB,
            "logits_saved_gib": statistics.median(row["logits_saved_bytes"] for row in rows) / GIB,
            "peak_increment_gib": (
                value / GIB if (value := _median_optional(rows, "peak_increment_bytes")) is not None else None
            ),
            "peak_allocated_gib": (
                value / GIB if (value := _median_optional(rows, "peak_allocated_bytes")) is not None else None
            ),
        }
    return summary


def _optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_summary(name: str, summary: dict) -> None:
    print(f"\n{name}")
    print(
        "\t".join(GROUP_FIELDS) + "\tn\tranks\tactive_pct\tlatency_p50_ms\tlatency_p95_ms\t"
        "dense_logits_GiB\tactual_logits_GiB\tlogits_saved_GiB\tpeak_increment_GiB\tpeak_allocated_GiB"
    )
    for key, row in sorted(summary.items()):
        print(
            "\t".join(map(str, key)) + f"\t{row['samples']}\t{row['ranks']}\t"
            f"{100 * row['active_ratio']:.2f}\t{1000 * row['latency_p50_s']:.3f}\t"
            f"{1000 * row['latency_p95_s']:.3f}\t{row['dense_logits_gib']:.3f}\t"
            f"{row['actual_logits_gib']:.3f}\t{row['logits_saved_gib']:.3f}\t"
            f"{_optional(row['peak_increment_gib'])}\t{_optional(row['peak_allocated_gib'])}"
        )


def compare(baseline: dict, optimized: dict) -> dict[tuple, dict]:
    if baseline.keys() != optimized.keys():
        raise ValueError(
            "profile groups differ: "
            f"missing from optimized={sorted(baseline.keys() - optimized.keys())}, "
            f"missing from baseline={sorted(optimized.keys() - baseline.keys())}"
        )

    comparison = {}
    for key in baseline:
        before = baseline[key]
        after = optimized[key]
        for field in ("samples", "ranks"):
            if before[field] != after[field]:
                raise ValueError(f"profile {field} differ for {key}: {before[field]} != {after[field]}")
        if not math.isclose(before["active_ratio"], after["active_ratio"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"profile active ratios differ for {key}: {before['active_ratio']} != {after['active_ratio']}"
            )
        if before["dense_logits_gib"] != after["dense_logits_gib"]:
            raise ValueError(
                f"profile dense logits sizes differ for {key}: "
                f"{before['dense_logits_gib']} != {after['dense_logits_gib']}"
            )
        peak_saved = None
        if before["peak_increment_gib"] is not None and after["peak_increment_gib"] is not None:
            peak_saved = before["peak_increment_gib"] - after["peak_increment_gib"]
        comparison[key] = {
            "forward_latency_reduction_pct": 100 * (1 - after["latency_p50_s"] / before["latency_p50_s"]),
            "logits_saved_gib": before["actual_logits_gib"] - after["actual_logits_gib"],
            "peak_increment_saved_gib": peak_saved,
        }
    return comparison


def print_comparison(comparison: dict) -> None:
    print("\nbaseline -> response-only")
    print("\t".join(GROUP_FIELDS) + "\tforward_latency_reduction_pct\tlogits_saved_GiB\tpeak_increment_saved_GiB")
    for key, row in sorted(comparison.items()):
        print(
            "\t".join(map(str, key)) + f"\t{row['forward_latency_reduction_pct']:.2f}\t"
            f"{row['logits_saved_gib']:.3f}\t{_optional(row['peak_increment_saved_gib'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Response-only Megatron LM-head A/B profile logs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    baseline_records = load_records(args.baseline)
    optimized_records = load_records(args.optimized)
    validate_records(baseline_records, response_only_enabled=False)
    validate_records(optimized_records, response_only_enabled=True)
    baseline = summarize(baseline_records, args.warmup)
    optimized = summarize(optimized_records, args.warmup)
    print_summary("baseline", baseline)
    print_summary("response-only", optimized)
    print_comparison(compare(baseline, optimized))


if __name__ == "__main__":
    main()
