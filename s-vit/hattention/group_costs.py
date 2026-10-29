"""Build transparent grouped-head cost tables from measured hybrid reports."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from .config import normalize_head_types

METADATA = ("backend", "world_size", "shape_b_h_n_d", "patch_grid", "has_cls_token",
            "rho", "rsa_profile", "input_precision_bits", "ring3_precision_bits",
            "fixture", "input_std", "seed", "repeats", "max_abs_error_threshold",
            "python", "torch", "crypten_import", "scope", "timing", "communication")
METRICS = ("online_seconds", "online_bytes", "online_phases", "offline_seconds", "offline_bytes")


def assemble_reports(paths):
    """Use only correctness-passed Ring3 hybrid measurements, no interpolation."""
    if not paths:
        raise ValueError("at least one comparison.json is required")
    metadata, entries, heads = None, {}, None
    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get("status") != "benchmark_passed_not_security_validated":
            raise ValueError(f"{path}: benchmark did not pass correctness")
        current = {key: report.get(key) for key in METADATA}
        if any(current[key] is None for key in METADATA if key != "patch_grid"):
            raise ValueError(f"{path}: missing benchmark metadata")
        if current["world_size"] != 3 or current["ring3_precision_bits"] != 17:
            raise ValueError(f"{path}: expected three-party Q17 Ring3")
        if metadata is not None and current != metadata:
            raise ValueError(f"{path}: incompatible measurement metadata")
        metadata = current
        shape = current["shape_b_h_n_d"]
        if len(shape) != 4 or any(not isinstance(x, int) or x <= 0 for x in shape):
            raise ValueError(f"{path}: invalid shape")
        heads = shape[1]
        paths_for_head = normalize_head_types(report.get("head_types", {}).get("hybrid", []), heads)
        key = (paths_for_head.count("rsa"), paths_for_head.count("mixer"))
        if key in entries:
            raise ValueError(f"{path}: duplicate count combination {key}")
        rows = [row["methods"]["hybrid"] for row in report["records"] if not row["warmup"]]
        if len(rows) != current["repeats"] or not rows or not all(row["correctness_passed"] for row in rows):
            raise ValueError(f"{path}: missing or invalid measured repetitions")
        metrics = {"online_seconds": statistics.mean(r["online"]["seconds_max"] for r in rows),
                   "online_bytes": statistics.mean(r["online"]["system_bytes"] for r in rows),
                   "online_phases": statistics.mean(max(r["online"]["rounds_per_rank"]) for r in rows),
                   "offline_seconds": statistics.mean(r["offline"]["seconds_max"] for r in rows),
                   "offline_bytes": statistics.mean(r["offline"]["system_bytes"] for r in rows)}
        if any(not math.isfinite(v) or v < 0 for v in metrics.values()):
            raise ValueError(f"{path}: invalid cost value")
        entries[key] = {"rsa_heads": key[0], "mixer_heads": key[1], "head_types": list(paths_for_head),
                        "metrics": metrics, "source": str(path.resolve()),
                        "source_sha256": hashlib.sha256(raw).hexdigest(), "origin": "measured"}
    required = {(i, j) for i in range(heads + 1) for j in range(heads + 1 - i)}
    missing = sorted(required - entries.keys())
    return {"format": "hattention_group_costs_v1", "metadata": metadata,
            "complete": not missing, "missing_counts": [list(x) for x in missing],
            "entries": [entries[key] for key in sorted(entries)],
            "notes": "Measured grouped attention core only; no projection cost, interpolation or security proof. "
                     "Counts assume costs depend on path counts, not head ordering; verify on the target system."}


def load_complete_tables(path):
    """Load only a complete measured table; caller maps it to compatible layers."""
    report = json.loads(Path(path).read_text())
    if report.get("format") != "hattention_group_costs_v1" or report.get("complete") is not True:
        raise ValueError("complete measured group-cost table required")
    heads = report["metadata"]["shape_b_h_n_d"][1]
    tables = {metric: torch.full((heads + 1, heads + 1), float("nan"), dtype=torch.float64)
              for metric in METRICS}
    seen = set()
    for entry in report["entries"]:
        i, j = entry["rsa_heads"], entry["mixer_heads"]
        if (i, j) in seen or i < 0 or j < 0 or i + j > heads or entry.get("origin") != "measured":
            raise ValueError("invalid or duplicated table entry")
        seen.add((i, j))
        for metric in METRICS:
            value = entry["metrics"][metric]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid cost value")
            tables[metric][i, j] = value
    if len(seen) != (heads + 1) * (heads + 2) // 2:
        raise ValueError("table is missing combinations")
    return tables, report["metadata"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    try:
        table = assemble_reports(args.reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(table, target, indent=2)
            target.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot build group-cost table: {error}\n")
    print(json.dumps({"output": str(args.output), "complete": table["complete"],
                      "measured": len(table["entries"]), "missing": len(table["missing_counts"])}))


if __name__ == "__main__":
    main()
