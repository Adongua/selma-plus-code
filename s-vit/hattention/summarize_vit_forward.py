"""Summarize full-forward reports with explicit correctness and timing scope."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(label, path):
    report = json.loads(Path(path).read_text())
    rows = [r for r in report["records"] if not r["warmup"]]
    if not rows:
        raise ValueError(f"{label}: no measured records")
    native = "online_ring_excluding_preprocessing" not in rows[0]
    mean = lambda values: statistics.mean(values)
    mean_at = lambda group, field: mean(r[group][field] for r in rows)
    segmented = report.get("preprocessing_mode") == "precomputed_segmented"
    instrumented = report.get("preprocessing_mode") == "inline_instrumented"
    segment_names = ([entry["name"] for entry in rows[0]["segmented_online"]["segments"]]
                     if segmented else None)
    if segmented and any([entry["name"] for entry in r["segmented_online"]["segments"]] != segment_names for r in rows):
        raise ValueError(f"{label}: measured repetitions use different segment boundaries")
    max_party_sum = mean_at("forward_wall", "seconds_max")
    segment_max_sum = (mean(r["segmented_online"]["sum_segment_max_seconds"] for r in rows)
                       if segmented else None)
    return {"label": label, "report": str(Path(path).resolve()), "status": report["status"],
            "valid": report["status"] == "benchmark_passed_not_security_validated",
            "backend": report["backend"], "world_size": report["world_size"],
            "image_sha256": report["image_sha256"], "precision": report["precision"],
            "truncation_method": report.get("truncation_method", "unreported"),
            "preprocessing_mode": report.get("preprocessing_mode", "inline"),
            "operator_profiling": report.get("operator_profiling", False),
            "offline_cache_storage": report.get("offline_cache_storage", "device" if report.get("preprocessing_mode") in ("precomputed_strict", "precomputed_segmented") else None),
            "crypten_preprocessing_seconds": (mean_at("crypten_preprocessing", "seconds_max")
                                               if report.get("preprocessing_mode") in ("precomputed_strict", "precomputed_segmented") else None),
            "inline_online_estimate_seconds": mean(r["inline_accounting"]["online_estimate_seconds_max"] for r in rows) if instrumented else None,
            "inline_provider_seconds_max": mean(max(p["crypten_provider_seconds"] for p in r["inline_accounting"]["per_rank"]) for r in rows) if instrumented else None,
            "inline_ring_seconds_max": mean(max(p["ring_preprocessing_seconds"] for p in r["inline_accounting"]["per_rank"]) for r in rows) if instrumented and not native else None,
            "sum_segment_max_seconds": segment_max_sum,
            "max_party_segment_sum_seconds": max_party_sum if segmented else None,
            "segment_names": segment_names,
            "repeats": len(rows), "forward_wall_seconds": segment_max_sum if segmented else max_party_sum,
            "ring_preprocessing_seconds": None if native else mean_at("ring_preprocessing", "seconds_max"),
            "forward_excluding_ring_preprocessing_seconds": mean_at(
                "online_crypten" if native else "online_ring_excluding_preprocessing", "seconds_max"),
            "crypten_system_bytes": mean_at("online_crypten", "system_bytes"),
            "online_ring_sent_bytes": None if native else mean_at("online_ring_excluding_preprocessing", "sent_bytes_sum"),
            "ring_preprocessing_sent_bytes": None if native else mean_at("ring_preprocessing", "sent_bytes_sum"),
            "max_logit_error": max(r["error_vs_plaintext_logits"]["max_abs"] or 0 for r in rows)
                               if all(r["error_vs_plaintext_logits"]["finite"] for r in rows) else None,
            "max_own_formula_logit_error": max(r["error_vs_own_formula_logits"]["max_abs"] or 0 for r in rows)
                               if all("error_vs_own_formula_logits" in r and r["error_vs_own_formula_logits"]["finite"] for r in rows) else None,
            "correctness_gate": report.get("correctness_gate", "max error against plaintext exact model")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True, help="new output directory")
    args = parser.parse_args()
    results = []
    for spec in args.report:
        if "=" not in spec:
            parser.error("--report must be LABEL=PATH")
        label, path = spec.split("=", 1)
        if not label or not path:
            parser.error("label and path must be nonempty")
        results.append(summarize(label, path))
    if len({r["label"] for r in results}) != len(results):
        parser.error("report labels must be unique")
    for key in ("backend", "world_size", "image_sha256"):
        if len({r[key] for r in results}) != 1:
            raise ValueError(f"reports have different {key}; comparison is not valid")
    if len({r["operator_profiling"] for r in results}) != 1:
        raise ValueError("reports differ in operator profiling; rerun under the same instrumentation")
    if len({r["preprocessing_mode"] for r in results}) != 1:
        raise ValueError("reports use different preprocessing modes; rerun under the same timing boundary")
    if results[0]["preprocessing_mode"] in ("precomputed_strict", "precomputed_segmented") and len({r["offline_cache_storage"] for r in results}) != 1:
        raise ValueError("reports use different material storage; compare under the same storage boundary")
    if results[0]["preprocessing_mode"] == "precomputed_segmented" and len({tuple(r["segment_names"]) for r in results}) != 1:
        raise ValueError("reports use different segment boundaries; comparison is not valid")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    segmented = results[0]["preprocessing_mode"] == "precomputed_segmented"
    strict = results[0]["preprocessing_mode"] in ("precomputed_strict", "precomputed_segmented")
    instrumented = results[0]["preprocessing_mode"] == "inline_instrumented"
    note = ("Mean of measured repetitions; maximum elapsed time across parties per repetition. "
            "Input sharing and output reconstruction excluded. " +
            ("Layer-segmented estimate: fresh materials were prepared between segments and excluded from the sum of measured online segments; "
             "extra synchronization and changes in timing state mean this is not uninterrupted end-to-end wall time. "
             "The table sums the slowest party's time per segment; the JSON also records the maximum party sum. "
             "CPU-to-GPU material transfers count online when CPU storage is used; local random generation remains online. "
             if segmented else
             "Fresh CrypTen provider and Ring3 materials were prepared before sharing and replayed once; "
             "forward wall is measured after explicit material preparation; CPU-to-GPU material transfers "
             "are included when CPU storage is used. Local random generation and "
             "other runtime setup in the forward remain online; no security validation is implied. "
             if strict else
             "Continuous forward with synchronized provider instrumentation; per-rank online estimate subtracts timed provider and Ring3 preparation from local wall before taking the maximum. Preparations remain interleaved, synchronization adds overhead, and local randomness outside timed calls remains online. CrypTen communication includes provider work. "
             if instrumented else
             "CrypTen provider work is included in forward; Ring3 preparation is separately counted. "
             "Full wall includes just-in-time Ring3 preparation, and forward excluding Ring3 preparation "
             "still includes CrypTen provider work. ") +
            "Communication counters, precision, and correctness-gate references may differ; "
            "results with failed numerical checks are diagnostic only.")
    report = {"scope": "full ViTLite forward", "note": note, "results": results}
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    header = ["# Full ViT forward comparison", "", note, ""]
    if instrumented:
        header.extend(["| Method | Status | Precision | CrypTen truncation | Forward wall (s) | Provider max (s) | Ring prep max (s) | Online estimate (s) | CrypTen total comm (MiB) | Ring online (MiB) | Max error vs exact | Max error vs own formula |",
                       "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    elif strict:
        timing_header = "Segmented online sum of maxima (s)" if segmented else "Online forward (s)"
        header.extend([f"| Method | Status | Precision | CrypTen truncation | Material storage | {timing_header} | CrypTen online comm (MiB) | Ring online (MiB) | Max error vs exact | Max error vs own formula |",
                       "|---|---|---|---|---|---:|---:|---:|---:|---:|"])
    else:
        header.extend(["| Method | Status | Precision | CrypTen truncation | Forward wall (s) | CrypTen prep (s) | Ring prep (s) | Forward excl. Ring prep (s) | CrypTen comm (MiB) | Ring online (MiB) | Max error vs exact | Max error vs own formula |",
                       "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for r in results:
        line = f"| {r['label']} | {r['status']} | {r['precision']} | {r['truncation_method']} | "
        if strict:
            line += f"{r['offline_cache_storage']} | "
        line += f"{fmt(r['forward_wall_seconds'])} | "
        if instrumented:
            line += (f"{fmt(r['inline_provider_seconds_max'])} | {fmt(r['inline_ring_seconds_max'])} | "
                     f"{fmt(r['inline_online_estimate_seconds'])} | ")
        elif not strict:
            line += (f"{fmt(r['crypten_preprocessing_seconds'])} | {fmt(r['ring_preprocessing_seconds'])} | "
                     f"{fmt(r['forward_excluding_ring_preprocessing_seconds'])} | ")
        line += (f"{fmt(r['crypten_system_bytes'] / 2**20)} | "
                 f"{fmt(None if r['online_ring_sent_bytes'] is None else r['online_ring_sent_bytes'] / 2**20)} | "
                 f"{fmt(r['max_logit_error'])} | {fmt(r['max_own_formula_logit_error'])} |")
        header.append(line)
    text = "\n".join(header) + "\n"
    (out / "comparison.md").write_text(text)
    print(text, end="")
    print(f"Reports: {out}")


if __name__ == "__main__":
    main()
