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
    return {"label": label, "report": str(Path(path).resolve()), "status": report["status"],
            "valid": report["status"] == "benchmark_passed_not_security_validated",
            "backend": report["backend"], "world_size": report["world_size"],
            "image_sha256": report["image_sha256"], "precision": report["precision"],
            "truncation_method": report.get("truncation_method", "unreported"),
            "preprocessing_mode": report.get("preprocessing_mode", "inline"),
            "offline_cache_storage": report.get("offline_cache_storage", "device" if report.get("preprocessing_mode") == "precomputed_strict" else None),
            "crypten_preprocessing_seconds": (mean_at("crypten_preprocessing", "seconds_max")
                                               if report.get("preprocessing_mode") == "precomputed_strict" else None),
            "repeats": len(rows), "forward_wall_seconds": mean_at("forward_wall", "seconds_max"),
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
    if len({r["preprocessing_mode"] for r in results}) != 1:
        raise ValueError("reports mix inline and precomputed preprocessing; rerun under the same timing boundary")
    if results[0]["preprocessing_mode"] == "precomputed_strict" and len({r["offline_cache_storage"] for r in results}) != 1:
        raise ValueError("reports use different material storage; compare under the same storage boundary")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    strict = results[0]["preprocessing_mode"] == "precomputed_strict"
    note = ("Mean of measured repetitions; maximum elapsed time across parties per repetition. "
            "Input sharing and output reconstruction excluded. " +
            ("Fresh CrypTen provider and Ring3 materials were prepared before sharing and replayed once; "
             "forward wall is measured after explicit material preparation; CPU-to-GPU material transfers "
             "are included when CPU storage is used. Local random generation and "
             "other runtime setup in the forward remain online; no security validation is implied. "
             if strict else
             "CrypTen provider work is included in forward; Ring3 preparation is separately counted. "
             "Full wall includes just-in-time Ring3 preparation, and forward excluding Ring3 preparation "
             "still includes CrypTen provider work. ") +
            "Communication counters, precision, and correctness-gate references may differ; "
            "results with failed numerical checks are diagnostic only.")
    report = {"scope": "full ViTLite forward", "note": note, "results": results}
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    header = ["# Full ViT forward comparison", "", note, ""]
    if strict:
        header.extend(["| Method | Status | Precision | CrypTen truncation | Material storage | Online forward (s) | CrypTen online comm (MiB) | Ring online (MiB) | Max error vs exact | Max error vs own formula |",
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
        if not strict:
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
