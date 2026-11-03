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
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    note = ("Mean of measured repetitions; maximum elapsed time across parties per repetition. "
            "Input sharing and output reconstruction excluded. CrypTen provider work is included "
            "in forward; Ring3 preparation is separately counted. Communication counters and "
            "precision and correctness-gate references differ; results with failed numerical checks are diagnostic only.")
    report = {"scope": "full ViTLite forward", "note": note, "results": results}
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    header = ["# Full ViT forward comparison", "", note, "",
              "| Method | Status | Precision | Forward wall (s) | Ring prep (s) | Forward excl. Ring prep (s) | CrypTen comm (MiB) | Ring online (MiB) | Max error vs exact | Max error vs own formula |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for r in results:
        header.append(f"| {r['label']} | {r['status']} | {r['precision']} | "
                      f"{fmt(r['forward_wall_seconds'])} | {fmt(r['ring_preprocessing_seconds'])} | "
                      f"{fmt(r['forward_excluding_ring_preprocessing_seconds'])} | "
                      f"{fmt(r['crypten_system_bytes'] / 2**20)} | "
                      f"{fmt(None if r['online_ring_sent_bytes'] is None else r['online_ring_sent_bytes'] / 2**20)} | "
                      f"{fmt(r['max_logit_error'])} | {fmt(r['max_own_formula_logit_error'])} |")
    text = "\n".join(header) + "\n"
    (out / "comparison.md").write_text(text)
    print(text, end="")
    print(f"Reports: {out}")


if __name__ == "__main__":
    main()
