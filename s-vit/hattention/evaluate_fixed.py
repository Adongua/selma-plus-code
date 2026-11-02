"""Evaluate a fixed hAttention student on the Tiny ImageNet validation split."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from .search_data import SearchDataset
from .search_mixer import balanced_items
from .student_checkpoint import load_fixed_student
from .train_search import evaluate, write_json
from .validate_teacher import validation_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed-student", "checkpoint", "mpcvit-root", "dataset-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the entire validation split")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_samples < 0 or args.seed < 0:
        parser.error("invalid sample, seed, or batch settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.output.exists():
        parser.error("output exists; choose a new file")
    root = args.dataset_root.resolve()
    classes = sorted((root / "wnids.txt").read_text(encoding="utf-8").splitlines())
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError("expected 200 unique Tiny ImageNet classes")
    items = validation_items(root, classes)
    total_items = len(items)
    if args.max_samples > total_items:
        parser.error("max-samples exceeds validation set")
    if args.max_samples:
        items = balanced_items(items, args.max_samples, args.seed)
    manifest = hashlib.sha256()
    for path, label in items:
        manifest.update(f"{label}\t{path}\n".encode())
    artifact = torch.load(args.fixed_student, map_location="cpu", weights_only=True)
    saved_classes = artifact.get("distillation_settings", {}).get("class_order")
    if saved_classes is not None and saved_classes != classes:
        raise ValueError("validation dataset class order differs from distillation")
    model, meta = load_fixed_student(args.fixed_student, args.checkpoint, args.mpcvit_root)
    device = torch.device(args.device)
    model = model.to(device)
    metrics = evaluate(model, SearchDataset(root, items), device, args.batch_size, len(items))
    metrics["scope"] = "official validation split" if len(items) == total_items else "balanced official validation subset"
    report = {"status": "evaluated_plaintext", "evaluation": metrics,
              "sample_manifest_sha256": manifest.hexdigest(), "seed": args.seed,
              "fixed_student_sha256": meta["sha256"], "source_checkpoint_sha256": meta["source"]["sha256"],
              "head_types": meta["head_types"],
              "path_counts": dict(Counter(p for layer in meta["head_types"] for p in layer)),
              "note": "Repeatedly selecting models using validation results makes this a tuning set; accuracy does not establish secure inference correctness."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"report": str(args.output), "evaluation": metrics, "path_counts": report["path_counts"]}), flush=True)


if __name__ == "__main__":
    main()
