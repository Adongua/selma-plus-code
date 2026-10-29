"""Public-data smoke check for an MPCViT checkpoint and searchable student."""

import argparse
import json
from pathlib import Path

import torch

from .search import search_layers
from .search_data import SearchDataset, collate_search, training_splits
from .student_checkpoint import load_student, make_checkpoint_search_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mpcvit-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if args.batch_size < 1 or args.batch_size > 16:
        parser.error("batch-size must be between 1 and 16")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    classes, weight_items, architecture_items, split = training_splits(args.dataset_root, seed=args.seed)
    source, checkpoint = load_student(args.checkpoint, args.mpcvit_root)
    student = make_checkpoint_search_student(source, prior_strength=args.prior_strength)
    source = source.to(args.device).eval()
    student = student.to(args.device).eval()
    batches = {}
    for name, items in (("weight", weight_items), ("architecture", architecture_items)):
        dataset = SearchDataset(args.dataset_root, items)
        inputs, _, targets = collate_search([dataset[i] for i in range(args.batch_size)])
        with torch.inference_mode():
            baseline_logits, _ = source(inputs.to(args.device))
            logits, features = student(inputs.to(args.device))
        if (logits.shape != (args.batch_size, 200) or not torch.isfinite(logits).all()
                or not torch.isfinite(baseline_logits).all()):
            raise ValueError(f"{name}: model produced invalid logits")
        batches[name] = {"sample_paths": [items[i][0] for i in range(args.batch_size)],
                         "targets": targets.tolist(), "logits_shape": list(logits.shape),
                         "feature_shape": list(features[-1].shape),
                         "baseline_logits_finite": True, "search_logits_finite": True}
    layers = search_layers(student)
    report = {"status": "smoke_passed", "scope": "plaintext public data, forward only",
              "checkpoint": checkpoint, "split": split, "class_order": classes,
              "search": {"layers": len(layers), "heads_per_layer": [l.config.num_heads for l in layers],
                         "prior_strength": args.prior_strength, "temperature": layers[0].temperature,
                         "prior_rsa_heads": sum(int(block.self_attn.alpha.sum())
                                                for block in source.classifier.blocks),
                         "attention_weight_dropout_removed": True},
              "preprocessing": "bicubic resize 64x64, MPCViT Tiny-ImageNet mean/std, no augmentation",
              "device": args.device, "batch_size": args.batch_size, "batches": batches,
              "limits": "No teacher, optimization, accuracy evaluation or secure computation; forward-only smoke."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "split": split, "search": report["search"], "batches": batches}, indent=2))


if __name__ == "__main__":
    main()
