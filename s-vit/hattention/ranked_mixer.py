"""Export and distill fixed-RSA students ranked by learned LSA alpha.

All evaluation and training is plaintext. Grouped attention costs are table
estimates, not secure end-to-end timings. Each candidate starts from the same
search checkpoint and uses the same recovery data order and random seed.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .group_costs import load_complete_tables
from .search import export_fixed_student, search_layers
from .search_data import SearchDataset, training_splits
from .search_mixer import balanced_items, measured_cost, recover
from .student_checkpoint import (configure_fixed_rsa_search, file_sha256, load_student,
                                 make_checkpoint_search_student)
from .teacher_guidance import load_teacher
from .train_search import evaluate, save_checkpoint, split_architecture_items, write_json
from .validate_teacher import validation_items


def ranked_paths(layers, mixer_count):
    """Return fixed-RSA paths, selecting lowest LSA alpha with stable ties."""
    ranked, paths = [], []
    for layer_index, layer in enumerate(layers):
        if not layer.fixed_rsa_positions or layer.lsa_alpha is None:
            raise ValueError("ranking requires fixed-RSA direct-alpha layers")
        allowed = layer.allowed_paths.detach().cpu()
        alpha = layer.lsa_alpha.detach().cpu()
        row = []
        for head, value in enumerate(alpha.tolist()):
            if bool(allowed[head, 0]):
                row.append("rsa")
            elif bool(allowed[head, 1]) and bool(allowed[head, 2]):
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("alpha must be finite and in [0,1]")
                row.append("lsa")
                ranked.append((value, layer_index, head))
            else:
                raise ValueError("unexpected fixed-RSA path mask")
        paths.append(row)
    if not 0 <= mixer_count <= len(ranked):
        raise ValueError("Mixer count exceeds searchable LSA heads")
    ranked.sort()
    selected = ranked[:mixer_count]
    for _, layer, head in selected:
        paths[layer][head] = "mixer"
    return paths, [{"layer": layer + 1, "head": head, "lsa_alpha": alpha}
                   for alpha, layer, head in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("search-checkpoint", "checkpoint", "mpcvit-root", "dataset-root",
                 "teacher-dir", "cost-table", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mixer-counts", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--recovery-steps", type=int, default=20)
    parser.add_argument("--recovery-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--eval-split", choices=("search_holdout", "official_val"), default="search_holdout")
    parser.add_argument("--lambda-kd", type=float, default=1.)
    parser.add_argument("--lambda-feature", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (not args.mixer_counts or len(set(args.mixer_counts)) != len(args.mixer_counts)
            or any(count < 0 for count in args.mixer_counts) or args.recovery_steps < 0
            or min(args.batch_size, args.eval_batch_size, args.eval_samples) < 1
            or not math.isfinite(args.recovery_lr) or args.recovery_lr <= 0
            or any(not math.isfinite(x) or x < 0 for x in (args.lambda_kd, args.lambda_feature))):
        parser.error("invalid counts, sample sizes, recovery parameters or loss weights")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    root = args.output_root.resolve()
    if root.exists():
        parser.error("output-root already exists; choose a new directory")
    artifact = torch.load(args.search_checkpoint, map_location="cpu", weights_only=True)
    config = artifact.get("config", {})
    if (artifact.get("format") != "hattention_search_checkpoint_v1"
            or config.get("optimization") != "joint_alpha_lasso_v1"
            or config.get("search_space") != "fixed_rsa_lsa_mixer"):
        raise ValueError("expected a fixed-RSA joint-alpha search checkpoint")
    if config.get("checkpoint_sha256") != file_sha256(args.checkpoint):
        raise ValueError("original MPCViT checkpoint hash mismatch")
    if config.get("cost_table_sha256") != file_sha256(args.cost_table):
        raise ValueError("cost table hash mismatch")
    classes, weight_items, architecture_items, split = training_splits(args.dataset_root, seed=config["seed"])
    if split != config.get("training_split"):
        raise ValueError("training split mismatch")
    _, heldout, holdout_meta = split_architecture_items(
        architecture_items, validation_per_class=config["validation_per_class"], seed=config["seed"])
    if holdout_meta != config.get("search_holdout"):
        raise ValueError("search holdout mismatch")
    if args.eval_split == "official_val":
        items = balanced_items(validation_items(args.dataset_root.resolve(), classes), args.eval_samples, args.seed)
        scope = "official validation subset; do not use as repeated tuning set"
    else:
        if args.eval_samples > len(heldout):
            parser.error("eval-samples exceeds search holdout")
        items = heldout[:args.eval_samples]
        scope = "search held-out training images; may have been used by pretrained MPCViT"
    dataset = SearchDataset(args.dataset_root, items)
    weights = SearchDataset(args.dataset_root, weight_items)
    teacher, teacher_meta = load_teacher(args.teacher_dir, args.dataset_root, mapping=config["mapping"])
    for name in ("weights", "config", "processor"):
        if teacher_meta[name + "_sha256"] != config["teacher_" + name + "_sha256"]:
            raise ValueError("teacher " + name + " hash mismatch")
    source, source_meta = load_student(args.checkpoint, args.mpcvit_root)
    model = make_checkpoint_search_student(source)
    configure_fixed_rsa_search(model, source, mixer_probability=config["mixer_init_probability"],
                               direct_alpha=True)
    model.search_feature_projection = nn.Linear(192, teacher_meta["teacher_hidden_size"])
    model.load_state_dict(artifact["model"], strict=True)
    layers = search_layers(model)
    for count in args.mixer_counts:
        ranked_paths(layers, count)
    tables, metadata = load_complete_tables(args.cost_table)
    if metadata.get("shape_b_h_n_d") != [1, 12, 257, 16] or metadata.get("world_size") != 3:
        raise ValueError("cost table must match three-party shape 1x12x257x16")
    table = tables["online_seconds"]
    device = torch.device(args.device)
    teacher = teacher.to(device).eval()
    root.mkdir(parents=True, exist_ok=False)
    manifest = hashlib.sha256()
    for path, target in items:
        manifest.update(f"{path}\t{target}\n".encode())
    report = {"status": "running", "search_checkpoint_sha256": file_sha256(args.search_checkpoint),
              "source_checkpoint_sha256": source_meta["sha256"], "cost_table_sha256": file_sha256(args.cost_table),
              "search_checkpoint": str(args.search_checkpoint.resolve()),
              "search_completed_epochs": artifact.get("completed_epochs"), "device": args.device,
              "teacher": teacher_meta, "mixer_counts": args.mixer_counts,
              "evaluation": {"scope": scope, "samples": len(items), "manifest_sha256": manifest.hexdigest()},
              "recovery": {"steps": args.recovery_steps, "lr": args.recovery_lr, "seed": args.seed,
                           "batch_size": args.batch_size, "lambda_kd": args.lambda_kd,
                           "lambda_feature": args.lambda_feature, "optimizer": "AdamW",
                           "gradient_clip_norm": 1.0},
              "fixed_rsa_observations": [
                  {"layer": index + 1, "threshold": layer.denominator_min,
                   "heads_unobserved_or_below_threshold": [
                       head for head, (allowed, observed) in enumerate(zip(
                           layer.allowed_paths[:, 0].detach().cpu().tolist(),
                           layer.observed_min_denominator.detach().cpu().tolist()))
                       if allowed and (not math.isfinite(observed) or observed < layer.denominator_min)]}
                  for index, layer in enumerate(layers)],
              "notes": "Plaintext search and evaluation; attention-core table costs exclude preprocessing and other ViT operators. "
                       "RSA range and Ruler privacy are not validated. No automatic selection or accuracy guarantee.",
              "candidates": []}
    write_json(root / "report.json", report)
    for count in args.mixer_counts:
        paths, selected = ranked_paths(layers, count)
        candidate = export_fixed_student(model, paths).to(device)
        before = evaluate(candidate, dataset, device, args.eval_batch_size, len(dataset))
        before["scope"] = scope
        info = {"format": "hattention_fixed_student_v1", "state_dict": candidate.state_dict(),
                "head_types": paths, "student_init": "checkpoint", "source_checkpoint_sha256": source_meta["sha256"],
                "search_checkpoint_sha256": report["search_checkpoint_sha256"]}
        save_checkpoint(root / f"mixer-{count:03d}-before.pt", info)
        print(json.dumps({"event": "candidate_before", "mixer_count": count, "evaluation": before}), flush=True)
        recovery = recover(candidate, teacher, weights, device, steps=args.recovery_steps,
                           batch_size=args.batch_size, lr=args.recovery_lr, seed=args.seed,
                           lambda_kd=args.lambda_kd, lambda_feature=args.lambda_feature)
        after = evaluate(candidate, dataset, device, args.eval_batch_size, len(dataset))
        after["scope"] = scope
        info["state_dict"] = candidate.state_dict()
        save_checkpoint(root / f"mixer-{count:03d}-after.pt", info)
        row = {"mixer_count": count, "path_counts": dict(Counter(path for layer in paths for path in layer)),
               "head_types": paths, "selected": selected, "estimated_online_seconds": measured_cost(paths, table),
               "before": before, "recovery": recovery, "after": after,
               "before_artifact": f"mixer-{count:03d}-before.pt", "after_artifact": f"mixer-{count:03d}-after.pt"}
        report["candidates"].append(row)
        write_json(root / "report.json", report, overwrite=True)
        print(json.dumps({"event": "candidate_complete", **row}), flush=True)
        del candidate
    report["status"] = "completed_plaintext_not_security_validated"
    write_json(root / "report.json", report, overwrite=True)
    print(json.dumps({"event": "complete", "report": str(root / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
