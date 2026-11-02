"""Distill a layer-search fixed student on public data; no secure inference."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch
from torch import nn

from .search import export_fixed_student
from .search_data import SearchDataset, training_splits
from .student_checkpoint import (configure_fixed_rsa_search, file_sha256, load_fixed_student,
                                 load_student, make_checkpoint_search_student)
from .teacher_guidance import load_teacher
from .train_search import evaluate, loader, save_checkpoint, split_architecture_items, write_json


def verify_export(search_artifact, fixed, selection, checkpoint, mpcvit_root):
    """Verify the fixed state actually comes from the selected search epoch."""
    config = search_artifact.get("config", {})
    if (search_artifact.get("format") != "hattention_search_checkpoint_v1"
            or config.get("search_space") != "fixed_rsa_layer_lsa_mixer"
            or config.get("optimization") != "joint_layer_alpha_lasso_v1"):
        raise ValueError("expected a layer-level fixed-RSA search checkpoint")
    if selection.get("head_types") != fixed["head_types"]:
        raise ValueError("selection and fixed-student head paths differ")
    if Path(selection.get("source_checkpoint", "")).name != Path(search_artifact["_path"]).name:
        raise ValueError("selection source checkpoint filename differs")
    if selection.get("cost_table_sha256") != config.get("cost_table_sha256"):
        raise ValueError("selection and search cost table hashes differ")
    source, meta = load_student(checkpoint, mpcvit_root)
    if config.get("checkpoint_sha256") != meta["sha256"]:
        raise ValueError("search and original student checkpoint hashes differ")
    model = make_checkpoint_search_student(source)
    configure_fixed_rsa_search(model, source, mixer_probability=config["mixer_init_probability"],
                               direct_alpha=True, layer_alpha=True)
    state = search_artifact["model"]
    projection = state.get("search_feature_projection.weight")
    if projection is None or projection.ndim != 2 or projection.shape[1] != 192:
        raise ValueError("invalid search feature projection")
    model.search_feature_projection = nn.Linear(192, projection.shape[0])
    model.load_state_dict(state, strict=True)
    exported = export_fixed_student(model, fixed["head_types"]).state_dict()
    actual = fixed["state_dict"]
    if exported.keys() != actual.keys() or any(not torch.equal(exported[k].cpu(), actual[k].cpu())
                                                for k in exported):
        raise ValueError("fixed student weights do not match the selected search checkpoint")
    return config


def train_epoch(model, teacher, dataset, device, optimizer, *, epoch, batch_size,
                max_steps, seed, lambda_kd, lambda_feature, log_every):
    model.train()
    totals = {"total": 0., "ce": 0., "kd": 0., "feature": 0.}
    steps = samples = 0
    for inputs, images, labels in loader(dataset, batch_size, shuffle=True, seed=seed + epoch):
        if max_steps and steps >= max_steps:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = teacher.loss(model, images, inputs, labels,
                                        lambda_kd=lambda_kd, lambda_feature=lambda_feature)
        if not torch.isfinite(loss):
            raise ValueError("non-finite distillation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        samples += labels.numel()
        values = {"total": float(loss.detach()), **{k: float(v.detach()) for k, v in components.items()}}
        for key in totals:
            totals[key] += values[key]
        if log_every and steps % log_every == 0:
            print(json.dumps({"event": "distill_step", "epoch": epoch, "step": steps, "loss": values}), flush=True)
    if not steps:
        raise ValueError("empty training dataset")
    return {"steps": steps, "samples": samples, "loss": {k: v / steps for k, v in totals.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed-student", "search-checkpoint", "selection", "checkpoint", "mpcvit-root",
                 "dataset-root", "teacher-dir", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="resume an epoch checkpoint in a new output directory")
    parser.add_argument("--epochs", type=int, required=True, help="total epochs including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-feature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.eval_batch_size, args.eval_samples) < 1
            or args.max_steps_per_epoch < 0 or args.log_every < 0 or args.seed < 0
            or not math.isfinite(args.lr) or args.lr <= 0
            or any(not math.isfinite(v) or v < 0 for v in (args.lambda_kd, args.lambda_feature))):
        parser.error("invalid training settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    root = args.output_root.resolve()
    if root.exists():
        parser.error("output-root already exists; use a new directory")
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    search = torch.load(args.search_checkpoint, map_location="cpu", weights_only=True)
    search["_path"] = str(args.search_checkpoint.resolve())
    fixed = torch.load(args.fixed_student, map_location="cpu", weights_only=True)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    search_config = verify_export(search, fixed, selection, args.checkpoint, args.mpcvit_root)
    del search, fixed
    classes, weight_items, architecture_items, split = training_splits(args.dataset_root, seed=search_config["seed"])
    _, heldout, holdout_meta = split_architecture_items(
        architecture_items, validation_per_class=search_config["validation_per_class"], seed=search_config["seed"])
    if split != search_config["training_split"] or holdout_meta != search_config["search_holdout"]:
        raise ValueError("training split or search holdout differs from search checkpoint")
    if args.eval_samples > len(heldout):
        parser.error("eval-samples exceeds search holdout")
    teacher, teacher_meta = load_teacher(args.teacher_dir, args.dataset_root, mapping=search_config["mapping"])
    for name in ("weights", "config", "processor"):
        if teacher_meta[name + "_sha256"] != search_config["teacher_" + name + "_sha256"]:
            raise ValueError("teacher " + name + " hash differs from search")
    if teacher_meta["class_order"] != classes:
        raise ValueError("teacher and student dataset class order differs")
    model, source_meta = load_fixed_student(args.fixed_student, args.checkpoint, args.mpcvit_root)
    if model.search_feature_projection.out_features != teacher_meta["teacher_hidden_size"]:
        raise ValueError("teacher and student feature dimensions differ")
    model, teacher = model.to(device), teacher.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_data = SearchDataset(args.dataset_root, weight_items)
    eval_data = SearchDataset(args.dataset_root, heldout[:args.eval_samples])
    settings = {"fixed_student_sha256": file_sha256(args.fixed_student),
                "search_checkpoint_sha256": file_sha256(args.search_checkpoint),
                "selection_sha256": file_sha256(args.selection),
                "source_checkpoint_sha256": source_meta["source"]["sha256"],
                "teacher_weights_sha256": teacher_meta["weights_sha256"],
                "teacher_config_sha256": teacher_meta["config_sha256"],
                "teacher_processor_sha256": teacher_meta["processor_sha256"],
                "training_split": split, "search_holdout": holdout_meta, "class_order": classes,
                "seed": args.seed, "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
                "eval_samples": args.eval_samples, "max_steps_per_epoch": args.max_steps_per_epoch,
                "lr": args.lr, "lambda_kd": args.lambda_kd, "lambda_feature": args.lambda_feature,
                "device": args.device}
    start = 0
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previous.get("distillation_settings") != settings or previous.get("format") != "hattention_fixed_student_v1":
            raise ValueError("resume checkpoint settings or provenance differ")
        if previous.get("head_types") != source_meta["head_types"]:
            raise ValueError("resume paths differ from original fixed student")
        start = previous["completed_epochs"]
        if not 0 < start < args.epochs:
            raise ValueError("resume epoch must be less than total epochs")
        model.load_state_dict(previous["state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(previous["cuda_rng"].cpu(), device)
    root.mkdir(parents=True, exist_ok=False)
    before = evaluate(model, eval_data, device, args.eval_batch_size, len(eval_data))
    paths = source_meta["head_types"]
    report = {"status": "running", "settings": settings, "teacher": teacher_meta,
              "head_types": paths, "path_counts": dict(Counter(p for row in paths for p in row)),
              "estimated_online_seconds": selection.get("costs", {}).get("online_seconds"),
              "start_epoch": start, "before": before, "epochs": [],
              "scope": "plaintext distillation; holdout consists of training images and may have been seen by MPCViT; not official validation"}
    write_json(root / "report.json", report)
    print(json.dumps({"event": "distill_started", "start_epoch": start, "before": before,
                      "path_counts": report["path_counts"]}), flush=True)
    for epoch in range(start + 1, args.epochs + 1):
        metrics = train_epoch(model, teacher, train_data, device, optimizer, epoch=epoch,
                              batch_size=args.batch_size, max_steps=args.max_steps_per_epoch, seed=args.seed,
                              lambda_kd=args.lambda_kd, lambda_feature=args.lambda_feature, log_every=args.log_every)
        metrics["evaluation"] = evaluate(model, eval_data, device, args.eval_batch_size, len(eval_data))
        metrics["epoch"] = epoch
        artifact = {"format": "hattention_fixed_student_v1", "state_dict": model.state_dict(),
                    "head_types": paths, "student_init": "checkpoint",
                    "source_checkpoint_sha256": settings["source_checkpoint_sha256"],
                    "distillation_settings": settings, "optimizer": optimizer.state_dict(),
                    "completed_epochs": epoch, "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        checkpoint = root / f"epoch-{epoch:04d}.pt"
        save_checkpoint(checkpoint, artifact)
        metrics["checkpoint"] = str(checkpoint)
        write_json(root / f"epoch-{epoch:04d}.json", metrics)
        report["epochs"].append(metrics)
        write_json(root / "report.json", report, overwrite=True)
        print(json.dumps({"event": "distill_epoch_complete", **metrics}), flush=True)
    report["status"] = "completed_plaintext_not_security_validated"
    write_json(root / "report.json", report, overwrite=True)
    print(json.dumps({"event": "distill_complete", "report": str(root / "report.json"),
                      "last_checkpoint": report["epochs"][-1]["checkpoint"]}), flush=True)


if __name__ == "__main__":
    main()
