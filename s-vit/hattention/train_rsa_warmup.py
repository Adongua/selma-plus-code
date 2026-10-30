"""Train a fixed all-RSA student, or diagnose learning on a small balanced subset.

This is plaintext training. Overfit diagnostics use training images for both
optimization and evaluation and must never be treated as validation accuracy.
"""

import argparse
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from .search import export_fixed_student, search_layers
from .search_data import SearchDataset, training_splits
from .student_checkpoint import create_random_student, make_checkpoint_search_student
from .teacher_guidance import load_teacher
from .train_search import evaluate, loader, save_checkpoint, split_architecture_items, write_json


def balanced_subset(items, classes, per_class, seed):
    """Choose a deterministic, class-balanced subset from the weight split."""
    grouped = {}
    for item in items:
        grouped.setdefault(item[1], []).append(item)
    if classes < 1 or per_class < 1 or classes > len(grouped):
        raise ValueError("invalid overfit subset size")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted(grouped), classes))
    subset = []
    for target in chosen:
        rows = sorted(grouped[target])
        if len(rows) < per_class:
            raise ValueError(f"class {target} has too few training samples")
        subset.extend(rng.sample(rows, per_class))
    return subset


def train_epoch(model, teacher, data, optimizer, device, *, batch_size, seed,
                max_steps=0, lambda_kd=1.0, lambda_feature=1.0, log_every=0, epoch=1):
    model.train()
    totals, steps, samples = {}, 0, 0
    for inputs, images, labels in loader(data, batch_size, shuffle=True, seed=seed):
        if max_steps and steps >= max_steps:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if teacher is None:
            logits, _ = model(inputs)
            ce = F.cross_entropy(logits, labels)
            loss, components = ce, {"ce": ce}
        else:
            loss, components = teacher.loss(model, images, inputs, labels,
                                            lambda_kd=lambda_kd, lambda_feature=lambda_feature)
        if not torch.isfinite(loss):
            raise ValueError("non-finite warmup loss")
        loss.backward()
        optimizer.step()
        count = labels.numel()
        samples += count
        steps += 1
        for key, value in {"total": loss, **components}.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
        if log_every and steps % log_every == 0:
            print(json.dumps({"event": "warmup_step", "epoch": epoch, "step": steps,
                              "loss": {key: float(value.detach()) for key, value in
                                       {"total": loss, **components}.items()}}), flush=True)
    if not steps:
        raise ValueError("warmup epoch had no batches")
    return {"steps": steps, "samples": samples,
            "loss": {key: value / samples for key, value in totals.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mpcvit-root", "dataset-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="resume a completed warmup epoch in output-root")
    parser.add_argument("--loss", choices=("ce", "distill"), default="ce")
    parser.add_argument("--mapping", choices=("strict-id", "first-name"), default="first-name")
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-feature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--overfit-classes", type=int, default=0,
                        help="positive enables a training-set overfit diagnostic")
    parser.add_argument("--overfit-per-class", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1
            or args.eval_samples < 1 or args.max_steps_per_epoch < 0 or args.log_every < 0
            or args.overfit_classes < 0 or args.overfit_per_class < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or any(not math.isfinite(x) or x < 0 for x in (args.lambda_kd, args.lambda_feature))):
        parser.error("invalid training arguments")
    if args.loss == "distill" and args.teacher_dir is None:
        parser.error("--loss distill requires --teacher-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    root = args.output_root.resolve()
    if args.resume:
        if not root.is_dir() or args.resume.resolve().parent != root:
            parser.error("resume checkpoint must belong to an existing output-root")
    elif root.exists():
        parser.error("output-root already exists; choose a new directory")
    torch.manual_seed(args.seed)
    classes, weight_items, architecture_items, split = training_splits(args.dataset_root, seed=args.seed)
    _, heldout, holdout = split_architecture_items(architecture_items, seed=args.seed)
    diagnostic = args.overfit_classes > 0
    if diagnostic:
        weight_items = balanced_subset(weight_items, args.overfit_classes, args.overfit_per_class, args.seed)
        evaluation_items = weight_items
    else:
        evaluation_items = heldout[:args.eval_samples]
        if len(evaluation_items) < args.eval_samples:
            parser.error("eval-samples exceeds the held-out search split")
    device = torch.device(args.device)
    source, source_metadata = create_random_student(args.mpcvit_root)
    search = make_checkpoint_search_student(source, prior_strength=0.0)
    paths = [["rsa"] * layer.config.num_heads for layer in search_layers(search)]
    model = export_fixed_student(search, paths)
    teacher = teacher_metadata = None
    if args.loss == "distill":
        teacher, teacher_metadata = load_teacher(args.teacher_dir, args.dataset_root, mapping=args.mapping)
        model.search_feature_projection = nn.Linear(192, teacher_metadata["teacher_hidden_size"])
        teacher = teacher.to(device).eval()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data = SearchDataset(args.dataset_root, weight_items)
    evaluation = SearchDataset(args.dataset_root, evaluation_items)
    config = {"loss": args.loss, "seed": args.seed, "batch_size": args.batch_size,
              "epochs": args.epochs, "learning_rate": args.lr,
              "max_steps_per_epoch": args.max_steps_per_epoch,
              "lambda_kd": args.lambda_kd, "lambda_feature": args.lambda_feature,
              "overfit_diagnostic": diagnostic, "training_samples": len(weight_items),
              "overfit_classes": args.overfit_classes, "overfit_per_class": args.overfit_per_class,
              "eval_samples": args.eval_samples, "eval_batch_size": args.eval_batch_size,
              "device": args.device, "mapping": args.mapping,
              "teacher_weights_sha256": teacher_metadata["weights_sha256"] if teacher_metadata else None,
              "teacher_config_sha256": teacher_metadata["config_sha256"] if teacher_metadata else None,
              "teacher_processor_sha256": teacher_metadata["processor_sha256"] if teacher_metadata else None,
              "class_order": classes, "training_split": split, "search_holdout": holdout}
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        saved_config = checkpoint.get("config")
        # The requested final epoch is a continuation target, not a training
        # hyperparameter. Permit extending a run while keeping its data,
        # optimizer, and objective settings identical.
        saved_compatible_config = dict(saved_config or {})
        current_compatible_config = dict(config)
        saved_compatible_config.pop("epochs", None)
        current_compatible_config.pop("epochs", None)
        if (checkpoint.get("format") != "hattention_rsa_warmup_v1"
                or saved_compatible_config != current_compatible_config):
            raise ValueError("warmup resume configuration or model/data hashes differ")
        start = checkpoint["completed_epochs"]
        if not 0 < start < args.epochs or args.resume.name != f"epoch-{start:04d}.pt":
            raise ValueError("resume target must be later than the checkpoint epoch; check its filename")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
        write_json(root / "run.json", {"config": config, "source": source_metadata,
                                       "teacher": teacher_metadata,
                                       "resumed_from": str(args.resume.resolve())},
                   overwrite=True)
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "run.json", {"config": config, "source": source_metadata,
                                       "teacher": teacher_metadata})
    for epoch in range(start, args.epochs):
        metrics = train_epoch(model, teacher, data, optimizer, device, batch_size=args.batch_size,
                              seed=args.seed + epoch, max_steps=args.max_steps_per_epoch,
                              lambda_kd=args.lambda_kd, lambda_feature=args.lambda_feature,
                              log_every=args.log_every, epoch=epoch + 1)
        metrics["evaluation"] = evaluate(model, evaluation, device, args.eval_batch_size, len(evaluation))
        metrics["evaluation"]["scope"] = ("same training images; overfit diagnostic only" if diagnostic
                                          else "held-out training images, not official validation")
        metrics["epoch"] = epoch + 1
        checkpoint = root / f"epoch-{epoch + 1:04d}.pt"
        save_checkpoint(checkpoint, {"format": "hattention_rsa_warmup_v1", "config": config,
                                     "completed_epochs": epoch + 1, "model": model.state_dict(),
                                     "optimizer": optimizer.state_dict(), "metrics": metrics,
                                     "cpu_rng": torch.get_rng_state(),
                                     "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None})
        write_json(root / f"epoch-{epoch + 1:04d}.json", metrics)
        print(json.dumps({"event": "warmup_epoch_complete", "checkpoint": str(checkpoint), **metrics}), flush=True)


if __name__ == "__main__":
    main()
