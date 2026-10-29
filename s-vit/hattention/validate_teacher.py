"""Validate a locally downloaded Hugging Face ViT on Tiny-ImageNet.

Only official Tiny-ImageNet val_annotations.txt format and explicit WordNet IDs
in the model's id2label are accepted. Weights are loaded with weights_only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


WNID = re.compile(r"n[0-9]{8}\Z")


def class_mapping(wnids, id2label):
    """Return teacher class indices in sorted student class order."""
    wnids = sorted(wnids)
    if len(wnids) != 200 or len(set(wnids)) != 200 or any(not WNID.fullmatch(w) for w in wnids):
        raise ValueError("wnids.txt must contain 200 distinct WordNet IDs")
    labels = {int(k): str(v).strip() for k, v in id2label.items()}
    if set(labels) != set(range(200)):
        raise ValueError("teacher must provide exactly 200 indexed labels")
    if len(set(labels.values())) != 200 or set(labels.values()) != set(wnids):
        raise ValueError("teacher id2label must identify the same 200 WordNet IDs; "
                         "do not guess a mapping from class names")
    inverse = {label: index for index, label in labels.items()}
    return wnids, [inverse[wnid] for wnid in wnids]


def validation_items(root, wnids):
    images = root / "val" / "images"
    annotations = root / "val" / "val_annotations.txt"
    indices = {wnid: i for i, wnid in enumerate(wnids)}
    result, seen = [], set()
    for line_number, line in enumerate(annotations.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) < 2 or parts[1] not in indices:
            raise ValueError(f"invalid validation annotation at line {line_number}")
        name, wnid = parts[:2]
        if Path(name).name != name or name in seen:
            raise ValueError(f"invalid or duplicate image name at line {line_number}")
        path = images / name
        if not path.is_file():
            raise ValueError(f"validation image missing: {path}")
        seen.add(name)
        result.append((path, indices[wnid]))
    if not result:
        raise ValueError("validation annotations are empty")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 evaluates the full validation set; a positive value is a smoke test")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_samples < 0:
        parser.error("batch size must be positive and max-samples nonnegative")
    if args.output.exists():
        parser.error("output already exists; choose a new output file")

    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    teacher_dir = args.teacher_dir.resolve()
    config = AutoConfig.from_pretrained(teacher_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != "vit":
        raise ValueError(f"expected a ViT model, got {config.model_type!r}")
    wnids = (args.dataset_root / "wnids.txt").read_text(encoding="utf-8").splitlines()
    classes, teacher_indices = class_mapping(wnids, config.id2label)
    items = validation_items(args.dataset_root, classes)
    if args.max_samples:
        items = items[:args.max_samples]

    # Explicitly use PyTorch's restricted loader: never execute remote code.
    weight_file = teacher_dir / "pytorch_model.bin"
    if not weight_file.is_file():
        raise FileNotFoundError(f"expected {weight_file}")
    digest = hashlib.sha256()
    with weight_file.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    state = torch.load(weight_file, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("expected a plain Hugging Face tensor state_dict")
    model = AutoModelForImageClassification.from_config(config, trust_remote_code=False)
    model.load_state_dict(state, strict=True)
    processor = AutoImageProcessor.from_pretrained(teacher_dir, local_files_only=True,
                                                   trust_remote_code=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(args.device).eval()
    permutation = torch.tensor(teacher_indices, dtype=torch.long, device=args.device)
    correct1 = correct5 = 0
    support = Counter()
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            chunk = items[start:start + args.batch_size]
            images = []
            for path, _ in chunk:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = {key: value.to(args.device) for key, value in
                      processor(images=images, return_tensors="pt").items()}
            logits = model(**inputs).logits
            if logits.ndim != 2 or logits.shape != (len(chunk), 200) or not torch.isfinite(logits).all():
                raise ValueError("teacher produced invalid logits")
            logits = logits.index_select(1, permutation)
            targets = torch.tensor([target for _, target in chunk], device=args.device)
            predictions = logits.topk(5, dim=1).indices
            correct1 += int((predictions[:, 0] == targets).sum().item())
            correct5 += int((predictions == targets[:, None]).any(dim=1).sum().item())
            support.update(target for _, target in chunk)
            print(f"evaluated {min(start + len(chunk), len(items))}/{len(items)}", flush=True)
    report = {"status": "evaluated", "scope": "smoke_subset" if args.max_samples else "full_annotations",
              "teacher_dir": str(teacher_dir), "dataset_root": str(args.dataset_root.resolve()),
              "model_type": config.model_type, "architecture": config.architectures,
              "hidden_size": config.hidden_size, "layers": config.num_hidden_layers,
              "heads": config.num_attention_heads, "patch_size": config.patch_size,
              "image_size": config.image_size, "num_classes": len(classes),
              "class_order": "sorted Tiny-ImageNet WordNet IDs", "teacher_indices": teacher_indices,
              "samples": len(items), "classes_observed": len(support),
              "top1": correct1 / len(items), "top5": correct5 / len(items),
              "device": args.device, "weights": str(weight_file), "weights_sha256": digest.hexdigest(),
              "security": "public pretrained candidate; restricted weights-only deserialization"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
