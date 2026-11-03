"""Open public first-block LayerNorm intermediates for numerical diagnosis.

This is not a private protocol or a performance benchmark. All parties open
the intermediate tensors and compare them against a floating-point reference.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.multiprocessing as mp

from slayernorm.benchmark_crypten_cuda import _plain
from .benchmark_vit_cuda import image_tensor, load_model
from .diagnose_vit_cuda import error_metrics
from .student_checkpoint import file_sha256
from .vit_executor import PlainOperators


def first_block_input(model, image):
    classifier = model.classifier
    if classifier.seq_pool or len(model.tokenizer.conv_layers) != 1:
        raise ValueError("only single-convolution CLS-token ViTLite is supported")
    tokenizer = model.tokenizer.conv_layers[0]
    if len(tokenizer) != 3 or not all(isinstance(m, torch.nn.Identity) for m in tokenizer[1:]):
        raise ValueError("tokenizer activation and pooling must be identities")
    ops = PlainOperators()
    value = ops.conv(image, tokenizer[0]).flatten(2, 3).transpose(-2, -1)
    value = ops.cat_cls(value, classifier.class_emb)
    return value + classifier.positional_emb.detach() if classifier.positional_emb is not None else value


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}" if options["device"] == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    if options["sqrt_iters"] is not None:
        crypten.cfg.functions.sqrt_nr_iters = options["sqrt_iters"]
    crypten.init(device=device)
    try:
        model, metadata = load_model(options["checkpoint"], options["mpcvit_root"], None)
        model = model.to(device).eval()
        image = image_tensor(options["image"]).to(device)
        with torch.no_grad():
            x = first_block_input(model, image)
            layer = model.classifier.blocks[0].pre_norm
            x64 = x.double()
            mean = x64.mean(dim=-1, keepdim=True)
            centered = x64 - mean
            variance = centered.square().mean(dim=-1, keepdim=True)
            variance_eps = variance + layer.eps
            root = variance_eps.sqrt()
            inverse = root.reciprocal()
            normalized = centered * inverse
            result = normalized * layer.weight.double() + layer.bias.double()
        references = {"input": x64, "mean": mean, "centered": centered,
                      "variance": variance, "variance_plus_epsilon": variance_eps,
                      "sqrt": root, "reciprocal_sqrt": inverse,
                      "normalized": normalized, "affine_output": result}
        rows = []
        def inspect(name, value):
            actual = _plain(value.get_plain_text())
            if rank == 0:
                metrics = error_metrics(actual, references[name])
                if metrics["finite"]:
                    metrics.update(reference_min=float(references[name].min()),
                                   reference_max=float(references[name].max()),
                                   actual_min=float(actual.min()), actual_max=float(actual.max()))
                row = {"stage": name, **metrics}
                rows.append(row)
                print(json.dumps({"event": "stage_complete", **row}, allow_nan=False), flush=True)
        with torch.no_grad():
            secret = crypten.cryptensor(x, src=0, precision=options["precision_bits"], device=device)
            inspect("input", secret)
            secret_mean = secret.mean(dim=-1, keepdim=True)
            inspect("mean", secret_mean)
            secret_centered = secret - secret_mean
            inspect("centered", secret_centered)
            secret_variance = secret_centered.square().mean(dim=-1, keepdim=True)
            inspect("variance", secret_variance)
            secret_variance_eps = secret_variance.add(layer.eps)
            inspect("variance_plus_epsilon", secret_variance_eps)
            secret_root = secret_variance_eps.sqrt()
            inspect("sqrt", secret_root)
            secret_inverse = secret_root.reciprocal()
            inspect("reciprocal_sqrt", secret_inverse)
            secret_normalized = secret_centered.mul(secret_inverse)
            inspect("normalized", secret_normalized)
            secret_output = secret_normalized.mul(layer.weight.detach()).add(layer.bias.detach())
            inspect("affine_output", secret_output)
        if rank == 0:
            report = {"status": "diagnostic_completed_not_security_validated",
                      "scope": "first block pre_norm; public image; intermediate tensors opened",
                      "model": metadata, "image": str(Path(options["image"]).resolve()),
                      "image_sha256": file_sha256(options["image"]),
                      "precision": f"Q{options['precision_bits']}", "device": options["device"],
                      "epsilon": layer.eps, "sqrt_nr_iters": crypten.cfg.functions.sqrt_nr_iters,
                      "reciprocal_nr_iters": crypten.cfg.functions.reciprocal_nr_iters, "stages": rows,
                      "note": "Each stage is compared with the exact plaintext computation on the same public input; earlier fixed-point error propagates.",
                      "security": "Intermediate values are opened. Use public test inputs only; not a benchmark."}
            root = Path(options["output"])
            root.mkdir(parents=True, exist_ok=False)
            with (root / "diagnosis.json").open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
            print(json.dumps({"report": str(root / "diagnosis.json"), "stages": len(rows)}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mpcvit-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--precision-bits", type=int, choices=(17, 20), default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sqrt-iters", type=int, help="diagnostic override for CrypTen inverse-square-root iterations")
    args = parser.parse_args()
    if args.sqrt_iters is not None and not 1 <= args.sqrt_iters <= 50:
        parser.error("sqrt-iters must be in [1,50]")
    if Path(args.output).exists():
        parser.error("output already exists")
    if args.device == "cuda" and torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-native-layernorm-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
