"""Open first-layer CrypTen Softmax stages on one public ViT fixture.

Numerical diagnosis only: intermediate values are revealed and it is not a
performance benchmark. The public block input is reshared, so its rounding and
randomness differ from running the preceding encrypted tokenizer.
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
from .benchmark_vit_native_cuda import CrypTenNativeOperators
from .diagnose_vit_cuda import error_metrics
from .diagnose_vit_native_layernorm import first_block_input
from .student_checkpoint import file_sha256


def trace_attention(crypten, q, k, v, q_ref, k_ref, v_ref, emit):
    """Run the same Softmax primitives as native CrypTen, with openings."""
    def inspect(name, value, reference):
        actual = _plain(value.get_plain_text())
        metrics = error_metrics(actual, reference)
        if metrics["finite"]:
            metrics.update(actual_min=float(actual.min()), actual_max=float(actual.max()),
                           reference_min=float(reference.min()), reference_max=float(reference.max()))
        emit({"stage": name, **metrics})
        return actual

    oq = inspect("q", q, q_ref)
    ok = inspect("k", k, k_ref)
    ov = inspect("v", v, v_ref)
    scale = q.size(-1) ** -0.5
    scores_reference = q_ref.matmul(k_ref.transpose(-2, -1)) * scale
    scores = q.matmul(k.transpose(-2, -1)) * scale
    scores_open = inspect("scores", scores, scores_reference)
    maximum = scores.max(dim=-1, keepdim=True)[0]
    maximum_open = inspect("maximum_vs_opened_scores", maximum, scores_open.max(dim=-1, keepdim=True).values)
    logits = scores - maximum
    logits_open = inspect("shifted_logits", logits, scores_open - maximum_open)
    numerator = logits.exp()
    numerator_open = inspect("exp_vs_exact_opened_logits", numerator, logits_open.exp())
    denominator = numerator.sum(dim=-1, keepdim=True)
    denominator_open = inspect("denominator_vs_opened_exp_sum", denominator,
                               numerator_open.sum(dim=-1, keepdim=True))
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inverse = denominator.reciprocal()
    inverse_open = inspect("reciprocal_vs_opened_denominator", inverse, denominator_open.reciprocal())
    weights = numerator * inverse
    weights_open = inspect("weights_vs_exact_softmax_of_opened_scores", weights,
                           torch.softmax(scores_open, dim=-1))
    output = weights.matmul(v)
    output_open = inspect("attention_vs_plain_reference", output,
                          torch.softmax(scores_reference, dim=-1).matmul(v_ref))
    extra = {"qkv_score_reconstruction_max_abs": float(
                 (scores_open - oq.matmul(ok.transpose(-2, -1)) * scale).abs().max()),
             "denominator_nonpositive": int((denominator_open <= 0).sum()),
             "reciprocal_residual_max_abs": float((1 - denominator_open * inverse_open).abs().max()),
             "output_matmul_max_abs_error": float((output_open - weights_open.matmul(ov)).abs().max())}
    if not all(torch.isfinite(torch.tensor(x)) for x in extra.values()):
        extra = {key: value if bool(torch.isfinite(torch.tensor(value))) else None
                 for key, value in extra.items()}
    return extra


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}" if options["device"] == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    crypten.cfg.functions.sqrt_nr_iters = options["sqrt_iters"]
    crypten.init(device=device)
    try:
        model, metadata = load_model(options["checkpoint"], options["mpcvit_root"], None)
        model = model.to(device).eval()
        image = image_tensor(options["image"]).to(device)
        with torch.no_grad():
            block_input = first_block_input(model, image)
            block = model.classifier.blocks[0]
            y_ref = block.pre_norm(block_input)
            b, n, dim = y_ref.shape
            heads = block.self_attn.config.num_heads
            qkv_ref = block.self_attn.qkv(y_ref).reshape(b, n, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
            q_ref, k_ref, v_ref = [item.double() for item in qkv_ref]
            encrypted = crypten.cryptensor(block_input, src=0, precision=options["precision_bits"], device=device)
            ops = CrypTenNativeOperators("chebyshev16", 12.0)
            y = ops.norm(encrypted, block.pre_norm)
            qkv = ops.linear(y, block.self_attn.qkv).reshape(b, n, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if options["head"] is not None:
                head = options["head"]
                if not 0 <= head < heads:
                    raise ValueError(f"head index must be between 0 and {heads - 1}")
                q, k, v = [item[:, head:head + 1] for item in (q, k, v)]
                q_ref, k_ref, v_ref = [item[:, head:head + 1] for item in (q_ref, k_ref, v_ref)]
            rows = []
            def emit(row):
                if rank == 0:
                    rows.append(row)
                    print(json.dumps({"event": "stage_complete", **row}, allow_nan=False), flush=True)
            extra = trace_attention(crypten, q, k, v, q_ref, k_ref, v_ref, emit)
        if rank == 0:
            report = {"status": "diagnostic_completed_not_security_validated",
                      "scope": "public first-layer block input reshared; selected head or all heads; intermediate values opened",
                      "model": metadata, "image": str(Path(options["image"]).resolve()),
                      "image_sha256": file_sha256(options["image"]),
                      "precision": f"Q{options['precision_bits']}", "sqrt_nr_iters": options["sqrt_iters"],
                      "device": options["device"], "head": options["head"], "stages": rows, "checks": extra,
                      "note": "Q/K/V arise from CrypTen LayerNorm and projection of reshared public block input; per-stage errors include earlier propagated error.",
                      "security": "Intermediate values opened; public fixture only. Not a timing benchmark."}
            root = Path(options["output"])
            root.mkdir(parents=True, exist_ok=False)
            with (root / "diagnosis.json").open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
            print(json.dumps({"report": str(root / "diagnosis.json"), "checks": extra}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mpcvit-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision-bits", type=int, choices=(17, 20), default=20)
    parser.add_argument("--sqrt-iters", type=int, default=7)
    parser.add_argument("--head", type=int, help="optional zero-based head index; default all heads")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.sqrt_iters <= 50:
        parser.error("sqrt-iters must be in [1,50]")
    if Path(args.output).exists():
        parser.error("output already exists")
    if args.device == "cuda" and torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-native-attention-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
