"""Open selected-layer CrypTen Softmax stages on one public ViT fixture.

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
from .benchmark_vit_native_cuda import AuditedPlainOperators, CrypTenNativeOperators
from .diagnose_vit_cuda import error_metrics
from .diagnose_vit_native_layernorm import first_block_input
from .student_checkpoint import file_sha256
from .vit_executor import forward_vit


def trace_attention(crypten, q, k, v, q_ref, k_ref, v_ref, emit, *, trace_exp=False,
                    reciprocal_scale=None):
    """Run the same Softmax primitives as native CrypTen, with openings."""
    def inspect(name, value, reference):
        actual = _plain(value.get_plain_text())
        metrics = error_metrics(actual, reference)
        if metrics["finite"]:
            metrics.update(actual_min=float(actual.min()), actual_max=float(actual.max()),
                           reference_min=float(reference.min()), reference_max=float(reference.max()),
                           elements=int(actual.numel()),
                           errors_above_0_01=int((actual.sub(reference).abs() > 0.01).sum()))
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
    if trace_exp:
        # Same sequence as CrypTen exp: 1 + x/2**d, then d squares. Each
        # reconstruction is diagnostic and does not alter the secret value.
        steps = crypten.cfg.functions.exp_iterations
        numerator = 1 + logits.div(2 ** steps)
        previous = inspect("exp_initial", numerator, 1 + logits_open / (2 ** steps))
        for step in range(steps):
            numerator = numerator.square()
            previous = inspect(f"exp_square_{step + 1}", numerator, previous.square())
    else:
        numerator = logits.exp()
    numerator_open = inspect("exp_vs_exact_opened_logits", numerator, logits_open.exp())
    denominator = numerator.sum(dim=-1, keepdim=True)
    denominator_open = inspect("denominator_vs_opened_exp_sum", denominator,
                               numerator_open.sum(dim=-1, keepdim=True))
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inverse = denominator.reciprocal()
    inverse_open = inspect("reciprocal_vs_opened_denominator", inverse, denominator_open.reciprocal())
    native_residual = (1 - denominator_open * inverse_open).abs().max()
    if reciprocal_scale is not None:
        native_weights = numerator * inverse
        inspect("native_weights_vs_exact_softmax", native_weights, torch.softmax(scores_open, dim=-1))
        inspect("native_attention_vs_plain_reference", native_weights.matmul(v),
                torch.softmax(scores_reference, dim=-1).matmul(v_ref))
        scaled_denominator = denominator.div(reciprocal_scale)
        inspect("scaled_denominator", scaled_denominator, denominator_open / reciprocal_scale)
        with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
            inverse = scaled_denominator.reciprocal().div(reciprocal_scale)
        inverse_open = inspect("scaled_reciprocal_vs_opened_denominator", inverse,
                               denominator_open.reciprocal())
    weights = numerator * inverse
    weights_open = inspect("weights_vs_exact_softmax_of_opened_scores", weights,
                           torch.softmax(scores_open, dim=-1))
    output = weights.matmul(v)
    output_open = inspect("attention_vs_plain_reference", output,
                          torch.softmax(scores_reference, dim=-1).matmul(v_ref))
    extra = {"qkv_score_reconstruction_max_abs": float(
                 (scores_open - oq.matmul(ok.transpose(-2, -1)) * scale).abs().max()),
             "denominator_nonpositive": int((denominator_open <= 0).sum()),
             "native_reciprocal_residual_max_abs": float(native_residual),
             "reciprocal_scale": reciprocal_scale,
             "reciprocal_residual_max_abs": float((1 - denominator_open * inverse_open).abs().max()),
             "output_matmul_max_abs_error": float((output_open - weights_open.matmul(ov)).abs().max())}
    if not all(value is None or torch.isfinite(torch.tensor(value)) for value in extra.values()):
        extra = {key: value if value is None or bool(torch.isfinite(torch.tensor(value))) else None
                 for key, value in extra.items()}
    return extra


def plaintext_block_input(model, image, layer_index, chebyshev_bound=12.0):
    """Extract a public reference block input from the plaintext Chebyshev model."""
    blocks = model.classifier.blocks
    if not 0 <= layer_index < len(blocks):
        raise ValueError(f"layer must be in [1, {len(blocks)}]")
    if layer_index == 0:
        return first_block_input(model, image)
    captured = {}
    ops = AuditedPlainOperators(chebyshev_bound)
    def remember(index, site, value):
        if index == layer_index and site == "block_input":
            captured["input"] = value.detach().clone()
    forward_vit(model, image, ops, on_stage=remember)
    if "input" not in captured:
        raise RuntimeError("selected block input was not captured")
    if any(maximum > chebyshev_bound for maximum in ops.gelu_max_abs[:layer_index]):
        raise ValueError("a preceding public Chebyshev input exceeds the fit interval")
    return captured["input"]


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
            layer_index = options["layer"] - 1
            block_input = plaintext_block_input(model, image, layer_index, options["chebyshev_bound"])
            block = model.classifier.blocks[layer_index]
            y_ref = block.pre_norm(block_input)
            b, n, dim = y_ref.shape
            heads = block.self_attn.config.num_heads
            qkv_ref = block.self_attn.qkv(y_ref).reshape(b, n, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
            q_ref, k_ref, v_ref = [item.double() for item in qkv_ref]
            if options["head"] is not None:
                head = options["head"]
                if not 0 <= head < heads:
                    raise ValueError(f"head index must be between 0 and {heads - 1}")
                q_ref, k_ref, v_ref = [item[:, head:head + 1] for item in (q_ref, k_ref, v_ref)]
            records = []
            for iteration in range(options["repeats"]):
                encrypted = crypten.cryptensor(block_input, src=0, precision=options["precision_bits"], device=device)
                ops = CrypTenNativeOperators("chebyshev16", options["chebyshev_bound"])
                y = ops.norm(encrypted, block.pre_norm)
                qkv = ops.linear(y, block.self_attn.qkv).reshape(b, n, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                if options["head"] is not None:
                    q, k, v = [item[:, head:head + 1] for item in (q, k, v)]
                rows = []
                def emit(row):
                    if rank == 0:
                        rows.append(row)
                        print(json.dumps({"event": "stage_complete", "iteration": iteration + 1, **row}, allow_nan=False), flush=True)
                extra = trace_attention(crypten, q, k, v, q_ref, k_ref, v_ref, emit,
                                        trace_exp=options["trace_exp"],
                                        reciprocal_scale=options["reciprocal_scale"])
                if rank == 0:
                    records.append({"iteration": iteration + 1, "stages": rows, "checks": extra})
        if rank == 0:
            report = {"status": "diagnostic_completed_not_security_validated",
                      "scope": "public selected-layer plaintext block input reshared; selected head or all heads; intermediate values opened",
                      "model": metadata, "image": str(Path(options["image"]).resolve()),
                      "image_sha256": file_sha256(options["image"]),
                      "precision": f"Q{options['precision_bits']}", "sqrt_nr_iters": options["sqrt_iters"],
                      "device": options["device"], "head": options["head"], "layer": options["layer"],
                      "chebyshev_bound": options["chebyshev_bound"], "repeats": options["repeats"],
                      "reciprocal_scale": options["reciprocal_scale"],
                      "trace_exp": options["trace_exp"],
                      "exp_iterations": crypten.cfg.functions.exp_iterations,
                      "stages": records[0]["stages"], "checks": records[0]["checks"], "records": records,
                      "note": "Q/K/V arise from CrypTen LayerNorm and projection of the selected plaintext block input reshared; it is not the actual secure-forward state. Extra openings affect random execution.",
                      "security": "Intermediate values opened; public fixture only. Not a timing benchmark."}
            root = Path(options["output"])
            root.mkdir(parents=True, exist_ok=False)
            with (root / "diagnosis.json").open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
            print(json.dumps({"report": str(root / "diagnosis.json"), "repeats": len(records)}), flush=True)
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
    parser.add_argument("--layer", type=int, default=1, help="one-based ViT block index")
    parser.add_argument("--repeats", type=int, default=1, help="fresh resharing and diagnostics per iteration")
    parser.add_argument("--chebyshev-bound", type=float, default=12.0)
    parser.add_argument("--head", type=int, help="optional zero-based head index; default all heads")
    parser.add_argument("--trace-exp", action="store_true", help="open each stage of CrypTen exp")
    parser.add_argument("--reciprocal-scale", type=int,
                        help="also diagnose scaled reciprocal 1/(d/S)/S using public power-of-two S")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.sqrt_iters <= 50:
        parser.error("sqrt-iters must be in [1,50]")
    if args.layer < 1 or not 1 <= args.repeats <= 20 or not 0 < args.chebyshev_bound < float("inf"):
        parser.error("layer must be positive, repeats in [1,20], Chebyshev bound finite and positive")
    if args.reciprocal_scale is not None and (args.reciprocal_scale <= 0 or
            args.reciprocal_scale > 2**20 or args.reciprocal_scale & (args.reciprocal_scale - 1)):
        parser.error("reciprocal-scale must be a power of two in [1, 2**20]")
    if Path(args.output).exists():
        parser.error("output already exists")
    if args.device == "cuda" and torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-native-attention-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
