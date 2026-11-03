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


def _raw_tensor(opened):
    return opened.tensor() if hasattr(opened, "tensor") else opened


def open_exact(value):
    """Open once; decode raw shares exactly in float64 and inspect bundled decoder."""
    raw_opened = value.reveal()
    raw = _raw_tensor(raw_opened)
    scale = value._tensor.encoder.scale
    exact = raw.to(torch.float64) / scale
    legacy = _plain(value._tensor.encoder.decode(raw_opened.clone()))
    return raw, exact, legacy


def diagnostic_raw_square(value):
    """Same Beaver square and truncation, with public raw openings around them."""
    import crypten
    from crypten.mpc.primitives import beaver

    if crypten.cfg.mpc.protocol != "beaver":
        raise ValueError("raw square tracing currently requires the beaver protocol")
    input_raw = _raw_tensor(value.reveal())
    scale = value._tensor.encoder.scale
    product = beaver.square(value._tensor)
    product_raw = _raw_tensor(product.reveal())
    expected_product = input_raw * input_raw  # int64 ring arithmetic
    product_mismatches = int((product_raw != expected_product).sum())
    expected_truncated = torch.div(product_raw, scale, rounding_mode="trunc")
    product.div_(scale)
    truncated_raw = _raw_tensor(product.reveal())
    delta = truncated_raw - expected_truncated
    max_index = int(delta.abs().argmax())
    coordinates, remaining = [], max_index
    for size in reversed(delta.shape):
        coordinates.append(remaining % size)
        remaining //= size
    coordinates.reverse()
    qbits = scale.bit_length() - 1
    one_wrap = 1 << (64 - qbits)
    probe = {"raw_product_mismatches": product_mismatches,
             "truncation_delta_max_abs": int(delta.abs().max()),
             "truncation_deltas_over_4": int((delta.abs() > 4).sum()),
             "expected_one_wrap_raw": one_wrap,
             "deltas_near_one_wrap": int(((delta.abs() - one_wrap).abs() <= 4).sum()),
             "max_delta_index": coordinates,
             "truncation_delta_at_max": int(delta.reshape(-1)[max_index]),
             "raw_product_at_max": int(product_raw.reshape(-1)[max_index]),
             "truncated_at_max": int(truncated_raw.reshape(-1)[max_index]),
             "expected_truncated_at_max": int(expected_truncated.reshape(-1)[max_index])}
    result = value.clone()
    result._tensor.share = product.share
    return result, probe


def diagnostic_qk_stages(q, k, factor):
    """Open raw QK Beaver matmul and public scaling, for public inputs only."""
    import crypten
    from crypten.mpc.primitives import beaver

    if crypten.cfg.mpc.protocol != "beaver":
        raise ValueError("raw QK tracing requires the beaver protocol")
    key = k.transpose(-2, -1)
    scale = q._tensor.encoder.scale
    if scale != key._tensor.encoder.scale or scale <= 1:
        raise ValueError("raw QK tracing requires matching fixed-point scales")
    qraw, kraw = _raw_tensor(q.reveal()), _raw_tensor(key.reveal())
    product = beaver.matmul(q._tensor, key._tensor)
    raw_product = _raw_tensor(product.reveal())
    # CPU integer matmul computes modulo 2^64, including overflow.
    expected = torch.matmul(qraw.cpu(), kraw.cpu()).to(raw_product.device)

    def truncation_probe(before, after, divisor):
        expected_trunc = torch.div(before, divisor, rounding_mode="trunc")
        delta = after - expected_trunc
        flat = int(delta.abs().argmax())
        coords, remaining = [], flat
        for size in reversed(delta.shape):
            coords.append(remaining % size)
            remaining //= size
        one_wrap = (1 << 64) // divisor
        return {"divisor": divisor, "max_delta_index": list(reversed(coords)),
                "truncation_delta_max_abs": int(delta.abs().max()),
                "truncation_deltas_over_4": int((delta.abs() > 4).sum()),
                "expected_one_wrap_raw": one_wrap,
                "deltas_near_one_wrap": int(((delta.abs() - one_wrap).abs() <= 4).sum()),
                "truncation_delta_at_max": int(delta.reshape(-1)[flat]),
                "raw_product_at_max": int(before.reshape(-1)[flat]),
                "truncated_at_max": int(after.reshape(-1)[flat]),
                "expected_truncated_at_max": int(expected_trunc.reshape(-1)[flat])}

    product.div_(scale)
    matmul_raw = _raw_tensor(product.reveal())
    matmul_probe = truncation_probe(raw_product, matmul_raw, scale)
    matmul_probe["raw_product_mismatches"] = int((raw_product != expected).sum())
    matmul = q.clone()
    matmul._tensor.share = product.share

    encoded = matmul._tensor.encoder.encode(factor, device=matmul._tensor.device)
    scaled = matmul._tensor.clone()
    scaled.share = scaled.share * encoded
    scaled_product_raw = _raw_tensor(scaled.reveal())
    scaled.div_(scale)
    scaled_raw = _raw_tensor(scaled.reveal())
    scaling_probe = truncation_probe(scaled_product_raw, scaled_raw, scale)
    scaling_probe["raw_product_mismatches"] = int(
        (scaled_product_raw != matmul_raw * int(factor * scale)).sum())
    scores = q.clone()
    scores._tensor.share = scaled.share
    return matmul, scores, matmul_probe, scaling_probe


def trace_attention(crypten, q, k, v, q_ref, k_ref, v_ref, emit, *, trace_exp=False,
                    reciprocal_scale=None, trace_raw_square=False, trace_raw_qk=False):
    """Run the same Softmax primitives as native CrypTen, with openings."""
    def inspect(name, value, reference, raw_square_probe=None, probe_key="raw_square_probe"):
        raw, actual, legacy = open_exact(value)
        metrics = error_metrics(actual, reference)
        if metrics["finite"]:
            delta = (actual - reference).reshape(-1)
            flat_index = int(delta.abs().argmax())
            coords, remaining = [], flat_index
            for size in reversed(actual.shape):
                coords.append(remaining % size)
                remaining //= size
            coords.reverse()
            metrics.update(actual_min=float(actual.min()), actual_max=float(actual.max()),
                           reference_min=float(reference.min()), reference_max=float(reference.max()),
                           elements=int(actual.numel()),
                           errors_above_0_01=int((delta.abs() > 0.01).sum()),
                           errors_above_0_0001=int((delta.abs() > 0.0001).sum()),
                           max_error_index=coords,
                           actual_at_max_error=float(actual.reshape(-1)[flat_index]),
                           reference_at_max_error=float(reference.reshape(-1)[flat_index]),
                           signed_error_at_max=float(delta[flat_index]),
                           legacy_decode_max_abs_error=float((legacy - actual).abs().max()),
                           legacy_decode_errors_above_0_1=int(((legacy - actual).abs() > 0.1).sum()))
        if raw_square_probe is not None:
            metrics[probe_key] = raw_square_probe
        emit({"stage": name, **metrics})
        return actual

    oq = inspect("q", q, q_ref)
    ok = inspect("k", k, k_ref)
    ov = inspect("v", v, v_ref)
    scale = q.size(-1) ** -0.5
    scores_reference = q_ref.matmul(k_ref.transpose(-2, -1)) * scale
    if trace_raw_qk:
        matmul, scores, matmul_probe, scaling_probe = diagnostic_qk_stages(q, k, scale)
        inspect("qk_matmul_raw_trace", matmul, oq.matmul(ok.transpose(-2, -1)), matmul_probe, "raw_qk_probe")
        inspect("qk_scale_raw_trace", scores, oq.matmul(ok.transpose(-2, -1)) * scale, scaling_probe, "raw_qk_probe")
    else:
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
        divided = logits.div(2 ** steps)
        divided_open = inspect("exp_division", divided, logits_open / (2 ** steps))
        numerator = 1 + divided
        previous = inspect("exp_initial", numerator, 1 + logits_open / (2 ** steps))
        inspect("exp_public_addition", numerator, 1 + divided_open)
        for step in range(steps):
            if trace_raw_square:
                numerator, probe = diagnostic_raw_square(numerator)
            else:
                numerator, probe = numerator.square(), None
            previous = inspect(f"exp_square_{step + 1}", numerator, previous.square(), probe)
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
                                        reciprocal_scale=options["reciprocal_scale"],
                                        trace_raw_square=options["trace_raw_square"],
                                        trace_raw_qk=options["trace_raw_qk"])
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
                      "trace_raw_square": options["trace_raw_square"],
                      "trace_raw_qk": options["trace_raw_qk"],
                      "opening_decode": "raw int64 divided by encoder scale in float64; bundled decoder also checked",
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
    parser.add_argument("--trace-raw-square", action="store_true", help="open raw Beaver product and truncation stages; implies --trace-exp")
    parser.add_argument("--trace-raw-qk", action="store_true", help="open raw QK matmul and public scale stages for public fixture diagnosis")
    parser.add_argument("--reciprocal-scale", type=int,
                        help="also diagnose scaled reciprocal 1/(d/S)/S using public power-of-two S")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.trace_raw_square:
        args.trace_exp = True
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
