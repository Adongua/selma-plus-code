"""Diagnose CrypTen attention on PUBLIC synthetic inputs; not a benchmark.

This intentionally opens intermediate values. Never use it with private data.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.multiprocessing as mp

from .benchmark_cuda import _error_metrics, _parse_shape


def _as_tensor(value):
    return (value.tensor() if hasattr(value, "tensor") else value).double()


def _open(value):
    return _as_tensor(value.get_plain_text())


def _range(value):
    return {"min": value.min().item(), "max": value.max().item()}


def _worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten

    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    crypten.init(device=device)
    try:
        shape = tuple(options["shape"])
        rng = torch.Generator().manual_seed(options["seed"])
        values = []
        for _ in range(3):
            value = (options["input_std"] * torch.randn(shape, generator=rng,
                     dtype=torch.float64)).clamp(-options["input_bound"], options["input_bound"])
            # Preserve the Q17 fixture of benchmark_compare_cuda even if the
            # CrypTen computational precision is varied for diagnosis.
            values.append((torch.round(value * (1 << 17)) / (1 << 17)).to(device))
        q, k, v = values
        scale = shape[-1] ** -0.5
        reference_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        reference_output = torch.matmul(torch.softmax(reference_scores, dim=-1), v)
        records = []
        for iteration in range(options["repeats"]):
            if rank == 0:
                print(json.dumps({"event": "diagnostic_iteration_started", "iteration": iteration}), flush=True)
            encrypted = [crypten.cryptensor(x, src=0, precision=options["precision_bits"],
                                           device=device) for x in values]
            cq, ck, cv = encrypted
            scores = cq.matmul(ck.transpose(-2, -1)) * scale
            score_open = _open(scores)
            maximum = scores.max(dim=-1, keepdim=True)[0]
            maximum_open = _open(maximum)
            logits = scores - maximum
            logits_open = _open(logits)
            numerator = logits.exp()
            numerator_open = _open(numerator)
            denominator = numerator.sum(dim=-1, keepdim=True)
            denominator_open = _open(denominator)
            if rank == 0:
                print(json.dumps({"event": "denominator_opened", "iteration": iteration,
                                  "range": _range(denominator_open)}), flush=True)
            with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
                inverse = denominator.reciprocal()
            inverse_open = _open(inverse)
            weights = numerator * inverse
            weights_open = _open(weights)
            output_open = _open(weights.matmul(cv))

            # Repeat the native default Newton update, opening intermediate
            # estimates only for this synthetic diagnostic. These extra
            # operations deliberately do not measure benchmark timing.
            initial = crypten.cfg.functions.reciprocal_initial
            if crypten.cfg.functions.reciprocal_method != "NR" or initial is not None:
                raise ValueError("diagnostic expects CrypTen default NR and reciprocal_initial=None")
            estimate = 3 * (1 - 2 * denominator).exp() + 0.003
            iterations = []
            inverse_target = denominator_open.reciprocal()
            for step in range(crypten.cfg.functions.reciprocal_nr_iters):
                estimate_open = _open(estimate)
                square = estimate.square()
                square_open = _open(square)
                iterations.append({"step": step,
                    "estimate_range": _range(estimate_open),
                    "estimate_vs_inverse_of_opened_denominator": _error_metrics(estimate_open, inverse_target),
                    "square_vs_opened_estimate_squared": _error_metrics(square_open, estimate_open.square()),
                    "max_abs_relative_residual": (1 - denominator_open * estimate_open).abs().max().item()})
                estimate += estimate - square.mul_(denominator)
                if rank == 0:
                    print(json.dumps({"event": "newton_step_complete", "iteration": iteration,
                                      "step": step}), flush=True)
            estimate_open = _open(estimate)
            native_weights = scores.softmax(dim=-1)
            native_weights_open = _open(native_weights)
            native_output_open = _open(native_weights.matmul(cv))
            if rank == 0:
                exp_steps = crypten.cfg.functions.exp_iterations
                approx_exp = (1 + logits_open / (2 ** exp_steps)).pow(2 ** exp_steps)
                row = {
                    "iteration": iteration,
                    "score_range": _range(score_open),
                    "score_vs_plain_qk": _error_metrics(score_open, reference_scores),
                    "maximum_vs_opened_score": _error_metrics(maximum_open, score_open.max(dim=-1, keepdim=True).values),
                    "logits_vs_opened_score_and_max": _error_metrics(logits_open, score_open - maximum_open),
                    "numerator_vs_exp_of_opened_logits": _error_metrics(numerator_open, logits_open.exp()),
                    "numerator_vs_crypten_float_approx_of_opened_logits": _error_metrics(numerator_open, approx_exp),
                    "denominator_range": _range(denominator_open),
                    "denominator_vs_sum_opened_numerator": _error_metrics(denominator_open, numerator_open.sum(dim=-1, keepdim=True)),
                    "native_inverse_vs_inverse_opened_denominator": _error_metrics(inverse_open, inverse_target),
                    "native_inverse_max_abs_relative_residual": (1 - denominator_open * inverse_open).abs().max().item(),
                    "newton_final_vs_inverse_opened_denominator": _error_metrics(estimate_open, inverse_target),
                    "newton_iterations": iterations,
                    "weights_vs_opened_numerator_times_opened_inverse": _error_metrics(weights_open, numerator_open * inverse_open),
                    "expanded_output_vs_opened_weights_times_plain_v": _error_metrics(output_open, torch.matmul(weights_open, v)),
                    "expanded_output_vs_plain_attention": _error_metrics(output_open, reference_output),
                    "native_weights_vs_plain_softmax": _error_metrics(native_weights_open, torch.softmax(reference_scores, dim=-1)),
                    "native_output_vs_opened_weights_times_plain_v": _error_metrics(native_output_open, torch.matmul(native_weights_open, v)),
                    "native_output_vs_plain_attention": _error_metrics(native_output_open, reference_output),
                }
                records.append(row)
                print(json.dumps({"event": "diagnostic_iteration_complete", "iteration": iteration,
                                  "denominator_range": row["denominator_range"],
                                  "reciprocal_relative_residual": row["native_inverse_max_abs_relative_residual"],
                                  "native_max_error": row["native_output_vs_plain_attention"]["max_abs"]}), flush=True)
        if rank == 0:
            report = {"purpose": "public-fixture numerical diagnosis; intermediate values opened; no timing claims",
                      "backend": "gloo", "world_size": 3, "shape_b_h_n_d": shape,
                      "fixture": "CPU synthetic normal, Q17 input quantization", "seed": options["seed"],
                      "precision_bits": options["precision_bits"], "repeats": options["repeats"],
                      "crypten_import": str(Path(crypten.__file__).resolve()),
                      "crypten_config": {"exp_iterations": crypten.cfg.functions.exp_iterations,
                                         "reciprocal_method": crypten.cfg.functions.reciprocal_method,
                                         "reciprocal_nr_iters": crypten.cfg.functions.reciprocal_nr_iters,
                                         "reciprocal_initial": crypten.cfg.functions.reciprocal_initial},
                      "records": records}
            path = Path(options["output"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"report": str(path)}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_parse_shape, default=(1, 4, 197, 256))
    parser.add_argument("--precision-bits", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--input-std", type=float, default=0.25)
    parser.add_argument("--input-bound", type=float, default=8.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not 8 <= args.precision_bits <= 24:
        parser.error("repeats must be positive and precision-bits must be in [8,24]")
    if args.input_std <= 0 or args.input_bound <= 0:
        parser.error("input-std and input-bound must be positive")
    if torch.cuda.device_count() < 3:
        parser.error("three CUDA devices are required")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options["output"] = str(args.output)
    with tempfile.TemporaryDirectory(prefix="crypten-softmax-diag-", dir=args.output.parent) as directory:
        mp.spawn(_worker, args=(Path(directory, "store").as_uri(), options), nprocs=3, join=True)


if __name__ == "__main__":
    main()
