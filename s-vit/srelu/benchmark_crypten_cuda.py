"""Unified three-party CUDA/NCCL microbenchmark for sReLU and ReLU baselines."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide

from .baselines import (
    METHOD_IMPLEMENTATION_CLASS,
    METHOD_MAIN_TABLE_SCOPE,
    METHOD_PROVENANCE,
    METHOD_YEAR,
    crypten_relu,
    effective_signed_bits,
    method_reference,
)
from .protocol import SReLUProfile, prepare_srelu, ruler_srelu


METHODS = (
    "crypten_native",
    "bicoptor2_keybits",
    "hummingbird_eco",
    "hummingbird_budget",
    "srelu",
)

EXACT_REFERENCE_METHODS = frozenset(("crypten_native", "hummingbird_eco"))


def _parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shape must use x-separated positive integers"
        ) from error
    if not shape or any(dimension < 1 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def _barrier(device):
    dist.barrier(device_ids=[device.index])


def _sync(device):
    torch.cuda.synchronize(device)


def _aggregate(device, seconds, stats, *, bidirectional_counter=False):
    row = torch.tensor(
        [seconds, stats["bytes"], stats["rounds"], stats["time"]],
        dtype=torch.float64,
        device=device,
    )
    rows = [torch.empty_like(row) for _ in range(3)]
    dist.all_gather(rows, row)
    values = torch.stack(rows).cpu()
    logged_bytes = int(values[:, 1].sum().item())
    system_bytes = logged_bytes // 2 if bidirectional_counter else logged_bytes
    return {
        "seconds_max": values[:, 0].max().item(),
        "sent_bytes_sum": logged_bytes,
        "system_bytes": system_bytes,
        "rounds_per_rank": values[:, 2].to(torch.int64).tolist(),
        "comm_time_sum": values[:, 3].sum().item(),
    }


def _crypten_measure(device, crypten, function):
    _barrier(device)
    crypten.reset_communication_stats()
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    seconds = time.perf_counter() - started
    stats = crypten.communicator.get().get_communication_stats()
    return result, _aggregate(
        device,
        seconds,
        stats,
        bidirectional_counter=True,
    )


def _ring_measure(device, base, function):
    _barrier(device)
    _sync(device)
    bytes_before = base.sent_bytes
    phases_before = base.phases
    started = time.perf_counter()
    result = function()
    _sync(device)
    seconds = time.perf_counter() - started
    return result, _aggregate(
        device,
        seconds,
        {
            "bytes": base.sent_bytes - bytes_before,
            "rounds": base.phases - phases_before,
            "time": 0.0,
        },
    )


def _zero_metrics():
    return {
        "seconds_max": 0.0,
        "sent_bytes_sum": 0,
        "system_bytes": 0,
        "rounds_per_rank": [0, 0, 0],
        "comm_time_sum": 0.0,
    }


def _error_metrics(actual, expected):
    difference = (actual - expected).abs().to(torch.float64)
    return {
        "mae": difference.mean().item(),
        "max_abs": difference.max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "p99_abs": torch.quantile(difference.reshape(-1), 0.99).item(),
    }


def _masked_mean(value, mask):
    selected = value[mask]
    return selected.mean().item() if selected.numel() else 0.0


def _semantic_diagnostics(actual, source):
    """Detect fast-but-wrong identity, absolute-value and sign-leakage paths."""

    negative = source < 0
    nonnegative = ~negative
    quantum = 1.0 / (1 << 17)
    return {
        "all_finite": bool(torch.isfinite(actual).all().item()),
        "error_vs_identity": _error_metrics(actual, source),
        "error_vs_absolute_value": _error_metrics(actual, source.abs()),
        "negative_output_abs_mean": _masked_mean(actual.abs(), negative),
        "negative_output_nonzero_fraction": _masked_mean(
            (actual.abs() > quantum).to(torch.float64), negative
        ),
        "nonnegative_payload_mae": _masked_mean(
            (actual - source).abs(), nonnegative
        ),
    }


def _correctness_gate(method, reference_error, diagnostics, options):
    threshold = (
        options["exact_max_abs"]
        if method in EXACT_REFERENCE_METHODS
        else options["adapted_max_abs"]
    )
    passed = (
        diagnostics["all_finite"]
        and reference_error["max_abs"] <= threshold
    )
    return {
        "passed": passed,
        "reference": (
            "exact_relu"
            if method in EXACT_REFERENCE_METHODS
            else "declared_method_reference"
        ),
        "max_abs_threshold": threshold,
        "observed_max_abs": reference_error["max_abs"],
        "timing_claim_allowed": passed,
    }


def _plain(value):
    return (value.tensor() if hasattr(value, "tensor") else value).double()


class CrypTenCudaSReLU:
    """Ruler-sReLU adapter on CrypTen's initialized NCCL process group."""

    def __init__(self, device, profile, reveal_mode):
        from crypten.cuda import CUDALongTensor
        from crypten.mpc import MPCTensor, ptype

        self.CUDALongTensor = CUDALongTensor
        self.MPCTensor = MPCTensor
        self.ptype = ptype
        self.base = CudaRing3(device=device, reveal_mode=reveal_mode)
        self.wide = CudaRing3Wide(self.base)
        self.profile = profile
        self.material = None

    def prepare(self, shape):
        if self.material is not None:
            raise RuntimeError("pending sReLU material must be consumed")
        self.material = prepare_srelu(
            self.base,
            self.wide,
            tuple(shape),
            self.profile,
        )

    def __call__(self, value):
        if value.ptype != self.ptype.arithmetic:
            raise ValueError("sReLU requires arithmetic CrypTen shares")
        if value.encoder._precision_bits != self.profile.input_fraction_bits:
            raise ValueError("sReLU requires Q17 input")
        if self.material is None:
            raise RuntimeError("prepare sReLU material before online execution")
        material, self.material = self.material, None
        output = ruler_srelu(
            self.base,
            self.wide,
            value.share.data,
            material,
        )
        return self.MPCTensor.from_shares(
            self.CUDALongTensor(output),
            precision=self.profile.input_fraction_bits,
        )


def _fixture(shape, options, seed_offset):
    generator = torch.Generator().manual_seed(options["seed"] + seed_offset)
    strict_bound = options["input_bound"] - 1.0 / (1 << 17)
    value = options["input_std"] * torch.randn(
        shape,
        generator=generator,
        dtype=torch.float64,
    )
    return value.clamp(-strict_bound, strict_bound)


def _method_metadata(options):
    metadata = {
        "crypten_native": {
            "year": METHOD_YEAR["crypten_native"],
            "provenance": METHOD_PROVENANCE["crypten_native"],
            "implementation_class": METHOD_IMPLEMENTATION_CLASS[
                "crypten_native"
            ],
            "main_table_scope": METHOD_MAIN_TABLE_SCOPE["crypten_native"],
            "comparison_bits": 64,
        },
        "bicoptor2_keybits": {
            "year": METHOD_YEAR["bicoptor2_keybits"],
            "provenance": METHOD_PROVENANCE["bicoptor2_keybits"],
            "implementation_class": METHOD_IMPLEMENTATION_CLASS[
                "bicoptor2_keybits"
            ],
            "main_table_scope": METHOD_MAIN_TABLE_SCOPE[
                "bicoptor2_keybits"
            ],
            "comparison_fraction_bits": options["bicoptor_fraction_bits"],
            "comparison_bits": effective_signed_bits(
                options["input_bound"], options["bicoptor_fraction_bits"]
            ),
        },
        "hummingbird_eco": {
            "year": METHOD_YEAR["hummingbird_eco"],
            "provenance": METHOD_PROVENANCE["hummingbird_eco"],
            "implementation_class": METHOD_IMPLEMENTATION_CLASS[
                "hummingbird_eco"
            ],
            "main_table_scope": METHOD_MAIN_TABLE_SCOPE["hummingbird_eco"],
            "comparison_fraction_bits": 17,
            "comparison_bits": effective_signed_bits(
                options["input_bound"], 17
            ),
        },
        "hummingbird_budget": {
            "year": METHOD_YEAR["hummingbird_budget"],
            "provenance": METHOD_PROVENANCE["hummingbird_budget"],
            "implementation_class": METHOD_IMPLEMENTATION_CLASS[
                "hummingbird_budget"
            ],
            "main_table_scope": METHOD_MAIN_TABLE_SCOPE[
                "hummingbird_budget"
            ],
            "comparison_fraction_bits": options["hummingbird_fraction_bits"],
            "comparison_bits": effective_signed_bits(
                options["input_bound"], options["hummingbird_fraction_bits"]
            ),
        },
        "srelu": {
            "year": 2026,
            "provenance": (
                "manuscript Ruler-sReLU using the standalone Ring3 adapter on "
                "the same CrypTen NCCL process group"
            ),
            "implementation_class": "ours_experimental",
            "main_table_scope": (
                "eligible after the corrected-sReLU numerical gate and the "
                "Ruler security caveat are reported"
            ),
            "comparison_bits": None,
        },
    }
    return metadata


def _summary(report):
    lines = [
        "# sReLU CUDA micro benchmark",
        "",
        "All values below exclude warmup. Time, communication and rounds are online only.",
        "",
        "A method that fails its declared-reference correctness gate is not allowed to make a timing claim.",
        "",
        "| Method | Year | Shape | Gate | Time (s) | Comm. (KB) | Rounds | MAE vs ReLU | Max error vs ReLU |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for shape in report["shapes"]:
        shape_records = [
            record
            for record in report["records"]
            if record["shape"] == shape and not record["warmup"]
        ]
        for method in report["methods"]:
            entries = [record["methods"][method] for record in shape_records]
            time_mean = statistics.mean(
                entry["online"]["seconds_max"] for entry in entries
            )
            comm_mean = statistics.mean(
                entry["online"]["system_bytes"] for entry in entries
            ) / 1024.0
            rounds_mean = statistics.mean(
                max(entry["online"]["rounds_per_rank"]) for entry in entries
            )
            mae_mean = statistics.mean(
                entry["error_vs_relu"]["mae"] for entry in entries
            )
            max_mean = statistics.mean(
                entry["error_vs_relu"]["max_abs"] for entry in entries
            )
            valid = all(
                entry["correctness_gate"]["passed"] for entry in entries
            )
            shape_text = "(" + ",".join(str(item) for item in shape) + ")"
            year = report["method_metadata"][method]["year"]
            if valid:
                performance = (
                    f"{time_mean:.6f} | {comm_mean:.1f} | "
                    f"{rounds_mean:.1f}"
                )
                gate = "PASS"
            else:
                performance = "-- | -- | --"
                gate = "INVALID"
            lines.append(
                f"| {method} | {year} | `{shape_text}` | {gate} | "
                f"{performance} | {mae_mean:.6e} | {max_mean:.6e} |"
            )
    lines.extend(
        [
            "",
            "`bicoptor2_keybits` is a same-backend key-bit adaptation, not the official two-round protocol.",
            "`hummingbird_*` are three-party adaptations of the paper's reduced-bit DReLU idea.",
            "An INVALID row remains in comparison.json for diagnosis, but its timing must not be used in a paper table.",
            "",
        ]
    )
    return "\n".join(lines)


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="3",
        RENDEZVOUS=rendezvous,
        DISTRIBUTED_BACKEND="nccl",
    )
    import crypten

    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    try:
        profile = SReLUProfile(
            radius=options["radius"],
            input_bound=options["input_bound"],
        )
        records = []
        for shape_index, shape_value in enumerate(options["shapes"]):
            shape = tuple(shape_value)
            plaintext_cpu = _fixture(shape, options, shape_index)
            quantized_cpu = (
                (plaintext_cpu * profile.input_scale).round()
                / profile.input_scale
            )
            quantized = quantized_cpu.to(device)
            exact_relu = torch.relu(quantized)
            corrected_srelu = profile.corrected_reference(quantized)
            references = {
                method: (
                    corrected_srelu
                    if method == "srelu"
                    else method_reference(
                        method,
                        quantized,
                        hummingbird_fraction_bits=options[
                            "hummingbird_fraction_bits"
                        ],
                        bicoptor_fraction_bits=options[
                            "bicoptor_fraction_bits"
                        ],
                    )
                )
                for method in options["methods"]
            }

            if rank == 0 and options["progress"]:
                print(
                    json.dumps(
                        {
                            "event": "shape_started",
                            "shape": list(shape),
                            "methods": options["methods"],
                        }
                    ),
                    flush=True,
                )

            for iteration in range(options["repeats"] + 1):
                encrypted, input_sharing = _crypten_measure(
                    device,
                    crypten,
                    lambda: crypten.cryptensor(
                        quantized,
                        src=0,
                        precision=17,
                        device=device,
                    ),
                )
                row = {
                    "shape": list(shape),
                    "warmup": iteration == 0,
                    "input_sharing": input_sharing,
                    "methods": {},
                }
                for method in options["methods"]:
                    if method == "srelu":
                        operator = CrypTenCudaSReLU(
                            device,
                            profile,
                            options["reveal_mode"],
                        )
                        _, offline = _ring_measure(
                            device,
                            operator.base,
                            lambda operator=operator: operator.prepare(
                                encrypted.share.shape
                            ),
                        )
                        output, online = _ring_measure(
                            device,
                            operator.base,
                            lambda operator=operator: operator(encrypted.clone()),
                        )
                    else:
                        output, online = _crypten_measure(
                            device,
                            crypten,
                            lambda method=method: crypten_relu(
                                method,
                                encrypted.clone(),
                                input_bound=options["input_bound"],
                                hummingbird_fraction_bits=options[
                                    "hummingbird_fraction_bits"
                                ],
                                bicoptor_fraction_bits=options[
                                    "bicoptor_fraction_bits"
                                ],
                            ),
                        )
                        offline = _zero_metrics()
                    opened, reconstruction = _crypten_measure(
                        device,
                        crypten,
                        lambda output=output: output.get_plain_text(),
                    )
                    if rank == 0:
                        actual = _plain(opened)
                        reference_error = _error_metrics(
                            actual,
                            references[method],
                        )
                        diagnostics = _semantic_diagnostics(actual, quantized)
                        row["methods"][method] = {
                            "offline": offline,
                            "online": online,
                            "output_reconstruction": reconstruction,
                            "error_vs_method_reference": reference_error,
                            "error_vs_relu": _error_metrics(
                                actual,
                                exact_relu,
                            ),
                            "output_min": actual.min().item(),
                            "output_max": actual.max().item(),
                            "semantic_diagnostics": diagnostics,
                            "correctness_gate": _correctness_gate(
                                method,
                                reference_error,
                                diagnostics,
                                options,
                            ),
                        }
                        if options["progress"]:
                            print(
                                json.dumps(
                                    {
                                        "event": "method_complete",
                                        "shape": list(shape),
                                        "warmup": iteration == 0,
                                        "method": method,
                                        "online_seconds": online["seconds_max"],
                                        "online_system_bytes": online[
                                            "system_bytes"
                                        ],
                                        "online_rounds": online[
                                            "rounds_per_rank"
                                        ],
                                        "correctness_passed": row["methods"][
                                            method
                                        ]["correctness_gate"]["passed"],
                                    }
                                ),
                                flush=True,
                            )
                if rank == 0:
                    records.append(row)

        if rank == 0:
            output_directory = Path(options["output"])
            output_directory.mkdir(parents=True, exist_ok=True)
            invalid_cases = [
                {
                    "shape": record["shape"],
                    "method": method,
                    "gate": record["methods"][method]["correctness_gate"],
                }
                for record in records
                if not record["warmup"]
                for method in options["methods"]
                if not record["methods"][method]["correctness_gate"]["passed"]
            ]
            report = {
                "status": (
                    "benchmark_passed"
                    if not invalid_cases
                    else "benchmark_failed_correctness"
                ),
                "framework": "CrypTen MPCTensor plus standalone Ring3 / NCCL / CUDA",
                "world_size": 3,
                "shapes": [list(shape) for shape in options["shapes"]],
                "methods": options["methods"],
                "method_metadata": _method_metadata(options),
                "repeats": options["repeats"],
                "radius": options["radius"],
                "input_bound": options["input_bound"],
                "input_std": options["input_std"],
                "seed": options["seed"],
                "reveal_mode": options["reveal_mode"],
                "correctness_policy": {
                    "exact_max_abs": options["exact_max_abs"],
                    "adapted_max_abs": options["adapted_max_abs"],
                    "invalid_timing_policy": (
                        "retain raw data for diagnosis but suppress timing in "
                        "summary and forbid performance claims"
                    ),
                },
                "invalid_cases": invalid_cases,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "fixture": (
                    "synthetic normal generated by CPU RNG before timing, "
                    "quantized to Q17"
                ),
                "communication": (
                    "system_bytes normalizes CrypTen's send-plus-receive log by "
                    "one half; Ring3 already counts three-party sent payloads"
                ),
                "timing": (
                    "CrypTen provider work occurs inside and is included in online; "
                    "Ruler-sReLU preprocessing is reported separately"
                ),
                "records": records,
                "security": (
                    "Ruler masked openings are not privacy validated; Bicoptor2 "
                    "is a key-bit same-backend adaptation rather than the official "
                    "two-round protocol; CUDA Philox is a benchmark PRG"
                ),
            }
            report_path = output_directory / "comparison.json"
            summary_path = output_directory / "summary.md"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            summary_path.write_text(_summary(report))
            print(
                json.dumps(
                    {
                        "report": str(report_path),
                        "summary": str(summary_path),
                        "records": len(records),
                        "status": report["status"],
                        "invalid_cases": len(invalid_cases),
                    },
                    indent=2,
                ),
                flush=True,
            )
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=_parse_shape,
        default=[(1, 197), (1, 257), (1, 577), (8, 257)],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--input-std", type=float, default=1.0)
    parser.add_argument("--input-bound", type=float, default=16.0)
    parser.add_argument("--radius", type=float, default=0.0055)
    parser.add_argument("--hummingbird-fraction-bits", type=int, default=8)
    parser.add_argument("--bicoptor-fraction-bits", type=int, default=2)
    parser.add_argument("--exact-max-abs", type=float, default=1e-3)
    parser.add_argument("--adapted-max-abs", type=float, default=5e-3)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--reveal-mode",
        choices=("all_reduce", "all_gather", "p2p"),
        default="p2p",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.input_std <= 0 or not math.isfinite(args.input_std):
        parser.error("input-std must be positive and finite")
    if args.input_bound <= 0 or not math.isfinite(args.input_bound):
        parser.error("input-bound must be positive and finite")
    if args.radius <= 0 or not math.isfinite(args.radius):
        parser.error("radius must be positive and finite")
    if not 0 <= args.hummingbird_fraction_bits <= 17:
        parser.error("hummingbird-fraction-bits must lie in [0,17]")
    if not 0 <= args.bicoptor_fraction_bits <= 17:
        parser.error("bicoptor-fraction-bits must lie in [0,17]")
    if args.exact_max_abs <= 0 or not math.isfinite(args.exact_max_abs):
        parser.error("exact-max-abs must be positive and finite")
    if args.adapted_max_abs <= 0 or not math.isfinite(args.adapted_max_abs):
        parser.error("adapted-max-abs must be positive and finite")
    if torch.cuda.device_count() < 3:
        parser.error("at least three visible CUDA devices are required")

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options["output"] = str(args.output)
    options["shapes"] = [tuple(shape) for shape in args.shapes]
    with tempfile.TemporaryDirectory(
        prefix="srelu-crypten-cuda-",
        dir=args.output.parent,
    ) as directory:
        mp.spawn(
            worker,
            args=(Path(directory, "store").as_uri(), options),
            nprocs=3,
            join=True,
        )
    report_path = args.output / "comparison.json"
    if report_path.exists():
        status = json.loads(report_path.read_text())["status"]
        if status != "benchmark_passed":
            raise SystemExit(
                "benchmark completed but failed correctness gates; inspect "
                f"{report_path}"
            )


if __name__ == "__main__":
    main()
