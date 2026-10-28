"""Three-party GPU RSA smoke benchmark with public synthetic inputs."""

import argparse
import json
import math
from pathlib import Path
import platform
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide
from .benchmark_cuda import _error_metrics, _init_process_group, _measure, _parse_shape
from .protocol import FixedPointProfile
from .reference import rsa_attention
from .rsa import RSAProfile, prepare_rsa, secure_rsa


def _worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    _init_process_group(rank, rendezvous, device, options["backend"])
    try:
        base = CudaRing3(device=device, reveal_mode="p2p")
        wide = CudaRing3Wide(base)
        shape = tuple(options["shape"])
        profile = RSAProfile()
        fixed = FixedPointProfile(input_bound=options["input_bound"])
        generator = torch.Generator().manual_seed(options["seed"])
        values = [(options["input_std"] * torch.randn(shape, generator=generator,
                  dtype=torch.float64)).clamp(-options["input_bound"], options["input_bound"]).to(device)
                  for _ in range(3)]
        encoded = [fixed.encode(value) for value in values]
        quantized = [fixed.decode(value) for value in encoded]
        reference, aux = rsa_attention(*quantized, epsilon=profile.effective_epsilon,
                                       radius=profile.radius, return_aux=True)
        score_abs_max = aux["scores"].abs().max().item()
        denominator_min = aux["denominator"].min().item()
        denominator_max = aux["denominator"].max().item()
        # This check is permitted only because these benchmark inputs are public
        # synthetic fixtures. It does not check the secret fixed-point runtime.
        if score_abs_max >= profile.score_bound or not (
            profile.denominator_min <= denominator_min <= denominator_max <= profile.denominator_max
        ):
            raise ValueError("public fixture violates RSA score/denominator contract: "
                             f"max_abs_score={score_abs_max}, denominator=[{denominator_min},{denominator_max}]")

        records = []
        for iteration in range(options["repeats"] + 1):
            shares, sharing = _measure(base, lambda: tuple(base.private(x, owner=0) for x in encoded))
            material, offline = _measure(base, lambda: prepare_rsa(base, wide, shape, profile))
            output, online = _measure(base, lambda: secure_rsa(base, wide, *shares, material))
            opened, reconstruction = _measure(base, lambda: base.reveal(output))
            if rank == 0:
                errors = _error_metrics(fixed.decode(opened), reference)
                records.append({"warmup": iteration == 0, "input_sharing": sharing,
                                "offline": offline, "online": online,
                                "output_reconstruction": reconstruction,
                                "error_vs_quantized_plaintext_rsa": errors,
                                "correctness_passed": all(math.isfinite(v) for v in errors.values())
                                and errors["max_abs"] <= options["max_abs_error"]})
        if rank == 0:
            output_dir = Path(options["output"])
            output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "status": ("benchmark_passed_not_security_validated" if all(
                    row["correctness_passed"] for row in records if not row["warmup"]
                ) else "benchmark_failed_correctness"),
                "framework": f"custom Ring3 / {options['backend'].upper()} / CUDA",
                "backend": options["backend"], "world_size": 3, "operator": "rsa",
                "shape_b_h_n_d": shape, "repeats": options["repeats"],
                "max_abs_error_threshold": options["max_abs_error"],
                "fixture": "public synthetic normal, generated on CPU before timing",
                "fixture_contract_check": {"max_abs_score": score_abs_max,
                                           "denominator_min": denominator_min,
                                           "denominator_max": denominator_max,
                                           "scope": "plaintext synthetic reference only; secret fixed-point intermediates are not checked"},
                "profile": {"epsilon": profile.epsilon, "effective_epsilon": profile.effective_epsilon,
                            "radius": profile.radius, "score_bound": profile.score_bound,
                            "denominator_min": profile.denominator_min, "denominator_max": profile.denominator_max,
                            "reciprocal_fraction_bits": profile.reciprocal_fraction_bits,
                            "newton_iterations": profile.iterations},
                "python": platform.python_version(), "torch": torch.__version__,
                "communication": "sum of Ring3 sent payload bytes across three parties; Gloo CPU staging included in phase time; metric gathers excluded",
                "security": "experimental Ruler openings are not privacy validated; CUDA Philox is not a reviewed CSPRNG; the secret range contract is not checked",
                "records": records,
            }
            report_path = output_dir / "comparison.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            last = records[-1]
            print(json.dumps({"report": str(report_path), "records": len(records),
                              "online_seconds_last": last["online"]["seconds_max"],
                              "online_bytes": last["online"]["sent_bytes_sum"],
                              "error_vs_quantized_plaintext_rsa": last["error_vs_quantized_plaintext_rsa"]}, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--shape", type=_parse_shape, default=(1, 1, 17, 16))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--input-std", type=float, default=0.25)
    parser.add_argument("--input-bound", type=float, default=8.0)
    parser.add_argument("--max-abs-error", type=float, default=0.005)
    args = parser.parse_args()
    if (args.repeats < 1 or any(not math.isfinite(x) or x <= 0 for x in
                               (args.input_std, args.input_bound, args.max_abs_error))):
        parser.error("repeats, input-std, input-bound and max-abs-error must be positive")
    if torch.cuda.device_count() < 3:
        parser.error("at least three visible CUDA devices are required")
    args.output = args.output.resolve()
    if (args.output / "comparison.json").exists():
        parser.error("output report already exists; choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options["output"] = str(args.output)
    with tempfile.TemporaryDirectory(prefix="hattention-rsa-", dir=args.output.parent) as directory:
        mp.spawn(_worker, args=(Path(directory, "store").as_uri(), options), nprocs=3, join=True)
    report = json.loads((args.output / "comparison.json").read_text())
    if report["status"] == "benchmark_failed_correctness":
        parser.exit(1, "RSA correctness check failed; inspect comparison.json\n")


if __name__ == "__main__":
    main()
