"""Compare standard Attention and grouped hAttention cores on three GPUs."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile

import torch
import torch.multiprocessing as mp

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide
from slayernorm.benchmark_crypten_cuda import _crypten_measure, _ring_measure, _zero_phase, _plain
from .benchmark_cuda import _error_metrics, _parse_grid, _parse_shape
from .config import infer_patch_grid, normalize_head_types
from .hybrid import prepare_hybrid, secure_hybrid
from .protocol import FixedPointProfile
from .reference import grouped_hattention, rsa_attention
from .rsa import RSAProfile

METHODS = ("standard", "rsa", "lsa", "mixer", "hybrid")


def standard_attention(q, k, v):
    """CrypTen-native approximate softmax, without projections or dropout."""
    scores = q.matmul(k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    return scores.softmax(dim=-1).matmul(v)


def _paths(method, count, hybrid):
    return hybrid if method == "hybrid" else (method,) * count


def _summary(report):
    lines = ["# Attention core comparison", "",
             "Three-party maximum time; mean of non-warmup repetitions. "
             "Communication is system-level sent-payload equivalent in KiB.", "",
             "| Method | Head paths | Online (s) | Offline (s) | Online (KiB) | Phases/rank | MAE | Max error | Valid |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for method in report["methods"]:
        rows = [r["methods"][method] for r in report["records"] if not r["warmup"]]
        mean = lambda fn: statistics.mean(fn(row) for row in rows)
        valid = all(row["correctness_passed"] for row in rows)
        paths = ", ".join(report["head_types"][method])
        lines.append(f"| {method} | {paths} | "
                     f"{mean(lambda r: r['online']['seconds_max']):.6f} | "
                     f"{mean(lambda r: r['offline']['seconds_max']):.6f} | "
                     f"{mean(lambda r: r['online']['system_bytes']) / 1024:.1f} | "
                     f"{mean(lambda r: max(r['online']['rounds_per_rank'])):.1f} | "
                     f"{mean(lambda r: r['error_vs_own_reference']['mae']):.6g} | "
                     f"{max(r['error_vs_own_reference']['max_abs'] for r in rows):.6g} | "
                     f"{'yes' if valid else 'INVALID'} |")
    lines.extend(["", "Standard uses CrypTen native softmax; others use experimental Ring3. "
                  "CrypTen provider work is included in online; Ring3 preparation is offline. "
                  "Framework phase counters and Gloo transport paths differ, so rounds and "
                  "times are engineering measurements, not equal protocol-round definitions.",
                  "", "Scope: Q/K/V to head output; projections and full ViT are excluded. "
                  "Each error is against its own plaintext formula. INVALID rows must not "
                  "be used for speedup claims. Ruler privacy is not validated."])
    return "\n".join(lines) + "\n"


def _worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND=options["backend"])
    import crypten
    crypten.cfg.encoder.precision_bits = 17
    if options["backend"] == "nccl":
        from crypten_nccl_compat import install_nccl_seed_sync_compat
        install_nccl_seed_sync_compat(crypten)
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    try:
        base = CudaRing3(device=device, reveal_mode="p2p")
        wide = CudaRing3Wide(base)
        fixed = FixedPointProfile(input_bound=options["input_bound"])
        rsa_profile = RSAProfile()
        shape = tuple(options["shape"])
        hybrid = tuple(options["head_types"])
        rho = shape[-1] ** -0.5
        grid = tuple(options["patch_grid"]) if options["patch_grid"] else None
        generator = torch.Generator().manual_seed(options["seed"])
        values = [(options["input_std"] * torch.randn(shape, generator=generator,
                  dtype=torch.float64)).clamp(-fixed.input_bound, fixed.input_bound).to(device)
                  for _ in range(3)]
        q, k, v = [fixed.decode(fixed.encode(value)) for value in values]
        head_types = {m: (("softmax",) * shape[1] if m == "standard" else
                          _paths(m, shape[1], hybrid)) for m in options["methods"]}
        references = {}
        checks = {}
        for method in options["methods"]:
            if method == "standard":
                scores = torch.matmul(q, k.transpose(-2, -1)) * rho
                references[method] = torch.matmul(torch.softmax(scores, dim=-1), v)
                continue
            paths = head_types[method]
            references[method] = grouped_hattention(
                q, k, v, paths, rho=rho, epsilon=rsa_profile.effective_epsilon,
                srelu_radius=rsa_profile.radius, patch_grid=grid,
                has_cls_token=options["has_cls_token"])
            indices = [i for i, path in enumerate(paths) if path == "rsa"]
            if indices:
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                _, aux = rsa_attention(q.index_select(1, idx), k.index_select(1, idx),
                                       v.index_select(1, idx), epsilon=rsa_profile.effective_epsilon,
                                       radius=rsa_profile.radius, return_aux=True)
                score = aux["scores"].abs().max().item()
                low, high = aux["denominator"].min().item(), aux["denominator"].max().item()
                checks[method] = {"max_abs_score": score, "denominator_min": low,
                                  "denominator_max": high}
                if score >= rsa_profile.score_bound or not (
                    rsa_profile.denominator_min <= low <= high <= rsa_profile.denominator_max
                ):
                    raise ValueError(f"public RSA fixture violates numerical contract: {checks[method]}")

        records = []
        for iteration in range(options["repeats"] + 1):
            encrypted, sharing = _crypten_measure(device, crypten, lambda: tuple(
                crypten.cryptensor(value, src=0, precision=17, device=device) for value in (q, k, v)))
            row = {"warmup": iteration == 0, "input_sharing": sharing, "methods": {}}
            methods = options["methods"]
            ordered = methods[iteration % len(methods):] + methods[:iteration % len(methods)]
            for method in ordered:
                if method == "standard":
                    offline = _zero_phase()
                    output, online = _crypten_measure(device, crypten,
                                                      lambda: standard_attention(*encrypted))
                    opened, reconstruction = _crypten_measure(device, crypten,
                                                               lambda: output.get_plain_text())
                    actual = _plain(opened)
                else:
                    shares = tuple(item.share.data for item in encrypted)
                    material, offline = _ring_measure(device, base, lambda: prepare_hybrid(
                        base, wide, shape, head_types[method], rho, fixed, rsa_profile))
                    output, online = _ring_measure(device, base, lambda: secure_hybrid(
                        base, wide, *shares, material, patch_grid=grid,
                        has_cls_token=options["has_cls_token"]))
                    opened, reconstruction = _ring_measure(device, base, lambda: base.reveal(output))
                    actual = fixed.decode(opened)
                errors = _error_metrics(actual, references[method])
                passed = all(math.isfinite(x) for x in errors.values()) and errors["max_abs"] <= options["max_abs_error"]
                row["methods"][method] = {"offline": offline, "online": online,
                    "output_reconstruction": reconstruction, "error_vs_own_reference": errors,
                    "correctness_passed": passed,
                    "online_includes_provider": method == "standard"}
                if rank == 0:
                    print(json.dumps({"event": "method_complete", "warmup": iteration == 0,
                                      "method": method, "seconds": online["seconds_max"],
                                      "max_abs_error": errors["max_abs"], "passed": passed}), flush=True)
            if rank == 0:
                records.append(row)

        if rank == 0:
            passed = all(x["correctness_passed"] for row in records if not row["warmup"]
                         for x in row["methods"].values())
            report = {"status": "benchmark_passed_not_security_validated" if passed else "benchmark_failed_correctness",
                "framework": "CrypTen MPCTensor standard; custom Ring3 hAttention",
                "backend": options["backend"], "world_size": 3, "shape_b_h_n_d": shape,
                "methods": options["methods"], "head_types": head_types, "patch_grid": grid,
                "has_cls_token": options["has_cls_token"], "rho": rho,
                "rsa_profile": vars(rsa_profile), "effective_epsilon": rsa_profile.effective_epsilon,
                "rsa_public_fixture_contract": checks,
                "fixture_contract_scope": "public synthetic plaintext only; secret values not checked",
                "fixture": "same CPU-generated normal Q17 Q/K/V for all methods",
                "seed": options["seed"], "input_std": options["input_std"],
                "repeats": options["repeats"], "max_abs_error_threshold": options["max_abs_error"],
                "crypten_import": str(Path(crypten.__file__).resolve()),
                "python": platform.python_version(), "torch": torch.__version__,
                "scope": "attention core Q/K/V to per-head output; excludes projections, residual, dropout and ViT",
                "timing": "CrypTen provider included online; Ring3 preprocessing separate; Gloo Ring3 stages via CPU",
                "communication": "CrypTen send+receive divided by two; Ring3 sent payload sum; collective payload estimated; round counters differ",
                "security": "experimental Ruler privacy not validated; CUDA Philox is not a reviewed CSPRNG",
                "records": records}
            output_dir = Path(options["output"])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            (output_dir / "summary.md").write_text(_summary(report))
            print(json.dumps({"status": report["status"], "report": str(output_dir / "comparison.json"),
                              "summary": str(output_dir / "summary.md")}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--shape", type=_parse_shape, default=(1, 4, 17, 16))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--head-types", nargs="+", choices=("rsa", "lsa", "mixer"))
    parser.add_argument("--patch-grid", type=_parse_grid)
    parser.add_argument("--no-cls-token", dest="has_cls_token", action="store_false")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--input-std", type=float, default=0.25)
    parser.add_argument("--input-bound", type=float, default=8.0)
    parser.add_argument("--max-abs-error", type=float, default=0.005)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must not contain duplicates")
    if args.repeats < 1 or any(not math.isfinite(x) or x <= 0 for x in
                               (args.input_std, args.input_bound, args.max_abs_error)):
        parser.error("repeats, input-std, input-bound and max-abs-error must be positive")
    if args.head_types is None:
        args.head_types = [("rsa", "lsa", "mixer")[i % 3] for i in range(args.shape[1])]
    try:
        args.head_types = normalize_head_types(args.head_types, args.shape[1])
        needs_mixer = "mixer" in args.methods or (
            "hybrid" in args.methods and "mixer" in args.head_types)
        if args.patch_grid is None and needs_mixer:
            args.patch_grid = infer_patch_grid(args.shape[2], args.has_cls_token)
        if args.patch_grid and math.prod(args.patch_grid) != args.shape[2] - int(args.has_cls_token):
            raise ValueError("patch-grid does not match token count")
    except ValueError as error:
        parser.error(str(error))
    if torch.cuda.device_count() < 3:
        parser.error("at least three visible CUDA devices are required")
    args.output = args.output.resolve()
    if any((args.output / name).exists() for name in ("comparison.json", "summary.md")):
        parser.error("output report already exists; choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options["output"] = str(args.output)
    with tempfile.TemporaryDirectory(prefix="hattention-compare-", dir=args.output.parent) as directory:
        mp.spawn(_worker, args=(Path(directory, "store").as_uri(), options), nprocs=3, join=True)
    report = json.loads((args.output / "comparison.json").read_text())
    if report["status"] == "benchmark_failed_correctness":
        parser.exit(1, "correctness check failed; inspect comparison.json and summary.md\n")


if __name__ == "__main__":
    main()
