"""Diagnose CrypTen's native comparison path without replacing it.

This runner intentionally calls ``MPCTensor._ltz()``, ``ge(0)`` and
``relu()`` separately.  It is useful for distinguishing a binary-sharing
comparison bug from a ReLU wrapper bug on CPU/Gloo versus CUDA/Gloo, mirroring
the CrypTen benchmark launcher rather than creating a separate NCCL group.
"""

import argparse
import json
from pathlib import Path
import platform
import sys

import torch


def _assert_vendored_crypten(crypten):
    expected = (
        Path(__file__).resolve().parents[2]
        / "CrypTen-main"
        / "benchmarks"
        / "crypten"
    ).resolve()
    actual = Path(crypten.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError(
            f"Expected benchmark CrypTen at {expected}, imported {actual}"
        )
    return str(actual / "__init__.py")


def _parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must use AxB notation") from error
    if not shape or any(dimension < 1 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def _fixture(shape):
    probe = torch.tensor(
        [-1.75, -0.5, -0.0625, 0.0, 0.0625, 0.5, 1.75],
        dtype=torch.float64,
    )
    return probe.repeat((probe.numel() + int(torch.tensor(shape).prod()) - 1) // probe.numel())[
        : int(torch.tensor(shape).prod())
    ].reshape(shape)


def _plain(value):
    result = value.get_plain_text()
    if hasattr(result, "tensor"):
        result = result.tensor()
    return result.detach().cpu().double()


def _metrics(actual, expected):
    difference = (actual - expected).abs()
    return {
        "mae": difference.mean().item(),
        "max_abs": difference.max().item(),
        "values": actual.reshape(-1).tolist(),
    }


def _describe(value):
    return {
        "class": type(value).__name__,
        "ptype": str(value.ptype),
        "share_class": type(value.share).__name__,
        "share_dtype": str(value.share.dtype),
        "share_device": str(value.share.device),
        "precision_bits": value.encoder._precision_bits,
        "scale": value.encoder.scale,
    }


def worker(options):
    """Run inside CrypTen's own MultiProcessLauncher worker."""
    import crypten
    import crypten.communicator as comm

    rank = comm.get().get_rank()
    use_cuda = options["device"] == "cuda"
    # Match benchmark.py's --device cuda behavior by default: each party uses
    # the default CUDA device unless the caller launches a multi-GPU variant.
    device = torch.device("cuda" if use_cuda else "cpu")

    crypten_path = _assert_vendored_crypten(crypten)
    plain = _fixture(options["shape"]).to(device)
    encrypted = crypten.cryptensor(
        plain,
        src=0,
        precision=17,
        device=device,
    )
    operations = {
        "ltz_private": lambda x: x._ltz(),
        "ge_zero": lambda x: x.ge(0),
        "relu_native": lambda x: x.relu(),
    }
    expected = {
        "ltz_private": (plain < 0).to(torch.float64).cpu(),
        "ge_zero": (plain >= 0).to(torch.float64).cpu(),
        "relu_native": torch.relu(plain).cpu(),
    }

    row = {
        "rank": rank,
        "crypten_import": crypten_path,
        "device": str(device),
        "input": plain.cpu().reshape(-1).tolist(),
        "encrypted": _describe(encrypted),
        "operations": {},
    }
    for name, operation in operations.items():
        output = operation(encrypted.clone())
        opened = _plain(output)
        row["operations"][name] = {
            "output": opened.reshape(-1).tolist(),
            "output_description": _describe(output),
        }
        row["operations"][name]["metrics"] = _metrics(
            opened,
            expected[name],
        )

    if rank == 0:
        output_path = Path(options["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "diagnostic_complete",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "backend": "Gloo/CUDA" if use_cuda else "Gloo/CPU",
            "world_size": 3,
            "shape": list(options["shape"]),
            "row": row,
            "note": (
                "Launched with the same CrypTen MultiProcessLauncher and "
                "default Gloo backend as CrypTen-main/benchmarks/benchmark.py. "
                "Calls unmodified MPCTensor methods; no adapter or sReLU."
            ),
        }
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--shape", type=_parse_shape, default=(1, 17))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and torch.cuda.device_count() < 3:
        parser.error("CUDA mode requires at least three visible GPUs")

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_crypten_parent = (
        Path(__file__).resolve().parents[2]
        / "CrypTen-main"
        / "benchmarks"
    ).resolve()
    sys.path.insert(0, str(benchmark_crypten_parent))
    import crypten
    from examples import multiprocess_launcher

    options = {
        "device": args.device,
        "shape": args.shape,
        "output": str(args.output),
    }
    # Mirror benchmark.py: initialize CrypTen in the parent, then use its own
    # launcher, which configures child processes with the default Gloo backend.
    crypten.init()
    launcher = multiprocess_launcher.MultiProcessLauncher(
        3,
        worker,
        fn_args=options,
    )
    try:
        launcher.start()
        launcher.join()
    finally:
        launcher.terminate()
        crypten.uninit()


if __name__ == "__main__":
    main()
