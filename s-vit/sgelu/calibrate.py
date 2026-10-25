"""Calibrate one public mu per named pre-GeLU activation array in an NPZ file."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .sgelu import calibrate_mu
from .validate import DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activations", type=Path, help="public pre-GeLU activations, one named array per layer")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "calibration.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with np.load(args.activations, allow_pickle=False) as bundle:
        if not bundle.files:
            parser.error("activation bundle has no layers")
        result = {name: calibrate_mu(torch.from_numpy(bundle[name])).to_dict() for name in bundle.files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": str(args.activations.resolve()),
                                      "formula": "G2-v2", "layers": result}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
