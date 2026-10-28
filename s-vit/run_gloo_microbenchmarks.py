"""Backward-compatible entry point for the complete Gloo/CUDA runs."""

from run_microbenchmarks import main


if __name__ == "__main__":
    main(default_backend="gloo")
