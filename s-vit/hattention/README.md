# hAttention

This directory contains the isolated implementation of the heterogeneous secure
attention method.  The original `mpcvit` sources remain unchanged.

## Plaintext reference scope

The current code provides plaintext, differentiable references for:

- RSA with either exact ReLU or a non-negative smooth ReLU;
- reassociated LSA, `rho / N * Q (K^T V)`, without an `N x N` intermediate;
- a public reshape-transpose-reshape Mixer that preserves a leading CLS token;
- grouped execution and restoration of the original head order;
- a fixed-structure ViT attention module with packed QKV weights.

For `N=197`, `257`, and `577`, Mixer preserves the CLS token and permutes the
remaining `14x14`, `16x16`, and `24x24` patch grids.

## sReLU convention

The method chapter writes RSA's activation in two explicit stages.  First use
the zero-at-origin shifted form

```text
(x + sqrt(x^2 + r^2) - r) / 2
```

Then add the public constant `r/2` to every element before row summation and
normalization.  Therefore the effective non-negative RSA weight is

```text
(x + sqrt(x^2 + r^2) - r) / 2 + r / 2
```

which is algebraically equal to `(x + sqrt(x^2 + r^2)) / 2`, with `r = 0.0055`.
The intermediate shifted form is retained as `legacy_zero_origin_srelu` for
auditability; it must not be normalized before the public `r/2` correction.

## Tests

From the repository root, with PyTorch installed:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main"
python -m unittest discover -s s-vit/hattention/tests -v
```

## Ring3 linear paths

`protocol.py` now provides the first secure deployment increment:

- one-round batched Ring3 matrix multiplication shared by CPU/Gloo and
  CUDA/NCCL runtimes;
- Q17 encode/decode profiles and one-time probabilistic truncation material;
- secure reassociated LSA with two secret matrix multiplications;
- grouped secure LSA plus communication-free public Mixer execution.

The grouped secure entry point deliberately rejects RSA configurations.  This
prevents an incomplete plaintext fallback from being reported as secure RSA;
Ruler-sReLU and the row reciprocal will be enabled only after their independent
fixed-point and three-party tests pass.

## CUDA smoke benchmark

The current three-party benchmark covers LSA and Mixer heads only. From the
repository root on a server with three visible GPUs:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main"
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
python -m hattention.benchmark_cuda \
  --shape 1x4x17x16 \
  --head-types lsa lsa mixer mixer \
  --repeats 1 \
  --output "$HOME/home/tmp/hattention-linear-smoke"
```

For a ViT-style operator case, use for example `--shape 1x4x257x64`; the
`16x16` patch grid is inferred while the CLS token is preserved.  The report
separates input sharing, offline truncation-material preparation, online
execution, output reconstruction, sent payload bytes, rounds, and error.  The
benchmark is experimental and does not claim production security validation.

Secure RSA normalization, the grouped three-path benchmark, cost-table-driven
search, pruning, and distillation remain subsequent phases.

## Standalone Ruler-sReLU

`srelu.py` implements the RSA activation as two explicit fixed-point stages:

1. securely compute `(x + sqrt(x^2 + r^2) - r) / 2` with the experimental
   96-bit Ruler masked-square-root path;
2. add public `r/2` elementwise before any row sum or reciprocal.

The one-time material contains an independently sampled scale per element, its
96-bit square, a secret Q50 reciprocal generated during preprocessing, and
fresh truncation masks. The online result is Q17 and the current fixed-point
contract requires `|x| < 16`, keeping the pre-truncation square-root recovery
strictly below the signed-ring bound. To run the smallest CUDA
smoke test on three GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
python -m hattention.benchmark_srelu_cuda \
  --shape 1x2x17x17 \
  --repeats 1 \
  --output "$HOME/home/tmp/hattention-srelu-smoke"
```

The report compares the secure result with the continuous two-stage sReLU, a
reference using the quantized Q16/Q17 public constants, and ordinary ReLU.
Ruler's masked opening remains experimental and is not presented as a completed
privacy proof.
