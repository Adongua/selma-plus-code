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
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
python -m unittest discover -s s-vit/hattention/tests -v
```

## Ring3 linear paths

`protocol.py` now provides the first secure deployment increment:

- one-round batched Ring3 matrix multiplication shared by CPU/Gloo and
  CUDA runtimes with Gloo or NCCL communication;
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
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2 \
python -m hattention.benchmark_cuda \
  --backend gloo \
  --shape 1x4x17x16 \
  --head-types lsa lsa mixer mixer \
  --repeats 1 \
  --output "$HOME/home/tmp/hattention-linear-gloo-smoke"
```

For a ViT-style operator case, use for example `--shape 1x4x257x64`; the
`16x16` patch grid is inferred while the CLS token is preserved.  The report
separates input sharing, offline truncation-material preparation, online
execution, output reconstruction, sent payload bytes, rounds, and error.  The
benchmark is experimental and does not claim production security validation.

The default backend is Gloo; use `--backend nccl` to select NCCL explicitly.
In Gloo mode the Ring3 runtime stages CUDA shares through CPU memory and the
copies are included in phase timing. Gloo metric gathering uses CPU tensors;
collective payload counts are estimates. Report the two backends separately.
The new Gloo/CUDA path requires server validation.

Validation and integration of standalone RSA, the grouped three-path benchmark,
cost-table-driven search, pruning, and distillation remain subsequent phases.

## Experimental standalone RSA

`rsa.py` provides an initial independent RSA path: Q17 `QK^T`, public score
scaling, standalone Ruler-sReLU with the public `r/2` correction, row sum,
Q24 Newton reciprocal, secret weight multiplication, and `AV`. It uses
one-time offline truncation and sReLU material. Its reciprocal is an
engineering baseline, not a specified manuscript protocol. The grouped
hAttention entry point still rejects RSA until this standalone implementation
is validated in three-party runs.

The experimental default profile requires every secret score to satisfy
`abs(score) < 16` and every secret corrected-sReLU row denominator to be
within `[0.01, 64]`. It also assumes that all fixed-point intermediate products
remain within the signed range required by probabilistic truncation. These
are caller obligations, not secret range checks. In particular, a positive
epsilon alone does not ensure the denominator satisfies this contract: sReLU
has a small possible fixed-point negative tail. The input fixture check in
the benchmark only examines public synthetic plaintext and cannot establish
the bounds of secret intermediate values.

From the repository root on a server with three GPUs:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2

python -m unittest discover -s s-vit/hattention/tests -v
timeout 1800 python -m hattention.benchmark_rsa_cuda \
  --backend gloo --shape 1x1x17x16 --repeats 1 \
  --output "$HOME/home/tmp/hattention-rsa-gloo-smoke"
```

The report compares against plaintext RSA on the same Q17 inputs using the
effective quantized epsilon. Formal runs exceeding the maximum-error
threshold cause `benchmark_failed_correctness` and a nonzero exit. Offline
material scales with the score matrix (`N^2`); start with the small shape.
The benchmark uses synthetic data and must not be treated as a privacy proof.
The server's `(1,1,17,16)` Gloo smoke passed its plaintext-reference error
gate; broader shapes and security remain unvalidated.

## Attention-core comparison

`benchmark_compare_cuda.py` evaluates the same Q17 synthetic Q/K/V with:

* `standard`: CrypTen MPCTensor matrix products and native approximate softmax;
* `rsa`, `lsa`, `mixer`: all heads assigned to the respective Ring3 path;
* `hybrid`: RSA, LSA and Mixer heads grouped according to `--head-types`.

All measurements cover only the attention core from shared Q/K/V to per-head
output. They exclude QKV/output projection, sharing and reconstruction from
online time. CrypTen provider work is included online; one-use Ring3 materials
are generated separately. Gloo stages Ring3 CUDA shares via CPU, whereas
CrypTen has its own transport; the two communication-round counters are not
identical protocol-round definitions. Error is against each method's own
plaintext formula, not standard Attention for every method. A failed error
gate marks the report INVALID and causes a nonzero exit. The public-fixture
RSA range check does not check secrets. Ruler security remains unvalidated.

Begin with a small shape (RSA material grows with the number of score entries):

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2
python -m unittest discover -s s-vit/hattention/tests -v
timeout 1800 python -m hattention.benchmark_compare_cuda \
  --backend gloo --shape 1x4x17x16 \
  --head-types rsa lsa mixer mixer --repeats 1 \
  --output "$HOME/home/tmp/hattention-compare-gloo-smoke"
```

Inspect `comparison.json` and `summary.md` before running larger shapes or
making speedup claims. Use `--methods standard rsa lsa mixer hybrid` to select
methods; `--patch-grid` or `--no-cls-token` controls Mixer token layout.
The `hybrid` measurement is a grouped attention-core prototype, not end-to-end
ViT inference or a trained/distilled model.

### Four-shape attention-core microbenchmark

Run `python -m hattention.run_microbenchmark --output-root PATH` to compare
standard, RSA, LSA and Mixer, with each shape/method in an isolated three-party
process group. Defaults are `(1,4,197,64)`, `(1,4,257,64)`, `(1,4,577,64)` and
`(8,4,257,64)` in `(B,H,N,D_head)` order, Gloo, one warmup and three measured
repetitions. These are attention cores, not mixed hAttention or full ViT.

All paths receive the same synthetic Q17 Q/K/V. CrypTen standard uses Q20
computation by default (`--crypten-precision-bits 17` selects the original Q17
baseline); RSA, LSA and Mixer stay Q17. This is a mixed-precision engineering
comparison and does not establish equal-precision speedup. Standard includes
CrypTen provider work online; Ring3 preprocessing is measured separately.
Gloo stages Ring3 via CPU; reported communication and phases are framework
estimates with differing definitions. The table omits MAE and marks a cell
INVALID if any measured run fails its own-reference maximum-error gate; raw
JSON retains correctness and offline measurements.

First use a small smoke case (17 tokens means 16 patches plus CLS):

```bash
python -m hattention.run_microbenchmark --backend gloo \
  --shapes 1x1x17x16 --repeats 1 \
  --output-root "$HOME/home/tmp/hattention-micro-smoke"
```

If the smoke passes, choose shapes based on available server memory. RSA
preprocessing scales at least quadratically with token count; `1x4x577x64`
and especially `8x4x257x64` may need substantial memory and time. Start with
smaller shapes or run them separately using `--shapes`; `--case-timeout` sets
the timeout for each shape/method. After interruption, repeat the same command
with `--resume` to reuse matching completed reports. Existing finished tables
are never overwritten.

The output root contains one `comparison.json` per shape/method plus
`hattention_microbenchmark.md` and `hattention_microbenchmark.tex`. Invalid
cells must not be used for performance claims. These experiments do not prove
Ruler privacy or measure accuracy against standard attention after distillation.

### Diagnose CrypTen standard Attention

`diagnose_crypten_softmax.py` intentionally reconstructs intermediate values;
it is restricted to its internally generated public synthetic fixture and must
never be used with private data. It reports errors of QK, row maximum,
exponential, row sum, reciprocal, attention weights and AV independently.
It also opens each default Newton reciprocal iteration to inspect Q17 square
quantization. The extra openings change random consumption and make the run
unsuitable for timing or performance comparisons.

```bash
timeout 1800 python -m hattention.diagnose_crypten_softmax \
  --shape 1x4x197x256 --precision-bits 17 --repeats 1 \
  --output "$HOME/home/tmp/hattention-softmax-diag-q17.json"
```

Repeat with `--precision-bits 20` and a distinct output file to investigate
fixed-point sensitivity. Q20 is a diagnostic setting, not the original Q17
baseline; do not substitute its timings for the existing benchmark.

## Standalone sReLU dependency

The secure sReLU protocol no longer lives inside this attention package.  Its
Ruler implementation, one-time material, recent ReLU baselines and CUDA
microbenchmark are isolated in [`../srelu`](../srelu/README.md).  hAttention
will import that package when the secure RSA normalization path is connected;
the plaintext `positive_srelu` formula remains here as part of the RSA
reference definition.
