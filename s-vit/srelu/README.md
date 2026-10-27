# sReLU

This package isolates the secure smooth-ReLU operator used by RSA from the
rest of `hattention`.  The attention package keeps only the plaintext formula
needed to describe RSA; all fixed-point Ruler code, one-time material and ReLU
microbenchmarks live here.

## Our operator

The manuscript convention is evaluated in two explicit stages:

```text
shifted   = (x + sqrt(x^2 + r^2) - r) / 2
corrected = shifted + r / 2
```

The public `r/2` correction is applied elementwise before RSA row summation and
normalization.  The final value is algebraically equal to
`(x + sqrt(x^2 + r^2)) / 2`.  The current profile uses `r=0.0055`, Q17
input/output, a Q16 radicand and a Q50 secret reciprocal.  It requires
`|x| < 16`.

## Compared methods

- `crypten_native` (CrypTen, 2021): the strict unmodified
  `MPCTensor.relu()` call. If the current CrypTen/CUDA combination returns an
  incorrect result, the benchmark records the raw result but suppresses its
  timing claim.
- `crypten_fullwidth_adapter` (engineering diagnostic): a separate
  correctness-repaired 64-bit A2B/B2A implementation built from CrypTen
  primitives. It is not the official `MPCTensor.relu()` implementation and
  is excluded from the strict native comparison.
- `bicoptor2_keybits` (Bicoptor 2.0, 2023): a same-backend structural
  adaptation of its key-bit precision policy.  It is not the paper's official
  UBL/RSS two-round protocol and is not an independent security claim.
- `hummingbird_eco` (HummingBird, 2024): a 3PC adaptation that removes only
  public-bound-proven redundant high bits while retaining all Q17 fractional
  bits.
- `hummingbird_budget` (HummingBird-style ablation, 2024): also reduces the
  fractional-bit budget.  This is intended for the accuracy/performance
  ablation, not the main exact-under-bound comparison.
- `srelu` (ours, 2026): the standalone Ruler-based smooth ReLU.

The reduced-ring methods use the same CrypTen NCCL process group and convert
the sign bit back to an arithmetic share before selecting the original Q17
payload.  Their prefix-adder depth is `ceil(log2(k))` for a public `k`-bit
ring, instead of always converting a full 64-bit word.

The sReLU benchmark defaults to `--reveal-mode all_reduce`: masked
reconstructions are revealed to all three parties and a collective completes
that reconstruction in one NCCL phase. `p2p` remains available only as a
transport ablation and is not the default performance setting.

Mosformer is not included in this same-backend main table: its maliciously
secure 3PC model and incomplete public offline implementation would make a
direct semi-honest online comparison misleading.  It can be evaluated later
as a separately labelled security-model experiment.

## Tests

From the repository root:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main"
python -m unittest discover -s s-vit/srelu/tests -v
python -m unittest discover -s s-vit/hattention/tests -v
```

## Three-GPU smoke test

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main"
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
timeout 600 \
python -m srelu.benchmark_crypten_cuda \
  --shapes 1x17 \
  --repeats 1 \
  --methods crypten_native bicoptor2_keybits hummingbird_eco hummingbird_budget srelu \
  --output "$HOME/home/tmp/srelu-cuda-smoke"
```

## Formal operator benchmark

Run the main four methods on the same shapes as the sGeLU microbenchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
timeout 7200 \
python -m srelu.benchmark_crypten_cuda \
  --shapes 1x197 1x257 1x577 8x257 \
  --repeats 3 \
  --methods crypten_native bicoptor2_keybits hummingbird_eco srelu \
  --output "$HOME/home/tmp/srelu-cuda-main"
```

The command writes `comparison.json` and `summary.md`.  The table excludes the
warmup and reports online wall time, normalized system-sent payload bytes,
rounds, MAE and maximum error against exact ReLU.  CrypTen provider work is
included in the CrypTen method's online interval; Ruler-sReLU preprocessing is
reported separately as offline work.

The benchmark is correctness-gated. Exact methods must match ReLU within
`1e-3`; adapted methods and sReLU must match their declared method reference
within `5e-3`. A failed method remains in `comparison.json` for diagnosis, but
its performance cells are replaced by `INVALID` in `summary.md`, and the
command exits unsuccessfully. The JSON also records error versus identity and
absolute-value mappings, negative-output leakage and positive-payload error to
detect fast but semantically incorrect comparison paths.

The benchmark also records `crypten_import` and refuses to run if Python
resolves `crypten` outside this repository's `CrypTen-main/crypten` directory.

To isolate a native-ReLU failure, run the diagnostic separately on CPU/Gloo
and CUDA/NCCL:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main"

python -m srelu.diagnose_crypten_relu \
  --device cpu \
  --shape 1x17 \
  --output "$HOME/home/tmp/crypten-relu-diagnostic-cpu.json"

CUDA_VISIBLE_DEVICES=0,1,2 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
timeout 600 \
python -m srelu.diagnose_crypten_relu \
  --device cuda \
  --shape 1x17 \
  --output "$HOME/home/tmp/crypten-relu-diagnostic-cuda.json"
```

The diagnostic invokes only CrypTen's native `_ltz()`, `ge(0)` and `relu()`
methods and reports each result separately.

The fairness and claim policy is recorded in
[`../../plan/experiment-protocol.md`](../../plan/experiment-protocol.md).

The code is experimental.  Ruler masked openings are not privacy validated,
the adapted baselines are not official implementations of their cited papers,
and CUDA Philox is a benchmark PRG rather than a reviewed CSPRNG.
