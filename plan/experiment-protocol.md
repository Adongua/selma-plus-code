# sReLU Microbenchmark Protocol

## Scope

This protocol evaluates the standalone three-party secure ReLU/sReLU operator.
It does not treat adapted code as an official implementation of another paper.

## Inputs and seeds

- Synthetic normal inputs, generated before timing and quantized to Q17.
- Public bound: `|x| < 16`.
- Shapes: `(1,197)`, `(1,257)`, `(1,577)`, `(8,257)` for comparison with
  the existing sGeLU table; attention-score matrix shapes are a separate
  workload experiment.
- Default seed: `20261027`; one warmup plus at least three measured repeats.

## Baselines and fairness

- CrypTen-compatible full-width: explicit 64-bit A2B/sign/B2A ReLU using
  CrypTen primitives. The direct vendored CUDA `MPCTensor.relu()` path is a
  failed diagnostic (identity output), not an eligible timed baseline.
- Bicoptor2-keybits: same-backend 3PC adaptation of the key-bit precision idea,
  not the official two-round UBL/RSS implementation.
- HummingBird-eco: same-backend 3PC adaptation of reduced-width DReLU; official
  two-party timing numbers are not reused.
- HummingBird-budget: accuracy/performance ablation only.
- sReLU: manuscript Ruler path; its masked-opening security caveat is retained.

The sReLU path uses NCCL `all_reduce` for masked all-party reconstructions by
default. `p2p` is retained only as a separately labelled transport ablation.

No baseline may be delayed with sleeps, redundant communication, intentionally
slow kernels or adverse parameters.  Conversely, a baseline may not silently
receive an optimization absent from its provenance label.  Shared runtime
optimizations must be disclosed and either applied to all applicable methods or
reported as a separate optimized-adaptation ablation.

## Metrics

- Maximum three-party online wall time.
- System-sent payload bytes with receive traffic not double-counted.
- Maximum online communication rounds per party.
- MAE, RMSE, P99 and maximum absolute error against the declared method
  reference and against exact ReLU.
- Diagnostic errors against identity and absolute-value mappings, negative
  output leakage and positive-payload error.

## Correctness gates

- Exact methods: maximum error versus exact ReLU at most `1e-3`.
- Adapted approximate methods and sReLU: maximum protocol error versus the
  declared method reference at most `5e-3`.
- Non-finite output always fails.
- Failed methods remain in raw JSON for diagnosis, but their time,
communication and round values are suppressed from the Markdown comparison.

## Allowed claims

Performance claims require a PASS gate, identical hardware/software, identical
input fixture and matching online/offline definitions.  Adapted methods must be
named as adaptations.  Results determine the claim; the benchmark must not be
modified to guarantee that sReLU wins.
