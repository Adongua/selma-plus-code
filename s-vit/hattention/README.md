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

### Fixed hAttention in a plaintext MPCViT student

`hattention.vit_integration.make_fixed_student` accepts an already constructed
`ViTLite` or `ViTLite_teacher`-style model, after loading any desired checkpoint.
It deep-copies the model, then replaces the selected `classifier.blocks[*].self_attn`
modules. The original model remains available as the teacher. QKV and output
projection weights, dtype, gradient flags and output dropout are copied. Each
layer requires its own public `HAttentionConfig`; CLS-token handling must match
the classifier's `seq_pool` setting.

```python
from hattention import HAttentionConfig, make_fixed_student

# teacher is an already constructed and checkpoint-loaded ViTLite model.
paths = ("rsa", "lsa", "mixer", "mixer")  # example for a 4-head model
configs = [HAttentionConfig(num_heads=4, head_types=paths)
           for _ in teacher.classifier.blocks]
student = make_fixed_student(teacher, configs, remove_attention_dropout=True)
# If the teacher was frozen or in evaluation mode, choose student training
# flags deliberately before constructing its optimizer:
student.requires_grad_(True)
student.train()
```

MPCViT normally uses nonzero dropout on the attention matrix; hAttention does
not apply this operation to all three paths. `remove_attention_dropout=True`
is therefore an explicit change to student training, not a weight conversion.
Projection dropout is retained. Load the original checkpoint before conversion;
the fixed student has a different attention state dict from the original teacher.
This adapter is plaintext only: it neither executes 3PC nor performs distillation,
structure search, Mixer Q/K pruning, or full-model secure benchmarking.

### Candidate pretrained Tiny-ImageNet teacher

The public [`tzhao3/vit-tiny-imagenet`](https://huggingface.co/tzhao3/vit-tiny-imagenet)
is a candidate teacher, not yet validated in this project. Its 344 MB PyTorch
weights and accompanying configs are published at revision
`20235a2bfae4169eb3c9791ad79f7336465c35a5`. Its model card does not document
training details; verify its license, architecture, labels and validation
accuracy before use. It is not a confirmed same-architecture MPCViT teacher.

On a machine with network access, install compatible `huggingface_hub`,
`transformers`, Pillow and PyTorch in a suitable environment, then download
only the necessary files into the project's external models directory:

```bash
export TEACHER_DIR="$HOME/home/models/tiny-imagenet-vit-teacher"
python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="tzhao3/vit-tiny-imagenet", revision="20235a2bfae4169eb3c9791ad79f7336465c35a5", allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"], local_dir=os.environ["TEACHER_DIR"])'
```

This server path assumes the documented `~/home/{selma-plus-code,models,datasets,tmp}`
layout; adjust it if the dataset or models directory is elsewhere. For the
official Tiny-ImageNet layout (`wnids.txt`, `val/images`, and
`val/val_annotations.txt`), first run a small local-file-only evaluation:

```bash
python -m hattention.validate_teacher \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --device cuda --max-samples 32 --batch-size 8 \
  --output "$HOME/home/tmp/teacher-validation-smoke.json"
```

For a meaningful top-1/top-5 result, rerun without `--max-samples` and use a
fresh output file. The evaluator loads only the local tensor state dict with
`weights_only=True`, records its SHA-256, and rejects missing or ambiguous
class labels rather than assuming class order. It accepts exact WordNet IDs in
the teacher's `id2label`; if the teacher uses a different label convention,
manually verify the mapping against its original training metadata before
extending the evaluator. It does not train or fine-tune the teacher.

### Mixed-head correctness and configuration costs

Each comparison JSON now includes `per_head_error_vs_own_reference` for every
head. This checks the grouped output against the matching plaintext path,
including head order; the error is not model accuracy or similarity to softmax.

Run five isolated configurations with a shared fixture: standard softmax,
four RSA heads, RSA/RSA/LSA/Mixer, RSA/LSA/Mixer/Mixer, and an interleaved
Mixer/RSA/LSA/Mixer order. The latter two have the same counts and test whether
head ordering is handled correctly. Start small to control RSA material memory:

```bash
python -m hattention.run_hybrid_microbenchmark --backend gloo \
  --shapes 1x4x17x16 --repeats 1 --case-timeout 1800 \
  --output-root "$HOME/home/tmp/hattention-hybrid-smoke"
```

If every row passes, increase repetitions or test `1x4x65x16` in a fresh
output directory. Avoid starting at 197 tokens on a shared GPU. Each case has
its own `comparison.json`; the combined Markdown table marks failed cases
`INVALID`. An interrupted run can use the same command with `--resume`.
Changing precision, seed, shape, configuration or repetitions requires a fresh
output directory. CrypTen rounds and Ring3 phases have different definitions;
the standard and mixed paths also differ in preprocessing accounting and
precision, so their times are engineering measurements, not equal-protocol
speedup claims.

### Four-shape attention-core microbenchmark

Run `python -m hattention.run_microbenchmark --output-root PATH` to compare
standard, RSA, LSA and Mixer, with each shape/method in an isolated three-party
process group. Defaults are `(1,1,17,32)`, `(1,1,65,32)`, `(1,1,101,32)` and
`(1,1,197,32)` in `(B,H,N,D_head)` order, Gloo, one warmup and three measured
repetitions. They use one batch, one head and at most 197 tokens to fit more
comfortably on shared GPU servers. These are attention cores, not mixed
hAttention or full ViT.

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

If the smoke passes, choose larger shapes based on available server memory. RSA
preprocessing scales at least quadratically with token count; increasing batch,
head count, head dimension or token count can sharply increase memory and
runtime. Check that the GPUs are not occupied by other jobs. Run larger shapes
separately using `--shapes`; `--case-timeout` sets the timeout for each
shape/method. After interruption, repeat the same command with `--resume` to
reuse matching completed reports. Existing finished tables are never
overwritten.

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
