# hAttention

## Distill exported layer-search students

`python -m hattention.distill_fixed` accepts the original fixed export,
`selection.json`, and the final layer-search checkpoint. It reconstructs the
export and verifies every fixed tensor and path against the search checkpoint;
it also verifies the MPCViT checkpoint, teacher files, training split, and
search holdout. The original 33 RSA locations and all LSA/Mixer choices stay
fixed. Train loss is CE + `lambda_kd * KD` + `lambda_feature * feature MSE`;
the inherited feature projection is trainable. This is plaintext recovery,
not secure inference. First-name teacher label mapping is still a hypothesis.

Smoke example for lambda=0.4 (replace `0.4` with `0.5` for the other case;
use a new output directory for every invocation):

```bash
case=0.4
run="$HOME/home/tmp/hattention-layer-gate-lasso-${case}-v1"
CUDA_VISIBLE_DEVICES=0 python -m hattention.distill_fixed \
  --fixed-student "$run/fixed-student.pt" --search-checkpoint "$run/epoch-0005.pt" \
  --selection "$run/selection.json" \
  --checkpoint "$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar" \
  --mpcvit-root "$PWD/mpcvit" --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --output-root "$HOME/home/tmp/hattention-distill-${case}-smoke-v1" \
  --epochs 1 --batch-size 1 --max-steps-per-epoch 2 --eval-samples 16 \
  --log-every 1 --device cuda
```

For a preliminary comparison, use identical training settings and seeds for
both cases, e.g. `--epochs 5 --batch-size 8 --max-steps-per-epoch 200
--eval-samples 1000 --log-every 50` with new output directories. A zero step
limit uses the whole weight-training split per epoch; 200 steps is a partial
pass, not a full dataset epoch. Reports include losses and search holdout
metrics before training and after every epoch. `epoch-NNNN.pt` is also a
loadable fixed-student artifact. `--resume path/to/epoch-NNNN.pt` resumes
optimizer and RNG state with the same configuration, data and inputs; set
`--epochs` to the desired total and use a new output root. The original
`--fixed-student` and search files remain required for provenance checks.

After deciding how to use the validation set, evaluate a fixed checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m hattention.evaluate_fixed \
  --fixed-student "$HOME/home/tmp/hattention-distill-0.4-trial-v1/epoch-0005.pt" \
  --checkpoint "$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar" \
  --mpcvit-root "$PWD/mpcvit" --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --output "$HOME/home/tmp/hattention-distill-0.4-official-val-v1.json" \
  --batch-size 8 --device cuda
```

By default this evaluates all official validation images; `--max-samples N`
uses a seeded class-balanced subset. Repeated model selection on this set turns
it into a tuning set. Reports do not imply privacy or secure numeric validity.

## Layer-level LSA/Mixer search with fixed RSA

`train_search --search-space fixed_rsa_layer_lsa_mixer` starts from a binary
MPCViT RSA/LSA checkpoint and holds RSA positions fixed. Each layer has one
trainable direct gate `alpha`: all of its non-RSA heads use
`alpha * LSA + (1-alpha) * Mixer` in the plaintext soft forward. Weights and
gates use the same CE/KD/feature, numeric, budget and L1 objective, and
independent optimizers. L1 is a **sum per searchable layer**, not per head;
do not equate its coefficient with the head-level experiment.

For each layer, the cost surrogate is `alpha*C(rsa,0) + (1-alpha)*C(rsa,k)`
where `k` is the number of original LSA heads. This is the expected measured
endpoint cost under a hypothetical binary layer gate; it is not the runtime of
the soft forward and does not assume individual heads choose independently.
Export uses only the two whole-layer options, retains fixed RSA, enforces the
estimated budget and existing CLS connectivity constraint, and maximizes the
learned gate score among feasible structures. A tight budget may force a layer
against its preferred gate and does not guarantee accuracy. `selection.json`
reports `layer_lsa_alpha`, `layer_choices`, paths, costs and fixed-model
holdout evaluation. `baseline.json` records the original fixed baseline and a
table-derived lower bound. Fixed-structure recovery and final validation are separate steps; see
`distill_fixed` and `evaluate_fixed` above. `ranked_mixer` accepts head-level
checkpoints only.

Use a new output root and the same measured cost table as the intended
deployment. Example pipeline smoke test (the wide budget is not an optimization
target):

```bash
CUDA_VISIBLE_DEVICES=0 python -m hattention.train_search \
  --checkpoint "$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar" \
  --mpcvit-root "$PWD/mpcvit" --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --cost-table "$HOME/home/tmp/hattention-cost-12h-remeasure-v1.json" \
  --output-root "$HOME/home/tmp/hattention-layer-gate-smoke-v1" \
  --search-space fixed_rsa_layer_lsa_mixer --architecture-warmup-epochs 1 \
  --epochs 2 --max-steps-per-epoch 2 --eval-samples 16 --batch-size 1 \
  --budget-online-seconds 100 --device cuda --mapping first-name --log-every 1
```

This is a plaintext search with cost estimates, not secure inference or a
guarantee of accuracy. Search holdout images are drawn from training data and
may have been seen by the original checkpoint. The first-name teacher mapping,
RSA secret range and Ruler privacy remain unvalidated.

## Remeasure selected Mixer candidate costs

`hattention.benchmark_candidate_costs` reads the ranked Mixer report, extracts
each selected candidate's actual layer path counts, and deduplicates identical
`(RSA, Mixer)` combinations. Each round measures each combination independently
with one warmup and one measured repetition; successive rounds reverse the
order. The summary preserves per-round source paths and hashes and sums the
same measured group cost for layers with equal counts. This is an estimate of
attention-core cost, not a timed ViT forward or a paired confidence interval.
Head ordering is canonicalized, so verify its timing impact before a final claim.
The original candidate accuracy and model checkpoints are not changed.

```bash
# Set this to the actual ranked_mixer report containing 0, 4 and 24 candidates.
export CANDIDATE_REPORT="$HOME/home/tmp/hattention-ranked-main-v1/report.json"
CUDA_VISIBLE_DEVICES=0,1,2 python -m hattention.benchmark_candidate_costs \
  --candidate-report "$CANDIDATE_REPORT" --mixer-counts 0 4 24 \
  --backend gloo --shape 1x12x257x16 --rounds 3 \
  --output-root "$HOME/home/tmp/hattention-candidate-costs-v1" --dry-run
```

Remove `--dry-run` to measure. This particular report uses 15 combinations,
so 3 rounds require 45 separate three-party runs, each with its own offline
preprocessing; reserve suitable GPUs and sufficient time. `summary.json` is
written only on success. To continue an interrupted run, rerun the exact same
command with `--resume`; existing cases are validated before reuse. A failed or
incompatible case is not silently overwritten. Timing is online-only and the
Gloo CPU staging is included; privacy and RSA ranges remain unvalidated.

## Diagnose hybrid path timings

`benchmark_compare_cuda --profile-hybrid` first measures the usual uninstrumented
online execution. It then prepares fresh one-use material and runs a separate
diagnostic execution, synchronizing the GPU around each group's select, compute
and scatter stages. Per-rank wall times, sent bytes and exchange phases are in
each record's `methods.hybrid.diagnostic_profile.stages`; the instrumented run
also has its own correctness check. Additional preprocessing makes this mode
considerably slower and more memory intensive. Stage timings include any waits
for peers and contention and are not exclusive GPU kernel times. Do not add
individual rank maxima: different ranks can be the slowest at different stages.
Use the ordinary `online` field for the uninstrumented measurement. Run both
configurations under comparable GPU load; diagnostic values are not a new cost
table or a security validation.

Example: run each command with a fresh output path and the same environment.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m hattention.benchmark_compare_cuda \
  --backend gloo --shape 1x12x257x16 --methods hybrid \
  --head-types rsa rsa rsa rsa lsa lsa lsa lsa lsa lsa mixer mixer \
  --patch-grid 16x16 --repeats 3 --profile-hybrid \
  --output "$HOME/home/tmp/hattention-profile-rsa4-mixer2-v1"

CUDA_VISIBLE_DEVICES=0,1,2 python -m hattention.benchmark_compare_cuda \
  --backend gloo --shape 1x12x257x16 --methods hybrid \
  --head-types rsa rsa rsa rsa lsa lsa lsa lsa mixer mixer mixer mixer \
  --patch-grid 16x16 --repeats 3 --profile-hybrid \
  --output "$HOME/home/tmp/hattention-profile-rsa4-mixer4-v1"
```

## Rank fixed LSA/Mixer candidates and distill independently

`python -m hattention.ranked_mixer` loads a **joint-alpha fixed-RSA** search
checkpoint, fixes the original RSA locations, and sorts searchable heads by
LSA alpha (ascending; layer and head index break ties). It exports independently
initialized candidates for the requested Mixer counts and records plaintext
accuracy and measured-table attention-core cost. Every candidate starts from
the same search weights; its distillation uses the same data split, steps,
optimizer settings, seed and initial feature-projection weights. It saves
before/after checkpoints and `report.json`; no candidate is selected
automatically. A count is a requested choice, not proof that the corresponding
head prefers Mixer. The default search holdout was already monitored during
search and the original MPCViT model may have seen these training images.
Use `--eval-split official_val` only for a separately planned validation;
repeated selection on that split makes it a tuning set.

Smoke example, using a **new output directory** and an existing search checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m hattention.ranked_mixer \
  --search-checkpoint "$HOME/home/tmp/hattention-alpha-03-lasso001-v1/epoch-0005.pt" \
  --checkpoint "$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar" \
  --mpcvit-root "$PWD/mpcvit" --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --cost-table "$HOME/home/tmp/hattention-cost-12h-full-v2.json" \
  --output-root "$HOME/home/tmp/hattention-ranked-smoke-v1" \
  --mixer-counts 0 4 --recovery-steps 2 --eval-samples 16 \
  --batch-size 1 --device cuda
```

For an actual comparison, use `--mixer-counts 0 4 8 12`, choose the same
recovery steps and data for all candidates, and increase evaluation samples.
Report estimates only attention-core online cost at the table's shape; it is
not a measured full-model runtime. Checkpoint, teacher, dataset split and cost
table hashes are checked against the search run. This command never overwrites
an existing output directory; interrupted runs leave their completed artifacts
in place and must be restarted in a new directory. Ruler privacy and secret RSA
range remain unvalidated. Stored fixed checkpoints use the existing
`hattention_fixed_student_v1` loader.

## Differentiable second-stage LSA/Mixer search

`hattention.train_search --search-space fixed_rsa_lsa_mixer` initializes from
the binary RSA/LSA MPCViT checkpoint. The original RSA head positions remain
RSA throughout search and export. Every original LSA head has a direct,
trainable alpha in [0,1] and computes `alpha * LSA + (1-alpha) * Mixer`.
Alpha starts at 0.95 by default. QKV, projections and other student weights
remain trainable. After optional `--architecture-warmup-epochs`, the model
weights and alpha are updated from the same batch and the same combined loss,
using separate AdamW optimizers and learning rates. The loss is teacher
guidance (CE, logit KD and feature loss), plus `--lambda-lasso * sum(abs(alpha))`
over searchable LSA heads, a measured-budget penalty, and the RSA numerical
penalty. The default L1 coefficient is 0.001; larger values encourage Mixer but
do not guarantee accuracy. `--lambda-task`, `--task-loss-scale`,
`--lambda-cost` and `--lambda-efficiency` also affect this combined loss.

This follows MPCViT's joint task-plus-L1 search pattern, adapted to a fixed
RSA set, a Mixer path and measured grouped costs. Alpha is projected into [0,1]
after each update. The exported structure uses the measured-cost dynamic
program, not MPCViT's global top-k rule. Checkpoints from the old alternating
fixed-RSA search cannot resume this changed optimization; use a new output
directory. Export does not perform a separate fixed-structure distillation.

The measured grouped-cost table provides the optimization objective and a
hard cost budget for export. The CLI checks the minimum table cost under the
fixed RSA assignments before training. The search objective has no hard
accuracy floor and the resulting argmax or budget-constrained fixed model can
lose accuracy; compare `baseline.json`, mixed metrics, `selection.json`, and
evaluate final candidates on the official validation set. Setting an
aggressive budget can force low-probability Mixer choices. `fixed_rsa_observations`
reports RSA heads whose observed denominators violate the numerical profile:
the fixed constraint retains these heads, so this warning must be investigated
before secure inference. No RSA-range or privacy guarantee is implied.

For a smoke run, use a new output directory, two epochs and one gate warmup
epoch. The large budget below only tests the training/export pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 python -m hattention.train_search \
  --checkpoint "$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.5-kd.pth.tar" \
  --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --cost-table "$HOME/home/tmp/hattention-cost-12h-full-v2.json" \
  --output-root "$HOME/home/tmp/hattention-fixed-rsa-smoke-v1" \
  --search-space fixed_rsa_lsa_mixer --architecture-warmup-epochs 1 \
  --epochs 2 --max-steps-per-epoch 2 --eval-samples 16 --batch-size 1 \
  --budget-online-seconds 100 --device cuda --mapping first-name --log-every 1
```

The search is plaintext. Cost table values describe three-party attention
cores, not measured end-to-end ViT runtime. The teacher's first-name mapping is
a hypothesis. Search holdout examples are from the training split and might
have been seen by the original MPCViT checkpoint.

This directory contains the isolated implementation of the heterogeneous secure
attention method.  The original `mpcvit` sources remain unchanged.

## Incremental Mixer search from an MPCViT checkpoint

`python -m hattention.search_mixer` begins with the checkpoint's RSA/LSA
head choices, converting RSA to the project's corrected sReLU. It measures
the new fixed baseline, then screens single-head Mixer replacements. Per stage,
it evaluates the best `--top-k` screened candidates, optionally recovers each
with CE or teacher distillation, and accepts one only if its selection Top-1
is at least both `baseline_top1 - --max-top1-drop` and `--min-top1`. It also
requires total estimated grouped attention-core cost to remain no greater than
the original baseline plus `--max-cost-increase`, and at least
`--minimum-non-mixer` heads to remain. The absolute-drop argument is in
fractional units: `0.05` means five percentage points, not five percent of the
baseline. The `--target-mixers` flag reports whether a goal was reached; it
never overrides the accuracy and cost requirements.

The sensitivity/selection set and holdout are disjoint for this search and
recovery uses a separate weight split. However, the original MPCViT checkpoint
may itself have seen those training images. These are not official validation
results; holdout is shown at completion and must not be presented as a fresh
test set. Screening, top-k truncation and greedy one-head replacement do not
guarantee the globally best structure or a target number of Mixer heads.
Estimated costs sum the measured same-shape group table, exclude other ViT
operators, and do not constitute a secure end-to-end benchmark. RSA range and
Ruler privacy remain unvalidated. Teacher first-name mapping is an unverified
hypothesis for 18 mismatched WordNet IDs.

Start with a small smoke run; its tiny sample and relaxed constraints are only
for verifying plumbing:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python -m hattention.search_mixer \
  --checkpoint "$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.5-kd.pth.tar" \
  --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --cost-table "$HOME/home/tmp/hattention-cost-12h-full-v2.json" \
  --output-root "$HOME/home/tmp/hattention-mixer-smoke-v1" \
  --device cuda --max-stages 1 --candidate-limit 2 --top-k 1 \
  --sensitivity-samples 8 --selection-samples 16 --holdout-samples 16 \
  --batch-size 4 --recovery-steps 1 --max-top1-drop 1 --max-cost-increase 100
```

For a real experiment choose a new output directory and explicitly set an
acceptable accuracy drop and optional absolute floor, for example
`--max-top1-drop 0.05 --min-top1 0.80 --max-stages 3 --target-mixers 3`.
`--budget-online-seconds 9` optionally records whether the final estimated
attention-core cost meets a 9-second goal; intermediate stages may exceed it.
By default all eligible heads are screened, which can be expensive. Use
`--recovery-loss distill --teacher-dir ...` to opt into CE+KD+feature recovery;
default recovery uses CE only. Every stage is saved with zero-based layer/head
indices. `report.json` explains any early stop and `final.pt` holds the fixed
model. A completed run that stopped at the stage limit may be extended with
the same arguments, `--resume`, and a larger total `--max-stages`; it preserves
the original baseline and constraints. Incomplete interrupted runs are not
resumable automatically; completed per-stage artifacts remain available.
Use `load_mixer_search_model` in `hattention.search_mixer` to load the exported
artifact with its original MPCViT checkpoint.

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
layout; adjust it if the dataset or models directory is elsewhere. Validation
images can be in the official `val/images` directory or reorganized into
`val/<wnid>/` class folders. Both require `val/val_annotations.txt`; filenames
must match, and class folder names are checked against annotations. First run
a small local-file-only evaluation:

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
class labels rather than assuming class order. The default requires exact
WordNet IDs in the teacher's `id2label`. It does not train or fine-tune the
teacher.

The downloaded `tzhao3` configuration uses 18 WordNet IDs different from the
dataset IDs, while all 200 first names in `words.txt` match uniquely. A small
training-set diagnostic supports the teacher's name-based index ordering but
does not establish synset equivalence or validate individual classes. To test
this explicit hypothesis on the held-out validation set, add
`--mapping first-name` to the command above. The default `strict-id` mode
still rejects mismatched IDs. The JSON report records all ID/index mappings,
the source file hashes, per-class support and top-1/top-5 counts, and separate
metrics for matching and differing IDs. Missing, duplicated or ambiguous names
are rejected; identical first names do not prove identical WordNet senses.
Smoke subsets can omit classes; inspect per-class support and run the full
validation set before interpreting group metrics. Treat this result as an
evaluation under an explicit mapping hypothesis, not verified ground truth.

### Differentiable head search: building blocks

`hattention.search` supplies a plaintext mixed-path attention layer, a
ViTLite conversion helper, differentiable grouped-count costs, numerical and
budget penalties, a first-order alternating-update helper, greedy budget
repair and fixed-path model export. `HEAD_TYPES` order is RSA, LSA, Mixer.
All three paths execute during search; these times are not deployment times.
The initial structure logits are uniform, not the manuscript's MPCViT prior.
The numerical guard uses observed batches only and does not prove secure
fixed-point range or privacy. Export currently retains packed QKV projections;
Mixer Q/K physical pruning remains future work.

`hattention.calibrate_group_costs` runs the existing three-party hybrid
benchmark for explicit `(number of RSA heads, number of Mixer heads)` pairs.
For a small server smoke test, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m hattention.calibrate_group_costs \
  --backend gloo --shape 1x2x17x8 --repeats 1 \
  --pairs 0,0 0,2 --case-timeout 600 \
  --output-root "$HOME/home/tmp/hattention-cost-smoke-cases" \
  --table-output "$HOME/home/tmp/hattention-cost-smoke.json"
```

The table contains only correctness-passed measured cells, input/environment
metadata and hashes of the source reports. Unmeasured cells are listed and
never inferred. `load_complete_tables` refuses an incomplete table. Pass
`--all-counts` instead of `--pairs` to measure all `(H+1)(H+2)/2` cells:
12 heads require 91 separate cases, and large RSA cases can exhaust GPU memory.
Use `--resume` to reuse matching completed cases after an interruption, with a
new table-output filename. Per-count costs assume head order does not affect
cost; rerun selected permutations and verify this assumption on the target
system. The tables measure attention core only, not projections or full ViT.

### Real checkpoint and one-step search smoke

`student_checkpoint` loads the active weights from an existing `vit_9_12_64`
checkpoint using restricted deserialization. The older checkpoint contains
extra experimental layers absent from the current source; known extra keys are
listed in the report, while unknown extras, missing active keys, or shape
mismatches are rejected. The original MPCViT source imports its inference CLI
just to obtain the `linear_gelu` flag; an isolated import compatibility shim
sets it explicitly to false without modifying that source. The student's binary
alpha values initialize RSA/LSA preferences and Mixer remains available.

`search_data` uses a seeded per-class shuffle to split the official 500 train
images into 450 weight-update and 50 structure-update images, preserving the
validation split. Split manifest hashes are saved. The current smoke uses a
64x64 bicubic resize and MPCViT Tiny-ImageNet mean/std without augmentation;
it does not reproduce the original training augmentation recipe.

After installing the existing project dependencies, run from the repository root:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export MPCVIT_CHECKPOINT="$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.5-kd.pth.tar"

CUDA_VISIBLE_DEVICES=0 python -m hattention.smoke_student \
  --checkpoint "$MPCVIT_CHECKPOINT" --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --device cuda --batch-size 1 --output "$HOME/home/tmp/hattention-student-smoke.json"

CUDA_VISIBLE_DEVICES=0 python -m hattention.smoke_search_step \
  --checkpoint "$MPCVIT_CHECKPOINT" --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --mapping first-name --device cuda --batch-size 1 --smoke-synthetic-costs \
  --output "$HOME/home/tmp/hattention-search-step-smoke.json"
```

The first command only checks forwards; the second performs one weight step
and one structure step with CE, teacher-logit KL and CLS-feature MSE. The
student's 192-dimensional CLS feature is projected to the teacher's feature
dimension; the teacher uses its own processor and frozen weights. Synthetic
costs are explicitly restricted to smoke diagnostics and are not evidence of
MPC performance. Alternatively `--cost-table` requires a *complete measured*
table of matching `(B,12,257,16)` shape. First-name mapping remains an explicit
unverified label hypothesis. These smoke commands do not export a structure.

### Multi-epoch architecture search

#### Fixed-RSA learning and CLS connectivity checks

For this CLS-token classifier, an all-Mixer network cannot transfer image
patch information to the classifier. Discretization now requires at least
one RSA or LSA head across the entire model; Pareto pruning keeps connected
and disconnected states separate. Export rejects all-Mixer CLS classifiers,
and hard-path training stops if its argmax collapses to all Mixer. This is a
necessary connectivity check, not a guarantee of useful accuracy. Individual
all-Mixer layers remain allowed, as do standalone Mixer microbenchmarks.

Before architecture search, check that a randomly initialized student can
learn using fixed all-RSA heads. This short diagnostic trains and evaluates
on the same 20 training images; its accuracy is an overfit check, not
generalization. It does not load a teacher:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python -m hattention.train_rsa_warmup \
  --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --output-root "$HOME/home/tmp/hattention-rsa-overfit-v1" \
  --device cuda --loss ce --epochs 40 --batch-size 4 \
  --overfit-classes 5 --overfit-per-class 4 --log-every 5
```

Examine the CE loss and training-set accuracy by epoch. If they do not improve,
diagnose learning before launching a large run. Diagnostic checkpoints cannot
initialize architecture search. To train on the full weight split, omit both
overfit options and choose a new output directory. With no `--max-steps-per-epoch`,
each epoch uses the entire 90,000-image weight split; this can take substantial
time. The evaluation is a disjoint subset of training images, not the official
validation set. `--loss distill --teacher-dir ...` additionally logs `kd` and
`feature`, besides `ce` and `total`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m hattention.train_rsa_warmup \
  --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --output-root "$HOME/home/tmp/hattention-rsa-warmup-v1" \
  --device cuda --loss ce --epochs 10 --batch-size 8 \
  --max-steps-per-epoch 1000 --eval-samples 1000 --log-every 100
```

Every epoch writes a checkpoint. An interrupted run can use the same arguments
plus `--resume .../epoch-XXXX.pt` (the `--epochs` value must remain unchanged).
After inspecting warmup accuracy, its normal checkpoint can initialize scratch
search by adding `--warmup-checkpoint .../epoch-XXXX.pt` to `train_search`. The
dataset split and class order must match; KD warmup also requires the same
teacher. This initializes only student weights (and the feature projection if
trained); architecture logits and RSA observations start fresh. The existing
search remains experimental: it uses hard straight-through paths and a soft
expected-cost surrogate, and can still collapse or fail the accuracy check.

`hattention.train_search` alternates weight and architecture updates using the
real teacher and a complete measured deployment cost table. The original
450/50 per-class split is refined into 450 weight, 40 architecture-update and
10 search-holdout images; official validation is kept for final evaluation.
Search holdout top-1/top-5 metrics are diagnostic, not official test accuracy.
The minibatch size can differ from the measured deployment batch of one.

#### Scratch, all-RSA starting point and budget-constrained search

`--student-init scratch` creates a new `vit_9_12_64` student without reading a
student checkpoint. The teacher is still loaded for CE/KD/feature guidance.
Unless overridden, scratch mode starts with `--architecture-init rsa` and
`--path-mode straight_through`: each head's forward path initially is exactly
RSA, while the softmax probabilities supply surrogate architecture gradients.
All three candidate outputs are still computed in this plaintext search; this
does not time secure execution. Unlike MPCViT's two-way continuous alpha search,
this uses three-way logits and a straight-through estimator. Initial *soft*
probabilities are not one-hot: with margin 2 and temperature 5 the initial RSA
probability is about 0.427, despite all 108 forward paths selecting RSA.

Scratch search treats latency as a budget constraint rather than rewarding
lower latency without limit. Its architecture objective defaults to
`task / 50 + 10 * max(expected_time / budget - 1, 0)^2`, plus numerical, RSA
count, and entropy terms. `--task-loss-scale` (default 50 for scratch),
`--lambda-task`, and `--lambda-cost` control these terms. There is no positive
efficiency reward below the budget (`--lambda-efficiency` defaults to 0).
After training, the Pareto selector chooses the highest learned path
probability among structures that satisfy the measured hard budget; it has no
extra cost preference by default (`--selection-cost-preference 0`). The
intended priority is to meet the latency budget, then preserve learned task
utility—not to choose the absolute cheapest structure. This is still a
penalty-based differentiable search, not a formal constrained-optimization
guarantee; check selected-model quality and measured runtime. The table is
measured for a single deployment batch and contains experimental noise.

The earlier scratch objective used a small task coefficient, a continuous
efficiency reward, and a cost-biased final selector. It could collapse to an
all-Mixer structure with near-zero latency and unusable accuracy. Runs created
with that objective are not valid search results; start a fresh output
directory after updating the code rather than resuming their checkpoints.

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python -m hattention.train_search \
  --student-init scratch --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --cost-table "$HOME/home/tmp/hattention-cost-12h-full-v2.json" \
  --output-root "$HOME/home/tmp/hattention-scratch-smoke-v1" \
  --device cuda --mapping first-name --epochs 1 --batch-size 1 \
  --max-steps-per-epoch 2 --eval-samples 16 --log-every 1 \
  --budget-online-seconds 9
```

This command is an integration smoke, not a trained result. `run.json` records
initial path counts; epoch reports include task, numerical, budget and
efficiency components, soft expected cost, and `argmax_structure_online_seconds`
(lookup estimate for the hard paths). The final selection can differ from the
argmax paths because of its budget and RSA denominator filter. To recalibrate
scratch runs use `calibrate_search_denominators` with the same arguments as the
checkpoint mode but omit `--checkpoint`; its fixed export can be reloaded as
`load_fixed_student(artifact, None, mpcvit_root)`. Historical minimum denominators
may be dominated by early random weights, so inspect final-model calibration.

`--architecture-init mpcvit` (default) starts with the binary MPCViT head
selection as a logit prior. `--architecture-init uniform` keeps the same
pretrained model weights but starts every head with equal RSA/LSA/Mixer
probabilities. Compare runs using the same checkpoint, seed, data split,
teacher, cost table, budget and training schedule; only change this option and
the output directory. Both modes share pretrained projection weights, so equal
probabilities do not establish equal path quality or performance. Repeat with
multiple seeds before drawing conclusions.

Each new run records initial path probabilities in `run.json` and per-epoch
mean probabilities and unique preferred path counts in `epoch-NNNN.json`.
Tied heads are counted separately instead of being assigned to the first path.
Architecture initialization is checked on resume. Legacy checkpoints without
this field are interpreted as `mpcvit` initialization.

Run a short integration check first, using a **new output directory**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m hattention.train_search \
  --checkpoint "$MPCVIT_CHECKPOINT" --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$HOME/home/datasets/tiny-imagenet-200" \
  --teacher-dir "$HOME/home/models/tiny-imagenet-vit-teacher" \
  --cost-table "$HOME/home/tmp/hattention-cost-12h-full-v2.json" \
  --output-root "$HOME/home/tmp/hattention-search-run-smoke" \
  --device cuda --mapping first-name --epochs 1 --batch-size 1 \
  --max-steps-per-epoch 2 --eval-samples 16 --log-every 1 \
  --budget-online-seconds 9
```

`9 s` is an example budget for the **sum of the nine attention cores**, not
an experimentally justified whole-ViT target. Cost is a soft training penalty;
the final global Pareto search chooses the highest-probability feasible count
configuration in the measured tables and reports an error if none fits (for
scratch mode the optional cost preference described above also applies). This
does not establish an optimum for the continuous or true secure system. `--max-rsa` limits the expected total across
all heads/layers, default 108 (no effective cap for this 9x12 model). The
temperature schedule is 5.0 to 0.1. `--max-steps-per-epoch 0` processes the
full weight split and can take a long time; choose epochs, budget and training
length explicitly for an actual search. Evaluation samples are interleaved by
class, but small subsets are only smoke checks.

Each completed epoch saves `epoch-NNNN.pt` and its metrics JSON. After
interruption, repeat the identical arguments and add `--resume` pointing to
the latest completed epoch checkpoint in the same output directory. A partial
epoch is rerun; the checkpoint verifies configuration, split manifests, cost
table and initial model/teacher file hashes. The output includes
`selection.json` and `fixed-student.pt` only after successful discretization.
The latter can be reloaded for later distillation with
`hattention.student_checkpoint.load_fixed_student(artifact, original_checkpoint, mpcvit_root)`.
Only trusted local checkpoints should be loaded.

The fixed export keeps QKV projections, and the model is not yet distilled.

### Final-search RSA denominator diagnostic

`python -m hattention.calibrate_search_denominators` reloads a completed search
checkpoint and measures final mixed-model plaintext RSA row denominators on a
class-interleaved held-out training subset. It preserves the historical training
minima, runs the same strict minimum threshold and budget selection with both
historical and newly measured minima, then compares both fixed exports and the
mixed model on a disjoint subset. The report records per-head minima, row
quantiles and below-threshold fractions, path assignments, and costs.

Use `--search-checkpoint`, `--checkpoint` (original MPCViT), `--mpcvit-root`,
`--dataset-root`, `--cost-table`, and a new `--output-root`. Calibration and
evaluation default to 1000 samples each; reduce both for a smoke test. The
checkpoint's split, source and cost-table hashes must match. The held-out split
was already monitored during search; this diagnostic uses part of it for
selection, and neither metric replaces a separate final test. Mixed-model
denominators do not prove the fixed model satisfies the Q17 3PC range contract.
Check fixed-model inputs and run secure numerical tests before using a result.
Costs assume grouping depends on head counts and sum across layers; the table
uses one measured repetition and differs in preprocessing accounting from some
baselines. Search uses approximate plaintext paths and observed denominators;
neither search accuracy nor measured costs prove secure inference correctness
or privacy. Mixer Q/K pruning and final validation remain future work.

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

## Experimental full-ViT execution and range audit

`hattention.benchmark_vit_cuda` contains a forward-only ViTLite prototype:
public model weights, a public benchmark image secret-shared for execution,
CrypTen public linear operations, Ring3 hAttention, sLayerNorm and sGeLU.
The source MPCViT checkpoint is converted to sReLU RSA for this experiment;
this adapted baseline is not the original MPCViT ReLU implementation. The
reference is the checkpoint's plaintext LayerNorm/GeLU model. Just-in-time
Ring3 material generation is counted separately and full forward wall time is
also recorded. The model is experimental and has not been run on three GPUs.

Before attempting the secure prototype, audit the real checkpoints and public
images on CPU. For example, set `IMAGE` to an existing Tiny-ImageNet image:

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
CHECKPOINT="$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
IMAGE="$HOME/home/datasets/tiny-imagenet-200/val/n01784675/example.JPEG"

python -m hattention.benchmark_vit_cuda \
  --checkpoint "$CHECKPOINT" --mpcvit-root "$PWD/mpcvit" \
  --fixed-student "$HOME/home/tmp/hattention-distill-0.4-resume200-v1/epoch-0189.pt" \
  --image "$IMAGE" --audit-only \
  --output "$HOME/home/tmp/hattention-vit-audit-04"
```

Replace the example image path with an actual file. Omit `--fixed-student` to
audit the source MPCViT checkpoint. Multiple image paths may follow `--image`
in audit mode. The generated `range-audit.json` reports per-layer plaintext
observations and violations, but passing a finite set of public examples is
not proof of secret ranges. The secure run refuses an observed violation by
default. `--allow-out-of-range` is only for debugging and marks the output
invalid; it does not repair the numerical contract. The local original
checkpoint and one real image already exhibit LayerNorm and RSA range violations,
so this prototype is not ready for formal full-model performance claims.

### Experimental wider numerical profile

The old RSA and LayerNorm protocols remain unchanged, and `legacy` remains the
default. `--numeric-profile wide-range` explicitly selects separate experimental
implementations for the full-ViT prototype:

- sLayerNorm: Q17 input, fixed public scale 16, Q21 centered values and Q28
  scaled variance; profile requires |input|<=64 and |gamma|,|beta|<=4.
  At epsilon=1e-5 the quantized effective epsilon is approximately 9.5367e-6.
- RSA: Q17 input with |score|<16, Q24 Newton reciprocal with 32 iterations,
  and a declared denominator interval [0.0005, 2048]. The sReLU primitive is
  unchanged, so its finite precision can still matter for very small rows.
- sGeLU: fixed |input|<=12 profile reuses the unchanged G2 protocol and
  validates the public radicand bound for mu in [0.25,4].
- LSA Q/K/V input bound is 16; the wide grouped path requires N<=257, D<=16.

For a plaintext check of the new declared range, add `--numeric-profile
wide-range` to the `--audit-only` example above and use a fresh output path.
The GPU forward prototype also accepts `--numeric-profile wide-range`. It
still requires a public range audit and compares reconstructed logits against
the corresponding plaintext model; `--allow-out-of-range` only permits an
invalid diagnostic execution. The plaintext audit does not enforce secret
intermediate ranges. Current CPU tests cover synthetic boundary and near-zero
cases, but the user's trained checkpoints, GPU implementation, larger sample
sets, and privacy guarantees have not been validated by these tests. The
three-party timings and numerical errors must be remeasured for this profile.

For public-fixture numerical diagnosis, `python -m hattention.diagnose_vit_cuda`
accepts the same checkpoint, fixed-student, mpcvit-root, image and numeric-profile
arguments as the full-ViT benchmark and writes `diagnosis.json` in a new output
directory. It reconstructs pre-norm, attention, norm1, GeLU and block outputs
on every layer and compares them against the corresponding plaintext model.
All three parties participate in every opening. Only use an explicitly public
fixture: intermediate activations are revealed, so this is not private inference
and its timings and communication cannot be used as benchmark measurements.
Differences at a site include errors inherited from previous sites; they are
not isolated measurements of that operator. The CUDA end-to-end diagnostic
requires validation on the target server.

### Full-forward CrypTen control

`python -m hattention.benchmark_vit_native_cuda` runs full ViTLite with
CrypTen softmax attention, CrypTen erf-based GeLU and the CrypTen
sqrt/reciprocal LayerNorm composition. It reuses the supplied MPCViT
checkpoint's public projections and other weights, but replaces every head
with standard softmax without retraining. Thus this is an operator-cost
control, not an accuracy measurement of a trained standard ViT. Input
sharing and output reconstruction are measured separately. CrypTen provider
work is included in forward and cannot be classified as zero-cost offline work.
Default Q20 matches the prior softmax engineering benchmark; `--precision-bits
17` is available for a same-Q comparison with Ring3. Validate both profiles:
CrypTen erf and softmax approximations can produce severe numerical error.

`--gelu chebyshev16 --chebyshev-bound 12` selects a separate CrypTen
adaptation of the reconstructed 16-degree Chebyshev interpolant, fitted on
[-12,12]. This wider fit is distinct from the earlier CryptGPU [-8,8] fit and
uses reconstructed, not original experiment, coefficients. Before the secure
forward, public exact-GeLU and Chebyshev plaintext models are audited; an
observed input beyond the bound aborts the run. This is not a secret-range
check and does not clamp secret inputs. The report records secure error against
both the exact-GeLU and Chebyshev plaintext models; the correctness gate uses
the selected formula, and neither gate measures classification accuracy. The
MPCViT weights remain unretrained for softmax or Chebyshev. Real CUDA results
and repeated samples are required before drawing performance conclusions.

`python -m hattention.summarize_vit_forward --report LABEL=PATH ... --output DIR`
compares compatible same-image reports, preserving failed correctness statuses
and separate communication counters. A failed numerical check makes a timing
diagnostic, and framework counters and preprocessing boundaries differ. The
original Ring3 full-ViT benchmark can supply the two optimized reports.
