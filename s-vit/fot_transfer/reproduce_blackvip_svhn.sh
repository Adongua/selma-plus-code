#!/usr/bin/env bash
# Run the upstream CVPR 2023 SVHN experiment. See BLACKVIP_SVHN.md for assets.
set -euo pipefail
code_root=$(cd "$(dirname "$0")/../.." && pwd)
upstream=$(cd "${BLACKVIP_DIR:-$code_root/BlackVIP}" && pwd)
data_root=${BLACKVIP_DATA:-$HOME/home/datasets}
out=${BLACKVIP_OUT:-$HOME/home/tmp/blackvip-original-svhn}
mae_dir=${MAE_DIR:-$HOME/home/models/vit-mae-base}
stage=${1:-help}
case "$stage" in
  prepare|zero|smoke|train) ;;
  *) echo "Usage: bash $0 {prepare|zero|smoke|train} [seed 1, 2 or 3 for train]"; exit 2 ;;
esac
export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
expected=66606fbc57dd9b8bfbe4d7296d0fd5662d6e83d2
actual=$(git -C "$upstream" rev-parse HEAD)
if [[ "$actual" != "$expected" ]]; then
  echo "Expected upstream revision $expected, found $actual" >&2
  exit 1
fi

if [[ "$stage" == prepare ]]; then
  mkdir -p "$data_root"
  python - "$data_root" <<'PY'
from datasets.svhn_dl import download_and_prepare
import sys
download_and_prepare('svhn', sys.argv[1])
PY
fi
# The loader silently creates a new random split when this filename is missing.
python - "$data_root/svhn" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
p = root / 'split_SVHN.json'
if not p.is_file():
    sys.exit(f'Place the author-provided split_mlai_SVHN.json at {p}')
split = json.loads(p.read_text())
for kind in ('train', 'val'):
    if not split[kind]:
        sys.exit(f'Empty {kind} split')
    for rel, label, name in split[kind]:
        if not (root / 'train' / rel).is_file():
            sys.exit(f'Missing {kind} image: {rel}')
for kind, expected in [('train', 73257), ('test', 26032)]:
    n = sum(1 for _ in (root / kind).glob('*/*.jpg'))
    if n != expected:
        sys.exit(f'{kind} has {n}/{expected} JPEGs in {root}')
PY
if [[ "$stage" == prepare ]]; then exit 0; fi

python -c 'import torch; assert torch.cuda.is_available(), "Upstream BLACKVIP needs CUDA"'
mkdir -p "$out/work"
cd "$out/work"
if [[ "$stage" != zero && -d "$mae_dir" ]]; then
  # Transformers treats this upstream model ID as a local path if present.
  mkdir -p facebook
  if [[ ! -e facebook/vit-mae-base ]]; then ln -s "$mae_dir" facebook/vit-mae-base; fi
  [[ facebook/vit-mae-base -ef "$mae_dir" ]] || { echo "MAE path conflict in $PWD/facebook" >&2; exit 1; }
fi
cfg=(--root "$data_root" --dataset-config-file "$upstream/configs/datasets/svhn.yaml"
     --config-file "$upstream/configs/trainers/BLACKVIP/vit_b16.yaml")
if [[ "$stage" == zero ]]; then
  python "$upstream/train.py" "${cfg[@]}" --trainer ZeroshotCLIP --eval-only \
    --seed 1 --output-dir "$out/zero" DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES all
  exit 0
fi
epochs=5000
if [[ "$stage" == smoke ]]; then epochs=1; fi
seeds=(1 2 3)
if [[ "$stage" == smoke || -n ${2:-} ]]; then seeds=("${2:-1}"); fi
for seed in "${seeds[@]}"; do
  case "$seed" in 1|2|3) ;; *) echo "Expected seed 1, 2 or 3" >&2; exit 2 ;; esac
  dir="$out/$stage/seed$seed"
  if [[ -e "$dir" ]]; then
    echo "Output already exists at $dir. Select a fresh BLACKVIP_OUT or specify an unrun seed." >&2
    exit 1
  fi
  python "$upstream/train.py" "${cfg[@]}" --trainer BLACKVIP --seed "$seed" --output-dir "$dir" \
    TRAIN.CHECKPOINT_FREQ 500 DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES all \
    OPTIM.MAX_EPOCH "$epochs" TRAINER.BLACKVIP.PT_BACKBONE vit-mae-base \
    TRAINER.BLACKVIP.SPSA_PARAMS '[1.0,0.005,0.01,0.4,0.2]' \
    TRAINER.BLACKVIP.OPT_TYPE spsa-gc TRAINER.BLACKVIP.MOMS 0.9 TRAINER.BLACKVIP.P_EPS 1.0
done
