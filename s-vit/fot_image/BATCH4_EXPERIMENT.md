# 原 FoT 与 MAE-CNN：明文精度曲线与 batch=4 三方开销

原 FoT 使用 `training.core.PlainAdam` 和原有球面方向；MAE-CNN 使用图像提示和 SPSA-GC 风格更新。`fot_image.compare_baseline` 是独立明文驱动，不修改原 FoT；两边同用现有冻结 hAttention ViT、三方轮流和数据顺序。表明网络、训练样本数量、每步使用 batch 大小、评估样本及训练步数一致；提示位置、参数数目、扰动、学习率、优化器不同，实验衡量完整方案，而非单一变量。

以下只生成训练图像中互不重叠的训练/留出 manifest，固定 backbone 在预训练和结构选择时可能见过这些图；因此不能称作完全未见过的最终测试集。明确报告类别顺序、样本数和 seed；反复使用这些样本调参会进一步使评估偏乐观。为防止按类别顺序只训练前几十类，对三个新建训练 manifest 做确定性打乱；同一数据与评估被两个方法复用。最终性能需要独立于训练和模型选择的数据集、多个种子及类别一致的准确率评估。

## 环境

```bash
conda activate selma-plus
cd "$HOME/home/selma-plus-code"
git rev-parse HEAD
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export DATA="$HOME/home/datasets/tiny-imagenet-200"
export CKPT="$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
export FIXED="$HOME/home/tmp/hattention-distill-0.5-resume200-v1/epoch-0189.pt"
export MAE="$HOME/home/models/vit-mae-base"
export ACC_SPLIT="$HOME/home/tmp/fot-accuracy-200cls-split-v1"
export BENCH_SPLIT="$HOME/home/tmp/zo-prompt-split-v1"
```

所有输出路径必须不存在。以下新建每方 600 张（每类 3 张）和 1,000 张留出图像（每类 5 张），无文件移动，清单使用固定种子。类别标签沿用仓库既有 sorted-wnid 映射，正式研究应核对 checkpoint 类别映射。

```bash
python -m training.prepare_data --dataset-root "$DATA" --output "$ACC_SPLIT" \
  --classes 200 --per-party-per-class 3 --eval-per-class 5 --seed 20261107

python - <<'PY'
import json, os, random
from pathlib import Path
root = Path(os.environ['ACC_SPLIT'])
for owner in range(3):
    path = root / f'party-{owner}.json'
    rows = json.loads(path.read_text())
    random.Random(20261107 + owner).shuffle(rows)
    path.write_text(json.dumps(rows, indent=2) + '\n')
    print(path, len(rows))
print('evaluation images:', len(json.loads((root/'eval.json').read_text())))
PY
```

## 明文 200 类曲线

每方案 300 次更新，每步每个数据方拿 4 张，轮换数据方，固定相同的顺序；每 50 步评估同一 1,000 张留出图并保存 checkpoint。评估耗时不纳入每步训练 benchmark。请在 GPU 0 空闲时依次执行，两命令不能写入同一输出目录；实际评估 1,000 张图需耗时。参数当前按两方法各自已使用的设置，超参未经过公平调优。

默认沿用三方 sGeLU 的 ±12 输入限制。如果明文数据集触发 `gelu_input` 范围错误，可在两条明文命令中**同时**加入 `--allow-plaintext-gelu-out-of-range`，使用 g2 明文公式对有限的越界输入继续评估，并在报告 `plaintext_range_audit` 中统计触发次数、元素数和最大值。重复查询会重复计数；该选项不是安全算子扩域，不代表越界时三方协议正确，非有限值和其他算子的范围检查仍然生效。3PC 禁止使用此选项。明文准确率和三方数值验证应分别报告。

```bash
CUDA_VISIBLE_DEVICES=0 python -m fot_image.compare_baseline \
  --device cuda --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$ACC_SPLIT/party-0.json" "$ACC_SPLIT/party-1.json" "$ACC_SPLIT/party-2.json" \
  --eval-root "$DATA" --eval-manifest "$ACC_SPLIT/eval.json" \
  --rank 4 --sigma 0.1 --lr 0.03 --epsilon 0.01 \
  --batch-size 4 --steps 300 --eval-every 50 --seed 20261107 \
  --output "$HOME/home/tmp/fot-accuracy-b4-baseline-v1"

CUDA_VISIBLE_DEVICES=0 python -m fot_image.run \
  --backend plaintext --device cuda --encoder mae --mae-path "$MAE" \
  --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$ACC_SPLIT/party-0.json" "$ACC_SPLIT/party-1.json" "$ACC_SPLIT/party-2.json" \
  --eval-root "$DATA" --eval-manifest "$ACC_SPLIT/eval.json" \
  --sigma 0.05 --lr 0.001 --momentum 0.9 --prompt-scale 0.1 \
  --batch-size 4 --steps 300 --eval-every 50 --seed 20261107 \
  --output "$HOME/home/tmp/fot-accuracy-b4-image-v1"
```

每个评估时的 `step-NNNNNN.pt` 含对应明文模型状态；如某次因数值检查中断，先前 checkpoint 仍保留，但程序目前不支持从 checkpoint 恢复，也不会在异常时写最终 report。

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path.home()/'home/tmp'
for label, name in [('Original FoT', 'fot-accuracy-b4-baseline-v1'),
                    ('MAE-CNN FoT', 'fot-accuracy-b4-image-v1')]:
    path = root/name/'report.json'
    if not path.exists():
        print(label, '未完成，检查日志及 checkpoint:', path.parent)
        continue
    result = json.loads(path.read_text())
    print('\n',label, 'initial:', result['initial'])
    for row in result['steps']:
        if 'evaluation' in row:
            e = row['evaluation']
            print(f"step={row['step']:3} loss={e['loss']:.6f} top1={e['top1']:.2%} n={e['samples']}")
PY
```

## 三方 batch=4

使用先前的 `zo-prompt-split-v1` 固定 batch，便于初步对照既有原 FoT 报告。先确认 GPU 1、2、3 空闲：

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=1,2,3 timeout 10800 python -m fot_image.run \
  --backend 3pc --device cuda --encoder mae --mae-path "$MAE" \
  --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$BENCH_SPLIT/party-0.json" "$BENCH_SPLIT/party-1.json" "$BENCH_SPLIT/party-2.json" \
  --sigma 0.05 --lr 0.001 --momentum 0.9 --prompt-scale 0.1 \
  --batch-size 4 --steps 1 --fixed-batch --trusted-dealer --check-forward-ranges --debug-progress \
  --output "$HOME/home/tmp/fot-image-3pc-b4-check-v1"
```

首次 smoke 的额外检查和日志会改变时延，不能当作 benchmark。成功后关闭额外检查与调试、采用新目录测 4 步：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 timeout 21600 python -m fot_image.run \
  --backend 3pc --device cuda --encoder mae --mae-path "$MAE" \
  --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$BENCH_SPLIT/party-0.json" "$BENCH_SPLIT/party-1.json" "$BENCH_SPLIT/party-2.json" \
  --sigma 0.05 --lr 0.001 --momentum 0.9 --prompt-scale 0.1 \
  --batch-size 4 --steps 4 --fixed-batch --trusted-dealer \
  --output "$HOME/home/tmp/fot-image-3pc-b4-benchmark-v1"

python - <<'PY'
import json, statistics
from pathlib import Path
p = Path.home()/'home/tmp/fot-image-3pc-b4-benchmark-v1/report.json'
r = json.loads(p.read_text())
rows = r['steps'][1:]
print('status:', r['status'], 'measured steps:', len(rows))
for stage in ('forward','optimizer','total'):
    print(stage,
        'online s:', round(statistics.mean(x['benchmark'][stage]['online_estimate_seconds'] for x in rows),3),
        'comm GB:', round(statistics.mean(x['benchmark'][stage]['online_communication_bytes'] for x in rows)/1e9,6))
print('local MAE s:', round(statistics.mean(x['feature_extraction_seconds_owner'] for x in rows),3))
PY
```

一条 3PC step 包含所有者本地 MAE、输入及特征分享、两次 CNN+ViT、差值分享及一次优化器更新。仅成本可与现有 batch=4 旧 FoT 在相似条件下做初步比较；未同步控制 GPU、通信和数值精度前，不是严格结论。指标为插桩估计、不是严格离线预处理后纯在线重放；通信不包含 dealer IPC/所有控制元数据。MAE 时间已包含在 forward 范围内。可信 dealer 仅是模拟，状态 `completed_not_security_validated` 不是安全或精度证明。

## 官方验证集上的统一明文对照

`fot_image.evaluate_official` 同时比较无提示、两份原 FoT checkpoint 和两份 MAE-CNN checkpoint。使用与微调相同的训练前向算子、同一 MAE 特征和相同 64×64 输入，在官方验证标注上计算 Top-1、Top-5、CE 和逐样本相对无提示配置的预测变化；batch 化并每 100 张汇报一次进度。`--max-samples 200` 为按类别平衡的 smoke 子集，不是完整验证集；`--max-samples 0` 使用全部官方验证图像。报告包含模型和 checkpoint 的 SHA-256，以及每个配置的 sGeLU 越界统计。checkpoint 旧格式没有保存来源权重 hash，程序只核对保存的路径，无法证明文件曾未被替换。

示例依赖本文环境变量和已完成的 `fot-accuracy-b4-baseline-v2` 与 `fot-accuracy-b4-image-v2`：

```bash
ROOT="$HOME/home/tmp"
CUDA_VISIBLE_DEVICES=0 python -m fot_image.evaluate_official \
  --device cuda --dataset-root "$DATA" \
  --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" --mae-path "$MAE" \
  --baseline-checkpoints "$ROOT/fot-accuracy-b4-baseline-v2/step-000200.pt" "$ROOT/fot-accuracy-b4-baseline-v2/step-000300.pt" \
  --image-checkpoints "$ROOT/fot-accuracy-b4-image-v2/step-000200.pt" "$ROOT/fot-accuracy-b4-image-v2/step-000300.pt" \
  --batch-size 4 --max-samples 200 --allow-plaintext-gelu-out-of-range \
  --output "$ROOT/fot-official-eval-smoke-v1.json"
```

完成 smoke 后，将 `--max-samples` 改为 `0` 并选择新的输出文件，可评价完整验证集。有限的 sGeLU 输入越界仅在明确指定选项时按明文 g2 公式继续，其他检查仍在；不代表三方协议可在越界时正确执行。官方验证集已用于先前模型和 checkpoint 的评估，不能再称为独立最终测试集。这里全部使用训练脚本的明文近似算子，之前 `hattention.evaluate_fixed` 使用的固定模型计算路径不同，不应把旧的 59.29% 作为完全同口径的无提示行。不要将 200 张 smoke 结果与 10,000 张结果混用。

只需比较无提示和初始 CNN 时，可不传其他 checkpoint 列表，改为传入 `--image-initial-from "$ROOT/fot-accuracy-b4-image-v2/step-000200.pt"`。程序从该 checkpoint 读取初始化 seed、投影配置和缩放系数，用当前代码重新生成 CNN 初始参数，并在报告中记录该配置来源及其 SHA-256。此结果是重建的初始参数，并不是读取 checkpoint 中已训练的 theta；若初始化代码或预训练 MAE 权重曾变化，需要另行核对。
