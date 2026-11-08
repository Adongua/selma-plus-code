# 输入相关 FoT：独立探索原型

该目录不修改 `training/` 原有 FoT 实现。目标是先验证「各方本地特征提取 → 秘密分享特征 → 共同生成输入相关提示 → 零阶更新」的可行性。当前解码器是可学习的线性条件层和原有公开低秩基，不是 BlackVIP 的完整 CNN 解码器；结果不能称为完整方案的最终结果。

每个数据所有者从本地图像提取冻结 MAE 的 CLS 特征，使用公开随机矩阵降到 `feature_dim` 维并执行 tanh，再将特征和原图分别秘密分享。三方计算 `z = bias + W feature`，通过公开正交基得到 patch 嵌入提示；随后在 `theta + beta * momentum` 处进行两次方向扰动查询，只有数据所有者接收 logits 并计算 CE 差值，三方秘密更新 `momentum = beta * momentum - lr * gradient` 和 `theta += momentum`。梯度估计使用球面随机方向及参数维度修正；它不是 BlackVIP 默认的 segmented-uniform SPSA。既有基线为共享提示和原有秘密 Adam。两者的模型、权重、批量、数据顺序及扰动查询次数可保持相同，但参数规模与优化器不同，不能将差异全部归因于条件提示。

当前实验限制：只支持 batch size 1；只支持三方 round-robin；完整网络的 3PC 版本目前仅输出开销和秘密参数份额，不计算独立验证集准确率。MAE 在各方本地运行，时间计入对应 3PC forward wall；因此与旧报表比较时要确保计时口径相同。条件参数、动量、特征和主模型在 3PC 中秘密分享，但本地数据所有者能看到其 logits；可信截断 dealer 只是性能模拟，不构成安全证明。秘密状态检查会泄露是否越界的一个失败标志。首次输出路径必须不存在。

## 本机公开 fixture 结果（不是完整网络的性能结果）

2026-11-09 本机 CPU、小模型、batch 1、可信截断 dealer，每种方法 2 次 step，仅列第 2 次估计：原 FoT forward 0.149 s、optimizer 0.972 s、total 1.121 s；条件原型分别 0.174 s、0.051 s、0.224 s。通信估计分别为 1,399,416 和 847,920 bytes。两次各只有一个去掉首步后的样本，不能外推到 GPU ViT。三方条件模型两步后秘密参数重构与对应明文最大误差约 0.00011，重构仅用于公开 fixture 的调试。

公开 fixture、固定训练 batch、lr=0.001 的 100 次明文更新中，原 FoT loss 3.34924→3.34352；条件原型 3.34924→3.34439；两者 Top-1 都为 0%。该 fixture 随机、很小且不是独立验证集，这不是精度结论。使用 lr=0.03 继续训练时两个方案都触发了 RSA 分母范围检查，提示需要调参，不能把完成两步看作长时间稳定的证明。

## 在 GPU 服务器上复测

这些命令要求代码已同步到服务器、当前环境可 import `torch`、`transformers`，且 `MAE` 指向一个本地 Hugging Face ViT-MAE 预训练模型目录（含 config 和权重）。没有 MAE 权重时不要用 fixture 结果冒充真实 MAE。请不要覆盖现有实验目录。

```bash
cd "$HOME/home/selma-plus-code"
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
DATA="$HOME/home/datasets/tiny-imagenet-200"
SPLIT="$HOME/home/tmp/fot-explore-split-v1"
CKPT="$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
FIXED="$HOME/home/tmp/hattention-distill-0.5-resume200-v1/epoch-0189.pt"
MAE="$HOME/home/models/vit-mae-base"

# 留出样本与三方训练 manifest 不重叠；预训练模型可能见过这些训练图像。
python -m training.prepare_data --dataset-root "$DATA" --output "$SPLIT" \
  --classes 20 --per-party-per-class 8 --eval-per-class 4

for method in baseline conditioned; do
  CUDA_VISIBLE_DEVICES=0 python -m fot_explore.run \
    --backend plaintext --method "$method" --device cuda \
    --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
    --dataset-roots "$DATA" "$DATA" "$DATA" \
    --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
    --eval-root "$DATA" --eval-manifest "$SPLIT/eval.json" \
    --encoder mae --mae-path "$MAE" --rank 4 --feature-dim 8 \
    --sigma 0.1 --lr 0.001 --momentum 0.9 --steps 100 --eval-every 20 \
    --output "$HOME/home/tmp/fot-explore-plain-${method}-v1" || break
done
```

比较 `report.json` 的 `initial` 和各步 `evaluation`。这里评价的是训练 manifest 留出的 80 张样本，不是官方验证集，且基于同一个 seed 的小样本结果不足以证明泛化；正式结论应增加数据、独立验证集和多个种子。

先在 GPU 资源充足时测试 2 步以排查 OOM，再延长到 4 步并排除首步：

```bash
for method in baseline conditioned; do
  CUDA_VISIBLE_DEVICES=1,2,3 timeout 14400 python -m fot_explore.run \
    --backend 3pc --method "$method" --device cuda \
    --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
    --dataset-roots "$DATA" "$DATA" "$DATA" \
    --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
    --encoder mae --mae-path "$MAE" --rank 4 --feature-dim 8 \
    --sigma 0.1 --lr 0.001 --momentum 0.9 --steps 4 --trusted-dealer \
    --output "$HOME/home/tmp/fot-explore-mpc-${method}-v1" || break
done

python - <<'PY'
import json, statistics
from pathlib import Path
for method in ('baseline', 'conditioned'):
    p = Path.home() / f'home/tmp/fot-explore-mpc-{method}-v1/report.json'
    if not p.exists():
        print(method, 'no completed report:', p)
        continue
    report = json.loads(p.read_text())
    rows = report['steps'][1:]
    print(method, 'measured steps:', len(rows))
    for stage in ('forward', 'optimizer', 'total'):
        print(' ', stage,
              'online_s=', round(statistics.mean(r['benchmark'][stage]['online_estimate_seconds'] for r in rows), 3),
              'online_GB=', round(statistics.mean(r['benchmark'][stage]['online_communication_bytes'] for r in rows)/1e9, 6))
PY
```

在线时间是连续执行中扣除已计时的 provider 和 Ring3 预处理后的估计值，包含本地特征提取及同步；未计时的预处理、控制消息和 dealer IPC 不在扣除/通信统计内。可信 dealer 仅替换 Ring3 截断材料生成，不代表所有预处理均已排除。若运行时 RSA、参数或方向系数越界，需保留报错并检查，不要将结果作为有效测试。
