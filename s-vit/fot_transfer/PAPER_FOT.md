# 基于论文的明文 FoT 对照

`paper_fot.py` 与既有训练入口独立。两条路径共享当前 ViT 和 CIFAR 分类头。

- `blackvip` 使用 MAE 的 768 维 CLS 特征、800 维可学习 trigger、五级反卷积提示解码器，合计 9,920 参数。每步在当前参数处估计分段均匀扰动的 SPSA 梯度，平均 q 个方向，执行 BlackVIP 代码中的梯度修正及学习率、扰动尺度衰减。
- `subspace` 从 `blackvip` checkpoint 初始化 trigger 和解码器，在公开数据上使用高斯双边差分逐列学习两个互相独立的子空间，再补充随机正交列。下游零阶训练使用学习到的子空间。子空间学习与微调的查询分别计数。

当前适配保留我们的 ViT 和数据。为了双边查询确定性，解码器 BatchNorm 使用固定运行统计，MAE 使用固定 token 顺序，生成的 224 像素提示缩放至 ViT 的 64 像素输入并做像素裁剪。该实现使用明文 GeLU、反卷积和 BatchNorm，尚未实现或验证其 3PC 协议。这里的目标是检验方法的明文效果。完整对齐论文实验还涉及 CLIP、数据、模型和训练设置。

输出中的训练 loss 为正负扰动损失均值。每步前向次数为 `2 × directions`，各阶段累计调用记录在 `queries`。`training-split public` 用公开集预训练提示解码器，`party` 轮询三方训练集。CIFAR 官方测试集此前已在开发过程中检查过。

## 服务器示例

在项目根目录执行，输出目录须不存在。以下先做短程联通检查，再做整轮实验。

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
WORK="$HOME/home/tmp"
EXP="$WORK/paper-fot-v1"
common=(
  --archive "$HOME/home/datasets/cifar-10-python.tar.gz"
  --splits "$WORK/cifar10-transfer-splits-v1.json"
  --checkpoint "$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
  --fixed-student "$WORK/hattention-distill-0.5-resume200-v1/epoch-0189.pt"
  --mpcvit-root "$PWD/mpcvit"
  --head "$WORK/cifar10-transfer-head-v1/head.pt"
  --mae-path "$HOME/home/models/vit-mae-base"
  --device cuda --batch-size 8 --eval-batch-size 16
  --allow-plaintext-gelu-out-of-range
)
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.paper_fot "${common[@]}" \
  --method blackvip --training-split public --epochs 1 --max-batches 10 \
  --val-samples 100 --directions 2 --log-every 1 --output "$EXP/smoke-pretrain" &&
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.paper_fot "${common[@]}" \
  --method subspace --initial-decoder "$EXP/smoke-pretrain/epoch-0001.pt" \
  --epochs 1 --max-batches 10 --val-samples 100 --directions 2 \
  --subspace-columns 2 --projection-width 4 --subspace-steps 3 \
  --random-columns 1 --log-every 1 --output "$EXP/smoke-subspace"
```

完成后使用公开样本训练初始化，再用该初始化开展两个方法的整轮训练。

```bash
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.paper_fot "${common[@]}" \
  --method blackvip --training-split public --epochs 1 --directions 5 \
  --val-samples 0 --output "$EXP/pretrain" &&
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.paper_fot "${common[@]}" \
  --method blackvip --training-split party --initial-decoder "$EXP/pretrain/epoch-0001.pt" \
  --epochs 3 --directions 5 --val-samples 0 --output "$EXP/blackvip" &&
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.paper_fot "${common[@]}" \
  --method subspace --training-split party --initial-decoder "$EXP/pretrain/epoch-0001.pt" \
  --epochs 3 --directions 5 --subspace-columns 5 --projection-width 8 \
  --subspace-steps 50 --random-columns 2 --val-samples 0 --output "$EXP/subspace"
```

训练结束后根据验证 Top-1、再以较低验证 loss 决定 epoch。测试时用同一组 `common` 和选中的路径运行一次 `--phase test --test-checkpoint ...`，并指定新输出目录；subspace 测试无需 `--initial-decoder`。两种方法从同一预训练权重开始时，预训练前向次数要与各自的训练查询相加，才能比较总训练开销。

## 全零阶轨迹子空间

`--basis-strategy trajectory` 从给定初始解码器出发，在公开数据上通过 SPSA-GC 零阶更新生成轨迹，对 trigger 和其他解码器参数的更新分别做 SVD，以主要右奇异向量作为子空间方向。整个轨迹通过双边前向计算，随后回到同一初始解码器，在生成的基底里执行零阶微调。可用 `--basis-strategy random` 生成同维随机正交基作为对照。两种策略使用独立且相同种子的后续优化随机序列，保持其余训练设置一致。轨迹构造耗时和查询应计入总开销；此法目前为明文研究路径。
