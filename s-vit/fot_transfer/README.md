# CIFAR-10 迁移实验（明文研究原型）

本目录独立于 `training/`、`fot_image/` 的已有训练方案。目标是在相同的预训练 λ=0.5 hAttention ViT、公开校准头、数据和任务条件下，比较无提示、原 FoT、MAE-CNN + 零阶、MAE-CNN + 一阶 Adam。三方开销和准确率并非本实验范围；当前仅实现明文。

## 数据与可比性

- 输入官方 CIFAR-10 Python 归档（`cifar-10-python.tar.gz`），验证标准 MD5 `c58f30108f718f92721af3b95e74349a`。加载器在校验后用 Python pickle 读取固定成员，并不解压路径；**只应使用可信来源的官方归档**。MD5 用于检测意外损坏，不是抗攻击的真实性证明。报告另记录归档 SHA-256。
- 按 CIFAR-10 原始类标签顺序：`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`。各类分出 500 张作为公开分类头数据、500 张验证，余下 4,000 张交给三方。总量分别是公开 5,000、验证 5,000、三方 13,334 / 13,333 / 13,333；官方 10,000 张测试图只在最终评估阶段解码。数据和种子固定，并校验无重复、无遗漏。
- 图像从 RGB 32×32 以双三次缩放至 64×64，再使用现有 Tiny-ImageNet 归一化。MAE 编码器按其既有实现反归一化、缩放至 224×224 并提取冻结 CLS 特征。CIFAR 与任何外部预训练数据之间可能存在重复或语义重合；不能未经核查声称每张图在所有预训练阶段都未出现过。
- 公开分类头从冻结 ViT 的 CLS 特征训练 `192→10`；按验证集 loss 选取分类头，之后对所有提示方案**共同冻结**。因此无提示起点已经用 5,000 张公开目标数据做过分类头准备，不是零样本。提示训练阶段三方只轮换各自的训练数据；选择提示 checkpoint 时仅使用验证集，测试命令须读取固定的选择记录。
- 微调期间默认从 5,000 张验证图中取类别均衡的 1,000 张，所有方法复用相同样本；`--val-samples 0` 使用全部。所选头曾用完整 5,000 张验证数据选 epoch，因此验证集是模型选择集，不是最终测试集。正式报告使用官方 test，且不要根据 test 再调参。
- 原 FoT 使用 4 维 patch 提示与球面 SPSA + Adam；图像 ZO 使用 91 参数 CNN、Rademacher SPSA-GC；一阶图像 Adam 是同 CNN 的明文参照。统一 backbone、分类头、训练轮换、样本顺序及评估；算法、超参数和每步前向次数不同，效果比较不能归因于某个单一变量。
- 此实验使用当前训练脚本的明文 g2、LayerNorm 和异构 attention 参考，开启 GeLU 明文范围放宽时会记录越界次数；不能据此推断三方在越界输入上正确。数据含所有三方明文，仅用于预实验；不是隐私保护训练。

## 依赖与源数据

需要 Python、PyTorch、NumPy、Pillow、transformers（用于 MAE）；无需 torchvision。权重使用服务器原有 `0.3-kd` 和 λ=0.5 第 189 步固定学生及本地 MAE。

本机归档位置：`/Users/adongua/Documents/PythonProject/selma-plus/datasets/cifar-10-python.tar.gz`，SHA-256：`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`。在本机将其复制到服务器 `~/home/datasets/`；以下假设服务器对应路径为 `$HOME/home/datasets/cifar-10-python.tar.gz`。

```bash
conda activate selma-plus
cd "$HOME/home/selma-plus-code"
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export CIFAR="$HOME/home/datasets/cifar-10-python.tar.gz"
export SPLITS="$HOME/home/tmp/cifar10-transfer-splits-v1.json"
export CKPT="$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
export FIXED="$HOME/home/tmp/hattention-distill-0.5-resume200-v1/epoch-0189.pt"
export MAE="$HOME/home/models/vit-mae-base"
export HEAD_DIR="$HOME/home/tmp/cifar10-transfer-head-v1"
export ROOT="$HOME/home/tmp/cifar10-transfer-v1"
sha256sum "$CIFAR"
```

所有新输出路径必须不存在；重复运行时换目录，不要删除以前的结果。实验会记录并检查归档、划分、分类头及 backbone 的 hash。

## 1. 创建划分

```bash
python -m fot_transfer.prepare --archive "$CIFAR" --output "$SPLITS" \
  --public-per-class 500 --val-per-class 500 --seed 20261109
```

预期 5,000 / 5,000 / 13,334 / 13,333 / 13,333 / 10,000。只生成索引，不复制或修改原始图片。

## 2. 训练并固定公开分类头

确认 GPU 0 空闲后执行。先批量提取公开集和验证集的冻结特征（各 5,000 张），再在缓存特征上训练线性分类头；每 80 张左右输出进度。只用验证 loss 选头，**不读取 test batch**。

```bash
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.calibrate \
  --archive "$CIFAR" --splits "$SPLITS" --checkpoint "$CKPT" \
  --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --device cuda --feature-batch-size 8 --epochs 20 --train-batch-size 128 \
  --lr 0.01 --seed 20261109 --allow-plaintext-gelu-out-of-range \
  --output "$HEAD_DIR"
export HEAD="$HEAD_DIR/head.pt"
```

## 3. 在同一数据上训练三种提示

示例为 batch=4、3000 步、每 200 步评估 1,000 张类别均衡验证图；每方法执行 12,000 张训练样本曝光，未必覆盖全部 40,000 张。范围、资源或成本需要先短跑时，可将三条命令的 `--steps` 同时改小并使用新目录。每条在每次评估后保存 `step-NNNNNN.pt`，暂不支持恢复训练。

```bash
for method in baseline image-zo image-adam; do
  if [ "$method" = baseline ]; then
    sigma=0.1
    lr=0.03
  else
    sigma=0.05
    lr=0.001
  fi
  CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.train \
    --method "$method" --archive "$CIFAR" --splits "$SPLITS" \
    --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
    --head "$HEAD" --mae-path "$MAE" --device cuda \
    --steps 3000 --eval-every 200 --batch-size 4 --eval-batch-size 8 --val-samples 1000 \
    --rank 4 --sigma "$sigma" --lr "$lr" --epsilon 0.01 --momentum 0.9 --prompt-scale 0.1 \
    --seed 20261107 --allow-plaintext-gelu-out-of-range \
    --output "$ROOT/$method" || break
done
```

各方法保存 `report.json`、初始指标、每 200 步的 loss/Top-1 和对应参数。记录耗时或使用 16 个 SPSA 方向时应另做等前向预算对照；不能用等步数声称训练开销相同。

## 4. 验证集选择，之后一次性评估测试集

预先固定选择规则：验证 Top-1 最大，若相同则验证 CE 最低，仍相同选较早步骤。该命令不读取 CIFAR test，且要求三种运行的 seed、步数、batch、验证大小与评估间隔一致。

```bash
python -m fot_transfer.select \
  --baseline "$ROOT/baseline" --image-zo "$ROOT/image-zo" --image-adam "$ROOT/image-adam" \
  --output "$ROOT/selection.json"
```

先用验证集的 40 张 smoke，确保评估器可用，**不会读取测试集**：

```bash
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.evaluate \
  --archive "$CIFAR" --splits "$SPLITS" --checkpoint "$CKPT" --fixed-student "$FIXED" \
  --mpcvit-root "$PWD/mpcvit" --head "$HEAD" --mae-path "$MAE" \
  --selection "$ROOT/selection.json" --device cuda --batch-size 8 \
  --evaluation-split validation --max-samples 40 --allow-plaintext-gelu-out-of-range \
  --output "$ROOT/eval-smoke.json"
```

确认选择和代码后，对完整 10,000 张官方测试图评估一次：

```bash
CUDA_VISIBLE_DEVICES=0 python -m fot_transfer.evaluate \
  --archive "$CIFAR" --splits "$SPLITS" --checkpoint "$CKPT" --fixed-student "$FIXED" \
  --mpcvit-root "$PWD/mpcvit" --head "$HEAD" --mae-path "$MAE" \
  --selection "$ROOT/selection.json" --device cuda --batch-size 8 \
  --evaluation-split test --allow-plaintext-gelu-out-of-range \
  --output "$ROOT/test-report.json"
```

结果含：`no-prompt`、`image-initial`、`baseline`、`image-zo`、`image-adam`。记录 loss、Top-1/Top-5 和相对无提示的成对预测改变计数；只有经过多个独立实验种子、数值范围核验和明确数据来源后才适合推广结论。官方测试被查看后即不再适合用来继续调参。
