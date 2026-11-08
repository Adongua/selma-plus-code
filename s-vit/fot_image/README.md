# MAE + 图像空间 CNN 提示的独立探索

本目录不更改 `training/` 或 `fot_explore/`。这是借鉴 BlackVIP 思路的独立研究原型，不是 BlackVIP 的逐行复现，也不以此替代原方案。

## 计算流程

1. 数据所有者在本地对 64×64 图像执行冻结 ViT-MAE 编码：将图像反归一化、调整为 224×224，使用 ImageNet 标准化和无随机遮挡的 MAE CLS 特征。公开固定随机投影和 tanh 将特征压到 32 维。冻结编码器每个 batch 只执行一次，±查询复用特征。
2. 三方分别秘密分享图像和特征。秘密 CNN 将 32 维特征重排为 2×4×4，经 3×3 卷积、sReLU、公开最近邻上采样、1×1 卷积生成 3×64×64 图像提示；共 91 个可训练参数。公开种子的非零初始参数以秘密份额保存，但初始化值本身可由所有方重现，不应声称初始化值保密。
3. 对 ViT 的已标准化输入加 `prompt_scale * prompt / channel_std`，等价于在标准化之前将 `prompt_scale * prompt` 加到像素上。当前**不做像素裁剪**，因此允许超出 [0,1]。不得把它描述为已实现安全裁剪；真实输入范围须另外检查。ViT 原有 patch 级提示固定为零。
4. 在 `theta + beta*velocity` 处对全部 CNN 参数同时做 Rademacher ±sigma 扰动。仅数据所有者获得两次 logits，计算本地标签的 CE 差值并秘密分享。三方计算 `(loss_plus-loss_minus)/(2*sigma) * direction`，执行 `velocity = beta*velocity - lr*gradient; theta += velocity`。这是 SPSA-GC 风格的前瞻动量更新；与 BlackVIP 默认 segmented-uniform 扰动和解码器结构不同。

3PC 代码对优化器系数和参数执行秘密范围检查，公开越界失败标志。`--check-forward-ranges` 额外开启前向范围检查（用于诊断，不要和关闭检查的性能数值直接比较）。`--diagnostic-loss` 明确授权公开数据所有者的两次扰动 loss 平均值，它不是更新后验证 loss。参数、动量与特征按秘密份额处理；最终模拟文件的各方份额被写入同一输出根目录，因此必须按真实部署的访问控制分别保管。运行使用现有实验性算子和可选可信截断 dealer，未完成安全证明。

当前仅支持 batch size 1、数据方轮流更新、不支持断点恢复。独立留出集 Top-1 由明文路径评估；3PC 路径验证执行/数值并测量开销，不将秘密参数重构进行真实数据评估。时间是连续执行中扣除已测 CrypTen provider 和 Ring3 准备的估计，包含本地 MAE；同步、未计时预处理和 dealer IPC 的限制与现有 benchmark 一致。`--debug-progress` 额外同步，适合排查；正式计时时关闭。

## 本机验证

- 64×64 公开输入的三方 CNN 输出及归一化路径，与明文结果的绝对误差在 0.003 内；测试只向指定数据所有者重构。
- 公开小模型、3 方各轮流一步，开启前向范围检查；3 步训练完成，测试重构的参数与相同明文配置最大误差约 `1.3e-5`（量化和随机截断会产生波动）。仅针对公开 fixture 测试重构，不用于真实秘密。
- 同一小模型 100 步明文运行，loss `3.349235→3.346297`，Top-1 始终 0%；该玩具实验不说明真实任务有效性。

本机没有服务器的固定学生模型和 MAE 运行环境，因此完整 ViT 和 GPU 性能须在服务器实测。

## 服务器：明文实验

代码需先同步到服务器。请使用已经存在的 `fot-explore-split-v1`，与之前的基线比较；前提是该目录仍包含 party-0/1/2 和 eval manifest。之前的 baseline 训练 100 步、lr=0.01，与这里 100 步的数据顺序一致，但优化器、参数规模、扰动方式和输入提示不同，差异应作为整体方案比较。该留出集与当前微调集不重叠，但原预训练模型可能见过这些训练集图像。

```bash
conda activate selma-plus
cd "$HOME/home/selma-plus-code"
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export DATA="$HOME/home/datasets/tiny-imagenet-200"
export SPLIT="$HOME/home/tmp/fot-explore-split-v1"
export CKPT="$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
export FIXED="$HOME/home/tmp/hattention-distill-0.5-resume200-v1/epoch-0189.pt"
export MAE="$HOME/home/models/vit-mae-base"

CUDA_VISIBLE_DEVICES=0 python -m fot_image.run \
  --backend plaintext --device cuda --encoder mae --mae-path "$MAE" \
  --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
  --eval-root "$DATA" --eval-manifest "$SPLIT/eval.json" \
  --sigma 0.05 --lr 0.01 --momentum 0.9 --prompt-scale 0.1 \
  --steps 100 --eval-every 20 \
  --output "$HOME/home/tmp/fot-image-plain-trial-v1"
```

检查每轮 `evaluation` 的 loss、top1 和 `pixel_prompt_rms`，以及状态报告。数值范围失败时保留报错，不应关闭检查后将结果视为有效。

## 服务器：三方 smoke 和计时

当三张 GPU 空闲后，先使用范围检查和详细日志做 1 步验证。这个步骤会比较慢，其计时不能作为关闭检查时的运行开销。

```bash
CUDA_VISIBLE_DEVICES=1,2,3 timeout 7200 python -m fot_image.run \
  --backend 3pc --device cuda --encoder mae --mae-path "$MAE" \
  --checkpoint "$CKPT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
  --sigma 0.05 --lr 0.01 --momentum 0.9 --prompt-scale 0.1 \
  --steps 1 --trusted-dealer --check-forward-ranges --debug-progress \
  --output "$HOME/home/tmp/fot-image-3pc-check-v1"
```

`tail -f "$HOME/home/tmp/fot-image-3pc-check-v1/progress.jsonl"` 可查看详细进度。检查通过后，去除 `--check-forward-ranges --debug-progress`，将步数改为 4 并使用新的输出目录；丢弃首步，汇总剩余三步的 `benchmark`。各方可分别保留输出的秘密份额；不要随意合并真实训练的份额。检查通过仅覆盖本次样本和轨迹，不代表所有输入安全或数值稳定。
