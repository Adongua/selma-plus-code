# 三方零阶提示微调（初步复现）

本目录实现《基于零阶优化的三方安全提示微调架构-v1》的双边随机方向估计和两种 Adam 更新模式。初步验证目标是：三方下计算不触发数值异常，固定批次 loss 出现下降；使用同样算法的明文训练另行评价效用。这里没有将小样本 smoke 的准确率作为质量指标。

## 1. 实现选择

- 提示位置：在 tokenizer 之后给 **patch embedding 加提示**，CLS 和 token 数不变，不追加 prompt token。固定公共基 `B` 通过带固定种子的高斯矩阵 QR 构造，`p=Bz`；列向量在浮点参考中正交归一，定点投影存在量化误差。
- 冻结骨干：所有参与前向的权重、偏置、CLS 和位置向量均通过 CrypTen 秘密分享。程序只有模型输入方 P0 读取权重，P1/P2 只接收公开结构和份额。当前模型所有者由 P0 兼任；未实现独立第四方模型输入服务。
- 每步：公开单位球面方向、同批次正负扰动、两次覆盖三方批次的前向。当前按数据所有者顺序执行，因此有 6 次本地批次前向；不表示算法需要 6 个不同方向。关闭训练随机层，不建立反向图。
- 各方只重构自己批次的 logits、在本地计算交叉熵，重新分享 `L_plus-L_minus` 后秘密加权。`g=r*delta*u/(2*sigma)`。
- `secret`：z/m/v 为 Q24 份额，二阶矩使用一次秘密标量 `c*c` 与公开 `u*u`。平方根用固定次数秘密二分，倒数用固定次数 Newton 迭代；epsilon 加在平方根外。由定点舍入造成的负二阶矩通过秘密选择投影到非负域，这是数值实现细节，不是替换 Adam 的归一化公式。
- `trusted`：独立的本地 T 进程（不在三方进程组中）接收三方各自的聚合差分份额，重构差分并执行明文 Adam；重新生成三份 Q24 z。T 不接收图片、标签、logits 或单方损失差。Queue 仅模拟本机服务传输，不是认证加密的远程服务。
- `plaintext`：相同基、方向、样本顺序、CE 和 Adam；前向使用同一布局、sReLU/G2 明文公式及量化后的 LayerNorm epsilon。其结果与 3PC 不要求逐位一致。

全秘密指优化状态不向协调方打开，不表示整个程序没有授权输出。当前范围检查会公开**是否越界的失败位**；最终输出采用数据所有者授权重构。原有 Ruler 打开消息及 CrypTen provider 假设没有因本目录实现而得到安全证明。

## 2. 文件

| 文件 | 职责 |
|---|---|
| `core.py` | 超参数、正交基、球面方向、明文 Adam |
| `model.py` | 冻结 ViTLite 导入、公开小模型、共用前向与提示注入 |
| `secure.py` | 秘密权重算子、按所有者重构、范围位检查、秘密 Adam |
| `trusted.py` | 三方组以外的可信 Adam 进程与提示重新分享 |
| `data.py` | 所有者本地样本清单与固定/轮换批次 |
| `run.py` | 明文 / 3PC 训练入口、检查点和报告 |
| `prepare_data.py` | 不移动原始图像，生成互不重叠的公开诊断清单 |
| `evaluate.py` | 明文提示检查点在指定清单上的 CE/Top-1 评估 |
| `tests/` | Adam 对照、定点比较、三方前向和选择性输出测试 |

该实现只新增 `training/`，不改写已有 RSA、LayerNorm、GeLU 或整网推理脚本。

## 3. 最快运行：先用公开小模型验证三方链路

在项目根目录设置：

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
OUT="$HOME/home/tmp/zo-prompt-smoke-v1"
```

输出目录必须尚不存在。下面的 fixture 是一个冻结的 1-block、3-head、3-class 小 Transformer，输入 `[1,3,4,4]`，包含 RSA/LSA/Mixer 和 sGeLU、sLayerNorm；样本和标签明确为公开的同类别过拟合诊断。它不代表 9 层 ViT 已验证，也不衡量分类能力。

```bash
python -m training.run --backend plaintext --fixture \
  --steps 8 --fixed-batch --eval-every 2 --output "$OUT/plaintext"

python -m training.run --backend 3pc --mode secret --fixture \
  --steps 8 --fixed-batch --eval-every 2 --output "$OUT/secret"

python -m training.run --backend 3pc --mode trusted --fixture \
  --steps 8 --fixed-batch --eval-every 2 --output "$OUT/trusted"
```

这三条默认在 CPU/Gloo 上运行。GPU smoke 使用同样参数，额外添加 `--device cuda` 并设置 `CUDA_VISIBLE_DEVICES=1,2,3`；三方各占一张可见 GPU。明文只需一张。

回归测试：

```bash
python -m unittest training.tests.test_core training.tests.test_secure
```

`report.json` 给出 `initial_loss`、`final_loss`、`loss_decreased` 及每个诊断点。默认最终 loss 至少降低 0.001 才标记 `smoke_passed_not_security_validated`；未下降标记 `loss_decrease_not_observed` 并返回退出码 2。范围错误直接终止，不自动换明文或裁掉异常样本。loss 不要求每一步严格单调。

本次本地 CPU 验证：8 步全秘密运行约 `3.35031 → 3.33888`，可信服务模式约 `3.35026 → 3.33891`，相同明文配置约 `3.35021 → 3.33886`。具体以运行报告为准；预处理随机性和硬件状态会造成变化，不能推断完整 ViT 或 GPU 已通过。

另外已验证 9 层 `0.3-kd` 骨干导入：零提示前向与原模型在相同 GeLU、RSA epsilon、LayerNorm epsilon 下的明文输出一致。该检查验证模型布局和权重导入，不替代两个蒸馏后 hAttention 检查点的三方验证。

## 4. 真实固定 ViT 的三方初步验证

先从现有 Tiny-ImageNet 生成互不重叠的小清单，文件路径相对于 dataset root，标签遵循排序后的 `wnids.txt`，不要另外重新编号子集标签：

```bash
DATA="$HOME/home/datasets/tiny-imagenet-200"
SPLIT="$HOME/home/tmp/zo-prompt-split-v1"
python -m training.prepare_data \
  --dataset-root "$DATA" --output "$SPLIT" \
  --classes 4 --per-party-per-class 1 --eval-per-class 1
```

每个数据方只读取自己的 `party-i.json`。实际私有数据可以自行提供三个清单和各自的数据目录。格式为：

```json
[{"path": "train/n01443537/images/n01443537_0.JPEG", "label": 0}]
```

上面的标签值只是格式示例，实际值须与模型的完整 class order 一致。数据不自动视为此前搜索/预训练从未见过。`prepare_data` 生成的 `eval.json` 与本次三方训练清单不重叠，但属于训练图像中的诊断留出集，不是官方验证集。

先跑短程全秘密检查，使用之前选择的 lambda=0.5、epoch 189：

```bash
CHECKPOINT="$PWD/mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
FIXED="$HOME/home/tmp/hattention-distill-0.5-resume200-v1/epoch-0189.pt"

CUDA_VISIBLE_DEVICES=1,2,3 timeout 14400 \
python -m training.run --backend 3pc --mode secret --device cuda \
  --checkpoint "$CHECKPOINT" --fixed-student "$FIXED" \
  --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
  --rank 4 --sigma 0.1 --lr 0.03 --epsilon 0.01 \
  --batch-size 1 --steps 5 --fixed-batch --eval-every 5 \
  --diagnostic-loss \
  --output "$HOME/home/tmp/zo-prompt-vit-secret-v1"
```

此命令使用公开 Tiny-ImageNet 诊断数据，`--diagnostic-loss` **显式允许公开聚合的评估 loss**。实际私有数据默认不输出全局 loss；各方在自己的 `party-i/local-loss.json` 保存其获授权的本地损失。不要将三方的私有目录集中交给任一计算方，当前同机目录隔离仅用于实验组织，不构成 OS 级隐私边界。

首次运行比原先公开权重推理更贵：模型权重也成为秘密输入，每步有正负两次覆盖三方数据的前向，还有安全范围检查。诊断 loss 需要额外的无扰动前向；报告中的 wall 时间包含预处理、检查、输入/输出和诊断，不是纯在线双前向时间。暂不缓存或复用协议材料。

若需要可信模式，将 `--mode secret` 改成 `--mode trusted` 并更换输出目录。若检查到 RSA 分母、GeLU/LayerNorm 输入或 Adam 状态越界，先保存完整报错；不要通过关闭数值检查宣布通过。

## 5. 明文微调与独立清单评估

先复制相同样本、超参数和步数，用 `--backend plaintext --device cuda` 对照 3PC。若评估长期效用，增加样本与步数、去掉 `--fixed-batch`，按确定性的轮换批次训练。例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.run \
  --backend plaintext --device cuda \
  --checkpoint "$CHECKPOINT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
  --rank 32 --sigma 0.1 --lr 0.01 --epsilon 0.01 \
  --batch-size 8 --steps 200 --eval-every 20 \
  --output "$HOME/home/tmp/zo-prompt-vit-plain-v1"
```

上例是调用形式，不是推荐用 12 张图像评估模型效用。应先用更大的新清单替换 `$SPLIT`，例如 `--classes 200 --per-party-per-class 10 --eval-per-class 5`。按样本数加权可传 `--weights n0 n1 n2`；程序归一化这些公开权重。默认三方等权。

训练日志中的诊断 loss 始终在各方固定首批样本上计算，因此只能用于快速观测。完整明文评估使用单独的 manifest：

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.evaluate \
  --prompt "$HOME/home/tmp/zo-prompt-vit-plain-v1/prompt.pt" \
  --checkpoint "$CHECKPOINT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-root "$DATA" --manifest "$SPLIT/eval.json" \
  --device cuda --batch-size 8 \
  --output "$HOME/home/tmp/zo-prompt-vit-plain-eval-v1.json"
```

该评估入口只接受明文 prompt artifact，不会自动拼接三个秘密状态文件或打开全秘密提示。当前首版保存最终状态，不提供跨后端转换或 resume；延长实验需新输出目录重新训练。

## 6. 数值配置与验证界限

- 前向 Q17、优化器 Q24，转换显式执行安全截断。平方根外 epsilon 默认 0.01；这是当前定点数值配置，不是 PyTorch 常见的 1e-8。`c_bound`、`z_bound` 默认 8；越界即失败，不静默裁剪方向估计或提示。
- 新增的精度独立秘密符号位避免本仓库 CrypTen 全局 Q17 配置比较 Q24 状态时可能产生的错误；回归测试同时覆盖正负数和授权重构。
- 当前循环没有安全反向传播，骨干参数不更新；3PC 不公开 delta、梯度或矩状态。可信模式 T 获得聚合 delta 和自身优化状态，这是授权输出。
- 范围检查公开一个失败位，是调试协议的额外可见信息；Ruler 的盲化打开和底层预处理 provider 的安全性仍须独立论证。
- 一次小模型 loss 下降不等于真实模型训练通过；必须在服务器执行第 4 节检查。明文效用实验也不能替代私有数据的最终验证。
- 三方输出目录只供本地复现。T 的 `trusted-service/state.pt` 包含明文提示与 Adam 状态，应只由 T 持有；`party-i/prompt-share.pt` 为第 i 方份额，不应集中公开。
