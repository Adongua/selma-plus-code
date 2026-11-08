# 三方零阶提示微调（初步复现）

本目录实现《基于零阶优化的三方安全提示微调架构-v1》的双边随机方向估计和两种 Adam 更新模式。初步验证目标是：三方下计算不触发数值异常，固定批次 loss 出现下降；使用同样算法的明文训练另行评价效用。这里没有将小样本 smoke 的准确率作为质量指标。

## 1. 实现选择

- 提示位置：在 tokenizer 之后给 **patch embedding 加提示**，CLS 和 token 数不变，不追加 prompt token。固定公共基 `B` 通过带固定种子的高斯矩阵 QR 构造，`p=Bz`；列向量在浮点参考中正交归一，定点投影存在量化误差。
- 冻结骨干：所有参与前向的权重、偏置、CLS 和位置向量均通过 CrypTen 秘密分享。程序只有模型输入方 P0 读取权重，P1/P2 只接收公开结构和份额。当前模型所有者由 P0 兼任；未实现独立第四方模型输入服务。
- 默认 `--update-schedule round-robin`：每步按 P0、P1、P2 轮流选择一方的一个 batch，正、负扰动各做一次三方联合安全前向，将该方 loss 差秘密分享后更新一次共享 prompt。每个 step 使用新方向，三步才覆盖三方各一个 batch。每方独立维护批次游标；`--fixed-batch` 使其始终使用各自首批样本。无反向图。
- 可选 `--update-schedule aggregate`：旧方案，每步各方各处理一个 batch，用同一方向计算三个 loss 差，秘密加权聚合后更新一次；每步六次安全前向。`g=r*delta*u/(2*sigma)`。
- `secret`：z/m/v 为 Q24 份额，二阶矩使用一次秘密标量 `c*c` 与公开 `u*u`。平方根用固定次数秘密二分，倒数用固定次数 Newton 迭代；epsilon 加在平方根外。由定点舍入造成的负二阶矩通过秘密选择投影到非负域，这是数值实现细节，不是替换 Adam 的归一化公式。
- `trusted`：独立的本地 T 进程（不在三方进程组中）接收三方的差分份额、重构并执行明文 Adam，再重新分享 Q24 z。**轮流更新时重构的是当前数据方的单方 loss 差**；聚合模式才是聚合后的 loss 差。使用前须授权 T 获取该信息。T 不直接接收图片、标签或 logits。Queue 仅模拟本机服务传输，不是认证加密的远程服务。
- `plaintext`：相同基、方向、样本顺序、CE 和 Adam；前向使用同一布局、sReLU/G2 明文公式及量化后的 LayerNorm epsilon。其结果与 3PC 不要求逐位一致。

全秘密指优化状态不向协调方打开，不表示整个程序没有授权输出。当前范围检查会公开**是否越界的失败位**；最终输出采用数据所有者授权重构。原有 Ruler 打开消息及 CrypTen provider 假设没有因本目录实现而得到安全证明。

## 2. 文件

| 文件 | 职责 |
|---|---|
| `core.py` | 超参数、正交基、球面方向、明文 Adam |
| `model.py` | 冻结 ViTLite 导入、公开小模型、共用前向与提示注入 |
| `secure.py` | 秘密权重算子、按所有者重构、范围位检查、秘密 Adam |
| `trace.py` | 可选进度心跳、分块时间和 Ring3 准备计时 |
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

旧的聚合模式曾完成本地 CPU 8 步验证；不同更新模式的一个 step 所含数据量不同，应分别查看运行报告。预处理随机性和硬件状态会造成变化，不能据小模型推断完整 ViT 或 GPU 已通过。

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
  --batch-size 1 --steps 6 --fixed-batch --eval-every 6 \
  --diagnostic-loss --update-schedule round-robin --debug-progress \
  --output "$HOME/home/tmp/zo-prompt-vit-round-robin-v1"
```

此命令使用公开 Tiny-ImageNet 诊断数据，`--diagnostic-loss` **显式允许公开聚合的评估 loss**。实际私有数据默认不输出全局 loss；各方在自己的 `party-i/local-loss.json` 保存其获授权的本地损失。不要将三方的私有目录集中交给任一计算方，当前同机目录隔离仅用于实验组织，不构成 OS 级隐私边界。

默认每步仅对当次数据方做正负各一次前向；初始化诊断和每次评估各额外覆盖三方，因此分别包含三次无扰动前向。`--steps 6` 表示六次 Adam 更新，每方依次参与两次；不是六个完整数据集 epoch。报告中的 wall 时间包含预处理、优化器检查、输入/输出和诊断，不是纯在线双前向时间。暂不缓存或复用协议材料。

`training_step` 和 `report.json` 的每步记录分别给出 `training_update_seconds`（从该步开始到 Adam 更新完成）、`evaluation_seconds`（该步评估时间，未评估时为 `null`）和 `wall_seconds`（含更新、评估和少量日志等开销）。三方模式对每个时间段分别取各方本地时长的最大值，因此三个最大值可能来自不同参与方，不一定严格相加。初始化评估单独由 `training_started` 中的 `initial_evaluation_seconds_rank0` 报告，不在每步时间中。更新和评估时间均包含即时预处理、通信、输出交付与同步，不等于纯在线时间。

三方运行默认打印模型加载、每次正负前向、差分分享、Adam 和评估的起止事件，并逐条写入输出目录下的 `progress.jsonl`，可在另一个终端用 `tail -f "$OUTPUT/progress.jsonl"` 查看。三方运行的 `--debug-progress` 额外打印每个 block、Ring3 材料准备起止和每 30 秒 `forward_heartbeat`；可用 `--heartbeat-seconds 10` 修改心跳间隔。`block_complete` 和 `forward_complete` 中的 `ring3_prepare_seconds_rank0`、`range_check_seconds_rank0` 是 rank 0 的本地墙钟时间，准备和检查以外的时间仍包含 CrypTen provider、通信、运算和等待，不等于纯在线时间；最后的 classifier LayerNorm 也有准备事件，但不属于某个 block。启用详细计时会增加 rank 0 GPU 同步、改变时间和等待行为，不能作为无插桩的性能基准；心跳只能说明日志线程存活，不能单独证明计算有进度。

默认秘密前向不执行中间值范围检查，与已有的推理基准口径一致；`run.json` 和 `report.json` 会记录该设置。需要诊断时加入 `--check-forward-ranges`，检查前向中的输入、RSA 分数和分母等秘密值，并在检测到越界时中止；这会增加三方交互开销。全秘密 Adam 的状态范围检查始终执行。关闭前向检查不代表训练过程中一定没有越界；应结合诊断运行、loss 和明文对照评估数值行为。

为加快测试，可在三方命令中加入 `--ring3-truncation-preprocessing trusted-dealer`。本机独立进程根据公开的张量形状和截断位数，生成全新的随机截断关联材料并分发三方份额；每次请求均生成新材料，不复用。此选项替代 Ring3 概率截断材料的分布式生成，RSA、LayerNorm、GeLU 的其他准备工作以及 CrypTen provider 和 Adam 仍会产生开销。`ring3_prepare_seconds_rank0` 包含可信进程生成及传输等待，不能直接视为在线时间；对比运行时应使用相同的日志设置。

这是本机可信预处理模拟，使用 NumPy 随机数与本地进程通信，没有经过密码学安全验证，也不代表独立机器上的安全部署。默认 `distributed` 模式继续使用原有实现；`run.json`、`report.json` 和启动事件会记录所选模式。

需要记录每步开销时可在全秘密三方训练命令中加入 `--benchmark-step`。每个 `training_step` 的 `benchmark` 包含 `forward`、`backward`、`optimizer`、`total`；FoT 的前向包含输入分享、两次扰动前向、损失和差值分享，反向为零，评估不计入。各阶段提供 `online_estimate_seconds` 和 `online_communication_bytes`，并保存每方原始计数。时间为即时执行时逐方扣除已计时的 CrypTen provider 与 Ring3 准备，再对各方取最大值；`total` 先在每方相加再取最大值。同步插桩改变运行时，局部随机生成和未计时的预处理仍在估计值内，因此这些数字不是严格预计算后的纯在线时间。

通信量按 CrypTen 双向计数的各方总和除以二，加上 Ring3 的各方发送载荷；计数口径为协议载荷估算，不包含 `torch.distributed` 直接发送的公共控制元数据、可信预处理进程 IPC 或指标收集，不能当作抓包测量。每次只使用新材料。基线的实现、数值正确性和统一统计口径需单独验证后才能把其数值与 FoT 放在同一个结果表中。

### 三种微调方法的每步开销

`training.benchmark_first_order` 提供标准 Hugging Face ViT 的三方 SFT 与 PT 基线：SFT 更新完整模型，PT 冻结模型、对所有 patch 的加性提示求梯度。三方均持有模型的秘密份额；当前数据所有者接收自己的 logits，用本地标签计算 logits 梯度，再秘密分享梯度供三方反向传播。两者使用 Q20 CrypTen、缩放 Softmax、CrypTen LayerNorm、CrypTen erf 近似 GeLU 和秘密 Adam。可用 `--gelu-method chebyshev16` 选择先前的实现。FoT 使用现有 Q17 hAttention、秩 4 加性提示和秘密 Adam。不同模型的参数量、输入尺寸、定点精度和算子都可能不同；因此这是端到端方案比较，不能将差异仅归因于零阶优化。

基线每次运行先由 P0 对自己的首个 batch 做明文参考与三方 logits 对照，超过 `--max-audit-error` 默认停止。`--cost-only` 允许审计或秘密 Adam 数值检查失败后继续，相关失败仍记录，结果状态为 `benchmark_completed_cost_only_not_numerically_validated`，仅供开销诊断；非有限的本地 logits 或梯度等无法继续的错误仍会终止。其他方只获得审计通过/失败标记；具体误差保存在 P0 的 `party-0/numerical-audit.json`，同机目录隔离只用于实验组织。此检查只覆盖一个 batch 和前向；完整教师模型的梯度、Adam 边界和多样本行为仍需验证。`first-name` 标签映射沿用现有教师设置，其中存在需要核实的类别对应关系。

开销诊断可以同时指定 `--cost-only --skip-numerical-audit`，跳过启动时的明文和安全前向对照；`report.json` 中的 `audit.skipped` 为 `true`，`audit.passed` 为 `null`，不生成审计误差文件。Adam 范围检查仍会执行并记录失败；实际步骤中无法继续的错误仍会终止。FoT 训练程序不受此选项影响。

排查优化器耗时可加 `--optimizer-progress`：按参数打包、梯度和方差检查、平方根及倒数的各轮迭代等输出 `optimizer_stage_started` 和 `optimizer_stage_complete`，心跳也显示当前阶段。阶段时间是 rank 0 同步后的墙钟时间，包含预处理和等待；额外同步与日志会改变性能，因此应使用单独运行排查，正式计时去掉该选项。该选项只影响新启动的进程。

与 FoT 对齐模型规模可指定 `--model-source standard-vitlite-random --mpcvit-root "$PWD/mpcvit"`。该模式使用 `vit_9_12_64` 的随机初始化参数，并以标准 Softmax 替换所有注意力头，输入和预处理与 FoT 相同：9 层、12 头、192 维、64×64 图像和 257 token，MLP 宽度为 384。模型种子由 `--model-seed` 设置；类别索引直接使用清单中的标签。随机模型只用于开销测量，不代表预训练分类准确率。PT 更新 256×192 个加性提示参数，FoT 更新 4 个子空间参数；两者的优化器、算子和数值精度也有区别。

设置 `DATA`、`SPLIT`、`CHECKPOINT`、`FIXED` 后，下面的命令每种方法做 4 个更新步骤，汇总时跳过第 1 步作为预热。首次运行建议将各命令的 `--steps` 改为 `1` 并使用独立输出目录，确认显存和完成情况后再进行正式测量。单个 SFT 步骤的全模型梯度与 Adam 状态可能占用大量显存；当前只在公开小模型上验证过三方流程。

```bash
ROOT="$HOME/home/tmp/zo-step-benchmark-v1"
for method in pt sft; do
  CUDA_VISIBLE_DEVICES=1,2,3 timeout 21600 \
  python -m training.benchmark_first_order \
    --method "$method" --device cuda \
    --model-source standard-vitlite-random --mpcvit-root "$PWD/mpcvit" \
    --dataset-roots "$DATA" "$DATA" "$DATA" \
    --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
    --gelu-method erf --cost-only --skip-numerical-audit --batch-size 1 --steps 4 --fixed-batch \
    --output "$ROOT/$method" || break
done

CUDA_VISIBLE_DEVICES=1,2,3 timeout 21600 \
python -m training.run --backend 3pc --mode secret --device cuda \
  --checkpoint "$CHECKPOINT" --fixed-student "$FIXED" --mpcvit-root "$PWD/mpcvit" \
  --dataset-roots "$DATA" "$DATA" "$DATA" \
  --manifests "$SPLIT/party-0.json" "$SPLIT/party-1.json" "$SPLIT/party-2.json" \
  --ring3-truncation-preprocessing trusted-dealer --benchmark-step \
  --update-schedule round-robin --rank 4 --sigma 0.1 --lr 0.03 --epsilon 0.01 \
  --batch-size 1 --steps 4 --fixed-batch --eval-every 4 --min-loss-drop 0 \
  --diagnostic-loss --output "$ROOT/fot"

python -m training.summarize_step_benchmark \
  --report "MPC-SFT=$ROOT/sft/report.json" \
  --report "MPC-PT=$ROOT/pt/report.json" \
  --report "MPC-FoT=$ROOT/fot/report.json" \
  --skip 1 --output "$ROOT/summary"
```

每次运行必须使用新输出目录。结果表是同步插桩的在线时间估计，记录的网络为实际运行时设置的网络；默认单机 Gloo 不等于自动配置 LAN/WAN。正式报告时应使用相同网络环境、batch size、采样方式和预热/重复次数，并记录不同方法的输入尺寸、token 数、模型参数量与优化器设置。

若需要可信模式，将 `--mode secret` 改成 `--mode trusted` 并更换输出目录。

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

上例是调用形式，不是推荐用 12 张图像评估模型效用。应先用更大的新清单替换 `$SPLIT`，例如 `--classes 200 --per-party-per-class 10 --eval-per-class 5`。轮流模式每方依次出现一次，`--weights` 只影响诊断 loss 的聚合，不能改变轮流频率；不同数据量时不会自动按样本量采样。聚合模式可用 `--weights n0 n1 n2` 指定训练与诊断权重，程序归一化权重。

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
- 当前循环没有安全反向传播，骨干参数不更新；全秘密模式不公开 delta、梯度或矩状态。可信模式 T 获得当前 step 的 delta：轮流模式为单方 delta，聚合模式为聚合 delta。
- 范围检查公开一个失败位，是调试协议的额外可见信息；Ruler 的盲化打开和底层预处理 provider 的安全性仍须独立论证。
- 一次小模型 loss 下降不等于真实模型训练通过；必须在服务器执行第 4 节检查。明文效用实验也不能替代私有数据的最终验证。
- 三方输出目录只供本地复现。T 的 `trusted-service/state.pt` 包含明文提示与 Adam 状态，应只由 T 持有；`party-i/prompt-share.pt` 为第 i 方份额，不应集中公开。
