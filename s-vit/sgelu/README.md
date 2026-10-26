# GeLU v2 算子复现：第一阶段

来源：仓库父目录的《基于双尺度平滑近似的安全GeLU计算方法-v2.md》、
《面向三方安全计算的高效视觉Transformer推理与微调方法_技术初稿.md》§3.5、附录 D/E。
冲突时以 GeLU v2 为准。当前范围是独立算子及 CrypTen 份额适配，尚未接入完整 ViT 图。

## 实现内容

- `sgelu.py`：G1/G2 浮点参考、`SmoothGeLU` 模块、公开激活的逐层尺度校准。
- `wide.py`：96-bit 加减乘、4×24-bit limb、3×32-bit 紧凑通信、公开整数平方根。
- `runtime.py`：CPU/Gloo 三进程 64/96-bit 加法分享、乘法、随机刷新。
- `runtime_cuda.py`：实验性 CUDA/NCCL 三方运行时；每个参与方绑定一张 GPU。
- `protocol.py`：联合掩码提升、概率截断、双分支 Ruler-G2 数值协议。
- `bridge.py`：Q17 `MPCTensor` 输入输出，显式 `prepare()`，仅前向、不支持反向传播。
- `validate.py`：三进程原语验收、一次预热和三次正式算子测量、JSON 报告。
- `calibrate.py`：从公开 pre-GeLU 激活 NPZ 逐层拟合，输出 JSON。
- `baseline.py` / `benchmark.py`：CrypTen 原生 `erf`-GeLU、MPCFormer-Quad 与 sGeLU 对比。

G2(x; μ) = x/2 + x²/2 · [4/(3√(x²+μ²)) − 1/(3√(x²+4μ²))]。
μ₀ = (7/6)√(π/2) ≈ 1.4622。公开校准最小化相对精确 GeLU 的经验 MSE；
在对数尺度网格上定位，再以黄金分割局部细化，并保留 μ₀ 作为候选。
拟合是有限区间内的数值搜索，不保证全局最优，也不把初稿旧参数当成实际校准结果。

## 与初稿参考代码的差异

1. **独立尺度**：两个分支打包为 `(2, *shape)`，每个元素与调用使用独立一次性尺度。
   初稿 E.6 两分支共用 `a`；v2 明确禁止这种复用。
2. **安全随机流原语**：成对零分享采用操作系统随机种子及 SHAKE256 流，不使用 `torch.Generator`
   生成协议秘密。均匀 29-bit 尺度由三方随机比特秘密 XOR 得到，再加 2²⁹。
3. **联合提升掩码**：三方分别贡献 60-bit 随机数，64/96-bit 中分享同一个掩码。
4. **批量分支**：先计算一次 Q16 共享平方，同时构造两个分母；份额提升、宽环乘法及打开批量执行。
   复用共享平方计算结果是允许的，独立截断/盲化随机材料不能复用。
5. **一次性材料**：材料绑定运行时、形状和公开 μ，开始求值即标记已消费；失败也不允许再使用。
6. **公开倒数**：用整数算术计算 Round(2⁵⁶/w)，恰好半整数时向上取整。

这些差异会改变初稿的实现级阶段数和性能。不能直接套用初稿的 270 阶段、1.508 秒或 5.90×。

## 正确性范围与安全结论

CPU/Gloo 路径要求 CPU int64；CUDA/NCCL 路径使用 CUDA int64。两者都要求恰好三方、Q17 输入输出，公开承诺 `|x| ≤ 8`、`0.25 ≤ μ ≤ 4`。
分母采用 Q16，倒平方根采用 Q24。盲化乘积小于 2⁸⁴，概率截断的重建输入必须在
`[-2⁶², 2⁶²)`。公开 isqrt 对超界值报错。秘密输入的范围由调用者保证；实现不会通过打开
原始秘密输入来检查范围，也不会静默裁剪。更大真实激活范围需要重新分析量化、范围和误差。

**目前 Ruler 打开步骤不能作为已验证的隐私安全协议使用。** 数值测试通过只证明正确性。
附录指定精确整数公开量 `Y=a²V`（无模溢出），存在以下可验证泄露：

- 令 ν₂(n) 为整数 n 含因子 2 的次数，则 `ν₂(Y) mod 2 = ν₂(V) mod 2`，与 a 无关。
  在允许域内，取 μ=1、Q17 输入 x=0 或 x=1，第一分支 V 分别为 2¹⁶ 和 2¹⁷；
  观察一个 Y 即可完美区分这两个候选输入。因此独立 a 也不足以支撑一般的 37-bit 统计隐藏声明。
- a ∈ [2²⁹,2³⁰) 还给出了 V 的范围信息。增大 a 的位数不自动消除乘法打开的泄露。
- 两个分支若共用 a，且共享平方编码一致，则 `Y₂−Y₁=3μ²_Q16 a²`
  （这里选择常数恰好可编码的 μ），能够恢复 a² 及平方值。独立分支尺度修复了这项额外泄露，
  但没有修复上一项。

测试包含这些反例。保留 Ruler 路径用于合成输入上的算法/工程复现，并显式标注 experimental。
用真实私有数据部署或在论文中声明统计安全前，需要修订打开协议并给出相应证明；
也可另行实现全秘密倒平方根基线，与 Ruler 的性能分开报告。
传输认证、主动攻击防护和跨进程重启的材料持久化不在此原型范围。

## 运行方式

在代码仓库根目录执行；激活自己的 PyTorch 环境。路径按本地/服务器共同布局设置：

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$(cd .. && pwd)/tmp"
export TORCH_HOME="$(cd .. && pwd)/models/torch-cache"
export OMP_NUM_THREADS=1
export GLOO_SOCKET_IFNAME=lo  # Linux；macOS 使用 lo0

python -m unittest discover -s s-vit/sgelu/tests -v
python -m sgelu.validate --output ../tmp/gelu-v2
python -m sgelu.benchmark --output ../tmp/gelu-baseline
```

GPU 功能验证使用独立入口，要求至少三张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m sgelu.benchmark_cuda \
  --count 17 --batch 1 --repeats 1 --output ../tmp/sgelu-cuda-smoke
```

CUDA benchmark 默认使用实测更快的 P2P 重构，并将平方截断和 `x/2` 截断合并打开，
在线 phase 从旧实现的 17 降至 15。使用 `--reveal-mode all_reduce` 或
`--reveal-mode all_gather` 可测试单轮 NCCL 集合重构；两种模式理论上为 9 phase，
但 `all_gather` 在当前服务器上实际更慢。两种模式的
通信字段都按三方环实际传输的等价 payload 计数，因此可直接比较。

使用 `--profile-online` 可记录平方/截断、尺度提升、宽乘法/平方根、逆根截断、
括号截断和最终乘法截断的同步分阶段时间。该选项会插入 CUDA 同步，只用于定位
瓶颈，不能用于正式性能表。

CUDA benchmark 现在同时测量同一 CUDA Ring3 上的 Quad 固定点移植，报告中对应
`quad_offline`、`quad_online`、`quad_output_reconstruction` 和 `quad_vs_torch_gelu`。
该 Quad 是与 CrypTen Quad 数学公式一致的工程 baseline，不代表 CrypTen GPU
实现或相同安全证明；CrypTen-erf 和 Fourier GPU baseline 仍未接入。

这是 CPU/Gloo 原型的工程移植，仍带有 Ruler 打开步骤的隐私限制。CUDA v2 的
公开 96-bit 整数平方根、精确修正与随机词展开均留在 GPU；初始化时仅交换一次随机
种子。GPU 随机词使用 PyTorch Philox，仅用于性能实验，不是经审查的密码学 PRG。
CrypTen-erf、Quad 和 Fourier 的 GPU baseline 尚未纳入该入口。

`validate` 默认运行 10,000 个三方原语样本、257 个合成激活、1 次预热和 3 次正式重复。
测试模块另行核验 10,000 对 96-bit 算术和 10,000 个随机公开整数根，并覆盖边界。
输出包含输入分享、离线准备、在线求值、输出打开的独立耗时。
通信统计是三方发送载荷之和；phase 是实现的 exchange 次数，不是 WAN RTT。
计时不包含 barrier、统计汇总和最终误差计算。CPU/Gloo 本地结果不是服务器论文测量。

`benchmark` 在相同三方进程、输入形状和 Q17 输入下比较 sGeLU、MPCFormer-Quad 与 CrypTen 原生
`0.5*x*(1+erf(x/sqrt(2)))`。CrypTen 的默认 `erf_iterations=8` 是 Taylor 展开，
不是专门优化的 GeLU 协议；因此报告同时记录开销和误差，不能只看速度。

本次本地环境放在 `../tmp/gelu-venv`；服务器应创建独立环境，不能复制 macOS 虚拟环境。
本算子仅需 PyTorch、NumPy；CrypTen bridge 需要原项目的附加依赖。
`requirements.txt` 和 `requirements-bridge.txt` 分别记录本次验证使用的核心依赖与
CrypTen 导入依赖版本，不是完整 MPCViT 环境或跨平台锁文件。请在独立环境中安装，
不要直接覆盖服务器现有训练环境；Linux CPU 测试可先安装 PyTorch 对应 CPU 构建。

如本地 CrypTen 已可导入，可额外测试真实 MPCTensor 份额边界：

```bash
python -m sgelu.validate --crypten --output ../tmp/gelu-v2-bridge
```

这个测试先初始化 CrypTen，再共用其 Gloo group，验证 MPCTensor 的 Q17 share 往返；
输入通过 `crypten.cryptensor` 分享，输出通过 `get_plain_text` 重建（仅合成测试允许）；
不代表整个 CrypTen 图导出通过。
应用集成时先初始化 CrypTen 三方进程组，再建立 Ring3/Ring3Wide，使用同一个 group。

```python
bridge = ExperimentalG2Bridge(base, wide, mu=1.3)
bridge.prepare(tuple(x_encrypted.shape))  # 单独离线计时；只接受公开形状与参数
y_encrypted = bridge(x_encrypted)        # 单次在线消费；不会在这里打开输出
```

校准输入是 NPZ：每个数组的键为稳定层名，值为公开数据在该层 **GeLU 之前** 的激活。
不提供虚构的九层校准参数，不将合成随机样本的拟合称为模型校准。

```bash
python -m sgelu.calibrate ../tmp/public_gelu_inputs.npz --output ../tmp/gelu-v2/calibration.json
```

## 验收与下一步

2026-10-24 本地验证（macOS ARM64、Python 3.12.14、PyTorch 2.5.1、CPU/Gloo）：

- 11 项单元测试通过，覆盖浮点公式、校准、96-bit 算术、整数根、隐私反例及校准 CLI。
- 三方原语 10,000 个样本通过，覆盖三种输入拥有方、环溢出、提升与概率截断边界。
- 257 个合成激活、1 次预热和 3 次正式重复：协议对浮点 G2 的最大误差约 `8.27e-5`。
- 4,097 个合成激活及实际 CrypTen MPCTensor 接口通过：上述最大误差约 `8.44e-5`。
- 最终回归进一步验证 CrypTen 自身加密→G2→解密链路；4,097 点、三次正式重复，
  协议对浮点 G2 的最大误差约 `9.30e-5`（使用了新的一次性随机材料）。
- 257 点未校准 G2（μ₀）对精确 GeLU 的最大误差约 `0.068424`，不是协议量化误差。

新增的 CrypTen-erf 对比报告（257 点、1 次预热、3 次正式重复）显示：

- sGeLU 在线约 `2.4–2.8 ms`，三方发送量合计约 `191 KB`；离线准备约 `15.6–16.3 ms`。
- CrypTen-erf 在线约 `36.9–39.0 ms`，CrypTen 统计的三方字节合计约 `1.36 MB`。
- 按统一的整体系统通信口径（CrypTen 三方合计除以 2），只比较在线阶段，sGeLU 约快 `15×`、
  通信约减少 `72%`；若直接比较报告中的原始三方载荷合计，则对应约 `86%`，但这不是同一口径。
  若把 sGeLU 离线准备也算入，
  当前本地样本约为 `2.1×` 总耗时优势。该比值只适用于本机和这个合成形状。
- CrypTen-erf 在 `[-8,8]` 上最大误差约 `1.07×10^9`，说明默认 Taylor 展开已经数值发散，
  不能把它当作精度合格的 baseline；它目前只适合作为 CrypTen 原生实现的工程开销参考。
- MPCFormer-Quad 使用 `0.125*x^2 + 0.25*x + 0.5`，只需要一次秘密平方；它是有效的
  MPC 开销/近似 baseline，但原论文依靠替换模型后的训练或知识蒸馏弥补精度损失，不能
  直接把未重训的 Quad 输出当作 sGeLU 的等精度对照。

在同一 `(1,257)` 合成输入上的 MPCFormer-Quad 测试约为：在线 `1.7–1.9 ms`、三方发送量
合计 `61.7 KB`、相对精确 GeLU 最大误差 `6.50`。这说明 Quad 是很好的低成本 MPC 参照，
但不能直接代表经过 MPCFormer 重新训练后的模型精度。

报告位于 `../tmp/gelu-baseline/comparison.json`。CrypTen 文档约定其通信统计的整体系统量
需要将三方字节合计除以 2；sGeLU 的自定义环统计则按三方发送载荷合计记录，阅读报告时需保持
这个口径差异。后续再加入 SecFormer 的分段 Fourier baseline，作为更高精度的协议对照。

本次原始报告分别位于 `../tmp/gelu-v2/three_party.json` 和
`../tmp/gelu-v2-bridge/three_party.json`，最终回归为 `../tmp/gelu-v2-final/three_party.json`，
不随 Git 同步。随机材料每次更新，数值误差和
耗时会波动。这里记录的是算子功能结果，不是隐私证明、完整模型复现或服务器性能结果。

报告分别列出：协议输出对浮点 G2（量化输入）的误差、浮点 G2 对精确 GeLU 的近似误差、
协议输出对精确 GeLU 的总误差。`0.005` 门槛用于**协议数值误差**，不能用于宣称 G2 的全域
GeLU 近似误差均小于这个数值。

后续：核定安全打开方案；在公开 Tiny-ImageNet 数据与已核对权重上采集 9 层 pre-GeLU 激活并校准；
解耦 MPCViT 模型导入时的参数解析；接入单 block 和完整图；测量精度、通信、时间及消融。
本阶段没有修改旧 CrypTen 的全局算子映射或 MPCViT 注意力行为。
