# sLayerNorm

依据《基于中心化二次型融合的安全LayerNorm计算方法-v2.md》和技术初稿 E.7 实现的实验性算子。仅做 micro benchmark；最后一维是归一化隐藏维度，其余维度为相互独立的 token。

## 实现

- `protocol.py`：Q17 输入/输出、Q20 公开平均系数、Q24 公开尺度倒数、Q34 平方、Q16 方差、Q24 倒平方根；秘密 gamma/beta。
- `reference.py`：实数域缩放恒等式，与 float64 PyTorch LayerNorm 对照。
- `bridge.py`：可选 CrypTen Q17 适配器，无需改动 CrypTen 或 mpcvit。
- `benchmark.py`：本机三方 Gloo，复用 `sgelu.runtime` 的 Ring3 和 96-bit 算术层以及一次性概率截断材料。

三种方法共享输入、仿射参数、定点精度和发送字节计数器：

| 方法 | 方差形成 | 倒平方根 |
|---|---|---|
| fused_ruler | 先求和再一次标量截断 | Ruler |
| unfused_ruler | 各通道平方先截断至 Q17，再归约并转换到 Q16 | 同一 Ruler |
| fused_newton | 与 fused_ruler 相同 | 32 次秘密 Newton 迭代，初值 0.125 |

Newton 使用 `r <- r*(3-v*r*r)/2`，通过先计算 `v*r` 避免先平方大倒数造成中间溢出。固定 32 步覆盖当前小方差范围，不是为普通随机输入调优的最少迭代次数。它是工程对照，不代表 CrypTen 原版或文献 SOTA。

## 数值及安全边界

当前验证域：CPU int64，`|x|<=4`，公开 `1<=s<=4`，秘密仿射参数绝对值不超过 2。参数检查用公开上界验证融合累加不超过截断合法范围；在线不打开输入来检查范围。调用方须满足契约。

Q16 稳定项编码为 `max(1, round(epsilon/s² * 65536))`。默认 `epsilon=1e-5,s=1` 对应有效稳定项 `1/65536`。这是对初稿离散精度的复现，不等同于浮点精确 LayerNorm。近常量输入的误差可能较大；报告同时保留相对请求 epsilon 和有效 epsilon 的误差。概率截断会造成重复运行间的差异。

Ruler 中公开 `Y=a²V`，其有限区间乘法盲化及现有跨环扩展的隐私性未获证明。本实现复现其计算行为，不宣称经过验证的安全协议；联合生成且不复用材料也不消除这一限制。

材料绑定运行时、shape、配置和方法；每次在线执行烧毁材料，禁止跨调用复用。对正式 `(B,T,D)` 输入，均值和方差在每个 `(b,t,:)` 行内计算，未构造中心化矩阵。

## 运行

从代码仓库根目录运行，使用已有安装 PyTorch 的环境：

```bash
export PYTHONPATH="$PWD/s-vit"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export TMPDIR="$(dirname "$PWD")/tmp"
python -m unittest discover -s s-vit/slayernorm/tests -v
python -m slayernorm.benchmark --output ../tmp/slayernorm-micro-v2
python -m slayernorm.benchmark --shapes 2x1 2x17 2x257 --distributions constant near_constant --repeats 3 --output ../tmp/slayernorm-edge-v1
```

默认正式测试为 `(B,T,D)` 的四种 ViT-like shape、normal 与 uniform 两种分布，每组 1 次预热 + 3 次测量，执行顺序轮换。每个阶段记录各方最大耗时、三方发送载荷合计和各方 exchange 次数。阶段包括输入分享、材料准备、在线计算、输出重构；同步和指标聚合不计入。`exchange` 次数是运行时交换阶段，并非跨不同后端通用的协议轮数。

GPU 功能验证使用独立的 `benchmark_cuda.py`，要求至少三张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m slayernorm.benchmark_cuda \
  --shapes 1x17x32 --distributions normal --methods fused_ruler \
  --repeats 1 --output ../tmp/slayernorm-cuda-smoke
```

该入口只测自研 sLayerNorm 方法。CUDA v2 的公开平方根、整数修正和随机词展开均留在
GPU；PyTorch Philox 仅用于性能实验，不是经审查的密码学 PRG。CrypTen GPU baseline
尚未纳入同一入口。

统一 GPU micro benchmark 使用 `benchmark_crypten_cuda.py`。它将自研 sLayerNorm
和 Newton 工程基线通过 Ring3 适配到同一 NCCL 进程组，其余方法使用 CrypTen
MPCTensor：

| CLI 方法 | 时间 | 实现口径 |
|---|---:|---|
| `slayernorm` | 本文 | 融合方差 + Ruler（本文方案） |
| `crypten_native` | 2021 | 顺序 `sqrt -> reciprocal` CrypTen 组合 |
| `secformer` | 2024 | 11 步 Goldschmidt，CrypTen 3PC 适配 |
| `puma` | 2025 | 直接 `InvSqrt` + broadcast，CrypTen 3PC 适配；预印本为 2023 |
| `sprint` | 2026 | `T=100,beta=1000` 分段 `InvSqrt`，CrypTen 3PC 适配 |
| `shaft_adapted_3pc` | 2025 | SHAFT 的 8 次指数初始化、5 次 rSqrt 迭代，3PC 适配 |
| `newton_engineering` | 工程对照 | 逐通道截断方差 + 32 步 Newton，同 Ring3 工程对照 |

除 `crypten_native` 外，文献 baseline 的结果都是在本项目相同 CrypTen 3PC 环境中的
算法移植结果，不是原作者系统的官方性能数字。运行完整的 D=256 测试：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
python -m slayernorm.benchmark_crypten_cuda \
  --shapes 1x197x256 1x257x256 1x577x256 8x257x256 \
  --distributions normal uniform --repeats 3 \
  --methods slayernorm crypten_native secformer puma sprint shaft_adapted_3pc newton_engineering \
  --output ../tmp/slayernorm-crypten-cuda-d256
```

所有阶段的原始数据写入 `comparison.json`；自动生成 `summary.md`。传统
`benchmark.py` 对普通 normal/uniform 输入执行最大绝对误差 `0.005` 的 guard；统一
GPU 入口会如实记录每个 baseline 的误差而不因某个近似误差较大提前终止整批实验。

同一入口也可通过 `--backend gloo` 使用 GPU 计算和 Gloo 通信。Ring3 的 CUDA 份额
经 CPU 缓冲区传输，拷贝时间计入测量；CrypTen 方法使用自身 Gloo/CUDA 通信路径。
报告记录导入路径、backend 和 Q17 精度，不能直接把此路径的运行时间当成 NCCL 结果。
先运行小形状并检查各方法误差：

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2 timeout 600 python -m slayernorm.benchmark_crypten_cuda \
  --backend gloo --shapes 1x17x32 --distributions normal --repeats 1 \
  --methods slayernorm crypten_native --output "$HOME/home/tmp/slayernorm-gloo-smoke"
```
