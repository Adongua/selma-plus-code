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

该入口只测自研 sLayerNorm 方法。公开平方根暂时回到 CPU 计算，因此 GPU 结果首先用于
功能验证，之后再单独优化这一数据搬运点；CrypTen GPU baseline 尚未纳入同一入口。

所有阶段的原始数据写入 `comparison.json`；自动生成 `summary.md`。普通 normal/uniform 输入最大绝对误差要求不超过 0.005，否则测试退出失败；边界输入如实记录是否达到该阈值。
