# NCCL/CUDA 算子微基准

从仓库根目录启动。使用项目的 CrypTen 环境，并确认三张 GPU 当前可用：

```bash
cd "$HOME/home/selma-plus-code"
conda activate s-crypten
git pull --ff-only origin main
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

python -m run_microbenchmarks --backend nccl --operator both \
  --repeats 3 --case-timeout 1800 \
  --output-root "$HOME/home/tmp/nccl-paper-v1"
```

若先想确认 NCCL 初始化和单算子能跑通，可先执行小型 smoke：

```bash
python -m sgelu.benchmark_crypten_cuda --backend nccl \
  --shape 1x17 --repeats 1 --methods sgelu erf \
  --output "$HOME/home/tmp/nccl-sgelu-smoke"

python -m slayernorm.benchmark_crypten_cuda --backend nccl \
  --shapes 1x17x32 --distributions normal --repeats 1 \
  --methods fused_ruler secformer_goldschmidt unfused_newton \
  --output "$HOME/home/tmp/nccl-slayernorm-smoke"
```

完整实验按顺序运行，每个 shape 独立启动三方进程。GeLU 测四个 shape
`1x197 1x257 1x577 8x257` 和 sGeLU、CrypTen-erf、masked Fourier、PUMA、SEAF
适配；LayerNorm 测 `1x197x256 1x257x256 1x577x256 8x257x256` 和 CrypTen-native、
SecFormer、PUMA、SPRINT、Newton、sLayerNorm。LayerNorm 默认测 normal 与 uniform。

输出根目录下保留每个 case 的 `comparison.json`，全部完成后生成含 Markdown 和 LaTeX
的表格：

- `sgelu/sgelu_microbenchmark_normal.md`
- `slayernorm/slayernorm_microbenchmark_normal.md`
- `slayernorm/slayernorm_microbenchmark_uniform.md`

每个 case 默认最多运行 1800 秒。超时会终止该 case 的进程组，保留之前已完成的 JSON，
并停止后续实验。更换 `--output-root` 后可从头重跑；有已有 JSON 时 runner 会拒绝覆盖。

表格默认以 `system_bytes / 1024` 报通信量，并从非 warmup 记录计算均值。它与旧
sGeLU 表中的 `sent_bytes_sum` 口径不同。CrypTen-erf 的误差可能很大，必须先查看误差列，
不能单凭时间认定性能更优。masked Fourier、PUMA 和 SEAF 均为适配实现；sGeLU Ruler
相关 masked opening 尚未完成隐私验证，表格不是安全性证明。
