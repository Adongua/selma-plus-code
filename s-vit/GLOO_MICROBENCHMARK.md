# Gloo/CUDA 算子微基准

批量实现现已统一在 `run_microbenchmarks`。运行 NCCL 完整版请使用
[`NCCL_MICROBENCHMARK.md`](NCCL_MICROBENCHMARK.md)；本文件中的命令仍运行 Gloo。

从仓库根目录启动，使用包含 PyTorch、CrypTen 依赖和三张可用 CUDA GPU 的环境：

```bash
export PYTHONPATH="$PWD/s-vit:$PWD/CrypTen-main/benchmarks"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2

python -m run_gloo_microbenchmarks --operator both \
  --repeats 3 --output-root "$HOME/home/tmp/gloo-paper-v1"
```

`--operator` 可选 `sgelu`、`slayernorm` 或 `both`。每个 shape 独立运行，每组一次预热
加三次测量；LayerNorm 默认运行 normal 和 uniform，可使用 `--distributions normal` 只运行
normal。实验按顺序运行，可能需要较长时间。若任何子任务失败，脚本停止并保留已完成的
`comparison.json`；为避免覆盖，再运行时使用新的 output root。

GeLU：shape 为 `1x197 1x257 1x577 8x257`，方法为 sGeLU、CrypTen-erf、
masked-Fourier 工程参考、PUMA 3PC 适配和 SEAF 3PC 适配。LayerNorm：shape 为
`1x197x256 1x257x256 1x577x256 8x257x256`，方法为 CrypTen-native、SecFormer、
PUMA、SPRINT、Newton engineering 和 sLayerNorm；不包含 SHAFT。

每个方法/shape 的原始记录写入 `sgelu/<shape>/comparison.json` 或
`slayernorm/<shape>/comparison.json`。完成后生成：

- `sgelu/sgelu_microbenchmark_normal.md`
- `slayernorm/slayernorm_microbenchmark_normal.md`
- `slayernorm/slayernorm_microbenchmark_uniform.md`（如果测量了 uniform）

表格包含 Markdown 和 LaTeX，按非预热重复次数求平均；通信列默认使用各报告的
`system_bytes / 1024`（KiB）。此前 sGeLU 文档曾使用 `sent_bytes_sum / 1024`，两者口径
不同。需要旧口径时可用 `python -m microbenchmark_table --comm-metric sent_bytes_sum ...`
从同一批 JSON 另行生成表格。运行 `python -m microbenchmark_table --help` 查看参数。

GPU Ring3 的 Gloo 通信需将本地份额暂存到 CPU，拷贝计入测量；CrypTen 使用其自己的
Gloo/CUDA 通信路径。两者的离线材料计时边界也不同，且文献 baseline 是标注的适配版。
`executed_not_security_validated` 不是正确性或安全证明：必须检查各方法数值误差，
尤其 CrypTen-erf 可能严重发散。表格只加粗原始最小值，不表示精度合格的 SOTA。
新 Gloo/CUDA 路径尚需服务器实际验证，不应将其与旧 NCCL 数据混为一组。
