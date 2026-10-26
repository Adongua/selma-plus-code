# CryptGPU GeLU operator suite

该目录在 vendored `CryptGPU/` 三方 replicated-secret-sharing GPU 框架上统一实现：

- `sgelu`：论文 Ruler-G2 协议，通过 CryptGPU communicator/相关随机流运行；
- `chebyshev16`：`[-8,8]` 上的 16 阶 Chebyshev GeLU，用作技术初稿主 baseline 的同框架重建；
- `puma`：PUMA 分段多项式及 CryptGPU 安全比较；
- `quad`：MPCFormer 二次近似；
- `erf`：CrypTen 风格 Taylor-erf；
- `fourier`：SecFormer masked-Fourier。

CryptGPU 官方代码针对 PyTorch 1.6。当前兼容层目标是先在服务器 PyTorch 2.5/CUDA
环境完成算子 smoke 与 benchmark，不宣称生产安全。默认关闭 `torchcsprng`，输出正确，
但随机数不是密码学安全 PRG；正式安全实验需按上游说明构建 PyTorch/torchcsprng。

不要在现有 `selma-plus` 环境直接执行上游 `pip install -r CryptGPU/requirements.txt`，
因为它会请求降级到 PyTorch 1.6。这里通过 `PYTHONPATH` 直接加载 vendored 源码；算子
测试不依赖上游旧 ONNX 转换器。

从仓库根目录运行，务必让 vendored CryptGPU 的 `crypten` 排在旧 CrypTen 前面：

```bash
PYTHONPATH="$PWD/CryptGPU:$PWD/s-vit" \
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m cryptgpu_gelu.benchmark \
  --batch 1 --count 17 --repeats 1 \
  --methods quad \
  --output "$HOME/home/tmp/cryptgpu-quad-smoke"
```

建议按 `quad -> chebyshev16 -> puma -> erf -> fourier -> sgelu` 逐项 smoke；全部通过后再去掉
`--methods` 运行统一表格。原始记录写入 `comparison.json`，正式重复的均值表写入
`summary.md`。

Transformer 的 `(b,t,d)` 输入可直接用 `--shape`：

```bash
PYTHONPATH="$PWD/CryptGPU:$PWD/s-vit" \
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m cryptgpu_gelu.benchmark \
  --shape 1x197x768 --distribution normal --repeats 3 \
  --methods sgelu chebyshev16 puma quad erf fourier \
  --output "$HOME/home/tmp/cryptgpu-gelu-1x197x768"
```

`chebyshev16` 的阶数和定义域与当前技术初稿目标一致，但原遗失实验的系数文件尚未
恢复，因此报告必须写作“同框架 16 阶重建 baseline”，不能直接把新时间等同于初稿的
`8.896 s`。
