# BlackVIP 官方 SVHN 对照实验

目标为复现 CVPR 2023 表 3 在 SVHN 上的零样本 CLIP（18.1%）与 BlackVIP（44.3%，16-shot、三个种子）。入口调用独立上游仓库的 train.py。模型、优化器、评估器和数据加载均来自上游提交 66606fbc57dd9b8bfbe4d7296d0fd5662d6e83d2。[论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Oh_BlackVIP_Black-Box_Visual_Prompting_for_Robust_Transfer_Learning_CVPR_2023_paper.pdf) · [作者配置](https://github.com/changdaeoh/BlackVIP/blob/66606fbc57dd9b8bfbe4d7296d0fd5662d6e83d2/docs/configuration.md)

## 环境和代码

服务器项目根目录内另行克隆上游仓库。已有该版本的本地副本也可通过 BLACKVIP_DIR 指定。下面的环境采用作者的 Python、PyTorch、torchvision 和 Transformers 版本。安装 safetensors 以读取项目已有 MAE 文件。my_dassl 通过 torch.utils.tensorboard 写入日志，使用兼容 Python 3.8 的稳定版 TensorBoard 代替其安装清单中的 tb-nightly。

```bash
cd "$HOME/home/selma-plus-code"
git clone https://github.com/changdaeoh/BlackVIP.git BlackVIP
git -C BlackVIP checkout --detach 66606fbc57dd9b8bfbe4d7296d0fd5662d6e83d2
conda create -y -n blackvip-original python=3.8
conda activate blackvip-original
conda install -y pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
sed 's/^tb-nightly$/tensorboard==2.13.0/' BlackVIP/my_dassl/requirements.txt > "$CONDA_PREFIX/blackvip-dassl-requirements.txt"
python -m pip install -r "$CONDA_PREFIX/blackvip-dassl-requirements.txt" -r BlackVIP/requirements.txt 'safetensors>=0.3,<0.5' pandas
python BlackVIP/train.py --help
```

## 数据和权重

- 下载作者的 SVHN 划分，将文件存到服务器的 `$HOME/home/datasets/svhn/split_SVHN.json`。README 里文件名含有 mlai，加载器实际查找上述文件名。建议使用独立目录或首次使用的 SVHN 数据目录，避免已有的 split_fewshot 缓存来自其他划分。[作者划分](https://drive.google.com/file/d/1dnjnMX-sr7FClb6EUywc-tNx6e1YaGpu/view)
- 准备 torchvision 的 train_32x32.mat 和 test_32x32.mat。可以在本机下载后放到服务器 `$HOME/home/datasets/` 下。prepare 调用作者的 JPEG 导出函数，并核对划分中的路径及 JPEG 总数。torchvision 会校验已有 mat 文件。[SVHN 数据集](http://ufldl.stanford.edu/housenumbers/)
- 将 OpenAI CLIP 权重放到服务器 `$HOME/.cache/clip/ViT-B-16.pt`，SHA256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`。这是原 loader 指定的 TorchScript 权重。服务器联网时 loader 可自行下载。[CLIP 权重](https://openaipublic.blob.core.windows.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)
- MAE 使用现有 `$HOME/home/models/vit-mae-base`。启动脚本会在独立工作目录将其链接到作者代码读取的 facebook/vit-mae-base。可通过 MAE_DIR 指定其他绝对路径。

## 在服务器运行

```bash
cd "$HOME/home/selma-plus-code"
conda activate blackvip-original
export CUDA_VISIBLE_DEVICES=0
RUN=s-vit/fot_transfer/reproduce_blackvip_svhn.sh
bash "$RUN" prepare
bash "$RUN" zero
bash "$RUN" smoke
bash "$RUN" train
```

zero 在完整测试集上评估未添加提示的 CLIP。smoke 用 seed 1 运行一个 epoch 并验证保存、重新加载和测试流程。train 分别使用种子 1、2、3，每个种子从头训练 5,000 个 epoch，再加载该种子的最后一个 checkpoint，测试 26,032 张图。每类 16 张训练图，一共有 160 张，batch size 为 128，作者 loader 使用 drop_last，因此每轮 1 步。每步 5 个双边方向，每种子 50,000 次训练分类器前向。可用 `bash "$RUN" train 2` 单独启动指定种子。

默认输出目录为 `$HOME/home/tmp/blackvip-original-svhn`，可设置绝对路径 BLACKVIP_OUT。上游日志输出到各次运行目录下的 log.txt。统计最后一次测试的正确数，保留三个种子的结果并计算平均值。

```bash
python - <<'PY'
from pathlib import Path
import os, re, statistics
root = Path(os.environ.get("BLACKVIP_OUT", str(Path.home() / "home/tmp/blackvip-original-svhn")))
values = []
for name in ["zero", "train/seed1", "train/seed2", "train/seed3"]:
    log = (root / name / "log.txt").read_text()
    total, correct = re.findall(r"\* total: ([\d,]+)\s+\* correct: ([\d,]+)", log)[-1]
    total, correct = int(total.replace(",", "")), int(correct.replace(",", ""))
    assert total == 26032
    acc = 100 * correct / total
    print(name, f"{acc:.3f}%")
    if name.startswith("train/"):
        values.append(acc)
print("BlackVIP mean", f"{statistics.mean(values):.3f}%")
PY
```
