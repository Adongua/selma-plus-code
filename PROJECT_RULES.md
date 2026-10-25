# Selma Plus 项目规则

本文件记录本项目在本地开发和远程服务器复现时的目录约定。

## 目录约定

### 代码

所有项目代码、配置文件、脚本和必要的文档统一存放在：

```text
/Users/adongua/Documents/PythonProject/selma-plus/selma-plus-code
```

远程服务器使用同样的项目目录结构；代码通过 GitHub 仓库同步。

### 数据集

本地数据集统一存放在：

```text
/Users/adongua/Documents/PythonProject/selma-plus/datasets
```

数据集默认不放入 Git 仓库，也不提交到 GitHub。远程服务器使用对应的服务器数据目录。

### 模型权重

本地模型权重统一存放在：

```text
/Users/adongua/Documents/PythonProject/selma-plus/models
```

模型权重默认不放入 Git 仓库，也不提交到 GitHub。远程服务器使用对应的服务器模型目录。

### 临时输出

本地运行测试、脚本和实验产生的临时文件、日志、缓存和中间结果统一存放在：

```text
/Users/adongua/Documents/PythonProject/selma-plus/tmp
```

除非用户明确要求，否则不要把临时输出写入代码仓库目录。

## 开发要求

1. 代码中不要硬编码只适用于单台机器的路径；优先使用配置文件、命令行参数或环境变量传入数据集、模型和输出目录。
2. 默认使用上述目录作为本地路径；在远程服务器上使用对应的服务器路径。
3. 修改代码前先确认输入数据、模型权重和输出目录的位置，避免误读或覆盖用户文件。
4. 测试和实验产生的文件优先写入 `tmp`，不要污染 Git 工作区。
5. 大型数据集、模型权重和实验结果不应直接提交到 Git；确需版本管理时，先评估 Git LFS 或独立对象存储方案。
