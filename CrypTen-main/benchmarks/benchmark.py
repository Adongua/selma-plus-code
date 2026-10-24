#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate function and model benchmarks

To Run:
$ python benchmark.py

# Only function benchmarks
$ python benchmark.py --only-functions
$ python benchmark.py --only-functions --world-size 2

# Benchmark functions and all models
$ python benchmark.py --advanced-models

# Run benchmarks on GPU
$ python benchmark.py --device cuda
$ python benchmark.py --device cuda --world-size 2

# Run benchmarks on different GPUs for each party
$ python benchmark.py --world-size=2 --multi-gpu

# Save benchmarks to csv
$ python benchmark.py -p ~/Downloads/
"""

import argparse
import functools
import os
import timeit
from collections import namedtuple

import crypten
import crypten.communicator as comm
import numpy as np
import pandas as pd
import torch
from examples import multiprocess_launcher


try:
    from . import data, models
except ImportError:
    # direct import if relative fails
    import data
    import models

Runtime = namedtuple("Runtime", "mid q1 q3")


def time_me(func=None, n_loops=10):
    """Decorator returning average runtime in seconds over n_loops：
    这个装饰器函数可以用于任何函数，用于计算函数的平均运行时间，并返回一个包含运行时间统计信息的对象

    Args:
        func (function): invoked with given args / kwargs（使用给定的args/kwargs调用）
        n_loops (int): number of times to invoke function for timing（调用函数进行计时的次数）

    Returns: tuple of (time in seconds, inner quartile range, function return value).
    """
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

    @functools.wraps(func) # 装饰器，用于将 timing_wrapper 函数的元数据（如名称、文档字符串等）设置为原始函数 func 的元数据，以保留函数的原始信息
    def timing_wrapper(*args, **kwargs): # 装饰函数 timing_wrapper，接受任意数量的位置参数和关键字参数
        return_val = func(*args, **kwargs) # 调用原始函数 func 并记录返回值
        times = [] # 创建一个空列表，用于存储每次函数调用的运行时间
        for _ in range(n_loops): # 循环 n_loops 次，即指定的计时次数
            start = timeit.default_timer() # 记录当前时间作为函数调用的起始时间
            func(*args, **kwargs) # 再次调用原始函数 func，对其进行计时
            times.append(timeit.default_timer() - start) # 计算函数调用的运行时间，并将其添加到 times 列表中
        mid_runtime = np.quantile(times, 0.5) # 计算运行时间列表 times 的中位数
        q1_runtime = np.quantile(times, 0.25) # 计算运行时间列表 times 的下四分位数
        q3_runtime = np.quantile(times, 0.75) # 计算运行时间列表 times 的上四分位数
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime) # 创建一个 Runtime 对象，存储中位数、下四分位数和上四分位数
        return runtime, return_val # 返回一个元组，包含运行时间对象和原始函数的返回值

    return timing_wrapper


class FuncBenchmarks:
    """Benchmarks runtime and error of crypten functions against PyTorch

    Args:
        tensor_size (int or tuple): size of tensor for benchmarking runtimes
    """

    BINARY = ["add", "sub", "mul", "matmul", "gt", "lt", "eq"]

    UNARY = [
        "sigmoid",
        "relu",
        "tanh",
        "exp",
        "log",
        "reciprocal",
        "cos",
        "sin",
        "sum",
        "mean",
        "neg",
    ]

    LAYERS = ["conv2d"]

    # 用于基准测试的常量
    DOMAIN = torch.arange(start=0.01, end=100, step=0.01) # 针对大部分函数
    # for exponential, sin, and cos
    TRUNCATED_DOMAIN = torch.arange(start=0.001, end=10, step=0.001) # 针对指数、三角函数

    def __init__(self, tensor_size=(100, 100), device="cpu"):
        self.device = torch.device(device)
        self.tensor_size = tensor_size

        # dataframe for benchmarks
        self.df = None

    # 后面打表用的
    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Function Benchmarks"

    @staticmethod
    @time_me
    def time_func(x, func, y=None):
        """
        Invokes func as a method of x：根据给定的参数调用相应的函数，并测量其运行时间
        getattr(x,func)<=>x.func
        """
        if y is None: # 一元函数
            return getattr(x, func)() # 将 func 视为 x 的方法，并通过 getattr(x, func) 获取该方法，并调用它。此时返回函数的结果，并使用 time_me 装饰器测量其运行时间

        # 二元函数
        if func in {"conv1d", "conv2d"}: # 检查 func 是否为 "conv1d" 或 "conv2d"
            if torch.is_tensor(x): # x是torch张量
                return getattr(torch.nn.functional, func)(x, y) # 使用 torch.nn.functional 模块中的对应函数，并传递 x 和 y 作为参数
            return getattr(x, func)(y) # 假定 x 是一个对象，并通过 getattr(x, func) 获取该方法，并将 y 作为参数调用它
            # 返回函数的结果，并使用 time_me 装饰器测量其运行时间
            # 需要注意的是，pytorch卷积层的卷积运算实质是调用torch.nn.functional里的conv方法，而crypten卷积层的卷积运算实质则是调用cryptensor里的conv方法

        return getattr(x, func)(y) # 假定 x 是一个对象，并通过 getattr(x, func) 获取该方法，并将 y 作为参数调用它。返回函数的结果，并使用 time_me 装饰器测量其运行时间


    def get_runtimes(self):
        """Returns plain text and crypten runtimes"""
        x, y = (
            torch.rand(self.tensor_size, device=self.device),
            torch.rand(self.tensor_size, device=self.device),
        ) # 明文
        x_enc, y_enc = crypten.cryptensor(x), crypten.cryptensor(y) # 密文

        runtimes, runtimes_enc = [], []

        for func in FuncBenchmarks.UNARY + FuncBenchmarks.BINARY:
            second_operand, second_operand_enc = None, None
            if func in FuncBenchmarks.BINARY:
                second_operand, second_operand_enc = y, y_enc # 二元函数多加一个操作数

            runtime, _ = FuncBenchmarks.time_func(x, func, y=second_operand) # 对明文运算计算运行耗时
            runtimes.append(runtime)

            runtime_enc, _ = FuncBenchmarks.time_func(x_enc, func, y=second_operand_enc) # 对密文运算计算运行耗时
            runtimes_enc.append(runtime_enc)

        # add layer runtimes（追加层的运行时间）
        runtime_layers, runtime_layers_enc = self.get_layer_runtimes()
        runtimes.extend(runtime_layers) # 在list最后追加
        runtimes_enc.extend(runtime_layers_enc)

        return runtimes, runtimes_enc

    def get_layer_runtimes(self):
        """Returns runtimes for layers"""

        runtime_layers, runtime_layers_enc = [], []

        for layer in FuncBenchmarks.LAYERS:
            if layer == "conv1d":
                x, x_enc, y, y_enc = self.random_conv1d_inputs()
            elif layer == "conv2d":
                x, x_enc, y, y_enc = self.random_conv2d_inputs()
            else: # 它只写了测试conv的代码，其他层的还得补
                raise ValueError(f"{layer} not supported")

            runtime, _ = FuncBenchmarks.time_func(x, layer, y=y)
            runtime_enc, _ = FuncBenchmarks.time_func(x_enc, layer, y=y_enc)

            runtime_layers.append(runtime)
            runtime_layers_enc.append(runtime_enc)

        return runtime_layers, runtime_layers_enc

    def random_conv2d_inputs(self):
        """
        Returns random input and weight tensors for 2d convolutions
        生成随机的输入和权重张量，用于进行二维卷积操作
        """
        filter_size = [size // 10 for size in self.tensor_size]
        # 对应torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)中的input和weight
        x_conv2d = torch.rand(1, 1, *self.tensor_size, device=self.device) # 生成一个形状为 (1, 1, *self.tensor_size) 的随机输入张量 x_conv2d，其中*self.tensor_size 表示将 self.tensor_size 中的元素作为单独的参数传递给 rand() 方法
        weight2d = torch.rand(1, 1, *filter_size, device=self.device) # 生成一个形状为 (1, 1, *filter_size) 的随机权重张量 weight2d
        x_conv2d_enc = crypten.cryptensor(x_conv2d) # 加密
        weight2d_enc = crypten.cryptensor(weight2d)
        return x_conv2d, x_conv2d_enc, weight2d, weight2d_enc

    def random_conv1d_inputs(self):
        """Returns random input and weight tensors for 1d convolutions"""
        size = self.tensor_size[0]
        filter_size = size // 10
        (x_conv1d,) = torch.rand(1, 1, size, device=self.device)
        weight1d = torch.rand(1, 1, filter_size, device=self.device)
        x_conv1d_enc = crypten.cryptensor(x_conv1d)
        weight1d_enc = crypten.cryptensor(weight1d)
        return x_conv1d, x_conv1d_enc, weight1d, weight1d_enc

    @staticmethod
    def calc_abs_error(ref, out):
        """Computes total absolute error"""
        ref, out = ref.cpu(), out.cpu()
        if ref.dtype == torch.bool:
            errors = (out != ref).numpy().sum() # 计算不匹配元素数量的总和
            return errors
        errors = torch.abs(out - ref).numpy() # 计算输出张量 out 与参考张量 ref 之间的差的绝对值，得到误差张量
        return errors.sum()

    @staticmethod
    def calc_relative_error(ref, out):
        """Computes average relative error"""
        ref, out = ref.cpu(), out.cpu()
        if ref.dtype == torch.bool:
            errors = (out != ref).numpy().sum() // ref.nelement() # 不匹配元素数量/元素数量=mean
            return errors
        errors = torch.abs((out - ref) / ref)
        # remove inf due to division by tiny numbers
        errors = errors[errors != float("inf")].numpy()
        return errors.mean()

    def call_function_on_domain(self, func):
        """Call plain text and CrypTen function on given function
        Uses DOMAIN, TRUNCATED_DOMAIN, or appropriate layer inputs

        Returns: tuple of (plain text result, encrypted result)
        """
        DOMAIN, TRUNCATED_DOMAIN = (
            FuncBenchmarks.DOMAIN,
            FuncBenchmarks.TRUNCATED_DOMAIN,
        )
        if hasattr(DOMAIN, "to") and hasattr(TRUNCATED_DOMAIN, "to"):
            DOMAIN, TRUNCATED_DOMAIN = (
                DOMAIN.to(device=self.device),
                TRUNCATED_DOMAIN.to(device=self.device),
            )
        y = torch.rand(DOMAIN.shape, device=self.device)
        DOMAIN_enc, y_enc = crypten.cryptensor(DOMAIN), crypten.cryptensor(y)
        TRUNCATED_DOMAIN_enc = crypten.cryptensor(TRUNCATED_DOMAIN)

        if func in ["exp", "cos", "sin"]:
            ref, out_enc = (
                getattr(TRUNCATED_DOMAIN, func)(),
                getattr(TRUNCATED_DOMAIN_enc, func)(),
            )
        elif func in FuncBenchmarks.UNARY:
            ref, out_enc = getattr(DOMAIN, func)(), getattr(DOMAIN_enc, func)()
        elif func in FuncBenchmarks.LAYERS:
            ref, out_enc = self._call_layer(func)
        elif func in FuncBenchmarks.BINARY:
            ref, out_enc = (getattr(DOMAIN, func)(y), getattr(DOMAIN_enc, func)(y_enc))
        else:
            raise ValueError(f"{func} not supported")

        return ref, out_enc

    def get_errors(self):
        """Computes the total error of approximations"""
        abs_errors, relative_errors = [], []

        functions = FuncBenchmarks.UNARY + FuncBenchmarks.BINARY
        functions += FuncBenchmarks.LAYERS

        for func in functions:
            ref, out_enc = self.call_function_on_domain(func)
            out = out_enc.get_plain_text()

            abs_error = FuncBenchmarks.calc_abs_error(ref, out)
            abs_errors.append(abs_error)

            relative_error = FuncBenchmarks.calc_relative_error(ref, out)
            relative_errors.append(relative_error)

        return abs_errors, relative_errors

    def _call_layer(self, layer):
        """Call supported layers：调用支持的层函数"""
        if layer == "conv1d":
            x, x_enc, y, y_enc = self.random_conv1d_inputs()
        elif layer == "conv2d":
            x, x_enc, y, y_enc = self.random_conv2d_inputs()
        else: # 暂时只支持计算conv层
            raise ValueError(f"{layer} not supported")

        ref = getattr(torch.nn.functional, layer)(x, y)
        out_enc = getattr(x_enc, layer)(y_enc)

        return ref, out_enc

    def save(self, path):
        if self.device.type == "cuda":
            csv_path = os.path.join(path, "func_benchmarks_cuda.csv")
        else:
            csv_path = os.path.join(path, "func_benchmarks.csv")
        self.df.to_csv(csv_path, index=False)

    def get_results(self):
        return self.df

    def run(self):
        """Runs and stores benchmarks in self.df"""
        runtimes, runtimes_enc = self.get_runtimes()

        abs_errors, relative_errors = self.get_errors()

        self.df = pd.DataFrame.from_dict(
            {
                "function": FuncBenchmarks.UNARY
                + FuncBenchmarks.BINARY
                + FuncBenchmarks.LAYERS,
                "runtime": [r.mid for r in runtimes],
                "runtime Q1": [r.q1 for r in runtimes],
                "runtime Q3": [r.q3 for r in runtimes],
                "runtime crypten": [r.mid for r in runtimes_enc],
                "runtime crypten Q1": [r.q1 for r in runtimes_enc],
                "runtime crypten Q3": [r.q3 for r in runtimes_enc],
                "total abs error": abs_errors,
                "average relative error": relative_errors,
            }
        )


class ModelBenchmarks:
    """Benchmarks runtime and accuracy of crypten models:
    比较普通 PyTorch 模型和 CrypTen 加密模型训练和推理效率、准确率

    Models are benchmarked on synthetically generated
    Gaussian clusters for binary classification. Resnet18 is
    benchmarks use image data.
    模型在合成生成的高斯簇上进行二值分类做基准实验，。Resnet18是使用图像数据的基准测试。

    Args:
        n_samples (int): number of samples for Gaussian cluster model training
        n_features (int): number of features for the Gaussian clusters.
        epochs (int): number of training epochs
        lr_rate (float): learning rate.
    """

    def __init__(self, device="cpu", advanced_models=False):
        self.device = torch.device(device)
        self.df = None
        self.models = models.MODELS
        if not advanced_models:
            self.remove_advanced_models()

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Model Benchmarks"

    def remove_advanced_models(self):
        """Removes advanced models from instance"""
        self.models = list(filter(lambda x: not x.advanced, self.models)) # filter 函数和 lambda 表达式来过滤出不包含 advanced 标志的模型，然后将结果存储回 self.models 变量中

    @time_me(n_loops=3)
    def train(self, model, x, y, epochs, lr, loss):
        """Trains PyTorch model

        Args:
            model (PyTorch model): model to be trained
            x (torch.tensor): inputs
            y (torch.tensor): targets
            epochs (int): number of training epochs
            lr (float): learning rate
            loss (str): type of loss to use for training

        Returns:
            model with update weights
        """
        assert isinstance(model, torch.nn.Module), "must be a PyTorch model"
        criterion = getattr(torch.nn, loss)()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        return model

    @time_me(n_loops=3)
    def train_crypten(self, model, x, y, epochs, lr, loss):
        """Trains crypten encrypted model

        Args:
            model (CrypTen model): model to be trained
            x (crypten.tensor): inputs
            y (crypten.tensor): targets
            epochs (int): number of training epochs
            lr (float): learning rate
            loss (str): type of loss to use for training

        Returns:
            model with update weights
        """
        assert isinstance(model, crypten.nn.Module), "must be a CrypTen model"
        criterion = getattr(crypten.nn, loss)()
        # 似乎没有optimizer

        for _ in range(epochs):
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            model.update_parameters(lr)

        return model

    def time_training(self):
        """Returns training time per epoch for plain text and CrypTen"""
        runtimes = []
        runtimes_enc = []

        for model in self.models:
            x, y = model.data.x, model.data.y
            x, y = x.to(device=self.device), y.to(device=self.device)
            model_plain = model.plain
            if hasattr(model_plain, "to"):
                model_plain = model_plain.to(self.device)
            runtime, _ = self.train(model_plain, x, y, 1, model.lr, model.loss)
            runtimes.append(runtime)

            if model.advanced: # 对于 CrypTen 加密模型，会先根据模型是否为高级模型，对标签数据进行处理
                y = model.data.y_onehot.to(self.device) # 转成onehot
            x_enc = crypten.cryptensor(x)
            y_enc = crypten.cryptensor(y)

            model_crypten = model.crypten
            if hasattr(model_crypten, "to"):
                model_crypten = model_crypten.to(self.device)
            model_enc = model_crypten.encrypt() # 加密model还需要做一次encrypt操作才算完成
            runtime_enc, _ = self.train_crypten(
                model_enc, x_enc, y_enc, 1, model.lr, model.loss
            )
            runtimes_enc.append(runtime_enc)

        return runtimes, runtimes_enc

    @time_me(n_loops=3)
    def predict(self, model, x):
        y = model(x)
        return y

    def time_inference(self):
        """Returns inference time for plain text and CrypTen"""
        runtimes = []
        runtimes_enc = []

        for model in self.models:
            x = model.data.x.to(self.device)
            model_plain = model.plain
            if hasattr(model_plain, "to"):
                model_plain = model_plain.to(self.device)
            runtime, _ = self.predict(model_plain, x)
            runtimes.append(runtime)

            model_crypten = model.crypten
            if hasattr(model_crypten, "to"):
                model_crypten = model_crypten.to(self.device)
            model_enc = model_crypten.encrypt()

            x_enc = crypten.cryptensor(x)
            runtime_enc, _ = self.predict(model_enc, x_enc)
            runtimes_enc.append(runtime_enc)

        return runtimes, runtimes_enc

    @staticmethod
    def calc_accuracy(output, y, threshold=0.5):
        """Computes percent accuracy

        Args:
            output (torch.tensor): model output
            y (torch.tensor): true label
            threshold (float): classification threshold

        Returns (float): percent accuracy
        """
        output, y = output.cpu(), y.cpu()
        predicted = (output > threshold).float() # 得到预测label
        correct = (predicted == y).sum().float() # 得到预测正确的总数
        accuracy = float((correct / y.shape[0]).cpu().numpy())
        return accuracy

    def evaluate(self):
        """Evaluates accuracy of crypten versus plain text models"""
        accuracies, accuracies_crypten = [], []

        for model in self.models:
            model_plain = model.plain
            if hasattr(model_plain, "to"):
                model_plain = model_plain.to(self.device)
            x, y = model.data.x, model.data.y
            x, y = x.to(device=self.device), y.to(device=self.device)
            _, model_plain = self.train(
                model_plain, x, y, model.epochs, model.lr, model.loss
            )

            x_test = model.data.x_test.to(device=self.device)
            y_test = model.data.y_test.to(device=self.device)
            accuracy = ModelBenchmarks.calc_accuracy(model_plain(x_test), y_test)
            accuracies.append(accuracy)

            model_crypten = model.crypten
            if hasattr(model_crypten, "to"):
                model_crypten = model_crypten.to(self.device)
            model_crypten = model_crypten.encrypt()
            if model.advanced:
                y = model.data.y_onehot.to(self.device)
            x_enc = crypten.cryptensor(x)
            y_enc = crypten.cryptensor(y)
            _, model_crypten = self.train_crypten(
                model_crypten, x_enc, y_enc, model.epochs, model.lr, model.loss
            )
            x_test_enc = crypten.cryptensor(x_test)

            output = model_crypten(x_test_enc).get_plain_text()
            accuracy = ModelBenchmarks.calc_accuracy(output, y_test)
            accuracies_crypten.append(accuracy)

        return accuracies, accuracies_crypten

    def save(self, path):
        if self.device.type == "cuda":
            csv_path = os.path.join(path, "model_benchmarks_cuda.csv")
        else:
            csv_path = os.path.join(path, "model_benchmarks.csv")
        self.df.to_csv(csv_path, index=False)

    def get_results(self):
        return self.df

    def run(self):
        """Runs and stores benchmarks in self.df"""
        training_runtimes, training_runtimes_enc = self.time_training()
        inference_runtimes, inference_runtimes_enc = self.time_inference()
        accuracies, accuracies_crypten = self.evaluate()
        model_names = [model.name for model in self.models]

        training_times_both = training_runtimes + training_runtimes_enc
        inference_times_both = inference_runtimes + inference_runtimes_enc

        half_n_rows = len(training_runtimes)
        self.df = pd.DataFrame.from_dict(
            {
                "model": model_names + model_names,
                "seconds per epoch": [t.mid for t in training_times_both],
                "seconds per epoch q1": [t.q1 for t in training_times_both],
                "seconds per epoch q3": [t.q3 for t in training_times_both],
                "inference time": [t.mid for t in inference_times_both],
                "inference time q1": [t.q1 for t in inference_times_both],
                "inference time q3": [t.q3 for t in inference_times_both],
                "is plain text": [True] * half_n_rows + [False] * half_n_rows,
                "accuracy": accuracies + accuracies_crypten,
            }
        )
        self.df = self.df.sort_values(by="model")


def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Functions")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=False,
        default=None,
        help="path to save function benchmarks",
    )
    parser.add_argument(
        "--only-functions",
        "-f",
        required=False,
        default=False,
        action="store_true",
        help="run only function benchmarks",
    )
    parser.add_argument(
        "--world-size",
        "-w",
        type=int,
        required=False,
        default=1,
        help="world size for number of parties",
    )
    parser.add_argument(
        "--device",
        "-d",
        required=False,
        default="cpu",
        help="the device to run the benchmarks",
    )
    parser.add_argument(
        "--multi-gpu",
        "-mg",
        required=False,
        default=False,
        action="store_true",
        help="use different gpu for each party. Will override --device if selected",
    )
    parser.add_argument(
        "--ttp",
        "-ttp",
        required=False,
        default=False,
        action="store_true",
        help="initialize a trusted third party (TTP) as beaver triples' provider, world_size should be greater than 2",
    )
    parser.add_argument(
        "--advanced-models",
        required=False,
        default=False,
        action="store_true",
        help="run advanced model (resnet, transformer, etc.) benchmarks",
    )
    args = parser.parse_args()
    return args


def multiprocess_caller(args):
    """Runs multiparty benchmarks and prints/saves from source 0"""
    rank = comm.get().get_rank()
    if args.multi_gpu:
        assert (
            args.world_size >= torch.cuda.device_count()
        ), f"Got {args.world_size} parties, but only {torch.cuda.device_count()} GPUs found"
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(args.device)

    benchmarks = [
        FuncBenchmarks(device=device),
        ModelBenchmarks(device=device, advanced_models=args.advanced_models),
    ]

    if args.only_functions:
        benchmarks = [FuncBenchmarks(device=device)]

    for benchmark in benchmarks:
        benchmark.run()
        rank = comm.get().get_rank()
        if rank == 0:
            pd.set_option("display.precision", 3)
            print(benchmark)
            if args.path:
                benchmark.save(args.path)


def main():
    """Runs benchmarks and saves if path is provided"""
    crypten.init()
    args = get_args()
    device = torch.device(args.device)

    if not hasattr(crypten.nn.Module, "to") or not hasattr(crypten.mpc.MPCTensor, "to"):
        if device.type == "cuda":
            print(
                "GPU computation is not supported for this version of CrypTen, benchmark will be skipped"
            )
            return

    benchmarks = [
        FuncBenchmarks(device=device),
        ModelBenchmarks(device=device, advanced_models=args.advanced_models),
    ]

    if args.only_functions:
        benchmarks = [FuncBenchmarks(device=device)]

    if args.world_size > 1:
        if args.ttp:
            crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedThirdParty)
        launcher = multiprocess_launcher.MultiProcessLauncher(
            args.world_size, multiprocess_caller, fn_args=args
        )
        launcher.start()
        launcher.join()
        launcher.terminate()

    else:
        pd.set_option("display.precision", 3)
        for benchmark in benchmarks:
            benchmark.run()
            print(benchmark)
            if args.path:
                benchmark.save(args.path)


if __name__ == "__main__":
    main()
