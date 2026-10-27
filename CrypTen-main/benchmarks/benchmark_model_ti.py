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
$ python benchmark_model_ti.py --device cuda --advanced-models -p /mnt/workspace/Downloads/v2/gpu/advanced_models/before
$ python benchmark_model.py --advanced-models --device cuda -p /mnt/workspace/Downloads/v3/gpu/advanced_models/1
$ python benchmark_model.py --advanced-models --device cuda --world-size 2 -p /data2/ljd/home/SELMA/CrypTen-main/result/v3/gpu/advanced_models/2/now/TI
"""

import argparse
import functools
import os
import timeit
import time
from collections import namedtuple

# mytest optim
# from crypten import crypten
import crypten
import crypten.communicator as comm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F # mytest for advanced model
from examples import multiprocess_launcher

#mytest
import random as rd

from crypten.config import cfg  # add this line

cfg.communicator.verbose = True # add this line


# try:
#     from . import data, models
# except ImportError:
#     # direct import if relative fails
#     import data
#     import models

import data
import models

# mytest
Runtime = namedtuple("Runtime", "mid q1 q3 comm")
# Runtime = namedtuple("Runtime", "mid q1 q3")


def time_me(func=None, n_loops=10):
    """Decorator returning average runtime in seconds over n_loops

    Args:
        func (function): invoked with given args / kwargs
        n_loops (int): number of times to invoke function for timing

    Returns: tuple of (time in seconds, inner quartile range, function return value).
    """
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

#     mytest 带测通信
    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        return_val = func(*args, **kwargs)
        times = []
        comm_bytes = []
        for i in range(n_loops):
            # mytest
            comm.get().reset_communication_stats()
            # start = timeit.default_timer()
            start = time.perf_counter()
            func(*args, **kwargs)
            # times.append(timeit.default_timer() - start)
            times.append(time.perf_counter() - start)
            val = comm.get().get_communication_stats()
            comm_bytes.append(val["bytes"])
        mid_runtime = np.quantile(times, 0.5)
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        mid_comm = np.quantile(comm_bytes, 0.5) / 1024**2
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime, mid_comm)
        return runtime, return_val

    return timing_wrapper

    # @functools.wraps(func) # 装饰器，用于将 timing_wrapper 函数的元数据（如名称、文档字符串等）设置为原始函数 func 的元数据，以保留函数的原始信息
    # def timing_wrapper(*args, **kwargs): # 装饰函数 timing_wrapper，接受任意数量的位置参数和关键字参数
    #     return_val = func(*args, **kwargs) # 调用原始函数 func 并记录返回值
    #     times = [] # 创建一个空列表，用于存储每次函数调用的运行时间
    #     for _ in range(n_loops): # 循环 n_loops 次，即指定的计时次数
    #         start = timeit.default_timer() # 记录当前时间作为函数调用的起始时间
    #         func(*args, **kwargs) # 再次调用原始函数 func，对其进行计时
    #         times.append(timeit.default_timer() - start) # 计算函数调用的运行时间，并将其添加到 times 列表中
    #     mid_runtime = np.quantile(times, 0.5) # 计算运行时间列表 times 的中位数
    #     q1_runtime = np.quantile(times, 0.25) # 计算运行时间列表 times 的下四分位数
    #     q3_runtime = np.quantile(times, 0.75) # 计算运行时间列表 times 的上四分位数
    #     runtime = Runtime(mid_runtime, q1_runtime, q3_runtime) # 创建一个 Runtime 对象，存储中位数、下四分位数和上四分位数
    #     return runtime, return_val # 返回一个元组，包含运行时间对象和原始函数的返回值

    # return timing_wrapper


class FuncBenchmarks:
    """Benchmarks runtime and error of crypten functions against PyTorch

    Args:
        tensor_size (int or tuple): size of tensor for benchmarking runtimes
    """

    BINARY = ["add", "sub", "mul", "matmul", "gt", "lt", "eq"]

    UNARY = [
        "sigmoid",
        "relu",
        "approx_relu",
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
    
    LAYERS = ["conv2d","max_pool2d","approxmax_pool2d","avg_pool2d","dropout"] # mytest

    """
    BINARY = []

    UNARY = [
      
    ]
    
    LAYERS = ["approxmax_pool2d"] # mytest
    
    # LAYERS = ["conv2d"] 
    """

    DOMAIN = torch.arange(start=0.01, end=100, step=0.01)
    # for exponential, sin, and cos
    TRUNCATED_DOMAIN = torch.arange(start=0.001, end=10, step=0.001)

    def __init__(self, tensor_size=(100, 100), device="cpu"):
        self.device = torch.device(device)
        self.tensor_size = tensor_size

        # dataframe for benchmarks
        self.df = None

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Function Benchmarks"

    @staticmethod
    @time_me
    def time_func(x, func, y=None):
        """Invokes func as a method of x"""
        if y is None:
            return getattr(x, func)()
 
        if func in {"conv1d", "conv2d", "max_pool2d","avg_pool2d","dropout"}: # mytest
        # if func in {"conv1d", "conv2d"}:
            if torch.is_tensor(x):
                return getattr(torch.nn.functional, func)(x, y)
            return getattr(x, func)(y)
 
        return getattr(x, func)(y)

    def get_runtimes(self):
        """Returns plain text and crypten runtimes"""
        x, y = (
            torch.rand(self.tensor_size, device=self.device),
            torch.rand(self.tensor_size, device=self.device),
        )
        x_enc, y_enc = crypten.cryptensor(x), crypten.cryptensor(y)

        runtimes, runtimes_enc = [], []

        for func in FuncBenchmarks.UNARY + FuncBenchmarks.BINARY:
            second_operand, second_operand_enc = None, None
            if func in FuncBenchmarks.BINARY:
                second_operand, second_operand_enc = y, y_enc

            # mytest for optim
            if func == "approx_relu":
                runtime, _ = FuncBenchmarks.time_func(x, "relu", y=second_operand)
            else:
                runtime, _ = FuncBenchmarks.time_func(x, func, y=second_operand)
            runtimes.append(runtime)

            runtime_enc, _ = FuncBenchmarks.time_func(x_enc, func, y=second_operand_enc)
            runtimes_enc.append(runtime_enc)

        # add layer runtimes
        runtime_layers, runtime_layers_enc = self.get_layer_runtimes()
        runtimes.extend(runtime_layers)
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
            # mytest
            elif layer == "max_pool2d" or layer == "avg_pool2d" or layer == "approxmax_pool2d" :
                x,x_enc,y,y_enc = self.random_pool2d_inputs()
            elif layer == "dropout":
                x,x_enc,y,y_enc = self.random_dropout_inputs()
            # else :
            #     raise ValueError(f"{layer} not supported")
            
            # mytest for optim
            if layer == "approxmax_pool2d":
                x,x_enc,y,y_enc = self.random_pool2d_inputs()
                runtime, _ = FuncBenchmarks.time_func(x, "max_pool2d", y=y) # runtime, _ = self.soft_pool2d(x,y)
                runtime_enc, _ = FuncBenchmarks.time_func(x_enc, layer, y=y_enc)
            else:
                runtime, _ = FuncBenchmarks.time_func(x, layer, y=y)
                runtime_enc, _ = FuncBenchmarks.time_func(x_enc, layer, y=y_enc)

            runtime_layers.append(runtime)
            runtime_layers_enc.append(runtime_enc)

        return runtime_layers, runtime_layers_enc

    #mytest
    def random_dropout_inputs(self):
        """得到2d dropout层的随机input和p"""
        x_dropout = torch.rand(1,1,*self.tensor_size,device=self.device)
        p = 0.5 # 除了输入以外其他参数不能随机，否则无法多方
        x_dropout_enc = crypten.cryptensor(x_dropout)
        return x_dropout,x_dropout_enc,p,p
    
    def random_pool2d_inputs(self):
        """得到2d 池化层的随机input和kernel"""
        x_maxpool2d = torch.rand(1,1,*self.tensor_size,device=self.device) 
        kernel_size = 10 #除了输入以外其他参数不能随机，否则无法多方
        x_maxpool2d_enc = crypten.cryptensor(x_maxpool2d)
        return x_maxpool2d,x_maxpool2d_enc,kernel_size,kernel_size
        
    
    def random_conv2d_inputs(self):
        """Returns random input and weight tensors for 2d convolutions"""
        filter_size = [size // 10 for size in self.tensor_size]
        x_conv2d = torch.rand(1, 1, *self.tensor_size, device=self.device)
        weight2d = torch.rand(1, 1, *filter_size, device=self.device)
        x_conv2d_enc = crypten.cryptensor(x_conv2d)
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
            errors = (out != ref).numpy().sum()
            return errors
        errors = torch.abs(out - ref).numpy()
        return errors.sum()

    @staticmethod
    def calc_relative_error(ref, out):
        """Computes average relative error"""
        ref, out = ref.cpu(), out.cpu()
        if ref.dtype == torch.bool:
            errors = (out != ref).numpy().sum() // ref.nelement()
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
            # mytest for optim
            if func == "approx_relu":
                ref = getattr(DOMAIN, "relu")()
            else:
                ref = getattr(DOMAIN, func)()
            out_enc = getattr(DOMAIN_enc, func)()
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
        """Call supported layers"""
        if layer == "conv1d":
            x, x_enc, y, y_enc = self.random_conv1d_inputs()
        elif layer == "conv2d":
            x, x_enc, y, y_enc = self.random_conv2d_inputs()
        # mytest
        elif layer == "max_pool2d" or layer == "avg_pool2d" or layer == "approxmax_pool2d":
            x,x_enc,y,y_enc = self.random_pool2d_inputs()
        elif layer == "dropout":
            x,x_enc,y,y_enc = self.random_dropout_inputs()
        # else:
        #     raise ValueError(f"{layer} not supported")

        # mytest for approxmax_pool2d
        if layer == "approxmax_pool2d":
            x,x_enc,y,y_enc = self.random_pool2d_inputs()
            ref = getattr(torch.nn.functional, "max_pool2d")(x, y) # _, ref = self.soft_pool2d(x,y)
            out_enc = getattr(x_enc, layer)(y_enc)
        else:
            ref = getattr(torch.nn.functional, layer)(x, y)
            out_enc = getattr(x_enc, layer)(y_enc)

        # print(ref)
        # print(out_enc.get_plain_text())
        # print()
        return ref, out_enc

    # 带测通信
    # def save(self, path):
    #     print("===============save==============")
    #     if self.device.type == "cuda":
    #         csv_path = os.path.join(path, "runtime_cuda.csv") # mytest for approxmax_pool2d 
    #     else:
    #         csv_path = os.path.join(path, "runtime.csv")
    #     self.df.to_csv(csv_path, index=False)
        
    # 不带测通信
    def save(self, path):
        if self.device.type == "cuda":
            csv_path = os.path.join(path, "func_benchmarks_cuda.csv")
        else:
            csv_path = os.path.join(path, "func_benchmarks.csv")
        self.df.to_csv(csv_path, index=False)

    def get_results(self):
        return self.df
    
    # 带测通信
#     def run(self):
#         """Runs and stores benchmarks in self.df"""
#         runtimes, runtimes_enc = self.get_runtimes()

#         abs_errors, relative_errors = self.get_errors()

#         # mytest
#         self.df = pd.DataFrame.from_dict(
#             {
#                 "function": FuncBenchmarks.UNARY
#                 + FuncBenchmarks.BINARY
#                 + FuncBenchmarks.LAYERS,
#                 "runtime": [r.mid for r in runtimes],
#                 "runtime comm": [r.comm for r in runtimes],
#                 "runtime crypten": [r.mid for r in runtimes_enc],
#                 "runtime crypten comm": [r.comm for r in runtimes_enc],
#                 "total abs error": abs_errors,
#                 "average relative error": relative_errors,
#             }
#         )    
    
    # 不带测通信
    
    def run(self):
        """Runs and stores benchmarks in self.df"""
        runtimes, runtimes_enc = self.get_runtimes()

        abs_errors, relative_errors = self.get_errors()

        # mytest
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
    """Benchmarks runtime and accuracy of crypten models

    Models are benchmarked on synthetically generated
    Gaussian clusters for binary classification. Resnet18 is
    benchmarks use image data.

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
        self.models = list(filter(lambda x: not x.advanced, self.models))

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

        for epoch in range(epochs):
            output_enc = model(x)
            loss = criterion(output_enc, y)
            if epoch%10 == 0:
                print(f"Loss:{loss.get_plain_text().item():.4f}")
            model.zero_grad()
            loss.backward()
            model.update_parameters(lr)

        return model
    
    @time_me(n_loops=3)
    def train_advanced(self, train_loader, model, criterion, optimizer, device, tag=-1, model_name=""):
        assert isinstance(model, torch.nn.Module), "must be a PyTorch model"
        
        with open(f"plain_loss_{model_name}.txt", "a") as f:  # 以追加模式打开文件
            f.write(f"Epoch {tag}\n")
        for i,(input,target) in enumerate(train_loader):
            input,target = input.to(device=device),target.to(device=device)
            output = model(input)
            loss = criterion(output,target)
            if i%10 == 0: 
                print(f"Plain Batch {i} Loss:{loss:.4f}")
                # 写入 loss 文件
                with open(f"plain_loss_{model_name}.txt", "a") as f:  # 以追加模式打开文件
                    f.write(f"Batch {i}, Loss: {loss.item():.4f}\n")
            optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            # if i%10 == 0:
            #     print(f"Gradient norm: {total_grad_norm:.6f}")
            #     with open("plain_loss_sq.txt", "a") as f:
            #         f.write(f"Batch {i}, Gradient norm: {total_grad_norm:.6f}\n")
            
            optimizer.step()
            if tag == -1:
                break

        return model
    
    @time_me(n_loops=3)
    def train_advanced_crypten(self, train_loader, model, criterion, lr, device, tag=-1, model_name=""):
        assert isinstance(model, crypten.nn.Module), "must be a CrypTen model"

        with open(f"enc_loss_{model_name}.txt", "a") as f:  # 以追加模式打开文件
            f.write(f"Epoch {tag}\n")
        for i,(input,target) in enumerate(train_loader):
            input,target = input.to(device=device),target.to(device=device)
            target = F.one_hot(target,200) # crypten不像pytorch可以自动转换成onehot，所以还得额外做一步
            input_enc = crypten.cryptensor(input, requires_grad=True)
            target_enc = crypten.cryptensor(target, requires_grad=True)
            output_enc = model(input_enc)
            loss = criterion(output_enc,target_enc)
            if i%10 == 0:
                # print("output_enc min:", output_enc.get_plain_text().min().item(), "max:", output_enc.get_plain_text().max().item())
                # print("target_enc min:", target_enc.get_plain_text().min().item(), "max:", target_enc.get_plain_text().max().item())
                print(f"Enc Batch {i} Loss:{loss.get_plain_text().item():.4f}")
                with open(f"enc_loss_{model_name}.txt", "a") as f:  # 以追加模式打开文件
                    f.write(f"Batch {i}, Loss: {loss.get_plain_text().item():.4f}\n")
            model.zero_grad()
            loss.backward()
            # backpropagation_done = all(param.grad is not None for param in model.parameters())
            # print(backpropagation_done)
            model.update_parameters(lr)
            if tag == -1:
                break

        return model

    def time_training(self):
        """Returns training time per epoch for plain text and CrypTen"""
        runtimes = []
        runtimes_enc = []
        print("=============Time Training=============")
        for model in self.models:
            print("=============",model.name,"=============")
            if model.advanced:
                train_loader = model.data.train_loader
                # 明文
                print("----------plain----------")
                model_plain = model.plain
                if hasattr(model_plain, "to"):
                    model_plain = model_plain.to(self.device)
                model_plain.train()
                criterion = getattr(torch.nn, model.loss)()
                optimizer = torch.optim.SGD(model_plain.parameters(), lr=model.lr)
                for i in range(0,model.epochs):
                    runtime, model_plain = self.train_advanced(
                        train_loader,model_plain,criterion,optimizer,self.device,-1,model.name
                    )
                runtimes.append(runtime)
                
                # 密文
                print("----------Enc----------")
                model_crypten = model.crypten
                if hasattr(model_crypten, "to"):
                    model_crypten = model_crypten.to(self.device)
                # model_crypten.train() # 设置为训练模式
                model_enc = model_crypten.encrypt()
                criterion = getattr(crypten.nn, model.loss)()
                for i in range(0,model.epochs):
                    runtime_enc, model_enc = self.train_advanced_crypten(
                        train_loader,model_enc,criterion,model.lr,self.device,i,model.name
                    )
                # runtime_enc = Runtime(0, 0, 0, 0)
                runtimes_enc.append(runtime_enc)
                continue
    
            # 明文
            print("----------plain----------")
            x, y = model.data.x, model.data.y
            x, y = x.to(device=self.device), y.to(device=self.device)
            model_plain = model.plain
            if hasattr(model_plain, "to"):
                model_plain = model_plain.to(self.device)
            runtime, _ = self.train(model_plain, x, y, 1, model.lr, model.loss)
            runtimes.append(runtime)
            
            # 密文
            print("----------Enc----------")
            x_enc = crypten.cryptensor(x)
            y_enc = crypten.cryptensor(y)
            model_crypten = model.crypten
            if hasattr(model_crypten, "to"):
                model_crypten = model_crypten.to(self.device)
            model_enc = model_crypten.encrypt()
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
        print("=============Time Inference=============")
        runtimes = []
        runtimes_enc = []

        for model in self.models:
            if model.advanced:
                train_loader = model.data.train_loader #似乎得用testloader
                # 明文
                print("----------plain----------")
                model_plain = model.plain
                if hasattr(model_plain, "to"):
                    model_plain = model_plain.to(self.device)
                for i,(input,target) in enumerate(train_loader):
                    input,target = input.to(device=self.device),target.to(device=self.device)
                    break
                runtime, _ = self.predict(model_plain, input)
                runtimes.append(runtime)
                
                # 密文
                print("----------Enc----------")
                model_crypten = model.crypten
                if hasattr(model_crypten, "to"):
                    model_crypten = model_crypten.to(self.device)
                model_enc = model_crypten.encrypt()
                for i,(input,target) in enumerate(train_loader):
                    input,target = input.to(device=self.device),target.to(device=self.device)
                    break
                input_enc = crypten.cryptensor(input)
                runtime_enc, _ = self.predict(model_enc, input_enc)
                runtimes_enc.append(runtime_enc)
                continue
                
                
            # 明文
            print("----------plain----------")
            x = model.data.x.to(self.device)
            model_plain = model.plain
            if hasattr(model_plain, "to"):
                model_plain = model_plain.to(self.device)
            runtime, _ = self.predict(model_plain, x)
            runtimes.append(runtime)
            
            # 密文
            print("----------Enc----------")
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
        if threshold == 1:
            predicted = torch.argmax(output, dim=1).float()
        else:
            predicted = (output > threshold).float()
        correct = (predicted == y).sum().float()
        accuracy = float((correct / y.shape[0]).cpu().numpy())
        return accuracy

    def evaluate(self):
        """Evaluates accuracy of crypten versus plain text models"""
        print("=============Evaluate=============")
        accuracies, accuracies_crypten = [], []

        for model in self.models:
            print("=============",model.name,"=============")
            if model.advanced:
                print("----------plain----------")
                # train_loader, test_loader = model.data.train_loader, model.data.test_loader
                # # 明文
                # model_plain = model.plain
                # if hasattr(model_plain, "to"):
                #     model_plain = model_plain.to(self.device)
                # model_plain.train()
                # criterion = getattr(torch.nn, model.loss)()
                # optimizer = torch.optim.SGD(model_plain.parameters(), lr=model.lr)
                # # 训练
                # for _ in range(0,model.epochs):
                #     _, model_plain = self.train_advanced(
                #         train_loader,model_plain,criterion,optimizer,self.device
                #     )
                #     break
                # 测试
                accuracy = 0
                # i = 0
                # for i,(input,target) in enumerate(test_loader):
                #     x_test,y_test = input.to(device=self.device),target.to(device=self.device)
                #     accuracy += ModelBenchmarks.calc_accuracy(model_plain(x_test), y_test, 1)
                #     break
                # accuracy /= (i+1)   
                accuracies.append(accuracy)
                
                
                # 密文
                print("----------Enc----------")
                # model_crypten = model.crypten
                # if hasattr(model_crypten, "to"):
                #     model_crypten = model_crypten.to(self.device)
                # # model_crypten.train()
                # model_enc = model_crypten.encrypt()
                # criterion = getattr(crypten.nn, model.loss)()
                # # 训练
                # for _ in range(0,model.epochs):
                #     _, model_enc = self.train_advanced_crypten(
                #         train_loader,model_enc,criterion,model.lr,self.device
                #     )
                #     break
                # 测试
                accuracy = 0
                # i = 0
                # for i,(input,target) in enumerate(test_loader):
                #     x_test,y_test = input.to(device=self.device),target.to(device=self.device)
                #     x_test_enc = crypten.cryptensor(x_test)
                #     output = model_crypten(x_test_enc).get_plain_text()
                #     accuracy += ModelBenchmarks.calc_accuracy(output, y_test, 1)
                #     break
                    
                # accuracy /= (i+1)   
                accuracies_crypten.append(accuracy)
                continue
             
            
            # 明文
            print("----------plain----------")
            model_plain = model.plain
            if hasattr(model_plain, "to"):
                model_plain = model_plain.to(self.device)
            x, y = model.data.x, model.data.y
            x, y = x.to(device=self.device), y.to(device=self.device)
            # 训练
            _, model_plain = self.train(
                model_plain, x, y, model.epochs, model.lr, model.loss
            )            
            # 测试
            x_test = model.data.x_test.to(device=self.device)
            y_test = model.data.y_test.to(device=self.device)
            accuracy = ModelBenchmarks.calc_accuracy(model_plain(x_test), y_test)
            accuracies.append(accuracy)
            
            # 密文
            print("----------Enc----------")
            model_crypten = model.crypten
            if hasattr(model_crypten, "to"):
                model_crypten = model_crypten.to(self.device)
            model_crypten = model_crypten.encrypt()
            x_enc = crypten.cryptensor(x)
            y_enc = crypten.cryptensor(y)
            # 训练
            _, model_crypten = self.train_crypten(
                model_crypten, x_enc, y_enc, model.epochs, model.lr, model.loss
            )
            # 测试
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
                "seconds per batch": [t.mid for t in training_times_both],
                "seconds per batch q1": [t.q1 for t in training_times_both],
                "seconds per batch q3": [t.q3 for t in training_times_both],
                "comm per batch": [t.comm for t in training_times_both],
                "inference time": [t.mid for t in inference_times_both],
                "inference time q1": [t.q1 for t in inference_times_both],
                "inference time q3": [t.q3 for t in inference_times_both],
                "inference comm": [t.comm for t in inference_times_both],
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
        # FuncBenchmarks(device=device),
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
        # FuncBenchmarks(device=device),
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
