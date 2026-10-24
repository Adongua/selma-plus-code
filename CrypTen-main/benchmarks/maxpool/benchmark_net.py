#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate function and model benchmarks

To Run:
$ python benchmark_net.py

# Benchmark functions and all models
$ python benchmark.py --advanced-models

# Run benchmarks on GPU
$ python benchmark_net.py --device cuda
$ python benchmark_net.py --device cuda --world-size 2

# Run benchmarks on different GPUs for each party
$ python benchmark_net.py --world-size=2 --multi-gpu

# Save benchmarks to csv
$ python benchmark_net.py --device cuda -p /mnt/workspace/Downloads/gpu/single_party/
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

# mytest
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from examples.util import NoopContextManager
import tempfile

from crypten.config import cfg  # add this line
cfg.communicator.verbose = True # add this line

# grad
from crypten.gradients import AutogradContext

try:
    from . import data, models
except ImportError:
    # direct import if relative fails
    import data
    import models

Runtime = namedtuple("Runtime", "mid q1 q3 mean comm comm_mean")


def time_me(func=None, n_loops=10):
    """Decorator returning average runtime in seconds over n_loops

    Args:
        func (function): invoked with given args / kwargs
        n_loops (int): number of times to invoke function for timing

    Returns: tuple of (time in seconds, inner quartile range, function return value).
    """
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        return_val = func(*args, **kwargs)
        times = []
        comm_times = []
        for i in range(n_loops):
            # mytest
            comm.get().reset_communication_stats()
            start = timeit.default_timer()
            func(*args, **kwargs)
            times.append(timeit.default_timer() - start)
            val = comm.get().get_communication_stats() # 通信开销
            comm_times.append(val["time"]) 
        mid_runtime = np.quantile(times, 0.5)
        mid_commtime = np.quantile(comm_times, 0.5)
        if n_loops == 1: 
            return mid_runtime,mid_commtime,return_val
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        mean_runtime = sum(times)/len(times)
        mean_commtime = sum(comm_times)/len(comm_times)
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime,mean_runtime,mid_commtime,mean_commtime)
        return runtime, return_val

    return timing_wrapper


class LeNet(nn.Sequential):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
        self.rt = {
            "conv1":[], 
            "conv2":[],
            "pool1":[],
            "pool2":[],
            "fc1":[],
            "fc2":[],
            "fc3":[],
            "relu1":[],
            "relu2":[],
            "relu3":[],
            "relu4":[]
        }
        self.rt_comm = {
            "conv1":[], 
            "conv2":[],
            "pool1":[],
            "pool2":[],
            "fc1":[],
            "fc2":[],
            "fc3":[],
            "relu1":[],
            "relu2":[],
            "relu3":[],
            "relu4":[]
        }

    @staticmethod
    @time_me(n_loops=1)
    def time_layer(self, layer, x=None):
        """对x调用某层的前向传播"""
        
        if layer in {"conv1", "conv2", "pool", "fc1", "fc2", "fc3"}: # mytest
            return getattr(self,layer)(x)
        elif layer in {"relu"}:
            return getattr(F,layer)(x)
        
        return getattr(self,layer)(x)
        
    def forward(self, x):
        # print(x)
        cost,cost_comm,x = LeNet.time_layer(self,"conv1",x)
        self.rt["conv1"].append(cost)
        self.rt_comm["conv1"].append(cost_comm)
        
        cost,cost_comm,x = LeNet.time_layer(self,"relu",x)
        self.rt["relu1"].append(cost)
        self.rt_comm["relu1"].append(cost_comm)
        
        cost,cost_comm,x = LeNet.time_layer(self,"pool",x)
        self.rt["pool1"].append(cost)
        self.rt_comm["pool1"].append(cost_comm)
        
        cost,cost_comm,x = LeNet.time_layer(self,"conv2",x)
        self.rt["conv2"].append(cost)
        self.rt_comm["conv2"].append(cost_comm)
        
        cost,cost_comm,x = LeNet.time_layer(self,"relu",x)
        self.rt["relu2"].append(cost)
        self.rt_comm["relu2"].append(cost_comm)
        
        cost,cost_comm,x = LeNet.time_layer(self,"pool",x)
        self.rt["pool2"].append(cost)
        self.rt_comm["pool2"].append(cost_comm)
                      
        x = x.view(-1,16*5*5)
        
        cost,cost_comm,x = LeNet.time_layer(self,"fc1",x)
        self.rt["fc1"].append(cost)
        self.rt_comm["fc1"].append(cost_comm)
        
        cost,cost_comm,x = LeNet.time_layer(self,"relu",x)
        self.rt["relu3"].append(cost)
        self.rt_comm["relu3"].append(cost_comm)
        
        cost,cost_comm,x = LeNet.time_layer(self,"fc2",x)
        self.rt["fc2"].append(cost)
        self.rt_comm["fc2"].append(cost_comm)
                      
        cost,cost_comm,x = LeNet.time_layer(self,"relu",x)
        self.rt["relu4"].append(cost)
        self.rt_comm["relu4"].append(cost_comm)
                      
        cost,cost_comm,x = LeNet.time_layer(self,"fc3",x)
        self.rt["fc3"].append(cost)
        self.rt_comm["fc3"].append(cost_comm)
                         
        return x
    
    
class ApproxMaxPool2D(crypten.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(ApproxMaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        
    def approxmax_pool2d(self, x_enc, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, is_print=False):
        if ceil_mode:
            raise NotImplementedError(
                "CrypTen does not support `ceil_mode` for `approxmax_pool2d`"
            )
        
        x2_enc = x_enc.square()
    
        x_enc_avg2 = x_enc.mean().square()
        x_enc_var = x_enc.var()
        approx_r = 7 * x_enc_avg2 + 9 * x_enc_var 
    
        tau, i = 260, 0
        bound = 10            
        r = torch.tensor([0.1],device=x_enc.device) 
        while(1):
            if approx_r.le(r).get_plain_text():
                break
            r *= 100
            tau /= 10
            i += 1
            if i >= bound:
                break
        x2_enc = tau**2 * x2_enc + 1
        
        # 套对应公式变形
        y = x_enc.mul(x2_enc)
        y = y.avg_pool2d(kernel_size,stride=stride, padding=padding) * (kernel_size**2)
        x2_enc_sum = x2_enc.avg_pool2d(kernel_size, stride=stride, padding=padding) * (kernel_size**2)
        x2_enc_sum = x2_enc_sum.get_plain_text()
        x2_enc_sum = torch.where(x2_enc_sum > 1, torch.floor(x2_enc_sum), x2_enc_sum)

        z = y / x2_enc_sum 
        return z
    
    def forward(self, x):
        # preprocess inputs:
        if self.stride is None:
            self.stride = self.kernel_size
        
        # perform max pooling:
        y = self.approxmax_pool2d(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode
        ) # 为先前写好的最大池化优化最终版
        
        return y

class LeNetCrypTen(crypten.nn.Sequential):
    def __init__(self):
        super(LeNetCrypTen,self).__init__()
        self.conv1 = crypten.nn.Conv2d(3,6,5)
        self.pool = crypten.nn.MaxPool2d(2,2)#ApproxMaxPool2D(2,2)
        self.conv2 = crypten.nn.Conv2d(6,16,5)
        self.fc1 = crypten.nn.Linear(16*5*5,120)
        self.fc2 = crypten.nn.Linear(120,84)
        self.fc3 = crypten.nn.Linear(84,10)
        
        self.rt_enc = {
            "conv1":[], 
            "conv2":[],
            "pool1":[],
            "pool2":[],
            "fc1":[],
            "fc2":[],
            "fc3":[],
            "relu1":[],
            "relu2":[],
            "relu3":[],
            "relu4":[]
        }
        self.rt_enc_comm = {
            "conv1":[], 
            "conv2":[],
            "pool1":[],
            "pool2":[],
            "fc1":[],
            "fc2":[],
            "fc3":[],
            "relu1":[],
            "relu2":[],
            "relu3":[],
            "relu4":[]
        }
    
    @staticmethod
    @time_me(n_loops=1)
    def time_layer(self, layer, x=None):
        """对x调用某层的前向传播"""
        
        if layer in {"conv1", "conv2", "pool", "fc1", "fc2", "fc3"}: # mytest
            return getattr(self,layer)(x)
        elif layer in {"relu"}:
            return getattr(x,layer)()
        
        return getattr(self,layer)(x)
        
    def forward(self, x):
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"conv1",x)
        self.rt_enc["conv1"].append(cost)
        self.rt_enc_comm["conv1"].append(cost_comm)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"relu",x)
        self.rt_enc["relu1"].append(cost)
        self.rt_enc_comm["relu1"].append(cost_comm)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"pool",x)
        self.rt_enc["pool1"].append(cost)
        self.rt_enc_comm["pool1"].append(cost_comm)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"conv2",x)
        self.rt_enc["conv2"].append(cost)
        self.rt_enc_comm["conv2"].append(cost_comm)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"relu",x)
        self.rt_enc["relu2"].append(cost)
        self.rt_enc_comm["relu2"].append(cost_comm)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"pool",x)
        self.rt_enc["pool2"].append(cost)
        self.rt_enc_comm["pool2"].append(cost_comm)
                      
        x = x.view(-1,16*5*5)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"fc1",x)
        self.rt_enc["fc1"].append(cost)
        self.rt_enc_comm["fc1"].append(cost_comm)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"relu",x)
        self.rt_enc["relu3"].append(cost)
        self.rt_enc_comm["relu3"].append(cost_comm)
        
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"fc2",x)
        self.rt_enc["fc2"].append(cost)
        self.rt_enc_comm["fc2"].append(cost_comm)
                      
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"relu",x)
        self.rt_enc["relu4"].append(cost)
        self.rt_enc_comm["relu4"].append(cost_comm)
                      
        cost,cost_comm,x = LeNetCrypTen.time_layer(self,"fc3",x)
        self.rt_enc["fc3"].append(cost)
        self.rt_enc_comm["fc3"].append(cost_comm)
                                   
        return x
    
    
def train(train_loader, model, criterion, optimizer, device):
    assert isinstance(model, torch.nn.Module), "must be a PyTorch model"

    for i,(input,target) in enumerate(train_loader):
        input,target = input.to(device=device),target.to(device=device)
        # print(input)
        # print(target)
        # print("###")
        # print(output)
        model.zero_grad()
        output = model(input)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss))
        # break

    return model

def train_enc(train_loader, model, criterion, lr, device):
    assert isinstance(model, crypten.nn.Module), "must be a CrypTen model"

    for i,(input,target) in enumerate(train_loader):
        input,target = input.to(device=device),target.to(device=device)
        target = F.one_hot(target,10) # crypten不像pytorch可以自动转换成onehot，所以还得额外做一步
        input_enc = crypten.cryptensor(input)
        target_enc = crypten.cryptensor(target)
        # print(target_enc)
        # print("###")
        # print(output_enc)
        model.zero_grad()
        output_enc = model(input_enc)
        loss = criterion(output_enc,target_enc)
        loss.backward()
        model.update_parameters(lr)
        print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss.get_plain_text()))
        # break

    return model

def time_training(device,epochs=10,lr=0.01,context_manager=None):
    
    def preprocess_data(context_manager,data_dirname):
        # 定义数据转换管道，用于对CIFAR10数据集进行预处理操作：将CIFAR10数据集中的图像数据转换为张量，并对每个通道的像素值进行标准化处理
        transform = transforms.Compose(
            [
                transforms.ToTensor(), # 将图像数据转换为张量（Tensor）格式。这个操作会将图片的通道维度从H×W×C（高度×宽度×通道数）转换为C×H×W（通道数×高度×宽度）的张量格式，并将像素值归一化到[0, 1]范围内
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 对张量进行归一化处理,均值是(0.5, 0.5, 0.5)，标准差也是(0.5, 0.5, 0.5)
            ]
        )
        # 下载、加载数据集
        with context_manager:
            trainset = datasets.CIFAR10(
                data_dirname,train=True,download=True,transform=transform
            )
            testset = datasets.CIFAR10(
                data_dirname,train=False,download=True,transform=transform
            )
        trainloader = torch.utils.data.DataLoader(
            trainset,batch_size=100,shuffle=True,num_workers=2 # mytest
        )
        # shuffle=True：这个参数表示是否要对数据进行随机重排（洗牌）。设置为True，则数据在每个epoch训练之前会被随机打乱顺序，从而增加模型学习的多样性
        # num_workers = 2：这个参数定义了用于数据加载的工作线程数。在这里，使用了2个工作线程来并行加载数据，可以提高数据加载的效率
        testloader = torch.utils.data.DataLoader(
            testset,batch_size=1,shuffle=True,num_workers=2
        )
        return trainloader,testloader
    
    if context_manager is None:
        context_manager = NoopContextManager()
    
    data_dirname = "/mnt/data/tmpd5wtqqcd/" # data_dir = tempfile.TemporaryDirectory()
    train_loader, val_loader = preprocess_data(context_manager, data_dirname) # 得到训练数据加载器和验证数据加载器


    runtimes,runtimes_enc = [],[]
    
    model_plain = LeNet()
    if hasattr(model_plain, "to"):
        model_plain = model_plain.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_plain.parameters(), lr=lr)
    
    for _ in range(0,epochs):
        model_plain = train(train_loader,model_plain,criterion,optimizer,device)

        # runtime = Runtime(mid_runtime, q1_runtime, q3_runtime) # 创建一个 Runtime 对象，存储中位数、下四分位数和上四分位数

    for key,val in model_plain.rt.items():
        mid_rt = np.quantile(val, 0.5)
        q1_rt = np.quantile(val, 0.25)
        q3_rt = np.quantile(val, 0.75)
        mean_rt = sum(val)/len(val)
        
        mid_rt_comm = np.quantile(model_plain.rt_comm[key], 0.5)
        mean_rt_comm = sum(model_plain.rt_comm[key])/len(model_plain.rt_comm[key])
        
        runtime = Runtime(mid_rt, q1_rt, q3_rt, mean_rt, mid_rt_comm, mean_rt_comm)
        runtimes.append(runtime)
            
    print("#################################################################")

    model_crypten = LeNetCrypTen()
    if hasattr(model_crypten, "to"):
        model_crypten = model_crypten.to(device)
    model_enc = model_crypten.encrypt()
    criterion = crypten.nn.CrossEntropyLoss()

    for _ in range(0,epochs):
        model_enc = train_enc(train_loader,model_enc,criterion,lr,device)

    for key,val in model_enc.rt_enc.items():
        mid_rt = np.quantile(val, 0.5)
        q1_rt = np.quantile(val, 0.25)
        q3_rt = np.quantile(val, 0.75)
        mean_rt = sum(val)/len(val)
        
        mid_rt_comm = np.quantile(model_enc.rt_enc_comm[key], 0.5)
        mean_rt_comm = sum(model_enc.rt_enc_comm[key])/len(model_enc.rt_enc_comm[key])
        
        runtime = Runtime(mid_rt, q1_rt, q3_rt, mean_rt, mid_rt_comm, mean_rt_comm)
        runtimes_enc.append(runtime)
    
    return runtimes,runtimes_enc
    
    
    # data_dir.cleanup()

def run(device):
    runtimes,runtimes_enc = time_training(device)
    layers = ["conv1","relu1","pool1","conv2","relu2","pool2","fc1","relu3","fc2","relu4","fc3"]
    df = pd.DataFrame.from_dict(
        {
            "layer" : layers,
            "runtime": [r.mid for r in runtimes],
            "runtime Q1": [r.q1 for r in runtimes],
            "runtime Q3": [r.q3 for r in runtimes],
            "runtime mean": [r.mean for r in runtimes],
            "commtime": [r.comm for r in runtimes],
            "commtime mean": [r.comm_mean for r in runtimes],
            "runtime crypten": [r.mid for r in runtimes_enc],
            "runtime crypten Q1": [r.q1 for r in runtimes_enc],
            "runtime crypten Q3": [r.q3 for r in runtimes_enc],
            "runtime crypten mean": [r.mean for r in runtimes_enc],
            "commtime crypten": [r.comm for r in runtimes_enc],
            "commtime mean crypten": [r.comm_mean for r in runtimes_enc]
        }
    )
    return df
    
def save(device,path,df):
    if device == "cuda":
        csv_path = os.path.join(path,"net_benchmarks_cuda.csv")
    else:
        csv_path = os.path.join(path,"net_benchmarks.csv")
    df.to_csv(csv_path,index=False)
    
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
    
    res = run(device)
    rank = comm.get().get_rank()
    if rank == 0:
        pd.set_option("display.precision", 3)
        print(res)
        if args.path:
            save(args.device,args.path,res)

    # benchmarks = [
    #     FuncBenchmarks(device=device),
    #     ModelBenchmarks(device=device, advanced_models=args.advanced_models),
    # ]

    # if args.only_functions:
    #     benchmarks = [FuncBenchmarks(device=device)]

    # for benchmark in benchmarks:
    #     benchmark.run()
    #     rank = comm.get().get_rank()
    #     if rank == 0:
    #         pd.set_option("display.precision", 3)
    #         print(benchmark)
    #         if args.path:
    #             benchmark.save(args.path)


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
        res = run(args.device)
        pd.set_option("display.precision", 3)
        print(res)
        if args.path:
            save(args.device,args.path,res)


if __name__ == "__main__":
    main()
