#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains models used for benchmarking
"""


from dataclasses import dataclass
from typing import Any

import crypten
import torch
import torch.nn as nn
from torchvision import models

# from dp_noise_mpc import add_noise


try:
    from . import data
except ImportError:
    # direct import if relative fails
    import data


N_FEATURES = 20


@dataclass
class Model:
    name: str
    plain: torch.nn.Module
    crypten: crypten.nn.Module
    # must contains x, y, x_test, y_test attributes
    data: Any
    epochs: int
    lr: float
    loss: str
    advanced: bool


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class LogisticRegressionCrypTen(crypten.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear = crypten.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).sigmoid()


class FeedForward(torch.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_features, n_features // 2)
        self.linear2 = torch.nn.Linear(n_features // 2, n_features // 4)
        self.linear3 = torch.nn.Linear(n_features // 4, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.relu(self.linear2(out))
        out = torch.sigmoid(self.linear3(out))
        return out


class FeedForwardCrypTen(crypten.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear1 = crypten.nn.Linear(n_features, n_features // 2)
        self.linear2 = crypten.nn.Linear(n_features // 2, n_features // 4)
        self.linear3 = crypten.nn.Linear(n_features // 4, 1)

    # mytest optim
    def forward(self, x):
        out = (self.linear1(x)).relu() # relu() 
        # add_noise(out)
        # print("out1:",out.get_plain_text())
        out = (self.linear2(out)).relu() # relu()
        # add_noise(out)
        # print("out2:",out.get_plain_text())
        out = (self.linear3(out)).sigmoid()
        # add_noise(out)
        # print("out3",out.get_plain_text())
        return out

class Residual(torch.nn.Module):
    def __init__(self, input_channels, output_channels,use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = torch.nn.Conv2d(output_channels, output_channels,kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(input_channels, output_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        
    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return torch.relu(y)
class ResNet(torch.nn.Module):
    def __init__(self, input_channels=3, num_classes=10, n_layers=18):
        super().__init__()
        assert n_layers in [18, 34]
        self.model = self._make_model(input_channels,num_classes,n_layers)

    def resnet_block(self, input_channels, output_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, output_channels,use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(output_channels, output_channels))
        return blk    
    
    def _make_model(self, input_channels, num_classes, n_layers=18):
        if n_layers == 18:
            rbs = [2,2,2,2]
        else:
            rbs = [3,4,6,3]
        b1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = torch.nn.Sequential(*self.resnet_block(64, 64, rbs[0], first_block=True))
        b3 = torch.nn.Sequential(*self.resnet_block(64, 128, rbs[1]))
        b4 = torch.nn.Sequential(*self.resnet_block(128, 256, rbs[2]))
        b5 = torch.nn.Sequential(*self.resnet_block(256, 512, rbs[3]))
        net = torch.nn.Sequential(
            b1, b2, b3, b4, b5, torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten(), torch.nn.Linear(512, num_classes)
        )
        return net
        
    def forward(self, x):
        return self.model(x)
    
class Residual50(torch.nn.Module):
    def __init__(self, input_channels, output_channels, downsample=None, strides=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels,output_channels,kernel_size=1)
        self.conv2 = torch.nn.Conv2d(output_channels,output_channels,kernel_size=3, padding=1,stride=strides)
        self.conv3 = torch.nn.Conv2d(output_channels,output_channels*4,kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        self.bn3 = torch.nn.BatchNorm2d(output_channels * 4)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        y = torch.relu(self.bn1(self.conv1(x)))
        y = torch.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        y = y + identity
        return torch.relu(y)
class ResNet50(torch.nn.Module):
    def __init__(self, input_channels=3, num_classes=10, n_layers=50):
        super().__init__()
        assert n_layers in [50]
        self.model = self._make_model(input_channels,num_classes,n_layers)

    def resnet_block(self, input_channels, output_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 :
                if first_block == False:
                    downsample = torch.nn.Sequential(
                        torch.nn.Conv2d(input_channels, output_channels * 4, kernel_size=1, stride=2),
                        torch.nn.BatchNorm2d(output_channels * 4)
                    )
                    blk.append(Residual50(input_channels, output_channels, downsample, strides=2))
                else:
                    downsample = torch.nn.Sequential(
                        torch.nn.Conv2d(input_channels, output_channels * 4, kernel_size=1, stride=1),
                        torch.nn.BatchNorm2d(output_channels * 4)
                    )
                    blk.append(Residual50(output_channels, output_channels, downsample))
            else:
                    blk.append(Residual50(output_channels*4, output_channels))
        return blk    
    
    def _make_model(self, input_channels, num_classes, n_layers=50):
        if n_layers == 50:
            rbs = [3,4,6,3]
        b1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = torch.nn.Sequential(*self.resnet_block(64, 64, rbs[0], first_block=True))
        b3 = torch.nn.Sequential(*self.resnet_block(256, 128, rbs[1]))
        b4 = torch.nn.Sequential(*self.resnet_block(512, 256, rbs[2]))
        b5 = torch.nn.Sequential(*self.resnet_block(1024, 512, rbs[3]))
        net = torch.nn.Sequential(
            b1, b2, b3, b4, b5, torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten(), torch.nn.Linear(2048, num_classes)
        )
        return net
        
    def forward(self, x):
        return self.model(x)


# class ResidualCrypTen(crypten.nn.Module):
#     def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1, dp=False):
#         super().__init__()
#         self.conv1 = crypten.nn.Conv2d(input_channels, output_channels,kernel_size=3, padding=1, stride=strides)
#         self.conv2 = crypten.nn.Conv2d(output_channels, output_channels,kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = crypten.nn.Conv2d(input_channels, output_channels,kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None
#         self.bn1 = crypten.nn.BatchNorm2d(output_channels)
#         self.bn2 = crypten.nn.BatchNorm2d(output_channels)
#         self.dp = dp
        
#     def forward(self, x):
#         y = self.bn1(self.conv1(x))
#         y = y.relu()
#         y = self.bn2(self.conv2(y))
#         if self.conv3:
#             x = self.conv3(x)
#         y = y + x
#         if self.dp:
#             add_noise(y.relu())
#         return y.relu()
# class ResNetCrypTen(crypten.nn.Module):
#     def __init__(self, input_channels=3, num_classes=10, n_layers=18):
#         super().__init__()
#         assert n_layers in [18, 34]
#         self.model = self._make_model(input_channels,num_classes,n_layers)

#     def resnet_block(self, input_channels, output_channels, num_residuals, first_block=False, noise_block=False):
#         blk = []
#         for i in range(num_residuals):
#             if i == num_residuals-1 and noise_block:
#                 blk.append(ResidualCrypTen(output_channels, output_channels, dp=True))
#             elif i == 0 and not first_block:
#                 blk.append(ResidualCrypTen(input_channels, output_channels, use_1x1conv=True, strides=2))
#             else:
#                 blk.append(ResidualCrypTen(output_channels, output_channels))
#         return blk    
    
#     def _make_model(self, input_channels, num_classes, n_layers=18):
#         if n_layers == 18:
#             rbs = [2,2,2,2]
#         else:
#             rbs = [3,4,6,3]
#         b1 = crypten.nn.Sequential(
#             crypten.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
#             crypten.nn.BatchNorm2d(64), 
#             crypten.nn.ReLU(),
#             crypten.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         b2 = crypten.nn.Sequential(*self.resnet_block(64, 64, rbs[0], first_block=True))
#         b3 = crypten.nn.Sequential(*self.resnet_block(64, 128, rbs[1], noise_block=False))
#         b4 = crypten.nn.Sequential(*self.resnet_block(128, 256, rbs[2]))
#         b5 = crypten.nn.Sequential(*self.resnet_block(256, 512, rbs[3]))
#         net = crypten.nn.Sequential(
#             b1, b2, b3, b4, b5, crypten.nn.AdaptiveAvgPool2d((1,1)), crypten.nn.Flatten(), crypten.nn.Linear(512, num_classes)
#         )
#         return net
        
#     def forward(self, x):
#         return self.model(x)


class ResidualCrypTen(crypten.nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = crypten.nn.Conv2d(input_channels, output_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = crypten.nn.Conv2d(output_channels, output_channels,kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = crypten.nn.Conv2d(input_channels, output_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = crypten.nn.BatchNorm2d(output_channels)
        self.bn2 = crypten.nn.BatchNorm2d(output_channels)
        
    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = y.relu()
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return y.relu()
class ResNetCrypTen(crypten.nn.Module):
    def __init__(self, input_channels=3, num_classes=10, n_layers=18):
        super().__init__()
        assert n_layers in [18, 34]
        self.model = self._make_model(input_channels,num_classes,n_layers)

    def resnet_block(self, input_channels, output_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResidualCrypTen(input_channels, output_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(ResidualCrypTen(output_channels, output_channels))
        return blk    
    
    def _make_model(self, input_channels, num_classes, n_layers=18):
        if n_layers == 18:
            rbs = [2,2,2,2]
        else:
            rbs = [3,4,6,3]
        b1 = crypten.nn.Sequential(
            crypten.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            crypten.nn.BatchNorm2d(64), 
            crypten.nn.ReLU(),
            crypten.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = crypten.nn.Sequential(*self.resnet_block(64, 64, rbs[0], first_block=True))
        b3 = crypten.nn.Sequential(*self.resnet_block(64, 128, rbs[1]))
        b4 = crypten.nn.Sequential(*self.resnet_block(128, 256, rbs[2]))
        b5 = crypten.nn.Sequential(*self.resnet_block(256, 512, rbs[3]))
        net = crypten.nn.Sequential(
            b1, b2, b3, b4, b5, crypten.nn.AdaptiveAvgPool2d((1,1)), crypten.nn.Flatten(), crypten.nn.Linear(512, num_classes)
        )
        return net
        
    def forward(self, x):
        return self.model(x)
    
    
class Residual50CrypTen(crypten.nn.Module):
    def __init__(self, input_channels, output_channels, downsample=None, strides=1):
        super().__init__()
        self.conv1 = crypten.nn.Conv2d(input_channels,output_channels,kernel_size=1)
        self.conv2 = crypten.nn.Conv2d(output_channels,output_channels,kernel_size=3, padding=1,stride=strides)
        self.conv3 = crypten.nn.Conv2d(output_channels,output_channels*4,kernel_size=1)
        self.bn1 = crypten.nn.BatchNorm2d(output_channels)
        self.bn2 = crypten.nn.BatchNorm2d(output_channels)
        self.bn3 = crypten.nn.BatchNorm2d(output_channels * 4)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        y = self.bn1(self.conv1(x))
        y = y.relu()
        y = self.bn2(self.conv2(y))
        y = y.relu()
        y = self.bn3(self.conv3(y))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        y = y + identity
        return y.relu()
class ResNet50CrypTen(crypten.nn.Module):
    def __init__(self, input_channels=3, num_classes=10, n_layers=50):
        super().__init__()
        assert n_layers in [50]
        self.model = self._make_model(input_channels,num_classes,n_layers)

    def resnet_block(self, input_channels, output_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 :
                if first_block == False:
                    downsample = crypten.nn.Sequential(
                        crypten.nn.Conv2d(input_channels, output_channels * 4, kernel_size=1, stride=2),
                        crypten.nn.BatchNorm2d(output_channels * 4)
                    )
                    blk.append(Residual50CrypTen(input_channels, output_channels, downsample, strides=2))
                else:
                    downsample = crypten.nn.Sequential(
                        crypten.nn.Conv2d(input_channels, output_channels * 4, kernel_size=1, stride=1),
                        crypten.nn.BatchNorm2d(output_channels * 4)
                    )
                    blk.append(Residual50CrypTen(output_channels, output_channels, downsample))
            else:
                    blk.append(Residual50CrypTen(output_channels*4, output_channels))
        return blk    
    
    def _make_model(self, input_channels, num_classes, n_layers=50):
        if n_layers == 50:
            rbs = [3,4,6,3]
        b1 = crypten.nn.Sequential(
            crypten.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            crypten.nn.BatchNorm2d(64), 
            crypten.nn.ReLU(),
            crypten.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = crypten.nn.Sequential(*self.resnet_block(64, 64, rbs[0], first_block=True))
        b3 = crypten.nn.Sequential(*self.resnet_block(256, 128, rbs[1]))
        b4 = crypten.nn.Sequential(*self.resnet_block(512, 256, rbs[2]))
        b5 = crypten.nn.Sequential(*self.resnet_block(1024, 512, rbs[3]))
        net = crypten.nn.Sequential(
            b1, b2, b3, b4, b5, crypten.nn.AdaptiveAvgPool2d((1,1)), crypten.nn.Flatten(), crypten.nn.Linear(2048, num_classes)
        )
        return net
        
    def forward(self, x):
        return self.model(x)
    
# 定义DenseBlock
class DenseBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self.conv_block(input_channels + i * output_channels, output_channels))

    def conv_block(self, input_channels, output_channels):
        return torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        for blk in self.layers:
            y = blk(x)
            x = torch.cat([x,y],dim=1)
        return x

# 定义Transition层
class Transition(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.trans_layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=1),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.trans_layer(x)

class DenseNet121(torch.nn.Module):
    def __init__(self, num_classes=10, growth_rate=32, block_config=(6, 12, 24, 16)):
        super().__init__()
        self.model = self._make_model(num_classes, growth_rate, block_config)
        
    def _make_model(self, num_classes=10, growth_rate=32, block_config=(6, 12, 24, 16)):
        # 初始卷积层
        b1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 4个DenseBlock和3个Transition层
        blks = []
        num_channels = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_channels, growth_rate, num_layers)
            blks.append(block)
            num_channels = num_channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_channels, num_channels // 2)
                blks.append(trans)
                num_channels = num_channels // 2
                
        # 连上最后的全局平均池化层和全连接层
        net = torch.nn.Sequential(
            b1, *blks,
            torch.nn.BatchNorm2d(num_channels), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(num_channels, num_classes)
        )
        return net

    def forward(self, x):
        out = self.model(x)
        return out
    

# 定义DenseBlock
class DenseBlockCrypTen(crypten.nn.Module):
    def __init__(self, input_channels, output_channels, num_layers):
        super().__init__()
        self.layers = crypten.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self.conv_block(input_channels + i * output_channels, output_channels))

    def conv_block(self, input_channels, output_channels):
        return crypten.nn.Sequential(
            crypten.nn.BatchNorm2d(input_channels), 
            crypten.nn.ReLU(),
            crypten.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        for blk in self.layers:
            y = blk(x)
            x = crypten.cat([x,y],dim=1)
        return x

# 定义Transition层
class TransitionCrypTen(crypten.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.trans_layer = crypten.nn.Sequential(
            crypten.nn.BatchNorm2d(input_channels),
            crypten.nn.ReLU(),
            crypten.nn.Conv2d(input_channels, output_channels, kernel_size=1),
            crypten.nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.trans_layer(x)

class DenseNet121CrypTen(crypten.nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16)):
        super().__init__()
        self.model = self._make_model(num_classes, growth_rate, block_config)
        
    def _make_model(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16)):
        # 初始卷积层
        b1 = crypten.nn.Sequential(
            crypten.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            crypten.nn.BatchNorm2d(64),
            crypten.nn.ReLU(),
            crypten.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 4个DenseBlock和3个Transition层
        blks = []
        num_channels = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlockCrypTen(num_channels, growth_rate, num_layers)
            blks.append(block)
            num_channels = num_channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionCrypTen(num_channels, num_channels // 2)
                blks.append(trans)
                num_channels = num_channels // 2
                
        # 连上最后的全局平均池化层和全连接层
        net = crypten.nn.Sequential(
            b1, *blks,
            crypten.nn.BatchNorm2d(num_channels), crypten.nn.ReLU(),
            crypten.nn.AdaptiveAvgPool2d((1, 1)),
            crypten.nn.Flatten(),
            crypten.nn.Linear(num_channels, num_classes)
        )
        return net

    def forward(self, x):
        out = self.model(x)
        return out

class VGG16(torch.nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.model = self._make_model(input_channels, num_classes)
    
    def vgg_block(self, num_convs, input_channels, output_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(torch.nn.Conv2d(input_channels, output_channels,kernel_size=3, padding=1))
            layers.append(torch.nn.ReLU())
            input_channels = output_channels
        layers.append(torch.nn.MaxPool2d(kernel_size=2,stride=2))
        return torch.nn.Sequential(*layers)
    
    def vgg(self, input_channels, num_classes, conv_arch):
        conv_blks = []
        for (num_convs, output_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, input_channels, output_channels))
            input_channels = output_channels

        return torch.nn.Sequential(
            *conv_blks, 
            torch.nn.Flatten(),
            # 全连接层部分
            torch.nn.Linear(output_channels, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.4), # torch.nn.Linear(output_channels * 2 * 2 , 4096), torch.nn.ReLU(), torch.nn.Dropout(0.4), # torch.nn.Linear(output_channels, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.4),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.4),
            torch.nn.Linear(4096, num_classes)
        )
    
    def _make_model(self, input_channels, num_classes):
        conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
        net = self.vgg(input_channels, num_classes, conv_arch)
        return net
    
    def forward(self, x):
        return self.model(x)
    
    
class VGG16CrypTen(crypten.nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.model = self._make_model(input_channels, num_classes)
    
    def vgg_block(self, num_convs, input_channels, output_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(crypten.nn.Conv2d(input_channels, output_channels,kernel_size=3, padding=1))
            layers.append(crypten.nn.BatchNorm2d(output_channels))
            layers.append(crypten.nn.ReLU())
            input_channels = output_channels
        layers.append(crypten.nn.MaxPool2d(kernel_size=2,stride=2))
        return crypten.nn.Sequential(*layers)
    
    def vgg(self, input_channels, num_classes, conv_arch):
        conv_blks = []
        for (num_convs, output_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, input_channels, output_channels))
            input_channels = output_channels

        return crypten.nn.Sequential(
            *conv_blks, 
            crypten.nn.Flatten(),
            # 全连接层部分
            crypten.nn.Linear(output_channels, 4096), crypten.nn.ReLU(), crypten.nn.Dropout(0.4),# crypten.nn.Linear(output_channels * 2 * 2, 4096), crypten.nn.ApproxReLU(), crypten.nn.Dropout(0.4), # crypten.nn.Linear(output_channels, 4096), crypten.nn.ApproxReLU(), crypten.nn.Dropout(0.4),
            crypten.nn.Linear(4096, 4096), crypten.nn.ReLU(), crypten.nn.Dropout(0.4),
            crypten.nn.Linear(4096, num_classes)
        )
    
    def _make_model(self, input_channels, num_classes):
        conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
        net = self.vgg(input_channels, num_classes, conv_arch)
        return net
    
    def forward(self, x):
        return self.model(x)

    
class FireModule(torch.nn.Module):
    def __init__(self, input_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = torch.nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = torch.nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = torch.nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        squeezed = torch.relu(self.squeeze(x))
        expanded1x1 = torch.relu(self.expand1x1(squeezed))
        expanded3x3 = torch.relu(self.expand3x3(squeezed))
        return torch.cat([expanded1x1, expanded3x3], 1)

class SqueezeNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(512, 64, 256, 256)
        )
        self.final_conv = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    

class FireModuleCrypTen(crypten.nn.Module):
    def __init__(self, input_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = crypten.nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = crypten.nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = crypten.nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        squeezed = self.squeeze(x)      
        squeezed = squeezed.relu()
        expanded1x1 = self.expand1x1(squeezed)
        expanded1x1 = expanded1x1.relu()
        expanded3x3 = self.expand3x3(squeezed)
        expanded3x3 = expanded3x3.relu()
        return crypten.cat([expanded1x1, expanded3x3], 1)
class SqueezeNetCrypTen(crypten.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = crypten.nn.Sequential(
            crypten.nn.Conv2d(3, 96, kernel_size=3, stride=1),
            crypten.nn.ReLU(),
            crypten.nn.MaxPool2d(kernel_size=3, stride=2),
            FireModuleCrypTen(96, 16, 64, 64),
            FireModuleCrypTen(128, 16, 64, 64),
            FireModuleCrypTen(128, 32, 128, 128),
            crypten.nn.MaxPool2d(kernel_size=3, stride=2),
            FireModuleCrypTen(256, 32, 128, 128),
            FireModuleCrypTen(256, 48, 192, 192),
            FireModuleCrypTen(384, 48, 192, 192),
            FireModuleCrypTen(384, 64, 256, 256),
            crypten.nn.MaxPool2d(kernel_size=3, stride=2),
            FireModuleCrypTen(512, 64, 256, 256)
        )
        self.final_conv = crypten.nn.Conv2d(512, num_classes, kernel_size=1)
        self.global_avgpool = crypten.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    

MODELS = [
    # Model(
    #     name="logistic regression",
    #     plain=LogisticRegression(),
    #     crypten=LogisticRegressionCrypTen(),
    #     data=data.GaussianClusters(),
    #     epochs=1,
    #     lr=0.1,
    #     loss="BCELoss",
    #     advanced=False,
    # ),
    # Model(
    #     name="feedforward neural network",
    #     plain=FeedForward(),
    #     crypten=FeedForwardCrypTen(),
    #     data=data.GaussianClusters(),
    #     epochs=10,
    #     lr=0.3,
    #     loss="BCELoss",
    #     advanced=False,
    # ),
    # Model(
    #     name="resnet18",
    #     plain=ResNet(n_layers=18),
    #     crypten=ResNetCrypTen(n_layers=18),
    #     data=data.C10Images(),
    #     epochs=1,
    #     lr=1e-4,
    #     loss="CrossEntropyLoss",
    #     advanced=True,
    # ),
    # Model(
    #     name="resnet34",
    #     plain=ResNet(n_layers=34),
    #     crypten=ResNetCrypTen(n_layers=34),
    #     data=data.Images(),
    #     epochs=1,
    #     lr=0.001,
    #     loss="CrossEntropyLoss",
    #     advanced=True,
    # ),
    # Model(
    #     name="resnet50",
    #     plain=ResNet50(n_layers=50),
    #     crypten=ResNet50CrypTen(n_layers=50),
    #     data=data.Images(),
    #     epochs=1,
    #     lr=1e-5,
    #     loss="CrossEntropyLoss",
    #     advanced=True,
    # ),
    # Model(
    #     name="DenseNet121",
    #     plain=DenseNet121(num_classes=200),
    #     crypten=DenseNet121CrypTen(num_classes=200),
    #     data=data.TIImages(),
    #     epochs=1,
    #     lr=1e-4,
    #     loss="CrossEntropyLoss",
    #     advanced=True,
    # ),
    # Model(
    #     name="VGG16",
    #     plain=VGG16(num_classes=10),
    #     crypten=VGG16CrypTen(num_classes=10),
    #     data=data.C10Images(),
    #     epochs=3,
    #     lr=1e-2,
    #     loss="CrossEntropyLoss",
    #     advanced=True,
    # ),
    # Model(
    #     name="SqueezeNet",
    #     plain=SqueezeNet(num_classes=10),
    #     crypten=SqueezeNetCrypTen(num_classes=10),
    #     data=data.C10Images(),
    #     epochs=1,
    #     lr=1e-3,
    #     loss="CrossEntropyLoss",
    #     advanced=True,
    # )
    
]
