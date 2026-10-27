#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains data used for training / testing model benchmarks
"""

import os
from pathlib import Path

import crypten
import PIL
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from examples.util import NoopContextManager
import tempfile


class GaussianClusters:
    """Generates Glussian clusters for binary classes"""

    def __init__(self, n_samples=5000, n_features=20):
        self.n_samples = n_samples
        self.n_features = n_features
        x, x_test, y, y_test = GaussianClusters.generate_data(n_samples, n_features)
        self.x, self.y = x, y
        self.x_test, self.y_test = x_test, y_test

    @staticmethod
    def generate_data(n_samples, n_features):
        """Generates Glussian clusters for binary classes

        Args:
            n_samples (int): number of samples
            n_features (int): number of features

        Returns: torch tensors with inputs and labels
        """
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            # by default, 2 features are redundant
            n_informative=n_features - 2,
            n_classes=2,
        )
        x = torch.tensor(x).float()
        y = torch.tensor(y).float().unsqueeze(-1)

        return train_test_split(x, y)


# class Images:
#     def __init__(self):
#         self.x = self.preprocess_image()
#         # image net 1k classes
#         class_id = 463
#         self.y = torch.tensor([class_id]).long()
#         self.y_onehot = F.one_hot(self.y, 1000)
#         self.x_test, self.y_test = self.x, self.y

#     def preprocess_image(self):
#         """Preprocesses sample image"""
#         path = os.path.dirname(os.path.realpath(__file__))
#         filename = "dog.jpg"
#         input_image = PIL.Image.open(Path(os.path.join(path, filename)))
#         preprocess = transforms.Compose(
#             [
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                 ),
#             ]
#         )
#         input_tensor = preprocess(input_image)
#         input_batch = input_tensor.unsqueeze(0)
#         return input_batch


class Images:
    def __init__(self,context_manager=None):
        if context_manager is None:
            context_manager = NoopContextManager()
    
        data_dirname = "/mnt/data/tmpd5wtqqcd/" # data_dir = tempfile.TemporaryDirectory()
        self.train_loader, self.test_loader = self.preprocess_data(context_manager, data_dirname) # 得到训练数据加载器和验证数据加载器
    
    def preprocess_data(self,context_manager,data_dirname):
        # 定义数据转换管道，用于对CIFAR10数据集进行预处理操作：将CIFAR10数据集中的图像数据转换为张量，并对每个通道的像素值进行标准化处理
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
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
            trainset,batch_size=4,shuffle=True,num_workers=2 # mytest
        )
        # shuffle=True：这个参数表示是否要对数据进行随机重排（洗牌）。设置为True，则数据在每个epoch训练之前会被随机打乱顺序，从而增加模型学习的多样性
        # num_workers = 2：这个参数定义了用于数据加载的工作线程数。在这里，使用了2个工作线程来并行加载数据，可以提高数据加载的效率
        testloader = torch.utils.data.DataLoader(
            testset,batch_size=4,shuffle=True,num_workers=2
        )
        return trainloader,testloader
    