#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.util import NoopContextManager
from torchvision import datasets, transforms


def run_mpc_autograd_cnn(
    context_manager=None,
    num_epochs=3,
    learning_rate=0.001,
    batch_size=5,
    print_freq=5,
    num_samples=100,
):
    """
    Args:
        context_manager: used for setting proxy settings during download.（用于在下载期间设置代理设置）
        num_epochs：训练轮数，控制训练过程的迭代次数
        learning_rate：学习率，控制模型训练时的步长大小
        batch_size：批量大小，表示每次训练中使用的样本数量
        print_freq：打印频率，控制训练过程中的日志打印频率
        num_samples：样本数量，表示训练集中用于训练的样本数量
    """
    crypten.init() # 通过调用crypten.init()方法初始化CrypTen框架

    data_alice, data_bob, train_labels = preprocess_mnist(context_manager) # 预处理MNIST数据集，获得Alice和Bob的数据以及训练标签
    rank = comm.get().get_rank() # 获取当前进程的等级（rank），用于区分不同的参与方

    # assumes at least two parties exist
    # broadcast dummy data with same shape to remaining parties
    # 如果当前进程的等级是0，则将Alice的数据赋值给x_alice，否则创建一个与Alice数据大小相同的空张量
    if rank == 0:
        x_alice = data_alice
    else:
        x_alice = torch.empty(data_alice.size())
    # 如果当前进程的等级是1，则将Bob的数据赋值给x_bob，否则创建一个与Bob数据大小相同的空张量
    if rank == 1:
        x_bob = data_bob
    else:
        x_bob = torch.empty(data_bob.size())

    # encrypt：将Alice和Bob的数据加密，得到加密后的张量x_alice_enc和x_bob_enc
    x_alice_enc = crypten.cryptensor(x_alice, src=0)
    x_bob_enc = crypten.cryptensor(x_bob, src=1)

    # combine feature sets
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2) # 按第三个维度拼接（还原数据的shape，但是已加密）
    x_combined_enc = x_combined_enc.unsqueeze(1) # 在第1个维度上添加一个维度

    # reduce training set to num_samples：将训练集大小减少到num_samples，获取加密后的输入特征张量x_reduced和对应的训练标签y_reduced
    x_reduced = x_combined_enc[:num_samples]
    y_reduced = train_labels[:num_samples]

    # encrypt plaintext model
    model_plaintext = CNN() # 创建一个基于PyTorch的CNN模型model_plaintext
    dummy_input = torch.empty((1, 1, 28, 28)) # 根据输入的形状创建一个空的张量dummy_input作为输入
    model = crypten.nn.from_pytorch(model_plaintext, dummy_input) # 还需要模型输入形状的伪输入dummy_input
    model.train() # 设置为训练模式
    model.encrypt() # 模型加密

    # encrypted training
    train_encrypted(
        x_reduced, y_reduced, model, num_epochs, learning_rate, batch_size, print_freq
    )


def train_encrypted(
    x_encrypted,
    y_encrypted,
    encrypted_model,
    num_epochs,
    learning_rate,
    batch_size,
    print_freq,
):
    """
    x_encrypted：加密后的x
    y_encrypted：加密后的y
    encrypted_model：加密后的模型
    num_epochs:训练代数
    learning_rate:学习率
    batch_size：批量大小
    print_freq：打印频率，控制训练过程中的日志打印频率
    """
    rank = comm.get().get_rank() # 获取当前进程等级
    loss = crypten.nn.MSELoss()

    num_samples = x_encrypted.size(0) # 训练样本数
    label_eye = torch.eye(2) # 创建一个大小为 2 的单位矩阵 label_eye，用于将标签转换为 one-hot 编码形式

    for epoch in range(num_epochs):
        last_progress_logged = 0
        # only print from rank 0 to avoid duplicates for readability
        if rank == 0:
            print(f"Epoch {epoch} in progress:")

        # 在每个训练批次中，遍历训练样本，每次处理一个批次大小（batch_size）的样本
        for j in range(0, num_samples, batch_size):

            # define the start and end of the training mini-batch
            start, end = j, min(j + batch_size, num_samples)

            # switch on autograd for training examples
            x_train = x_encrypted[start:end]
            x_train.requires_grad = True
            y_one_hot = label_eye[y_encrypted[start:end]] # 获取对应的 one-hot 编码的训练标签 y_one_hot
            y_train = crypten.cryptensor(y_one_hot, requires_grad=True)

            # perform forward pass:
            output = encrypted_model(x_train)
            loss_value = loss(output, y_train)

            # backprop
            encrypted_model.zero_grad()
            loss_value.backward()
            encrypted_model.update_parameters(learning_rate)

            # log progress
            if j + batch_size - last_progress_logged >= print_freq:
                last_progress_logged += print_freq
                print(f"Loss {loss_value.get_plain_text().item():.4f}")

        # compute accuracy every epoch：每个训练代数结束后，计算模型的准确率
        pred = output.get_plain_text().argmax(1) # 每个类别的概率值代表了模型预测该类别的置信度
        correct = pred.eq(y_encrypted[start:end])
        correct_count = correct.sum(0, keepdim=True).float() # 沿着0维进行求和
        accuracy = correct_count.mul_(100.0 / output.size(0))  # acc%

        loss_plaintext = loss_value.get_plain_text().item()
        print(
            f"Epoch {epoch} completed: "
            f"Loss {loss_plaintext:.4f} Accuracy {accuracy.item():.2f}"
        )


def preprocess_mnist(context_manager):
    # 检查是否提供了上下文管理器（context_manager），如果没有则创建一个空的NoopContextManager作为默认值
    if context_manager is None:
        context_manager = NoopContextManager()

    # 通过调用tempfile.TemporaryDirectory()方法创建一个临时目录data_dir，并在其中下载并加载MNIST数据集
    with context_manager:
        # each party gets a unique temp directory
        with tempfile.TemporaryDirectory() as data_dir:
            mnist_train = datasets.MNIST(data_dir, download=True, train=True)
            mnist_test = datasets.MNIST(data_dir, download=True, train=False)

    # modify labels so all non-zero digits have class label 1
    mnist_train.targets[mnist_train.targets != 0] = 1
    mnist_test.targets[mnist_test.targets != 0] = 1
    mnist_train.targets[mnist_train.targets == 0] = 0
    mnist_test.targets[mnist_test.targets == 0] = 0

    # compute normalization factors：计算所有数据特征的平均值和标准差
    data_all = torch.cat([mnist_train.data, mnist_test.data]).float()
    data_mean, data_std = data_all.mean(), data_all.std()
    tensor_mean, tensor_std = data_mean.unsqueeze(0), data_std.unsqueeze(0) # 转换为形状为(1,)的张量

    # normalize data：归一化数据
    data_train_norm = transforms.functional.normalize(
        mnist_train.data.float(), tensor_mean, tensor_std
    )

    # partition features between Alice and Bob：将其特征分为两份，分别赋值给Alice和Bob
    data_alice = data_train_norm[:, :, :20]
    print(f"data_alice: {data_alice.shape}")
    data_bob = data_train_norm[:, :, 20:]
    print(f"data_bob: {data_bob.shape}")
    train_labels = mnist_train.targets

    return data_alice, data_bob, train_labels


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0) # 输入通道数为1，输出通道数为16（输出维度为16），卷积核大小为5*5
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
