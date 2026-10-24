#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import crypten
import torch
from examples.meters import AverageMeter


def train_linear_svm(features, labels, epochs=50, lr=0.5, print_time=False):
    """
    features：训练数据集的特征，维度为(D, N)，其中D为每个样本的特征维度，N为样本数量
    labels：训练数据集的标签，维度为(N,)，其中每个元素表示对应样本的类别标签
    epochs：迭代次数，默认为50
    lr：学习率，默认为0.5
    print_time：是否打印训练时间信息，默认为False
    """
    # Initialize random weights
    w = features.new(torch.randn(1, features.size(0))) # (1*D)，tensor取值均值为0，方差为1的正态分布中随机采样得到的数
    b = features.new(torch.randn(1)) # (1)，tensor取值均值为0，方差为1的正态分布中随机采样得到的数
    # features.new() 是一个用于创建与给定张量 features 具有相同设备和数据类型的新张量的方法。它的作用是创建一个新的张量，其设备和数据类型与给定张量 features 保持一致

    if print_time:
        pt_time = AverageMeter()
        end = time.time()

    for epoch in range(epochs):
        # Forward：通过计算线性模型的输出，得到预测的标签结果
        label_predictions = w.matmul(features).add(b).sign() # wX+b，w:1*D,X:D*N,b:1

        # Compute accuracy：根据预测结果和真实标签，计算模型的准确率
        correct = label_predictions.mul(labels) # 同-1/1则1，否则为-1
        accuracy = correct.add(1).div(2).mean() #-1+1=0,1+1=2 -> 0,1 -> mean
        # 如果使用了加密计算，在此处获取准确率的明文值
        if crypten.is_encrypted_tensor(accuracy):
            accuracy = accuracy.get_plain_text()

        # Print Accuracy once
        # 如果当前进程的排名为0，即主进程，将打印每个迭代的准确率
        if crypten.communicator.get().get_rank() == 0:
            print(
                f"Epoch {epoch} --- Training Accuracy %.2f%%" % (accuracy.item() * 100)
            )

        # Backward
        # 反向传播：根据准确率计算损失函数的梯度，并按照梯度下降更新模型的参数
        loss_grad = -labels * (1 - correct) * 0.5  # Hinge loss：如果预测正确，则对应位置的元素为0，否则为真实标签对应位置的相反数
        b_grad = loss_grad.mean() # 1
        w_grad = loss_grad.matmul(features.t()).div(loss_grad.size(1)) # (1*D)=(1*N) mul (N*D)

        # Update
        w -= w_grad * lr
        b -= b_grad * lr

        if print_time:
            iter_time = time.time() - end
            pt_time.add(iter_time)
            logging.info("    Time %.6f (%.6f)" % (iter_time, pt_time.value()))
            end = time.time()

    return w, b


def evaluate_linear_svm(features, labels, w, b):
    """
    Compute accuracy on a test set
    features：训练数据集的特征，维度为(D, N)，其中D为每个样本的特征维度，N为样本数量
    labels：训练数据集的标签，维度为(N,)，其中每个元素表示对应样本的类别标签
    w：train return w
    b：train return b
    """
    predictions = w.matmul(features).add(b).sign()
    correct = predictions.mul(labels)
    accuracy = correct.add(1).div(2).mean().get_plain_text()
    if crypten.communicator.get().get_rank() == 0:
        print("Test accuracy %.2f%%" % (accuracy.item() * 100))


def run_mpc_linear_svm(epochs=50, examples=50, features=100, lr=0.5, skip_plaintext=False):
    """
    epochs: 训练轮数，控制训练过程的迭代次数
    examples: 样本数量，表示训练数据中包含多少个样本
    features: 特征数量，表示每个样本包含多少个特征
    lr: 学习率，控制模型训练时的步长大小
    skip_plaintext: 是否跳过纯文本（PyTorch）实现的训练及输出。如果设置为True，则只运行CrypTen实现，并输出相应的加密结果
    """
    crypten.init() # 调用crypten.init()方法初始化CrypTen框架

    # Set random seed for reproducibility，通过torch.manual_seed(1)设置随机种子以确保结果可重复
    torch.manual_seed(1)

    # Initialize x, y, w, b
    x = torch.randn(features, examples) # 生成一个形状为(features, examples)的随机张量x，表示输入特征。其中features是特征的维度，examples是样本的数量
    w_true = torch.randn(1, features) # 生成一个形状为(1, features)的随机张量w_true，表示权重。其中1表示输出维度为1，features表示权重的维度，与输入特征的维度对应
    b_true = torch.randn(1) # 生成一个形状为(1,)的随机张量b_true，表示偏置
    y = w_true.matmul(x) + b_true # 计算目标值y。通过将权重w_true与输入特征x进行矩阵乘法，并加上偏置b_true，得到目标值y
    y = y.sign() # 对目标值y进行符号函数操作，将其转换为{-1, 1}之间的值。这是因为该线性回归模型是一个二分类模型，将目标值转换为{-1, 1}来表示两个类别

    if not skip_plaintext:
        logging.info("==================")
        logging.info("PyTorch Training")
        logging.info("==================")
        w_torch, b_torch = train_linear_svm(x, y, lr=lr, print_time=True)

    # Encrypt features / labels
    x = crypten.cryptensor(x)
    y = crypten.cryptensor(y)

    logging.info("==================")
    logging.info("CrypTen Training")
    logging.info("==================")
    w, b = train_linear_svm(x, y, lr=lr, print_time=True)

    if not skip_plaintext:
        logging.info("PyTorch Weights  :")
        logging.info(w_torch)
    logging.info("CrypTen Weights:")
    logging.info(w.get_plain_text())

    if not skip_plaintext:
        logging.info("PyTorch Bias  :")
        logging.info(b_torch)
    logging.info("CrypTen Bias:")
    logging.info(b.get_plain_text())
