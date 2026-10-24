#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 这个example并不是对加密后的模型进行直接的训练，而是训练的明文的模型，然后转成加密的模型，进行评估之类的操作
import logging
import os
import random
import shutil
import tempfile
import time

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from examples.meters import AverageMeter
from examples.util import NoopContextManager
from torchvision import datasets, transforms


def run_mpc_cifar(
    epochs=25,
    start_epoch=0,
    batch_size=1,
    lr=0.001,
    momentum=0.9,
    weight_decay=1e-6,
    print_freq=10,
    model_location="",
    resume=False,
    evaluate=True,
    seed=None,
    skip_plaintext=False,
    context_manager=None,
):
    """
    epochs：训练轮数
    start_epoch：起始轮数
    batch_size：批次大小
    lr：学习率
    momentum：动量参数
    weight_decay：权重衰减系数
    print_freq：输出训练和验证结果的频率
    model_location：模型文件路径
    resume：是否从checkpoint恢复训练
    evaluate：是否只进行模型评估
    seed：随机数种子
    skip_plaintext：是否跳过明文训练和验证
    context_manager：多方计算环境管理器

    cifar10数据集：60000*32*32 RGB
    """
    # 如果有设置随机种子，则使用该种子对随机数生成器进行初始化
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # 使用crypten.init()初始化多方计算环境
    crypten.init()

    # create model：创建LeNet模型，并定义交叉熵损失函数和SGD优化器
    model = LeNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # optionally resume from a checkpoint：
    best_prec1 = 0 # 定义best_prec1变量来记录最佳准确率，如果需要恢复训练，则将其设置为checkpoint中记录的最佳准确率
    # 如果设置了resume参数为True，则从指定路径加载checkpoint以恢复之前的训练进度。加载模型参数和优化器状态
    if resume:
        if os.path.isfile(model_location):
            logging.info("=> loading checkpoint '{}'".format(model_location))
            checkpoint = torch.load(model_location)
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    model_location, checkpoint["epoch"]
                )
            )
        else:
            raise IOError("=> no checkpoint found at '{}'".format(model_location))

    # Data loading code：通过preprocess_data()函数进行数据预处理和加载。这个函数内部使用了上下文管理器，确保在使用数据集时进行正确的加密和解密操作。
    def preprocess_data(context_manager, data_dirname):
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
                data_dirname, train=True, download=True, transform=transform
            )
            testset = datasets.CIFAR10(
                data_dirname, train=False, download=True, transform=transform
            )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2
        )
        # shuffle=True：这个参数表示是否要对数据进行随机重排（洗牌）。设置为True，则数据在每个epoch训练之前会被随机打乱顺序，从而增加模型学习的多样性
        # num_workers = 2：这个参数定义了用于数据加载的工作线程数。在这里，使用了2个工作线程来并行加载数据，可以提高数据加载的效率
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return trainloader, testloader

    if context_manager is None:
        context_manager = NoopContextManager()

    data_dir = tempfile.TemporaryDirectory()
    train_loader, val_loader = preprocess_data(context_manager, data_dir.name) # 得到训练数据加载器和验证数据加载器

    # 如果evaluate参数为True，则会进行模型评估
    if evaluate:
        if not skip_plaintext:
            logging.info("===== Evaluating plaintext LeNet network =====")
            validate(val_loader, model, criterion, print_freq) # 调用validate()函数对明文下的LeNet模型进行评估
        logging.info("===== Evaluating Private LeNet network =====")
        input_size = get_input_size(val_loader, batch_size)
        private_model = construct_private_model(input_size, model) # 加密
        validate(val_loader, private_model, criterion, print_freq) # 调用validate()函数对密文下的私有LeNet模型进行评估
        # logging.info("===== Validating side-by-side ======")
        # validate_side_by_side(val_loader, model, private_model)
        return

    # define loss function (criterion) and optimizer
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr) # 调整学习率

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_freq) # train()函数对训练集进行训练

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, print_freq) # validate()函数在验证集上评估模型性能

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "LeNet",
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )
    data_dir.cleanup()


def train(train_loader, model, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.add(loss.item(), input.size(0))
        top1.add(prec1[0], input.size(0))
        top5.add(prec5[0], input.size(0))

        # compute gradient and do SGD step：按pytorch那套走
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        current_batch_time = time.time() - end
        batch_time.add(current_batch_time)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                "Epoch: [{}][{}/{}]\t"
                "Time {:.3f} ({:.3f})\t"
                "Loss {:.4f} ({:.4f})\t"
                "Prec@1 {:.3f} ({:.3f})\t"
                "Prec@5 {:.3f} ({:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    current_batch_time,
                    batch_time.value(),
                    loss.item(),
                    losses.value(),
                    prec1[0],
                    top1.value(),
                    prec5[0],
                    top5.value(),
                )
            )


def validate_side_by_side(val_loader, plaintext_model, private_model):
    """
    Validate the plaintext and private models side-by-side on each example
    用于对明文模型（plaintext_model）和私有模型（private_model）进行并行验证
    """
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output for plaintext
            output_plaintext = plaintext_model(input)
            # encrypt input and compute output for private
            # assumes that private model is encrypted with src=0
            input_encr = encrypt_data_tensor_with_src(input)
            output_encr = private_model(input_encr)
            # log all info
            logging.info("==============================")
            logging.info("Example %d\t target = %d" % (i, target))
            logging.info("Plaintext:\n%s" % output_plaintext)
            logging.info("Encrypted:\n%s\n" % output_encr.get_plain_text())
            # only use the first 1000 examples
            if i > 1000:
                break


def get_input_size(val_loader, batch_size):
    input, target = next(iter(val_loader)) # 使用iter()函数将验证集加载器转换为迭代器对象，并使用next()函数获取下一个批次的数据。这里假设每个批次包含一个输入图像和对应的目标标签
    return input.size()


def construct_private_model(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = LeNet()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model


def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size())
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input


def validate(val_loader, model, criterion, print_freq=10):
    """
    val_loader：数据加载器，用于在验证集上批量加载图像数据
    model：待评估的模型
    criterion：损失函数，用于计算模型预测结果和真实标签之间的误差
    print_freq=10：指定打印日志的频率
    """
    batch_time = AverageMeter() # 用于对这些数据求平均
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # 禁用梯度计算
    with torch.no_grad():
        end = time.time() # 记录当前时间
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                input
            ):
                input = encrypt_data_tensor_with_src(input) # 如果模型是crypten.nn.Module类型（使用加密计算），且输入张量input不是加密张量，则使用encrypt_data_tensor_with_src()函数将输入张量进行加密
            # compute output
            output = model(input) # 将输入图像传递给模型进行前向推理，得到预测输出
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text() # 如果预测输出是加密张量，则使用get_plain_text()函数获取其明文表示
            loss = criterion(output, target) # 计算预测输出和真实标签之间的损失值

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5)) # 计算预测输出和真实标签之间的Top-1和Top-5精度
            losses.add(loss.item(), input.size(0)) # 将当前批次的损失值添加到losses对象中
            top1.add(prec1[0], input.size(0)) # 将当前批次的Top-1精度添加到top1对象中
            top5.add(prec5[0], input.size(0))

            # measure elapsed time
            current_batch_time = time.time() - end # 计算当前批次的处理时间
            batch_time.add(current_batch_time) # 将当前批次的处理时间添加到batch_time对象中
            end = time.time() # 记录当前时间，为下一个批次的处理做准备

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec5[0],
                        top5.value(),
                    )
                )

        logging.info(
            " * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value())
        )
    return top1.value()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint of plaintext model"""
    # only save from rank 0 process to avoid race condition
    rank = comm.get().get_rank()
    if rank == 0:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // 5)) # 根据当前epoch计算新的学习率。每隔5个epoch，学习率会按照0.1的倍数进行衰减
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # tensor.topk(k)，返回tensor的最大的前k个值
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
