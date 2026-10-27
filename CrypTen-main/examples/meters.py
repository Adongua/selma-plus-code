#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class AverageMeter:
    """Measures average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value, n=1):
        self.sum += value * n
        self.count += n

    def value(self):
        return self.sum / self.count


class AccuracyMeter:
    """Measures top-k accuracy of multi-class predictions."""

    def __init__(self, topk=(1,)):
        self.reset()
        self.topk = topk
        self.maxk = max(self.topk)

    def reset(self):
        self.values = []

    def add(self, output, ground_truth):

        # compute predicted classes (ordered):
        _, prediction = output.topk(self.maxk, 1, True, True)
        prediction = prediction.t()

        # store correctness values:
        # ground_truth.view(1,-1)是对ground_truth张量进行形状变换的操作。在PyTorch中，可以使用.view()方法来改变张量的形状。其中的参数表示新的维度大小，-1表示自动推断该维度的大小。
        # 在这里，ground_truth.view(1,-1)将ground_truth张量的形状变换为一个行向量（1行，自动推断列数）。具体来说，如果ground_truth张量的原始形状是(m,n)，那么ground_truth.view(1,-1)将其变换为一个行向量，形状为(1,m*n)。
        correct = prediction.eq(ground_truth.view(1, -1).expand_as(prediction))
        self.values.append(correct[: self.maxk])

    def value(self):
        result = {}
        correct = torch.stack(self.values, 0) # values是一个tensor list，stack用于将tensor堆叠变成一个(tensor.shape,list.length)的tensor
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # 从correct张量中取前k个元素，并将其展平为一维张量。然后将其转换为浮点型，并进行求和操作，得到top-k正确的数量。这里使用keepdim=True保持结果张量的维度和原始正确张量correct的维度一致。
            result[k] = correct_k.mul_(100.0 / correct.size(0))
        return result
