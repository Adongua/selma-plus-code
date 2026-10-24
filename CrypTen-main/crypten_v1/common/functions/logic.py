#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
from crypten.common.tensor_types import is_tensor


__all__ = [
    "__eq__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "__ne__",
    "abs",
    "eq",
    "ge",
    "gt",
    "hardtanh",
    "le",
    "lt",
    "ne",
    "relu",
    "sign",
    "where",
]


def ge(self, y):
    """Returns self >= y"""
    return 1 - self.lt(y) # 1-[x<y]


def gt(self, y):
    """Returns self > y"""
    return (-self + y)._ltz() # [y<x] <=> [-x+y<0]


def le(self, y):
    """Returns self <= y"""
    return 1 - self.gt(y) # 1-[y<x]


def lt(self, y):
    """Returns self < y"""
    return (self - y)._ltz() # [x<y] <=> [x-y<0]


def eq(self, y):
    """Returns self == y"""
    return 1 - self.ne(y) # 1-[x!=y]


def ne(self, y):
    """Returns self != y"""
    difference = self - y # x-y，相等则为0
    difference = type(difference).stack([difference, -difference]) # 将差值张量和它的相反数张量在同一个张量中进行排列
    return difference._ltz().sum(0) # 利用_ltz()方法判断difference张量中的元素是否小于零，返回一个布尔类型的张量。然后使用sum(0)对每一列（或者说沿着0维）进行求和操作，最终得到每列中小于零的元素个数，这个结果即为不相等的情况个数


__eq__ = eq
__ge__ = ge
__gt__ = gt
__le__ = le
__lt__ = lt
__ne__ = ne


def sign(self):
    """Computes the sign value of a tensor (0 is considered positive)"""
    return 1 - 2 * self._ltz() # -1,x<0 ;1,x>0


def abs(self):
    """Computes the absolute value of a tensor"""
    return self * self.sign() # x*sign


def relu(self):
    """Compute a Rectified Linear function on the input tensor."""
    return self * self.ge(0) # 0,x<0 ;x,x>=0


def hardtanh(self, min_value=-1, max_value=1):
    r"""Applies the HardTanh function element-wise

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
    """
    intermediate = crypten.stack([self - min_value, self - max_value]).relu()
    intermediate = intermediate[0].sub(intermediate[1])
    return intermediate.add_(min_value)


def where(self, condition, y):
    """Selects elements from self or y based on condition

    Args:
        condition (torch.bool or MPCTensor): when True yield self,
            otherwise yield y
        y (torch.tensor or MPCTensor): values selected at indices
            where condition is False.

    Returns: MPCTensor or torch.tensor
    """
    if is_tensor(condition):
        condition = condition.float()
        y_masked = y * (1 - condition)
    else:
        # encrypted tensor must be first operand
        y_masked = (1 - condition) * y

    return self * condition + y_masked
