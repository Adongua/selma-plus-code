#!/usr/bin/env python3

"""
网络模拟器模块
用于模拟WAN环境的网络延迟和带宽限制
"""

import time
import threading
import queue
from typing import Any, Optional
import torch
import torch.distributed as dist


class NetworkSimulator:
    """
    网络模拟器类，用于模拟WAN环境的网络条件
    
    Args:
        bandwidth_mbps (float): 带宽限制，单位Mbps
        rtt_ms (float): 往返时延，单位毫秒
        enabled (bool): 是否启用网络模拟
    """
    
    def __init__(self, bandwidth_mbps: float = 392.0, rtt_ms: float = 70.0, enabled: bool = True):
        self.bandwidth_mbps = bandwidth_mbps
        self.rtt_ms = rtt_ms
        self.enabled = enabled
        
        # 计算传输延迟
        self.bandwidth_bytes_per_second = bandwidth_mbps * 1024 * 1024 / 8  # 转换为字节/秒
        self.rtt_seconds = rtt_ms / 1000.0  # 转换为秒
        
        # 用于存储延迟的队列
        self._delay_queue = queue.Queue()
        self._lock = threading.Lock()
        
        print(f"网络模拟器初始化: 带宽={bandwidth_mbps}Mbps, RTT={rtt_ms}ms, 启用={enabled}")
    
    def calculate_transmission_delay(self, data_size_bytes: int) -> float:
        """
        根据数据大小计算传输延迟
        
        Args:
            data_size_bytes (int): 数据大小（字节）
            
        Returns:
            float: 传输延迟（秒）
        """
        if not self.enabled or self.bandwidth_bytes_per_second <= 0:
            return 0.0
        
        # 传输延迟 = 数据大小 / 带宽
        transmission_delay = data_size_bytes / self.bandwidth_bytes_per_second
        
        # 总延迟 = 传输延迟 + RTT/2 (单向延迟)
        total_delay = transmission_delay + self.rtt_seconds / 2
        
        return total_delay
    
    def simulate_network_delay(self, data_size_bytes: int):
        """
        模拟网络延迟
        
        Args:
            data_size_bytes (int): 数据大小（字节）
        """
        if not self.enabled:
            return
        
        delay = self.calculate_transmission_delay(data_size_bytes)
        if delay > 0:
            time.sleep(delay)
    
    def get_tensor_size_bytes(self, tensor: torch.Tensor) -> int:
        """
        计算张量的字节大小
        
        Args:
            tensor (torch.Tensor): 输入张量
            
        Returns:
            int: 张量字节大小
        """
        if tensor is None:
            return 0
        
        # 计算张量的总字节数
        return tensor.numel() * tensor.element_size()
    
    def simulate_tensor_operation(self, operation_func, *args, **kwargs):
        """
        模拟张量操作的网络延迟
        
        Args:
            operation_func: 要执行的操作函数
            *args: 操作函数的参数
            **kwargs: 操作函数的关键字参数
            
        Returns:
            操作函数的返回值
        """
        if not self.enabled:
            return operation_func(*args, **kwargs)
        
        # 计算所有张量参数的总大小
        total_size = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                total_size += self.get_tensor_size_bytes(arg)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        total_size += self.get_tensor_size_bytes(item)
        
        # 模拟网络延迟
        self.simulate_network_delay(total_size)
        
        # 执行原始操作
        return operation_func(*args, **kwargs)


class NetworkSimulatedCommunicator:
    """
    带网络模拟的通信器包装类
    """
    
    def __init__(self, communicator, network_simulator: NetworkSimulator):
        self.communicator = communicator
        self.network_simulator = network_simulator
    
    def __getattr__(self, name):
        """代理所有其他属性到原始通信器"""
        return getattr(self.communicator, name)
    
    def __getstate__(self):
        """支持pickle序列化"""
        return {'communicator': self.communicator, 'network_simulator': self.network_simulator}
    
    def __setstate__(self, state):
        """支持pickle反序列化"""
        self.communicator = state['communicator']
        self.network_simulator = state['network_simulator']
    
    def send(self, tensor, dst, group=None, batched=False):
        """发送张量，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.send(tensor, dst, group, batched)
        
        # 计算数据大小
        if batched and isinstance(tensor, list):
            total_size = sum(self.network_simulator.get_tensor_size_bytes(t) for t in tensor)
        else:
            total_size = self.network_simulator.get_tensor_size_bytes(tensor)
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(total_size)
        
        # 执行发送操作
        return self.communicator.send(tensor, dst, group, batched)
    
    def recv(self, tensor, src, group=None, batched=False):
        """接收张量，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.recv(tensor, src, group, batched)
        
        # 计算数据大小
        if batched and isinstance(tensor, list):
            total_size = sum(self.network_simulator.get_tensor_size_bytes(t) for t in tensor)
        else:
            total_size = self.network_simulator.get_tensor_size_bytes(tensor)
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(total_size)
        
        # 执行接收操作
        return self.communicator.recv(tensor, src, group, batched)
    
    def reduce(self, input, op, group=None, batched=False):
        """规约操作，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.reduce(input, op, group, batched)
        
        # 计算数据大小
        if batched and isinstance(input, list):
            total_size = sum(self.network_simulator.get_tensor_size_bytes(t) for t in input)
        else:
            total_size = self.network_simulator.get_tensor_size_bytes(input)
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(total_size)
        
        # 执行规约操作
        return self.communicator.reduce(input, op, group, batched)
    
    def all_reduce(self, input, op, group=None, batched=False):
        """全规约操作，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.all_reduce(input, op, group, batched)
        
        # 计算数据大小
        if batched and isinstance(input, list):
            total_size = sum(self.network_simulator.get_tensor_size_bytes(t) for t in input)
        else:
            total_size = self.network_simulator.get_tensor_size_bytes(input)
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(total_size)
        
        # 执行全规约操作
        return self.communicator.all_reduce(input, op, group, batched)
    
    def broadcast(self, input, src, group=None, batched=False):
        """广播操作，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.broadcast(input, src, group, batched)
        
        # 计算数据大小
        if batched and isinstance(input, list):
            total_size = sum(self.network_simulator.get_tensor_size_bytes(t) for t in input)
        else:
            total_size = self.network_simulator.get_tensor_size_bytes(input)
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(total_size)
        
        # 执行广播操作
        return self.communicator.broadcast(input, src, group, batched)
    
    def gather(self, tensor, dst):
        """收集操作，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.gather(tensor, dst)
        
        # 计算数据大小
        total_size = self.network_simulator.get_tensor_size_bytes(tensor)
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(total_size)
        
        # 执行收集操作
        return self.communicator.gather(tensor, dst)
    
    def all_gather(self, tensor):
        """全收集操作，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.all_gather(tensor)
        
        # 计算数据大小
        total_size = self.network_simulator.get_tensor_size_bytes(tensor)
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(total_size)
        
        # 执行全收集操作
        return self.communicator.all_gather(tensor)
    
    def send_obj(self, obj, dst, group=None):
        """发送对象，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.send_obj(obj, dst, group)
        
        # 估算对象大小（序列化后）
        import pickle
        try:
            serialized = pickle.dumps(obj)
            data_size = len(serialized)
        except:
            data_size = 1024  # 默认大小
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(data_size)
        
        # 执行发送操作
        return self.communicator.send_obj(obj, dst, group)
    
    def recv_obj(self, src, group=None):
        """接收对象，带网络延迟模拟"""
        if not self.network_simulator.enabled:
            return self.communicator.recv_obj(src, group)
        
        # 估算对象大小
        data_size = 1024  # 默认大小
        
        # 模拟网络延迟
        self.network_simulator.simulate_network_delay(data_size)
        
        # 执行接收操作
        return self.communicator.recv_obj(src, group)
