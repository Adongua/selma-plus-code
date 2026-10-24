#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import uuid

import crypten


class MultiProcessLauncher:

    # run_process_fn will be run in subprocesses.
    def __init__(self, world_size, run_process_fn, fn_args=None):
        """
        world_size：MPC中的对等方数量（用子进程来模拟）
        run_process_fn：一个函数，将在每个子进程中运行。这是使用者自定义的函数
        fn_args：可选参数，传递给run_process_fn的参数
        """
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)
        multiprocessing.set_start_method("spawn") # 设置multiprocessing模块的启动方式为spawn，这个方法会启动新的Python解释器，从而确保在不同的进程中运行代码

        # Use random file so multiple jobs can be run simultaneously
        # 生成一个随机的文件路径作为初始化方法(INIT_METHOD)，并将其存储在env字典的RENDEZVOUS键上。这个初始化方法将用于多个进程之间的通信
        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1()) # uuid.uuid1():由 MAC 地址（主机物理地址）、当前时间戳、随机数生成。可以保证全球范围内的唯一性
        env["RENDEZVOUS"] = INIT_METHOD

        # 添加进程
        self.processes = []
        for rank in range(world_size):
            process_name = "process " + str(rank)
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(rank, world_size, env, run_process_fn, fn_args),
            )
            self.processes.append(process)

        # 第三方服务器
        if crypten.mpc.ttp_required():
            ttp_process = multiprocessing.Process(
                target=self.__class__._run_process,
                name="TTP",
                args=(
                    world_size,
                    world_size,
                    env,
                    crypten.mpc.provider.TTPServer,
                    None,
                ),
            )
            self.processes.append(ttp_process)

    @classmethod
    def _run_process(cls, rank, world_size, env, run_process_fn, fn_args):
        """
        rank：子进程的等级或位置
        world_size：进程的总数
        env：子进程中的环境变量
        run_process_fn：要在子进程中运行的函数
        fn_args：由用户传递的参数
        """
        for env_key, env_value in env.items():
            os.environ[env_key] = env_value # 将env字典中的环境变量设为当前进程的环境变量。
        os.environ["RANK"] = str(rank) # 将env字典中的环境变量设为当前进程的环境变量。
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO) # 设置日志级别为INFO，以便在日志中看到子进程的输出
        crypten.init() # 调用crypten.init()初始化Crypten分布式加密框架！！！
        logging.getLogger().setLevel(orig_logging_level) # 恢复原始的日志级别
        # 如果fn_args为None，则直接调用run_process_fn；否则，将fn_args作为参数传递给run_process_fn
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

    # 启动所有子进程
    def start(self):
        for process in self.processes:
            process.start()

    # 等待所有子进程完成
    def join(self):
        for process in self.processes:
            process.join()
            assert (
                process.exitcode == 0
            ), f"{process.name} has non-zero exit code {process.exitcode}"

    # 终止所有子进程
    def terminate(self):
        for process in self.processes:
            process.terminate()
