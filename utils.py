# coding:UTF-8
# @Time: 2022/10/26 21:20
# @Author: Lulu Cao
# @File: utils.py
# @Software: PyCharm
import os
import torch

# GPU select for DDP training
def select_device(device='', batch_size=0):
    # device = 'cpu' or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        s = ''
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        print(s)                                                                # 之后放入日志记录
    return torch.device('cuda:0' if cuda else 'cpu')


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel) else model