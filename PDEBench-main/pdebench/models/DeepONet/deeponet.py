# coding:UTF-8
# @Time: 2023/3/16 10:31
# @Author: Lulu Cao
# @File: deeponet.py
# @Software: PyCharm
import pandas as pd


import scipy.io as scio
import torch
import torch.nn.functional as F

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DeepONet1d(torch.nn.Module):
    def __init__(self,trunk_input,branch_input,width):
        super(DeepONet1d,self).__init__()
        self.width = width

        self.fc1 = torch.nn.Linear(trunk_input, self.width)
        self.fc2 = torch.nn.Linear(self.width, self.width)
        self.fc3 = torch.nn.Linear(self.width, self.width)

        self.fc21 = torch.nn.Linear(branch_input,  self.width)
        self.fc22 = torch.nn.Linear(self.width, self.width)
    def forward(self,x, grid ,position):
        """

        :param param:
        :param position:
        :return:
        """
        # x1 dim = [b, x1, ch]
        batch_size = x.size(0)

        x1 = x.view(batch_size, -1, x.size(-1))
        # x dim = [b, ch, x1]
        x1 = x1.permute(0, 2, 1)


        x1 = self.fc1(x1)
        x1 = F.gelu(x1)

        x1 = self.fc2(x1)
        x1 = F.gelu(x1)

        x1 = self.fc3(x1)
        x1 = F.gelu(x1)



        x2 = position

        x2 = F.gelu(self.fc21(x2))
        x2 = F.gelu(self.fc22(x2))
        outputs = torch.einsum('ncb,nmb->nmc', x1, x2)

        return outputs