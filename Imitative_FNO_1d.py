# coding:UTF-8
# @Time: 2022/10/30 16:21
# @Author: Lulu Cao
# @File: Imitative_FNO_1d.py
# @Software: PyCharm

'''
Burgers Equation
'''
import pandas as pd

from utilities3 import *
import scipy.io as scio
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm


torch.manual_seed(0)
np.random.seed(0)



# 1. 准备数据集
class IFNODataset(Dataset):
    def __init__(self,filepath,istrain,number,sub):
    # df.to_csv("../../data/burgers_data_v10_sample"+str(sample_num)+".csv")
        df = pd.read_csv(filepath)
        self.data = torch.from_numpy(df.values)

        if istrain==True:
            self.a = self.data[:number,1:-2]
            self.x = self.data[:number,-2]
            self.x = self.data[:number,-1]

        else:
            self.a = self.data[-number:, :-2]
            self.x = self.data[-number:, -2]
            self.x = self.data[-number:, -1]

        self.len = number
        print(self.a.shape)
        print(self.u.shape)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.a[index],self.u[index],self.grid[:]





# TODO:2. 设计模型

class IFNO1d(torch.nn.Module):
    def __init__(self,fft_input_size,position_size):
        super(IFNO1d,self).__init__()
        self.fc0 = nn.Linear(fft_input_size, 64) #[batch_size,inputsize] to [batch_size,outputsize] (a(x),fft(a(x)),x)
        self.fc1 = nn.Linear(64+position_size, 256)
    def forward(self,a,position):
        """
        :param a: 常数系数
        :param f: 强迫函数
        :param position: 位置坐标
        :return: u 解
        """
        a_ft = torch.fft.rfft2(a,n=a.size(),dim=-1)
        # f_ft = torch.fft.rfft2(f)
        x1 = torch.cat((a,a_ft, position), dim=-1) # 输入(a,f)
        x1 = self.fc0(x1)
        x1 = F.gelu(x1)

        x1 = torch.cat((x1, position), dim=-1)
        x1 = self.fc1(x1)
        x1 = F.gelu(x1)
        x1 = torch.fft.irfft2(x1)


        x1 = self.fc2(x1)




