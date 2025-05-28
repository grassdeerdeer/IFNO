# coding:UTF-8
# @Time: 2022/11/7 18:02
# @Author: Lulu Cao
# @File: sample_data.py
# @Software: PyCharm

"""
从建模图中随机采样数据点
"""
import scipy.io
import torch
import numpy as np
import random
import pandas as pd
torch.manual_seed(0)
np.random.seed(0)
from sklearn.utils import shuffle


def SampleData1d(filepath,sample_num):
    data = scipy.io.loadmat(filepath)
    grid = np.linspace(0, 1, data['input'].shape[-1])
    a_size = data['input'].shape[0]
    x_list = range(a_size)

    a_sample = np.array([])
    u_sample = np.array([])
    position = np.array([])
    for i in range(len(data['input'])):

        sample_index = random.sample(x_list, sample_num)
        temp = data['input'][i, sample_index]
        a_sample = np.hstack((a_sample, data['input'][i, sample_index]))
        u_sample = np.hstack((u_sample, data['output'][i, sample_index]))
        position = np.hstack((position,grid[sample_index]))


        print(sample_index)

    df = pd.DataFrame({'a': a_sample,
                       'position': position,
                       'u': u_sample,
                       })
    df = shuffle(df)
    df.to_csv("../../data/burgers_data_v10_sample"+str(sample_num)+".csv")


if __name__=="__main__":
    loadpath = '../../data/burgers_data_v10.mat'
    sample_num = 1000 # 每个仿真案列采样几个数据
    SampleData1d(loadpath,sample_num)

    # df = pd.read_csv("../../data/burgers_data_v10_sample9.csv")
    # data = torch.from_numpy(df.values)
    # print(data[0:2,:-2],data[0:2,-2],data[0:2,-1])