# coding:UTF-8
# @Time: 2022/11/19 20:46
# @Author: Lulu Cao
# @File: IFNO.py
# @Software: PyCharm

import pandas as pd


import scipy.io as scio
import torch
import torch.nn.functional as F

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# class IFNO1d(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size, num_layers):
#         super(IFNO1d, self).__init__()  # 调用父类的构造，必须要有
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#
#         # self.rnn = torch.nn.RNN(input_size=self.input_size,
#         #                         hidden_size=self.hidden_size,
#         #                         num_layers=num_layers)
#         #self.weights1 = torch.nn.Parameter(torch.rand(hidden_size, dtype=torch.cfloat))
#
#         if num_layers==2:
#             self.rnn1=torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
#             self.weights1 = torch.nn.Parameter(torch.rand(hidden_size, dtype=torch.float))
#
#             self.rnn2 = torch.nn.RNNCell(input_size=hidden_size, hidden_size=hidden_size)
#             self.weights2 = torch.nn.Parameter(torch.rand(hidden_size, dtype=torch.float))
#
#             self.weights3 = torch.nn.Parameter(torch.rand(hidden_size, dtype=torch.float))
#             self.weights4 = torch.nn.Parameter(torch.rand(hidden_size, dtype=torch.float))
#
#             self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
#             self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
#             self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
#
#
#         #print(self.weights1.shape)
#
#     def forward(self,hidden, x_input):
#         hidden_t = torch.zeros(hidden.shape[:],dtype=torch.float,device=hidden.device)
#         hidden1 = self.rnn1(x_input, hidden[0])
#         hidden1 = F.gelu(torch.einsum("bx,x->bx", hidden1, self.weights1))
#         hidden_t[0]=hidden1
#
#         hidden2 = self.rnn2(hidden1, hidden[1])
#
#         x1 = self.fc1(hidden2)
#         hidden2 = F.gelu(x1+torch.einsum("bx,x->bx", hidden2, self.weights2))
#
#         x1 = self.fc2(x1)
#         hidden2 = F.gelu(x1 + torch.einsum("bx,x->bx", hidden2, self.weights3))
#
#         x1 = self.fc3(x1)
#         hidden2 = x1+torch.einsum("bx,x->bx", hidden2, self.weights4)
#         hidden_t[1]=hidden2
#
#
#         # out, hidden = self.rnn(x_input, hidden)
#         #
#         #
#         #
#         #
#         # for i in range(out.shape[0]):
#         #     out_t[i,...]=torch.einsum("bx,x->bx", out[i], self.weights1)
#         # return out_t.permute(permute_idx).contiguous() # batchsize,seq(t),hiddensize
#         return hidden2,hidden_t
#


class SpectralConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class IFNO1d(torch.nn.Module):
    def __init__(self,input_channel,spatial,blength,num_channels=1, modes=16, width=64, initial_step=10,):
        super(IFNO1d,self).__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic

        self.fc0 = torch.nn.Sequential(
            torch.nn.Linear(initial_step * num_channels + 1, self.width),
            torch.nn.LayerNorm(self.width)
        )
        # self.fc0 = torch.nn.Linear(initial_step*num_channels + 1, self.width)
        # Sequential
        # torch.nn.LayerNorm(self.width)

        self.w0 = torch.nn.Conv1d(self.width, self.width, 1)
        self.w1 = torch.nn.Conv1d(self.width, self.width, 1)
        self.w2 = torch.nn.Conv1d(self.width, self.width, 1)
        self.w3 = torch.nn.Conv1d(self.width, self.width, 1)

        # self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.nb0 = torch.nn.BatchNorm1d(self.width)
        self.nb1 = torch.nn.BatchNorm1d(self.width)
        self.nb2 = torch.nn.BatchNorm1d(self.width)
        self.nb3= torch.nn.BatchNorm1d(self.width)

        self.fc1 = torch.nn.Linear(self.width, 128)
        self.fc2 = torch.nn.Linear(128, num_channels)

        self.fc3 = torch.nn.Linear(spatial, blength)

        self.fc21 = torch.nn.Linear(input_channel, blength)
        self.fc22 = torch.nn.Linear(blength, blength)

        self.init_weight()
    def forward(self,x, grid ,position):
        """

        :param param:
        :param position:
        :return:
        """
        # x dim = [b, x1, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = F.gelu(x)

        # x dim = [b, t*v, x1,]
        x = x.permute(0, 2, 1)
        #x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic #最后一个维度添加0

        #x1 = self.conv0(x)
        x = self.w0(x)
        #x = x1 + x2
        #x = self.nb0(x)
        x = F.relu(x)

        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        x = F.relu(x)
        x = self.w3(x)
        x = F.relu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.relu(x)
        #
        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.relu(x)
        #
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)

        batch_size = x.size(0)  # 有多少数据

        x1 = x.view(batch_size, -1,x.size(-1))
        x1 = x1.permute(0, 2, 1)
        x1 = self.fc3(x1)
        x1 = F.gelu(x1)

        x2 = position

        x2 = F.gelu(self.fc21(x2))
        x2 = F.gelu(self.fc22(x2))
        outputs = torch.einsum('ncb,nmb->nmc', x1, x2)

        return outputs

    def init_weight(self):
        for m in self.modules():
            t = type(m)
            if t is torch.nn.Conv2d:
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))

                m.weight.data.normal_(0.0, 0.02)
            elif t is torch.nn.BatchNorm2d:
                # m.eps = 1e-3
                # m.momentum = 0.1
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()

                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

            elif t in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU]:
                m.inplace = True






class SpectralConv2d_fast(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
class IFNO2d(torch.nn.Module):
    def __init__(self,input_channel,spatial,blength,num_channels=1, modes1=12, modes2=12, width=64, initial_step=10,):
        super(IFNO2d,self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic

        self.fc0 = torch.nn.Linear(initial_step * num_channels + 2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w1 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w2 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w3 = torch.nn.Conv2d(self.width, self.width, 1)

        self.fc1 = torch.nn.Linear(self.width, 128)
        self.fc2 = torch.nn.Linear(128, num_channels)

        self.fc3 = torch.nn.Linear(spatial, blength)

        self.fc21 = torch.nn.Linear(input_channel, blength)
        self.fc22 = torch.nn.Linear(blength, blength)

    def forward(self, x, grid, position):
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]  # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        batch_size = x.size(0)  # 有多少数据

        x1 = x.view(batch_size, -1, x.size(-1))
        x1 = x1.permute(0, 2, 1)
        x1 = self.fc3(x1)
        x1 = F.gelu(x1)

        x2 = position

        x2 = F.gelu(self.fc21(x2))
        x2 = F.gelu(self.fc22(x2))
        outputs = torch.einsum('ncb,nmb->nmc', x1, x2)
        return outputs

class IFNO3d(torch.nn.Module):
    def __init__(self):
        pass
