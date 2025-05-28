# coding:UTF-8
# @Time: 2022/9/25 21:14
# @Author: Lulu Cao
# @File: Imitative_FNO_2d_time.py
# @Software: PyCharm

import torch
import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utilities3 import *
from configs import *

from timeit import default_timer
import joblib
import tqdm
import torch.nn.functional as F

import utils

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

torch.manual_seed(0)
np.random.seed(0)




# 1. 准备数据集
class IFNODataset(Dataset):
    def __init__(self,filepath,istrain,number,sub,T_in,T_after_delta):

        if istrain==True:
            par = tqdm.tqdm(range(number // batch_size),desc="Get train dataset")
            for j in par:
                reader = MatReader(args.filepath+str(j)+'.mat')
                a = reader.read_field('u')[:, ::sub, ::sub, :T_in]
                u = reader.read_field('u')[:, ::sub, ::sub, T_in:T_in + T_after_delta]
                a_para = reader.read_field('a')[:, ::sub, ::sub]
                a_para = a_para.reshape(a_para.shape[0], a_para.shape[1], a_para.shape[2], 1)

                if j == 0:
                    self.a = a
                    self.u = u
                    self.a_para = a_para
                else:
                    self.a = torch.cat((self.a,a),0)
                    self.u = torch.cat((self.u,u), 0)
                    self.a_para = torch.cat((self.a_para, a_para), 0)

        else:
            temp = range(N // batch_size)
            par = tqdm.tqdm(temp[-number // batch_size:N//batch_size],desc="Get test dataset")
            flag=0
            for j in par:
                reader = MatReader(args.filepath + str(j) + '.mat')
                a = reader.read_field('u')[:, ::sub, ::sub, :T_in]
                u = reader.read_field('u')[:, ::sub, ::sub, T_in:T_in + T_after_delta]
                a_para = reader.read_field('a')[:, ::sub, ::sub]
                a_para = a_para.reshape(a_para.shape[0], a_para.shape[1], a_para.shape[2], 1)

                if flag == 0:
                    flag = 1
                    self.a = a
                    self.u = u
                    self.a_para = a_para
                else:
                    self.a = torch.cat((self.a, a), 0)
                    self.u = torch.cat((self.u, u), 0)
                    self.a_para = torch.cat((self.a_para, a_para), 0)

        self.len = number
        print(self.a.shape)
        print(self.u.shape)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.a[index],self.u[index],self.a_para[index]


def dataPrepare(filepath):
    """
    将数据转化为pytorch的dataloader格式
    :param filepath: 数据所在路径 mat格式
    :return: train_loader, test_loader
    """

    train_dataset = IFNODataset(filepath,istrain=True,number=ntrain,sub=sub,T_in=T_in,T_after_delta=T_after_delta)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                             batch_size = bsize,
                             sampler=train_sampler)

    test_dataset = IFNODataset(filepath,istrain=False,number=ntest,sub=sub,T_in=T_in,T_after_delta=T_after_delta)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=bsize,
                              sampler=test_sampler)
    return train_loader, test_loader


# TODO:2. 设计模型
class Rainbow(torch.nn.Module):
    def __init__(self,in_channels,out_channels, modes1, modes2):
        super(Rainbow,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x_input, weights):
        """
        定义input和权重如何相乘
        :param input: [batch,in_channel,x,y]
        :param weights: [in_channel,out_channel,x,y]
        :return:[batch,out_channel,x,y]
        """
        print(x_input.size(),weights.size())
        return torch.einsum("bixy,ioxy->boxy", x_input, weights)

    def forward(self,x):
        """
        Magic construction
        :param x:[bathsize,channel by fc0，resolution，resolution]
        :return:
        """
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize,self.out_channels,x_ft.size(-2),x_ft.size(-1), dtype=torch.cfloat,device=x.device)
        out_ft[:,:,:self.modes1,:self.modes2] = \
            self.compl_mul2d(x_ft[:,:,:self.modes1,:self.modes2],self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


        return x


class IFNO(torch.nn.Module):
    def __init__(self,modes1, modes2,width):
        super(IFNO,self).__init__()

        """
                The overall network. It contains 4 layers of the Fourier layer.
                1. Lift the input to the desire channel dimension by self.fc0 .
                2. 4 layers of the integral operators u' = (W + K)(u).
                    W defined by self.conv; K defined by self.fconv .
                3. Project from the channel space to the output space by self.fc1 and self.fc2 .
                input shape: (batchsize, x=s, y=s, c=T_in+2)
                output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width


        self.fc0 = nn.Linear(T_in+2, self.width)  # [batch_size,inputsize 12] (u(t-10, x, y), ..., u(t-1, x, y),  x, y) to [batch_size,outputsize]
        self.fconv0 = Rainbow(self.width+1, self.width, self.modes1, self.modes2)
        self.fconv1 = Rainbow(self.width, self.width, self.modes1, self.modes2)
        self.fconv2 = Rainbow(self.width, self.width, self.modes1, self.modes2)
        self.fconv3 = Rainbow(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width+1, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self,x,a_para):
        """

        :param x: [batchsize,s,s,recordtime]  [0.T_in]
        :param a_para: [batchsize,s,s,1]
        :return:
        """
        grid = self.get_grid_para(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)

        x = torch.cat((x, a_para), dim=-1)
        x = x.permute(0, 3, 1, 2)
        a_para = a_para.permute(0,3,1,2)
        x = torch.einsum('nkij,nmij->nkij', x, a_para)

        x1 = self.fconv0(x)  # batch_size, channels, height_1, width_1
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.fconv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.fconv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.fconv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)  ##batch_size, s, s,channels,
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x



    def get_grid_para(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

        return torch.cat((gridx, gridy), dim=-1).to(device)


def Settings():
    """
    构造损失函数和优化器
    :return: 模型，优化器，损失函数 model,optimizer,myloss
    """
    model = IFNO().half().float()
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')
    # print(count_params(model))

    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=1e-4) # 告诉优化器对哪些Tensor做梯度优化
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma) #每多少轮循环后更新一次学习率,每次更新lr的gamma倍

    myloss = LpLoss(size_average=False)
    return model,optimizer,myloss,scheduler

# 4. 训练周期：forward、backward、update
def train(epochs,train_loader,model,optimizer,myloss,scheduler,device):

    for epoch in range(epochs):
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        par = tqdm.tqdm(train_loader, desc="Train batchsize")
        for xx,yy,a_para in par:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            a_para = a_para.to(device)
            # 这里需要注意的是，对t在[10,20）的输出方式是滚动的
            # 即输入T[0,10)，输出T[10]，得到T[10]后，下一轮输入T[0,11)，输出T[11]
            # 每次只输出一个时间点的结果，输入该时间点之前的所有结果
            tar = tqdm.tqdm(range(0, T_after_delta, step), desc="Time Step")
            for t in tar:
                # 具体时间点label的值，该场景下step被设置为1
                y = yy[..., t:t + step].to(device)
                # 输入xx得到FNO的输出im

                im = model(xx,a_para).to(device)
                # loss是累加的，在输出完[10,20）所有结果后再更新参数
                loss += myloss(im.reshape(bsize, -1), y.reshape(bsize, -1))
                # 如果t=0则模型输出直接等于pred，如果不为0，则把本次输出和之前的输出拼接起来，即把每个时间点的输出拼起来
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1) #最后一维增加，其余维度不变
                # 将本次输出的im拼接到xx上，用作下一轮的输入
                xx = torch.cat((xx[..., step:], im), dim=-1).to(device)
                # 计算总loss，更新参数
            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(bsize, -1), yy.reshape(bsize, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #梯度Update

        t2 = default_timer()
        print(t2-t1)
        print(train_l2_step,train_l2_full)
        scheduler.step()
    return model,

def test(test_loader,model,myloss,device):
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():  # 主要是用于停止autograd模块的工作,以起到加速和节省显存的作用
        for xx,yy,a_para in test_loader:
            t1 = default_timer()
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            a_para = a_para.to(device)
            # 这里需要注意的是，对t在[10,20）的输出方式是滚动的
            # 即输入T[0,10)，输出T[10]，得到T[10]后，下一轮输入T[0,11)，输出T[11]
            # 每次只输出一个时间点的结果，输入该时间点之前的所有结果
            for t in range(0, T_after_delta, step):
                # 具体时间点label的值，该场景下step被设置为1
                y = yy[..., t:t + step]
                # 输入xx得到FNO的输出im
                im = model(xx,a_para)
                # loss是累加的，在输出完[10,20）所有结果后再更新参数
                loss += myloss(im.reshape(bsize, -1), y.reshape(bsize, -1))
                # 如果t=0则模型输出直接等于pred，如果不为0，则把本次输出和之前的输出拼接起来，即把每个时间点的输出拼起来
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)  # 最后一维增加，其余维度不变
                # 将本次输出的im拼接到xx上，用作下一轮的输入
                xx = torch.cat((xx[..., step:], im), dim=-1)
                # 计算总loss，更新参数
            t2 = default_timer()
            print(t2 - t1)

            test_l2_step += loss.item()
            l2_full = myloss(pred.reshape(bsize, -1), yy.reshape(bsize, -1))
            test_l2_full += l2_full.item()
            print(test_l2_step, test_l2_full)


    #print(f"Accuracy on test set:{100 * test_l2_step}%,{100 * test_l2_full}%")

def main(args):

    # 设置文件路径
    path = args.filepath + str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
    path_model = path.replace("data/","model/")+"model.pkl"
    path_train_err = path.replace("data/","results/")+"train.txt"
    path_test_err = path.replace("data/","results/")+"test.txt"
    path_image = path.replace("data/","image/")

    # 1. 准备数据集
    filepath=args.filepath
    train_loader, test_loader = dataPrepare(filepath)

    # 3. 构造损失函数和优化器
    model,optimizer,myloss,scheduler = Settings()
    model.to(device)
    model = DistributedDataParallel(model,device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


    # 4. 训练周期：forward、backward、update
    model = train(epochs,train_loader,model,optimizer,myloss,scheduler,device)
    torch.save(model, path_model)

    test(test_loader,model,myloss,device)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Imitative Fourier Nerual Operator')

    parser.add_argument("--filepath", type=str, default='data/ns_data_V100_N1200_T50_train',
                        help="dataset path")
    parser.add_argument("--device_ids", type=str, default="0,2,3", help="Training Devices")
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()
    print(args)
    main(args)