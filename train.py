# coding:UTF-8
# @Time: 2022/11/6 14:56
# @Author: Lulu Cao
# @File: train.py
# @Software: PyCharm

import argparse
import torch
from Imitative_FNO_1d import *
import torch.nn.functional as F
from timeit import default_timer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def dataPrepare(filepath,args):
    """
    将数据转化为pytorch的dataloader格式
    :param filepath: 数据所在路径 mat格式
    :return: train_loader, test_loader
    """

    train_dataset = IFNODataset(filepath,istrain=True,number=args.ntrain,sub=args.sub)

    train_loader = DataLoader(dataset=train_dataset,
                               batch_size = args.batchsize,
                             shuffle=True)

    test_dataset = IFNODataset(filepath,istrain=False,number=args.ntest,sub=args.sub)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=args.batchsize,
                              shuffle=False)
    return train_loader, test_loader


def Settings(args):
    """
    构造损失函数和优化器
    :return: 模型，优化器，损失函数 model,optimizer,myloss
    """
    model = IFNO1d()
    # model = IFNO().half().float()
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')
    # print(count_params(model))

    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,weight_decay=1e-4) # 告诉优化器对哪些Tensor做梯度优化
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma) #每多少轮循环后更新一次学习率,每次更新lr的gamma倍

    myloss = torch.nn.MSELoss(size_average=True)
    # myloss = LpLoss(size_average=True)
    return model,optimizer,myloss,scheduler

def train(args,train_loader,model,optimizer,myloss,scheduler):
    device = args.device
    mse_list = []
    for epoch in range(args.epochs):
        t1 = default_timer()
        running_loss = 0
        par = tqdm.tqdm(train_loader, desc="Train batchsize")
        for a,u,grid in par:
            a, u, grid = a.to(device),u.to(device),grid.to(device)

            optimizer.zero_grad()

            im = model(a, u, grid).to(device)
            loss = myloss(im,u)


            loss.backward()
            optimizer.step() #梯度Update

            running_loss += loss.item()

        mse_list.append(running_loss/len(par))
        t2 = default_timer()
        print(t2-t1)
        scheduler.step()
    return model,mse_list

def test(args,test_loader,model,myloss):
    device = args.device
    running_loss= 0

    with torch.no_grad():  # 主要是用于停止autograd模块的工作,以起到加速和节省显存的作用
        par = tqdm.tqdm(test_loader, desc="Train batchsize")
        for a, u, grid in par:
            t1 = default_timer()
            a, u, grid = a.to(device), u.to(device), grid.to(device)

            im = model(a, u, grid).to(device)
            loss = myloss(im, u)
            running_loss += loss.item()

        t2 = default_timer()
        print(t2 - t1)
    normal_err=running_loss/len(par)
    print(normal_err)
    return normal_err
    #print(f"Accuracy on test set:{100 * test_l2_step}%,{100 * test_l2_full}%")

def main(args):
    device = args.device
    # 设置文件路径
    path = args.filepath +'_ntrain'+str(args.ntrain)+'_ep' + str(args.epochs)
    path_model = path.replace("data/","model/")+"model.pkl"
    path_train_err = path.replace("data/","results/")+"train.txt"
    path_test_err = path.replace("data/","results/")+"test.txt"
    path_image = path.replace("data/","image/")

    # 1. 准备数据集
    train_loader, test_loader = dataPrepare(args.filepath+'.mat',args)

    # 3. 构造损失函数和优化器
    model,optimizer,myloss,scheduler = Settings(args)
    model.to(device)


    # 4. 训练周期：forward、backward、update
    model,mse_list = train(args,train_loader,model,optimizer,myloss,scheduler)
    torch.save(model, path_model)

    test_err = test(args,test_loader,model,myloss)
    scipy.io.savemat(path_train_err + '.mat',
                     mdict={'train': mse_list,'test':test_err})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Imitative Fourier Nerual Operator')

    parser.add_argument("--filepath", type=str, default='data/ns_data_V100_N1200_T50_train',
                        help="dataset path")
    parser.add_argument("--ntrain", type=int, default=1000,
                        help="number of train dataset")
    parser.add_argument("--ntest", type=int, default=200,
                        help="number of test dataset")
    parser.add_argument("--batchsize", type=int, default=20,
                        help="batch size")
    parser.add_argument("--s", type=int, default=8192,
                        help="resolution")
    parser.add_argument("--sub", type=int, default=1,
                        help="change resolution to s/sub")
    parser.add_argument("--epochs", type=int, default=100,
                        help="train epoch")
    parser.add_argument("--learning_rate", type=int, default=100,
                        help="train epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--scheduler_step", type=int, default=100,
                        help="scheduler step")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5,
                        help="scheduler gamma")

    args = parser.parse_args()
    args.filepath = "data/burgers_data_v10"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(args)
    main(args)




