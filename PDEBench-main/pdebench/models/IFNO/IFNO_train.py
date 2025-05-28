# coding:UTF-8
# @Time: 2022/11/13 17:52
# @Author: Lulu Cao
# @File: IFNO_train.py
# @Software: PyCharm


import deepxde as dde
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys
import torch

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
from timeit import default_timer
from torch.nn.parallel import DataParallel

sys.path.append(".")

device_ids = [0]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else 'cpu')


from .IFNO_utils  import IFNODatasetSingle,IFNODatasetMult
from .IFNO import IFNO1d,IFNO2d,IFNO3d
#from metrics import *
from .PDE_figure import *

def adjust_learning_rate(optimizer, learning_rate, scheduler_gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * scheduler_gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def run_training(if_training,continue_training,
                 epochs,learning_rate,scheduler_step,scheduler_gamma,batch_size,
                 flnm,single_file,test_ratio = 0.9,
                 base_path='../data_download/data/1D/Advection/Train/',initial_step=10,num_channels=1,plot=True):
    print(f'Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}')
    print("device",device)
    ################################################################
    # load data
    ################################################################
    if single_file:
        # filename
        model_name = flnm[:-5] + '_IFNO'
        print("IFNODatasetSingle")

        # Initialize the dataset and dataloader
        train_data = IFNODatasetSingle(flnm,saved_folder=base_path,
                 reduced_resolution=1,reduced_resolution_t=1,reduced_batch=1,
                 if_test=False,test_ratio=test_ratio,num_samples_max=-1,
                 initial_step=initial_step)
        val_data = IFNODatasetSingle(flnm,saved_folder=base_path,
                 reduced_resolution=1,reduced_resolution_t=1,reduced_batch=1,
                 if_test=True,test_ratio=test_ratio,num_samples_max=-1,
                 initial_step=initial_step)
    else:
        # filename
        model_name = flnm + '_IFNO'

        train_data = IFNODatasetMult(flnm,
                 initial_step=initial_step,
                 saved_folder=base_path,
                 if_test=False, test_ratio=0.1)
        val_data = IFNODatasetMult(flnm,
                 initial_step=initial_step,
                 saved_folder=base_path,
                 if_test=True, test_ratio=0.1)
    train_loader = torch.utils.data.DataLoader(train_data,num_workers=0, batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,num_workers=0, batch_size=batch_size,shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    _data, _param, _,_ = next(iter(val_loader))
    dimensions = len(train_data.data_shape)-3
    print('Spatial Dimension', dimensions)

    if dimensions == 1:
        model = IFNO1d(_data.shape[-1],_param.shape[-3],100,initial_step=initial_step,num_channels=num_channels).to(device)
        #model = IFNO1d(train_data.input_size,train_data.hiddensize,batch_size,train_data.num_layers).to(device)
    elif dimensions == 2:
        model = IFNO2d(_data.shape[-1],_param.shape[1]*_param.shape[2],100,initial_step=initial_step,num_channels=num_channels).to(device)
    elif dimensions == 3:
        model = IFNO3d().to(device)

    #model = DataParallel(model, device_ids=device_ids, output_device=device)  # 将最后的结果gather到一个服务器上
    model_path = model_name + ".pt"
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}, model path = {model_path}')

    ###### 构造损失函数和优化器 #####
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 告诉优化器对哪些Tensor做梯度优化
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step,
    #                                             gamma=scheduler_gamma)  # 每多少轮循环后更新一次学习率,每次更新lr的gamma倍
    loss_fn = torch.nn.MSELoss(reduction="mean") #在batch和特征维度上都做了平均

    start_epoch = 0





    ###### 测试 #####
    if not if_training:
        itot = 0
        checkpoint = torch.load(model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(model.state_dict())
        model.to(device)
        model.eval()
        val_l2_full=0
        running_loss = []
        with torch.no_grad():
            sum_time = 0
            min_loss=100
            for data_input, xx, yy, grid in val_loader:
                t1 = default_timer()
                data_input = data_input.to(device)
                xx = xx.to(device)
                yy = yy.to(device)
                grid = grid.to(device)

                inputMax = torch.max(xx)
                inputMin = torch.min(xx)
                #outputMax = torch.max(yy)
                #outputMin = torch.min(yy)

                xx = 2 * (xx - inputMin) / (inputMax - inputMin) - 1

                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                inp = xx.reshape(inp_shape)


                outputs = model(inp, grid, data_input)  # [b,m,num_layers]
                #outputs = (outputs + 1) * (outputMax - outputMin) / 2 + outputMin

                t2 = default_timer()
                sum_time+=t2 - t1
                # print(t2 - t1)
                out_shape = list(outputs.shape)
                inp_shape = list(yy.shape)
                out_inp = torch.zeros(inp_shape[:], device=xx.device, dtype=torch.float)
                inp_shape[-2] = -1
                outputs = outputs.reshape(inp_shape)
                out_inp[..., :initial_step, :] = xx
                out_inp[..., initial_step:, :] = outputs





                # [batch,xn,t,ch]
                err_mean = torch.mean((out_inp.view([yy.shape[0], -1, yy.shape[-2], yy.shape[-1]]) - yy.view([yy.shape[0], -1, yy.shape[-2], yy.shape[-1]])) ** 2, dim=1)
                #print(err_mean)


                loss = loss_fn(out_inp.reshape(yy.shape[0], -1), yy.reshape(yy.shape[0], -1))
                running_loss.append(loss.item())
                if loss<min_loss:
                    with open(flnm+".time.pkl", 'wb') as f:
                        pickle.dump(err_mean.detach().cpu(), f)
                    min_loss = loss
                    #print(err_mean)
                    pred_plot = out_inp[:1]
                    target_plot = yy[:1]

                val_l2_full += loss.item()

            val_l2_full = val_l2_full / len(val_loader)
            plot_data_pred(dimensions, target_plot, pred_plot, 0, model_name, initial_step)
            print(f'mean_time:{sum_time/len(val_loader)},testloss: {val_l2_full}')
            print(np.mean(running_loss), np.var(running_loss))
        # plot=True
        # channel_plot=0
        # x_min,x_max,y_min,y_max,t_min,t_max = 0,1,0,1,0,1
        # Lx, Ly, Lz = 1., 1., 1.
        # errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
        #                model_name, x_min, x_max, y_min, y_max,
        #                t_min, t_max,initial_step=initial_step)
        # pickle.dump(errs, open(model_name + '.pickle', "wb"))

        return

    min_loss = 100
    min_test_loss = 100

    if continue_training:
        print('Restoring model (that is the network\'s weights) from file...')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()

        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint['epoch']
        min_test_loss = checkpoint['loss']



    for ep in range(start_epoch, epochs):
        model.train()
        t1 = default_timer()
        running_loss = 0
        loss = 0
        for data_input, xx, yy, grid in train_loader:
            data_input = data_input.to(device)
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)

            #inputMax = torch.max(xx)
            #inputMin = torch.min(xx)
            #outputMax = torch.max(yy)
            #outputMin = torch.min(yy)

            #xx = 2 * (xx - inputMin) / (inputMax - inputMin) - 1

            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)
            # Reshape input tensor into [b, x1, ..., xd, t_init*v]
            inp = xx.reshape(inp_shape)



            outputs = model(inp, grid, data_input)  # [b,m,num_layers]
            #outputs = (outputs + 1) * (outputMax - outputMin) / 2 + outputMin

            inp_shape = list(yy.shape) # [batch,xn,t,ch]
            out_inp = torch.zeros(inp_shape[:], device=xx.device, dtype=torch.float) # [batch,xn,t,ch]
            inp_shape[-2]=-1 # [batch,xn,-1,ch]
            outputs = outputs.reshape(inp_shape)
            out_inp[...,:initial_step,:] = xx
            out_inp[...,initial_step:, :] = outputs

            loss = loss_fn(out_inp.reshape(yy.shape[0], -1), yy.reshape(yy.shape[0], -1))
            optimizer.zero_grad()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            #

        t2 = default_timer()
        print(t2-t1)
        running_loss = running_loss/len(train_loader)
        if min_loss>running_loss:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss
            }, model_path)
            min_loss = running_loss


        val_l2_full = 0
        if ep%10==0:
            with torch.no_grad():
                for data_input, xx, yy, grid in val_loader:
                    data_input = data_input.to(device)
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)

                    # inputMax = torch.max(xx)
                    # inputMin = torch.min(xx)
                    #outputMax = torch.max(yy)
                    #outputMin = torch.min(yy)

                    #xx = 2 * (xx - inputMin) / (inputMax - inputMin) - 1

                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)
                    outputs = model(inp, grid, data_input)  # [b,m,num_layers]
                    #outputs = (outputs + 1) * (outputMax - outputMin) / 2 + outputMin

                    out_shape = list(outputs.shape)
                    inp_shape = list(yy.shape)
                    out_inp = torch.zeros(inp_shape[:], device=xx.device, dtype=torch.float)
                    inp_shape[-2] = -1
                    outputs = outputs.reshape(inp_shape)
                    out_inp[..., :initial_step, :] = xx
                    out_inp[..., initial_step:, :] = outputs

                    loss = loss_fn(out_inp.reshape(yy.shape[0], -1), yy.reshape(yy.shape[0], -1))
                    val_l2_full += loss.item()

                val_l2_full = val_l2_full / len(val_loader)
                print(model.parameters())
                #print(torch.max(model.parameters()),torch.min(model.parameters()))
                if val_l2_full<min_test_loss:
                    min_test_loss = val_l2_full
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_l2_full
                    }, model_path)

        print(f'epoch: {ep}, loss: {running_loss}, testloss: {val_l2_full}')

if __name__ == "__main__":
    run_training()