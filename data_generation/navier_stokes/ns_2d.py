# coding:UTF-8
# @Time: 2022/9/25 21:37
# @Author: Lulu Cao
# @File: ns_2d.py
# @Software: PyCharm

import torch
import argparse
import tqdm

import math

from random_fields import GaussianRF

from timeit import default_timer

import scipy.io
import hdf5storage
import pickle
import joblib

def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    #Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0) #扩张数据维度

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias* F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            #Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1


    return sol, sol_t

def main(args):
    if args.gpu == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    pass

    # Resolution
    s = args.resolution

    #Number of solutions to generate
    N = args.N

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s + 1, device=device)
    t = t[0:-1]

    # 生成网格坐标 (X[i][j],Y[i][j])
    X, Y = torch.meshgrid(t, t)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Number of snapshots from solution
    record_steps = 200

    # Inputs
    a = torch.zeros(N, s, s)
    # Solutions
    u = torch.zeros(N, s, s, record_steps)

    # Solve equations in batches (order of magnitude speed-up)

    # Batch size
    bsize = args.bsize

    c = 0
    t0 = default_timer()
    par = tqdm.tqdm(range(N // bsize))
    for j in par:
        par.set_description(
            f"Now that's all")
        # Sample random feilds
        w0 = GRF.sample(bsize)

        # Solve NS
        # 小粘性系数1e-3,时间[0,args.T]区间记录 record_steps个,时间离散度1e-4,
        sol, sol_t = navier_stokes_2d(w0, f, args.v, args.T, 1e-4, record_steps)

        a[c:(c + bsize), ...] = w0
        u[c:(c + bsize), ...] = sol

        c += bsize
        t1 = default_timer()
        print(j, c, t1 - t0)
        scipy.io.savemat(args.savepath+str(j)+'.mat', mdict={'a': w0.detach().cpu().numpy(), 'u': sol.detach().cpu().numpy(),
                                                't': sol_t.detach().cpu().numpy()})
        print("saved")
    # temp_dict = {'a': a.detach().cpu().numpy(), 'u': u.detach().cpu().numpy(), 't': sol_t.detach().cpu().numpy()}
    # print([temp_dict[k][1] for k, v in temp_dict.items()])
    # joblib.dump(temp_dict, str(int(args.v * 100000)) + '_' + str(j) + '.txt')
    # scipy.io.savemat(args.savepath, mdict={'a': a.detach().cpu().numpy(), 'u': u.detach().cpu().numpy(), 't': sol_t.detach().cpu().numpy()})

    #     print("use hdf5storage")
    #     hdf5storage.savemat(args.savepathu, mdict={'u': u.detach().cpu().numpy()}, format='7.3')
    #     hdf5storage.savemat(args.savepatha , mdict={'a': a.detach().cpu().numpy()}, format='7.3')
    #     hdf5storage.savemat(args.savepatht, mdict={'t': sol_t.detach().cpu().numpy()}, format='7.3')



if __name__=="__main__":
    print("data generation…………")
    parser = argparse.ArgumentParser(description='Imitative Fourier Nerual Operator')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu:1, cpu:0")
    parser.add_argument("--resolution", type=int, default=256,
                        help="resolution")
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of solutions to generate")
    parser.add_argument("--bsize", type=int, default=20,
                        help="batch size  N//batch_size")
    parser.add_argument("--T", type=int, default=50,
                        help="time interval [0,T]")
    parser.add_argument("--v", type=float, default=1e-3,
                        help="viscosity coefficient")
    parser.add_argument("--dataset", type=str, default="train",
                        help="if dataset is equal to train, it is train dataset."
                             "if dataset is equal to test, it is test dataset.")

    args = parser.parse_args()
    args.savepath = '../../data/ns_data_V'+ str(int(args.v*100000))+'_N'+str(args.N)+'_T'+str(args.T)+'_'+args.dataset
    # args.savepathu = '../../data/ns_data_V' + str(int(args.v * 100000)) + '_N' + str(args.N) + '_T' + str(
    #     args.T) + '_' + args.dataset + 'u.mat'
    # args.savepatht = '../../data/ns_data_V' + str(int(args.v * 100000)) + '_N' + str(args.N) + '_T' + str(
    #     args.T) + '_' + args.dataset + 't.mat'
    print(args)

    main(args)