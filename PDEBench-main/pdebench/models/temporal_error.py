# coding:UTF-8
# @Time: 2023/1/23 7:48
# @Author: Lulu Cao
# @File: temporal_error.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import h5py
import numpy as np
import pickle

# shallow water
flnm = "2D_rdb_NA_NA"
base_path= '../data_download/data/2D/shallow-water/'
file_path = os.path.abspath(base_path + flnm + ".h5")
initial_step = 10
plt.figure(1)
with h5py.File(file_path, 'r') as h5_file:
    data_list = sorted(h5_file.keys())
    seed_group = h5_file[data_list[0]]
    # data dim = [t, x1, ..., xd, v]
    time = np.array(seed_group["grid"]["t"])
    with open(flnm+".time.pkl", 'rb') as f:
        err_mean = pickle.load(f)
        err_mean = err_mean.numpy()#[batch,t,ch]
plt.subplot(311)
plt.subplots_adjust(hspace=1)
plt.plot(time[initial_step:],err_mean[0,initial_step:,0], )
plt.axvline(x=time[len(time)//2],ymin=0,ymax=1,ls="-.",c="g")
plt.axis([0, 1.1, -1e-2 , 1e-2])
plt.ylabel('MSE')
plt.xlabel('time(s)')
plt.title("Shallow Water")












# diffusion reaction
flnm = "2D_diff-react_NA_NA"
base_path= '../data_download/data/2D/diffusion-reaction/'
file_path = os.path.abspath(base_path + flnm + ".h5")
initial_step = 10

with h5py.File(file_path, 'r') as h5_file:
    data_list = sorted(h5_file.keys())
    seed_group = h5_file[data_list[0]]
    # data dim = [t, x1, ..., xd, v]
    time = np.array(seed_group["grid"]["t"])
    with open(flnm+".time.pkl", 'rb') as f:
        err_mean = pickle.load(f)
        err_mean = err_mean.numpy()#[batch,t,ch]
plt.subplot(312)
plt.plot(time[initial_step:],err_mean[0,initial_step:,0], )
plt.plot(time[initial_step:],err_mean[0,initial_step:,1], )
plt.axvline(x=time[len(time)//2],ymin=0,ymax=1,ls="-.",c="g")
plt.axis([0, 5.5, -1e-1, 1e-1])

plt.ylabel('MSE')
plt.xlabel('time(s)')
plt.title("Diffusion Reaction")


# diffusion sorption
flnm = "1D_diff-sorp_NA_NA"
base_path= '../data_download/data/1D/diffusion-sorption/'
file_path = os.path.abspath(base_path + flnm + ".h5")
initial_step = 10

with h5py.File(file_path, 'r') as h5_file:
    data_list = sorted(h5_file.keys())
    seed_group = h5_file[data_list[0]]
    # data dim = [t, x1, ..., xd, v]
    time = np.array(seed_group["grid"]["t"])
    with open(flnm+".time.pkl", 'rb') as f:
        err_mean = pickle.load(f)
        err_mean = err_mean.numpy()#[batch,t,ch]
plt.subplot(313)
plt.plot(time[initial_step:],err_mean[0,initial_step:,0], )
plt.title("Diffusion Sorption")
plt.axvline(x=time[len(time)//2],ymin=0,ymax=1,ls="-.",c="g")
plt.axis([0, 505, -1e-2, 1e-2])



plt.ylabel('MSE')
plt.xlabel('time(s)')
#plt.show()
plt.savefig("temporal_analyze.png")
