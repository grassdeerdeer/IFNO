# coding:UTF-8
# @Time: 2022/11/11 9:14
# @Author: Lulu Cao
# @File: getDataset.py
# @Software: PyCharm


import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt

class IFNODatasetSingle(Dataset):
    def __init__(self,filename,saved_folder='../data/',
                 reduced_resolution=1,reduced_resolution_t=1,reduced_batch=1,
                 if_test=False,test_ratio=0.1,num_samples_max=-1,
                 initial_step=10,notime=False):
        # Define path to files
        root_path = os.path.abspath(saved_folder + filename)
        self.initial_step = initial_step
        self.notime = notime
        assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'

        with h5py.File(root_path, 'r') as f:
            self.keys = list(f.keys())
            self.keys.sort()
            if 'tensor' not in self.keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                if len(idx_cfd) == 3:  # 1D
                    self.data = np.zeros([idx_cfd[0] // reduced_batch,
                                          idx_cfd[2] // reduced_resolution,
                                          mt.ceil(idx_cfd[1] / reduced_resolution_t),
                                          3],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[..., 0] = _data  # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[..., 1] = _data  # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[..., 2] = _data  # batch, x, t, ch


                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)

                    self.data_grid_x = torch.tensor(f["x-coordinate"][::reduced_resolution], dtype=torch.float)
                    self.data_grid_t = torch.tensor(f["t-coordinate"][::reduced_resolution_t], dtype=torch.float)
                    self.tdim = self.data.shape[-2]
                    self.data_grid_t = self.data_grid_t[:self.tdim]
                    #
                    XX, TT = torch.meshgrid(
                        [self.data_grid_x, self.data_grid_t[initial_step:]]
                    )
                    self.data_input = torch.vstack([XX.ravel(), TT.ravel()]).T

                if len(idx_cfd) == 4:  # 2D
                    self.data = np.zeros([idx_cfd[0] // reduced_batch,
                                          idx_cfd[2] // reduced_resolution,
                                          idx_cfd[3] // reduced_resolution,
                                          mt.ceil(idx_cfd[1] / reduced_resolution_t),
                                          4],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 0] = _data  # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 1] = _data  # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 2] = _data  # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 3] = _data  # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y)
                    self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]


                    self.data_grid_t = torch.tensor(f["t-coordinate"][::reduced_resolution_t], dtype=torch.float)
                    self.tdim = self.data.shape[-2]
                    self.data_grid_t = self.data_grid_t[:self.tdim]
                    #
                    XX, YY, TT = torch.meshgrid(
                        [x,y, self.data_grid_t[initial_step:]]
                    )
                    self.data_input = torch.vstack([XX.ravel(), YY.ravel(), TT.ravel()]).T

                if len(idx_cfd) == 5:  # 3D
                    self.data = np.zeros([idx_cfd[0] // reduced_batch,
                                          idx_cfd[2] // reduced_resolution,
                                          idx_cfd[3] // reduced_resolution,
                                          idx_cfd[4] // reduced_resolution,
                                          mt.ceil(idx_cfd[1] / reduced_resolution_t),
                                          5],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution,
                            ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 0] = _data  # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution,
                            ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 1] = _data  # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution,
                            ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 2] = _data  # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution,
                            ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 3] = _data  # batch, x, t, ch
                    # Vz
                    _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution,
                            ::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 4] = _data  # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    z = np.array(f["z-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    z = torch.tensor(z, dtype=torch.float)
                    X, Y, Z = torch.meshgrid(x, y, z)
                    self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution, \
                                ::reduced_resolution, \
                                ::reduced_resolution]

                    self.data_grid_t = torch.tensor(f["t-coordinate"][::reduced_resolution_t], dtype=torch.float)
                    self.tdim = self.data.shape[-2]
                    self.data_grid_t = self.data_grid_t[:self.tdim]
                    #
                    XX, YY, ZZ, TT = torch.meshgrid(
                        [x, y, z,self.data_grid_t[initial_step:]]
                    )
                    self.data_input = torch.vstack([XX.ravel(), YY.ravel(), ZZ.ravel(),TT.ravel()]).T

            else:  # scalar equations
                #self._data = torch.tensor(f['tensor'], dtype=torch.float)
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    #
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]

                    ## convert to [x1, ..., xd, t,v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch
                    # init_data = np.repeat(_data[...,0,None,None],_data.shape[-1],axis=-2)
                    # init_data = np.transpose(init_data[:, :, :], (1, 2, 0))
                    # init_data = init_data[:,:,:,None]
                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)

                    self.data_grid_x = torch.tensor(f["x-coordinate"][::reduced_resolution], dtype=torch.float)
                    self.data_grid_t = torch.tensor(f["t-coordinate"][::reduced_resolution_t], dtype=torch.float)
                    self.tdim = self.data.shape[-2]
                    self.data_grid_t = self.data_grid_t[:self.tdim]
                    #
                    XX, TT = torch.meshgrid(
                        [self.data_grid_x, self.data_grid_t[initial_step:]]
                    )
                    # self.list_t = list(self.data_grid_t)
                    # self.list_x = list(self.data_grid_x)
                    #
                    self.data_input = torch.vstack([XX.ravel(), TT.ravel()]).T


                if len(_data.shape) == 4:  # 2D Darcy flow
                    # u: label
                    _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    #if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    self.data = _data
                    # nu: input
                    _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1)) # batch,xn,time
                    self.data = np.concatenate([_data, self.data], axis=-1)
                    self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y)
                    self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

                    self.tdim = self.data.shape[-2]
                    self.data_grid_t = torch.zeros(self.tdim, dtype=torch.float)
                    XX, YY, TT = torch.meshgrid(
                        [x, y, self.data_grid_t[initial_step:]]
                    )
                    self.data_input = torch.vstack([XX.ravel(), YY.ravel(),TT.ravel()]).T

        self.data_shape = self.data.shape
        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            # self.hidden = self.hidden[:test_idx]
            # self.output = self.output[:test_idx]
            self.data = self.data[:test_idx]
        else:
            # self.hidden = self.hidden[test_idx:num_samples_max]
            # self.output = self.output[test_idx:num_samples_max]
            self.data = self.data[test_idx:num_samples_max]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # [x1, ..., xd, t, v]
        return self.data_input, self.data[idx,...,:self.initial_step,:],self.data[idx], self.grid
        # return self.data_input, self.data[idx,:1,...], self.data[idx,1:,...]
        #return self.coordinate, self.hidden[idx,...,], self.x_input, self.output[idx]







class IFNODatasetMult(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data_download/data/2D/diffusion-reaction/',
                 if_test=False, test_ratio=0.1,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 temporal_train=False,
                 ):
        # Define path to files
        self.file_path = os.path.abspath(saved_folder + filename + ".h5")

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            seed_group = h5_file[data_list[0]]
            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        self.reduced_batch = reduced_batch

        self.data_shape = [0]+list(data.shape)
        self.temporal_train=temporal_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend(list([0, -1]))  # 在原来的list后追加[0,-1]
            data = data.permute(permute_idx)

            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                data = data[::self.reduced_resolution,::self.reduced_resolution_t,:]

                grid = np.array(seed_group["grid"]["x"], dtype='f')
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)

                data_grid_x = torch.tensor(seed_group["grid"]["x"][::self.reduced_resolution], dtype=torch.float)
                self.data_grid_t = torch.tensor(seed_group["grid"]["t"][::self.reduced_resolution_t], dtype=torch.float)
                tdim = data.shape[-2]
                self.data_grid_t = self.data_grid_t[:tdim]

                # For Temporal Error Analysis
                if self.temporal_train:
                    self.data_grid_t = self.data_grid_t[:self.data_shape[1] // 2]

                #
                XX, TT = torch.meshgrid(
                    [data_grid_x, self.data_grid_t[self.initial_step:]]
                )
                data_input = torch.vstack([XX.ravel(), TT.ravel()]).T

            elif dim == 2:
                # test Temporal Error
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y)
                grid = torch.stack((X, Y), axis=-1)

                self.data_grid_t= torch.tensor(seed_group["grid"]["t"][::self.reduced_resolution_t], dtype=torch.float)
                tdim = data.shape[-2]
                self.data_grid_t = self.data_grid_t[:tdim]

                # For Temporal Error Analysis
                if self.temporal_train:
                    self.data_grid_t=self.data_grid_t[:self.data_shape[1]//2]
                #
                XX, YY, TT = torch.meshgrid(
                    [x, y, self.data_grid_t[self.initial_step:]]
                )
                data_input = torch.vstack([XX.ravel(), YY.ravel(), TT.ravel()]).T

            elif dim == 3:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                z = np.array(seed_group["grid"]["z"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z)
                grid = torch.stack((X, Y, Z), axis=-1)

                self.data_grid_t = torch.tensor(seed_group["grid"]["t"][::self.reduced_resolution_t], dtype=torch.float)
                tdim = data.shape[-2]
                self.data_grid_t= self.data_grid_t[:tdim]
                #
                XX, YY, ZZ, TT = torch.meshgrid(
                    [x, y, z, self.data_grid_t[self.initial_step:]]
                )
                data_input = torch.vstack([XX.ravel(), YY.ravel(), ZZ.ravel(), TT.ravel()]).T

            # For Temporal Error Analysis
            if self.temporal_train:
                return data_input, data[..., :self.initial_step, :], data[..., :self.data_shape[1]//2, :], grid
        return data_input, data[...,:self.initial_step,:], data, grid





if __name__=="__main__":
    filename = "1D_Advection_Sols_beta0.1.hdf5"
    root_path = "../../data_download/data/1D/Advection/Train/"
