# coding:UTF-8
# @Time: 2023/1/24 15:52
# @Author: Lulu Cao
# @File: PDE_figure.py
# @Software: PyCharm
import torch
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from timeit import default_timer
def plot_data_pred(dim,target_plot,pred_plot,channel_plot,model_name,initial_step=10):
    plt.ioff()
    x_min, x_max, y_min, y_max, t_min, t_max = 0, 1, 0, 1, 0, 1
    if dim == 1:

        fig, ax = plt.subplots(figsize=(6.5, 6))
        h = ax.imshow(pred_plot[..., channel_plot].squeeze().detach().cpu(),
                      extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto') #热图
        h.set_clim(target_plot[..., channel_plot].min(), target_plot[..., channel_plot].max())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        ax.set_title("Prediction", fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel("$x$", fontsize=20)
        ax.set_xlabel("$t$", fontsize=20)
        plt.tight_layout()
        filename = model_name + '_pred.pdf'
        plt.savefig(filename)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        h = ax.imshow(target_plot[..., channel_plot].squeeze().detach().cpu(),
                      extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
        h.set_clim(target_plot[..., channel_plot].min(), target_plot[..., channel_plot].max())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        ax.set_title("Data", fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel("$x$", fontsize=20)
        ax.set_xlabel("$t$", fontsize=20)
        plt.tight_layout()
        filename = model_name + '_data.pdf'
        plt.savefig(filename)

    elif dim == 2:
        time = [x for x in range(10,100,10)]
        for t in time:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(pred_plot[..., t, channel_plot].squeeze().t().detach().cpu(),
                          extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[..., t, channel_plot].min(), target_plot[..., t, channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=20)
            ax.set_title(f"Prediction t={t/100*t_max}s ", fontsize=20)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_ylabel("$y$", fontsize=20)
            ax.set_xlabel("$x$", fontsize=20)
            plt.tight_layout()
            filename = model_name + str(t)+'_pred.pdf'
            plt.savefig(filename)

            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(target_plot[..., t, channel_plot].squeeze().t().detach().cpu(),
                          extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[..., t, channel_plot].min(), target_plot[..., t, channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=20)
            ax.set_title(f"Data  t={t/100*t_max}s ", fontsize=20)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_ylabel("$y$", fontsize=20)
            ax.set_xlabel("$x$", fontsize=20)
            plt.tight_layout()
            filename = model_name + str(t)+'_data.pdf'
            plt.savefig(filename)

    # plt.figure(figsize=(8,8))
    # plt.semilogy(torch.arange(initial_step,yy.shape[-2]),
    #              val_l2_time[initial_step:].detach().cpu())
    # plt.xlabel('$t$', fontsize=30)
    # plt.ylabel('$MSE$', fontsize=30)
    # plt.title('MSE vs unrolled time steps', fontsize=30)
    # plt.tight_layout()
    # filename = model_name + '_mse_time.pdf'
    # plt.savefig(filename)


