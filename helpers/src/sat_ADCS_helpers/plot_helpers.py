import numpy as np
import math
import pytest
import matplotlib.pyplot as plt
from .helpers import *


class extra:
    def __init__(self):
        pass

def plot3d_on_move(event,axlist,fig):
    for ax in axlist:
        if event.inaxes == ax:
            for k in axlist:
                if k != ax:
                    k.view_init(elev=ax.elev, azim=ax.azim)
    fig.canvas.draw_idle()

def ct(arr):
    if len(np.shape(arr))>=3:
        ons = 2*np.pi*np.ones([1]+list(arr.shape[1:]))
        a = np.concatenate([ons, arr],axis = 0)
    si = np.sin(a)
    si[0,...] = 1
    si = np.cumprod(si,axis = 0)
    co = np.cos(a)
    co = np.roll(co, -1,axis = 0)
    return si*co


def plot_the_thing(data,data_est = None,norm=False,act_v_est = False,transpose_auto=True,title=None,xlabel=None,ylabel=None,legend_list = None,xdata = None,plot_now = True,save = False,save_name = None,scatter=False):
    plt.figure()
    ax = plt.subplot()
    vals = np.copy(np.array(data))
    nr = vals.shape[0]
    flat_mat = False
    if len(vals.shape)==0:
        raise ValueError("What? how? array is zero-dimensional")
    elif len(vals.shape)==1:
        vals = vals.reshape((vals.shape[0],1))
        flat_mat = True
    elif len(vals.shape)>2:
        raise ValueError("This function can currently only plot 1- and 2-D arrays. This has more.")
    nc = vals.shape[1]
    if nr<nc and transpose_auto:
        vals = vals.T

    if save and save_name is None:
        raise ValueError("need a name to save.")

    if data_est is None and act_v_est:
        raise ValueError("want to compare actual and estimated but only one data set provided")

    nr = vals.shape[0]
    nc = vals.shape[1]
    if legend_list is None:
        legend_list = [str(i) for i in range(nc)]
    if title is None:
        title = "The Thing!"
    if xlabel is None:
        xlabel = "proabably time?"
    if ylabel is None:
        ylabel = "data"
    if not scatter:

        if act_v_est:
            vals2 = np.copy(np.array(data_est))
            act_style = style = ["-b","-r","-g","-m","-k"]
            est_style = [":b",":r",":g",":m",":k"]
            if xdata is None:
                raise ValueError("not yet implemented")
            else:
                # print(xdata.shape,vals.shape)
                if len(xdata)>vals.shape[0]:

                    # style = ["-b","-r","-g",":b",":r",":g",]
                    # breakpoint()
                    for i in range(6):
                        ax.plot(data[1].t_hist,use_data[:,i],style[i],label=legends[i], linewidth=1)


                    ll = vals.shape[0]
                    for i in range(nc):
                        ax.plot(xdata[:ll],vals[:ll,i],act_style[i],label=legend_list[i]+" actual", linewidth=1)
                    for i in range(nc):
                        ax.plot(xdata[:ll],vals2[:ll,i],est_style[i],label=legend_list[i]+" est", linewidth=1)
                    if norm:
                        raise ValueError("not yet implemented")
                else:
                    for i in range(nc):
                        ax.plot(xdata,vals[:,i],act_style[i],label=legend_list[i]+" actual", linewidth=1)
                    for i in range(nc):
                        ax.plot(xdata,vals2[:,i],est_style[i],label=legend_list[i]+" est", linewidth=1)
                    if norm:
                        raise ValueError("not yet implemented")

        else:
            if xdata is None:
                for i in range(nc):
                    ax.plot(vals[:,i],label=legend_list[i])
                if norm:
                    if flat_mat:
                        ax.plot(abs(vals),label='norm')
                    else:
                        ax.plot(matrix_row_norm(vals),label='norm')
            else:
                # print(xdata.shape,vals.shape)
                if len(xdata)>vals.shape[0]:
                    ll = vals.shape[0]
                    for i in range(nc):
                        ax.plot(xdata[:ll],vals[:ll,i],label=legend_list[i])
                    if norm:
                        if flat_mat:
                            ax.plot(xdata[:ll],abs(vals[:ll]),label='norm')
                        else:
                            ax.plot(xdata[:ll],matrix_row_norm(vals[:ll,:]),label='norm')
                else:
                    for i in range(nc):
                        ax.plot(xdata,vals[:,i],label=legend_list[i])
                    if norm:
                        if flat_mat:
                            ax.plot(xdata,abs(vals),label='norm')
                        else:
                            ax.plot(xdata,matrix_row_norm(vals),label='norm')
    else:
        if xdata is None:
            for i in range(1,nc):
                ax.scatter(vals[:,0],vals[:,i],label=legend_list[i])
            # if norm:
            #     if flat_mat:
            #         ax.plot(abs(vals),label='norm')
            #     else:
            #         ax.plot(matrix_row_norm(vals),label='norm')
        else:
            # print(xdata.shape,vals.shape)
            if len(xdata)>vals.shape[0]:
                ll = vals.shape[0]
                for i in range(nc):
                    ax.scatter(xdata[:ll],vals[:ll,i],label=legend_list[i])
                # if norm:
                #     if flat_mat:
                #         ax.plot(xdata[:ll],abs(vals[:ll]),label='norm')
                #     else:
                #         ax.plot(xdata[:ll],matrix_row_norm(vals[:ll,:]),label='norm')
            else:
                for i in range(nc):
                    ax.scatter(xdata,vals[:,i],label=legend_list[i])
                # if norm:
                #     if flat_mat:
                #         ax.plot(xdata,abs(vals),label='norm')
                #     else:
                #         ax.plot(xdata,matrix_row_norm(vals),label='norm')
    # Bb = np.hstack([(rot_mat(sim.state_history[3:7,j]).T@sim.orbit_history[j].B).reshape((3,1)) for j in range(sim.sim_length)])
    ax.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(save_name+'.png')

    if plot_now:
        plt.draw()
        plt.pause(1)
    else:
        plt.close()
