#estimation plots for paper
from sat_ADCS_estimation import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
import numpy as np
import math
from scipy.integrate import odeint, solve_ivp, RK45
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import erfinv
import time
import dill as pickle
import copy
import os
import re
from squaternion import Quaternion


def get_data(dirlist,name_core,path=""):
    opts = [(d,d.replace(name_core,'')) for d in dirlist if name_core in d]
    use = [d[0] for d in opts if re.match('^[0-9\.\ \_\-]*$',d[1])]
    with open(path+"/"+use[0]+"/data","rb") as fp:
        output = pickle.load(fp)
    return output

basepath = "disturbance_paper_test_files"
dirnames = next(os.walk(basepath), (None, None, []))[1]  # [] if no file

names = [("Wie","Wie","Wie","wie"),("Lovera","Lovera","Lovera","lovera"),("Wisniewski","Wisniewski","Wisniewski","wisniewski"),("Wisniewski_twist","Modified Wisniewski","Wisniewski","wisniewski_twist")]
ctrl_units = ["Nm","Am^2","Am^2","Am^2"]

if False:
    #Wie test case; his vs. mine and 1 vs 10 sec time step; not close; w/ dipole
    for ii in range(len(names)):
        basename = names[ii][0]
        use_name = names[ii][1]
        sat_name = names[ii][2]
        lc_basename = names[ii][3]
        names_1 = [basename+k for k in ["_matching1","_disturbed1","_disturbed_w_control1","_disturbed_w_gencontrol1"]]
        # names_1 = ["Wie_matching1","Wie_disturbed1","Wie_disturbed_w_control1","Wie_disturbed_w_gencontrol1"]
        legends_1 = ["Clean","Disturbed","Disturbance Control","General Disturbance"]
        # code_1 = ["mine_10","mine_1","usque_10","usque_1"]
        data_1 = [get_data(dirnames,n,basepath) for n in names_1]
        angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip([np.sum(d.state_hist[j,3:7]*d.goal_hist[j].state[3:7]) for j in range(len(d.state_hist))],-1,1)**2.0 ) for d in data_1]
        mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_1]
        rpy_err = [np.stack([Quaternion(*quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7])).to_euler(degrees=True) for j in range(d.state_hist.shape[0])]) for d in data_1]

        #ang diff overall
        plt.figure()
        ylim = [-5,190]
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
            # print(data_1[i].state_hist[0,3:7])
            # print(quat_mult(data_1[i].state_hist[0,3:7],mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4))))
        # plt.show()
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = '',color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title(use_name+" Law on "+sat_name+" Satellite Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("angular_error_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)


        #q0
        plt.figure()
        ylim = [-0.05,1.05]
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,np.array([data_1[i].goal_hist[j].state[3] for j in range(len(data_1[i].goal_hist))]),label=legends_1[i])

        # ax.set_yscale('log')
        # plt.show()
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = '',color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title(use_name+" Law on "+sat_name+" Satellite Quat0 Goal")
        plt.xlabel("Time (s)")
        plt.ylabel("Quat")
        plt.savefig("quat0_goal_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)

        #log ang diff overall
        plt.figure()
        ylim = [1e-3,350]
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])


        # ax.set_ylim(ylim[0],ylim[1])
        ax.set_yscale('log')
        # plt.show()
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-6), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        plt.title(use_name+" Law on "+sat_name+" Satellite Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("log_angular_error_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)
        #
        # #mrp
        # sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        # ylim = [-0.1,0.1]
        # for i in range(4):
        #     axs = [0,0,0]
        #     fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        #     axs = [ax1,ax2,ax3]
        #     fig.suptitle("TRMM Satellite MRP Error with Initial Atittude and Bias Errors (" + legends_1[i] +")")
        #     for r in range(3):
        #         # if r>0:
        #         #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        #         # else:
        #         #     axs[r] = plt.subplot(3,1,r+1)
        #         #     axs[r].set_xlabel("Time (s)")
        #         axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
        #         axs[r].set_ylabel(sp_labels[r])
        #         axs[r].set_ylim(ylim[0],ylim[1])
        #     axs[r].set_xlabel("Time (s)")
        #     plt.savefig(code_1[i]+"mrp_TRMM_initially_off.png")
        #     plt.draw()
        #     plt.pause(1)

        #mrp combo
        sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        ylim = [-1,1]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(use_name+" Law on "+sat_name+" Satellite MRP Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(4):
                axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])

            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "",color = 'k')

            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("mrp_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)


        #rpy combo
        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-1,1]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(use_name+" Law on "+sat_name+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(4):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "",color = 'k')

            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+"_limited.png")
        plt.draw()
        plt.pause(1)

        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-180,180]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(use_name+" Law on "+sat_name+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(4):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "",color = 'k')

            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)

        #AV by axis
        sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
        ylim = [-0.01,0.01]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))])) for d in data_1]
        fig.suptitle(use_name+" Law on "+sat_name+" Satellite Rate Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(4):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])
            axs[r].set_ylim(ylim[0],ylim[1])
            ylim = axs[r].get_ylim()
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "",color = 'k')

            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("axes_av_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)


        #AV norm
        fig = plt.figure()
        ylim = [5e-5,2]
        ax = fig.add_subplot(1,1,1)
        use_data = [matrix_row_norm((180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))]))) for d in data_1]
        ax.set_title(use_name+" Law on "+sat_name+" Satellite Rate Error ")

        for i in range(4):
            ax.plot(data_1[i].t_hist,use_data[i],label = legends_1[i])
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlabel("Time (s)")
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_ylabel("Rate Error Norm (deg/s)")
        ax.set_yscale('log')
        plt.savefig("log_norm_av_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)

        #Ctrl by axis
        sp_labels = [k+" "+ctrl_units[ii] for k in ["x axis","y axis","z axis"]]
        # ylim = [-0.05,0.05]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(np.array([data_1[i].control_hist[j,0:3]+data_1[i].state_hist[j,7:10]*(i>0) for j in range(len(data_1[i].state_hist))])) for i in range(len(data_1))]
        fig.suptitle(use_name+" Law on "+sat_name+" Satellite Control Effort")#"(Bias Removed)")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(4):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])

            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "",color = 'k')
            if r==0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("ctrl_"+lc_basename+".png")
        plt.draw()
        plt.pause(1)

        plt.pause(3)
        plt.close('all')


#cubesat for lovera, wisniewski, twistniewski

cubesat_colors = ["tab:orange","tab:green","tab:red"]
names = [("Lovera","lovera"),("Wisniewski","wisniewski"),("Wisniewski_twist","wisniewski_twist")]
names = [("Lovera","Lovera","lovera"),("Wisniewski","Wisniewski","wisniewski"),("Wisniewski_twist","Modified Wisniewski","wisniewski_twist")]

if False:
    #Wie test case; his vs. mine and 1 vs 10 sec time step; not close; w/ dipole
    for ii in range(len(names)):
        basename = names[ii][0]
        use_name = names[ii][1]
        lc_basename = names[ii][2]
        names_1 = [basename+"_on_cubesat"+k for k in ["_disturbed","","_gen"]]
        # names_1 = ["Wie_matching1","Wie_disturbed1","Wie_disturbed_w_control1","Wie_disturbed_w_gencontrol1"]

        legends_1 = ["Disturbed","Disturbance Control","General Disturbance"]
        # code_1 = ["mine_10","mine_1","usque_10","usque_1"]
        data_1 = [get_data(dirnames,n,basepath) for n in names_1]
        angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip([np.sum(d.state_hist[j,3:7]*d.goal_hist[j].state[3:7]) for j in range(len(d.state_hist))],-1,1)**2.0 ) for d in data_1]
        mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_1]
        rpy_err = [np.stack([Quaternion(*quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7])).to_euler(degrees=True) for j in range(d.state_hist.shape[0])]) for d in data_1]

        #ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i],color = cubesat_colors[i])
            # print(data_1[i].state_hist[0,3:7])
            # print(quat_mult(data_1[i].state_hist[0,3:7],mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4))))
        # plt.show()
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

        ax.set_ylim(ylim[0],ylim[1])
        plt.title(use_name+" Law on CubeSat Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("angular_error_"+lc_basename+"_cubesat.png")
        plt.draw()
        plt.pause(1)


        #log ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i],color = cubesat_colors[i])
        ax.set_yscale('log')
        # plt.show()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

        ax.set_ylim(ylim[0],ylim[1])
        plt.title(use_name+" Law on CubeSat Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("log_angular_error_"+lc_basename+"_cubesat.png")
        plt.draw()
        plt.pause(1)
        #
        # #mrp
        # sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        # ylim = [-0.1,0.1]
        # for i in range(4):
        #     axs = [0,0,0]
        #     fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        #     axs = [ax1,ax2,ax3]
        #     fig.suptitle("TRMM Satellite MRP Error with Initial Atittude and Bias Errors (" + legends_1[i] +")")
        #     for r in range(3):
        #         # if r>0:
        #         #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        #         # else:
        #         #     axs[r] = plt.subplot(3,1,r+1)
        #         #     axs[r].set_xlabel("Time (s)")
        #         axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
        #         axs[r].set_ylabel(sp_labels[r])
        #         axs[r].set_ylim(ylim[0],ylim[1])
        #     axs[r].set_xlabel("Time (s)")
        #     plt.savefig(code_1[i]+"mrp_TRMM_initially_off.png")
        #     plt.draw()
        #     plt.pause(1)

        #mrp combo
        sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        ylim = [-10,10]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(use_name+" Law on CubeSat MRP Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r],color = cubesat_colors[i])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("mrp_"+lc_basename+"_cubesat.png")
        plt.draw()
        plt.pause(1)


        #rpy combo
        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-5,5]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(use_name+" Law on CubeSat Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r],color = cubesat_colors[i])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+"_limited_cubesat.png")
        plt.draw()
        plt.pause(1)

        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-180,180]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle(use_name+" Law on CubeSat Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r],color = cubesat_colors[i])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+"_cubesat.png")
        plt.draw()
        plt.pause(1)

        #AV by axis
        sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
        ylim = [-0.05,0.05]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))])) for d in data_1]
        fig.suptitle(use_name+" Law on CubeSat Rate Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r],color = cubesat_colors[i])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

            axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("axes_av_"+lc_basename+"_cubesat.png")
        plt.draw()
        plt.pause(1)


        #AV norm
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        use_data = [matrix_row_norm((180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))]))) for d in data_1]
        ax.set_title(use_name+" Law on CubeSat Rate Error ")

        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,use_data[i],label = legends_1[i],color = cubesat_colors[i])
        # ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.set_yscale('log')
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')

        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

        ax.set_ylim(ylim[0],ylim[1])
        ax.set_ylabel("Rate Error Norm (deg/s)")
        plt.savefig("log_norm_av_"+lc_basename+"_cubesat.png")
        plt.draw()
        plt.pause(1)


        #Ctrl by axis
        sp_labels = ["x axis (Am^2)","y axis (Am^2)","z axis (Am^2)"]
        # ylim = [-0.05,0.05]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(np.array([data_1[i].control_hist[j,0:3]+data_1[i].state_hist[j,7:10] for j in range(len(data_1[i].state_hist))])) for i in range(len(data_1))]
        fig.suptitle(use_name+" Law on CubeSat Control Effort")# (Bias Removed)")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r],color = cubesat_colors[i])
            axs[r].set_ylabel(sp_labels[r])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("ctrl_"+lc_basename+"_cubesat.png")
        plt.draw()
        plt.pause(1)

        plt.pause(3)
        plt.close('all')




#wisniewski vs twistniewski
cases = [("_matching1","Clean"),("_disturbed1","Disturbed"),("_disturbed_w_control1","Disturbance-Aware"),("_disturbed_w_gencontrol1","Generalized Disturbance")]
extra_name = ""
extra_name_title = ""
if True:
    #Wie test case; his vs. mine and 1 vs 10 sec time step; not close; w/ dipole
    for ii in range(len(cases)):

        basename = cases[ii][1]
        lc_basename = cases[ii][0][1:-1]
        names_1 = ["Wisniewski"+extra_name+k+"_"+lc_basename+"1" for k in ["","_twist"]]
        # names_1 = ["Wie_matching1","Wie_disturbed1","Wie_disturbed_w_control1","Wie_disturbed_w_gencontrol1"]
        legends_1 = ["Original","Modified"]
        # code_1 = ["mine_10","mine_1","usque_10","usque_1"]
        data_1 = [get_data(dirnames,n,basepath) for n in names_1]
        angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip([np.sum(d.state_hist[j,3:7]*d.goal_hist[j].state[3:7]) for j in range(len(d.state_hist))],-1,1)**2.0 ) for d in data_1]
        mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_1]
        rpy_err = [np.stack([Quaternion(*quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7])).to_euler(degrees=True) for j in range(d.state_hist.shape[0])]) for d in data_1]

        #ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
            # print(data_1[i].state_hist[0,3:7])
            # print(quat_mult(data_1[i].state_hist[0,3:7],mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4))))
        # plt.show()
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title(basename + " Original vs Modified Law on Oersted Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("angular_error_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #q0
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,np.array([data_1[i].goal_hist[j].state[3] for j in range(len(data_1[i].goal_hist))]),label=legends_1[i])
        ylim = ax.get_ylim()
        # ax.set_yscale('log')
        # plt.show()
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite Quat0 Goal")
        # plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Quat0 Goal")
        plt.xlabel("Time (s)")
        plt.ylabel("Quat")
        plt.savefig("quat0_goal_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        #log ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
        ax.set_yscale('log')
        ylim = ax.get_ylim()
        # plt.show()
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite Angular Error")
        # plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("log_angular_error_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)
        #
        # #mrp
        # sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        # ylim = [-0.1,0.1]
        # for i in range(len(names_1)):
        #     axs = [0,0,0]
        #     fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        #     axs = [ax1,ax2,ax3]
        #     fig.suptitle("TRMM Satellite MRP Error with Initial Atittude and Bias Errors (" + legends_1[i] +")")
        #     for r in range(3):
        #         # if r>0:
        #         #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        #         # else:
        #         #     axs[r] = plt.subplot(3,1,r+1)
        #         #     axs[r].set_xlabel("Time (s)")
        #         axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
        #         axs[r].set_ylabel(sp_labels[r])
        #         axs[r].set_ylim(ylim[0],ylim[1])
        #     axs[r].set_xlabel("Time (s)")
        #     plt.savefig(code_1[i]+"mrp_TRMM_initially_off_wis_twist_comp.png")
        #     plt.draw()
        #     plt.pause(1)

        #mrp combo
        sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        ylim = [-1,1]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite MRP Error")
        fig.suptitle("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite MRP Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
            ylim = axs[r].get_ylim()
            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("mrp_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #rpy combo
        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-1,1]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        fig.suptitle("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")

            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            ylim = axs[r].get_ylim()
            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+extra_name+"_limited_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-180,180]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        fig.suptitle("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            ylim = axs[r].get_ylim()
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        #AV by axis
        sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
        ylim = [-0.01,0.01]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))])) for d in data_1]
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Rate Error")
        fig.suptitle("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite Rate Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])
            ylim = axs[r].get_ylim()
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("axes_av_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #AV norm
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        use_data = [matrix_row_norm((180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))]))) for d in data_1]
        # ax.set_title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Rate Error ")
        fig.suptitle("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite Rate Error")

        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,use_data[i],label = legends_1[i])
        # ax.set_ylim(ylim[0],ylim[1])
        ax.set_yscale('log')
        ax.set_xlabel("Time (s)")
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_ylabel("Rate Error Norm (deg/s)")
        plt.savefig("log_norm_av_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #Ctrl by axis
        sp_labels = ["x axis (Am^2)","y axis (Am^2)","z axis (Am^2)"]
        # ylim = [-0.05,0.05]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(np.array([data_1[i].control_hist[j,0:3]+data_1[i].state_hist[j,7:10] for j in range(len(data_1[i].state_hist))])) for i in range(len(data_1))]
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Control Effort")# (Bias Removed)")
        fig.suptitle("Original vs Modified "+extra_name_title+"Sliding Law on "+basename+" Satellite Control Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylabel(sp_labels[r])
            axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("ctrl_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        plt.pause(3)
        plt.close('all')

#wisniewski vs twistniewski alt
cases = [("_disturbed1","Disturbed"),("_disturbed_w_control1","Disturbance Control"),("_disturbed_w_gencontrol1","Generalized Control")]
extra_name = "_alt"
extra_name_title = "Alternative "
if False:
    #Wie test case; his vs. mine and 1 vs 10 sec time step; not close; w/ dipole
    for ii in range(len(cases)):

        basename = cases[ii][1]
        lc_basename = cases[ii][0][1:-1]
        names_1 = ["Wisniewski"+extra_name+k+"_"+lc_basename+"1" for k in ["","_twist"]]
        # names_1 = ["Wie_matching1","Wie_disturbed1","Wie_disturbed_w_control1","Wie_disturbed_w_gencontrol1"]
        legends_1 = ["Sliding","Twisting"]
        # code_1 = ["mine_10","mine_1","usque_10","usque_1"]
        data_1 = [get_data(dirnames,n,basepath) for n in names_1]
        angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip([np.sum(d.state_hist[j,3:7]*d.goal_hist[j].state[3:7]) for j in range(len(d.state_hist))],-1,1)**2.0 ) for d in data_1]
        mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_1]
        rpy_err = [np.stack([Quaternion(*quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7])).to_euler(degrees=True) for j in range(d.state_hist.shape[0])]) for d in data_1]

        #ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
            # print(data_1[i].state_hist[0,3:7])
            # print(quat_mult(data_1[i].state_hist[0,3:7],mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4))))
        # plt.show()
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        plt.title("Original vs Modified "+extra_name_title+basename+" Law on "+basename+" Satellite Quat0 Goal")
        # plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("angular_error_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #q0
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,np.array([data_1[i].goal_hist[j].state[3] for j in range(len(data_1[i].goal_hist))]),label=legends_1[i])
        # ax.set_yscale('log')
        # plt.show()
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Quat0 Goal")
        plt.xlabel("Time (s)")
        plt.ylabel("Quat")
        plt.savefig("quat0_goal_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        #log ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
        ax.set_yscale('log')
        # plt.show()
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("log_angular_error_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)
        #
        # #mrp
        # sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        # ylim = [-0.1,0.1]
        # for i in range(len(names_1)):
        #     axs = [0,0,0]
        #     fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        #     axs = [ax1,ax2,ax3]
        #     fig.suptitle("TRMM Satellite MRP Error with Initial Atittude and Bias Errors (" + legends_1[i] +")")
        #     for r in range(3):
        #         # if r>0:
        #         #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        #         # else:
        #         #     axs[r] = plt.subplot(3,1,r+1)
        #         #     axs[r].set_xlabel("Time (s)")
        #         axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
        #         axs[r].set_ylabel(sp_labels[r])
        #         axs[r].set_ylim(ylim[0],ylim[1])
        #     axs[r].set_xlabel("Time (s)")
        #     plt.savefig(code_1[i]+"mrp_TRMM_initially_off_wis_twist_comp.png")
        #     plt.draw()
        #     plt.pause(1)

        #mrp combo
        sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        ylim = [-10,10]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite MRP Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("mrp_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #rpy combo
        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-10,10]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+extra_name+"_limited_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-180,180]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        #AV by axis
        sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
        ylim = [-0.03,0.03]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))])) for d in data_1]
        fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Rate Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("axes_av_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #AV norm
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        use_data = [matrix_row_norm((180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))]))) for d in data_1]
        ax.set_title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Rate Error ")

        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,use_data[i],label = legends_1[i])
        # ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.set_ylabel("Rate Error Norm (deg/s)")
        ax.set_yscale('log')
        plt.savefig("log_norm_av_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #Ctrl by axis
        sp_labels = ["x axis (Am^2)","y axis (Am^2)","z axis (Am^2)"]
        # ylim = [-0.05,0.05]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(np.array([data_1[i].control_hist[j,0:3]+data_1[i].state_hist[j,7:10] for j in range(len(data_1[i].state_hist))])) for i in range(len(data_1))]
        fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Control Effort")# (Bias Removed)")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            # axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("ctrl_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        plt.pause(3)
        plt.close('all')

#wisniewski vs twistniewski cubesat
cases = [("_disturbed","Disturbed"),("","Disturbance-Aware"),("_gen","Generalized Control")]
extra_name = "_cubesat"
extra_name_title = "CubeSat "
if True:
    #Wie test case; his vs. mine and 1 vs 10 sec time step; not close; w/ dipole
    for ii in range(len(cases)):

        basename = cases[ii][1]
        lc_basename = cases[ii][0]
        names_1 = ["Wisniewski"+k+"_on_cubesat"+lc_basename for k in ["","_twist"]]
        # names_1 = ["Wie_matching1","Wie_disturbed1","Wie_disturbed_w_control1","Wie_disturbed_w_gencontrol1"]
        legends_1 = ["Original","Modified"]
        # code_1 = ["mine_10","mine_1","usque_10","usque_1"]
        # print(names_1)
        data_1 = [get_data(dirnames,n,basepath) for n in names_1]
        angdiff = [(180/np.pi)*np.arccos(-1 + 2*np.clip([np.sum(d.state_hist[j,3:7]*d.goal_hist[j].state[3:7]) for j in range(len(d.state_hist))],-1,1)**2.0 ) for d in data_1]
        mrp_err = [(180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7]),0)/2 for j in range(d.state_hist.shape[0])])) for d in data_1]
        rpy_err = [np.stack([Quaternion(*quat_mult(quat_inv(d.goal_hist[j].state[3:7]),d.state_hist[j,3:7])).to_euler(degrees=True) for j in range(d.state_hist.shape[0])]) for d in data_1]

        #ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i] )
            # print(data_1[i].state_hist[0,3:7])
            # print(quat_mult(data_1[i].state_hist[0,3:7],mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4))))
        # plt.show()
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title(basename + " Original vs Modified Law on Cubesat Angular Error")
        # plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("angular_error_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #q0
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,np.array([data_1[i].goal_hist[j].state[3] for j in range(len(data_1[i].goal_hist))]),label=legends_1[i])
        # ax.set_yscale('log')
        # plt.show()
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title("Original vs Modified Sliding Law on "+basename+" Cubesat Quat0 Goal")
        # plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Quat0 Goal")
        plt.xlabel("Time (s)")
        plt.ylabel("Quat")
        plt.savefig("quat0_goal_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        #log ang diff overall
        plt.figure()
        ax = plt.subplot()
        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,angdiff[i],label=legends_1[i])
        ax.set_yscale('log')
        # plt.show()
        ax.legend()
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        plt.title("Original vs Modified Sliding Law on "+basename+" Cubesat Angular Error")
        # plt.title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Error (deg)")
        plt.savefig("log_angular_error_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)
        #
        # #mrp
        # sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        # ylim = [-0.1,0.1]
        # for i in range(len(names_1)):
        #     axs = [0,0,0]
        #     fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        #     axs = [ax1,ax2,ax3]
        #     fig.suptitle("TRMM Satellite MRP Error with Initial Atittude and Bias Errors (" + legends_1[i] +")")
        #     for r in range(3):
        #         # if r>0:
        #         #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
        #         # else:
        #         #     axs[r] = plt.subplot(3,1,r+1)
        #         #     axs[r].set_xlabel("Time (s)")
        #         axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
        #         axs[r].set_ylabel(sp_labels[r])
        #         axs[r].set_ylim(ylim[0],ylim[1])
        #     axs[r].set_xlabel("Time (s)")
        #     plt.savefig(code_1[i]+"mrp_TRMM_initially_off_wis_twist_comp.png")
        #     plt.draw()
        #     plt.pause(1)

        #mrp combo
        sp_labels = ["MRP x (deg)","MRP y (deg)","MRP z (deg)"]
        ylim = [-1,1]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle("Original vs Modified Sliding Law on "+basename+" Cubesat MRP Error")

        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite MRP Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,mrp_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("mrp_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #rpy combo
        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-1,1]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle("Original vs Modified Sliding Law on "+basename+" Cubesat Angular Error")
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+extra_name+"_limited_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        sp_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        ylim = [-180,180]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        fig.suptitle("Original vs Modified Sliding Law on "+basename+" Cubesat Angular Error")
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Angular Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(3,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(3,1,r+1)
            #     axs[r].set_xlabel("Time (s)")
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,rpy_err[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylim(ylim[0],ylim[1])
        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("rpy_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        #AV by axis
        sp_labels = ["x axis (deg/s)","y axis (deg/s)","z axis (deg/s)"]
        ylim = [-0.01,0.01]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))])) for d in data_1]
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Rate Error")
        fig.suptitle("Original vs Modified Sliding Law on "+basename+" Cubesat Rate Error")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])
            axs[r].set_ylabel(sp_labels[r])
            ylim = axs[r].get_ylim()
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r == 0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("axes_av_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #AV norm
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        use_data = [matrix_row_norm((180/np.pi)*(np.array([d.state_hist[j,0:3]-d.goal_hist[j].state[0:3] for j in range(len(d.state_hist))]))) for d in data_1]
        # ax.set_title("Sliding vs Twisting "+extra_name_title+basename+" Satellite Rate Error ")
        ax.set_title("Original vs Modified Sliding Law on "+basename+" Cubesat Rate Error")

        for i in range(len(names_1)):
            ax.plot(data_1[i].t_hist,use_data[i],label = legends_1[i])
        # ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.set_yscale('log')
        ylim = ax.get_ylim()
        ax.vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
        ax.annotate("Control Activated", xy=(200, ylim[1]),
            xytext=(5,-5), textcoords="offset points",
            verticalalignment="top",
            weight="normal",
            bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_ylabel("Rate Error Norm (deg/s)")
        plt.savefig("log_norm_av_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)


        #Ctrl by axis
        sp_labels = ["x axis (Am^2)","y axis (Am^2)","z axis (Am^2)"]
        # ylim = [-0.05,0.05]
        axs = [0,0,0]
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        axs = [ax1,ax2,ax3]
        use_data = [(np.array([data_1[i].control_hist[j,0:3]+data_1[i].state_hist[j,7:10] for j in range(len(data_1[i].state_hist))])) for i in range(len(data_1))]
        fig.suptitle("Original vs Modified Sliding Law on "+basename+" Cubesat Control Effort")
        # fig.suptitle("Sliding vs Twisting "+extra_name_title+basename+" Satellite Control Effort")# (Bias Removed)")
        for r in range(3):
            # if r>0:
            #     axs[r] = plt.subplot(4,1,r+1,sharex = axs[0])
            # else:
            #     axs[r] = plt.subplot(4,1,r+1)
            for i in range(len(names_1)):
                axs[r].plot(data_1[i].t_hist,use_data[i][:,r])

            ylim = axs[r].get_ylim()
            axs[r].set_ylabel(sp_labels[r])
            axs[r].vlines(200,ylim[0],ylim[1],linestyles='dashed',label = "Control Activated",color = 'k')
            if r==0:
                axs[r].annotate("Control Activated", xy=(200, ylim[1]),
                    xytext=(5,-5), textcoords="offset points",
                    verticalalignment="top",
                    weight="normal",
                    bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
            axs[r].set_ylim(ylim[0],ylim[1])

        axs[r].set_xlabel("Time (s)")
        axs[r].legend(legends_1,loc='right')
        plt.savefig("ctrl_"+lc_basename+extra_name+"_wis_twist_comp.png")
        plt.draw()
        plt.pause(1)

        plt.pause(3)
        plt.close('all')
