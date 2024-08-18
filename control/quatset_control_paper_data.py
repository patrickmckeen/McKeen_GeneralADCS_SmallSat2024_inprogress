#estimation results for paper
from sat_ADCS_estimation import *
from sat_ADCS_control import *
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
from squaternion import Quaternion
import traceback

class metrics:
    def __init__(self):
        pass


class simsave:
    def __init__(self):
        pass

def pickling_test(ttt):
    dirlist = dir(ttt)
    print(dirlist)
    for j in dirlist:
        print(j)
        att = getattr(ttt,j)
        try:
            with open("test", "wb") as fp:   #Pickling
                pickle.dump(att, fp)
            print("OK ",j)
        except Exception as e:
            print("      BREAKS  ",j)
            print("     ")
            print(e)
            print("     ")


# analysis functions
def find_metrics(tlist,anglist,deg_threshold = 1):
    #find first and last time it falls under threshold
    # log_angdiff = np.log(angdiff[int(0.1*ind):int(0.9*ind)])
    # tc = np.polyfit(log_angdiff,range(int(0.9*ind)-int(0.1*ind)),1)[0]
    res = metrics()
    less_than_inds = np.where(anglist<deg_threshold)[0]
    greater_than_inds = np.where(anglist>=deg_threshold)[0]
    if len(less_than_inds)>0 and len(greater_than_inds)>0:
        converged_ind1 = less_than_inds[0]
        if greater_than_inds[-1]+1<len(anglist):
            converged_ind2 = greater_than_inds[-1]+1

            res.ind_conv = (converged_ind1,converged_ind2,int(0.5*(converged_ind1+converged_ind2)))
            res.time_to_conv = (tlist[converged_ind1],tlist[converged_ind2],tlist[int(0.5*(converged_ind1+converged_ind2))])
            log_angdiff = np.log(anglist[int(0.1*res.ind_conv[2]):int(0.9*res.ind_conv[2])])
            log_t_list = tlist[int(0.1*res.ind_conv[2]):int(0.9*res.ind_conv[2])]
            if log_t_list.size >0:
                res.tc_est = np.nan
                try:
                    res.tc_est = np.polyfit(log_angdiff,log_t_list,1)[0]
                except:
                    pass
            else:
                res.tc_est = np.nan
            res.steady_state_err_mean = tuple([np.mean(anglist[j:]) for j in res.ind_conv])
            res.steady_state_err_max = tuple([np.max(anglist[j:]) for j in res.ind_conv])
        else:
            converged_ind2 = np.nan

            res.ind_conv = (converged_ind1,np.nan,np.nan)
            res.time_to_conv = (tlist[converged_ind1],np.nan,np.nan)
            log_angdiff = np.log(anglist[int(0.1*res.ind_conv[0]):int(0.9*res.ind_conv[0])])
            log_t_list = tlist[int(0.1*res.ind_conv[0]):int(0.9*res.ind_conv[0])]
            res.tc_est = np.nan
            res.steady_state_err_mean = np.nan
            res.steady_state_err_max = np.nan
            try:
                res.tc_est = np.polyfit(log_angdiff,log_t_list,1)[0]
                res.steady_state_err_mean = (np.mean(anglist[res.ind_conv[0]:]) ,np.nan,np.nan)
                res.steady_state_err_max =  (np.max(anglist[res.ind_conv[0]:]) ,np.nan,np.nan)
            except:
                pass
    else:
        res.ind_conv = (np.nan,np.nan,np.nan)
        res.time_to_conv = (np.nan,np.nan,np.nan)
        res.tc_est = np.nan
        res.steady_state_err_mean = np.nan
        res.steady_state_err_max = np.nan
    return res

def make_lovera_orb(dt):
    inc = 87*math.pi/180.0
    alt = 450
    dur = 60*60*12
    v = math.sqrt(mu_e/(alt+R_e))
    os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*(alt+R_e),v*np.array([0,math.cos(inc),math.sin(inc)]))
    lovera_orb = Orbit(os0,end_time = 0.22+1.05*dur*sec2cent,dt = dt,use_J2 = True)
    with open("lovera_orb_"+str(dt), "wb") as fp:   #Pickling
        pickle.dump(lovera_orb, fp)
    return lovera_orb

def make_wie_orb(dt):
    #based on Hubble orbit
    inc = 28.5*math.pi/180.0
    alt = 540
    dur = 60*60*12
    v = math.sqrt(mu_e/(alt+R_e))
    os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*(alt+R_e),v*np.array([0,math.cos(inc),math.sin(inc)]))
    wie_orb = Orbit(os0,end_time = 0.22+1.05*dur*sec2cent,dt = dt,use_J2 = True)
    with open("wie_orb_"+str(dt), "wb") as fp:   #Pickling
        pickle.dump(wie_orb, fp)
    return wie_orb

def make_wisniewski_orb(dt):
    dur = 60*60*12
    os0 = Orbital_State(0.22-1*sec2cent,np.array([-1884.2,6935,-4.8]),np.array([0.77,0.21,7.51]))
    wie_orb = Orbit(os0,end_time = 0.22+1.05*dur*sec2cent,dt = dt,use_J2 = True)
    with open("wisniewski_orb_"+str(dt), "wb") as fp:   #Pickling
        pickle.dump(wie_orb, fp)
    return wie_orb


def run_sim(orb,state0,est,real_sat,control_laws,goals,j2000_0 = 0.22,tf=60*60*3,dt = 1,alt_title = None,rand = False):
    if not rand:
        np.random.seed(1)

    #
    #real_sat

    with open(orb, "rb") as fp:   #unPickling
        orb = pickle.load(fp)

    if (orb.times[-1]-j2000_0)*cent2sec < 1.01*tf:
        raise ValueError("orbit too short for this.")

    est_sat = est.sat
    t0 = 0
    q0 = state0[3:7]
    w0 = state0[0:3]

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),real_sat.state_len+sum([j.has_bias for j in real_sat.actuators])+sum([j.has_bias for j in real_sat.sensors])+sum([j.time_varying*j.main_param.size for j in real_sat.disturbances if j.time_varying])))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))

    goal_hist =  []
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    prev_control = control.copy()
    state = state0
    orbt = orb.get_os(j2000_0+(t-t0)*sec2cent)
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))

    est_sat0 = copy.deepcopy(est_sat)

    while t<tf:

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        prev_est_state = est.use_state.val.copy()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        print(t)
        # autovar = np.sqrt(np.diagonal(est.use_state.cov))
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        # print(autovar[3:6])
        # print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi,4*norm(np.arctan(autovar[3:6]/2.0))*180.0/math.pi)
        # wa,va = np.linalg.eigh(est.use_state.cov[3:6,3:6])
        # print(va)
        # print(wa)
        # breakpoint()
        print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)


        mode = goals.get_control_mode(orbt.J2000)
        ctrlr = [j for j in control_laws if j.modename==mode][0]
        goal_state = goals.get_pointing_info(orbt)
        # breakpoint()next_goal_stat
        nextorb = orb.get_os(j2000_0+(dt+t-t0)*sec2cent)
        next_goal_state = goals.get_pointing_info(nextorb)

        prev_control = control.copy()
        control = ctrlr.find_actuation(est_state,orbt,nextorb,goal_state,[],next_goal_state,sens,[],False)
        print('quat',state[3:7])#,norm(q_err[1:]),norm(np.cross(q_err[1:],est.os_vecs['b']))/norm(est.os_vecs['b']))
        print(goal_state.eci_vec,goal_state.body_vec)
        print('goalquat',goal_state.state[3:7])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(goal_state.state[3:7],state[3:7]),-1,1)**2.0 ))
        print(goal_state.body_vec@rot_mat(goal_state.state[3:7]).T)
        print(goal_state.eci_vec)
        print((180/np.pi)*math.acos(np.dot(goal_state.eci_vec,goal_state.body_vec@rot_mat(state[3:7]).T )))
        print('ctrl',control)
        print('av',state[0:3]*180.0/math.pi,norm(state[0:3])*180.0/math.pi)#,(180.0/math.pi)*norm(np.cross(state[0:3],est.os_vecs['b']))/norm(est.os_vecs['b']))
        # print(goal_state.state)
        # print(orbt.in_eclipse())

        print('dist ',real_sat.last_dist_torq,norm(real_sat.last_dist_torq))
        disttorqest = est.sat.dist_torque(est_state[:est.sat.state_len],est.os_vecs).copy()
        print('dist est ',disttorqest,norm(disttorqest))
        print(real_sat.last_dist_torq - disttorqest,norm(real_sat.last_dist_torq - disttorqest))
        # print(est_state[-3:])
        # print(real_sat.disturbances[-2].main_param)
        # if np.any([isinstance(j,General_Disturbance) for j in est_sat.disturbances]):
        #     print('dist gen dist est ',est_state[-3:],norm(est_state[-3:]))
        #     print(real_sat.last_dist_torq - est_state[-3:],norm(real_sat.last_dist_torq - est_state[-3:]))
        # print('act ',real_sat.last_act_torq)
        # print("B real", real_vecs["b"])
        # print("B est", Bbody)
        # print("B should meas",real_vecs["b"]*real_sat.sensors[0].scale + np.hstack([j.bias for j in real_sat.sensors[0:3]]))
        # print("B meas", sens[0:3])
        # print("B est meas", Bbody*est_sat.sensors[0].scale + est_state[10:13])
        # print("rot ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(Bbody,normalize(quat_mult(quat_inv(q),state[3:7])[1:])))/norm(Bbody)))
        # print(va[:,-1],np.sqrt(wa[-1]))
        # print("cov ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(Bbody,va[:,-1]))/norm(Bbody)))

        # print(real_sat.sensors[0].bias[0],est.sat.sensors[0].bias[0]-real_sat.sensors[0].bias[0])
        # print([j.bias for j in real_sat.actuators if j.has_bias])
        # gb_real = np.concatenate([j.bias for j in real_sat.sensors if isinstance(j,Gyro)])
        # gb_est = np.concatenate([j.bias for j in est.sat.sensors if isinstance(j,Gyro)])
        # print(gb_real,norm(gb_real))
        # print(gb_est,norm(gb_est))
        # print(gb_est-gb_real,norm(gb_est-gb_real))
        # print(est_state[7:10],norm(est_state[7:10]))
        # print(np.concatenate([j.bias for j in real_sat.sensors[3:6] if j.has_bias])-est_state[7:10],norm(np.concatenate([j.bias for j in real_sat.sensors[3:6] if j.has_bias])-est_state[7:10]))
        # print(np.concatenate([j.bias for j in real_sat.sensors[6:] if j.has_bias])-est_state[10:13],norm(np.concatenate([j.bias for j in real_sat.sensors[6:] if j.has_bias])-est_state[10:13]),norm(np.concatenate([j.bias for j in real_sat.sensors[6:] if j.has_bias])))
        # print([j.bias for j in real_sat.sensors[3:6] if j.has_bias])
        # print(est.use_state.val[7:10])

        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        goal_hist += [goal_state.copy()]
        # breakpoint()
        orb_hist += [orbt]
        control_hist[ind,:] = prev_control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)
        dist_torq_hist[ind,:] = real_sat.last_dist_torq.copy()
        est_dist_torq_hist[ind,:] = est.sat.dist_torque(prev_est_state[:est.sat.state_len],est.prev_os_vecs).copy()
        act_torq_hist[ind,:] = real_sat.last_act_torq.copy()
        est_act_torq_hist[ind,:] =  est.sat.act_torque(prev_est_state[:est.sat.state_len],prev_control,est.prev_os_vecs,False)
        sens_hist[ind,:] = sens.copy()
        eclipse_hist[ind] = orbt.in_eclipse()
        # if ind>1:
        #     print(np.abs(dist_torq_hist[ind,:]-dist_torq_hist[ind-1,:]),norm(dist_torq_hist[ind,:]-dist_torq_hist[ind-1,:]))

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(j2000_0+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-2, atol=1e-6)#,jac = ivp_jac)
        real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        est_sat.dynamics(est.use_state.val[:7],control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)

    # breakpoint()
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    ctrlangdiff = np.stack([(180/np.pi)*np.arccos(-1 + 2*np.clip(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7]),-1,1)**2.0)  for j in range(state_hist.shape[0])])
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7])*np.sign(np.dot(est_state_hist[j,3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    ctrlquatdiff = np.stack([quat_mult(quat_inv(goal_hist[j].state[3:7]),state_hist[j,3:7])*np.sign(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])

    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    ctrlmrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_mrp(ctrlquatdiff[j,:]) for j in range(quatdiff.shape[0])]))
    goalav_ECI = np.stack([goal_hist[j].state[0:3]@rot_mat(goal_hist[j].state[3:7]).T*180.0/np.pi for j in range(len(goal_hist))])
    goalav_body = np.stack([goalav_ECI[j,:]@rot_mat(state_hist[j,3:7]) for j in range(len(goal_hist))])

    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi

    point_vec_body = np.stack([j.body_vec for j in goal_hist])
    goal_vec_eci = np.stack([j.eci_vec for j in goal_hist])

    # sens_est_biases = np.array([j.has_bias*j.output_len for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],SunSensorPair) and real_sat.attitude_sensors[j].has_bias ])
    # sens_biases = np.array([j.has_bias*j.output_len for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],SunSensorPair) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias])

    est_sb_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],SunSensorPair) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias]) + est.sat.state_len+est.sat.act_bias_len).astype('int')
    est_mb_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],MTM) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias]) + est.sat.state_len+est.sat.act_bias_len).astype('int')
    est_gb_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],Gyro) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias]) + est.sat.state_len+est.sat.act_bias_len).astype('int')
    est_ab_inds = (np.array([np.sum([k.input_len for k in est.sat.actuators[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.actuators)) if est.sat.actuators[j].has_bias and est.sat.actuators[j].estimated_bias]) + est.sat.state_len).astype('int')
    try:
        est_dipole_inds = (np.concatenate([np.sum([k.main_param.size for k in est.sat.disturbances[:j] if k.estimated_param])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(est.sat.disturbances)) if est.sat.disturbances[j].estimated_param and isinstance(est.sat.disturbances[j],Dipole_Disturbance)]) + est.sat.state_len+est.sat.act_bias_len+est.sat.att_sens_bias_len).astype('int')
    except:
        est_dipole_inds = np.array([])
    try:
        est_gendist_inds = (np.concatenate([np.sum([k.main_param.size for k in est.sat.disturbances[:j] if k.estimated_param])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(est.sat.disturbances)) if est.sat.disturbances[j].estimated_param and isinstance(est.sat.disturbances[j],General_Disturbance)]) + est.sat.state_len+est.sat.act_bias_len+est.sat.att_sens_bias_len).astype('int')
    except:
        est_gendist_inds = np.array([])

    real_sb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],SunSensorPair) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_mb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],MTM) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_gb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],Gyro) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_ab_inds = (np.array([np.sum([k.input_len for k in real_sat.actuators[:j] if k.has_bias]) for j in range(len(real_sat.actuators)) if real_sat.actuators[j].has_bias]) + real_sat.state_len).astype('int')
    try:
        real_dipole_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(real_sat.disturbances)) if  isinstance(real_sat.disturbances[j],Dipole_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_length for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_dipole_inds = np.array([])
    try:
        real_gendist_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(real_sat.disturbances)) if isinstance(real_sat.disturbances[j],General_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_length for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_gendist_inds = np.array([])


    if alt_title is None:
        base_title = "baseline_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        base_title = alt_title+"_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("quatset_control_paper_test_files/"+base_title)
    folder_name = "quatset_control_paper_test_files"

    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/quat_plot")
    plot_the_thing(np.stack([j.state[3:7] for j in goal_hist]),title = "Goal Quat",xlabel='Time (s)',ylabel = 'Goal Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalquat_plot")
    plot_the_thing(np.stack([j.state[0:3]*180.0/np.pi for j in goal_hist]),title = "Goal AV (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalav_plot")
    plot_the_thing(goalav_body,title = "Goal AV (body frame) (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (body frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalav_body_plot")
    plot_the_thing(goalav_ECI,title = "Goal AV (ECI frame) (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (ECI frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalav_ECI_plot")
    plot_the_thing(goalav_body-state_hist[:,0:3]*180.0/math.pi,title = "Goal AV err (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV err (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalaverr_body_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ang_plot")
    plot_the_thing(ctrlangdiff,title = "Ctrl Angular Error",xlabel='Time (s)',ylabel = 'Ctrl Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlang_plot")

    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mrp_plot")
    plot_the_thing(ctrlmrpdiff,title = "Ctrl MRP Error",xlabel='Time (s)',ylabel = 'Ctrl MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlmrp_plot")
    plot_the_thing(ctrlquatdiff,title = "Ctrl Quat Error",xlabel='Time (s)',ylabel = 'Ctrl Quat Error',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlquat_plot")

    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_av_plot")

    plot_the_thing(est_dist_torq_hist,title = 'Estimated Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/est_dist_torq_plot")
    plot_the_thing(dist_torq_hist,title = 'Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/dist_torq_plot")
    plot_the_thing(est_dist_torq_hist-dist_torq_hist,title = 'Dist Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'Dist Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/dist_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_dist_torq_hist-dist_torq_hist)),title = 'Log Dist Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log Dist Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/_log_dist_torq_err_plot")

    plot_the_thing(est_act_torq_hist,title = 'Estimated act Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/est_act_torq_plot")
    plot_the_thing(act_torq_hist,title = 'act Torq',xlabel = 'Time (s)',norm = True, ylabel = 'act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/act_torq_plot")
    plot_the_thing(est_act_torq_hist-act_torq_hist,title = 'act Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'act Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/act_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_act_torq_hist-act_torq_hist)),title = 'Log act Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log act Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/_log_act_torq_err_plot")

    plot_the_thing(est_dist_torq_hist+est_act_torq_hist,title = 'Estimated Combo Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Combo Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/est_combo_torq_plot")
    plot_the_thing(dist_torq_hist+act_torq_hist,title = 'Combo Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Combo Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/combo_torq_plot")
    plot_the_thing(est_dist_torq_hist+est_act_torq_hist-dist_torq_hist-act_torq_hist,title = 'Combo Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'Combo Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/combo_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_dist_torq_hist+est_act_torq_hist-dist_torq_hist-act_torq_hist)),title = 'Log Combo Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log Combo Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/_log_combo_torq_err_plot")

    if est_ab_inds.size > 0:
        abest = est_state_hist[:,est_ab_inds]
    else:
        abest = np.zeros((est_state_hist.shape[0],3))
    if real_ab_inds.size > 0:
        abreal = state_hist[:,real_ab_inds]
    else:
        abreal = np.zeros((est_state_hist.shape[0],3))
    abdiff = abest-abreal
    plot_the_thing(abest,title = "Est Actuator Bias",xlabel='Time (s)',norm = True,ylabel = 'Actuator Bias ([units])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_abias_plot")
    plot_the_thing(abreal,title = "Actuator Bias",xlabel='Time (s)',norm = True,ylabel = 'Actuator Bias ([units])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/abias_plot")
    plot_the_thing(abdiff,title = "Actuator Bias Error",xlabel='Time (s)',ylabel = 'Actuator Bias Error ([units])', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/aberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(abdiff)),title = "Log Actuator Bias Error",xlabel='Time (s)',ylabel = 'Log Actuator Bias Error (log [units]])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logab_plot")


    if est_gb_inds.size > 0:
        gbest = est_state_hist[:,est_gb_inds]*180.0/math.pi
    else:
        gbest = np.zeros((est_state_hist.shape[0],3))
    if real_gb_inds.size > 0:
        gbreal = state_hist[:,real_gb_inds]*180.0/math.pi
    else:
        gbreal = np.zeros((est_state_hist.shape[0],3))
    gbdiff = gbest-gbreal
    plot_the_thing(gbest,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_gbias_plot")
    plot_the_thing(gbreal,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/gbias_plot")
    plot_the_thing(gbdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/gberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(gbdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_loggb_plot")

    if est_mb_inds.size > 0:
        mbest = est_state_hist[:,est_mb_inds]#*180.0/math.pi
    else:
        mbest = np.zeros((est_state_hist.shape[0],3))
    if real_mb_inds.size > 0:
        mbreal = state_hist[:,real_mb_inds]#*180.0/math.pi
    else:
        mbreal = np.zeros((est_state_hist.shape[0],3))
    mbdiff = mbest-mbreal
    plot_the_thing(mbest,title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_mbias_plot")
    plot_the_thing(mbreal,title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mbias_plot")
    plot_the_thing(mbdiff,title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(mbdiff)),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logmb_plot")

    if est_sb_inds.size > 0:
        sbest = est_state_hist[:,est_sb_inds]#*180.0/math.pi
    else:
        sbest = np.zeros((est_state_hist.shape[0],3))
    if real_sb_inds.size > 0:
        sbreal = state_hist[:,real_sb_inds]#*180.0/math.pi
    else:
        sbreal = np.zeros((est_state_hist.shape[0],3))
    sbdiff = sbest-sbreal
    plot_the_thing(sbest,title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_sbias_plot")
    plot_the_thing(sbreal,title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/sbias_plot")
    plot_the_thing(sbdiff,title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/sberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(sbdiff)),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logsb_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrl_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logav_plot")
    plot_the_thing(np.log10(ctrlangdiff),title = "Log Ctrl Angular Error",xlabel='Time (s)',ylabel = 'Ctrl Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logctrlang_plot")
    if est_dipole_inds.size > 0:
        dpest = est_state_hist[:,est_dipole_inds]
    else:
        dpest = np.zeros((est_state_hist.shape[0],3))
    if real_dipole_inds.size > 0:
        dpreal = state_hist[:,real_dipole_inds]
    else:
        dpreal = np.zeros((est_state_hist.shape[0],3))
    dpdiff = dpest-dpreal
    plot_the_thing(dpest,title = "Est Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dipole (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_dipole_plot")
    plot_the_thing(dpreal,title = "Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dipole (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/dipole_plot")
    plot_the_thing(dpdiff,title = "Dipole Error",xlabel='Time (s)',ylabel = 'Dipole Error (Am^2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/dipole_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(dpdiff)),title = "Log Dipole Error",xlabel='Time (s)',ylabel = 'Log Dipole Error (log (Am^2))',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_log_dipole_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrl_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logav_plot")
    plot_the_thing(np.log10(ctrlangdiff),title = "Log Ctrl Angular Error",xlabel='Time (s)',ylabel = 'Ctrl Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logctrlang_plot")

    plot_the_thing(np.stack([orb_hist[j].B@rot_mat(state_hist[j,3:7]) for j in range(len(orb_hist))]),title = "B Body frame",xlabel='Time (s)',norm = True,ylabel = 'B Body Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bbody_plot")
    plot_the_thing(np.stack([orb_hist[j].B for j in range(len(orb_hist))]),title = "B ECI frame",xlabel='Time (s)',norm = True,ylabel = 'B ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bECI_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(state_hist[j,3:7])),normalize(ctrlquatdiff[j,1:])))) for j in range(len(orb_hist))]),title = "Ang between B body and angular ctrl error",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/b_q_ctrlang_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(state_hist[j,3:7])),normalize(state_hist[j,:3])))) for j in range(len(orb_hist))]),title = "Ang between B body and av ctrl error",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/b_w_ctrlang_plot")

    Rmats = np.dstack([rot_mat(state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    orbitRmats = np.dstack([rot_mat(two_vec_to_quat(-orb_hist[j].R,orb_hist[j].V,unitvecs[0],unitvecs[1])) for j in range(len(orb_hist))]) #stacked matrices such that if you take R=[:,;,i], R@unitvecs[0] would give the nadir direction coordinates in ECI, R@unitvecs[0] is ram direction
    plot_the_thing(Rmats[0,:,:],title = "Body x-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bxECI_plot")
    plot_the_thing(Rmats[1,:,:],title = "Body y-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/byECI_plot")
    plot_the_thing(Rmats[2,:,:],title = "Body z-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bzECI_plot")
    plot_the_thing(orbitRmats[0,:,:],title = "Orbit Nadir in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/oxECI_plot")
    plot_the_thing(orbitRmats[1,:,:],title = "Orbit Ram in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/oyECI_plot")
    plot_the_thing(orbitRmats[2,:,:],title = "Orbit anti-normal in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ozECI_plot")
    plot_the_thing(np.stack([orbitRmats[:,:,j].T@Rmats[0,:,j] for j in range(Rmats.shape[2])]),title = "Body x-axis in x=nadir,ram=y frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bxo_plot")
    plot_the_thing(np.stack([orbitRmats[:,:,j].T@Rmats[1,:,j] for j in range(Rmats.shape[2])]),title = "Body y-axis in x=nadir,ram=y frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/byo_plot")
    plot_the_thing(np.stack([orbitRmats[:,:,j].T@Rmats[2,:,j] for j in range(Rmats.shape[2])]),title = "Body z-axis in x=nadir,ram=y frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bzo_plot")


    point_vec_eci = np.stack([point_vec_body[j,:]@Rmats[:,:,j].T for j in range(point_vec_body.shape[0])])
    goal_vec_body = np.stack([goal_vec_eci[j,:]@Rmats[:,:,j] for j in range(goal_vec_eci.shape[0])])
    pt_err = np.arccos(np.array([np.dot(point_vec_eci[j,:],goal_vec_eci[j,:]) for j in range(point_vec_body.shape[0])]))*180.0/np.pi
    plot_the_thing(point_vec_body,title = "Body Pointing Vector",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/point_vec_body_plot")
    plot_the_thing(point_vec_eci,title = "Body Pointing Vector in ECI",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/point_vec_eci_plot")
    plot_the_thing(goal_vec_body,title = "Goal Pointing Vector in Body",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goal_vec_body_plot")
    plot_the_thing(goal_vec_eci,title = "ECI Goal Pointing Vector",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goal_vec_eci_plot")
    plot_the_thing(pt_err,title = "Angular Error between Pointing and Goal Vectors",xlabel='Time (s)',norm = False,ylabel = 'Angle (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/vecang")
    plot_the_thing(np.log10(pt_err),title = "Log10 Angular Error between Pointing and Goal Vectors",xlabel='Time (s)',norm = False,ylabel = 'Angle (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logvecang")


    #generate statistics
    metrics = find_metrics(t_hist,angdiff)
    ctrlmetrics = find_metrics(t_hist,ctrlangdiff)
    labels = ["title","al","kap","bet","dt","werrcov","mrpcov","mtmscale"]+["covest0","intcov"]+["ctrl conv time","ctrl tc","last 100 ctrlang err mean","last 100 ctrlang err max"]
    info = [base_title,est.al,est.kap,est.bet,dt,werrcov,mrperrcov,1]+[np.diag(cov_estimate.copy()),int_cov.copy()]+[ctrlmetrics.time_to_conv,ctrlmetrics.tc_est,np.mean(ctrlangdiff[-100:]),np.amax(ctrlangdiff[-100:])]
    with open("quatset_control_paper_test_files/"+base_title+"/info", 'w') as f:
        for j in range(len(labels)):
            f.write(labels[j])
            f.write(": ")
            f.write(str(info[j]))
            f.write("\n")
    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
    sim.goal_hist = goal_hist
    sim.orb_hist = orb_hist
    sim.cov_hist = cov_hist
    sim.kap = est.kap
    sim.al = est.al
    sim.cov_est0 = cov_estimate.copy()
    sim.int_cov = int_cov.copy()
    sim.t_hist = t_hist
    # sim.real_sat = real_sat
    # sim.est_sat = est_sat
    # sim.est_sat0 = est_sat0
    # sim.est = est
    sim.metrics = metrics
    sim.ctrlmetrics = ctrlmetrics
    sim.comp_time = comp_time_hist
    sim.dist_torq_hist = dist_torq_hist
    sim.est_dist_torq_hist = est_dist_torq_hist
    sim.act_torq_hist = act_torq_hist
    sim.est_act_torq_hist = est_act_torq_hist
    sim.sens_hist = sens_hist
    sim.eclipse_hist = eclipse_hist
    # try:
    with open("quatset_control_paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)




#paper comparisons.  -- sensor biases and non, actuator biases and non, disturbances and non. Specific prop disturbance case

if __name__ == '__main__':
    # worb = make_wie_orb(10)
    # worb1 = make_wie_orb(1)
    # oorb = make_wisniewski_orb(10)
    # oorb1 = make_wisniewski_orb(1)
    with open("quatset_control_paper_test_files/new_tests_marker"+time.strftime("%Y%m%d-%H%M%S")+".txt", 'w') as f:
        f.write("just a file to show when new runs of tests started.")

    np.random.seed(1)
    avcov = ((0.5*math.pi/180)/(60))**2.0
    angcov = (0.5*math.pi/180)**2.0
    werrcov = 1e-17
    mrperrcov = 1e-12


    wie_q0 = zeroquat#normalize(np.array([0.153,0.685,0.695,0.153]))
    wie_w0 = np.array([0.01,0.01,0.001])#np.array([0.53,0.53,0.053])#/(180.0/math.pi)

    wie_base_sat = create_Wie_sat(real=True,rand=False,include_magicbias = False,include_magic_noise = False)
    wie_base_est_sat = create_Wie_sat(  real = False, rand=False,include_magicbias = False,estimate_magic_bias = False,include_magic_noise = False)
    wie_sat = create_Wie_sat(real=True,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,include_magicbias = False,include_magic_noise = False)
    wie_est_sat = create_Wie_sat(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,estimate_dipole = False,include_magicbias = False,estimate_magic_bias = False,include_magic_noise = False)

    wie_orb_file = "wie_orb_1"
    wie_goals_base = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wie_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wie_goals_ang =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD_QUATSET_ANG},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})
    # wie_goals_ang =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD_QUATSET_ANG},{0.2:PointingGoalVectorMode.ZENITH},{0.2:unitvecs[2]})
    # wie_goals_B =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD_QUATSET_B},{0.2:PointingGoalVectorMode.ZENITH},{0.2:unitvecs[2]})
    wie_goals_Lyap =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD_QUATSET_LYAP},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})
    wie_goals_LyapR =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD_QUATSET_LYAPR},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})

    wie_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)
    wie_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*2*2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(3**2.0))
    wie_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)#,np.eye(3)*(10**2.0))

    lovera_q0 = normalize(random_n_unit_vec(4))
    lovera_w0 = np.array([2,2,-2])/(180.0/math.pi)
    lovera_w0_slow = np.array([0.01,0.01,-0.01])

    lovera_base_sat = create_Lovera_sat(real=True,rand=False,mtq_max = 20.0*np.ones(3),include_mtq_noise = False)
    lovera_base_est_sat = create_Lovera_sat(  real = False, rand=False,mtq_max = 20.0*np.ones(3),include_mtq_noise = False)
    lovera_sat = create_Lovera_sat(real=True,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,include_mtqbias = False,mtq_max = 20.0*np.ones(3),include_mtq_noise = False)
    lovera_est_sat = create_Lovera_sat(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,estimate_dipole = False,include_mtqbias = False,estimate_mtq_bias = False,mtq_max = 20.0*np.ones(3),include_mtq_noise = False)

    lovera_orb_file = "lovera_orb_10"
    lovera_orb_file1 = "lovera_orb_1"
    lovera_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    lovera_goals_ang =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD_QUATSET_ANG},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})
    lovera_goals_ang =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD_QUATSET_ANG},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})

    lovera_goals_B =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD_QUATSET_B},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})
    lovera_goals_Lyap =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD_QUATSET_LYAP},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})
    lovera_goals_LyapR =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD_QUATSET_LYAPR},{0.2:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[2])},{0.2:unitvecs[2]})
    lovera_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)

    lovera_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*(1.0*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)#,np.eye(3)*(1**2.0))
    lovera_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(1.0*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)#,np.eye(3)*(1**2.0))

    wisniewski_orb_file = "wisniewski_orb_10"
    wisniewski_orb_file1 = "wisniewski_orb_1"
    wisniewski_q0 = normalize(np.array([0,0,0,1]))
    wisniewski_w0 = np.array([-0.002,0.002,0.002])

    with open(wisniewski_orb_file, "rb") as fp:   #unPickling
        orb = pickle.load(fp)
    orbst = orb.get_os(0.22)
    offset_quat =  quat_inv(np.array([x for x in Quaternion.from_euler(60, 100, -100, degrees=True)]))#mrp_to_quat(2*np.tan(np.array([60,100,-100])*(math.pi/180.0)/4))
    # offset_quat =  np.array([x for x in Quaternion.from_euler(100, 60, -100, degrees=True)])
    # offset_quat =  np.array([x for x in Quaternion.from_euler(60, 100, -100, degrees=True)])
    # print(offset_quat)
    wisniewski_q0 = quat_mult(two_vec_to_quat(orbst.R/norm(orbst.R),normalize(np.cross(orbst.R/norm(orbst.R),orbst.V/norm(orbst.V))),unitvecs[2],unitvecs[0]),offset_quat)

    # print(0.5*Wmat(offset_quat)@(wisniewski_w0-rot_mat(offset_quat).T@np.array([0.06*math.pi/180,0,0])))
    # print(np.array([x for x in Quaternion.from_euler(60, 100, -100, degrees=True)]))
    # print(np.array([x for x in Quaternion.from_euler(100, 60, -100, degrees=True)]))
    # print(np.array([x for x in Quaternion.from_euler(60, -100, 100, degrees=True)]))
    # print(np.array([x for x in Quaternion.from_euler(-100, 60, 100, degrees=True)]))
    # print(np.array([x for x in Quaternion.from_euler(100, -100, 60, degrees=True)]))
    # print(np.array([x for x in Quaternion.from_euler(-100, 100, 60, degrees=True)]))
    # print(mrp_to_quat(2*np.tan(np.array([60,100,-100])*(math.pi/180.0)/4)))
    # print(mrp_to_quat(2*np.tan(np.array([100,60,-100])*(math.pi/180.0)/4)))


    wisniewski_base_sat = create_Wisniewski_stowed_sat(real=True,rand=False,include_mtqbias = False,include_mtq_noise = False)
    wisniewski_base_est_sat = create_Wisniewski_stowed_sat(  real = False, rand=False,include_mtqbias = False,include_mtq_noise = False,estimate_mtq_bias = False)
    wisniewski_sat = create_Wisniewski_stowed_sat(real=True,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,include_mtqbias = False,include_mtq_noise = False)#,care_about_eclipse = False,gyro_std = np.ones(3)*(0.000025*math.pi/180.0),gyro_bsr = np.ones(3)*(0.000025*math.pi/180.0))#,include_mtq_noise = True,mtq_std = 0.001*np.ones(3))
    wisniewski_est_sat = create_Wisniewski_stowed_sat(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,estimate_dipole = False,include_mtqbias = False,estimate_mtq_bias = False,include_mtq_noise = False)

    wisniewski_goals =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wisniewski_goals_ang =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING_QUATSET_ANG},{0.2:PointingGoalVectorMode.ZENITH},{0.2:unitvecs[2]})
    wisniewski_goals_B =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING_QUATSET_B},{0.2:PointingGoalVectorMode.ZENITH},{0.2:unitvecs[2]})
    wisniewski_goals_Lyap =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING_QUATSET_LYAP},{0.2:PointingGoalVectorMode.ZENITH},{0.2:unitvecs[2]})
    wisniewski_goals_LyapR =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING_QUATSET_LYAPR},{0.2:PointingGoalVectorMode.ZENITH},{0.2:unitvecs[2]})
    wisniewski_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)
    wisniewski_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)#,np.eye(3)*(5**2.0))
    wisniewski_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)#,np.eye(3)*(5**2.0))

    bc_q0 = normalize(np.array([0,0,0,1]))
    bc_w0 = (np.pi/180.0)*random_n_unit_vec(3)#

    bc_real = create_BC_sat(real=True,rand=False,care_about_eclipse = True,use_dipole = False)
    bc_est = create_BC_sat( real = False, rand=False,care_about_eclipse = True,use_dipole = False)

    bc_orb_file = "../estimation/myorb_1"
    cubesat_goals_lovera =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    cubesat_goals_wisniewski =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})

    bc_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(1e-7*1)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-3)**2.0)

    dt = 1

    wie = ["Wie_disturbed", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals,0.5*60*60,wie_orb_file,""]
    lovera = ["Lovera_disturbed", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals,10*60*60,    lovera_orb_file1,""]
    wisniewski = ["Wisniewski_disturbed", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals,10*60*60,wisniewski_orb_file1,""]

    wie_ang = ["Wie_ang", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_ang,0.5*60*60,wie_orb_file,"Ang"]
    lovera_ang = ["Lovera_ang", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_ang,10*60*60,    lovera_orb_file1,"Ang"]
    wisniewski_ang = ["Wisniewski_ang", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_ang,10*60*60,wisniewski_orb_file1,"Ang"]

    lovera_B = ["Lovera_B", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_B,10*60*60,    lovera_orb_file1,"B"]
    wisniewski_B = ["Wisniewski_B", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_B,10*60*60,wisniewski_orb_file1,"B"]

    wie_LyapV = ["Wie_LyapV", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_Lyap,0.5*60*60,wie_orb_file,"Lyap0"]
    lovera_LyapV = ["Lovera_LyapV", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_Lyap,10*60*60,    lovera_orb_file1,"Lyap0"]
    wisniewski_LyapV = ["Wisniewski_LyapV", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_Lyap,10*60*60,wisniewski_orb_file1,"Lyap0"]

    wie_LyapVdot = ["Wie_LyapVdot", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_Lyap,0.5*60*60,wie_orb_file,"Lyap1"]
    lovera_LyapVdot = ["Lovera_LyapVdot", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_Lyap,10*60*60,    lovera_orb_file1,"Lyap1"]
    wisniewski_LyapVdot = ["Wisniewski_LyapVdot", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_Lyap,10*60*60,wisniewski_orb_file1,"Lyap1"]

    wie_LyapNorm = ["Wie_LyapNorm", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_Lyap,0.5*60*60,wie_orb_file,"Lyap2"]
    lovera_LyapNorm = ["Lovera_LyapNorm", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_Lyap,10*60*60,    lovera_orb_file1,"Lyap2"]
    wisniewski_LyapNorm = ["Wisniewski_LyapNorm", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_Lyap,10*60*60,wisniewski_orb_file1,"Lyap2"]

    wie_LyapNorm2 = ["Wie_LyapNorm2", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_Lyap,0.5*60*60,wie_orb_file,"Lyap3"]
    lovera_LyapNorm2 = ["Lovera_LyapNorm2", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_Lyap,10*60*60,    lovera_orb_file1,"Lyap3"]
    wisniewski_LyapNorm2 = ["Wisniewski_LyapNorm2", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_Lyap,10*60*60,wisniewski_orb_file1,"Lyap3"]

    wie_LyapRmatV = ["Wie_LyapRmatV", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_LyapR,0.5*60*60,wie_orb_file,"LyapRmat0"]
    lovera_LyapRmatV = ["Lovera_LyapRmatV", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_LyapR,10*60*60,    lovera_orb_file1,"LyapRmat0"]
    wisniewski_LyapRmatV = ["Wisniewski_LyapRmatV", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_LyapR,10*60*60,wisniewski_orb_file1,"LyapRmat0"]

    wie_LyapRmatVdot = ["Wie_LyapRmatVdot", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_LyapR,0.5*60*60,wie_orb_file,"LyapRmat1"]
    lovera_LyapRmatVdot = ["Lovera_LyapRmatVdot", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_LyapR,10*60*60,    lovera_orb_file1,"LyapRmat1"]
    wisniewski_LyapRmatVdot = ["Wisniewski_LyapRmatVdot", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_LyapR,10*60*60,wisniewski_orb_file1,"LyapRmat1"]

    wie_LyapRmatNorm = ["Wie_LyapRmatNorm", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_LyapR,0.5*60*60,wie_orb_file,"LyapRmat2"]
    lovera_LyapRmatNorm = ["Lovera_LyapRmatNorm", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_LyapR,10*60*60,    lovera_orb_file1,"LyapRmat2"]
    wisniewski_LyapRmatNorm = ["Wisniewski_LyapRmatNorm", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_LyapR,10*60*60,wisniewski_orb_file1,"LyapRmat2"]

    wie_LyapRmatNorm2 = ["Wie_LyapRmatNorm2", wie_est_sat,wie_sat,1,wie_w0,wie_q0,wie_cov0_estimate,True,wie_goals_LyapR,0.5*60*60,wie_orb_file,"LyapRmat3"]
    lovera_LyapRmatNorm2 = ["Lovera_LyapRmatNorm2", lovera_est_sat,lovera_sat,1,lovera_w0,lovera_q0,lovera_cov0_estimate,True,lovera_goals_LyapR,10*60*60,    lovera_orb_file1,"LyapRmat3"]
    wisniewski_LyapRmatNorm2 = ["Wisniewski_LyapRmatNorm2", wisniewski_est_sat,wisniewski_sat,dt,wisniewski_w0,wisniewski_q0,wisniewski_cov0_estimate,True,wisniewski_goals_LyapR,10*60*60,wisniewski_orb_file1,"LyapRmat3"]

    tests_baseline = [wie,lovera,wisniewski]
    tests_ang = [wie_ang,lovera_ang,wisniewski_ang]
    tests_B = [lovera_B,wisniewski_B]
    tests_LyapV = [wie_LyapV,lovera_LyapV,wisniewski_LyapV]
    tests_LyapVdot = [wie_LyapVdot,lovera_LyapVdot,wisniewski_LyapVdot]
    tests_LyapNorm = [wie_LyapNorm,lovera_LyapNorm,wisniewski_LyapNorm]
    tests_LyapNorm2 = [wie_LyapNorm2,lovera_LyapNorm2,wisniewski_LyapNorm2]
    tests_LyapRmatV = [wie_LyapRmatV,lovera_LyapRmatV,wisniewski_LyapRmatV]
    tests_LyapRmatVdot = [wie_LyapRmatVdot,lovera_LyapRmatVdot,wisniewski_LyapRmatVdot]
    tests_LyapRmatNorm = [wie_LyapRmatNorm,lovera_LyapRmatNorm,wisniewski_LyapRmatNorm]
    tests_LyapRmatNorm2 = [wie_LyapRmatNorm2,lovera_LyapRmatNorm2,wisniewski_LyapRmatNorm2]
    # tests_func = [wie_match_test]
    # tests_baseline = [wie_match,lovera_match0,lovera_match,wisniewski_match0,wisniewski_match]
    # tests_disturbed = [wie,lovera0,lovera,wisniewski0,wisniewski]
    # tests_ctrl = [wie_w_control1,lovera_w_control10,lovera_w_control1,wisniewski_w_control10,wisniewski_w_control1]
    # tests_genctrl = [wie_w_gencontrol1,lovera_w_gencontrol10,lovera_w_gencontrol1,wisniewski_w_gencontrol10,wisniewski_w_gencontrol1]#,wisniewski_w_gencontrol_gg_in_gen10,wisniewski_w_gencontrol_gg_in_gen1]
    # tests_cubesat = [lovera_on_cubesat,wisniewski_on_cubesat,lovera_on_cubesat_gencontrol,wisniewski_on_cubesat_gencontrol]
    # gen_ctrl_tests = [wie_control_mini,wie_gencontrol_mini,lovera_control_mini,lovera_gencontrol_mini,wisniewski_control_mini,wisniewski_gencontrol_mini,wisniewski_genggcontrol_mini,wisniewski_simp]
    #
    # # tests = tests_baseline + tests_disturbed + tests_ctrl + tests_genctrl + tests_cubesat
    # tests = tests_genctrl[0:3]+tests_cubesat
    # # tests = [wisniewski_ctrl0_mini]
    # # tests = gen_ctrl_tests[-3:]
    # # tests = tests_disturbed
    # # tests = [wisniewski_match]
    tests = tests_ang+tests_B+tests_LyapV+tests_LyapVdot+tests_LyapNorm+tests_LyapNorm2+tests_LyapRmatV+tests_LyapRmatVdot+tests_LyapRmatNorm+tests_LyapRmatNorm2+tests_baseline
    tests = tests_LyapNorm+tests_B+tests_LyapVdot+tests_LyapNorm2+tests_LyapRmatV+tests_LyapRmatVdot+tests_LyapRmatNorm+tests_LyapRmatNorm2+tests_baseline+tests_ang+tests_LyapV
    # tests = tests_LyapRmatV[1:]

    # tests = tests_baseline[1:3] + tests_disturbed[1:3] + tests_ctrl[1:3] + tests_genctrl[1:3] + tests_cubesat
    for j in tests:
        try:
            [title,est_sat,real_sat,dt,w0,q0,cov_estimate,dist_control,goals,tf,orb_file,quatset_type] = j
            est_sat = copy.deepcopy(est_sat)
            estimate = np.zeros(est_sat.state_len+est_sat.act_bias_len+est_sat.att_sens_bias_len+est_sat.dist_param_len)
            estimate[0:3] = w0
            estimate[3:7] = q0
            if np.any([j.estimated_param for j in est_sat.disturbances]):
                dist_ic = block_diag(*[j.std**2.0 for j in est_sat.disturbances if j.estimated_param])
            else:
                dist_ic = np.zeros((0,0))
            int_cov =  dt*block_diag(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),dist_ic)
            # breakpoint()
            print(int_cov.shape)
            print(estimate.shape)
            print(est_sat.state_len,est_sat.act_bias_len,est_sat.att_sens_bias_len,est_sat.dist_param_len)
            print(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]).shape,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]).shape,np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]).shape,dist_ic.shape)
            est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True)
            est.use_cross_term = True
            est.al = 1.0#1e-3#e-3#1#1e-1#al# 1e-1
            est.kap =  3 - (estimate.size - 1 - sum([j.use_noise for j in est.sat.actuators]))
            est.include_int_noise_separately = False
            est.include_sens_noise_separately = False
            est.scale_nonseparate_adds = False
            est.included_int_noise_where = 2

            if isinstance(real_sat.actuators[0],MTQ):
                control_laws =  [NoControl(est.sat),Bdot(1e8,est.sat),Lovera([0.01,50,50],est.sat,include_disturbances=dist_control,quatset=(len(quatset_type)>0),quatset_type = quatset_type,calc_av_from_quat = True),WisniewskiSliding([np.eye(3)*0.002,np.eye(3)*0.003],est.sat,include_disturbances=dist_control,calc_av_from_quat = True,quatset=(len(quatset_type)>0),quatset_type = quatset_type)]
            else:
                control_laws =  [NoControl(est.sat),Magic_PD([np.eye(3)*200,np.eye(3)*5.0],est.sat,include_disturbances=dist_control,quatset=(len(quatset_type)>0),quatset_type = quatset_type,calc_av_from_quat = True)]

            state0 = np.zeros(real_sat.state_len)
            state0[0:3] = w0
            state0[3:7] = q0
            run_sim(orb_file,state0,est,copy.deepcopy(real_sat),control_laws,goals,tf=tf,dt = dt,alt_title = title,rand=False)
        except Exception as ae:
            if isinstance(ae, KeyboardInterrupt):
                raise
            else:
                traceback.print_exc()
                pass
