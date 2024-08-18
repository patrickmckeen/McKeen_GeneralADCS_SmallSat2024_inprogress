#estimation results for paper
from sat_ADCS_estimation import *
from sat_ADCS_control import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
from sat_ADCS_ADCS import *
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
import traceback,sys,code
from bdb import BdbQuit

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



def run_sim(orb,state0,real_sat,adcsys,base_title, j2000_0 = 0.22,tf=60*60*3,dt = 1,rand = False):

    ##save a copy of this file bc I'm bad at debugging and SVN
    this_script = os.path.basename(__file__)
    with open(this_script, 'r') as f2:
        script_copy = f2.read()
    with open("thesis_test_files/"+base_title+"/program_copy.txt", 'w') as f:
        f.write(script_copy)

    ##rest of the script
    #crab crab crab
    if not rand:
        np.random.seed(1)

    #
    #real_sat

    with open(orb, "rb") as fp:   #unPickling
        orb = pickle.load(fp)

    if (orb.times[-1]-j2000_0)*cent2sec < 1.01*tf:
        raise ValueError("orbit too short for this.")

    if isinstance(adcsys,ADCS_Bx):
        mag_field_magic_wrapper(orb)

    est=adcsys.estimator

    est_sat = est.sat
    t0 = 0
    q0 = state0[3:7]
    w0 = state0[0:3]
    h0 = state0[7:]

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),real_sat.state_len+sum([j.has_bias for j in real_sat.actuators])+sum([j.has_bias for j in real_sat.sensors])+sum([j.time_varying*j.main_param.size for j in real_sat.disturbances if j.time_varying])))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),adcsys.estimator.state_len))
    plan_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),adcsys.estimator.state_len))

    goal_hist =  []
    orb_hist = []
    orb_est_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    plan_control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(est_sat.control_len)
    prev_control = control.copy()
    state = state0
    orbt = orb.get_os(j2000_0+(t-t0)*sec2cent)
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    plan_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9+len(real_sat.momentum_inds)))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))
    int_err_hist =  np.nan*np.zeros((int((tf-t0)/dt),real_sat.state_len-1))

    est_sat0 = copy.deepcopy(est_sat)
    int_err = np.zeros(real_sat.state_len-1)

    first_ADCS = True
    while t<tf:

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        gps_sens = real_sat.GPS_values(state,real_vecs)

        # print(sens)
        # print(len(sens))
        # print(sens[-6:-3])

        #run estimator
        # print('real sensors outside ',sens)
        # print('ideal sensors outside',real_sat.noiseless_sensor_values(state,real_vecs))

        prev_est_state = adcsys.estimator.use_state.val.copy()

        tt0 = time.process_time()
        prev_control = control.copy()
        # breakpoint()
        # print(state)
        control = adcsys.ADCS_update(j2000_0+(t-t0)*sec2cent,sens,gps_sens,state_truth = state,first_ADCS = first_ADCS)
        first_ADCS = False
        if adcsys.prop_status:
            real_sat.prop_dist_on()
        else:
            real_sat.prop_dist_off()
        # print('1',real_sat.disturbances[-1].active,est.sat.disturbances[-1].active)
        # print(np.sqrt(np.diag(est.use_state.cov)[-3:]),np.log10(np.sqrt(np.diag(est.use_state.cov)[-3:])))
        # print(np.sqrt(np.diag(est.full_state.cov)[-3:]),np.log10(np.sqrt(np.diag(est.full_state.cov)[-3:])))
        # print(np.sqrt(np.diag(est.use_state.int_cov)[-3:]),np.log10(np.sqrt(np.diag(est.use_state.int_cov)[-3:])))
        # print(np.sqrt(np.diag(est.full_state.int_cov)[-3:]),np.log10(np.sqrt(np.diag(est.full_state.int_cov)[-3:])))
        # print(est.variable_dist_info[3].value)
        est_state = adcsys.estimator.use_state.val
        # adcsys.orbital_state_update(gps_sens,j2000_0+(t-t0)*sec2cent)
        # adcsys.estimation_update(sens,j2000_0+(t-t0)*sec2cent,control)
        est_vecs = os_local_vecs(orbt,est_state[3:7])

        # control = adcsys.actuation(j2000_0+(t-t0)*sec2cent,sens)
        goal_state = adcsys.current_goal
        # next_goal_state = adcsys.next_goal
        tt1 = time.process_time()

        # if t>250:
        #     breakpoint()
        # control = ctrlr.find_actuation(est_state,orbt,nextorb,goal_state,[],next_goal_state,sens,[],False)
        print('======TIME======',t)
        print('quat',state[3:7])#,norm(q_err[1:]),norm(np.cross(q_err[1:],est.os_vecs['b']))/norm(est.os_vecs['b']))
        print('est quat',est_state[3:7])#,norm(q_err[1:]),norm(np.cross(q_err[1:],est.os_vecs['b']))/norm(est.os_vecs['b']))
        # print(adcsys.control_laws)


        # breakpoint()
        try:
            print('plan quat',goal_state.state[3:7])
            print(goal_state.eci_vec,goal_state.body_vec)
            print('goalquat',goal_state.state[3:7])
            print('ang bw state and goal: ', (180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(goal_state.state[3:7],state[3:7]),-1,1)**2.0 ))
            # print('ang bw est state and goal: ', (180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(goal_state.state[3:7],est_state[3:7]),-1,1)**2.0 ))
            print('ang bw est state and state: ', (180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
            # print(goal_state.body_vec@rot_mat(goal_state.state[3:7]).T)
            # print(goal_state.eci_vec)
            # print('ang bw eci vec and body vec: ', (180/np.pi)*math.acos(np.dot(goal_state.eci_vec,goal_state.body_vec@rot_mat(state[3:7]).T )))
        except:
            pass

        print('ctrl',control)
        print('av',state[0:3]*180.0/math.pi,norm(state[0:3])*180.0/math.pi)#,(180.0/math.pi)*norm(np.cross(state[0:3],est.os_vecs['b']))/norm(est.os_vecs['b']))
        print('est av',est_state[0:3]*180.0/math.pi,norm(est_state[0:3])*180.0/math.pi)#,(180.0/math.pi)*norm(np.cross(state[0:3],est.os_vecs['b']))/norm(est.os_vecs['b']))
        print('est av err',(est_state[0:3]-state[0:3])*180.0/math.pi,norm((est_state[0:3]-state[0:3]))*180.0/math.pi)#,(180.0/math.pi)*norm(np.cross(state[0:3],est.os_vecs['b']))/norm(est.os_vecs['b']))

        plan_av = np.nan*np.zeros(3)
        try:
            plan_av = goal_state.state[0:3]
            print('plan av',plan_av*180.0/math.pi,norm(plan_av*180.0/math.pi))
        except:
            pass
        print('state extra',state[7:])
        print('est state extra',est_state[7:].reshape((int((est_state.size-7)/3),3)))
        # print(state[7:]*1e4)
        # print(sens)
        # print(orbt.in_eclipse())
        #
        print('dist ',real_sat.last_dist_torq,norm(real_sat.last_dist_torq))
        disttorqest = adcsys.estimator.sat.dist_torque(est_state[:adcsys.estimator.sat.state_len],est_vecs).copy()
        print('dist est ',disttorqest,norm(disttorqest))
        print(real_sat.last_dist_torq - disttorqest,norm(real_sat.last_dist_torq - disttorqest))
        # # print(est_state[-3:])
        # print(real_sat.disturbances[-2].main_param)
        # for j in est_sat.disturbances:
        #     print(j.main_param)
        # for j in real_sat.disturbances:
        #     print(j.main_param)
        # for j in real_sat.disturbances:
        #     if isinstance(j,GG_Disturbance):
        #         tq = j.torque(real_sat,real_vecs)
        #         print(tq, norm(tq))
        # for j in est_sat.disturbances:
        #     if isinstance(j,GG_Disturbance):
        #         tq = j.torque(est_sat,est_vecs)
        #         print(tq, norm(tq))
        # print('act ',real_sat.last_act_torq)
        # print("B real", real_vecs["b"])
        # print("B est", Bbody)
        # print("B should meas",real_vecs["b"]*real_sat.sensors[0].scale + np.hstack([j.bias for j in real_sat.sensors[0:3]]))
        # print("B meas", sens[0:3])
        # Bbody = real_vecs["b"]@rot_mat(state[3:7])
        # # print("B est meas", Bbody*est_sat.sensors[0].scale + est_state[10:13])
        # quat_err = quat_mult(quat_inv(goal_state.state[3:7]),state[3:7])
        # nBbody = normalize(Bbody)
        # quat_err_unitvec = normalize(quat_err[1:])
        # w_err_body = state[0:3]-plan_av@rot_mat(quat_err)
        # w_err_body_unitvec = normalize(w_err_body)
        # print("ang bw rot err & Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,quat_err_unitvec))))
        # print("ang bw w err & Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,w_err_body_unitvec))))
        # print("ang bw rot err & w err", (180.0/np.pi)*np.arccos(np.abs(np.dot(quat_err_unitvec,w_err_body_unitvec))))
        # print("ang bw Bbody & rot err diff with w err", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(quat_err_unitvec-w_err_body_unitvec)))))
        # print(va[:,-1],np.sqrt(wa[-1]))
        # print("cov ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(Bbody,va[:,-1]))/norm(Bbody)))
        cov_tr = np.diag(adcsys.estimator.use_state.cov)
        # est_quat_err = quat_mult(quat_inv(est_state[3:7]),state[3:7])
        # print(est_quat_err)
        # print('mrp err',np.log10(np.abs(quat_to_vec3(est_quat_err,0))),np.log10(np.sqrt(cov_tr[3:6])))
        # print('av err',np.log10(np.abs(est_state[0:3]-state[0:3])),np.log10(np.sqrt(cov_tr[0:3])))
        # print('h err',np.log10(np.abs(est_state[7:est.sat.state_len]-state[7:est.sat.state_len])),np.log10(np.sqrt(cov_tr[6:est.sat.state_len-1])))
        # ii = est.sat.state_len
        # for j in range(len(real_sat.actuators)):
        #     if est.sat.actuators[j].has_bias:
        #         print('act err',j,np.log10(np.abs(est.sat.actuators[j].bias - real_sat.actuators[j].bias)),np.log10(np.sqrt(cov_tr[ii-1:ii+2])))
        #         ii += 3
        # for j in range(len(real_sat.sensors)):
        #     if est.sat.sensors[j].has_bias:
        #         print('sens err',j,np.log10(np.abs(est.sat.sensors[j].bias - real_sat.sensors[j].bias)),np.log10(np.sqrt(cov_tr[ii-1:ii])))
        #         ii += 1
        # try:
        #     for j in range(len(real_sat.disturbances)):
        #         if  real_sat.disturbances[j].time_varying:
        #             print('dist err',j,np.log10(np.abs(est.sat.disturbances[j].main_param - real_sat.disturbances[j].main_param)),np.log10(np.sqrt(cov_tr[ii-1:ii+2])))
        #             ii += 3
        # except:
        #     pass
        # acov = np.sqrt(np.diag(est.use_state.cov))
        # print('acov',acov.reshape((int(acov.size/3),3)) )

        #
        # real_dp = np.concatenate([j.main_param for j in real_sat.disturbances if j.time_varying and j.active]+[np.zeros(0)])
        # # real_dp = []
        # real_sbias = np.concatenate([real_sat.attitude_sensors[j].bias for j in range(len(real_sat.attitude_sensors)) if est.sat.attitude_sensors[j].has_bias]+[np.zeros(0)])
        # real_abias = np.concatenate([real_sat.actuators[j].bias for j in range(len(real_sat.actuators)) if est.sat.actuators[j].has_bias]+[np.zeros(0)])
        # real_full_state = np.concatenate([state.copy(),real_abias.copy(),real_sbias.copy(),real_dp.copy()])
        # errvec = est_state-real_full_state
        # errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0),errvec[7:]])
        # print('err vec ')
        # print(errvec.reshape((int(errvec.size/3),3)) )
        # # print('% err')
        # # print(errvec[6:].reshape((int((errvec.size-6)/3),3))/est_state[7:].reshape((int((est_state.size-7)/3),3)))
        # mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov)@errvec
        # print('mahalanobis_dist ',np.sqrt(mahalanobis_dist2))
        # pd = chi2.pdf(mahalanobis_dist2,errvec.size)
        # print(' prob ',pd)
        # print('rel prob ',pd/chi2.pdf(errvec.size-2,errvec.size))
        # cd = chi2.cdf(mahalanobis_dist2,errvec.size)
        # print('cdf ',cd)
        # stdeq = math.sqrt(2)*erfinv(cd)
        # print('std dev eq ',stdeq)
        # print('err/acov')
        # print(errvec.reshape((int((errvec.size)/3),3))/acov.reshape((int((acov.size)/3),3)))
        # if np.isinf(stdeq):
        #     breakpoint()
        # inv_sr = np.linalg.cholesky(np.linalg.inv(est.use_state.cov)).T

        # print('sens',sens)

        # for j in range(3):
        #     print(j)
        #     print(sens[j])
        #     print(real_sat.attitude_sensors[j].bias,real_sat.attitude_sensors[j].bias_std_rate,real_sat.attitude_sensors[j].clean_reading(state,real_vecs),real_sat.attitude_sensors[j].no_noise_reading(state,real_vecs),real_sat.attitude_sensors[j].reading(state,real_vecs),real_sat.attitude_sensors[j].reading(state,real_vecs)-real_sat.attitude_sensors[j].no_noise_reading(state,real_vecs),real_sat.attitude_sensors[j].no_noise_reading(state,real_vecs)-real_sat.attitude_sensors[j].clean_reading(state,real_vecs) ,real_sat.attitude_sensors[j].std)
        #     print(est.sat.attitude_sensors[j].bias,est.sat.attitude_sensors[j].bias_std_rate,est.sat.attitude_sensors[j].clean_reading(state,est_vecs),est.sat.attitude_sensors[j].no_noise_reading(state,est_vecs),est.sat.attitude_sensors[j].reading(state,est_vecs),est.sat.attitude_sensors[j].reading(state,est_vecs)-est.sat.attitude_sensors[j].no_noise_reading(state,est_vecs),est.sat.attitude_sensors[j].no_noise_reading(state,est_vecs)-est.sat.attitude_sensors[j].clean_reading(state,est_vecs),est.sat.attitude_sensors[j].std)
        #     print(np.sqrt(est.use_state.cov[10+j-1,10+j-1]))


        # w,v = np.linalg.eig(est.use_state.cov)
        # # w = np.abs(np.real(w))
        # srw = np.sqrt(w)#[math.sqrt(j) for j in np.abs(np.real(w))]
        # # v = np.real(v)
        # # inv_sr = np.linalg.inv()
        # wt_err = np.abs(np.diagflat((1.0/srw))@v.T@errvec)
        # maxerrind = np.argmax(wt_err)
        # print(max(wt_err),maxerrind)
        # print('direction most off',srw[maxerrind]*v[:,maxerrind])

        # breakpoint()
        # print(real_sat.sensors[0].bias[0],est.sat.sensors[0].bias[0]-real_sat.sensors[0].bias[0])
        # print([j.bias for j in real_sat.actuators if j.has_bias])
        # print([j.bias for j in est_sat.actuators if j.has_bias])
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
        int_err_hist[ind,:] = int_err
        est_state_hist[ind,:] = adcsys.estimator.full_state.val
        goal_hist += [goal_state.copy()]
        # breakpoint()
        orb_hist += [orbt]
        orb_est_hist += [adcsys.orbital_state]
        control_hist[ind,:] = prev_control
        cov_hist += [adcsys.estimator.full_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)
        dist_torq_hist[ind,:] = real_sat.last_dist_torq.copy()
        est_dist_torq_hist[ind,:] = adcsys.estimator.sat.dist_torque(prev_est_state[:adcsys.estimator.sat.state_len],est_vecs).copy()
        act_torq_hist[ind,:] = real_sat.last_act_torq.copy()
        est_act_torq_hist[ind,:] =  adcsys.estimator.sat.act_torque(prev_est_state[:adcsys.estimator.sat.state_len],prev_control,est_vecs,False)
        sens_hist[ind,:] = sens.copy()
        eclipse_hist[ind] = orbt.in_eclipse()
        if adcsys.current_mode in PlannerModeList:
            plan_state_hist[ind,0:adcsys.virtual_sat.state_len] = adcsys.planned_info[0]
            plan_control_hist[ind,:] = adcsys.planned_info[1]
            plan_dist_torq_hist[ind,:] = adcsys.planned_info[4]

        # print(adcsys.prop_status,est.sat.disturbances[-1].active,real_sat.disturbances[-1].active)
        # print(est.sat.disturbances[-1].torque(est.sat,est_vecs),real_sat.disturbances[-1].torque(real_sat,real_vecs))
        # breakpoint()

        # if ind>1:
        #     print(np.abs(dist_torq_hist[ind,:]-dist_torq_hist[ind-1,:]),norm(dist_torq_hist[ind,:]-dist_torq_hist[ind-1,:]))

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(j2000_0+(t-t0)*sec2cent)
        # orbt.B = 1e-4*unitvecs[0]

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6,atol = 1e-8)#3e-14, atol=1e-12)#np.concatenate([1e-8*math.pi/180.0*np.ones(3),1e-7*math.pi/180.0*np.ones(4),[1e-8*j.max_h for j in real_sat.actuators if j.has_momentum]]))#1e-8)#,jac = real_sat.dynamics_jac_for_solver)
        real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        est_sat.dynamics(adcsys.estimator.use_state.val[:est_sat.state_len],control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        basic_int,est_int_err = real_sat.rk4(state,control,dt,prev_os,orbt,verbose = False,quat_as_vec = True,save_info = False,give_err_est = True)



        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        int_err[0:3] = basic_int[0:3] - state[0:3]
        int_err[3:6] = quat_to_vec3(quat_mult(quat_inv(state[3:7]),basic_int[3:7]),0)
        int_err[6:real_sat.state_len-1] = basic_int[7:real_sat.state_len] - state[7:real_sat.state_len]

        print("int err",int_err)
        print("est int err",est_int_err)
        real_sat.update_RWhs(state)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)

    # breakpoint()
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])


    folder_name = "thesis_test_files"
    os.mkdir("thesis_test_files/"+base_title+"/environment")
    os.mkdir("thesis_test_files/"+base_title+"/actual")
    os.mkdir("thesis_test_files/"+base_title+"/estimated")
    os.mkdir("thesis_test_files/"+base_title+"/plan")
    os.mkdir("thesis_test_files/"+base_title+"/goal")
    os.mkdir("thesis_test_files/"+base_title+"/plan_v_actual")
    os.mkdir("thesis_test_files/"+base_title+"/estimated_v_actual")
    os.mkdir("thesis_test_files/"+base_title+"/actual_v_goal")
    os.mkdir("thesis_test_files/"+base_title+"/plan_v_goal")
    os.mkdir("thesis_test_files/"+base_title+"/estimated_v_plan")
    os.mkdir("thesis_test_files/"+base_title+"/estimated_v_goal")


    #preparation
    orbitRmats = np.dstack([rot_mat(two_vec_to_quat(-orb_hist[j].R,orb_hist[j].V,unitvecs[0],unitvecs[1])) for j in range(len(orb_hist))]) #stacked matrices such that if you take R=[:,;,i], R@unitvecs[0] would give the nadir direction coordinates in ECI, R@unitvecs[0] is ram direction
    plot_the_thing(eclipse_hist,title = "Eclipse History",xlabel='Time (s)',norm = False,ylabel = 'Eclipse',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/eclipse_plot")

    plot_the_thing(orbitRmats[0,:,:],title = "Orbit Nadir in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/oxECI_plot")
    plot_the_thing(orbitRmats[1,:,:],title = "Orbit Ram in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/oyECI_plot")
    plot_the_thing(orbitRmats[2,:,:],title = "Orbit anti-normal in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/ozECI_plot")

    plot_the_thing(np.stack([orb_hist[j].B for j in range(len(orb_hist))]),title = "B ECI frame",xlabel='Time (s)',norm = True,ylabel = 'B ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/bECI_plot")
    plot_the_thing(np.stack([orb_hist[j].R for j in range(len(orb_hist))]),title = "R ECI frame",xlabel='Time (s)',norm = True,ylabel = 'R ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/rECI_plot")
    plot_the_thing(np.stack([orb_hist[j].V for j in range(len(orb_hist))]),title = "V ECI frame",xlabel='Time (s)',norm = True,ylabel = 'V ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/vECI_plot")
    plot_the_thing(np.stack([orb_hist[j].S for j in range(len(orb_hist))]),title = "S ECI frame",xlabel='Time (s)',norm = True,ylabel = 'S ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/environment/sECI_plot")

    #goal
    point_vec_body = np.stack([j.body_vec for j in goal_hist])
    goal_vec_eci = np.stack([j.eci_vec for j in goal_hist])
    goalav_ECI = np.stack([goal_hist[j].state[0:3]@rot_mat(goal_hist[j].state[3:7]).T*180.0/np.pi for j in range(len(goal_hist))])
    plot_the_thing(np.stack([j.state[3:7] for j in goal_hist]),title = "Goal Quat",xlabel='Time (s)',ylabel = 'Goal Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goal/goalquat_plot")
    plot_the_thing(np.stack([j.state[0:3]*180.0/np.pi for j in goal_hist]),title = "Goal AV (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goal/goalav_plot")
    plot_the_thing(goalav_ECI,title = "Goal AV (ECI frame) (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (ECI frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goal/goalav_ECI_plot")
    plot_the_thing(point_vec_body,title = "Body Pointing Vector",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goal/point_vec_body_plot")
    plot_the_thing(goal_vec_eci,title = "ECI Goal Pointing Vector",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goal/goal_vec_eci_plot")

    #actual
    real_sb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],SunSensorPair) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_mb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],MTM) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_gb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],Gyro) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_ab_inds = (np.array([np.sum([k.input_len for k in real_sat.actuators[:j] if k.has_bias]) for j in range(len(real_sat.actuators)) if real_sat.actuators[j].has_bias]) + real_sat.state_len).astype('int')

    mtq_ctrl_inds = np.array([j for j in range(len(real_sat.actuators)) if isinstance(real_sat.actuators[j],MTQ)]).astype('int')
    rw_ctrl_inds = np.array([j for j in range(len(real_sat.actuators)) if isinstance(real_sat.actuators[j],RW)]).astype('int')
    thr_ctrl_inds = np.array([j for j in range(len(real_sat.actuators)) if isinstance(real_sat.actuators[j],Magic)]).astype('int')

    g_sens_inds = np.array([j for j in range(len(real_sat.sensors)) if isinstance(real_sat.sensors[j],Gyro)]).astype('int')
    m_sens_inds = np.array([j for j in range(len(real_sat.sensors)) if isinstance(real_sat.sensors[j],MTM)]).astype('int')
    s_sens_inds = np.array([j for j in range(len(real_sat.sensors)) if isinstance(real_sat.sensors[j],SunSensorPair)]).astype('int')

    Bbody = np.stack([orb_hist[j].B@rot_mat(state_hist[j,3:7]) for j in range(len(orb_hist))])


    try:
        real_dipole_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(real_sat.disturbances)) if  isinstance(real_sat.disturbances[j],Dipole_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_length for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_dipole_inds = np.array([]).astype('int')
    try:
        real_gendist_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(real_sat.disturbances)) if isinstance(real_sat.disturbances[j],General_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_length for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_gendist_inds = np.array([]).astype('int')
    try:
        real_prop_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(real_sat.disturbances)) if isinstance(real_sat.disturbances[j],Prop_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_length for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_prop_inds = np.array([]).astype('int')

    if real_ab_inds.size > 0:
        abreal = state_hist[:,real_ab_inds]
    else:
        abreal = np.zeros((est_state_hist.shape[0],3))
    if real_gb_inds.size > 0:
        gbreal = state_hist[:,real_gb_inds]*180.0/math.pi
    else:
        gbreal = np.zeros((est_state_hist.shape[0],3))

    if real_mb_inds.size > 0:
        mbreal = state_hist[:,real_mb_inds]#*180.0/math.pi
    else:
        mbreal = np.zeros((est_state_hist.shape[0],3))

    if real_sb_inds.size > 0:
        sbreal = state_hist[:,real_sb_inds]#*180.0/math.pi
    else:
        sbreal = np.zeros((est_state_hist.shape[0],3))

    if real_dipole_inds.size > 0:
        dpreal = state_hist[:,real_dipole_inds]
    else:
        dpreal = np.zeros((est_state_hist.shape[0],3))

    dpreal_nobb = np.stack([-np.cross(Bbody[j,:],np.cross(Bbody[j,:],dpreal[j,:]))/np.dot(Bbody[j,:],Bbody[j,:]) for j in range(dpreal.shape[0])])


    if real_gendist_inds.size > 0:
        genreal = state_hist[:,real_gendist_inds]
    else:
        genreal = np.zeros((est_state_hist.shape[0],3))

    if real_prop_inds.size > 0:
        propreal = state_hist[:,real_prop_inds]
    else:
        propreal = np.zeros((est_state_hist.shape[0],3))
    Rmats = np.dstack([rot_mat(state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    av_ECI = np.stack([state_hist[j,0:3]@rot_mat(state_hist[j,3:7]).T*180.0/np.pi for j in range(state_hist.shape[0])])
    angmom_ECI = np.stack([state_hist[j,0:3]@real_sat.J@rot_mat(state_hist[j,3:7]).T*180.0/np.pi for j in range(state_hist.shape[0])])

    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/av_plot")
    plot_the_thing(state_hist[:,7:real_sat.state_len],title = "Stored AM",xlabel='Time (s)',norm = True,ylabel = 'Angular Momentum (Nms)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/stored_am_plot")

    plot_the_thing(av_ECI,title = "AV (ECI frame) (deg/s)",xlabel='Time (s)',ylabel = 'AV (ECI frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/av_ECI_plot")
    plot_the_thing(angmom_ECI,title = "Angular Momentum (ECI frame) (deg/s)",xlabel='Time (s)',ylabel = 'H (ECI frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/angmom_ECI_plot")

    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/quat_plot")

    plot_the_thing(act_torq_hist,title = 'act Torq',xlabel = 'Time (s)',norm = True, ylabel = 'act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/actual/act_torq_plot")
    plot_the_thing(dist_torq_hist,title = 'Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/actual/dist_torq_plot")
    plot_the_thing(dist_torq_hist+act_torq_hist,title = 'Combo Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Combo Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/actual/combo_torq_plot")

    plot_the_thing(abreal,title = "Actuator Bias",xlabel='Time (s)',norm = True,ylabel = 'Actuator Bias ([units])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/abias_plot")
    plot_the_thing(gbreal,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/gbias_plot")
    plot_the_thing(mbreal,title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/mbias_plot")
    plot_the_thing(sbreal,title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/sbias_plot")
    plot_the_thing(dpreal,title = "Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dipole (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/dipole_plot")
    plot_the_thing(dpreal_nobb,title = "Dipole w/o Bbody",xlabel='Time (s)',norm = True,ylabel = 'Dipole (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/dipole_nobb_plot")

    plot_the_thing(genreal,title = "Gen Torque",xlabel='Time (s)',norm = True,ylabel = 'Gen Torque (Nm)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/gen_torq_plot")
    plot_the_thing(propreal,title = "Prop Torque",xlabel='Time (s)',norm = True,ylabel = 'Prop Torque (Nm)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/prop_torq_plot")


    plot_the_thing(sens_hist[:,g_sens_inds]*180.0/math.pi,title = "Gyro Sensor",xlabel='Time (s)',norm = True,ylabel = 'Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/gsens_plot")
    plot_the_thing(sens_hist[:,g_sens_inds]*180.0/math.pi - gbreal,title = "Unbiased Gyro Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/gsens_unbiased_plot")
    plot_the_thing(sens_hist[:,m_sens_inds],title = "MTM Sensor",xlabel='Time (s)',norm = True,ylabel = 'MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/msens_plot")
    plot_the_thing(sens_hist[:,m_sens_inds] - mbreal,title = "Unbiased MTM Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/msens_unbiased_plot")
    plot_the_thing(sens_hist[:,s_sens_inds],title = "Sun Sensor",xlabel='Time (s)',norm = True,ylabel = 'Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/ssens_plot")
    plot_the_thing(sens_hist[:,s_sens_inds] - sbreal,title = "Unbiased Sun Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/ssens_unbiased_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/ctrl_plot")
    plot_the_thing(control_hist[:,mtq_ctrl_inds],title = "Control MTQ",xlabel='Time (s)',norm = True,ylabel = 'Control MTQ (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/ctrlmtq_plot")
    plot_the_thing(control_hist[:,rw_ctrl_inds],title = "Control RW",xlabel='Time (s)',norm = True,ylabel = 'Control RW (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/ctrlrw_plot")
    plot_the_thing(control_hist[:,thr_ctrl_inds],title = "Control Thrusters",xlabel='Time (s)',norm = True,ylabel = 'Control Thrusters (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/ctrlthr_plot")

    plot_the_thing(Bbody,title = "B Body frame",xlabel='Time (s)',norm = True,ylabel = 'B Body Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/bbody_plot")

    plot_the_thing(Rmats[0,:,:],title = "Body x-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/bxECI_plot")
    plot_the_thing(Rmats[1,:,:],title = "Body y-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/byECI_plot")
    plot_the_thing(Rmats[2,:,:],title = "Body z-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/bzECI_plot")

    plot_the_thing(np.stack([orbitRmats[:,:,j].T@Rmats[0,:,j] for j in range(Rmats.shape[2])]),title = "Body x-axis in x=nadir,ram=y frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/bxo_plot")
    plot_the_thing(np.stack([orbitRmats[:,:,j].T@Rmats[1,:,j] for j in range(Rmats.shape[2])]),title = "Body y-axis in x=nadir,ram=y frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/byo_plot")
    plot_the_thing(np.stack([orbitRmats[:,:,j].T@Rmats[2,:,j] for j in range(Rmats.shape[2])]),title = "Body z-axis in x=nadir,ram=y frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/bzo_plot")

    point_vec_eci = np.stack([point_vec_body[j,:]@Rmats[:,:,j].T for j in range(point_vec_body.shape[0])])
    goal_vec_body = np.stack([goal_vec_eci[j,:]@Rmats[:,:,j] for j in range(goal_vec_eci.shape[0])])
    plot_the_thing(point_vec_eci,title = "Body Pointing Vector in ECI",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/point_vec_eci_plot")
    plot_the_thing(goal_vec_body,title = "Goal Pointing Vector in Body",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/goal_vec_body_plot")



    #planned
    plannedRmats = np.dstack([rot_mat(plan_state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    plannedav_ECI = np.stack([plan_state_hist[j,0:3]@rot_mat(plan_state_hist[j,3:7]).T*180.0/np.pi for j in range(plan_state_hist.shape[0])])
    plannedav_realbody = np.stack([plan_state_hist[j,0:3]@rot_mat(plan_state_hist[j,3:7]).T@rot_mat(state_hist[j,3:7])*180.0/np.pi for j in range(plan_state_hist.shape[0])])
    av_ECI_planbody = np.stack([state_hist[j,0:3]@rot_mat(state_hist[j,3:7]).T@rot_mat(plan_state_hist[j,3:7])*180.0/np.pi for j in range(state_hist.shape[0])])
    plannedangmom_ECI = np.stack([plan_state_hist[j,0:3]@real_sat.J@rot_mat(plan_state_hist[j,3:7]).T*180.0/np.pi for j in range(plan_state_hist.shape[0])])

    plot_the_thing(plan_state_hist[:,0:3]*180.0/math.pi,title = "Planned AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_av_plot")
    plot_the_thing(plannedav_ECI,title = "Planned AV (ECI frame)",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_av_eci_plot")
    plot_the_thing(plannedangmom_ECI,title = "Planned Angular Momentum (ECI frame) (deg/s)",xlabel='Time (s)',ylabel = 'H (ECI frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_angmom_ECI_plot")

    plot_the_thing(plan_state_hist[:,3:7],title = "Planned Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_quat_plot")

    plan_av_quat = np.stack([2*quat_mult(quat_inv(plan_state_hist[j,3:7]),plan_state_hist[j+1,3:7])[1:] for j in range(plan_state_hist.shape[0]-1)])
    # breakpoint()
    plot_the_thing(plan_av_quat*180.0/math.pi,title = "Planned AV from Quat",xlabel='Time (s)',norm = True,ylabel = 'AV estimated from Quats',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_av_from_quat_plot")

    plot_the_thing(plan_dist_torq_hist,title = 'Planned Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_dist_torq_plot")
    plot_the_thing(plan_control_hist,title = "Planned Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/planctrl_plot")
    plot_the_thing(plan_control_hist[:,mtq_ctrl_inds],title = "Planned Control MTQ",xlabel='Time (s)',norm = True,ylabel = 'Control MTQ (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/planctrlmtq_plot")
    plot_the_thing(plan_control_hist[:,rw_ctrl_inds],title = "Planned Control RW",xlabel='Time (s)',norm = True,ylabel = 'Control RW (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/planctrlrw_plot")
    plot_the_thing(plan_control_hist[:,thr_ctrl_inds],title = "Planned Control Thrusters",xlabel='Time (s)',norm = True,ylabel = 'Control Thrusters (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/planctrlthr_plot")

    plan_point_vec_eci = np.stack([point_vec_body[j,:]@plannedRmats[:,:,j].T for j in range(point_vec_body.shape[0])])
    plan_goal_vec_body = np.stack([goal_vec_eci[j,:]@plannedRmats[:,:,j] for j in range(point_vec_body.shape[0])])
    plan_alignment_rot_ax_body = np.stack([np.cross(plan_goal_vec_body[j,:],point_vec_body[j,:])for j in range(point_vec_body.shape[0])])
    plan_alignment_rot_ax_eci = np.stack([np.cross(plan_point_vec_eci[j,:],goal_vec_eci[j,:])for j in range(point_vec_body.shape[0])])
    plot_the_thing(plan_point_vec_eci,title = "Planned Body Pointing Vector in ECI",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_point_vec_eci_plot")
    plot_the_thing(plan_goal_vec_body,title = "Planned Goal ECI Pointing Vector in Body",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_goal_vec_body_plot")



    #estimated
    est_sb_inds = (np.array([np.sum([k.output_length for k in adcsys.estimator.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(adcsys.estimator.sat.attitude_sensors)) if isinstance(adcsys.estimator.sat.attitude_sensors[j],SunSensorPair) and adcsys.estimator.sat.attitude_sensors[j].has_bias and adcsys.estimator.sat.attitude_sensors[j].estimated_bias]) + adcsys.estimator.sat.state_len+adcsys.estimator.sat.act_bias_len).astype('int')
    est_mb_inds = (np.array([np.sum([k.output_length for k in adcsys.estimator.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(adcsys.estimator.sat.attitude_sensors)) if isinstance(adcsys.estimator.sat.attitude_sensors[j],MTM) and adcsys.estimator.sat.attitude_sensors[j].has_bias and adcsys.estimator.sat.attitude_sensors[j].estimated_bias]) + adcsys.estimator.sat.state_len+adcsys.estimator.sat.act_bias_len).astype('int')
    est_gb_inds = (np.array([np.sum([k.output_length for k in adcsys.estimator.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(adcsys.estimator.sat.attitude_sensors)) if isinstance(adcsys.estimator.sat.attitude_sensors[j],Gyro) and adcsys.estimator.sat.attitude_sensors[j].has_bias and adcsys.estimator.sat.attitude_sensors[j].estimated_bias]) + adcsys.estimator.sat.state_len+adcsys.estimator.sat.act_bias_len).astype('int')
    est_ab_inds = (np.array([np.sum([k.input_len for k in adcsys.estimator.sat.actuators[:j] if k.estimated_bias and k.has_bias]) for j in range(len(adcsys.estimator.sat.actuators)) if adcsys.estimator.sat.actuators[j].has_bias and adcsys.estimator.sat.actuators[j].estimated_bias]) + adcsys.estimator.sat.state_len).astype('int')
    try:
        est_dipole_inds = (np.concatenate([np.sum([k.main_param.size for k in adcsys.estimator.sat.disturbances[:j] if k.estimated_param])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(adcsys.estimator.sat.disturbances)) if adcsys.estimator.sat.disturbances[j].estimated_param and isinstance(adcsys.estimator.sat.disturbances[j],Dipole_Disturbance)]) + adcsys.estimator.sat.state_len+adcsys.estimator.sat.act_bias_len+adcsys.estimator.sat.att_sens_bias_len).astype('int')
    except:
        est_dipole_inds = np.array([]).astype('int')
    try:
        est_gendist_inds = (np.concatenate([np.sum([k.main_param.size for k in adcsys.estimator.sat.disturbances[:j] if k.estimated_param])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(adcsys.estimator.sat.disturbances)) if adcsys.estimator.sat.disturbances[j].estimated_param and isinstance(adcsys.estimator.sat.disturbances[j],General_Disturbance)]) + adcsys.estimator.sat.state_len+adcsys.estimator.sat.act_bias_len+adcsys.estimator.sat.att_sens_bias_len).astype('int')
    except:
        est_gendist_inds = np.array([]).astype('int')
    try:
        est_prop_inds = (np.concatenate([np.sum([k.main_param.size for k in adcsys.estimator.sat.disturbances[:j] if k.estimated_param])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(adcsys.estimator.sat.disturbances)) if adcsys.estimator.sat.disturbances[j].estimated_param and isinstance(adcsys.estimator.sat.disturbances[j],Prop_Disturbance)]) + adcsys.estimator.sat.state_len+adcsys.estimator.sat.act_bias_len+adcsys.estimator.sat.att_sens_bias_len).astype('int')
    except:
        est_prop_inds = np.array([]).astype('int')

    if est_ab_inds.size > 0:
        abest = est_state_hist[:,est_ab_inds]
    else:
        abest = np.zeros((est_state_hist.shape[0],3))

    if est_gb_inds.size > 0:
        gbest = est_state_hist[:,est_gb_inds]*180.0/math.pi
    else:
        gbest = np.zeros((est_state_hist.shape[0],3))

    if est_mb_inds.size > 0:
        mbest = est_state_hist[:,est_mb_inds]#*180.0/math.pi
    else:
        mbest = np.zeros((est_state_hist.shape[0],3))

    if est_sb_inds.size > 0:
        sbest = est_state_hist[:,est_sb_inds]#*180.0/math.pi
    else:
        sbest = np.zeros((est_state_hist.shape[0],3))
    if est_dipole_inds.size > 0:
        dpest = est_state_hist[:,est_dipole_inds]
    else:
        dpest = np.zeros((est_state_hist.shape[0],3))

    dpest_nobb = np.stack([-np.cross(Bbody[j,:],np.cross(Bbody[j,:],dpest[j,:]))/np.dot(Bbody[j,:],Bbody[j,:]) for j in range(dpest.shape[0])])
    if est_prop_inds.size > 0:
        propest = est_state_hist[:,est_prop_inds]
    else:
        propest = np.zeros((est_state_hist.shape[0],3))
    if est_gendist_inds.size > 0:
        genest = est_state_hist[:,est_gendist_inds]
    else:
        genest = np.zeros((est_state_hist.shape[0],3))
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_av_plot")
    plot_the_thing(est_state_hist[:,7:adcsys.estimator.sat.state_len],title = "Est Stored AM",xlabel='Time (s)',norm = True,ylabel = 'Angular Momentum (Nms)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_stored_am_plot")

    plot_the_thing(est_act_torq_hist,title = 'Estimated act Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_act_torq_plot")
    plot_the_thing(est_dist_torq_hist,title = 'Estimated Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_dist_torq_plot")
    plot_the_thing(est_dist_torq_hist+est_act_torq_hist,title = 'Estimated Combo Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Combo Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_combo_torq_plot")

    plot_the_thing(abest,title = "Est Actuator Bias",xlabel='Time (s)',norm = True,ylabel = 'Actuator Bias ([units])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_abias_plot")
    plot_the_thing(gbest,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_gbias_plot")
    plot_the_thing(mbest,title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_mbias_plot")
    plot_the_thing(sbest,title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_sbias_plot")
    plot_the_thing(dpest,title = "Est Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dipole (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_dipole_plot")
    plot_the_thing(dpest_nobb,title = "Est Dipole w/o Bbody",xlabel='Time (s)',norm = True,ylabel = 'Est Dipole (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_dipole_nobb_plot")

    plot_the_thing(genest,title = "Est General Torque",xlabel='Time (s)',norm = True,ylabel = 'Est Gen Torq (Nm)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_gentorq_plot")
    plot_the_thing(propest,title = "Est Prop Torque",xlabel='Time (s)',norm = True,ylabel = 'Est Prop Torq (Nm)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_proptorq_plot")

    plot_the_thing(np.stack([orb_est_hist[j].R for j in range(len(orb_hist))]),title = "Estimated R ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Estimated R ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/rECI_est_plot")
    plot_the_thing(np.stack([orb_est_hist[j].V for j in range(len(orb_hist))]),title = "Estimated V ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Estimated V ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/vECI_est_plot")

    plot_the_thing(autocov_hist[:,0:3],title = "AV Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'AV AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/av_acov_plot")
    plot_the_thing(autocov_hist[:,3:6],title = "Ang Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'Ang AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/mrp_acov_plot")
    plot_the_thing(autocov_hist[:,6:adcsys.estimator.sat.state_len-1],title = "h Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'h AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/h_acov_plot")
    plot_the_thing(autocov_hist[:,est_ab_inds-1],title = "Act Bias Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'Act Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/abias_acov_plot")
    plot_the_thing(autocov_hist[:,est_mb_inds-1],title = "MTM Bias Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/mbias_acov_plot")
    plot_the_thing(autocov_hist[:,est_gb_inds-1],title = "Gyro Bias Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/gbias_acov_plot")
    plot_the_thing(autocov_hist[:,est_sb_inds-1],title = "Sun Bias Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/sbias_acov_plot")
    plot_the_thing(autocov_hist[:,est_dipole_inds-1],title = "Dipole Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'Dipole AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/dp_acov_plot")
    plot_the_thing(autocov_hist[:,est_gendist_inds-1],title = "Gen Torq Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'Gen Torq AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/gen_acov_plot")
    plot_the_thing(autocov_hist[:,est_prop_inds-1],title = "Prop Torq Cov Trace",xlabel='Time (s)',norm = True,ylabel = 'Prop Torq AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/prop_acov_plot")

    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,0:3])),title = "AV Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'AV AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logav_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,3:6])),title = "Ang Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'Ang AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logmrp_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,6:adcsys.estimator.sat.state_len-1])),title = "h Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'h AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logh_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,est_ab_inds-1])),title = "Act Bias Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'Act Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logabias_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,est_mb_inds-1])),title = "MTM Bias Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'MTM Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logmbias_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,est_gb_inds-1])),title = "Gyro Bias Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'Gyro Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_loggbias_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,est_sb_inds-1])),title = "Sun Bias Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'Sun Bias AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logsbias_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,est_dipole_inds-1])),title = "Dipole Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'Dipole AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logdp_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,est_gendist_inds-1])),title = "Gen Torq Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'Gen Torq AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_loggen_acov_plot")
    plot_the_thing(np.log10(matrix_row_norm(autocov_hist[:,est_prop_inds-1])),title = "Prop Torq Cov Trace",xlabel='Time (s)',norm = False,ylabel = 'Prop Torq AutoCov',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/_logprop_acov_plot")




    expected_clean_reading = np.array([[j.reading(est_state_hist[k,:],os_local_vecs(orb_hist[k],est_state_hist[k,3:7])) for j in adcsys.estimator.sat.attitude_sensors] for k in range(est_state_hist.shape[0])])
    expected_clean_msens = expected_clean_reading[:,m_sens_inds]
    expected_clean_gsens = expected_clean_reading[:,g_sens_inds]
    expected_clean_ssens = expected_clean_reading[:,s_sens_inds]
    plot_the_thing(expected_clean_gsens*180.0/math.pi + gbest,title = "Expected Gyro Sensor",xlabel='Time (s)',norm = True,ylabel = 'Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/exp_gsens_plot")
    plot_the_thing(expected_clean_gsens*180.0/math.pi,title = "Expected Unbiased Gyro Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/exp_gsens_unbiased_plot")
    plot_the_thing(expected_clean_msens + mbest,title = "Expected MTM Sensor",xlabel='Time (s)',norm = True,ylabel = 'MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/exp_msens_plot")
    plot_the_thing(expected_clean_msens,title = "Expected Unbiased MTM Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/exp_msens_unbiased_plot")
    plot_the_thing(expected_clean_ssens + sbest,title = "Expected Sun Sensor",xlabel='Time (s)',norm = True,ylabel = 'Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/exp_ssens_plot")
    plot_the_thing(expected_clean_ssens,title = "Expected Unbiased Sun Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/exp_ssens_unbiased_plot")


    #estimated v actual
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7])*np.sign(np.dot(est_state_hist[j,3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    hdiff = est_state_hist[:,7:adcsys.estimator.sat.state_len]-state_hist[:,7:adcsys.estimator.sat.state_len]
    abdiff = abest-abreal
    gbdiff = gbest-gbreal
    mbdiff = mbest-mbreal
    sbdiff = sbest-sbreal
    dpdiff = dpest-dpreal
    dpdiff_nobb = dpest_nobb-dpreal_nobb
    gendiff = genest-genreal
    propdiff = propest-propreal

    plot_the_thing(quatdiff,title = "Quat Error",xlabel='Time (s)',ylabel = 'Quat Error', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/quaterr_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/averr_plot")
    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/mrp_plot")
    plot_the_thing(hdiff,title = "Stored AM Error",xlabel='Time (s)',ylabel = 'Stored AM Error (Nms)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/herr_plot")

    plot_the_thing(int_err_hist[:,0:3],title = "AV int Error",xlabel='Time (s)',ylabel = 'AV Int Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/av_int_err_plot")
    plot_the_thing(int_err_hist[:,3:6],title = "MRP int Error",xlabel='Time (s)',ylabel = 'MRP Int Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/mrp_int_err_plot")
    plot_the_thing(int_err_hist[:,6:],title = "Stored AM int Error",xlabel='Time (s)',ylabel = 'Stored AM Int Error (Nms)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/h_int_err_plot")

    plot_the_thing(np.log10(matrix_row_norm(int_err_hist[:,0:3])),title = "Log10 AV int Error",xlabel='Time (s)',ylabel = 'AV Int Error (log deg/s)', xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_av_int_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(int_err_hist[:,3:6])),title = "Log10 MRP int Error",xlabel='Time (s)',ylabel = 'MRP Int Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_mrp_int_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(int_err_hist[:,6:])),title = "Log10 Stored AM int Error",xlabel='Time (s)',ylabel = 'Stored AM Int Error (log Nms)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_h_int_err_plot")

    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logav_plot")
    plot_the_thing(np.log10(matrix_row_norm(hdiff)),title = "Log Stored AM Error",xlabel='Time (s)',ylabel = 'AV Error (log Nms)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logh_plot")

    plot_the_thing(state_hist[:,3:7],est_state_hist[:,3:7],title = "Actual v Estimated Quat",xlabel='Time (s)',ylabel = 'Quat', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/quatcomp_plot")
    plot_the_thing(state_hist[:,0:3],est_state_hist[:,0:3],title = "Actual v Estimated AV",xlabel='Time (s)',ylabel = 'AV (deg/s)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/avcomp_plot")
    plot_the_thing(state_hist[:,7:adcsys.estimator.sat.state_len],est_state_hist[:,7:adcsys.estimator.sat.state_len],title = "Actual v Estimated Stored AM",xlabel='Time (s)',ylabel = 'Stored AM (Nms)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/hcomp_plot")
    plot_the_thing(abreal,abest,title = "Actual v Estimated Act Bias",xlabel='Time (s)',ylabel = 'Act Bias', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/abcomp_plot")
    plot_the_thing(gbreal*np.pi/180.0,gbest*np.pi/180.0,title = "Actual v Estimated Gyro Bias",xlabel='Time (s)',ylabel = 'Gyro Bias (deg/s)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/gbcomp_plot")
    plot_the_thing(mbreal,mbest,title = "Actual v Estimated MTM Bias",xlabel='Time (s)',ylabel = 'MTM Bias (scaled nT)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/mbcomp_plot")
    plot_the_thing(sbreal,sbest,title = "Actual v Estimated Sun Bias",xlabel='Time (s)',ylabel = 'Sun Bias', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/sbcomp_plot")
    plot_the_thing(dpreal,dpest,title = "Actual v Estimated Dipole",xlabel='Time (s)',ylabel = 'Dipole (Am^2)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/dpcomp_plot")
    plot_the_thing(dpreal_nobb,dpest_nobb,title = "Actual v Estimated Dipole, w/o Bbody",xlabel='Time (s)',ylabel = 'Dipole (Am^2)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/dpcomp_nobb_plot")

    plot_the_thing(genreal,genest,title = "Actual v Estimated Gen Torq",xlabel='Time (s)',ylabel = 'Gen Torq (Nm)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/gentorqcomp_plot")
    plot_the_thing(propreal,propest,title = "Actual v Estimated Prop Torq",xlabel='Time (s)',ylabel = 'Prop Torq (Nm)', norm=False,act_v_est = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/proptorqcomp_plot")

    plot_the_thing(act_torq_hist,est_act_torq_hist,title = 'Actual v Estimated Act Torq ',xlabel = 'Time (s)',norm = False,act_v_est = True, ylabel = 'Act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/act_torq_comp_plot")
    plot_the_thing(dist_torq_hist,est_dist_torq_hist,title = 'Actual v Estimated Dist Torq ',xlabel = 'Time (s)',norm = False,act_v_est = True, ylabel = 'Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/dist_torq_comp_plot")
    plot_the_thing(dist_torq_hist+act_torq_hist,est_dist_torq_hist+est_act_torq_hist,title = 'Actual v Estimated Combo Torq ',xlabel = 'Time (s)',norm = False,act_v_est = True, ylabel = 'Combo Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/combo_torq_comp_plot")


    plot_the_thing(est_act_torq_hist-act_torq_hist,title = 'act Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'act Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/act_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_act_torq_hist-act_torq_hist)),title = 'Log act Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log act Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_act_torq_err_plot")
    plot_the_thing(est_dist_torq_hist+est_act_torq_hist-dist_torq_hist-act_torq_hist,title = 'Combo Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'Combo Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/combo_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_dist_torq_hist+est_act_torq_hist-dist_torq_hist-act_torq_hist)),title = 'Log Combo Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log Combo Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_combo_torq_err_plot")
    plot_the_thing(est_dist_torq_hist-dist_torq_hist,title = 'Dist Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'Dist Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/dist_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_dist_torq_hist-dist_torq_hist)),title = 'Log Dist Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log Dist Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_dist_torq_err_plot")
    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/ang_plot")

    plot_the_thing(abdiff,title = "Actuator Bias Error",xlabel='Time (s)',ylabel = 'Actuator Bias Error ([units])', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/aberr_plot")
    plot_the_thing(gbdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/gberr_plot")
    plot_the_thing(mbdiff,title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/mberr_plot")
    plot_the_thing(sbdiff,title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/sberr_plot")
    plot_the_thing(dpdiff,title = "Dipole Error",xlabel='Time (s)',ylabel = 'Dipole Error (Am^2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/dipole_err_plot")
    plot_the_thing(gendiff,title = "Gen Torq Error",xlabel='Time (s)',ylabel = 'Gen Torq Error (Nm)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/gentorq_err_plot")
    plot_the_thing(dpdiff_nobb,title = "Dipole Error w/o Bbody",xlabel='Time (s)',ylabel = 'Dipole Error (Am^2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/dipole_err_nobb_plot")

    plot_the_thing(propdiff,title = "Prop Torq Error",xlabel='Time (s)',ylabel = 'Prop Torq Error (Nm)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/proptorq_err_plot")

    plot_the_thing(np.log10(matrix_row_norm(dpdiff)),title = "Log Dipole Error",xlabel='Time (s)',ylabel = 'Log Dipole Error (log (Am^2))',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_dipole_plot")
    plot_the_thing(np.log10(matrix_row_norm(dpdiff_nobb)),title = "Log Dipole Error w/o Bbody",xlabel='Time (s)',ylabel = 'Log Dipole Error (log (Am^2))',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_dipole_nobb_plot")

    plot_the_thing(np.log10(matrix_row_norm(gendiff)),title = "Log Gen Torq Error",xlabel='Time (s)',ylabel = 'Log Gen Torq Error (log (Nm))',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_gentorq_plot")
    plot_the_thing(np.log10(matrix_row_norm(propdiff)),title = "Log Prop Torq Error",xlabel='Time (s)',ylabel = 'Log Prop Torq Error (log (Nm))',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_proptorq_plot")

    plot_the_thing(np.log10(matrix_row_norm(abdiff)),title = "Log Actuator Bias Error",xlabel='Time (s)',ylabel = 'Log Actuator Bias Error (log [units]])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logab_plot")
    plot_the_thing(np.log10(matrix_row_norm(gbdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_loggb_plot")
    plot_the_thing(np.log10(matrix_row_norm(mbdiff)),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logmb_plot")
    plot_the_thing(np.log10(matrix_row_norm(sbdiff)),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logsb_plot")

    plot_the_thing(np.stack([orb_est_hist[j].R - orb_hist[j].R for j in range(len(orb_hist))]),title = "Error Estimated R ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Error Estimated R ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/rECI_est_err_plot")
    plot_the_thing(np.array([np.log10(norm(orb_est_hist[j].R - orb_hist[j].R)) for j in range(len(orb_hist))]),title = "Log Error Estimated R ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Log Error Estimated R ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/rECI_est_logerr_plot")
    plot_the_thing(np.stack([orb_est_hist[j].V - orb_hist[j].V for j in range(len(orb_hist))]),title = "Error Estimated V ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Error Estimated V ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/vECI_est_err_plot")
    plot_the_thing(np.array([np.log10(norm(orb_est_hist[j].V - orb_hist[j].V)) for j in range(len(orb_hist))]),title = "Log Error Estimated V ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Log Error Estimated V ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/vECI_est_logerr_plot")


    plot_the_thing(expected_clean_gsens*180.0/math.pi + gbest - (sens_hist[:,g_sens_inds]*180.0/math.pi),title = "Actual v Expected Gyro Sensor",xlabel='Time (s)',norm = True,ylabel = 'Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/diff_gsens_plot")
    plot_the_thing(expected_clean_gsens*180.0/math.pi - (sens_hist[:,g_sens_inds]*180.0/math.pi - gbreal),title = "Actual v Expected Unbiased Gyro Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/diff_gsens_unbiased_plot")
    plot_the_thing(expected_clean_msens + mbest - (sens_hist[:,m_sens_inds]),title = "Actual v Expected MTM Sensor",xlabel='Time (s)',norm = True,ylabel = 'MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/diff_msens_plot")
    plot_the_thing(expected_clean_msens - (sens_hist[:,m_sens_inds] - mbreal),title = "Actual v Expected Unbiased MTM Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/diff_msens_unbiased_plot")
    plot_the_thing(expected_clean_ssens + sbest - (sens_hist[:,s_sens_inds]),title = "Actual v Expected Sun Sensor",xlabel='Time (s)',norm = True,ylabel = 'Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/diff_ssens_plot")
    plot_the_thing(expected_clean_ssens - (sens_hist[:,s_sens_inds] - sbreal),title = "Actual v Expected Unbiased Sun Sensor",xlabel='Time (s)',norm = True,ylabel = 'Unbiased Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/diff_ssens_unbiased_plot")


    plot_the_thing(expected_clean_gsens*180.0/math.pi + gbest, (sens_hist[:,g_sens_inds]*180.0/math.pi),title = "Actual v Expected Gyro Sensor",xlabel='Time (s)',norm = False,act_v_est = True,ylabel = 'Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/comp_gsens_plot")
    plot_the_thing(expected_clean_gsens*180.0/math.pi, (sens_hist[:,g_sens_inds]*180.0/math.pi - gbreal),title = "Actual v Expected Unbiased Gyro Sensor",xlabel='Time (s)',norm = False,act_v_est = True,ylabel = 'Unbiased Gyro Sensing (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/comp_gsens_unbiased_plot")
    plot_the_thing(expected_clean_msens + mbest, (sens_hist[:,m_sens_inds]),title = "Actual v Expected MTM Sensor",xlabel='Time (s)',norm = False,act_v_est = True,ylabel = 'MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/comp_msens_plot")
    plot_the_thing(expected_clean_msens, (sens_hist[:,m_sens_inds] - mbreal),title = "Actual v Expected Unbiased MTM Sensor",xlabel='Time (s)',norm = False,act_v_est = True,ylabel = 'Unbiased MTM Sensing (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/comp_msens_unbiased_plot")
    plot_the_thing(expected_clean_ssens + sbest, (sens_hist[:,s_sens_inds]),title = "Actual v Expected Sun Sensor",xlabel='Time (s)',norm = False,act_v_est = True,ylabel = 'Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/comp_ssens_plot")
    plot_the_thing(expected_clean_ssens, (sens_hist[:,s_sens_inds] - sbreal),title = "Actual v Expected Unbiased Sun Sensor",xlabel='Time (s)',norm = False,act_v_est = True,ylabel = 'Unbiased Sun Sensing ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/comp_ssens_unbiased_plot")





    #actual v goal
    ctrlangdiff = np.stack([(180/np.pi)*np.arccos(-1 + 2*np.clip(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7]),-1,1)**2.0)  for j in range(state_hist.shape[0])])
    ctrlquatdiff = np.stack([quat_mult(quat_inv(goal_hist[j].state[3:7]),state_hist[j,3:7])*np.sign(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    ctrlmrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(ctrlquatdiff[j,:],0) for j in range(quatdiff.shape[0])]))
    plot_the_thing(ctrlangdiff,title = "Ctrl Angular Error",xlabel='Time (s)',ylabel = 'Ctrl Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/ctrlang_plot")
    plot_the_thing(ctrlmrpdiff,title = "Ctrl MRP Error",xlabel='Time (s)',ylabel = 'Ctrl MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/ctrlmrp_plot")
    plot_the_thing(ctrlquatdiff,title = "Ctrl Quat Error",xlabel='Time (s)',ylabel = 'Ctrl Quat Error',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/ctrlquat_plot")
    plot_the_thing(np.log10(ctrlangdiff),title = "Log Ctrl Angular Error",xlabel='Time (s)',ylabel = 'Ctrl Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/_logctrlang_plot")

    goalav_body = np.stack([goalav_ECI[j,:]@rot_mat(state_hist[j,3:7]) for j in range(len(goal_hist))])
    plot_the_thing(goalav_body-state_hist[:,0:3]*180.0/math.pi,title = "Goal AV err (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV err (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/goalaverr_body_plot")
    plot_the_thing(goalav_body,title = "Goal AV (body frame) (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (body frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/goalav_body_plot")


    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(state_hist[j,3:7])),normalize(ctrlquatdiff[j,1:])))) for j in range(len(orb_hist))]),title = "Ang between B body and angular ctrl error to goal",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/b_q_ctrlang_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(state_hist[j,3:7])),normalize(state_hist[j,:3])))) for j in range(len(orb_hist))]),title = "Ang between B body and av ctrl error to goal",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/b_w_ctrlang_plot")

    pt_err = np.arccos(np.array([np.dot(point_vec_eci[j,:],goal_vec_eci[j,:]) for j in range(point_vec_body.shape[0])]))*180.0/np.pi
    plot_the_thing(pt_err,title = "Angular Error between Pointing and Goal Vectors",xlabel='Time (s)',norm = False,ylabel = 'Angle (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/vecang")
    plot_the_thing(np.log10(pt_err),title = "Log10 Angular Error between Pointing and Goal Vectors",xlabel='Time (s)',norm = False,ylabel = 'Angle (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/_logvecang")

    #plan v goal
    plannedangdiff = np.stack([(180/np.pi)*np.arccos(-1 + 2*np.clip(np.dot(goal_hist[j].state[3:7],plan_state_hist[j,3:7]),-1,1)**2.0)  for j in range(plan_state_hist.shape[0])])
    plannedquatdiff = np.stack([quat_mult(quat_inv(goal_hist[j].state[3:7]),plan_state_hist[j,3:7])*np.sign(np.dot(goal_hist[j].state[3:7],plan_state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    plannedmrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(plannedquatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    plot_the_thing(plannedmrpdiff,title = "Planned MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_goal/plannedmrp_plot")
    plot_the_thing(np.log10(plannedangdiff),title = "Log Planned Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_goal/_logplannedang_plot")
    plot_the_thing(plannedangdiff,title = "Planned Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_goal/plannedang_plot")
    plot_the_thing(plan_alignment_rot_ax_eci,title = "Planned goal alignment axis ECI",xlabel='Time (s)',ylabel = 'axis*sin(err)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_goal/planned_eci_alignment_axis_plot")
    plot_the_thing(plan_alignment_rot_ax_body,title = "Planned goal alignment axis Body",xlabel='Time (s)',ylabel = 'axis*sin(err)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_goal/planned_body_alignment_axis_plot")

    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(plan_state_hist[j,3:7])),normalize(ctrlquatdiff[j,1:])))) for j in range(len(orb_hist))]),title = "Ang between B body and angular ctrl error to goal",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/b_q_ctrlang_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(plan_state_hist[j,3:7])),normalize(plan_state_hist[j,:3])))) for j in range(len(orb_hist))]),title = "Ang between B body and av ctrl error to goal",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual_v_goal/b_w_ctrlang_plot")
    planpt_err = np.arccos(np.array([np.dot(plan_point_vec_eci[j,:],goal_vec_eci[j,:]) for j in range(point_vec_body.shape[0])]))*180.0/np.pi
    plot_the_thing(planpt_err,title = "Planned Angular Error between Pointing and Goal Vectors",xlabel='Time (s)',norm = False,ylabel = 'Angle (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_goal/planvecang")
    plot_the_thing(np.log10(planpt_err),title = "Log10 Angular Error between Pointing and Goal Vectors",xlabel='Time (s)',norm = False,ylabel = 'Angle (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_goal/_logplanvecang")
    goalav_body_planned = np.stack([goalav_ECI[j,:]@rot_mat(plan_state_hist[j,3:7]) for j in range(len(goal_hist))])


    #plan v actual
    angdiff_from_plan = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(state_hist[:,3:7]*plan_state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff_from_plan = np.stack([quat_mult(quat_inv(plan_state_hist[j,3:7]),state_hist[j,3:7])*np.sign(np.dot(plan_state_hist[j,3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    mrpdiff_from_plan = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff_from_plan[j,:],0) for j in range(quatdiff.shape[0])]))
    avdiff_from_plan = (state_hist[:,0:3] - plan_state_hist[:,0:3])*180.0/math.pi

    plot_the_thing(angdiff_from_plan,title = "Angular Error from Plan",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/ang_from_plan_plot")
    plot_the_thing(quatdiff_from_plan,title = "Quaternion Error from Plan",xlabel='Time (s)',ylabel = 'Quaternion Error',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/quaterr_from_plan_plot")

    plot_the_thing((state_hist[:,0:3]-plan_state_hist[:,0:3])*180.0/math.pi,title = "AV Error From Plan (body to body)",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/averr_from_plan_plot")
    plot_the_thing((av_ECI_planbody-plan_state_hist[:,0:3]*180.0/math.pi),title = "AV Error From Plan (all in planned body frame)",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/averr_from_plan_plan_body_frame_plot")
    plot_the_thing((state_hist[:,0:3]*180.0/math.pi-plannedav_realbody),title = "AV Error From Plan (all in actual body frame)",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/averr_from_plan_actual_body_frame_plot")
    plot_the_thing((av_ECI-plannedav_ECI),title = "AV Error From Plan (ECI to ECI)",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/averr_from_plan_ECI_plot")

    plot_the_thing(mrpdiff_from_plan,title = "MRP Error from Plan",xlabel='Time (s)',ylabel = 'Ctrl MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/mrp_from_plan_plot")
    plot_the_thing(np.log10(matrix_row_norm((plan_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi)),title = "Log Planned AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/_logav_from_plan_plot")
    plot_the_thing(np.log10(angdiff_from_plan),title = "Log Angular Error from Plan",xlabel='Time (s)',ylabel = ' Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/_logang_from_plan_plot")

    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(state_hist[j,3:7])),normalize(quatdiff_from_plan[j,1:])))) for j in range(len(orb_hist))]),title = "Ang between B body and angular ctrl error to plan",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/b_q_ang_from_plan_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(normalize(orb_hist[j].B@rot_mat(state_hist[j,3:7])),normalize(avdiff_from_plan[j,:])))) for j in range(len(orb_hist))]),title = "Ang between B body and av ctrl error to plan",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan_v_actual/b_w_ang_from_plan_plot")


    #estimated v plan
    estangdiff_from_plan = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*plan_state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    estquatdiff_from_plan =  np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),plan_state_hist[j,3:7])*np.sign(np.dot(est_state_hist[j,3:7],plan_state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    estmrpdiff_from_plan = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(estquatdiff_from_plan[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    estavdiff_from_plan =  (est_state_hist[:,0:3] - plan_state_hist[:,0:3])*180.0/math.pi

    plot_the_thing(estangdiff_from_plan,title = "Est Angular Error from Plan",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_plan/estang_from_plan_plot")
    plot_the_thing(estmrpdiff_from_plan,title = "Est MRP Error from Plan",xlabel='Time (s)',ylabel = 'Ctrl MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_plan/estmrp_from_plan_plot")
    plot_the_thing(estavdiff_from_plan,title = "Est AV Error from Plan",xlabel='Time (s)',ylabel = 'Ctrl AV Error (deg/s)',norm = True, xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_plan/estav_from_plan_plot")

    #estimated v goal




















    metrics = find_metrics(t_hist,angdiff)
    ctrlmetrics = find_metrics(t_hist,ctrlangdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
    sim.goal_hist = goal_hist
    sim.orb_hist = orb_hist
    sim.cov_hist = cov_hist
    sim.kap = adcsys.estimator.kap
    sim.al = adcsys.estimator.al
    # sim.cov_est0 = cov_estimate.copy()
    # sim.int_cov = int_cov.copy()
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
    sim.plan_state_hist = plan_state_hist
    sim.plan_control_hist = plan_control_hist
    sim.plan_dist_torq_hist = plan_dist_torq_hist
    sim.int_err_hist = int_err_hist
    # try:
    with open("thesis_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)

    #generate statistics
    labels = ["title","al","kap","bet","dt"]+["ctrl conv time","ctrl tc","last 100 ctrlang err mean","last 100 ctrlang err max"]
    info = [base_title,adcsys.estimator.al,adcsys.estimator.kap,adcsys.estimator.bet,dt]+[ctrlmetrics.time_to_conv,ctrlmetrics.tc_est,np.mean(ctrlangdiff[-100:]),np.amax(ctrlangdiff[-100:])]
    with open("thesis_test_files/"+base_title+"/info", 'w') as f:
        for j in range(len(labels)):
            f.write(labels[j])
            f.write(": ")
            f.write(str(info[j]))
            f.write("\n")



def run_sim_wrapper(title,est_sat,real_sat,dt,w0,q0,h0,cov_estimate,dist_control,goals,prop_schedule,tf,orb_file,base_int_cov,att_sen_bias_len,dp_len,act_bias_len,al=1e-3,kap = 0,quatset_type="",estimated_int_noise_scale=0.0,estimated_int_noise_other_scale=1.0,est_w0 = np.zeros(3),est_q0 = unitvecs4[0], est_h0 = None):

    if est_h0 is None:
        est_h0 = np.zeros(est_sat.state_len-7)
    if title is None:
        base_title = "baseline_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        base_title = title+"_"+time.strftime("%Y%m%d-%H%M%S")

    os.mkdir("thesis_test_files/"+base_title)
    try:


        if np.any([j.estimated_param for j in est_sat.disturbances]):
            dist_ic = block_diag(*[j.std**2.0 for j in est_sat.disturbances if j.estimated_param])
        else:
            dist_ic = np.zeros((0,0))

        int_cov = dt*block_diag(base_int_cov,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),dist_ic)

        est_sat = copy.deepcopy(est_sat)
        estimate = np.zeros(est_sat.state_len+act_bias_len+att_sen_bias_len+dp_len)
        estimate[0:3] = est_w0
        estimate[3:7] = est_q0
        estimate[7:est_sat.state_len] = est_h0
        # breakpoint()
        # print(int_cov.shape)
        # print(estimate.shape)
        # print(est_sat.state_len,est_sat.act_bias_len,est_sat.att_sens_bias_len,est_sat.dist_param_len)
        # print(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]).shape,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]).shape,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]).shape,dist_ic.shape)

        # print('cov @ 0',0.5*np.log10(np.diag(cov_estimate)))
        est = SRUKF(np.nan,est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True,verbose = False)
        # est = PerfectEstimatoGr(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True)
        if kap is None:
            kap = 3 - (estimate.size - 1 + 0*sum([j.use_noise for j in real_sat.actuators]))
        est.use_cross_term = True
        est.bet = 2.0
        est.al = al#1.0#1e-3#1.0#1.0#1.0#1.0#1e-3#e-3#1#1e-1#al# 1e-1
        est.kap =  kap#3 - (estimate.size - 1 + 0*sum([j.use_noise for j in real_sat.actuators]))
        est.include_int_noise_separately = False
        est.include_sens_noise_separately = False
        est.use_estimated_int_noise = bool(estimated_int_noise_scale != 0)
        est.estimated_int_noise_scale = estimated_int_noise_scale
        est.estimated_int_noise_other_scale = estimated_int_noise_other_scale
        est.scale_nonseparate_adds = False
        est.included_int_noise_where = 2




        mpc_dt = 1
        mpc_ang_limit = 5
        mpc_angwt_low = 1e12#1e10#e0#1e6#1e4#1e4#1e4
        mpc_angwt_high = 1e14#1e12#1e5#1e12
        mpc_avwt = 1e7#1e3#1e8 #1e2
        mpc_avangwt = 1e9#1e14#1e8#1e8#1e0#1e0#1e6#1e2
        mpc_extrawt = 0
        mpc_uwt_from_plan = 1e3#$0.0#1e1#0#1e-4#1e-6
        mpc_uwt_from_prev = 1e0#0.0#1e0#1e0#1e5
        mpc_lqrwt_mult = 0.0#1e-12#1e4#0.0#1.0
        mpc_extra_tests = 0
        mpc_tol = 1e0
        mpc_Nplot = 0
        mpc_gain_info = [mpc_dt,mpc_ang_limit,mpc_angwt_low,mpc_angwt_high,mpc_avwt,mpc_avangwt,mpc_extrawt,mpc_uwt_from_plan,mpc_uwt_from_prev,mpc_lqrwt_mult,mpc_extra_tests,mpc_tol,mpc_Nplot]#[1,10,100,1e6,1,0,1e-6,0]
        # tests = tests_baseline[1:3] + tests_disturbed[1:3] + tests_ctrl[1:3] + tests_genctrl[1:3] + tests_cubesat
        # breakpoint()
        # print('cov @ 1',0.5*np.log10(np.diag(est.use_state.cov)))
        # est = PerfectEstimator(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True)

        if any([isinstance(j,MTQ) for j in real_sat.actuators ]) and any([isinstance(j,RW) for j in real_sat.actuators ]):
            control_laws =  [NoControl(est.sat),Bdot(1e8,est.sat),RWBdot(est.sat,gain=0.005,include_disturbances = True),BdotEKF(0.05,est.sat,maintain_RW = False,include_disturbances=True),Lovera([0.01,50,50],est.sat,include_disturbances=dist_control,quatset=(len(quatset_type)>0),quatset_type = quatset_type,calc_av_from_quat = True),MTQ_W_RW_PD([np.eye(3)*0.025,np.eye(3)*0.005,0.005],est.sat),WisniewskiSliding([np.eye(3)*0.002,np.eye(3)*0.003],est.sat,include_disturbances=dist_control,calc_av_from_quat = True,quatset=(len(quatset_type)>0),quatset_type = quatset_type),TrajectoryMPC(mpc_gain_info,est.sat),Trajectory_Mag_PD([0.01,50,50],est.sat,include_disturbances=dist_control,calc_av_from_quat = True),TrajectoryLQR([],est.sat)]

        elif isinstance(real_sat.actuators[0],MTQ):
            control_laws =  [NoControl(est.sat),Bdot(1e8,est.sat),BdotEKF(1e4,est.sat,maintain_RW = True,include_disturbances=True),Lovera([0.01,50,50],est.sat,include_disturbances=dist_control,quatset=(len(quatset_type)>0),quatset_type = quatset_type,calc_av_from_quat = True),WisniewskiSliding([np.eye(3)*0.002,np.eye(3)*0.003],est.sat,include_disturbances=dist_control,calc_av_from_quat = True,quatset=(len(quatset_type)>0),quatset_type = quatset_type),TrajectoryMPC(mpc_gain_info,est.sat),Trajectory_Mag_PD([0.01,50,50],est.sat,include_disturbances=dist_control,calc_av_from_quat = True),TrajectoryLQR([],est.sat)]

        elif isinstance(real_sat.actuators[0],Magic):
            control_laws =  [NoControl(est.sat),Magic_PD([np.eye(3)*20,np.eye(3)*5.0],est.sat,include_disturbances=dist_control),MagicBdot(est.sat,gain=0.05),TrajectoryMPC(mpc_gain_info,est.sat),TrajectoryLQR([],est.sat)]

        else:
            control_laws =  [NoControl(est.sat),RW_PD([np.eye(3)*20,np.eye(3)*5.0],est.sat,include_disturbances=dist_control),RWBdot(est.sat,gain=0.05),TrajectoryMPC(mpc_gain_info,est.sat),TrajectoryLQR([],est.sat)]

        orbit_estimator = OrbitalEstimator(est_sat,state_estimate = Orbital_State(0,np.ones(3),np.ones(3)))
        # adcsys = ADCS(est_sat,orbit_estimator,est,control_laws,use_planner = False,planner = None,planner_settings = None,goals=goals,prop_schedule=None,dt = 1,control_dt = None,tracking_LQR_formulation = 0)

        adcsys = ADCS(est_sat,orbit_estimator,est,control_laws,use_planner = True,planner = None,planner_settings = None,goals=goals,prop_schedule=prop_schedule,dt = 1,control_dt = 1,tracking_LQR_formulation = 0)
        # adcsys.planner.setVerbosity(True)
        # breakpoint()
        state0 = np.zeros(real_sat.state_len)
        state0[0:3] = w0
        state0[3:7] = q0
        state0[7:] = h0
        real_sat.update_RWhs(state0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*TimedeltaIndex")
            run_sim(orb_file,state0,copy.deepcopy(real_sat),adcsys,base_title,tf=tf,dt = dt,rand=False)
    except Exception as e:
        if isinstance(e, KeyboardInterrupt) or isinstance(e,BdbQuit):
            raise
        tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        with open("thesis_test_files/"+base_title+"/error.txt", 'w') as f:
            f.write(tb_str)
