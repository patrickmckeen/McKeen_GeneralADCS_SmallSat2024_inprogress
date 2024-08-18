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
                res.tc_est = np.polyfit(log_angdiff,log_t_list,1)[0]
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
    os0 = Orbital_State(0.22-1*sec2cent,np.array([1.0,0,0])*(alt+R_e),v*np.array([0,math.cos(inc),math.sin(inc)]))
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
    os0 = Orbital_State(0.22-1*sec2cent,np.array([1.0,0,0])*(alt+R_e),v*np.array([0,math.cos(inc),math.sin(inc)]))
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

    wa,va = np.linalg.eigh(real_sat.J)
    # print(va)
    # print(wa)
    principal_axes = va

    # w0 = np.cross(w0,rot_mat(state[3:7]).T@normalize(orbt.B))
    # state0[0:3] = w0
    #
    # breakpoint()

    # breakpoint()
    try:
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
            print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)
            # print('av ECI',rot_mat(state[3:7])@state[0:3])


            mode = goals.get_control_mode(orbt.J2000)
            ctrlr = [j for j in control_laws if j.modename==mode][0]
            goal_state = goals.get_pointing_info(orbt)
            # breakpoint()next_goal_stat
            nextorb = orb.get_os(j2000_0+(dt+t-t0)*sec2cent)
            next_goal_state = goals.get_pointing_info(nextorb)

            # print('xgoal in ECI',unitvecs[0]@rot_mat(goal_state.state[3:7]).T)
            # print('ygoal in ECI',unitvecs[1]@rot_mat(goal_state.state[3:7]).T)
            # print('zgoal in ECI',unitvecs[2]@rot_mat(goal_state.state[3:7]).T)
            # print('zen in goal',normalize(orbt.R)@rot_mat(goal_state.state[3:7]))
            # print('ram in goal',normalize(orbt.V)@rot_mat(goal_state.state[3:7]))
            # print('normal in goal',normalize(np.cross(orbt.R,orbt.V))@rot_mat(goal_state.state[3:7]))
            prev_control = control.copy()
            control = ctrlr.find_actuation(est_state,orbt,nextorb,goal_state,[],next_goal_state,sens,[],False)
            print('quat',state[3:7])#,norm(q_err[1:]),norm(np.cross(q_err[1:],est.os_vecs['b']))/norm(est.os_vecs['b']))
            print('goalquat',goal_state.state[3:7])

            state_err = ctrlr.state_err(state,goal_state,print_info=False)
            print('ctrlquaterr ',state_err[3:7])
            est_state_err = ctrlr.state_err(est_state,goal_state,print_info=False)
            print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(goal_state.state[3:7],state[3:7]),-1,1)**2.0 ))
            # print('cmd ',control)
            print('cmd ',control+np.array([j.bias[0]*j.has_bias for j in real_sat.actuators]))
            # print('ctrl',control+np.array([j.noise+j.bias[0] for j in real_sat.actuators]))
            # print('noise',np.array([j.noise for j in real_sat.actuators]))
            # print('abias',np.array([j.bias [0]for j in real_sat.actuators]))
            # print('abias est',np.array([j.bias[0] for j in est.sat.actuators]))
            # print('abias err',np.array([j.bias[0] for j in est.sat.actuators])-np.array([j.bias[0] for j in real_sat.actuators]),norm(np.array([j.bias[0] for j in est.sat.actuators])-np.array([j.bias[0] for j in real_sat.actuators])))

            print('av',state[0:3]*180.0/math.pi,norm(state[0:3])*180.0/math.pi,state[0:3],norm(state[0:3]))#,(180.0/math.pi)*norm(np.cross(state[0:3],est.os_vecs['b']))/norm(est.os_vecs['b']))
            print('averr',state_err[0:3]*180.0/math.pi,norm(state_err[0:3])*180.0/math.pi,state_err[0:3],norm(state_err[0:3]))#,(180.0/math.pi)*norm(np.cross(state[0:3],est.os_vecs['b']))/norm(est.os_vecs['b']))

            # print('dists')

            # print(goal_state.state)
            # print(goal_state.state)
            # print(orbt.in_eclipse())

            # print('dist ',real_sat.last_dist_torq,norm(real_sat.last_dist_torq))
            # for j in range(len(real_sat.disturbances)):
            #     print('   dist ',j,real_sat.disturbances[j])
            #     print(real_sat.last_dist_list[j],norm(real_sat.last_dist_list[j]))
            #     print(est.sat.last_dist_list[j],norm(est.sat.last_dist_list[j]))
            #     print(real_sat.last_dist_list[j]-est.sat.last_dist_list[j],norm(real_sat.last_dist_list[j]-est.sat.last_dist_list[j]))

            # print('dipole ',real_sat.disturbances[2].main_param,norm(real_sat.disturbances[2].main_param))
            # print('dipole est ',est.sat.disturbances[2].main_param,norm(est.sat.disturbances[2].main_param))
            # print('dipole err ', real_sat.disturbances[2].main_param-est.sat.disturbances[2].main_param,norm(real_sat.disturbances[2].main_param-est.sat.disturbances[2].main_param))
            # disttorqest = est.sat.dist_torque(est_state[:est.sat.state_len],est.os_vecs).copy()
            # disttorqestsave = ctrlr.saved_dist
            # print('dist est ',disttorqest,norm(disttorqest))
            # print('dist est save ',disttorqestsave,norm(disttorqestsave))
            # # print('dist err',real_sat.last_dist_torq - disttorqest,norm(real_sat.last_dist_torq - disttorqest))
            # print('dist err save',real_sat.last_dist_torq - disttorqestsave,norm(real_sat.last_dist_torq - disttorqestsave))
            print('act ',real_sat.last_act_torq,norm(real_sat.last_act_torq))
            print('ang from qerr',(180.0/np.pi)*np.arccos(np.dot(real_sat.last_act_torq+real_sat.last_dist_torq,state_err[4:7])/norm(real_sat.last_act_torq+real_sat.last_dist_torq)/norm(state_err[4:7])))
            print('ang from werr',(180.0/np.pi)*np.arccos(np.dot(real_sat.last_act_torq+real_sat.last_dist_torq,state_err[0:3])/norm(real_sat.last_act_torq+real_sat.last_dist_torq)/norm(state_err[0:3])))

            # print(ctrlr)
            # if isinstance(ctrlr,WisniewskiSliding):
            #     s = state_err[0:3]@real_sat.J@ctrlr.params.Lambda_All + np.sign(state_err[3])*state_err[4:7]@ctrlr.params.Lambda_q
            #     print('s',s)
            #     sdot = -np.cross(state[0:3],state[0:3]@real_sat.J)@ctrlr.params.Lambda_All + (real_sat.last_act_torq + real_sat.last_dist_torq)@ctrlr.params.Lambda_All + np.sign(state_err[3])*0.5*state_err[0:3]@Wmat(state_err[3:7]).T@quat2vec_mat.T@ctrlr.params.Lambda_q - np.cross(state[0:3],state_err[0:3])@real_sat.J@ctrlr.params.Lambda_All
            #
            #     print('sdot',sdot)
            #     print('snorm2', 0.5*np.dot(s,s))
            #     print('snorm2dot',np.dot(s,sdot))
            #     print("s ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(s)))))
            #     print("sdot ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(sdot)))))

            #     print('snorm2dot breakout',np.array([[np.dot(state[0:3]@real_sat.J, -np.cross(state[0:3],state[0:3]@real_sat.J)),np.dot(state[0:3]@real_sat.J,real_sat.last_act_torq),np.dot(state[0:3]@real_sat.J,real_sat.last_dist_torq),np.dot(state[0:3]@real_sat.J,np.sign(state[3])*0.5*state[0:3]@Wmat(state[3:7]).T@quat2vec_mat.T@ctrlr.params.Lambda_q)],
            #                                         [np.dot(np.sign(state[3])*state[4:7]@ctrlr.params.Lambda_q, -np.cross(state[0:3],state[0:3]@real_sat.J)),np.dot(np.sign(state[3])*state[4:7]@ctrlr.params.Lambda_q,real_sat.last_act_torq),np.dot(np.sign(state[3])*state[4:7]@ctrlr.params.Lambda_q,real_sat.last_dist_torq),np.dot(np.sign(state[3])*state[4:7]@ctrlr.params.Lambda_q,np.sign(state[3])*0.5*state[0:3]@Wmat(state[3:7]).T@quat2vec_mat.T@ctrlr.params.Lambda_q)]]))
                # print(0.5*state[0:3]@Wmat(state[3:7]).T)
                # print(0.5*quat_mult(state[3:7],np.concatenate([[0],state[0:3]])))
            if np.any([isinstance(j,General_Disturbance) for j in est_sat.disturbances]):
                print('dist gen dist est ',est_state[-3:],norm(est_state[-3:]))
                print('dist est ',est_sat.last_dist_torq,norm(est_sat.last_dist_torq))
                print('dist real ',real_sat.last_dist_torq,norm(real_sat.last_dist_torq))
                print(real_sat.last_dist_torq - est_sat.last_dist_torq,norm(real_sat.last_dist_torq - est_sat.last_dist_torq))
            # print('act ',real_sat.last_act_torq)
            # print("B real", real_vecs["b"])
            # print("B est", Bbody)
            # print("B should meas",real_vecs["b"]*real_sat.sensors[0].scale + np.hstack([j.bias for j in real_sat.sensors[0:3]]))
            # print("B meas", sens[0:3])
            # print("B est meas", Bbody*est_sat.sensors[0].scale + est_state[10:13])
            # print(va[:,-1],np.sqrt(wa[-1]))
            # print("cov ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(Bbody,va[:,-1]))/norm(Bbody)))

            # print(real_sat.sensors[0].bias[0],est.sat.sensors[0].bias[0]-real_sat.sensors[0].bias[0])
            # print([j.bias for j in real_sat.actuators if j.has_bias])
            # print([j.bias for j in est.sat.actuators if j.has_bias])
            # gb_real = np.concatenate([j.bias for j in real_sat.sensors if isinstance(j,Gyro)])
            # gb_est = np.concatenate([j.bias for j in est.sat.sensors if isinstance(j,Gyro)])
            # print(gb_real,norm(gb_real))
            # print(gb_est,norm(gb_est))
            # print(gb_est-gb_real,norm(gb_est-gb_real))
            # # print(est_state[7:10],norm(est_state[7:10]))
            # print(np.concatenate([j.bias for j in real_sat.sensors[3:6] if j.has_bias])-est_state[7:10],norm(np.concatenate([j.bias for j in real_sat.sensors[3:6] if j.has_bias])-est_state[7:10]))
            # print(np.concatenate([j.bias for j in real_sat.sensors[6:] if j.has_bias])-est_state[10:13],norm(np.concatenate([j.bias for j in real_sat.sensors[6:] if j.has_bias])-est_state[10:13]),norm(np.concatenate([j.bias for j in real_sat.sensors[6:] if j.has_bias])))
            # print([j.bias for j in real_sat.sensors[3:6] if j.has_bias])
            # print(est.use_state.val[7:10])

            #save info
            state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
            est_state_hist[ind,:] = est.use_state.val
            goal_hist += [goal_state.copy()]
            # print(goal_state.state[3:7])
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

            out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-8)#,jac = ivp_jac)
            real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False,save_details = True)
            est_sat.dynamics(est.use_state.val[:7],control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False,save_details = True)

            # wdot = (real_sat.last_act_torq + real_sat.last_dist_torq - np.cross(state[0:3],state[0:3]@real_sat.J))@real_sat.invJ
            # werrdot = (real_sat.last_act_torq + real_sat.last_dist_torq - np.cross(state[0:3],state[0:3]@real_sat.J))@real_sat.invJ - np.cross(state[0:3],state_err[0:3])
            #
            # # print('wdot',wdot)
            # # print('wnorm2 dot',np.dot(state[0:3],wdot))
            # print('werr norm2 dot',np.dot(state_err[0:3],werrdot))
            # # print('wJw dot',np.dot(state[0:3]@est.sat.J,wdot))
            # # print('wJw dot from dist torq',np.dot(state[0:3],real_sat.last_dist_torq))
            # # for j in range(len(real_sat.disturbances)):
            # #     print('   wjw dot dist ',j,real_sat.disturbances[j],np.dot(state[0:3],real_sat.last_dist_list[j]))
            # # print('wJw dot from act torq',np.dot(state[0:3],real_sat.last_act_torq))
            #
            # print('weJwe dot',np.dot(state_err[0:3]@est.sat.J,werrdot))
            # print('weJwe dot from dist torq',np.dot(state_err[0:3],real_sat.last_dist_torq))
            # # for j in range(len(real_sat.disturbances)):
            # #     print('   wejwe dot dist ',j,real_sat.disturbances[j],np.dot(state_err[0:3],real_sat.last_dist_list[j]))
            # print('weJwe dot from act torq',np.dot(state_err[0:3],real_sat.last_act_torq))
            # Bbody= rot_mat(state[3:7]).T@orbt.B
            # nBbody = normalize(Bbody)
            # nRbody = normalize(rot_mat(state[3:7]).T@orbt.R)
            # if isinstance(ctrlr,Lovera):
            #     Bbodyest = rot_mat(est_state[3:7]).T@orbt.B
            #     Bbodyest /= np.dot(Bbodyest,Bbodyest)
            #     ctrl_err_quat = quat_mult(quat_inv(goal_state.state[3:7]),est_state[3:7])
            #     ctrl_err_quat *= np.sign(ctrl_err_quat[0])
            #     dv_est = ctrl_err_quat[1:]
            #     print('weJwe dot from kv torq',-ctrlr.params.gain_eps*ctrlr.params.kv_gain*np.dot(np.cross(Bbody,state_err[0:3]),np.cross(Bbodyest,est_state_err[0:3]@est.sat.invJ)) )
            #     print('weJwe dot from kp torq',-ctrlr.params.gain_eps**2.0*ctrlr.params.kp_gain*np.dot(np.cross(Bbody,state_err[0:3]),np.cross(Bbodyest,dv_est@est.sat.invJ)) )
            #     # print('wJw dot from dist torq counter',-np.dot(np.cross(Bbody,state[0:3]),np.cross(Bbodyest,disttorqest)) )
            #     print('weJwe dot from dist torq counter save',-np.dot(np.cross(Bbody,state_err[0:3]),np.cross(Bbodyest,disttorqestsave)) )
            # print('qdot',0.5*state[0:3]@Wmat(state[3:7]).T)
            # print('qedot',0.5*state_err[0:3]@Wmat(state_err[3:7]).T)
            # # print("rot ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,np.sign(state_err[3])*normalize(state_err[4:7])))))
            # # print("av ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(state_err[0:3])))))
            # # print("Jav ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(state_err[0:3]@real_sat.J)))))
            # # # print("rot ax ang to Bbody,withJinv", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(real_sat.invJ@quat_mult(quat_inv(goal_state.state[3:7]),state[3:7])[1:])))))
            # # # print("av ang to Bbody,withJinv", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(real_sat.invJ@state[0:3])))))
            # # print("rot ax ang to av", (180.0/np.pi)*np.arccos(np.dot(normalize(state_err[0:3]),np.sign(state_err[3])*normalize(state_err[4:7]))))
            # # print("rot ax ang to Jav", (180.0/np.pi)*np.arccos(np.dot(normalize(state_err[0:3]@real_sat.J),np.sign(state_err[3])*normalize(state_err[4:7]))))
            #
            # if isinstance(ctrlr,WisniewskiSliding) or isinstance(ctrlr,WisniewskiTwisting)  or isinstance(ctrlr,WisniewskiTwisting2) or isinstance(ctrlr,WisniewskiSlidingMagic) or isinstance(ctrlr,WisniewskiTwistingMagic) :
            #     s = ctrlr.scalc(state_err[3:7]*np.sign(state_err[3]),state_err[0:3],state,do_print = False)
            #     print('s',s)
            #     sdot = ctrlr.sdotcalc_noact(state_err[3:7]*np.sign(state_err[3]),state_err[0:3],state,real_vecs,save=False,do_print = False)+real_sat.last_act_torq@ctrlr.params.Lambda_All
            #     print('sdot',sdot)
            #     print('snorm2', 0.5*np.dot(s,s))
            #     print('snorm2dot',np.dot(s,sdot))
            #     print('snorm2dot from atorq,dtorq',np.dot(s,(real_sat.last_act_torq)@ctrlr.params.Lambda_All),np.dot(s,(real_sat.last_dist_torq)@ctrlr.params.Lambda_All))
            #     print('snorm2dot from wxjw,wxwe',np.dot(s,-np.cross(state[0:3],state[0:3]@real_sat.J)@ctrlr.params.Lambda_All ),np.dot(s,- np.cross(state[0:3],state_err[0:3])@real_sat.J@ctrlr.params.Lambda_All))
            #     # print('snorm2dot from qdot',np.dot(s,np.sign(state_err[3])*0.5*state_err[0:3]@Wmat(state_err[3:7]).T@quat2vec_mat.T@ctrlr.params.Lambda_q@ctrlr.params.Lambda_All))
            #
            #     print("s ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(s)))))
            #     # print("sdot ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(sdot)))))
            #     Javerrdes = -(s@ctrlr.params.inv_Lambda_All-state_err[0:3]@est_sat.J)
            #     print("des averr for q ", Javerrdes@real_sat.invJ*(180.0/np.pi),norm(Javerrdes@real_sat.invJ*(180.0/np.pi)))
            #     print("real averr      ", state_err[0:3]*180.0/math.pi,norm(state_err[0:3]*180.0/math.pi))
            #
            #     print("des Javerr for q ",Javerrdes,norm(Javerrdes))
            #     print("real Javerr      ", state_err[0:3]@real_sat.J,norm(state_err[0:3]@real_sat.J))
            #     # cqvd =-state_err[0:3]@real_sat.J@np.linalg.inv(ctrlr.params.Lambda_q)
            #     # print("des cqv for av ", cqvd,(180.0/np.pi)*2*np.arccos(np.sqrt(1-np.dot(cqvd,cqvd))))
            #     # print(" real cqv      ",state_err[4:7]*np.sign(state_err[3]),(180.0/np.pi)*2*np.arccos(np.abs(state_err[3])))
            #
            #     print("principal axes alignment with zenith: ",[(180.0/np.pi)*np.arccos(np.dot(principal_axes[:,kk],nRbody)) for kk in range(3)])
            state = out.y[:,-1]
            state[3:7] = normalize(state[3:7])

            real_sat.update_actuator_noise()
            real_sat.update_actuator_biases(orbt.J2000)
            real_sat.update_sensor_biases(orbt.J2000)
            real_sat.update_disturbances(orbt.J2000)

    except KeyboardInterrupt:
        pass
    # breakpoint()
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    controlwb_hist = control_hist + state_hist[:,real_sat.state_len:real_sat.state_len+real_sat.control_len]

    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    ctrlangdiff = np.stack([(180/np.pi)*np.arccos(-1 + 2*np.clip(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7]),-1,1)**2.0)  for j in range(len(goal_hist))])
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7])*np.sign(np.dot(est_state_hist[j,3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    ctrlquatdiff = np.stack([quat_mult(quat_inv(goal_hist[j].state[3:7]),state_hist[j,3:7])*np.sign(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7])) for j in range(len(goal_hist))])

    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    ctrlmrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_mrp(ctrlquatdiff[j,:]) for j in range(ctrlquatdiff.shape[0])]))
    goalav_ECI = np.stack([goal_hist[j].state[0:3]@rot_mat(goal_hist[j].state[3:7]).T*180.0/np.pi for j in range(len(goal_hist))])
    goalav_body = np.stack([goalav_ECI[j,:]@rot_mat(state_hist[j,3:7]) for j in range(len(goal_hist))])

    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi

    # sens_est_biases = np.array([j.has_bias*j.output_len for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],SunSensorPair) and real_sat.attitude_sensors[j].has_bias ])
    # sens_biases = np.array([j.has_bias*j.output_len for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],SunSensorPair) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias])

    sens_s_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j]]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],SunSensorPair)])).astype('int')
    sens_m_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j]]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],MTM)])).astype('int')
    sens_g_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j]]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],Gyro)])).astype('int')

    est_sb_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],SunSensorPair) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias]) + est.sat.state_len+est.sat.act_bias_len).astype('int')
    est_mb_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],MTM) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias]) + est.sat.state_len+est.sat.act_bias_len).astype('int')
    est_gb_inds = (np.array([np.sum([k.output_length for k in est.sat.attitude_sensors[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.attitude_sensors)) if isinstance(est.sat.attitude_sensors[j],Gyro) and est.sat.attitude_sensors[j].has_bias and est.sat.attitude_sensors[j].estimated_bias]) + est.sat.state_len+est.sat.act_bias_len).astype('int')
    est_ab_inds = (np.array([np.sum([k.input_len for k in est.sat.actuators[:j] if k.estimated_bias and k.has_bias]) for j in range(len(est.sat.actuators)) if est.sat.actuators[j].has_bias and est.sat.actuators[j].estimated_bias]) + est.sat.state_len).astype('int')
    try:
        est_dipole_inds = (np.concatenate([np.sum([k.main_param.size for k in est.sat.disturbances[:j] if k.estimated_param]) for j in range(len(est.sat.disturbances)) if est.sat.disturbances[j].estimated_param and isinstance(est.sat.disturbances[j],Dipole_Disturbance)]) + est.sat.state_len+est.sat.act_bias_len+est.att_sens_bias_len).astype('int')
    except:
        est_dipole_inds = np.array([])
    try:
        est_gendist_inds = (np.concatenate([np.sum([k.main_param.size for k in est.sat.disturbances[:j] if k.estimated_param]) for j in range(len(est.sat.disturbances)) if est.sat.disturbances[j].estimated_param and isinstance(est.sat.disturbances[j],General_Disturbance)]) + est.sat.state_len+est.sat.act_bias_len+est.att_sens_bias_len).astype('int')
    except:
        est_gendist_inds = np.array([])

    real_sb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],SunSensorPair) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_mb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],MTM) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_gb_inds = (np.array([np.sum([k.output_length for k in real_sat.attitude_sensors[:j] if k.has_bias]) for j in range(len(real_sat.attitude_sensors)) if isinstance(real_sat.attitude_sensors[j],Gyro) and real_sat.attitude_sensors[j].has_bias]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])).astype('int')
    real_ab_inds = (np.array([np.sum([k.input_len for k in real_sat.actuators[:j] if k.has_bias]) for j in range(len(real_sat.actuators)) if real_sat.actuators[j].has_bias]) + real_sat.state_len).astype('int')
    try:
        real_dipole_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying]) for j in range(len(real_sat.disturbances)) if  isinstance(real_sat.disturbances[j],Dipole_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_len for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_dipole_inds = np.array([])
    try:
        real_gendist_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying]) for j in range(len(real_sat.disturbances)) if isinstance(real_sat.disturbances[j],General_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_len for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_gendist_inds = np.array([])


    if alt_title is None:
        base_title = "baseline_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        base_title = alt_title+"_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("disturbance_paper_test_files/"+base_title)
    folder_name = "disturbance_paper_test_files"

    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/quat_plot")
    plot_the_thing(np.stack([j.state[3:7] for j in goal_hist]),title = "Goal Quat",xlabel='Time (s)',ylabel = 'Goal Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalquat_plot")
    plot_the_thing(np.stack([j.state[0:3]*180.0/np.pi for j in goal_hist]),title = "Goal AV (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalav_plot")
    plot_the_thing(goalav_body,title = "Goal AV (body frame) (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (body frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalav_body_plot")
    plot_the_thing(goalav_ECI,title = "Goal AV (ECI frame) (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV (ECI frame) (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalav_ECI_plot")
    plot_the_thing(goalav_body-state_hist[:goalav_body.shape[0],0:3]*180.0/math.pi,title = "Goal AV err (deg/s)",xlabel='Time (s)',ylabel = 'Goal AV err (deg/s)',norm = True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/goalaverr_body_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ang_plot")
    plot_the_thing(ctrlangdiff,title = "Ctrl Angular Error",xlabel='Time (s)',ylabel = 'Ctrl Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlang_plot")

    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/averr_plot")
    plot_the_thing(np.log10(np.abs(avdiff)),title = "Log10 AV Error",xlabel='Time (s)',ylabel = 'Log AV Error Log(deg/s)', norm=False,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/log_averr_plot")

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
    plot_the_thing(sens_hist[:,sens_g_inds],title = "Gyro Readings (Raw)",xlabel='Time (s)',norm = True,ylabel = 'Gyro Readings (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/gyro_out_plot")
    plot_the_thing(sens_hist[:,sens_g_inds]-gbreal,title = "Gyro Readings (Real Bias Removed)",xlabel='Time (s)',norm = True,ylabel = 'Gyro Readings (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/gyro_no_bias_plot")
    plot_the_thing(sens_hist[:,sens_g_inds]-gbest,title = "Gyro Readings (Est Bias Removed)",xlabel='Time (s)',norm = True,ylabel = 'Gyro Readings (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/gyro_no_estbias_plot")



    if est_mb_inds.size > 0:
        mbest = est_state_hist[:,est_mb_inds]*180.0/math.pi
    else:
        mbest = np.zeros((est_state_hist.shape[0],3))
    if real_mb_inds.size > 0:
        mbreal = state_hist[:,real_mb_inds]*180.0/math.pi
    else:
        mbreal = np.zeros((est_state_hist.shape[0],3))
    mbdiff = mbest-mbreal
    plot_the_thing(mbest,title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_mbias_plot")
    plot_the_thing(mbreal,title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mbias_plot")
    plot_the_thing(mbdiff,title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(mbdiff)),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logmb_plot")
    plot_the_thing(sens_hist[:,sens_m_inds],title = "MTM Readings (Raw)",xlabel='Time (s)',norm = True,ylabel = 'MTM Readings (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mtm_out_plot")
    plot_the_thing(sens_hist[:,sens_m_inds]-mbreal,title = "MTM Readings (Real Bias Removed)",xlabel='Time (s)',norm = True,ylabel = 'MTM Readings (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mtm_no_bias_plot")
    plot_the_thing(sens_hist[:,sens_m_inds]-mbest,title = "MTM Readings (Est Bias Removed)",xlabel='Time (s)',norm = True,ylabel = 'MTM Readings (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/mtm_no_estbias_plot")


    if est_sb_inds.size > 0:
        sbest = est_state_hist[:,est_sb_inds]*180.0/math.pi
    else:
        sbest = np.zeros((est_state_hist.shape[0],3))
    if real_sb_inds.size > 0:
        sbreal = state_hist[:,real_sb_inds]*180.0/math.pi
    else:
        sbreal = np.zeros((est_state_hist.shape[0],3))
    sbdiff = sbest-sbreal
    plot_the_thing(sbest,title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/est_sbias_plot")
    plot_the_thing(sbreal,title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/sbias_plot")
    plot_the_thing(sbdiff,title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/sberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(sbdiff)),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logsb_plot")
    plot_the_thing(sens_hist[:,sens_s_inds],title = "Sun Readings (Raw)",xlabel='Time (s)',norm = True,ylabel = 'Sun Sensor Readings ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/sun_out_plot")
    plot_the_thing(sens_hist[:,sens_s_inds]-sbreal,title = "Sun Readings (Real Bias Removed)",xlabel='Time (s)',norm = True,ylabel = 'Sun Sensor Readings ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/sun_no_bias_plot")
    plot_the_thing(sens_hist[:,sens_s_inds]-sbest,title = "Sun Readings (Est Bias Removed)",xlabel='Time (s)',norm = True,ylabel = 'Sun Sensor Readings ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/sun_no_estbias_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrl_plot")
    plot_the_thing(np.log10(matrix_row_norm(control_hist)),title = "Log Control",xlabel='Time (s)',norm = True,ylabel = 'Control (log N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_log_ctrl_plot")
    plot_the_thing(controlwb_hist,title = "Control+Bias",xlabel='Time (s)',norm = True,ylabel = 'Control with Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrl_w_bias_plot")
    plot_the_thing(np.log10(matrix_row_norm(controlwb_hist)),title = "Log Control+Bias",xlabel='Time (s)',norm = True,ylabel = 'Control with Bias (log N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_log_ctrl_w_bias_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logav_plot")
    plot_the_thing(np.log10(ctrlangdiff),title = "Log Ctrl Angular Error",xlabel='Time (s)',ylabel = 'Ctrl Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_logctrlang_plot")

    Bbody_hist = np.stack([orb_hist[j].B@rot_mat(state_hist[j,3:7]) for j in range(len(orb_hist))])
    nBbody_hist = np.stack([normalize(Bbody_hist[j,:]) for j in range(len(Bbody_hist))])
    plot_the_thing(np.array(eclipse_hist).astype('int'),title = "Eclipse?",xlabel='Time (s)',norm = True,ylabel = 'Eclipse Y/N',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/eclipse_plot")

    plot_the_thing(Bbody_hist,title = "B Body frame",xlabel='Time (s)',norm = True,ylabel = 'B Body Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bbody_plot")
    plot_the_thing(np.stack([orb_hist[j].B for j in range(len(orb_hist))]),title = "B ECI frame",xlabel='Time (s)',norm = True,ylabel = 'B ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bECI_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody_hist[j,:],normalize(ctrlquatdiff[j,1:])))) for j in range(len(orb_hist))]),title = "Ang between B body and angular ctrl error",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/b_q_ctrlang_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody_hist[j,:],normalize(state_hist[j,:3]-goalav_body[j,:]*math.pi/180)))) for j in range(len(orb_hist))]),title = "Ang between B body and av ctrl error",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/b_w_ctrlang_plot")
    plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody_hist[j,:],normalize((state_hist[j,:3]-goalav_body[j,:]*math.pi/180)@real_sat.J)))) for j in range(len(orb_hist))]),title = "Ang between B body and av ctrl error",xlabel='Time (s)',norm = False,ylabel = 'Ang Err (Deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/b_Jw_ctrlang_plot")


    plot_the_thing(np.array([ctrlquatdiff[j,1:]@(np.eye(3)-np.outer(nBbody_hist[j,:],nBbody_hist[j,:])) for j in range(len(orb_hist))]),title = "quatvec ctrl error w/o Bbody component",xlabel='Time (s)',norm = True,ylabel = 'ctrl quat vec',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlquatvec_wo_B_plot")
    plot_the_thing(np.array([2*(180.0/np.pi)*np.arccos(np.sqrt(1-norm(ctrlquatdiff[j,1:]@(np.eye(3)-np.outer(nBbody_hist[j,:],nBbody_hist[j,:])))**2.0)) for j in range(len(orb_hist))]),title = "ctrl ang error w/o Bbody component",xlabel='Time (s)',norm = True,ylabel = 'Ctrl Ang Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlang_wo_B_plot")
    plot_the_thing(np.array([(state_hist[j,:3]-goalav_body[j,:]*math.pi/180)@(np.eye(3)-np.outer(nBbody_hist[j,:],nBbody_hist[j,:])) for j in range(len(orb_hist))]),title = "av ctrl error w/o Bbody component",xlabel='Time (s)',norm = True,ylabel = 'Jw',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlav_wo_B_plot")
    plot_the_thing(np.array([((state_hist[j,:3]-goalav_body[j,:]*math.pi/180)@real_sat.J)@(np.eye(3)-np.outer(nBbody_hist[j,:],nBbody_hist[j,:])) for j in range(len(orb_hist))]),title = "Jav ctrl error w/o Bbody component",xlabel='Time (s)',norm = True,ylabel = 'Jw (Nm2*deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ctrlJw_wo_B_plot")

    Rmats = np.dstack([rot_mat(state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    orbitRmats = np.dstack([rot_mat(two_vec_to_quat(normalize(orb_hist[j].R),normalize(np.cross(orb_hist[j].R,orb_hist[j].V)),unitvecs[2],unitvecs[0])) for j in range(len(orb_hist))]) #stacked matrices such that if you take R=[:,;,i], R@unitvecs[0] would give the orbit normal direction coordinates in ECI, R@unitvecs[2] is zenith direction
    plot_the_thing(Rmats[0,:,:],title = "Body x-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bxECI_plot")
    plot_the_thing(Rmats[1,:,:],title = "Body y-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/byECI_plot")
    plot_the_thing(Rmats[2,:,:],title = "Body z-axis in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bzECI_plot")
    plot_the_thing(orbitRmats[0,:,:],title = "Orbit Nadir in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/oxECI_plot")
    plot_the_thing(orbitRmats[1,:,:],title = "Orbit Ram in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/oyECI_plot")
    plot_the_thing(orbitRmats[2,:,:],title = "Orbit anti-normal in ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/ozECI_plot")
    plot_the_thing(np.stack([(orbitRmats[:,:,j].T@Rmats[:,:,j])[:,0] for j in range(orbitRmats.shape[2])]),title = "Body x-axis in z=zen,x=norm frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bxo_plot")
    plot_the_thing(np.stack([(orbitRmats[:,:,j].T@Rmats[:,:,j])[:,1] for j in range(orbitRmats.shape[2])]),title = "Body y-axis in z=zen,x=norm frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/byo_plot")
    plot_the_thing(np.stack([(orbitRmats[:,:,j].T@Rmats[:,:,j])[:,2] for j in range(orbitRmats.shape[2])]),title = "Body z-axis in z=zen,x=norm frame",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/bzo_plot")


    # ws = [j for j in control_laws if j.modename in [ GovernorMode.WISNIEWSKI_SLIDING, GovernorMode.WISNIEWSKI_TWISTING,GovernorMode.WISNIEWSKI_TWISTING2,GovernorMode.WISNIEWSKI_SLIDING_MAGIC]]
    used_modes = [goals.get_control_mode(k.J2000) for k in orb_hist]
    unique_used_modes = list(set(used_modes))
    s_modes = [j for j in control_laws if (j.modename in [ GovernorMode.WISNIEWSKI_SLIDING, GovernorMode.WISNIEWSKI_TWISTING,GovernorMode.WISNIEWSKI_TWISTING2,GovernorMode.WISNIEWSKI_SLIDING_MAGIC] and j.modename in unique_used_modes) ]
    if len(s_modes)>0:
        cc = s_modes[0]
        # qgain = cc.params.Lambda_q
        # breakpoint()
        s = np.stack([cc.scalc(ctrlquatdiff[j,:],state_hist[j,0:3]-goalav_body[j,:]*math.pi/180,state_hist[j,:]) for j in range(goalav_body.shape[0])])
        sdot = np.stack([-np.cross(state_hist[j,0:3],state_hist[j,0:3]@real_sat.J)@cc.params.Lambda_All + (act_torq_hist[j,:] + dist_torq_hist[j,:])@cc.params.Lambda_All + np.sign(ctrlquatdiff[j,0])*0.5*(state_hist[j,0:3]-goalav_body[j,:]*math.pi/180)@Wmat(ctrlquatdiff[j,:]).T@quat2vec_mat.T@cc.params.Lambda_q@cc.params.Lambda_All - np.cross(state_hist[j,0:3],state_hist[j,0:3]-goalav_body[j,:]*math.pi/180)@real_sat.J@cc.params.Lambda_All for j in range(goalav_body.shape[0])])

        # s = (state_hist[:goalav_body.shape[0],0:3]-goalav_body*math.pi/180)@real_sat.J@cc.params.Lambda_All + ctrlquatdiff[:goalav_body.shape[0],1:]@qgain@cc.params.Lambda_All
        plot_the_thing(s,title = "S",xlabel='Time (s)',norm = True,ylabel = 'S',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/s_plot")
        plot_the_thing(np.array([2*np.dot(s[j,:],sdot[j,:]) for j in range(goalav_body.shape[0])]),title = "Snorm2 Poincare",xlabel='|s|^2',norm = False,ylabel = 'd/dt |s|^2',xdata = np.array([j**2 for j in matrix_row_norm(s)]),save =True,plot_now = False,scatter=True, save_name = folder_name+"/"+base_title+"/snorm2_poincare_plot")
        plot_the_thing(sdot[:,0],title = "S_x Poincare",xlabel='s_x',norm = False,ylabel = 'd/dt s_x',xdata = s[:,0],save =True,plot_now = False,scatter=True, save_name = folder_name+"/"+base_title+"/sx_poincare_plot")
        plot_the_thing(sdot[:,1],title = "S_y Poincare",xlabel='s_y',norm = False,ylabel = 'd/dt s_y',xdata = s[:,1],save =True,plot_now = False,scatter=True, save_name = folder_name+"/"+base_title+"/sy_poincare_plot")
        plot_the_thing(sdot[:,2],title = "S_z Poincare",xlabel='s_z',norm = False,ylabel = 'd/dt s_z',xdata = s[:,2],save =True,plot_now = False,scatter=True, save_name = folder_name+"/"+base_title+"/sz_poincare_plot")
        plot_the_thing(np.array([(180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody_hist[j,:],normalize(s[j,:])))) for j in range(len(orb_hist))]),title = "Ang between B body and s",xlabel='Time (s)',norm = False,ylabel = 's angle (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/b_s_ctrlang_plot")
        plot_the_thing(np.array([s[j,:]@(np.eye(3)-np.outer(nBbody_hist[j,:],nBbody_hist[j,:])) for j in range(len(orb_hist))]),title = "s w/o Bbody component",xlabel='Time (s)',norm = True,ylabel = 's',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/s_wo_B_plot")
        plot_the_thing(np.log10(matrix_row_norm(s)),title = "log norm s w/o Bbody component",xlabel='Time (s)',norm = False,ylabel = 's',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_log_s_plot")
        plot_the_thing(np.log10(matrix_row_norm(np.array([s[j,:]@(np.eye(3)-np.outer(nBbody_hist[j,:],nBbody_hist[j,:])) for j in range(len(orb_hist))]))),title = "Log s norm",xlabel='Time (s)',norm = False,ylabel = 's',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/_log_s_wo_B_plot")



    #generate statistics
    metrics = find_metrics(t_hist,angdiff)
    ctrlmetrics = find_metrics(t_hist,ctrlangdiff)
    labels = ["title","al","kap","bet","dt","werrcov","mrpcov","mtmscale","J"]+["covest0","intcov"]+["ctrl conv time","ctrl tc","last 100 ctrlang err mean","last 100 ctrlang err max"]
    info = [base_title,est.al,est.kap,est.bet,dt,werrcov,mrperrcov,1,real_sat.J]+[np.diag(cov_estimate.copy()),int_cov.copy()]+[ctrlmetrics.time_to_conv,ctrlmetrics.tc_est,np.mean(ctrlangdiff[-100:]),np.amax(ctrlangdiff[-100:])]
    np.set_printoptions(precision=10)
    with open("disturbance_paper_test_files/"+base_title+"/info", 'w') as f:
        for j in range(len(labels)):
            f.write(labels[j])
            f.write(": ")
            f.write(str(info[j]))
            f.write("\n")
        f.write("\n\nctrlr info:")
        for j in control_laws:
            f.write("\n")
            f.write(str(j.modename))
            for k in list(vars(j.params).keys()):
                f.write("\n")
                f.write(str(k))
                f.write(": ")
                f.write("\n")
                f.write(str(vars(j.params)[k]))
                try:
                    f.write("\n")
                    f.write(str(np.diag(vars(j.params)[k])))
                    f.write("\n")
                    for jj in range(np.diag(vars(j.params)[k]).shape[0]):
                        f.write(str(np.diag(vars(j.params)[k])[jj]))
                        f.write(" ")
                        # f.write(str(float(np.diag(vars(j.params)[k])[jj])))
                except:
                    pass


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
    sim.real_sat_J = real_sat.J
    sim.est_sat_J = est.sat.J
    # try:
    with open("disturbance_paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    print('saved!')




#paper comparisons.  -- sensor biases and non, actuator biases and non, disturbances and non. Specific prop disturbance case

if __name__ == '__main__':
    # worb = make_wie_orb(10)
    # worb1 = make_wie_orb(1)
    # oorb = make_wisniewski_orb(10)
    # oorb1 = make_wisniewski_orb(1)
    with open("disturbance_paper_test_files/new_tests_marker"+time.strftime("%Y%m%d-%H%M%S")+".txt", 'w') as f:
        f.write("just a file to show when new runs of tests started.")

    np.random.seed(1)
    avcov = ((0.5*math.pi/180)/(60))**2.0
    angcov = (0.5*math.pi/180)**2.0
    werrcov = 1e-17
    mrperrcov = 1e-12


    wie_q0 = zeroquat#normalize(np.array([0.153,0.685,0.695,0.153]))
    wie_w0 = np.array([0.01,0.01,0.001])#np.array([0.53,0.53,0.053])#/(180.0/math.pi)

    wie_base_sat = create_Wie_sat(real=True,rand=False)
    wie_base_est_sat = create_Wie_sat( real = False, rand=False)
    wie_disturbed_sat = create_Wie_sat(    real=True,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,include_mtmbias = True)
    wie_disturbed_est_sat = create_Wie_sat(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True, estimate_dipole = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,estimate_magic_bias = True,include_mtmbias = True,estimate_mtm_bias = True)
    # wie_disturbed_gen_sat = create_Wie_sat(   real=True,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True,include_magicbias = False,include_magic_noise = False,include_mtmbias = False,J=np.diagflat(np.array([10,9,12]))*1000,dipole_mag_max=50)
    wie_disturbed_genest_sat = create_Wie_sat(real=False,rand=False,use_gg = False, use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_gen_torq = True,gen_torq_std = 1e-4, gen_mag_max = 1e-1,include_magic_noise = True)#,include_mtmbias = True,estimate_mtm_bias = True)
    wie_disturbed_genest_sat = create_Wie_sat(real=False,rand=False,use_gg = False, use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_gen_torq = True,gen_torq_std = 2e-3, gen_mag_max = 20,include_magic_noise = True,include_mtmbias = True,estimate_mtm_bias = True)

    wie_orb_file = "wie_orb_1"
    wie_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wie_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)
    wie_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*2*2,np.eye(3)*1e-6**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(25)**2.0)
    # wie_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(25)**2.0)
    wie_gendisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*2*2*0.01,np.eye(3)*1e-6**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(1e-2**2.0))
    wie_gendisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*1e-6**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(3**2.0))

    lovera_q0 = normalize(random_n_unit_vec(4))#$normalize(np.array([0.5,-0.25,0.25,0.75]))#normalize(np.array([0,0,0,1.0]))
    lovera_w0 = np.array([2,2,-2])/(180.0/math.pi)
    lovera_w0_slow = np.array([1,1,-1])/(180.0/math.pi)

    lovera_base_sat = create_Lovera_sat(real=True,rand=False,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_base_est_sat = create_Lovera_sat(  real = False, rand=False,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_disturbed_sat = create_Lovera_sat(real=True,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,include_mtqbias = True,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_disturbed_est_sat = create_Lovera_sat(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,estimate_dipole = False,include_mtqbias = True,estimate_mtq_bias = True,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_disturbed_genest_sat = create_Lovera_sat(real=False,rand=False,use_gg = True, use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_dipole = False,estimate_gen_torq = True,include_mtqbias = True,estimate_mtq_bias = True,gen_torq_std = 1e-8, gen_mag_max = 1e-2,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)

    lovera_orb_file = "lovera_orb_10"
    lovera_orb_file1 = "lovera_orb_1"
    lovera_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    lovera_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)
    lovera_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*(1.0*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)#,np.eye(3)*(1**2.0))
    lovera_gendisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*(1.0*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0,np.eye(3)*(1e-8**2.0))

    wisniewski_orb_file = "wisniewski_orb_10"
    wisniewski_orb_file1 = "wisniewski_orb_1"
    wisniewski_q0 = normalize(np.array([0,0,0,1.0]))
    wisniewski_w0 = np.array([-0.002,0.002,0.002])

    with open(wisniewski_orb_file, "rb") as fp:   #unPickling
        orb = pickle.load(fp)
    orbst = orb.get_os(0.22)
    offset_quat =  quat_inv(np.array([x for x in Quaternion.from_euler(60, 100, -100, degrees=True)]))#mrp_to_quat(2*np.tan(np.array([60,100,-100])*(math.pi/180.0)/4))

    wisniewski_q0 = quat_mult(two_vec_to_quat(orbst.R/norm(orbst.R),normalize(np.cross(orbst.R/norm(orbst.R),orbst.V/norm(orbst.V))),unitvecs[2],unitvecs[0]),offset_quat)

    wisniewski_base_sat = create_Wisniewski_stowed_sat(real=True,rand=False,include_mtq_noise = True,mtq_std = 0.00001*np.ones(3))
    wisniewski_base_est_sat = create_Wisniewski_stowed_sat(  real = False, rand=False,include_mtq_noise = True,mtq_std = 0.00001*np.ones(3))

    wisniewski_disturbed_sat = create_Wisniewski_stowed_sat(real=True,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,include_mtqbias = True,include_mtq_noise = True,include_mtmbias = True,mtq_std = 0.00001*np.ones(3),mtq_max = 10*np.ones(3))#,care_about_eclipse = False,gyro_std = np.ones(3)*(0.000025*math.pi/180.0),gyro_bsr = np.ones(3)*(0.000025*math.pi/180.0))#,include_mtq_noise = True,mtq_std = 0.001*np.ones(3))
    wisniewski_disturbed_est_sat = create_Wisniewski_stowed_sat(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,estimate_dipole = False,include_mtqbias = True,estimate_mtq_bias = True,include_mtq_noise = True,include_mtmbias = True,mtq_std = 0.00001*np.ones(3),mtq_max = 10*np.ones(3))
    wisniewski_disturbed_genest_sat = create_Wisniewski_stowed_sat(real=False,rand=False,use_gg = True, use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_dipole = False,estimate_gen_torq = True,include_mtq_noise = True,include_mtmbias = True,include_mtqbias = True,estimate_mtq_bias = True,gen_torq_std = 1e-16, gen_mag_max = 1e-2,mtq_std = 0.00001*np.ones(3),mtq_max = 10*np.ones(3))
    wisniewski_disturbed_genggest_sat = create_Wisniewski_stowed_sat(real=False,rand=False,use_gg = True , use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_dipole = False,estimate_gen_torq = True,include_mtq_noise = True,include_mtmbias = True,include_mtqbias = True,estimate_mtq_bias = True,gen_torq_std = 1e-16, gen_mag_max = 1e-2,mtq_std = 0.00001*np.ones(3),mtq_max = 10*np.ones(3))

    wisniewski_match_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wisniewski_twisting_match_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_TWISTING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wisniewski_twisting2_match_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_TWISTING2},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})

    wisniewski_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wisniewski_alt_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wisniewski_twisting_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_TWISTING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wisniewski_alt_twisting_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WISNIEWSKI_TWISTING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})

    wisniewski_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)
    wisniewski_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*1e-4**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)#,np.eye(3)*(5**2.0))
    wisniewski_gendisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*1e-4**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0,np.eye(3)*(1e-15**2.0))
    wisniewski_genggdisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*1e-4**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0,np.eye(3)*(1e-7**2.0))


    bc_orb_file = "lovera_orb_1"#"../estimation/myorb"
    bcj = np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
        [5.88304e-05, 0.03409127827, -0.00012334756],
        [-0.00671361357, -0.00012334756, 0.01004091997]])
    wa,va = np.linalg.eigh(bcj)
    print(va)
    print(wa)
    for j in range(3):
        print(j)
        print(wa[j])
        print(va[:,j])
        print([(180.0/math.pi)*np.arccos(np.abs(np.dot(normalize(va[:,j]),i))) for i in unitvecs])


    with open(bc_orb_file, "rb") as fp:   #unPickling
        orb = pickle.load(fp)
    orbst = orb.get_os(0.22)
    bc_q0 = normalize(np.array([1.0,-1,1,-1]))#normalize(np.array([1.0,0,1,0]))#two_vec_to_quat(normalize(orbst.R),-normalize(np.cross(orbst.R,orbst.V)),unitvecs[2],unitvecs[1])#two_vec_to_quat(-normalize(orbst.R),-normalize(np.cross(orbst.R,orbst.V)),va[:,0],unitvecs[1])#two_vec_to_quat(normalize(orbst.R),-normalize(np.cross(orbst.R,orbst.V)),unitvecs[2],unitvecs[1])#normalize(np.array([1.0,0,1,0]))#normalize(np.array([1.0,-1,1,-1]))#
    bc_q0_2 = normalize(np.array([1.0,0,1,0]))
    bc_w0 = 0.25*(np.pi/180.0)*normalize(np.array(np.ones(3)))#random_n_unit_vec(3)# np.array([-0.002,0.002,0.002])

    bc_real = create_BC_sat(real=True,rand=False,care_about_eclipse = True,use_dipole = False,use_SRP = True,use_gg=True,use_drag = True,mtq_bias0 = 0.0005*normalize(np.array([1.0,1,4])),use_J_diag = True)#,include_mtqbias = False,estimate_mtq_bias = False,include_mtq_noise = False)#,mtq_bias0 = 0.0005*normalize(np.array([1.0,1,4])), J =  Jtest)
    bc_est = create_BC_sat( real = False, rand=False,care_about_eclipse = True,use_dipole = False, estimate_dipole = False,use_SRP = True,use_gg=True,use_drag = True,use_J_diag = True)#,mtq_bias0 = 0.0005*normalize(np.array([1.0,1,4])),use_J_diag = True)#,include_mtqbias = False,estimate_mtq_bias = False,include_mtq_noise = False)#,mtq_bias0 = 0.0005*normalize(np.array([1.0,1,4])), J =  Jtest)
    bc_est_w_gen = create_BC_sat( real = False, rand=False,care_about_eclipse = True,use_dipole = False, estimate_dipole = False,use_drag = False,use_gen = True, estimate_gen_torq = True, gen_torq_std = 1e-14,gen_mag_max = 1e-6,use_gg=True,use_J_diag = True)

    # bc_real = create_BC_sat_more_drag(real=True,rand=False,care_about_eclipse = True,use_dipole = False,use_SRP = True,use_gg=True,use_drag = True,mtq_bias0 = 0.0005*normalize(np.array([1.0,1,4])),use_J_diag = True,drag_zshift = 0.001)
    # # bc_real_drag_nogg = create_BC_sat_more_drag(real=True,rand=False,care_about_eclipse = True,use_dipole = False,use_SRP = True,use_gg=False)
    # bc_est = create_BC_sat_more_drag( real = False, rand=False,care_about_eclipse = True,use_dipole = False, estimate_dipole = False,use_SRP = True,use_gg=True,use_drag = True,use_J_diag = True,drag_zshift = 0.001)
    # # bc_est_drag_nogg = create_BC_sat_more_drag( real = False, rand=False,care_about_eclipse = True,use_dipole = False, estimate_dipole = False,use_SRP = True,use_gg=False)
    # bc_est_w_gen = create_BC_sat_more_drag( real = False, rand=False,care_about_eclipse = True,use_dipole = False, estimate_dipole = False,use_SRP= False, use_drag = False,use_gen = True, estimate_gen_torq = True, gen_torq_std = 1e-14,gen_mag_max = 1e-6,use_gg=True,use_J_diag = True)
    # # bc_est_w_gen_nogg = create_BC_sat( real = False, rand=False,care_about_eclipse = True,use_dipole = False, estimate_dipole = False,use_drag = False,use_gen = True, estimate_gen_torq = True, gen_torq_std = 1e-10,gen_mag_max = 1e-6,use_gg=False)

    cubesat_goals_lovera =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.LOVERA_MAG_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})

    # cubesat_goals_wisniewski =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    cubesat_goals_wisniewski =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,unitvecs[2]*np.sqrt(2)/(1+0.5*np.sqrt(2)))},{0.2:np.nan*np.zeros(3)})
    # cubesat_goals_wisniewski =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    # cubesat_goals_wisniewski_2 =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    cubesat_goals_wisniewski_twisting =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.WISNIEWSKI_TWISTING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,unitvecs[2]*np.sqrt(2)/(1+0.5*np.sqrt(2)))},{0.2:np.nan*np.zeros(3)})
    # cubesat_goals_wisniewski_twisting =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.WISNIEWSKI_TWISTING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI,np.zeros(3))},{0.2:np.nan*np.zeros(3)})

    # cubesat_goals_wisniewski_mini =  Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.WISNIEWSKI_SLIDING},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})

    bc_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(1e-7*1)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-3)**2.0)
    bc_cov0_estimate_gen = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(1e-7*1)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-3)**2.0,np.eye(3)*(1e-11)**2.0)

    wie_match_10 = ["Wie_matching",          wie_base_est_sat,           wie_base_sat,       10, wie_w0,         wie_q0,         wie_base_cov0_estimate,         False,  wie_goals,         3*60*60,    wie_orb_file,[1.0,1.0,1.0]]
    wie_match_1 = ["Wie_matching1",          wie_base_est_sat,           wie_base_sat,       1,  wie_w0,         wie_q0,         wie_base_cov0_estimate,         False,  wie_goals,         3*60*60,    wie_orb_file,[1.0,1.0,1.0]]
    lovera_match_10 = ["Lovera_matching",           lovera_base_est_sat,        lovera_base_sat,    10, lovera_w0_slow,      lovera_q0,      lovera_base_cov0_estimate,      False,  lovera_goals,      12*60*60,    lovera_orb_file,[1.0,1.0,1.0]]
    lovera_match_1 = ["Lovera_matching1",           lovera_base_est_sat,        lovera_base_sat,    1,  lovera_w0_slow,      lovera_q0,      lovera_base_cov0_estimate,      False,  lovera_goals,      10*60*60,    lovera_orb_file1,[1.0,1.0,1.0]]
    wisniewski_match_10 = ["Wisniewski_matching",   wisniewski_base_est_sat,    wisniewski_base_sat,10, wisniewski_w0,  wisniewski_q0,  wisniewski_base_cov0_estimate,  True,  wisniewski_match_goals,  12*60*60,    wisniewski_orb_file,[1.0,1.0,1.0]]
    wisniewski_match_1 = ["Wisniewski_matching1",   wisniewski_base_est_sat,    wisniewski_base_sat,1,  wisniewski_w0,  wisniewski_q0,  wisniewski_base_cov0_estimate,  True,  wisniewski_match_goals,  10*60*60,    wisniewski_orb_file1,[1.0,1.0,1.0]]
    wisniewski_twist_match_1 = ["Wisniewski_twist_matching1",   wisniewski_base_est_sat,    wisniewski_base_sat,1,  wisniewski_w0,  wisniewski_q0,  wisniewski_base_cov0_estimate,  True,  wisniewski_twisting_match_goals,  10*60*60,    wisniewski_orb_file1,[2.0,1.0,1.0,0,2.0,1.0,0]]

    wie_disturbed_10 = ["Wie_disturbed", wie_disturbed_est_sat,wie_disturbed_sat,10,wie_w0,wie_q0,wie_disturbed_cov0_estimate,False,wie_goals,3*60*60,wie_orb_file,[1.0,1.0,1.0]]
    wie_disturbed_1 = ["Wie_disturbed1", wie_disturbed_est_sat,wie_disturbed_sat,1,wie_w0,wie_q0,wie_disturbed_cov0_estimate,False,wie_goals,3*60*60,wie_orb_file,[1.0,1.0,1.0]]
    lovera_disturbed_10 = ["Lovera_disturbed", lovera_disturbed_est_sat,lovera_disturbed_sat,10,lovera_w0_slow,lovera_q0,lovera_disturbed_cov0_estimate,False,lovera_goals,18*60*60,    lovera_orb_file,[1.0,1.0,1.0]]
    lovera_disturbed_1 = ["Lovera_disturbed1", lovera_disturbed_est_sat,lovera_disturbed_sat,1,lovera_w0_slow,lovera_q0,lovera_disturbed_cov0_estimate,False,lovera_goals,10*60*60,    lovera_orb_file1,[1.0,1.0,1.0]]
    wisniewski_disturbed_10 = ["Wisniewski_disturbed", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,10,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,False,wisniewski_goals,6*60*60,wisniewski_orb_file,[1.0,1.0,1.0]]
    wisniewski_disturbed_1 = ["Wisniewski_disturbed1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,False,wisniewski_goals,10*60*60,wisniewski_orb_file1,[1.0,1.0,1.0]]
    wisniewski_twist_disturbed_1 = ["Wisniewski_twist_disturbed1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,False,wisniewski_twisting_goals,10*60*60,wisniewski_orb_file1,[2.0,1.0,1.0,0,2.0,1.0,0]]
    wisniewski_alt_disturbed_1 = ["Wisniewski_alt_disturbed1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,False,wisniewski_alt_goals,10*60*60,wisniewski_orb_file1,[2.0,2.0,1.0]]
    wisniewski_alt_twist_disturbed_1 = ["Wisniewski_alt_twist_disturbed1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,False,wisniewski_alt_twisting_goals,10*60*60,wisniewski_orb_file1,[4.0,2.0,1.0,0,2.0,1.0,0]]


    wie_disturbed_w_control10 = ["Wie_disturbed_w_control", wie_disturbed_est_sat,wie_disturbed_sat,10,wie_w0,wie_q0,wie_disturbed_cov0_estimate,True,wie_goals,3*60*60,wie_orb_file,[1.0,1.0,1.0]]
    wie_disturbed_w_control1 = ["Wie_disturbed_w_control1", wie_disturbed_est_sat,wie_disturbed_sat,1,wie_w0,wie_q0,wie_disturbed_cov0_estimate,True,wie_goals,3*60*60,wie_orb_file,[1.0,1.0,1.0]]
    lovera_disturbed_w_control10 = ["Lovera_disturbed_w_control", lovera_disturbed_est_sat,lovera_disturbed_sat,10,lovera_w0_slow,lovera_q0,lovera_disturbed_cov0_estimate,True,lovera_goals,18*60*60,    lovera_orb_file,[1.0,1.0,1.0]]
    lovera_disturbed_w_control1 = ["Lovera_disturbed_w_control1", lovera_disturbed_est_sat,lovera_disturbed_sat,1,lovera_w0_slow,lovera_q0,lovera_disturbed_cov0_estimate,True,lovera_goals,10*60*60,    lovera_orb_file1,[1.0,1.0,1.0]]
    wisniewski_disturbed_w_control10 = ["Wisniewski_disturbed_w_control", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,10,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,True,wisniewski_goals,6*60*60,wisniewski_orb_file,[1.0,1.0,1.0]]
    wisniewski_disturbed_w_control1 = ["Wisniewski_disturbed_w_control1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,True,wisniewski_goals,10*60*60,wisniewski_orb_file1,[1.0,1.0,1.0]]
    wisniewski_twist_disturbed_w_control1 = ["Wisniewski_twist_disturbed_w_control1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,True,wisniewski_twisting_goals,10*60*60,wisniewski_orb_file1,[2.0,1.0,1.0,0,2.0,1.0,0]] #here
    wisniewski_alt_disturbed_w_control1 = ["Wisniewski_alt_disturbed_w_control1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,True,wisniewski_alt_goals,10*60*60,wisniewski_orb_file1,[2.0,2.0,1.0]]
    wisniewski_alt_twist_disturbed_w_control1 = ["Wisniewski_alt_twist_disturbed_w_control1", wisniewski_disturbed_est_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_disturbed_cov0_estimate,True,wisniewski_alt_twisting_goals,10*60*60,wisniewski_orb_file1,[4.0,2.0,1.0,0,2.0,1.0,0]]


    wie_disturbed_w_gencontrol10 = ["Wie_disturbed_w_gencontrol", wie_disturbed_genest_sat,wie_disturbed_sat,10,wie_w0,wie_q0,wie_gendisturbed_cov0_estimate,True,wie_goals,3*60*60,wie_orb_file,[1.0,1.0,1.0]]
    wie_disturbed_w_gencontrol1 = ["Wie_disturbed_w_gencontrol1", wie_disturbed_genest_sat,wie_disturbed_sat,1,wie_w0,wie_q0,wie_gendisturbed_cov0_estimate,True,wie_goals,3*60*60,wie_orb_file,[1.0,1.0,1.0]]
    lovera_disturbed_w_gencontrol10 = ["Lovera_disturbed_w_gencontrol", lovera_disturbed_genest_sat,lovera_disturbed_sat,10,lovera_w0_slow,lovera_q0,lovera_gendisturbed_cov0_estimate,True,lovera_goals,18*60*60,lovera_orb_file,[1.0,1.0,1.0]]
    lovera_disturbed_w_gencontrol1 = ["Lovera_disturbed_w_gencontrol1", lovera_disturbed_genest_sat,lovera_disturbed_sat,1,lovera_w0_slow,lovera_q0,lovera_gendisturbed_cov0_estimate,True,lovera_goals,10*60*60,lovera_orb_file1,[1.0,1.0,1.0]]
    wisniewski_disturbed_w_gencontrol10 = ["Wisniewski_disturbed_w_gencontrol", wisniewski_disturbed_genest_sat,wisniewski_disturbed_sat,10,wisniewski_w0,wisniewski_q0,wisniewski_gendisturbed_cov0_estimate,True,wisniewski_goals,6*60*60,wisniewski_orb_file,[1.0,1.0,1.0]]
    wisniewski_disturbed_w_gencontrol1 = ["Wisniewski_disturbed_w_gencontrol1", wisniewski_disturbed_genest_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_gendisturbed_cov0_estimate,True,wisniewski_goals,10*60*60,wisniewski_orb_file1,[1.0,1.0,1.0]]
    wisniewski_twist_disturbed_w_gencontrol1 = ["Wisniewski_twist_disturbed_w_gencontrol1", wisniewski_disturbed_genest_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_gendisturbed_cov0_estimate,True,wisniewski_twisting_goals,10*60*60,wisniewski_orb_file1,[2.0,1.0,1.0,0,2.0,1.0,0]]
    wisniewski_alt_disturbed_w_gencontrol1 = ["Wisniewski_alt_disturbed_w_gencontrol1", wisniewski_disturbed_genest_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_gendisturbed_cov0_estimate,True,wisniewski_alt_goals,10*60*60,wisniewski_orb_file1,[2.0,2.0,1.0]]
    wisniewski_alt_twist_disturbed_w_gencontrol1 = ["Wisniewski_alt_twist_disturbed_w_gencontrol1", wisniewski_disturbed_genest_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_gendisturbed_cov0_estimate,True,wisniewski_alt_twisting_goals,10*60*60,wisniewski_orb_file1,[4.0,2.0,1.0,0,2.0,1.0,0]]

    # wisniewski_disturbed_w_gencontrol_gg_in_gen10 = ["Wisniewski_disturbed_w_genggcontrol", wisniewski_disturbed_genggest_sat,wisniewski_disturbed_sat,10,wisniewski_w0,wisniewski_q0,wisniewski_genggdisturbed_cov0_estimate,True,wisniewski_goals,10*60*60,wisniewski_orb_file,[1.0,1.0,1.0]]
    # wisniewski_disturbed_w_gencontrol_gg_in_gen1 = ["Wisniewski_disturbed_w_genggcontrol1", wisniewski_disturbed_genggest_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_genggdisturbed_cov0_estimate,True,wisniewski_goals,10*60*60,wisniewski_orb_file1,[1.0,1.0,1.0]]
    # wisniewski_twist_disturbed_w_gencontrol_gg_in_gen10 = ["Wisniewski_twist_disturbed_w_genggcontrol", wisniewski_disturbed_genggest_sat,wisniewski_disturbed_sat,10,wisniewski_w0,wisniewski_q0,wisniewski_genggdisturbed_cov0_estimate,True,wisniewski_goals,10*60*60,wisniewski_orb_file,[2.0,1.0,1.0,0,2.0,1.0,0]]
    # wisniewski_twist_disturbed_w_gencontrol_gg_in_gen1 = ["Wisniewski_twist_disturbed_w_genggcontrol1", wisniewski_disturbed_genggest_sat,wisniewski_disturbed_sat,1,wisniewski_w0,wisniewski_q0,wisniewski_genggdisturbed_cov0_estimate,True,wisniewski_goals,10*60*60,wisniewski_orb_file1,[2.0,1.0,1.0,0,2.0,1.0,0]]


    lovera_on_cubesat = ["Lovera_on_cubesat", bc_est,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate,True,cubesat_goals_lovera,10*60*60,bc_orb_file,[0.1,2e-4,3e-4]]#,[1.0,1e-4,1e-4]]#,[0.1,0.01,0.001]]#,[0.01,1.0,0.1]] #,[0.01,0.1,0.01]]
    wisniewski_twist_on_cubesat = ["Wisniewski_twist_on_cubesat", bc_est,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate,True,cubesat_goals_wisniewski_twisting,10*60*60,bc_orb_file,[0.05,2.0,1.0,0,2.0,1.0,0]] #try 0.02,0.5?
    wisniewski_on_cubesat = ["Wisniewski_on_cubesat", bc_est,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate,True,cubesat_goals_wisniewski,10*60*60,bc_orb_file,[0.05,0.1,1.0]] #[0.003,0.03,100.0] seemed to work with J = 0.03*I

    # lovera_on_cubesat_nogg = ["Lovera_on_cubesat_nogg", bc_est_nogg,bc_real_nogg,1,bc_w0,bc_q0,bc_cov0_estimate,True,cubesat_goals_lovera,10*60*60,bc_orb_file,[0.1,1.5e-4,1e-4]]#
    # lovera_on_cubesat_zen = ["Lovera_on_cubesat_zen", bc_est,bc_real,1,bc_w0,bc_q0_2,bc_cov0_estimate,True,cubesat_goals_lovera_zen,10*60*60,bc_orb_file,[0.1,2e-4,3e-4]]#
    # wisniewski_on_cubesat_nogg = ["Wisniewski_on_cubesat_nogg", bc_est_nogg,bc_real_nogg,1,bc_w0,bc_q0,bc_cov0_estimate,True,cubesat_goals_wisniewski,6*60*60,bc_orb_file,[0.1,0.1,1.0]]
    lovera_on_cubesat_dist = ["Lovera_on_cubesat_disturbed", bc_est,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate,False,cubesat_goals_lovera,10*60*60,bc_orb_file,[0.1,2e-4,3e-4]]#,[1.0,1e-4,1e-4]]#,[0.1,0.01,0.001]]#,[0.01,1.0,0.1]] #,[0.01,0.1,0.01]]
    wisniewski_on_cubesat_dist = ["Wisniewski_on_cubesat_disturbed", bc_est,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate,False,cubesat_goals_wisniewski,10*60*60,bc_orb_file,[0.05,0.1,1.0]]
    wisniewski_twist_on_cubesat_dist = ["Wisniewski_twist_on_cubesat_disturbed", bc_est,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate,False,cubesat_goals_wisniewski_twisting,10*60*60,bc_orb_file,[0.05,2.0,1.0,0,2.0,1.0,0]]


    # wie_on_cubesat_gencontrol = ["Wie_on_cubesat_gen", bc_est_w_gen,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate_gen,True,cubesat_goals,6*60*60,bc_orb_file]
    lovera_on_cubesat_gencontrol = ["Lovera_on_cubesat_gen", bc_est_w_gen,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate_gen,True,cubesat_goals_lovera,10*60*60,bc_orb_file,[0.1,2e-4,3e-4]]
    wisniewski_on_cubesat_gencontrol = ["Wisniewski_on_cubesat_gen", bc_est_w_gen,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate_gen,True,cubesat_goals_wisniewski,10*60*60,bc_orb_file,[0.05,0.1,1.0]]
    wisniewski_twist_on_cubesat_gencontrol = ["Wisniewski_twist_on_cubesat_gen", bc_est_w_gen,bc_real,1,bc_w0,bc_q0,bc_cov0_estimate_gen,True,cubesat_goals_wisniewski_twisting,10*60*60,bc_orb_file,[0.05,2.0,1.0,0,2.0,1.0,0]]

    lovera_slowbase = ["Lovera_slowbase",           lovera_base_est_sat,        lovera_base_sat,    1,  lovera_w0_slow,      lovera_q0,      lovera_base_cov0_estimate,      False,  lovera_goals,      10*60*60,    lovera_orb_file1,[1.0,1.0,1.0]]


    # tests_func = [wie_match_1_test]
    tests_baseline = [wie_match_1,lovera_match_10,lovera_match_1,wisniewski_match_1,wisniewski_twist_match_1]
    tests_disturbed = [wie_disturbed_1,lovera_disturbed_10,lovera_disturbed_1,wisniewski_disturbed_10,wisniewski_disturbed_1,wisniewski_twist_disturbed_1]
    tests_disturbed_ctrl = [wie_disturbed_w_control1,lovera_disturbed_w_control10,lovera_disturbed_w_control1,wisniewski_disturbed_w_control1,wisniewski_twist_disturbed_w_control1]
    tests_disturbed_genctrl = [wie_disturbed_w_gencontrol1,lovera_disturbed_w_gencontrol10,lovera_disturbed_w_gencontrol1,wisniewski_disturbed_w_gencontrol1,wisniewski_twist_disturbed_w_gencontrol1]#,wisniewski_disturbed_w_gencontrol_gg_in_gen10,wisniewski_disturbed_w_gencontrol_gg_in_gen1]
    # tests_cubesat = [wisniewski_twist_on_cubesat,lovera_on_cubesat_gencontrol_zen,lovera_on_cubesat_gencontrol_nogg,wisniewski_on_cubesat_nogg,wisniewski_on_cubesat_gencontrol_nogg,lovera_on_cubesat_nogg,lovera_on_cubesat_zen,wisniewski_on_cubesat_dist,lovera_on_cubesat_dist,lovera_on_cubesat_gencontrol,wisniewski_on_cubesat_gencontrol,lovera_on_cubesat,lovera_on_cubesat2,lovera_on_cubesat3,lovera_on_cubesat4]
    tests_wisniewski_alt = [wisniewski_alt_disturbed_1,wisniewski_alt_twist_disturbed_1,wisniewski_alt_disturbed_w_control1,wisniewski_alt_twist_disturbed_w_control1,wisniewski_alt_disturbed_w_gencontrol1,wisniewski_alt_twist_disturbed_w_gencontrol1]
    tests_cubesat = [wisniewski_on_cubesat,wisniewski_on_cubesat_dist,wisniewski_on_cubesat_gencontrol,wisniewski_twist_on_cubesat,wisniewski_twist_on_cubesat_dist,wisniewski_twist_on_cubesat_gencontrol,lovera_on_cubesat,lovera_on_cubesat_dist,lovera_on_cubesat_gencontrol]
    # twist_tests = [wisniewski_twist_disturbed_w_control1,wisniewski_twist_match_1,wisniewski_twist_disturbed_1,wisniewski_twist_disturbed_w_gencontrol1]#,wisniewski_twist_on_cubesat,wisniewski_twist_on_cubesat_dist,wisniewski_twist_on_cubesat_gencontrol]


    tests = tests_baseline + tests_disturbed + tests_disturbed_ctrl + tests_disturbed_genctrl + tests_cubesat + tests_wisniewski_alt

    tests = tests_cubesat + tests_wisniewski_alt#+[wisniewski_magic_disturbed_w_control1,wisniewski_twist2_disturbed_w_control1]#+new_tests#tests_cubesat#+lovera4_tests#lovera4_tests
    tests = [wisniewski_match_1,wisniewski_disturbed_1,wisniewski_disturbed_w_control1,wisniewski_disturbed_w_gencontrol1]+tests_cubesat + tests_wisniewski_alt
    tests = tests_baseline[-2:]+tests_disturbed[-2:]+tests_disturbed_genctrl[-2:]+tests_cubesat+tests_wisniewski_alt
    tests = [wisniewski_twist_on_cubesat_gencontrol,wisniewski_twist_on_cubesat_dist,wisniewski_on_cubesat_dist,wisniewski_on_cubesat_gencontrol,lovera_on_cubesat,lovera_on_cubesat_dist,lovera_on_cubesat_gencontrol]
    tests = [lovera_slowbase]#,wie_disturbed_w_gencontrol1]#,wie_disturbed_w_control1,wie_disturbed_w_gencontrol1]
    # tests = tests
    # tests = [tests_disturbed_genctrl[1],tests_disturbed_genctrl[2]] + tests_baseline[1:3] + tests_disturbed[1:3] + tests_disturbed_ctrl[1:3] + tests_cubesat
    for j in tests:
        [title,est_sat,real_sat,dt,w0,q0,cov_estimate,dist_control,goals,tf,orb_file,ctrl_mults] = j
        est_sat = copy.deepcopy(est_sat)
        estimate = np.zeros(est_sat.state_len+est_sat.act_bias_len+est_sat.att_sens_bias_len+est_sat.dist_param_len)
        estimate[0:3] = w0
        estimate[3:7] = q0
        # estimate[7:10] = np.concatenate([j.bias for j in real_sat.actuators])
        if np.any([j.estimated_param for j in est_sat.disturbances]):
            dist_ic = block_diag(*[j.std**2.0 for j in est_sat.disturbances if j.estimated_param])
            print("distic", dist_ic)
        else:
            dist_ic = np.zeros((0,0))
        int_cov =  dt*block_diag(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),dist_ic)
        # breakpoint()
        print(int_cov.shape)
        print(estimate.shape)
        print(est_sat.state_len,est_sat.act_bias_len,est_sat.att_sens_bias_len,est_sat.dist_param_len)
        print(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]).shape,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]).shape,np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]).shape,dist_ic.shape)
        # wa,va = np.linalg.eigh(real_sat.J)
        # print(va)
        # print(wa)
        est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = False)
        est.use_cross_term = True
        est.al = 1.0#1e-3#e-3#1#1e-1#al# 1e-1
        est.kap =  3 - (estimate.size - 1 - sum([j.use_noise for j in est.sat.actuators]))
        est.include_int_noise_separately = False
        est.include_sens_noise_separately = False
        est.scale_nonseparate_adds = False
        est.included_int_noise_where = 2
        # cm = np.array(ctrl_mults)
        ctrl_mults = np.pad(ctrl_mults,[(0,10)],mode = 'constant',constant_values = np.nan)
        # print(ctrl_mults,cm)
        if isinstance(real_sat.actuators[0],MTQ):
            control_laws =  [NoControl(est.sat),Bdot(1e8,est.sat),BdotEKF(1e8,est.sat),Lovera([0.01*ctrl_mults[0],50*ctrl_mults[1],50*ctrl_mults[2]],est.sat,include_disturbances=dist_control,calc_av_from_quat = True),Lovera04([0.01*ctrl_mults[0],50*ctrl_mults[1],50*ctrl_mults[2]],est.sat,include_disturbances=dist_control),WisniewskiTwisting([est.sat.J/np.mean(np.linalg.eigvals(est.sat.J))*0.002*ctrl_mults[0],np.eye(3)*0.003*ctrl_mults[1],np.eye(3)*ctrl_mults[2],np.eye(3)*ctrl_mults[3],ctrl_mults[4],ctrl_mults[5],ctrl_mults[6],0,0],est.sat,include_disturbances=dist_control,calc_av_from_quat = True),WisniewskiSliding([np.eye(3)*0.002*ctrl_mults[0],np.eye(3)*0.003*ctrl_mults[1],np.eye(3)*ctrl_mults[2]],est.sat,include_disturbances=dist_control,calc_av_from_quat = True),WisniewskiTwisting2([np.eye(3)*0.002*ctrl_mults[0],np.eye(3)*0.003*ctrl_mults[1],np.eye(3)*ctrl_mults[2]],est.sat,include_disturbances=dist_control,calc_av_from_quat = True)]
        else:
            control_laws =  [NoControl(est.sat),Magic_PD([np.eye(3)*200*ctrl_mults[0],np.eye(3)*5.0*ctrl_mults[1]],est.sat,include_disturbances=dist_control),WisniewskiSlidingMagic([np.eye(3)*0.002*ctrl_mults[0],np.eye(3)*0.003*ctrl_mults[1],np.eye(3)*ctrl_mults[2]],est.sat,include_disturbances=dist_control,calc_av_from_quat = True),WisniewskiTwistingMagic([np.eye(3)*0.002*ctrl_mults[0],np.eye(3)*0.003*ctrl_mults[1],np.eye(3)*ctrl_mults[2]],est.sat,include_disturbances=dist_control,calc_av_from_quat = True)]

        state0 = np.zeros(real_sat.state_len)
        state0[0:3] = w0
        state0[3:7] = q0
        run_sim(orb_file,state0,est,copy.deepcopy(real_sat),control_laws,goals,tf=tf,dt = dt,alt_title = title,rand=False)
