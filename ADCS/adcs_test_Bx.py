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


def run_sim(orb,state0,real_sat,adcsys,j2000_0 = 0.22,tf=60*60*3,dt = 1,alt_title = None,rand = False):
    if alt_title is None:
        base_title = "baseline_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        base_title = alt_title+"_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("adcs_test_files/"+base_title)

    ##save a copy of this file bc I'm bad at debugging and SVN
    this_script = os.path.basename(__file__)
    with open(this_script, 'r') as f2:
        script_copy = f2.read()
    with open("adcs_test_files/"+base_title+"/program_copy.txt", 'w') as f:
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

    est_sat = est.sat
    t0 = 0
    q0 = state0[3:7]
    w0 = state0[0:3]

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),real_sat.state_len+sum([j.has_bias for j in real_sat.actuators])+sum([j.has_bias for j in real_sat.sensors])+sum([j.time_varying*j.main_param.size for j in real_sat.disturbances if j.time_varying])))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    plan_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))

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
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))

    est_sat0 = copy.deepcopy(est_sat)

    while t<tf:

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        gps_sens = real_sat.GPS_values(state,real_vecs)

        #run estimator

        prev_est_state = adcsys.estimator.use_state.val.copy()

        tt0 = time.process_time()
        prev_control = control.copy()
        # breakpoint()
        print(state)
        control = adcsys.ADCS_update(j2000_0+(t-t0)*sec2cent,sens,gps_sens,state_truth = state)
        # adcsys.orbital_state_update(gps_sens,j2000_0+(t-t0)*sec2cent)
        # adcsys.estimation_update(sens,j2000_0+(t-t0)*sec2cent,control)
        est_state = adcsys.estimator.use_state.val

        # control = adcsys.actuation(j2000_0+(t-t0)*sec2cent,sens)
        goal_state = adcsys.current_goal
        # next_goal_state = adcsys.next_goal
        tt1 = time.process_time()

        # if t>250:
        #     breakpoint()
        # control = ctrlr.find_actuation(est_state,orbt,nextorb,goal_state,[],next_goal_state,sens,[],False)
        print(t)
        print('quat',state[3:7])#,norm(q_err[1:]),norm(np.cross(q_err[1:],est.os_vecs['b']))/norm(est.os_vecs['b']))
        try:
            print('plan quat',goal_state.state[3:7])
        except:
            pass
        print(goal_state.eci_vec,goal_state.body_vec)
        print('goalquat',goal_state.state[3:7])
        print('ang bw state and goal: ', (180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(goal_state.state[3:7],state[3:7]),-1,1)**2.0 ))
        print('ang bw est state and goal: ', (180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(goal_state.state[3:7],est_state[3:7]),-1,1)**2.0 ))
        print('ang bw est state and state: ', (180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        # print(goal_state.body_vec@rot_mat(goal_state.state[3:7]).T)
        # print(goal_state.eci_vec)
        print('ang bw eci vec and body vec: ', (180/np.pi)*math.acos(np.dot(goal_state.eci_vec,goal_state.body_vec@rot_mat(state[3:7]).T )))

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
        # print(goal_state.state)
        # print(orbt.in_eclipse())

        # print('dist ',real_sat.last_dist_torq,norm(real_sat.last_dist_torq))
        # disttorqest = est.sat.dist_torque(est_state[:est.sat.state_len],est.os_vecs).copy()
        # print('dist est ',disttorqest,norm(disttorqest))
        # print(real_sat.last_dist_torq - disttorqest,norm(real_sat.last_dist_torq - disttorqest))
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
        Bbody = real_vecs["b"]@rot_mat(state[3:7])
        # print("B est meas", Bbody*est_sat.sensors[0].scale + est_state[10:13])
        quat_err = quat_mult(quat_inv(goal_state.state[3:7]),state[3:7])
        nBbody = normalize(Bbody)
        quat_err_unitvec = normalize(quat_err[1:])
        w_err_body = state[0:3]-plan_av@rot_mat(quat_err)
        w_err_body_unitvec = normalize(w_err_body)
        print("ang bw rot err & Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,quat_err_unitvec))))
        print("ang bw w err & Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,w_err_body_unitvec))))
        print("ang bw rot err & w err", (180.0/np.pi)*np.arccos(np.abs(np.dot(quat_err_unitvec,w_err_body_unitvec))))
        print("ang bw Bbody & rot err diff with w err", (180.0/np.pi)*np.arccos(np.abs(np.dot(nBbody,normalize(quat_err_unitvec-w_err_body_unitvec)))))
        # print(va[:,-1],np.sqrt(wa[-1]))
        # print("cov ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(Bbody,va[:,-1]))/norm(Bbody)))

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
        est_state_hist[ind,:] = est.use_state.val
        goal_hist += [goal_state.copy()]
        # breakpoint()
        orb_hist += [orbt]
        orb_est_hist += [adcsys.orbital_state]
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
        if adcsys.current_mode in PlannerModeList:
            plan_state_hist[ind,0:adcsys.virtual_sat.state_len] = adcsys.planned_info[0]
            plan_control_hist[ind,:] = adcsys.planned_info[1]
            plan_dist_torq_hist[ind,:] = adcsys.planned_info[4]

        # if ind>1:
        #     print(np.abs(dist_torq_hist[ind,:]-dist_torq_hist[ind-1,:]),norm(dist_torq_hist[ind,:]-dist_torq_hist[ind-1,:]))

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(j2000_0+(t-t0)*sec2cent)
        # orbt.B = 1e-4*unitvecs[0]

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


    folder_name = "adcs_test_files"
    os.mkdir("adcs_test_files/"+base_title+"/environment")
    os.mkdir("adcs_test_files/"+base_title+"/actual")
    os.mkdir("adcs_test_files/"+base_title+"/estimated")
    os.mkdir("adcs_test_files/"+base_title+"/plan")
    os.mkdir("adcs_test_files/"+base_title+"/goal")
    os.mkdir("adcs_test_files/"+base_title+"/plan_v_actual")
    os.mkdir("adcs_test_files/"+base_title+"/estimated_v_actual")
    os.mkdir("adcs_test_files/"+base_title+"/actual_v_goal")
    os.mkdir("adcs_test_files/"+base_title+"/plan_v_goal")
    os.mkdir("adcs_test_files/"+base_title+"/estimated_v_plan")
    os.mkdir("adcs_test_files/"+base_title+"/estimated_v_goal")


    #preparation
    orbitRmats = np.dstack([rot_mat(two_vec_to_quat(-orb_hist[j].R,orb_hist[j].V,unitvecs[0],unitvecs[1])) for j in range(len(orb_hist))]) #stacked matrices such that if you take R=[:,;,i], R@unitvecs[0] would give the nadir direction coordinates in ECI, R@unitvecs[0] is ram direction

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
    try:
        real_dipole_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(real_sat.disturbances)) if  isinstance(real_sat.disturbances[j],Dipole_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_length for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_dipole_inds = np.array([])
    try:
        real_gendist_inds = (np.concatenate([np.sum([k.main_param.size for k in real_sat.disturbances[:j] if k.time_varying])+np.arange(real_sat.disturbances[j].main_param.size) for j in range(len(real_sat.disturbances)) if isinstance(real_sat.disturbances[j],General_Disturbance)]) + real_sat.state_len+sum([j.input_len for j in real_sat.actuators if j.has_bias])+sum([j.output_length for j in real_sat.attitude_sensors if j.has_bias])).astype('int')
    except:
        real_gendist_inds = np.array([])

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
    Rmats = np.dstack([rot_mat(state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    av_ECI = np.stack([state_hist[j,0:3]@rot_mat(state_hist[j,3:7]).T*180.0/np.pi for j in range(state_hist.shape[0])])
    angmom_ECI = np.stack([state_hist[j,0:3]@real_sat.J@rot_mat(state_hist[j,3:7]).T*180.0/np.pi for j in range(state_hist.shape[0])])

    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/av_plot")
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

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/ctrl_plot")

    plot_the_thing(np.stack([orb_hist[j].B@rot_mat(state_hist[j,3:7]) for j in range(len(orb_hist))]),title = "B Body frame",xlabel='Time (s)',norm = True,ylabel = 'B Body Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/actual/bbody_plot")

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

    plan_point_vec_eci = np.stack([point_vec_body[j,:]@plannedRmats[:,:,j].T for j in range(point_vec_body.shape[0])])
    plan_goal_vec_body = np.stack([goal_vec_eci[j,:]@plannedRmats[:,:,j] for j in range(point_vec_body.shape[0])])
    plan_alignment_rot_ax_body = np.stack([np.cross(plan_goal_vec_body[j,:],point_vec_body[j,:])for j in range(point_vec_body.shape[0])])
    plan_alignment_rot_ax_eci = np.stack([np.cross(plan_point_vec_eci[j,:],goal_vec_eci[j,:])for j in range(point_vec_body.shape[0])])
    plot_the_thing(plan_point_vec_eci,title = "Planned Body Pointing Vector in ECI",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_point_vec_eci_plot")
    plot_the_thing(plan_goal_vec_body,title = "Planned Goal ECI Pointing Vector in Body",xlabel='Time (s)',norm = False,ylabel = 'Coordinates',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/plan/plan_goal_vec_body_plot")



    #estimated
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
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_av_plot")

    plot_the_thing(est_act_torq_hist,title = 'Estimated act Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_act_torq_plot")
    plot_the_thing(est_dist_torq_hist,title = 'Estimated Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_dist_torq_plot")
    plot_the_thing(est_dist_torq_hist+est_act_torq_hist,title = 'Estimated Combo Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Combo Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_combo_torq_plot")

    plot_the_thing(abest,title = "Est Actuator Bias",xlabel='Time (s)',norm = True,ylabel = 'Actuator Bias ([units])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_abias_plot")
    plot_the_thing(gbest,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_gbias_plot")
    plot_the_thing(mbest,title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_mbias_plot")
    plot_the_thing(sbest,title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_sbias_plot")
    plot_the_thing(dpest,title = "Est Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dipole (Am^2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/est_dipole_plot")

    plot_the_thing(np.stack([orb_est_hist[j].R for j in range(len(orb_hist))]),title = "Estimated R ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Estimated R ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/rECI_est_plot")
    plot_the_thing(np.stack([orb_est_hist[j].V for j in range(len(orb_hist))]),title = "Estimated V ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Estimated V ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated/vECI_est_plot")


    #estimated v actual
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7])*np.sign(np.dot(est_state_hist[j,3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    abdiff = abest-abreal
    gbdiff = gbest-gbreal
    mbdiff = mbest-mbreal
    sbdiff = sbest-sbreal
    dpdiff = dpest-dpreal
    plot_the_thing(avdiff,title = "Quat Error",xlabel='Time (s)',ylabel = 'Quat Error', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/quaterr_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/averr_plot")
    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/mrp_plot")

    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logav_plot")

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
    plot_the_thing(np.log10(matrix_row_norm(dpdiff)),title = "Log Dipole Error",xlabel='Time (s)',ylabel = 'Log Dipole Error (log (Am^2))',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_log_dipole_plot")

    plot_the_thing(np.log10(matrix_row_norm(abdiff)),title = "Log Actuator Bias Error",xlabel='Time (s)',ylabel = 'Log Actuator Bias Error (log [units]])',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logab_plot")
    plot_the_thing(np.log10(matrix_row_norm(gbdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_loggb_plot")
    plot_the_thing(np.log10(matrix_row_norm(mbdiff)),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logmb_plot")
    plot_the_thing(np.log10(matrix_row_norm(sbdiff)),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/_logsb_plot")

    plot_the_thing(np.stack([orb_est_hist[j].R - orb_hist[j].R for j in range(len(orb_hist))]),title = "Error Estimated R ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Error Estimated R ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/rECI_est_err_plot")
    plot_the_thing(np.array([np.log10(norm(orb_est_hist[j].R - orb_hist[j].R)) for j in range(len(orb_hist))]),title = "Log Error Estimated R ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Log Error Estimated R ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/rECI_est_logerr_plot")
    plot_the_thing(np.stack([orb_est_hist[j].V - orb_hist[j].V for j in range(len(orb_hist))]),title = "Error Estimated V ECI frame",xlabel='Time (s)',norm = True,ylabel = 'Error Estimated V ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/vECI_est_err_plot")
    plot_the_thing(np.array([np.log10(norm(orb_est_hist[j].V - orb_hist[j].V)) for j in range(len(orb_hist))]),title = "Log Error Estimated V ECI frame",xlabel='Time (s)',norm = False,ylabel = 'Log Error Estimated V ECI Frame',xdata = np.array(t_hist),save =True,plot_now = False, save_name = folder_name+"/"+base_title+"/estimated_v_actual/vECI_est_logerr_plot")





    #actual v goal
    ctrlangdiff = np.stack([(180/np.pi)*np.arccos(-1 + 2*np.clip(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7]),-1,1)**2.0)  for j in range(state_hist.shape[0])])
    ctrlquatdiff = np.stack([quat_mult(quat_inv(goal_hist[j].state[3:7]),state_hist[j,3:7])*np.sign(np.dot(goal_hist[j].state[3:7],state_hist[j,3:7])) for j in range(state_hist.shape[0])])
    ctrlmrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_mrp(ctrlquatdiff[j,:]) for j in range(quatdiff.shape[0])]))
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
    mrpdiff_from_plan = (180/np.pi)*4*np.arctan(np.stack([quat_to_mrp(quatdiff_from_plan[j,:]) for j in range(quatdiff.shape[0])]))
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





















    #generate statistics
    metrics = find_metrics(t_hist,angdiff)
    ctrlmetrics = find_metrics(t_hist,ctrlangdiff)
    labels = ["title","al","kap","bet","dt","werrcov","mrpcov","mtmscale"]+["covest0","intcov"]+["ctrl conv time","ctrl tc","last 100 ctrlang err mean","last 100 ctrlang err max"]
    info = [base_title,est.al,est.kap,est.bet,dt,werrcov,mrperrcov,1]+[np.diag(cov_estimate.copy()),int_cov.copy()]+[ctrlmetrics.time_to_conv,ctrlmetrics.tc_est,np.mean(ctrlangdiff[-100:]),np.amax(ctrlangdiff[-100:])]
    with open("adcs_test_files/"+base_title+"/info", 'w') as f:
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
    sim.plan_state_hist = plan_state_hist
    sim.plan_control_hist = plan_control_hist
    sim.plan_dist_torq_hist = plan_dist_torq_hist
    # try:
    with open("adcs_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)




#paper comparisons.  -- sensor biases and non, actuator biases and non, disturbances and non. Specific prop disturbance case

if __name__ == '__main__':
    # worb = make_wie_orb(10)
    # worb1 = make_wie_orb(1)
    # oorb = make_wisniewski_orb(10)
    # oorb1 = make_wisniewski_orb(1)
    with open("adcs_test_files/new_tests_marker"+time.strftime("%Y%m%d-%H%M%S")+".txt", 'w') as f:
        f.write("just a file to show when new runs of tests started.")

    np.random.seed(1)
    avcov = ((0.0005*math.pi/180)/(60))**2.0
    angcov = (0.1*math.pi/180)**2.0
    werrcov = 1e-17
    mrperrcov = 1e-12

    wie_q0 = normalize(np.array([0.153,0.685,0.695,0.153]))#zeroquat#normalize(np.array([0.153,0.685,0.695,0.153]))
    wie_w0 = np.array([0.01,0.01,0.001])#np.array([0.53,0.53,0.053])#/(180.0/math.pi)

    wie_base_sat_w_GPS = create_Wie_sat_w_GPS(real=True,rand=False)
    wie_base_est_sat_w_GPS = create_Wie_sat_w_GPS( real = False, rand=False)
    wie_disturbed_sat_w_GPS = create_Wie_sat_w_GPS(    real=True,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,include_mtmbias = True)
    wie_disturbed_est_sat_w_GPS = create_Wie_sat_w_GPS(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True, estimate_dipole = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,estimate_magic_bias = True,include_mtmbias = True,estimate_mtm_bias = True)

    wie_disturbed_sat_w_GPS = create_Wie_sat_w_GPS(    real=True,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,include_mtmbias = True)
    wie_disturbed_est_sat_w_GPS = create_Wie_sat_w_GPS(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True, estimate_dipole = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,estimate_magic_bias = True,include_mtmbias = True,estimate_mtm_bias = True)
    # wie_disturbed_gen_sat = create_Wie_sat(   real=True,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True,include_magicbias = False,include_magic_noise = False,include_mtmbias = False,J=np.diagflat(np.array([10,9,12]))*1000,dipole_mag_max=50)
    wie_disturbed_genest_sat = create_Wie_sat(real=False,rand=False,use_gg = False, use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_gen_torq = True,gen_torq_std = 1e-4, gen_mag_max = 1e-1,include_magic_noise = True)#,include_mtmbias = True,estimate_mtm_bias = True)
    wie_disturbed_genest_sat = create_Wie_sat(real=False,rand=False,use_gg = False, use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_gen_torq = True,gen_torq_std = 2e-3, gen_mag_max = 20,include_magic_noise = True,include_mtmbias = True,estimate_mtm_bias = True)

    wie_orb_file = "wie_orb_1"
    wie_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.WIE_MAGIC_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    wie_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)
    wie_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*2*2,np.eye(3)*1e-6**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(25)**2.0)
    # wie_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*2*2,np.eye(3)*1e-6**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)

    # wie_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*2*2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(25)**2.0)
    # wie_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)

    # wie_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(25)**2.0)
    wie_gendisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*2*2*0.01,np.eye(3)*1e-6**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(1e-2**2.0))
    wie_gendisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*1e-6**2,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0,np.eye(3)*(3**2.0))


    bc_real = create_GPS_BC_sat(real=True,rand=False,care_about_eclipse = False,use_dipole = False)
    bc_est = create_GPS_BC_sat( real = False, rand=False,care_about_eclipse = False,use_dipole = False)



    bc_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(1e-7*1)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-3)**2.0)

    bc_real = create_GPS_BC_sat(real=True,rand=False,include_mtmbias = False, include_mtm_noise = False,care_about_eclipse = False,use_dipole = False,use_drag = False)
    bc_est = create_GPS_BC_sat( real = False, rand=False,include_mtmbias = False, include_mtm_noise = False,estimate_mtm_bias = False,care_about_eclipse = False,use_dipole = False,use_drag = False)
    bc_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-3)**2.0)


    bc_real = create_GPS_BC_sat(real=True,rand=False,include_mtmbias = False, include_mtm_noise = True,care_about_eclipse = False,use_dipole = False,use_drag = False,include_mtqbias = False,include_sbias = False,include_gbias = False)
    bc_est = create_GPS_BC_sat( real = False, rand=False,include_mtmbias = False, include_mtm_noise = True,estimate_mtm_bias = False,care_about_eclipse = False,use_dipole = False,use_drag = False,include_mtqbias = False,estimate_mtq_bias = False,include_sbias = False, estimate_sun_bias = False,include_gbias = False,estimate_gyro_bias = False)
    bc_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov)


    # bc_real = create_GPS_BC_sat(real=True,rand=False,include_mtmbias = False, include_mtm_noise = False,care_about_eclipse = False,use_dipole = False,use_drag = False,include_mtqbias = False,include_sbias = False)
    # bc_est = create_GPS_BC_sat( real = False, rand=False,include_mtmbias = False, include_mtm_noise = False,estimate_mtm_bias = False,care_about_eclipse = False,use_dipole = False,use_drag = False,include_mtqbias = False,estimate_mtq_bias = False,include_sbias = False,estimate_sun_bias = False)
    # bc_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(0.2*math.pi/180)**2.0)

    bc_q0 = normalize(np.array([0,0,0,1]))
    # bc_q0 = normalize(np.array([-1,-1,1,1]))
    bc_w0 = (np.pi/180.0)*random_n_unit_vec(3)#

    lovera_q0 = normalize(random_n_unit_vec(4))#$normalize(np.array([0.5,-0.25,0.25,0.75]))#normalize(np.array([0,0,0,1.0]))
    lovera_w0 = np.array([2,2,-2])/(180.0/math.pi)
    lovera_w0_slow = np.array([1,1,-1])/(180.0/math.pi)
    lovera_w0_vslow = 0.01*np.array([1,1,-1])/(180.0/math.pi)

    lovera_base_sat_GPS = create_Lovera_sat_GPS(real=True,rand=False,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_base_est_sat_GPS = create_Lovera_sat_GPS(  real = False, rand=False,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_disturbed_sat_GPS = create_Lovera_sat_GPS(real=True,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,include_mtqbias = True,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_disturbed_est_sat_GPS = create_Lovera_sat_GPS(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = False, use_SRP = True,estimate_dipole = False,include_mtqbias = True,estimate_mtq_bias = True,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)
    lovera_disturbed_genest_sat_GPS = create_Lovera_sat_GPS(real=False,rand=False,use_gg = True, use_drag = False, use_dipole = False, use_SRP = False,use_gen = True,estimate_dipole = False,estimate_gen_torq = True,include_mtqbias = True,estimate_mtq_bias = True,gen_torq_std = 1e-8, gen_mag_max = 1e-2,mtq_max = 50.0*np.ones(3),include_mtq_noise = True)

    lovera_orb_file = "lovera_orb_10"
    lovera_orb_file1 = "lovera_orb_1"
    lovera_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.LOVERA_MAG_PD},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.nan*np.zeros(3)})
    lovera_base_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(2.5*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.3*0.02)**2.0)
    lovera_disturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*(1.0*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0)#,np.eye(3)*(1**2.0))
    lovera_gendisturbed_cov0_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.5*0.5,np.eye(3)*(1.0*((math.pi/180.0)/3600))**2.0,np.eye(3)*(0.03)**2.0,np.eye(3)*(1e-8**2.0))

    dt = 1
    planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+120*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.zeros(3)})
    planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+120*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR}, {0.2:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),0.22+220*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)),0.22+250*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),0.22+400*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),0.22+700*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),0.22+1000*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})
    planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+50*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC}, {0.22:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})
    planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+100*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC}, {0.22:(PointingGoalVectorMode.NADIR,np.zeros(3)),0.22+1500*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),0.22+3000*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})
    # planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+200*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR}, {0.22:(PointingGoalVectorMode.NADIR,np.zeros(3)),0.22+1500*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),0.22+3000*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})

    # planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+50*sec2cent:GovernorMode.TRAJ_MAG_PD}, {0.22:(PointingGoalVectorMode.NADIR,np.zeros(3)),0.22+1500*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),0.22+3000*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})

    # planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC}, {0.22:(PointingGoalVectorMode.NADIR,np.zeros(3)),0.22+500*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),0.22+1000*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})

    # planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.WIE_MAGIC_PD,0.22+50*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC}, {0.22:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3)),0.22+500*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP,unitvecs[0]),0.22+1000*sec2cent:(PointingGoalVectorMode.PROVIDED_MRP,unitvecs[2])}, {0.2:unitvecs[0]})
    # planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+50*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR}, {0.22:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0]),0.22+500*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,-unitvecs[0]),0.22+1000*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0])}, {0.2:unitvecs[0]})
    # planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+50*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC},  {0.22:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0]),0.22+1000*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,-unitvecs[0]),0.22+2000*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0])}, {0.2:unitvecs[0]})
    # planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+200*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC}, {0.22:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0]),0.22+1500*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,-unitvecs[0]),0.22+3000*sec2cent:(PointingGoalVectorMode.PROVIDED_ECI_VECTOR,unitvecs[0])}, {0.2:unitvecs[0]})


    lovera_match_1 = ["Lovera_matching1",           lovera_base_est_sat_GPS,        lovera_base_sat_GPS,    1,  lovera_w0_slow,      lovera_q0,      lovera_base_cov0_estimate,      False,  lovera_goals,      60*60,    lovera_orb_file1,""]
    lovera_disturbed_1 = ["Lovera_dist1",           lovera_disturbed_est_sat_GPS,        lovera_disturbed_sat_GPS,    1,  lovera_w0_slow,      lovera_q0,      lovera_disturbed_cov0_estimate,      False,  lovera_goals,      60*60,    lovera_orb_file1,""]
    wie_disturbed_w_control1 = ["Wie_dist", wie_disturbed_est_sat_w_GPS,wie_disturbed_sat_w_GPS,1,wie_w0,wie_q0,wie_disturbed_cov0_estimate,True,wie_goals,10*60,wie_orb_file,""]
    wie_base1 = ["Wie_base", wie_base_est_sat_w_GPS,wie_base_sat_w_GPS,1,wie_w0,wie_q0,wie_base_cov0_estimate,True,wie_goals,10*60,wie_orb_file,""]
    # wie_base1 = ["Wie_base", wie_base_est_sat_w_GPS,wie_base_sat_w_GPS,1,wie_w0,wie_q0,wie_base_cov0_estimate,True,wie_goals,10*60,wie_orb_file,""]

    plannertest = ["planner", bc_est,bc_real,1,0.01*bc_w0,bc_q0,bc_cov0_estimate,True,planner_goals,3000,"../estimation/myorb",""]
    # plannertest = ["planner", lovera_base_est_sat_GPS,lovera_base_sat_GPS,True,lovera_w0_vslow,lovera_q0,lovera_base_cov0_estimate,True,planner_goals,2400,lovera_orb_file1,""]
    # plannertest = ["planner", wie_base_est_sat_w_GPS,wie_base_sat_w_GPS,True,lovera_w0_vslow,lovera_q0,wie_base_cov0_estimate,True,planner_goals,1200,lovera_orb_file1,""]

    tests = [plannertest]

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
            # print(int_cov.shape)
            # print(estimate.shape)
            # print(est_sat.state_len,est_sat.act_bias_len,est_sat.att_sens_bias_len,est_sat.dist_param_len)
            # print(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]).shape,np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]).shape,np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]).shape,dist_ic.shape)
            est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True)
            est = PerfectEstimator(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True)
            est.use_cross_term = True
            est.bet = 2.0
            est.al = 1.0#1e-3#e-3#1#1e-1#al# 1e-1
            est.kap =  3 - (estimate.size - 1 - sum([j.use_noise for j in est.sat.actuators]))
            est.include_int_noise_separately = False
            est.include_sens_noise_separately = False
            est.scale_nonseparate_adds = False
            est.included_int_noise_where = 2
            # est = PerfectEstimator(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True)

            if isinstance(real_sat.actuators[0],MTQ):
                control_laws =  [NoControl(est.sat),Bdot(1e8,est.sat),BdotEKF(1e4,est.sat,maintain_RW = True,include_disturbances=True),Lovera([0.01,50,50],est.sat,include_disturbances=dist_control,quatset=(len(quatset_type)>0),quatset_type = quatset_type,calc_av_from_quat = True),WisniewskiSliding([np.eye(3)*0.002,np.eye(3)*0.003],est.sat,include_disturbances=dist_control,calc_av_from_quat = True,quatset=(len(quatset_type)>0),quatset_type = quatset_type),TrajectoryMPC(mpc_gain_info,est.sat),Trajectory_Mag_PD([0.01,50,50],est.sat,include_disturbances=dist_control,calc_av_from_quat = True),TrajectoryLQR([],est.sat)]
            else:
                control_laws =  [NoControl(est.sat),Magic_PD([np.eye(3)*200,np.eye(3)*5.0],est.sat,include_disturbances=dist_control,quatset=(len(quatset_type)>0),quatset_type = quatset_type,calc_av_from_quat = True),TrajectoryLQR([],est.sat),RWBdot(est.sat),TrajectoryMPC(mpc_gain_info,est.sat),TrajectoryLQR([],est.sat)]

            orbit_estimator = OrbitalEstimator(est_sat,state_estimate = Orbital_State(0,np.ones(3),np.ones(3)))
            # adcsys = ADCS(est_sat,orbit_estimator,est,control_laws,use_planner = False,planner = None,planner_settings = None,goals=goals,prop_schedule=None,dt = 1,control_dt = None,tracking_LQR_formulation = 0)

            planner_settings = PlannerSettings(est_sat,tvlqr_len = 250,tvlqr_overlap = 100,dt_tp = 1,precalculation_time = 10,traj_overlap = 500,debug_plot_on = False,bdot_on = 1,
                        include_gg = True,include_resdipole = False, include_prop = False, include_drag = False, include_srp = False, include_gendist = False)
            planner_settings.maxIter = 2000
            planner_settings.maxIter2 = 2000
            planner_settings.gradTol = 1.0e-6#1.0e-6#1e-9#1e-09
            planner_settings.costTol =      1.0e-4#1.0e-5#1e-6#0.0000001
            planner_settings.ilqrCostTol =  1.0e-2#1.0e-3#1e-4#0.000001
            planner_settings.zCountLim = 5#10
            planner_settings.maxOuterIter = 15#25#70#50#20#20#25#25#1#15#diop
            planner_settings.maxIlqrIter = 25#40#350#0#100#300#diop0#50#150#50#25#25#1#30
            planner_settings.maxOuterIter2 = 15#25#70#50#20#20#25#25#1#15#diop
            planner_settings.maxIlqrIter2 = 25#40
            planner_settings.default_traj_length = 800
            planner_settings.penInit = 1.0
            planner_settings.penMax = 1.0e30
            planner_settings.penInit2 = 1.0e5
            planner_settings.penScale2 = 20
            planner_settings.penMax2 = 1.0e30
            planner_settings.maxLsIter = 10

            planner_settings.mtq_control_weight = 1000.0
            planner_settings.rw_control_weight = 0.001
            planner_settings.magic_control_weight = 0.0001
            planner_settings.rw_AM_weight = 0.1
            planner_settings.rw_stic_weight = 0.01

            planner_settings.angle_weight = 1.0#0.1#200.0
            planner_settings.angvel_weight = 1.0#0*10.0#0.1#0.01#0*10.0#.01#01#0.000001
            planner_settings.u_weight_mult = 1e4#10.0#1000000.0#1e-1#1e-1 #0*0.0001#0.0#0.0000001
            planner_settings.u_with_mag_weight = 0.0
            planner_settings.av_with_mag_weight = 0.0#0.1
            planner_settings.ang_av_weight = 0.1#0*10.0#0*100.0
            planner_settings.angle_weight_N = 1.0#2000.0
            planner_settings.angvel_weight_N = 1.0#0*10.0#0.1#0*10.0
            planner_settings.av_with_mag_weight_N = 0.0
            planner_settings.ang_av_weight_N = 0.1#0*100.0

            planner_settings.angle_weight2 = 1.0#0.1#100.0
            planner_settings.angvel_weight2 = 10.0#0.1#1e-2#1# 1e-1
            planner_settings.u_weight_mult2 = 1.0#1.0#1.0
            planner_settings.u_with_mag_weight2 = 0.0
            planner_settings.av_with_mag_weight2 = 0.0
            planner_settings.ang_av_weight2 = 0.0#0.01#0*0.2#0.1#0.0001*0
            planner_settings.angle_weight_N2 = 1.0#1000.0
            planner_settings.angvel_weight_N2 = 10.0#0.1#1.0#0*1.0#0.0
            planner_settings.av_with_mag_weight_N2 = 0.0
            planner_settings.ang_av_weight_N2 = 0.0#0.01#1.0#0*0.2

            planner_settings.angle_weight_tvlqr = 1.0#10.0#30#0.1#10.0
            planner_settings.angvel_weight_tvlqr = 10.0#0.1#0.1#0.01#0#0.001#0.01#0.01#0*0.01#1#$5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0.01#0.00001
            planner_settings.u_weight_mult_tvlqr = 0.1 #1e-2#0.001#0.00001#0.001#0.00000001#0.3#5#0.1#0.000000001#0.0000000001
            planner_settings.u_with_mag_weight_tvlqr = 0.0#0.01
            planner_settings.av_with_mag_weight_tvlqr = 0.1#0.1#0.01#0.0#1.0#5.0#0#0.005#0.01
            planner_settings.ang_av_weight_tvlqr = 0.0#0.02#0*2#0.05#0.01#0.5#0.5#0#10#0.01#0.1#0*0.1#0*10#1000#-0.1#0.0001#0.01
            planner_settings.angle_weight_N_tvlqr = 10.0#10.0#1*500#1#500#200#30#20.0
            planner_settings.angvel_weight_N_tvlqr = 100.0#0.1#$1.0*10#0.1*10#0.1#0.1#0#0.01#1#1#0*0.1#0*0.01#1#5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0*0.00001
            planner_settings.av_with_mag_weight_N_tvlqr = 0.0#1.0#1.0
            planner_settings.ang_av_weight_N_tvlqr = 1.0#0.2#0.1#0*2#5.0#0#0.05*100#0*0.01#0.01#0.5#0.5#0#10#0.01#0.1#0.01#0.0

            planner_settings.regScale = 1.6#1.6#1.6#1.6##2#1.6#4#1.6#2#1.6#1.8#2.0#1.8#5#10#1e3#1.8#2#1.6#1.6#2#1.6#2.0#1.6#3#1.6#5#2#5#1.6#5#1.8
            planner_settings.regMax = 1.0e20
            planner_settings.regMax2 = 1.0e20
            planner_settings.regMin = 1.0e-8#16
            planner_settings.regBump = 10.0#5#1e-3#1e-2#10.0#1e-1#2.0#10.0#1.0#10.0#1.0 #1.0#10.0#$0.01#0.1#2.0#1e-2#0.1#1.0#0.1#10#1e-10#2.0#1.0#1.0#1.0#10.0#100#0.1#10#1.0 #1.0#1.0#50#10#20.0#100.0
            planner_settings.regMinCond = 2 #0 means that regmin is basically ignored, and regulaization goes up and down without bounds, case 1 means regularization is always equal to or greater than regmin, case 2 means if the regularization falls below regmin then it clamps to 0
            planner_settings.regMinCond2 = 2
            planner_settings.regBumpRandAddRatio = 0.0#1e-20#1e-16#1e-3#4e-3#*1e-4
            # planner_settings.useEVmagic = 0;#1 #use the eigendecomposition rather than simple regularization
            # planner_settings.SPDEVreg = 1;#0#1 #regularize/add even if matrix is SPD
            # planner_settings.SPDEVregAll = 0;#0 reg SPD matrix by adding rho*identity matrix (otherwise do the EV magic reg)
            # planner_settings.rhoEVregTest = 1;#1 #test if reset is needed (in EV magic case) by comparing to a multiple of rho (otherwise compare to regmin)
            planner_settings.useDynamicsHess = 0
            # planner_settings.EVregTestpreabs = 1;#0 #complete the reset test before absolute value is taken
            # planner_settings.EVaddreg = 0;#0 #do EV magic by adding a value to the eigs that are too small (otherwise clamp to a minimum value)
            # planner_settings.EVregIsRho = 1; #1 #clamp to or add rho (otherwise regmin)
            planner_settings.dt_tp = 10.0
            planner_settings.dt_tvlqr = 1.0
            planner_settings.useConstraintHess = 0

            planner_settings.control_limit_scale = 0.5
            planner_settings.rho = 0.0#1e-10#0.01#1.0#0.1#1.0#0.1#0.001#1.0#0.01#1.0
            planner_settings.wmax = 1.0/60.0
            planner_settings.considerVectorInTVLQR = 0
            planner_settings.useRawControlCost = False#True
            planner_settings.whichAngCostFunc = 2#2 seems best.0.1#0*2#5.0#0#0.05*100#0*0.01#0.01#0.5#0.5#0#10#0.01#0.1#0.01#0.0
            planner_settings.bdotgain = 10000000/(planner_settings.dt_tp**2)

            adcsys = ADCS(est_sat,orbit_estimator,est,control_laws,use_planner = True,planner = None,planner_settings = planner_settings,goals=goals,prop_schedule=None,dt = 1,control_dt = 1,tracking_LQR_formulation = 0)
            # adcsys.planner.setVerbosity(True)

            state0 = np.zeros(real_sat.state_len)
            state0[0:3] = w0
            state0[3:7] = q0

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*TimedeltaIndex")
                run_sim(orb_file,state0,copy.deepcopy(real_sat),adcsys,tf=tf,dt = dt,alt_title = title,rand=False)
        except Exception as ae:
            if isinstance(ae, KeyboardInterrupt):
                raise
            else:
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
                frame = last_frame().tb_frame
                ns = dict(frame.f_globals)
                ns.update(frame.f_locals)
                code.interact(local=ns)
                pass
