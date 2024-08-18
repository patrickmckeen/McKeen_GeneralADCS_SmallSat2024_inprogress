#estimation results for paper
from attitude_EKF import *
from attitude_UKF import *
from attitude_SRUKF import *
from crassidis_UKF import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
from common_sats import *
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
    if len(less_than_inds)>0:
        converged_ind1 = less_than_inds[0]
        if greater_than_inds[-1]+1<len(anglist):
            converged_ind2 = greater_than_inds[-1]+1

            res.ind_conv = (converged_ind1,converged_ind2,int(0.5*(converged_ind1+converged_ind2)))
            res.time_to_conv = (tlist[converged_ind1],tlist[converged_ind2],tlist[int(0.5*(converged_ind1+converged_ind2))])
            log_angdiff = np.log(anglist[int(0.1*res.ind_conv[2]):int(0.9*res.ind_conv[2])])
            log_t_list = tlist[int(0.1*res.ind_conv[2]):int(0.9*res.ind_conv[2])]
            res.tc_est = np.polyfit(log_angdiff,log_t_list,1)[0]
            res.steady_state_err_mean = tuple([np.mean(anglist[j:]) for j in res.ind_conv])
            res.steady_state_err_max = tuple([np.max(anglist[j:]) for j in res.ind_conv])
        else:
            converged_ind2 = np.nan

            res.ind_conv = (converged_ind1,np.nan,np.nan)
            res.time_to_conv = (tlist[converged_ind1],np.nan,np.nan)
            log_angdiff = np.log(anglist[int(0.1*res.ind_conv[0]):int(0.9*res.ind_conv[0])])
            log_t_list = tlist[int(0.1*res.ind_conv[0]):int(0.9*res.ind_conv[0])]
            res.tc_est = np.polyfit(log_angdiff,log_t_list,1)[0]
            res.steady_state_err_mean = (np.mean(anglist[res.ind_conv[0]:]) ,np.nan,np.nan)
            res.steady_state_err_max =  (np.max(anglist[res.ind_conv[0]:]) ,np.nan,np.nan)
    else:
        res.ind_conv = (np.nan,np.nan,np.nan)
        res.time_to_conv = (np.nan,np.nan,np.nan)
        res.tc_est = np.nan
        res.steady_state_err_mean = np.nan
        res.steady_state_err_max = np.nan
    return res


#Crassidis replication case 2
def crassidis_UKF_attitude_errors_replication(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True)
    est_sat = create_Crassidis_UKF_sat(real=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[7:10] = 0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(0.2*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 5*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*10)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 10)
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val

        #control law
        control = np.zeros(0)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-12, atol=1e-15)#,jac = ivp_jac)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_repl_attOnly_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")
    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist

    #
    # ttt = copy.deepcopy(real_sat.sensors[0])
    # pickling_test(ttt)
    #
    # breakpoint()


    # save_real = copy.deepcopy(real_sat)
    # # breakpoint()
    # dirlist = dir(save_real)
    # print(dirlist)
    # for j in dirlist:
    #     print(j)
    #     att = getattr(save_real,j)
    #     try:
    #         with open("test", "wb") as fp:   #Pickling
    #             pickle.dump(att, fp)
    #         print("OK ",j)
    #     except Exception as e:
    #         print("      BREAKS  ",j)
    #         print("     ")
    #         print(e)
    #         print("     ")
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis replication case 3
def crassidis_UKF_attNbias_errors_replication(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True)
    est_sat = create_Crassidis_UKF_sat(real=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 5*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*10)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 10)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator

        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val

        #control law
        control = np.zeros(0)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-12, atol=1e-15)#,jac = ivp_jac)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_repl_attNbias_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "disturbed_world" has disturbances
def crassidis_UKF_disturbed_world(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,estimate_dipole=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 0.5*dt*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*dt)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 40
    ka = 1

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # # print('real dipole ',real_sat.disturbances[0].main_param)
        # # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        control = np.zeros(0)#est_sat.disturbances[0
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_disturbed_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "disturbed_world" has disturbances
def crassidis_UKF_disturbed_world_mine(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,estimate_dipole=True)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(13)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(1)**2.0)
    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    int_cov =  (dt**2.0)*block_diag(np.eye(3)*1e-16,1e-10*np.eye(3),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)

    est.use_cross_term = True
    est.al = 1e-2#e-3#1#1e-1#al# 1e-1
    est.kap =  12#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 40
    ka = 1

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # # print('real dipole ',real_sat.disturbances[0].main_param)
        # # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        control = np.zeros(0)#est_sat.disturbances[0
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "mine_disturbed_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "baby_world" has disturbances and sun sensors, but not control or biases
def crassidis_UKF_baby_world(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,include_sbias=False,estimate_sun_bias=False,estimate_dipole=False,mtm_scale=1e4)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,include_sbias=False,estimate_sun_bias=False,estimate_dipole=False,mtm_scale=1e4)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 0.5*dt*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*dt)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 40
    ka = 1

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # # print('real dipole ',real_sat.disturbances[0].main_param)
        # # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        control = np.zeros(0)#est_sat.disturbances[0
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    gbiasdiff = (180/np.pi)*(est_state_hist[:,7:10]-state_hist[:,7:10])
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_baby_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")

    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")
    # plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")

    plot_the_thing(est_state_hist[:,7:10]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,7:10]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")


    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "baby_world" has disturbances and sun sensors, but not control or biases
def crassidis_UKF_baby_world_mine(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,include_sbias=False,estimate_sun_bias=False,estimate_dipole=True,mtm_scale=1e4)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,estimate_mtm_bias=False,include_mtmbias=False,include_sbias=False,estimate_sun_bias=False,estimate_dipole=True,mtm_scale=1e4)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(13)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(1)**2.0)
    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    int_cov =  block_diag(np.eye(3)*1e-16,1e-10*np.eye(3),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)*dt**2.0

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.use_cross_term = True
    est.al = 1e-2#e-3#1#1e-1#al# 1e-1
    est.kap =  12#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est_sat0 = copy.deepcopy(est_sat)

    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 40
    ka = 1

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # # print('real dipole ',real_sat.disturbances[0].main_param)
        # # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        control = np.zeros(0)#est_sat.disturbances[0
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,7:10]-state_hist[:,7:10])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    base_title = "mine_baby_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")


    plot_the_thing(est_state_hist[:,7:10]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,7:10]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "ctrl_world" has disturbances and control, but not bias or sunsensors
def crassidis_UKF_ctrl_world(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*10
    tlim0 = 60*60*1
    tlim1 = 60*60*3
    tlim2 = 60*60*7

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_dipole=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 0.5*dt*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*dt)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 50
    ka = 1.0

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        #control law
        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            ud = -kw*w_err
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mag_max)
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,7:10]-state_hist[:,7:10])

    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_ctrl_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")


    plot_the_thing(est_state_hist[:,7:10]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,7:10]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "ctrl_world" has disturbances and control, but not bias bias or sunsensors
def crassidis_UKF_ctrl_world_mine(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*10
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_dipole=True)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = False,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_dipole=True)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(13)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(1)**2.0)
    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    int_cov =  dt*dt*block_diag(np.eye(3)*1e-16,1e-10*np.eye(3),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)

    est.use_cross_term = True
    est.al = 1e-2#e-3#1#1e-1#al# 1e-1
    est.kap =  12#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 50
    ka = 1.0

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        #control law
        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            ud = -kw*w_err
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mag_max)
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,7:10]-state_hist[:,7:10])

    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "mine_ctrl_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")


    plot_the_thing(est_state_hist[:,7:10]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,7:10]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "real_world"
def crassidis_UKF_real_world(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*10
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = True,estimate_mag_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = True,estimate_mag_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 0.5*dt*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*dt)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 50
    ka = 5.0

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        #control law

        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = est_state[7:10]
        elif t<tlim0:
            ud = -kw*w_err
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,7:10]-state_hist[:,13:16])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_real_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")


    plot_the_thing(est_state_hist[:,7:10]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")
    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")


    # plot_the_thing(est_state_hist[:,7:10],title = "Est Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_abias_plot")
    plot_the_thing(state_hist[:,7:10],title = "Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/abias_plot")
    # plot_the_thing(est_state_hist[:,7:10]-state_hist[:,7:10],title = "Magic Bias Error",xlabel='Time (s)',ylabel = 'Magic Bias Error (N)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/aberr_plot")

    # plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    # plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,19:22],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")


    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

def crassidis_UKF_real_world_no_abias(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*10
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,include_magbias = False,estimate_mag_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = False,include_magbias = False,estimate_mag_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 0.5*dt*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*dt)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 50
    ka = 5.0

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        #control law

        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = est_state[7:10]
        elif t<tlim0:
            ud = -kw*w_err
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,7:10]-state_hist[:,10:13])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_real_world_no_abias_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")


    plot_the_thing(est_state_hist[:,7:10]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,10:13]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")
    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")


    # plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,7:10],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    # plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,13:16],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")


    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "real_world"
def crassidis_UKF_real_world_no_dipole(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*10
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,include_mtqbias = False,estimate_mtq_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 0.5*dt*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*dt)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 1000
    ka = 10

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        #control law

        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = est_state[7:10]
        elif t<tlim0:
            ud = -kw*w_err
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,7:10]-state_hist[:,13:16])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_real_world_no_dipole_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")


    plot_the_thing(est_state_hist[:,7:10]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")
    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")


    # plot_the_thing(est_state_hist[:,7:10],title = "Est Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_abias_plot")
    plot_the_thing(state_hist[:,7:10],title = "Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/abias_plot")
    # plot_the_thing(est_state_hist[:,7:10]-state_hist[:,7:10],title = "Magic Bias Error",xlabel='Time (s)',ylabel = 'Magic Bias Error (N)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/aberr_plot")

    # plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    # plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,19:22],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    # plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")


    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#my controller on the Crassidis test
def crassidis_UKF_mine(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*5
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = True,estimate_mag_bias = True,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole = True,mtm_scale = 1e4)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = True,estimate_mag_bias = True,estimate_mtm_bias=True,include_mtmbias=True,estimate_sun_bias=True,estimate_dipole = True,mtm_scale = 1e4)#,mag_std=np.zeros(3))

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10+6+3+3)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    # estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(1e-3)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.array([1])*(0.05*0.3)**2.0,np.eye(3)*(1)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    mtm_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,MTM)])
    mtm_std = np.array([j.std for j in est_sat.sensors if isinstance(j,MTM)])
    sun_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,SunSensorPair)])
    sun_std = np.array([j.std for j in est_sat.sensors if isinstance(j,SunSensorPair)])

    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*1.5**2.0,np.eye(3)*(1e-3)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.05*0.3)**2.0,np.eye(3)*(1)**2.0)
    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    werrcov = 0#1e-20#0#1e-20#0#1e-20
    mrperrcov = 0#1e-10#16
    int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*(1/dt)*werrcov,0.5*np.eye(3)*werrcov],[0.5*np.eye(3)*werrcov,(1/3)*np.eye(3)*dt*werrcov + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)

    est_sat0 = copy.deepcopy(est_sat)

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.use_cross_term = True
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.al = 0.99#0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = 3-21#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    bdotgain = 1e10
    kw = 50
    ka = 1.0

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        print(t,ind)

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])
        real_abias = np.concatenate([j.bias for j in real_sat.actuators if j.has_bias])
        real_dist = np.concatenate([j.main_param for j in real_sat.disturbances if j.time_varying])
        real_full_state = np.concatenate([state.copy(),real_abias.copy(),real_sbias.copy(),real_dist.copy()])


        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',real_sbias)
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm(4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0))
        # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi)
        print('gb ',norm(est_state[13:16]-real_sbias[3:6]))
        print((est_state[13:16]-real_sbias[3:6]))
        print(autovar[12:15]*180.0/math.pi)

        print('mb ',norm(est_state[10:13]-real_sbias[0:3]))
        print((est_state[10:13]-real_sbias[0:3]))
        print(autovar[9:12])

        print('sb ',norm(est_state[16:19]-real_sbias[6:9]))
        print((est_state[16:19]-real_sbias[6:9]))
        print(autovar[15:18])

        print('magb ',norm(est_state[7:10]-real_abias))
        print((est_state[7:10]-real_abias))
        print(autovar[7:10])

        print('dip ',norm(est_state[19:22]-real_dist))
        print(est_state[19:22]-real_dist)
        print(autovar[18:21])
        errvec = est_state-real_full_state
        errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(real_full_state[3:7]),est_state[3:7]),5),errvec[7:]])
        print('err vec ',errvec )
        # coverr = np.sqrt(errvec.T@np.linalg.inv(est.use_state.cov)@errvec)
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        # print('cov err ',coverr)
        # print(real_sat.last_act_torq)
        # print(real_sat.last_dist_torq)

        try:
            ea,ee = np.linalg.eig(est.use_state.cov)
            sqrt_invcov = ee@np.diagflat(1/np.sqrt(ea))#*np.linalg.cholesky(invcov)
            # sqrt_invcov2 = np.linalg.cholesky(np.linalg.inv(est.use_state.cov))
            # sqrt_invcov3 = np.linalg.inv(np.linalg.cholesky(est.use_state.cov))
            # sqrt_cov = np.linalg.cholesky(est.use_state.cov)
            # sqrt_cov = np.linalg.cholesky(est.use_state.cov)
            # print(sum(np.abs(ee@np.diagflat(ea)@ee.T - invcov)))
            # breakpoint()
            # print(sqrt_invcov@sqrt_invcov.T - invcov)
            # print('scaled state err', sqrt_invcov@errvec)
            sc_err = sqrt_invcov.T@errvec
            biggest = np.argmax(np.abs(sc_err))
            print('cov-eig state err',sc_err)
            tt = ee@((np.sign(sc_err)*sc_err[biggest]*(sc_err/sc_err[biggest])**2.0)*ea)
            print('rescaled state err',np.sign(tt)*np.sqrt(np.abs(tt)))
            print('rel rescaled state err',np.sign(tt)*np.sqrt(np.abs(tt))/errvec)
            # print('rescaled state err', ee@sqrt_invcov.T@errvec)
            print('biggest dir state err', np.dot(ee[:,biggest],errvec)*ee[:,biggest],sc_err[biggest])
            print('rel biggest dir state err', np.dot(ee[:,biggest],errvec)*ee[:,biggest]/errvec,sc_err[biggest])
            # print('rescaled state err', ee@np.diagflat(np.sqrt(ea))@sqrt_invcov.T@errvec)
            print('overall err', norm(sc_err))
            # print('scaled state err3', sqrt_cov@errvec)


            # print('scaled state err4', sqrt_invcov3@errvec)
            # print('rescaled state err4', (1/np.mean(matrix_row_norm(sqrt_invcov3)))*matrix_row_normalize(sqrt_invcov3).T@sqrt_invcov3@errvec)
            # print('overall err', math.sqrt(np.dot(sqrt_invcov3@errvec,sqrt_invcov3@errvec)))
        #
        #
        #
        except:
            pass

        #control law

        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = est_state[7:10]
        elif t<tlim0:
            ud = -kw*w_err
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)

        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,13:16]-state_hist[:,13:16])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "mine_real_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    plot_the_thing(est_state_hist[:,13:16]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")


    plot_the_thing(est_state_hist[:,7:10],title = "Est Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_abias_plot")
    plot_the_thing(state_hist[:,7:10],title = "Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/abias_plot")
    plot_the_thing(est_state_hist[:,7:10]-state_hist[:,7:10],title = "Magic Bias Error",xlabel='Time (s)',ylabel = 'Magic Bias Error (N)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/aberr_plot")

    plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")

    plot_the_thing(est_state_hist[:,19:22],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,19:22],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")

    plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#my controller on the Crassidis test
def crassidis_UKF_mine_no_abias(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*5
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole = True)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_mtm_bias=True,include_mtmbias=True,estimate_sun_bias=True,estimate_dipole = True)#,mag_std=np.zeros(3))

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10+6+3)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    # estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(1e-7)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.array([1])*(0.05*0.3)**2.0,np.eye(3)*(1)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    mtm_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,MTM)])
    mtm_std = np.array([j.std for j in est_sat.sensors if isinstance(j,MTM)])
    sun_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,SunSensorPair)])
    sun_std = np.array([j.std for j in est_sat.sensors if isinstance(j,SunSensorPair)])

    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(1e-7)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.05*0.3)**2.0,np.eye(3)*(1)**2.0)
    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    werrcov = 0#1e-20#0#1e-20#0#1e-20
    mrperrcov = 0#1e-10#16
    int_cov =  dt*block_diag(dt*np.block([[np.eye(3)*(1/dt)*werrcov,0.5*np.eye(3)*werrcov],[0.5*np.eye(3)*werrcov,(1/3)*np.eye(3)*dt*werrcov + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)

    est_sat0 = copy.deepcopy(est_sat)

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.use_cross_term = True
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.al = 0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = 3-18#-10#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    bdotgain = 1e10
    kw = 50
    ka = 5.0

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        print(t,ind)

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        print(sens)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        est_sens = est.sat.sensor_values(est_state,est_vecs)
        print(est_sens)

        print(normalize(sens[0:3]),normalize(est_sens[0:3]),np.arccos(np.dot(normalize(sens[0:3]),normalize(est_sens[0:3])))*(180.0/math.pi))

        print(normalize(sens[6:9]),normalize(est_sens[6:9]),np.arccos(np.dot(normalize(sens[6:9]),normalize(est_sens[6:9])))*(180.0/math.pi))



        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real dip ', real_sat.disturbances[2].main_param)
        print('est dip ', est_state[-3:])
        # print('real abias ',np.concatenate([j.bias for j in real_sat.actuators if j.has_bias]))
        # print('est abias ',est_state[7:10])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        # print(real_sat.last_act_torq)
        # print(real_sat.last_dist_torq)

        #control law

        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = est_state[7:10]
        elif t<tlim0:
            ud = -kw*w_err
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)

        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,10:13]-state_hist[:,10:13])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "mine_real_world_no_abias_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    plot_the_thing(est_state_hist[:,10:13]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,10:13]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")


    plot_the_thing(est_state_hist[:,7:10],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,7:10],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    plot_the_thing(est_state_hist[:,7:10]-state_hist[:,7:10],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")

    plot_the_thing(est_state_hist[:,13:16],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,13:16],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    plot_the_thing(est_state_hist[:,13:16]-state_hist[:,13:16],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")

    plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#my controller on the Crassidis test
def crassidis_UKF_mine_no_dipole(orb=None,t0 = 0, tf = 60*60*8,dt=10,extra_tag = None,mrperrcov = 0,werrcov = 0,kap = 0,mtm_scale=1e3,jmult = 1):
    np.random.seed(1)
    tlim00 = 60*5
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    # mtm_scale = 1e3
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole = False,mtm_scale =  mtm_scale,jmult = jmult)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtq_bias = True,estimate_mtm_bias=True,include_mtmbias=True,estimate_sun_bias=True,estimate_dipole = False,mtm_scale = mtm_scale,jmult = jmult)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10+6+3)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    # estimate[0:3] = 0.1*(math.pi/180.0)/3600.0
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    # gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    # gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    # mtm_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,MTM)])
    # mtm_std = np.array([j.std for j in est_sat.sensors if isinstance(j,MTM)])
    # sun_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,SunSensorPair)])
    # sun_std = np.array([j.std for j in est_sat.sensors if isinstance(j,SunSensorPair)])

    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0/5)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*5**2.0,np.eye(3)*(1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.1*0.3)**2.0)
    alt_cov_estimate = block_diag(np.eye(3)*(math.pi/180.0/10)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,0.25*np.eye(3)*5**2.0,np.eye(3)*(0.5*1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.5*0.1*0.3)**2.0)

    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    # werrcov = 0#1e-20#1e-20#co*()**2#1e-16#1e-12#1e-20#0#1e-20#0#1e-20
    # mrperrcov = 1e-30#1e-20# 1e-16#1e-20#1e-10#16
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*(1/dt)*werrcov,0.5*np.eye(3)*werrcov],[0.5*np.eye(3)*werrcov,(1/3)*np.eye(3)*dt*werrcov + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    int_cov =  dt*block_diag(dt*np.block([[np.eye(3)*(1/dt)*werrcov,0.5*np.eye(3)*werrcov],[0.5*np.eye(3)*werrcov,(1/3)*np.eye(3)*dt*werrcov + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    # breakpoint()
    est_sat0 = copy.deepcopy(est_sat)

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.use_cross_term = True
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.al = 0.99#1e-3#0.99#0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = kap#3-18#0#3-21#6-18#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    # est.bet = 2
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    bdotgain = 1e10
    kw = 1000*jmult
    ka = 10*jmult

    mag_max = [j.max*math.sqrt(jmult) for j in est_sat.actuators]
    while t<tf:
        print("======================================")
        print(t,ind)

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        # print(sens)

        #run estimator
        tt0 = time.process_time()
        aa,extra = est.update(control,sens,orbt)
        # pred_dyn_state = extra[0]
        # psens = extra[1]
        # sens1 = extra[2]
        tt1 = time.process_time()
        est_state = est.use_state.val
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        est_sens = est.sat.sensor_values(est_state,est_vecs)
        # print(est_sens)

        # print('est sens')
        # print(normalize(sens[0:3]),normalize(est_sens[0:3]),np.arccos(np.dot(normalize(sens[0:3]),normalize(est_sens[0:3])))*(180.0/math.pi))
        # print(normalize(sens[6:9]),normalize(est_sens[6:9]),np.arccos(np.dot(normalize(sens[6:9]),normalize(est_sens[6:9])))*(180.0/math.pi))
        # print('predictive step sens')
        # print(normalize(sens[0:3]),normalize(psens[0:3]),np.arccos(np.dot(normalize(sens[0:3]),normalize(psens[0:3])))*(180.0/math.pi))
        # print(normalize(sens[6:9]),normalize(psens[6:9]),np.arccos(np.dot(normalize(sens[6:9]),normalize(psens[6:9])))*(180.0/math.pi))
        # # print('weighted pred step sens')
        # print(normalize(sens[0:3]),normalize(sens1[0:3]),np.arccos(np.dot(normalize(sens[0:3]),normalize(sens1[0:3])))*(180.0/math.pi))
        # print(normalize(sens[6:9]),normalize(sens1[6:9]),np.arccos(np.dot(normalize(sens[6:9]),normalize(sens1[6:9])))*(180.0/math.pi))

        # print('real state ',state)
        # print('est  state ',est_state[0:7])
        # print('pred state ',pred_dyn_state[0:7])
        # print('real dip ', real_sat.disturbances[2].main_param)
        # print('est dip ', est_state[-3:])
        print('real abias ',np.concatenate([j.bias for j in real_sat.actuators if j.has_bias]))
        print('est abias ',est_state[7:10])
        # print('pred abias ',pred_dyn_state[7:10])
        # real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])
        # print('real sbias',real_sbias)
        # print('est sbias ',est_state[10:19])
        # print('pred sbias ',pred_dyn_state[10:19])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print('est state')
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)
        # print('gb err est ',norm(est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # print('av err est ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        # print('gb+av err est ',norm(est_state[13:16]+est_state[0:3]-state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # # print('gb-av err est ',norm(est_state[13:16]-est_state[0:3]+state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # dsens = sens-est_sens
        # print(norm(dsens[0:3]),norm(dsens[3:6])*(180.0/math.pi),norm(dsens[6:9]),dsens/sens)
        # print('pred state')
        # print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(pred_dyn_state[3:7],state[3:7]),-1,1)**2.0 ))
        # print('real,pred av ',(norm(state[0:3])*180.0/math.pi),norm(pred_dyn_state[0:3])*180.0/math.pi)
        # print('gb err pred ',norm(pred_dyn_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # print('av err pred ',(norm(pred_dyn_state[0:3]-state[0:3])*180.0/math.pi))
        # print('gb+av err pred ',norm(pred_dyn_state[13:16]+pred_dyn_state[0:3]-state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # # print('gb-av err pred ',norm(pred_dyn_state[13:16]-pred_dyn_state[0:3]+state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # dsens = sens-psens
        # # pred_av_sens_err = dsens[3:6]
        # #
        # # print(pred_av_sens_err*180.0/math.pi,norm(pred_av_sens_err)*180.0/math.pi)
        # # print('gb pred to est adj ',(est_state[13:16]-pred_dyn_state[13:16])*180.0/math.pi,norm((est_state[13:16]-pred_dyn_state[13:16])*180.0/math.pi),norm(-pred_av_sens_err+est_state[13:16]-pred_dyn_state[13:16])*180.0/math.pi)
        # # print('av pred to est adj ',(est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi,norm((est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi),norm(-pred_av_sens_err+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi)
        # # print('gb+av pred to est adj ',(est_state[13:16]-pred_dyn_state[13:16]+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi,norm((est_state[13:16]-pred_dyn_state[13:16]+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi),norm((-pred_av_sens_err+est_state[13:16]-pred_dyn_state[13:16]+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi))
        #
        # print(norm(dsens[0:3]),norm(dsens[3:6])*(180.0/math.pi),norm(dsens[6:9]),dsens/sens)
        # # dsens = sens-sens1
        # # print(norm(dsens[0:3]),norm(dsens[3:6])*(180.0/math.pi),norm(dsens[6:9]),dsens/sens)
        # #
        # if ind>1:
        #     print(state[0:3]*180/math.pi)
        #     print(est_state[0:3]*180/math.pi)
        #     print(quat_log(quat_mult(quat_inv(state_hist[ind-1,3:7]),state[3:7]))*180/math.pi/dt)
        #     print(quat_log(quat_mult(quat_inv(est_state_hist[ind-1,3:7]),est_state[3:7]))*180/math.pi/dt)
            # print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
            # print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(quat_mult(est_state_hist[ind-1,3:7],rot_exp(dt*est_state_hist[ind-1,0:3])),state[3:7]),-1,1)**2.0 ))

        #control law

        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = est_state[7:10]
        elif t<tlim0:
            ud = -kw*w_err
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
            # ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)

        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,13:16]-state_hist[:,13:16])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "mine_real_world_no_dipole_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")

    plot_the_thing(est_state_hist[:,13:16]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(gbiasdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_loggb_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")


    plot_the_thing(est_state_hist[:,7:10],title = "Est Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_abias_plot")
    plot_the_thing(state_hist[:,7:10],title = "Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/abias_plot")
    plot_the_thing(est_state_hist[:,7:10]-state_hist[:,7:10],title = "Magic Bias Error",xlabel='Time (s)',ylabel = 'Magic Bias Error (N)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/aberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,7:10]-state_hist[:,7:10])),title = "Log Magic Bias Error",xlabel='Time (s)',ylabel = 'Log Magic Bias Error (log N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logab_plot")


    plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,10:13]-state_hist[:,10:13])),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logmb_plot")

    plot_the_thing(est_state_hist[:,16:19],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,16:19],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    plot_the_thing(est_state_hist[:,16:19]-state_hist[:,16:19],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,16:19]-state_hist[:,16:19])),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logsb_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    # plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist

    labels = ["al","kap","bet","dt","jmult","werrcov","mrpcov","mtmscale","covest0","intcov","conv time","tc","last 100 ang err mean","last 100 ang err max","last 100 av err mean","last 100 av err max"]
    info = [est.al,est.kap,est.bet,dt,jmult,werrcov,mrperrcov,mtm_scale,np.diag(cov_estimate.copy()),int_cov.copy(),metrics.time_to_conv,metrics.tc_est,np.mean(angdiff[-100:]),np.amax(angdiff[-100:]),np.mean(matrix_row_norm(avdiff)[-100:]),np.amax(matrix_row_norm(avdiff)[-100:])]
    with open("paper_test_files/"+base_title+"/info", 'w') as f:
        for j in range(len(labels)):
            f.write(labels[j])
            f.write(": ")
            f.write(str(info[j]))
            f.write("\n")
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#my controller on the Crassidis test
def crassidis_UKF_mine_no_dipole_Jmod(orb=None,t0 = 0, tf = 60*60*8,dt=10,extra_tag = None,mrperrcov = 0,werrcovmult = 0,kap = 0,mtm_scale=1e3,jmult = 1,invjpow = 3,scaled_UT = False,al = 0.99,bet = 2,xtermmult = -0.5,ang_werr_mult = 1/3.0,useSR = False,est_ctrl_cov = None,av_init_cov_est = (math.pi/180.0/5)**2.0):
    np.random.seed(1)
    tlim00 = 60*5
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    # mtm_scale = 1e3
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole = False,mtm_scale =  mtm_scale,jmult = jmult,mtq_std = est_ctrl_cov)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtq_bias = True,estimate_mtm_bias=True,include_mtmbias=True,estimate_sun_bias=True,estimate_dipole = False,mtm_scale = mtm_scale,jmult = jmult,mtq_std = est_ctrl_cov)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10+6+3)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    # estimate[0:3] = 0.1*(math.pi/180.0)/3600.0
    estimate[14] = 20.1*(math.pi/180.0)/3600.0
    # gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    # gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    # mtm_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,MTM)])
    # mtm_std = np.array([j.std for j in est_sat.sensors if isinstance(j,MTM)])
    # sun_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,SunSensorPair)])
    # sun_std = np.array([j.std for j in est_sat.sensors if isinstance(j,SunSensorPair)])

    cov_estimate = block_diag(np.eye(3)*av_init_cov_est,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*5**2.0,np.eye(3)*(1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.1*0.3)**2.0)
    # alt_cov_estimate = block_diag(np.eye(3)*(math.pi/180.0/10)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,0.25*np.eye(3)*5**2.0,np.eye(3)*(0.5*1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.5*0.1*0.3)**2.0)

    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    # werrcov = 0#1e-20#1e-20#co*()**2#1e-16#1e-12#1e-20#0#1e-20#0#1e-20
    # mrperrcov = 1e-30#1e-20# 1e-16#1e-20#1e-10#16
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*(1/dt)*werrcov,0.5*np.eye(3)*werrcov],[0.5*np.eye(3)*werrcov,(1/3)*np.eye(3)*dt*werrcov + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    werr = (np.linalg.matrix_power(est_sat.invJ.copy(),int(invjpow)))*1e-16*werrcovmult
    int_cov =  dt*block_diag(dt*np.block([[(1/dt)*werr,xtermmult*werr],[xtermmult*werr,ang_werr_mult*dt*werr + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    # breakpoint()
    est_sat0 = copy.deepcopy(est_sat)

    if useSR:
        est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    else:
        est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.use_cross_term = True
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.al = al#1e-3#0.99#0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = kap#3-18#0#3-21#6-18#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    # est.bet = 2
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    est.scaled_UT = scaled_UT
    est.bet = bet
    est.vec_mode = 6
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    bdotgain = 1e10
    kw = 1000*jmult
    ka = 10*jmult
    real_vecs = os_local_vecs(orbt,state[3:7])
    sens = real_sat.sensor_values(state,real_vecs)

    # est.initialize_estimate(sens,[np.array([0,1,2]),np.array([6,7,8])],[orbt.B*est.sat.attitude_sensors[0].scale,est_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)],np.array([3,4,5]),orbt)


    mag_max = [j.max*math.sqrt(jmult) for j in est_sat.actuators]
    while t<tf:
        print("======================================")
        print(t,ind)

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        real_sbias = np.concatenate([j.bias for j in real_sat.attitude_sensors if j.has_bias])
        real_abias = np.concatenate([j.bias for j in real_sat.actuators if j.has_bias])
        # print(sens)

        #run estimator
        tt0 = time.process_time()
        aa,extra = est.update(control,sens,orbt)
        pred_dyn_state = extra.mean1
        # psens = extra[1]
        # sens1 = extra[2]
        tt1 = time.process_time()
        est_state = est.use_state.val
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        est_sens = est.sat.sensor_values(est_state,est_vecs)


        real_full_state = np.concatenate([state.copy(),real_abias.copy(),real_sbias.copy()])
        # print(est_sens)

        # print('est sens')
        # print(normalize(sens[0:3]),normalize(est_sens[0:3]),np.arccos(np.dot(normalize(sens[0:3]),normalize(est_sens[0:3])))*(180.0/math.pi))
        # print(normalize(sens[6:9]),normalize(est_sens[6:9]),np.arccos(np.dot(normalize(sens[6:9]),normalize(est_sens[6:9])))*(180.0/math.pi))
        # print('predictive step sens')
        # print(normalize(sens[0:3]),normalize(psens[0:3]),np.arccos(np.dot(normalize(sens[0:3]),normalize(psens[0:3])))*(180.0/math.pi))
        # print(normalize(sens[6:9]),normalize(psens[6:9]),np.arccos(np.dot(normalize(sens[6:9]),normalize(psens[6:9])))*(180.0/math.pi))
        # # print('weighted pred step sens')
        # print(normalize(sens[0:3]),normalize(sens1[0:3]),np.arccos(np.dot(normalize(sens[0:3]),normalize(sens1[0:3])))*(180.0/math.pi))
        # print(normalize(sens[6:9]),normalize(sens1[6:9]),np.arccos(np.dot(normalize(sens[6:9]),normalize(sens1[6:9])))*(180.0/math.pi))

        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        unbiased_sens = sens-est_state[10:]
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('pred state ',pred_dyn_state[0:7])
        # print('real dip ', real_sat.disturbances[2].main_param)
        # print('est dip ', est_state[-3:])
        print('real abias ',real_abias)
        print('est  abias ',est_state[7:10])
        print('pred abias ',pred_dyn_state[7:10])
        print('real sbias ',real_sbias)
        print('est  sbias ',est_state[10:])
        print('pred sbias ',pred_dyn_state[10:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print('est state')
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi)-np.pi))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi) - np.pi))
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi,4*norm(np.arctan(autovar[3:6]/2.0))*180.0/math.pi)
        sbvec_est = unbiased_sens[6:9]
        bbvec_est = unbiased_sens[0:3]
        srvec_real = real_sat.attitude_sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)
        brvec_real = est.sat.attitude_sensors[0].scale*orbt.B
        print('ang to S/B est ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(two_vec_to_quat(srvec_real,brvec_real,sbvec_est,bbvec_est),state[3:7]),-1,1)**2.0 ))
        print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)
        print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)

        print('ab  ',norm(est_state[7:10]-real_abias))
        print((est_state[7:10]-real_abias))
        print(autovar[6:9])

        print('gb ',norm(est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        print((est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        print(autovar[12:15]*180.0/math.pi)

        print('mb (x1e8/scale) ',norm(est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        print((est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        print(autovar[9:12]*1e8/est.sat.attitude_sensors[0].scale)

        print('sb (x1e4) ',norm(est_state[16:19]-real_sbias[6:9])*1e4)
        print((est_state[16:19]-real_sbias[6:9])*1e4)
        print(autovar[15:18]*1e4)

        errvec = est_state-real_full_state
        errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),5),errvec[7:]])
        print('err vec ',errvec )
        mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
        print('mahalanobis_dist ',np.sqrt(mahalanobis_dist2))
        print('prob ',chi2.pdf(mahalanobis_dist2,18))
        cd = chi2.cdf(mahalanobis_dist2,18)
        print('cdf ',cd)
        print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))
        print(est.wts_m)
        print(est.wts_c)
        # print('gb err est ',norm(est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # print('av err est ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        # print('gb+av err est ',norm(est_state[13:16]+est_state[0:3]-state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # # print('gb-av err est ',norm(est_state[13:16]-est_state[0:3]+state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # dsens = sens-est_sens
        # print(norm(dsens[0:3]),norm(dsens[3:6])*(180.0/math.pi),norm(dsens[6:9]),dsens/sens)
        # print('pred state')
        # print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(pred_dyn_state[3:7],state[3:7]),-1,1)**2.0 ))
        # print('real,pred av ',(norm(state[0:3])*180.0/math.pi),norm(pred_dyn_state[0:3])*180.0/math.pi)
        # print('gb err pred ',norm(pred_dyn_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # print('av err pred ',(norm(pred_dyn_state[0:3]-state[0:3])*180.0/math.pi))
        # print('gb+av err pred ',norm(pred_dyn_state[13:16]+pred_dyn_state[0:3]-state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # # print('gb-av err pred ',norm(pred_dyn_state[13:16]-pred_dyn_state[0:3]+state[0:3]-real_sbias[3:6])*180.0/math.pi)
        # dsens = sens-psens
        # # pred_av_sens_err = dsens[3:6]
        # #
        # # print(pred_av_sens_err*180.0/math.pi,norm(pred_av_sens_err)*180.0/math.pi)
        # # print('gb pred to est adj ',(est_state[13:16]-pred_dyn_state[13:16])*180.0/math.pi,norm((est_state[13:16]-pred_dyn_state[13:16])*180.0/math.pi),norm(-pred_av_sens_err+est_state[13:16]-pred_dyn_state[13:16])*180.0/math.pi)
        # # print('av pred to est adj ',(est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi,norm((est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi),norm(-pred_av_sens_err+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi)
        # # print('gb+av pred to est adj ',(est_state[13:16]-pred_dyn_state[13:16]+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi,norm((est_state[13:16]-pred_dyn_state[13:16]+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi),norm((-pred_av_sens_err+est_state[13:16]-pred_dyn_state[13:16]+est_state[0:3]-pred_dyn_state[0:3])*180.0/math.pi))
        #
        # print(norm(dsens[0:3]),norm(dsens[3:6])*(180.0/math.pi),norm(dsens[6:9]),dsens/sens)
        # # dsens = sens-sens1
        # # print(norm(dsens[0:3]),norm(dsens[3:6])*(180.0/math.pi),norm(dsens[6:9]),dsens/sens)
        # #
        # if ind>1:
        #     print(state[0:3]*180/math.pi)
        #     print(est_state[0:3]*180/math.pi)
        #     print(quat_log(quat_mult(quat_inv(state_hist[ind-1,3:7]),state[3:7]))*180/math.pi/dt)
        #     print(quat_log(quat_mult(quat_inv(est_state_hist[ind-1,3:7]),est_state[3:7]))*180/math.pi/dt)
            # print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
            # print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(quat_mult(est_state_hist[ind-1,3:7],rot_exp(dt*est_state_hist[ind-1,0:3])),state[3:7]),-1,1)**2.0 ))



        # print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        # print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        # print(autovar[0:3]*180.0/math.pi)
        # print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))


        # print('real state ',state)
        # print('est  state ',est_state[0:7])
        # print('real sbias ',real_sbias)
        # print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        # autovar = np.sqrt(np.diagonal(est.use_state.cov))
        # est_vecs = os_local_vecs(orbt,est_state[3:7])

        # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))


        #control law

        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = est_state[7:10]
        elif t<tlim0:
            ud = -kw*w_err
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
            # ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[real_abias]+[real_sbias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)

        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbiasdiff = (180/np.pi)*(est_state_hist[:,13:16]-state_hist[:,13:16])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "mine_real_world_no_dipole_Jmod_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")

    plot_the_thing(est_state_hist[:,13:16]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(gbiasdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_loggb_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")


    plot_the_thing(est_state_hist[:,7:10],title = "Est Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_abias_plot")
    plot_the_thing(state_hist[:,7:10],title = "Magic Bias",xlabel='Time (s)',norm = True,ylabel = 'Magic Bias (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/abias_plot")
    plot_the_thing(est_state_hist[:,7:10]-state_hist[:,7:10],title = "Magic Bias Error",xlabel='Time (s)',ylabel = 'Magic Bias Error (N)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/aberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,7:10]-state_hist[:,7:10])),title = "Log Magic Bias Error",xlabel='Time (s)',ylabel = 'Log Magic Bias Error (log N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logab_plot")


    plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,10:13]-state_hist[:,10:13])),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logmb_plot")

    plot_the_thing(est_state_hist[:,16:19],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,16:19],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    plot_the_thing(est_state_hist[:,16:19]-state_hist[:,16:19],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,16:19]-state_hist[:,16:19])),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logsb_plot")

    # plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    # plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:

    labels = ["title","al","kap","bet","dt","jmult","invjpow","werrcovmult","mrpcov","scaled_UT","xtermmult","ang_werr_mult","mtmscale","SR","ctrl cov est"]+["covest0","intcov"]+["conv time","tc","last 100 ang err mean","last 100 ang err max","last 100 av err mean","last 100 av err max"]
    info = [base_title,est.al,est.kap,est.bet,dt,jmult,invjpow,werrcovmult,mrperrcov,scaled_UT,xtermmult,ang_werr_mult,mtm_scale,useSR,est_ctrl_cov]+[np.diag(cov_estimate.copy()),int_cov.copy()]+[metrics.time_to_conv,metrics.tc_est,np.mean(angdiff[-100:]),np.amax(angdiff[-100:]),np.mean(matrix_row_norm(avdiff)[-100:]),np.amax(matrix_row_norm(avdiff)[-100:])]
    with open("paper_test_files/"+base_title+"/info", 'w') as f:
        for j in range(len(labels)):
            f.write(labels[j])
            f.write(": ")
            f.write(str(info[j]))
            f.write("\n")
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()


def crassidis_UKF_mine_no_dipole_best_cov0(orb=None,t0 = 0, tf = 60*60*8,dt=10,extra_tag = None,mrperrcov = 0,werrcovmult = 0,kap = 0,mtm_scale=1e3,jmult = 1,invjpow = 3,scaled_UT = False,al = 0.99,bet = 2,xtermmult = -0.5,ang_werr_mult = 1/3.0,useSR = False,est_ctrl_cov = None,av_init_cov_est = (math.pi/180.0/5)**2.0):
    np.random.seed(1)
    tlim00 = 60*5
    tlim0 = np.round(tf/20)
    tlim1 = np.round(tf*1/8)
    tlim2 = np.round(tf*3/8)
    tlim3 = np.round(tf*7/8)

    #real_sat
    # mtm_scale = 1e3
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole = False,mtm_scale =  mtm_scale,jmult = jmult,mtq_std = est_ctrl_cov)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtq_bias = True,estimate_mtm_bias=True,include_mtmbias=True,estimate_sun_bias=True,estimate_dipole = False,mtm_scale = mtm_scale,jmult = jmult,mtq_std = est_ctrl_cov)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10+6+3)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    # estimate[0:3] = 0.1*(math.pi/180.0)/3600.0
    estimate[14] = 20.1*(math.pi/180.0)/3600.0
    # gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    # gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    # mtm_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,MTM)])
    # mtm_std = np.array([j.std for j in est_sat.sensors if isinstance(j,MTM)])
    # sun_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,SunSensorPair)])
    # sun_std = np.array([j.std for j in est_sat.sensors if isinstance(j,SunSensorPair)])

    cov_estimate = block_diag(np.eye(3)*av_init_cov_est,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*5**2.0,np.eye(3)*(1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.1*0.3)**2.0)
    # alt_cov_estimate = block_diag(np.eye(3)*(math.pi/180.0/10)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,0.25*np.eye(3)*5**2.0,np.eye(3)*(0.5*1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.5*0.1*0.3)**2.0)

    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    # werrcov = 0#1e-20#1e-20#co*()**2#1e-16#1e-12#1e-20#0#1e-20#0#1e-20
    # mrperrcov = 1e-30#1e-20# 1e-16#1e-20#1e-10#16
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*(1/dt)*werrcov,0.5*np.eye(3)*werrcov],[0.5*np.eye(3)*werrcov,(1/3)*np.eye(3)*dt*werrcov + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    werr = (np.linalg.matrix_power(est_sat.invJ.copy(),int(invjpow)))*1e-16*werrcovmult
    int_cov =  dt*block_diag(dt*np.block([[(1/dt)*werr,xtermmult*werr],[xtermmult*werr,ang_werr_mult*dt*werr + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    # breakpoint()
    est_sat0 = copy.deepcopy(est_sat)

    if useSR:
        est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    else:
        est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.use_cross_term = True
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.al = al#1e-3#0.99#0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = kap#3-18#0#3-21#6-18#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    # est.bet = 2
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    est.scaled_UT = scaled_UT
    est.bet = bet
    est.vec_mode = 6
    t = t0
    ind = 0
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    bdotgain = 1e10
    kw = 1000*jmult
    ka = 10*jmult
    real_vecs = os_local_vecs(orbt,state[3:7])
    sens = real_sat.sensor_values(state,real_vecs)

    # est.initialize_estimate(sens,[np.array([0,1,2]),np.array([6,7,8])],[orbt.B*est.sat.attitude_sensors[0].scale,est_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)],np.array([3,4,5]),orbt)


    mag_max = [j.max*math.sqrt(jmult) for j in est_sat.actuators]
    print("======================================")
    print(t,ind)

    #simulate sensors
    real_vecs = os_local_vecs(orbt,state[3:7])
    sens = real_sat.sensor_values(state,real_vecs)
    real_sbias = np.concatenate([j.bias for j in real_sat.attitude_sensors if j.has_bias])
    real_abias = np.concatenate([j.bias for j in real_sat.actuators if j.has_bias])
    # print(sens)

    #run estimator
    tt0 = time.process_time()
    aa,extra = est.update(control,sens,orbt)

#Crassidis "cubesat world" -- more serious bias, noise
def crassidis_UKF_cubesat_world(orb=None,t0 = 0, tf = 60*60*8,dt=10):
    np.random.seed(1)
    tlim00 = 60*10
    tlim0 = 60*60*1
    tlim1 = 60*60*3
    tlim2 = 60*60*7

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mag = True,include_magbias = False,estimate_mag_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[8] = 20.1*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 0.5*dt*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*dt)**2.0),np.diagflat(gyro_bsr**2.0))
    est_sat0 = copy.deepcopy(est_sat)

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),10+3+3+3))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    # bdotgain = 1e14
    kw = 1
    ka = 0.1

    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        # print(ind,t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        # print('t est',tt1-tt0)
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print((norm(state[0:3])*180.0/math.pi))

        #control law
        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            ud = -kw*w_err
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mag_max)
        # print(control)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # print('state',state)
        # print(real_sat.last_dist_torq)
        # print('avn',norm(state[0:3])*180.0/math.pi)
        # print(norm(orbt.R))
        # print([j.main_param for j in real_sat.disturbances if j.time_varying])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        # real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        # print('t sim', time.process_time()-tt1)

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])

    base_title = "crassidis_real_world__"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#my base case
def bc_UKF_baseline(orb = None, t0 = 0, tf = 60*90,dt = 1):
    np.random.seed(1)
    tlim00 = max(10,dt*1.5)
    tlim0 = 5*60
    tlim1 = 20*60
    tlim2 = 50*60
    tlim3 = 70*60

    #
    #real_sat
    real_sat = create_BC_sat(real=True,use_dipole = False,include_mtqbias = False,rand=False)
    w0 = random_n_unit_vec(3)*1.0*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    if orb is None:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,use_dipole = False,include_mtqbias = False)
    estimate = np.zeros(16)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*10,np.eye(3)*(1e-3)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0)


    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    werrcov = 0#1e-20#0#1e-20#0#1e-20
    mrperrcov = 0#1e-10#16
    int_cov =  dt*block_diag(dt*np.block([[np.eye(3)*(1/dt)*werrcov,0.5*np.eye(3)*werrcov],[0.5*np.eye(3)*werrcov,(1/3)*np.eye(3)*dt*werrcov + np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))

    mtq_max = [j.max for j in est_sat.actuators]
    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False)

    est.use_cross_term = True
    est.al = 0.99#1e-3#e-3#1#1e-1#al# 1e-1
    est.kap =  3-15#15#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)

    est_sat0 = copy.deepcopy(est_sat)

    bdotgain = 1e8
    kw = 1500
    ka = 10

    while t<tf:

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val

        #control law
        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err =w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = np.zeros(3)
        elif t<tlim0:
            #bdot
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/dt
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat
            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)

    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    base_title = "baseline_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)


    plot_the_thing(est_state_hist[:,10:13]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,10:13]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing((est_state_hist[:,10:13]-state_hist[:,10:13])*180.0/math.pi,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")

    plot_the_thing(est_state_hist[:,7:10],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,7:10],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    plot_the_thing(est_state_hist[:,7:10]-state_hist[:,7:10],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")

    plot_the_thing(est_state_hist[:,13:16],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,13:16],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    plot_the_thing(est_state_hist[:,13:16]-state_hist[:,13:16],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")

    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")
    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)

#crassidis on my base case
def bc_crassidis_UKF(orb = None, t0 = 0, tf = 60*90,dt = 1):
    np.random.seed(1)
    tlim00 = max(10,dt*1.5)
    tlim0 = 5*60
    tlim1 = 20*60
    tlim2 = 50*60
    tlim3 = 70*60

    #
    #real_sat
    real_sat = create_BC_sat(real=True,use_dipole = False,include_mtqbias = False,rand=False)
    w0 = random_n_unit_vec(3)*1.0*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    if orb is None:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,use_dipole = False,include_mtqbias = False)
    estimate = np.zeros(10)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 5*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*10)**2.0),np.diagflat(gyro_bsr**2.0))

    mtq_max = [j.max for j in est_sat.actuators]
    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt)
    est.lam = 0
    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)

    est_sat0 = copy.deepcopy(est_sat)

    bdotgain = 1e8
    kw = 1500
    ka = 10

    while t<tf:

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val

        #control law
        nB2 = norm(orbt.B)
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err =w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = np.zeros(3)
        elif t<tlim0:
            #bdot
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/dt
            offset_vec = np.zeros(3)# + np.concatenate([j.bias for j in est_sat.actuators])
        else:
            if t<tlim1:
                #PID to zeroquat
                qdes = zeroquat
            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

            offset_vec = est_sat.disturbances[2].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-12)#,jac = ivp_jac)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)

    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    base_title = "crassidis_baseline_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
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
    sim.comp_time = comp_time_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)

#my base case with great sensors


#paper comparisons.  -- sensor biases and non, actuator biases and non, disturbances and non. Specific prop disturbance case

if __name__ == '__main__':
    newcrassorbs = False
    newmyorb = False
    crass_tf = 60*60*32
    crass_tf_1 = 60*60*12
    my_tf = 60*60*4
    if newcrassorbs:
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        crass_orb = Orbit(os0,end_time = 0.22+1.05*crass_tf*sec2cent,dt = 10,use_J2 = True)
        with open("crassorb", "wb") as fp:   #Pickling
            pickle.dump(crass_orb, fp)
        crass_orb_1 = Orbit(os0,end_time = 0.22+1.05*crass_tf_1*sec2cent,dt = 1,use_J2 = True)
        with open("crassorb1", "wb") as fp:   #Pickling
            pickle.dump(crass_orb_1, fp)
    else:
        with open("crassorb", "rb") as fp:   #unPickling
            crass_orb = pickle.load(fp)
        with open("crassorb1", "rb") as fp:   #unPickling
            crass_orb_1 = pickle.load(fp)

    if newmyorb:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        myorb = Orbit(os0,end_time = 0.22+1.05*my_tf*sec2cent,dt = 1,use_J2 = True)
        with open("myorb", "wb") as fp:   #Pickling
            pickle.dump(myorb, fp)
    else:
        with open("myorb", "rb") as fp:   #unPickling
            myorb = pickle.load(fp)
    with open("paper_test_files/new_tests_marker"+time.strftime("%Y%m%d-%H%M%S")+".txt", 'w') as f:
        f.write("just a file to show when new runs of tests started.")
    np.set_printoptions(precision=3)
    # crassidis_UKF_attitude_errors_replication(crass_orb)
    # # crassidis_UKF_attNbias_errors_replication(crass_orb)
    # crassidis_UKF_baby_world(crass_orb,tf=crass_tf)
    # crassidis_UKF_baby_world_mine(crass_orb,tf=crass_tf)
    # crassidis_UKF_baby_world_mine(crass_orb_1,tf=crass_tf_1,dt = 1)
    # crassidis_UKF_disturbed_world(crass_orb,tf=crass_tf)
    # crassidis_UKF_disturbed_world_mine(crass_orb,tf=crass_tf)
    # crassidis_UKF_disturbed_world_mine(crass_orb_1,tf=crass_tf_1,dt = 1)
    # crassidis_UKF_ctrl_world(crass_orb,tf=crass_tf)
    # crassidis_UKF_ctrl_world_mine(crass_orb,tf=crass_tf)
    # crassidis_UKF_ctrl_world_mine(crass_orb_1,tf=crass_tf_1,dt = 1)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-5,werrcov = 1e-5)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcov = 1e-10)

    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcov = 1e-5)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-5,werrcov = 1e-10)



    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcov = 1e-12) # good! kinda bad in AV though.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-4,werrcov = 1e-8) #not as good as -6,-12, but ok.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-4,werrcov = 1e-4) #bad!
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-4,werrcov = 0) #angle is very bad, but not as bad as -4,-4. AV is alright, actually a bit better than -6,-12
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 1e-4) #bad!
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcov = 0) #quite good AV, not quite as good at ang as -6,-12
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcov = 0) #not as good as -6,-inf on either.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 1e-12) #pretty meh.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcov = 1e-12) #also meh. seems almost identical to previous.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcov = 1e-14) #ok but not great. less noisy on ang than -8,-inf but similar perforamce, worse on AV than -8,-inf.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcov = 1e-16) #not as good at AV as -6,-inf but good and better tahn -8,-inf. noisy ang like -8,-inf but better performance. maybe not quite as good as -6.-inf
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-4,werrcov = 1e-8) #bad AV. consistent approach on ang
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcov = 1e-16) #similar AV to -6,-inf, maybe even a bit better! slightly better ang than -6,-inf. maybe as good as -6,-12. probably new baseline.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcov = 1e-12) #subpar in both.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcov = 1e-18) #marginally worse on ang than -6,-16. similar performace on AV. myabe slightly worse.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcov = 1e-14) #save ang as -6,-16 but worse AV.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-5,werrcov = 1e-16) #worse on both.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-16) #slightly better on both!
    #longer times as to better evaluate whole. eventually will go up more.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-6,werrcov = 1e-16) #repeat of prev baseline with longer time. about as good as previous. some larger-sacle noise on ang that should be investigagted.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-7,werrcov = 1e-16) #repeat of current baseline with longer time.about as good as previous. some larger-sacle noise on ang that should be investigagted.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-8,werrcov = 1e-16) #more large scale noise than baseline, but less HF noise (in ang). AV isn't as good.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-7,werrcov = 1e-17) #ang similar to baseline, maybe slightly worse. AV slightly worse than baseline.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-7,werrcov = 1e-15) # maybe slightly better in ang and slightly worse in AV? hard to tell.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-7,werrcov = 1e-16) #longer run of baseline. about as expected.large scale noise is just middle frequency stuff.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-16,jmult = 0.001) #does MOI have an impact? YES! Very bad on both AV and ang.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-13,jmult = 0.001) #does MOI have an impact? marginally better on ang. same on AV
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-10,jmult = 0.001) #does MOI have an impact? pretty find on ang, kinda bad on AV.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-19,jmult = 0.001) #does MOI have an impact? very bad.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-22,jmult = 0.001) #does MOI have an impact? very bad.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcov = 1e-16,jmult = 0.001) #does MOI have an impact? not as bad as some of the others?
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcov = 1e-16,jmult = 0.001) #does MOI have an impact? bad.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcov = 1e-9,jmult = 0.001)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-7,jmult = 0.001) #maybe the best of these.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcov = 1e-6,jmult = 0.001)
    #testing scaling theory
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-4,jmult = 0.001) #not very good.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-8,jmult = 0.01) #pretty good
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-10,jmult = 0.01) #better than the other one for this scale, as expected.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-12,jmult = 0.1) #not bad
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-13,jmult = 0.1) #better
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-14,jmult = 0.1) #better on AV, worse on ang.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-12,jmult = 0.01) #meh
    #seems to scale with J^-3, or maybe J^-2 or J^-4
    #testing timing scaling theory
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-7,werrcov = 1e-16) #not amazing.
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-7,werrcov = 1e-15) #slightly better? marginal.
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-7,werrcov = 1e-14) #slightly worse.
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-8,werrcov = 1e-15) # a little better in ang..
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-6,werrcov = 1e-15) #best of these I think??
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-8,werrcov = 1e-16)
    #
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-7,werrcov = 1e-17)
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-7,werrcov = 1e-18)
    # back to modifying kap
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*1,mrperrcov = 1e-7,werrcov = 1e-16,kap = 3-18) #meh
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*1,mrperrcov = 1e-7,werrcov = 1e-16,kap = 3-21)#meh
    #
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*1,mrperrcov = 1e-7,werrcov = 1e-16,kap = -3) #not bad!
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*1,mrperrcov = 1e-7,werrcov = 1e-16,kap = -1) #meh
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*1,mrperrcov = 1e-7,werrcov = 1e-16,kap = -9) #quite good.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*1,mrperrcov = 1e-7,werrcov = 1e-16,kap = 0) #still best.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-7,werrcov = 1e-16,kap = -12) #bad!
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-7,werrcov = 1e-16,kap = -9) #not good. fine-ish on AV
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-7,werrcov = 1e-16,kap = -3)#not good. fine-ish on AV
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*3,mrperrcov = 1e-7,werrcov = 1e-16,kap = -6)




    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*90,mrperrcov = 1e-7,werrcov = 1e-18,kap = 3-18) #quite good!
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*90,mrperrcov = 1e-7,werrcov = 1e-18,kap = 0) #bad
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*90,mrperrcov = 1e-7,werrcov = 1e-16,kap = 0) #not great
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*90,mrperrcov = 1e-7,werrcov = 1e-14,kap = 0) #okay, better than others.
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*90,mrperrcov = 1e-7,werrcov = 1e-20,kap = 0) #bad
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*90,mrperrcov = 1e-6,werrcov = 1e-16,kap = 0) #not great
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*90,mrperrcov = 1e-8,werrcov = 1e-18,kap = 0) #bad
    #
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-7,werrcov = 1e-20) #not great--maybe bad
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*15,mrperrcov = 1e-10,werrcov = 1e-20) #not good.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-9,werrcov = 1e-14,jmult = 0.1) #not very good.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-9,werrcov = 1e-12,jmult = 0.01) #meh
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-15,jmult = 0.1) #meh
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcov = 1e-14,jmult = 0.01) #bad
    #appears that kappa is more key in dt=1! need to learn why.
    #updating file saving info
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*0.1,mrperrcov = 1e-7,werrcov = 1e-16)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*0.1,mrperrcov = 1e-7,werrcovmult = 1)
    #replacing int cov with a J term!
    #testing with kappa
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcov = 1e-16) #current baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1) #Jmod baseline. #using J in the int_cov! #just about equals the baseline--very slightly worse.
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcov = 1e-16)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-8,werrcovmult = 1) #worse in all respects to Jmod baseline. ~3-4x worse in ang metrics, 4-5x in AV metrics
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-9,werrcovmult = 1) #about 2x worse in ang metrics than -8,1 and ~3x worse in av metrics

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-6,werrcovmult = 1) #a bit better than Jmod baseline for AV. mixed results for ang (one metric a bit better, one a bit worse.)
    #was int_cov scaling implemented incorrectly?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e10) #--better than Jmod baseline on ang(~0.6-0.8x), worse on AV (~4x)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8) #slightly better than baseline on AV, mixed resolts on ang, but only small differences.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e9) #compared to jmod baseline, ang mean is 0.6x, ang max is slightly worese (~1.1x), av metrics are like 1.2-1.3x worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 10) #almost identical to jmod baseline. seems like the higher values are correct--looking at how jmod is implemented, it seems that the scaling is wrong. between 1e8 and 1e3?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 0.1)#almost identical to jmod baseline. seems like the higher values are correct--looking at how jmod is implemented, it seems that the scaling is wrong. between 1e8 and 1e3?
    # #was int_cov J power wrong? ^3 doesn't really make sense!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1,invjpow = 2) #almost identical to jmod baseline. seems like the higher values are correct--looking at how jmod is implemented, it seems that the scaling is wrong. between 1e8 and 1e3? but probably needs to vary with power.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e5,invjpow = 2) #better on AV than jmod baseline, mixed on ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,invjpow = 2) #better on ang than jmod baseline, slightly worse on one AV metric, better on another AV metric.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e7,invjpow = 2) #about twice as worse on AV than jmod baseline, clearly better on ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2) #mixed resutls on ang--pretty much identical to jmod baseline, a bit better on AV.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-6,werrcovmult = 1e5,invjpow = 2) #better on AV, worse ang max, better ang mean.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-8,werrcovmult = 1e5,invjpow = 2) #worse on both.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1,kap = 3-18) #bad!
    #compare to the werrcovmult 1e8 and 136 (but with invjpow = 2)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-6,werrcovmult = 1) #a bit better than Jmod baseline for AV. mixed results for ang (one metric a bit better, one a bit worse.)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e7) # a little better than Jmod baseline.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e4) # almost identical to Jmod basline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 3-18) #bad!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 1) #very bad!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 5) #better on ang than Jmod baseline, about 3x worse on AV.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 20) #not good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 100) #not good
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,invjpow = 2,kap = 3-18) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,invjpow = 2,kap = 1) #very bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,invjpow = 2,kap = 5) #better on ang than Jmod baseline, about 5x worse on AV.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,invjpow = 2,kap = 20) #not great.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,invjpow = 2,kap = 100) #bad

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 5) #repeat of previous
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 8) #bad!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 3) #not a good as kap=5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 12) #not good
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 16) #not great
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-8,werrcovmult = 1e8,kap = 5) #not great
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-6,werrcovmult = 1e8,kap = 5) #about as good as Jmod baseline, a bit worse on AV
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-9,werrcovmult = 1e8,kap = 5) #not bad but not excellent.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-5,werrcovmult = 1e8,kap = 5) #ok but not great
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e7,kap = 5) #beats jmod baseline on ang, but worse on AV
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e9,kap = 5) #beats jmod baseline on ang, but worse on AV
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,kap = 5) #beats jmod baseline on ang, but worse on AV--not bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e10,kap = 5) #meh
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e7,invjpow = 2,kap = 5) #meh
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e5,invjpow = 2,kap = 5) #beats both baselines on ang, but worse on AV.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-6,werrcovmult = 1e6,invjpow = 2,kap = 5) #not as good
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-8,werrcovmult = 1e6,invjpow = 2,kap = 5) #bad

#
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 1,kap = 1,scaled_UT = True,bet = 0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 1,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.99,kap = 1,scaled_UT = True,bet = 0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-8,werrcovmult = 1e6,al = 0.99,kap = 5,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-8,werrcovmult = 0,al = 1,kap = 10,invjpow = 2,scaled_UT = True)
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 1,kap = 3-18,scaled_UT = True)
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.99,kap = 3-18,scaled_UT = True)
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e8,invjpow = 2,al = 0.1,kap = 1,scaled_UT = True,bet = 2)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.01,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.001,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.1,kap = 3,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.01,kap = 3,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.001,kap = 3,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.1,kap = 10,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.01,kap = 10,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.001,kap = 10,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.1,kap = 3-18,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.01,kap = 3-18,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.001,kap = 3-18,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.1,kap = 3-21,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.01,kap = 3-21,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,al = 0.001,kap = 3-21,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 0,invjpow = 2,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e5,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 4)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e8,kap = 7)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,kap = 6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,kap = 4)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e6,kap = 7)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e5,invjpow = 2,kap = 4)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e5,invjpow = 2,kap = 6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1e5,invjpow = 2,kap = 3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,kap = 5)




    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1,jmult=0.01)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 1e-7,werrcovmult = 1,jmult=0.1)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb_1,dt = 1,tf=60*60,mrperrcov = 1e-7,werrcovmult = 1)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb_1,dt = 1,tf=60*60,mrperrcov = 1e-7,werrcovmult = 0.1)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb_1,dt = 1,tf=60*60,mrperrcov = 1e-7,werrcovmult = 0.01)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb_1,dt = 1,tf=60*60,mrperrcov = 1e-8,werrcovmult = 1)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb_1,dt = 1,tf=60*60,mrperrcov = 1e-9,werrcovmult = 1)

    #TODO: tuning. of pow 2v3, kap, werrcovmult, mrperrcov.
    #Next: try new eig decomp. ry different scalings of the 3 terms in int_cov associated with werr.   check J scaling on werr (next 2 tests). try to judge time scaling on mrperr,werr




    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = 3)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = 1)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = -1)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = -3)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = -7)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = -11)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = -15)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 0,kap = -18)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 1e-5)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 1e-10)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 1e-20)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcov = 1e-30)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-5,werrcov = 0)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcov = 0)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-20,werrcov = 0)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-30,werrcov = 0)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-30,werrcov = 1e-20)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2) #best so far

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e4,invjpow = 2) #about 10x worse than new baseline in AV, a little worse in ang (like old baseline or old Jmod baseline.)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcovmult = 1e4,invjpow = 2) #about 2-3x worse than new baseline in AV, about equal in ang mean, better in ang max.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e3,invjpow = 2)#about 2-3x worse than new baseline in AV, better in ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e5,invjpow = 2)#about 2x worse than new baseline in AV, better in ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,xtermmult=-1)#about 2-3x worse than new baseline in AV, better in ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,xtermmult=1) #about 2-3x worse than new baseline in AV, better in ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,xtermmult=-5)#about 2-3x worse than new baseline in AV, better in ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,xtermmult=5)#about 2-3x worse than new baseline in AV, better in ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,xtermmult=0)#about 2-3x worse than new baseline in AV, better in ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,ang_werr_mult=1/30.0)#about 2-3x worse than new baseline in AV, better in ang.

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2) #not as good as new baseline, but good trend and behavior looks good. reference for this batch. (with 0 mrperrcov)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2) #better than reference.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e3,invjpow = 2) #a little worse than reference
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=-1) #pretty much equal to reference.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=1) #pretty much equal to reference.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=0)#pretty much equal to reference!!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=-5)#pretty much equal to reference.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=5) #pretty much equal to reference, maybe a tiny bit worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,ang_werr_mult = 1/30.0) #pretty much equal to reference.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,ang_werr_mult = 10.0/3.0)#pretty much equal to reference.

    #new ref is crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2) #about equal to reference, a bit better in AV, a little worse ang mean, a little better ang max. overall behavior of ang graph looks worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=-0.05,ang_werr_mult = 1/30.0) #about equal to reference
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=0) #about equal to reference, maybe a bit worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 1/30.0)  #about equal to reference, maybe a bit worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=-0.05,ang_werr_mult = 1/300.0) #about equal to reference, maybe a bit worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=-0.005,ang_werr_mult = 1/300.0) #about equal to reference, maybe a bit better.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 1/300.0) #about equal to reference,
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #about equal to reference,
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #about equal to reference,
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=-0.005)#about equal to reference,
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,ang_werr_mult = 1/30.0) #about equal to reference,
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,ang_werr_mult = 1/300.0) #about equal to reference,
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,ang_werr_mult = 10.0/3.0)#about equal to reference,
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=-5) #about equal to reference, better on ang
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=5) #worse than reference by a bit
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=-50) #errors out

    # new best idea is crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #about equal to reference,
    #not different than reference and has a lot of 0 terms!

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #very good results. does impreove with time!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #repeat of best idea. still best I've seen.***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #kinda worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #equal to ref, as expected.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e7,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0) # a bit worse than ref
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e8,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0) #excellent. best yet (in this length). ***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e9,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0) #beats ref in both, not quite as good as previous
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0) #Kinda worse than ref
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #just like ref
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e4,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #kinda worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #as good as ref, maybe a bit better. ***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = 1) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = 1) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = 1) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = -1) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = -1) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = -1) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e2,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #ok, not as good as baseline. good trend
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #better than prev, not as good as baseline. weirder trend.***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #best of these J^0.***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #mixed results compared to previous. chaotic trend.***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) # pretty bad given its 8 hours long.


    #how does this one compare to the first 2 runs of the previous session, with werrcovmlut = 1e6 and 1e5. Is this worse or better or unchanged?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e7,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #wowrse than 1e6, but still pretty good. seems to level of in AV. 1e6 is still the best!
    #previous session got good results with short runs with J^-3. Does it hold up over a longer run?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e8,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0)  #wowrse than J^-2. but still pretty good. Try higher multiple?
    # #previous session got good results with short runs with J^-3. Does J^-4 work better? an odd power seems odd.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e10,invjpow = 4,xtermmult=0,ang_werr_mult = 0.0) #not bad. might be worth considering. a little worse in AV I think but maybe a bit better in ang. interesting pattern
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e11,invjpow = 4,xtermmult=0,ang_werr_mult = 0.0) #very similar to previous on ang (or converges to it at least), maybe a bit better in AV.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e9,invjpow = 4,xtermmult=0,ang_werr_mult = 0.0) #doesnt seem better than the others here. probably worse.
    #pretty good results in previous senssion with J^0! How do they hold up on longer timelines?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #about equal to main one on AV. a bit worse on ang. So seems to hold up ok??
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #better than best in AV. almost as good on angle--and better pattern. ***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #about as good as the previous.
    # #comparing best result from first run of previous session (which had al=0.99) to this with al = 1
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1) #doing badly!


    #al = 1 is bad. but does varying it do anything?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.9) #not as good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.99) #performed well like the baseline we would expect
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.999) #bad!
    #does 1e9 with J^-3 work better on the long run than 1e8?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e9,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0) #not super clear but seems to be better than before! Solid contender for best yet?? ***
    #do we need a higher multiple for J^-4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e12,invjpow = 4,xtermmult=0,ang_werr_mult = 0.0) #not as good as many others we've seen. I think it might be out. 2 or 3 or 0 is where it is at.
    #whats the effect of a little mrp error at long run times. do they allow less ang error?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-16,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #pretty much identical to mrp=0
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-12,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) # maybe a little bit better mut hard to tell
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-16,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #pretty close to best yet--maybe better in AV? *** --could be possible to bring up MRP and reduce werr
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-12,werrcovmult = 1e5,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0) #about as good as best yet. a little worse in AV. maybe a bit beter in ang.
    #J^0 has best yet for AV it seems. can it be best overall?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #unclear if better than werrcovmult=1e0. --doesn't seem to be, though converges a bit faster at first. not as good as best yet.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-16,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #slight imporvement over mrp=0. very slight.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-16,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #not as good.


    #small alpha???
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.1) #bad!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.01) #better but still not good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.001) #much better. Still not great. ***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1) #not as bad as it could be but bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.0001) #bad!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e3,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.001) #worse AV but better on ang than werrcovmult = 1e6
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.001) #pretty good--smooth.


    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult= 0,ang_werr_mult = 0.0,al = 0.01,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult= 0,ang_werr_mult = 0.0,al = 0.001,kap = 1,scaled_UT = True) #singular at 0
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult= 0,al = 0.1,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult= 0,al = 0.01,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.001,kap = 1,scaled_UT = True)#singular at 0
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult= 0,al = 0.1,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult= 0,al = 0.01,kap = 1,scaled_UT = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.001,kap = 1,scaled_UT = True) #singular at 0
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.1,kap = 1,scaled_UT = True) #nonPD at 3
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult= 0,ang_werr_mult = 0.0,al = 0.1,kap = 1,scaled_UT = True)#prob nonPD at 3
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.1,kap = 1,scaled_UT = True)#prob nonPD at 3
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.5,kap = 1,scaled_UT = True) #very meh
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.99,kap = 1,scaled_UT = True) #meh
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.99,kap = 1,scaled_UT = True) #not bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult= 0,ang_werr_mult = 0.0,al = 0.99,kap = 1,scaled_UT = True) #better but not amazing

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e1,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.001)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.001) #okay ish
    #al=0.99 shouldn't be the only thing that works. is it tied to integration covariance? or kappa?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e4,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.999) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e2,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.999) #bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1) #not good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 1)  #not good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = -1)  #not good
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = -3) #bleh.

    #square root!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.001,kap = 1,scaled_UT = False) #singular at 0
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e6,invjpow = 0,xtermmult= 0,ang_werr_mult = 0.0,al = 0.0001,kap = 0,scaled_UT = False) #nonPD at 3
    # #goddamnit I want 0 to work
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 10)
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = 1)
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = -1)
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = -10)
    # # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = -100)
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,kap = 10)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 100)
    # #further tests of J^-3
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e10,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0)#***
    # #testing with adding mrp
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-12,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0)#***
    # #longer run with small alpha
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.001)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.001)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.001)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 0.99)
    # #testing kap
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 10)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e6,invjpow = 2,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 10)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e9,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e9,invjpow = 3,xtermmult=0,ang_werr_mult = 0.0,al = 1,kap = 10)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #best of these 3
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True) #best of these 3, equal to no SR
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-10,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-10,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-10,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-10,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True)
    #next: testing with varying alpha, kappa, beta(?). checking J and dt scaling. effect of control cov going to 0 in estimated sat. one-by-one adjustment of how "good" each sensor is.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*1,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True)#test of software
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True) #worse than baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0)#equal to SR
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #worse than baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True) #equal to non SR
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-16,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0) #euqal to baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-16,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True)#euqal to baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,est_ctrl_cov = 0*np.zeros(3))
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True,est_ctrl_cov = 0*np.zeros(3))
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1.0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True,al = 1.0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.01)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True,al = 0.01)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,kap = -3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True,kap = -3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,kap = 3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True,kap = 3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,kap = 100)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True,kap = 100)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,kap = -15)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,useSR = True,kap = -15)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.999,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9,useSR = True)


    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.999,useSR = True) #5,0.002
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True) #0.16,3e-5 # *****
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9,useSR = True) #0.25,3e-4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True) #4,0.002
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.999,useSR = True) #10.5,0.009
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True) #0.22,7e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9,useSR = True) #12,0.005
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True) #19,0.01
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.999,useSR = True) #100,0.09
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True)  #0.19,3e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9,useSR = True) #0.65,6e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True) #0.62,5e-5, good pattern...

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True) #2.65,9e-4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e2,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True) #0.53, 9e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e2,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True) #0.45,1e-4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e4,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True) #0.31,6.5e-5 ***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e4,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True) #1.3,1.2e-4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -1) #1.75,6e-4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -3) #0.9,2.5e-4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -6) #0.2,7e-5 ***
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -9) #0.32,4.5e-5 **
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -15) skipped
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -18)skipped
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -21)skipped
    #next: testing with varying beta(?). checking J and dt scaling.  one-by-one adjustment of how "good" each sensor is.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1.0,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*4,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.001,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True)#0.1,2e-5 # *****
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True)#1.7,1.5e-3
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True) #0.26,5e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True) #23,0.005
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True) #140,0.27
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True) #18,0.0017
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.01,useSR = True) #11,0.005
    #bug fixed. Need to re run to understand...
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0) #1.7,0.0015
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -6) #10,0.01
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -3) #150,0.3
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -9) #2.1,0.0021
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -12) #130,0.23
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -15)#160,0.2
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -18) #140,0.6
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -21) #went singular?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #baseline: 25,0.02
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-18) #different kappa?: 30,0.012 -- mixed
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-17,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #reduced ang ic by 1 OoM: 25,0.019--not diff from baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-18,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #reduced ang ic by 2 OoM: 25,0.019--not diff from baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #reduced av ic by 1 OoM: 46,0.022 -- worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-17,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #reduced both ang and av ic by 1 OoM:  46,0.022 -- worse.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-18,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #skipped based on previous results

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-15,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #increased ang ic by 1 OoM: 25,0.019--not diff from baseline
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #increased av ic by 1 OoM: 23.5,0.015--better than baseline. Best so far?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-15,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #increased both ang and av ic by 1 OoM: 23.5,0.015--better than baseline. ang ic does very little at this scale.

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e2,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21)#increased av ic by 2 OoM: 43,0.013
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-18) #increased av ic by 1 OoM,changed kappa: 37,0.012
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #big increase in ang ic, 1 OoM increase in av ic: 21,0.015--best so far?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-14,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) # 23.5,0.015 -- equal to -16,+1
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #2.8,0.016!!--best so far!!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e2,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #41.5,0.012 -- smilar to 1e-15,1e2
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21) #18.5,0.017 --similar to 1e-12,1e1

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,mtm_scale = 1.0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #1.6,0.009!! best so far.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,mtm_scale = 1.0,av_init_cov_est =(math.pi/180.0)**2.0) # seems bad when running. freezes a lot.

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #5,0.006--best on AV, good on ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e2,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #2,0.010--good on AV, good on ang. but the previous best beats it on both (though only a small amount)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.9,0.010--best on ang, good on AV. best overall so far, I think.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #19.5,0.008--great on AV, bad on ang.

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.9,0.023 -- equal to best on ang, not great on AV.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) ##2.6,0.012--not better.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-18,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.8,0.009--best so far!!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.4,0.006. Best yet!! kap = 0 seems good. need to test with other values of J, t, my cubesat, etc. ****. run other estimatino test and see if still performs well-ish
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-1,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #not as good;4.3,0.06 (NOT 0.006)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-1,useSR = True,kap = 3-18,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #4,0.04. not as good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-1,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.17,0.014--mixed results! Could be quite good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #95,0.65 . BAD
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-18,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #6.7,0.11 bad
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #1.1,0.038 pretty good in ang, ok in av.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0,mtm_scale = 1.0) #0.9,0.010 quite good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0,mtm_scale = 1e5)  #0.9,0.010 quite good.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.34,0.007--tied for best yet!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 3-18,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #6.7,0.1. not very good. indistinguisable from the al = 1 case.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.27,0.01, very good pattern.
    #mtm scale doesn't make a difference except in speed.
    #comp to 0.4/0.006, 0.17/0.014, 0.34/0.007,0.27/0.010
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #1.95,0.009 --worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-1,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.86,0.010 -- worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.28,0.006 -- improved! acceptable pattern, too.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #7.3,0.04, bad pattern -- worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-3,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #2.56, 0.012 -- worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-1,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #8.27,0.045 -- worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #2.3,0.006 -- worse
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-10,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #2.5,0.007 -- worse
    #first not good in small test; then good but not amazing in small; quite good in small; quite good in small

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1.0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0)
    #figure out why 0.99 works and 1 doesn't. otherwise, try the above combos.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e-1,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) 4.3,0.06 (NOT 0.006)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) 7.8, 0.12
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.75,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.32,0.025
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #3.4,0.04
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.34,0.007
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.999,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #30,0.34
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9999,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #14,0.37
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99999,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #149,0.4
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #95,0.65 . BAD
    #al=1 is a problem becuase of SR?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0)
    #no, problem is av doesn't converge in either SRUKF or UKF--but it covnerges when kap = 3-21,esp. if int cov is low it seems.

    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #7.8,0.003
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #1.1,0.038
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #12.1,0.09
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #9.3,0.003
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #3.5,0.002
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #6.5,0.0025
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #good pattern.3.5,0.0017 best yet on AV, good on ang.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #21.2,0.1
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #13,0.08
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-12,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #2.7,0.002
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-16,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #3.6,0.0017. not different with mrperr = 0.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 0,werrcovmult = 1e-3,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #3.3,0.002.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e-3,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #73,0.09
    #Longer tests
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.15,0.00013
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.25,0.0002
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e-3,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0)#0.25,0.0002
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.15,0.00013
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.45,0.00014 -- great pattern!
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e3,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #3.2,0.0003. very bad for  this length
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-8,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #2.4,0.017, bad for length
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-12,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.34, 0.0002
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-8,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #4.0,0.003
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.31,5e-5, very good pattern, conitnuation of above.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 1e-12,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.06,0.00010 #best yet in ang, near best in av.
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 1e-16,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.25,0.00013
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 1e-12,werrcovmult = 1e-3,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.13,0.00018
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #.17, 7e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 1e-12,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-18,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #16,0.008
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 1e-12,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #56,0.017
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 1e-12,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-18,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #16,0.008
    #time scaling?
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 1,tf=60*60*6,mrperrcov = 1e-12,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.46,0.00011
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 1,tf=60*60*6,mrperrcov = 1e-14,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.49,6.3e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 1,tf=60*60*6,mrperrcov = 1e-10,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.24,0.00044
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 1,tf=60*60*6,mrperrcov = 1e-12,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #0.08,1.3e-5
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 1,tf=60*60*6,mrperrcov = 1e-12,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1e0,useSR = True,kap = 3-21,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0) #2.6,0.0003

    #with dt = 10,al =1, kap = 3-N (including control cov!),mrperrcov =  1e-12,werrcovmult = 1e-1; for dt = 1 seems like exact same!




    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1.0,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1.0,useSR = True,kap = 0,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0)


    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 10)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-12,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-12,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-16,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 1e-8,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1.0,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 1.0,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.01,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*8,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.01,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e6,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True) #0.75,3e-4
    # # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e6,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True) #skipped
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.999,useSR = True,kap = -6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9,useSR = True,kap = -6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True,kap = -6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True,kap = -6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -6)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -6)
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -9)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.999,useSR = True,kap = -9)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.9,useSR = True,kap = -9)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True,kap = -9)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True,kap = -9)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -9)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e-1,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -9)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -9)
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -7)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -8)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = -10)
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 1)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 5)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 10)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e0,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.99,useSR = True,kap = 100)
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e8,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.5,useSR = True)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*12,mrperrcov = 0,werrcovmult = 1e8,invjpow = 0,xtermmult=0,ang_werr_mult = 0.0,al = 0.1,useSR = True)



    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,ang_werr_mult=1/30.0)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e4,invjpow = 2,ang_werr_mult=0)
    #
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e7,invjpow = 3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e8,invjpow = 3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-7,werrcovmult = 1e6,invjpow = 3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-6,werrcovmult = 1e7,invjpow = 3)
    # crassidis_UKF_mine_no_dipole_Jmod(crass_orb,dt = 10,tf=60*60*2,mrperrcov = 1e-8,werrcovmult = 1e7,invjpow = 3)


    #
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=60*60*5)
    # crassidis_UKF_mine_no_dipole(crass_orb_1,dt = 1,tf=60*60*10)
    # crassidis_UKF_real_world_no_dipole(crass_orb,tf=60*60*24)
    #
    # # crassidis_UKF_mine_no_abias(crass_orb,dt = 10,tf=60*60*24)
    # # crassidis_UKF_mine_no_abias(crass_orb_1,dt = 1,tf=60*60*8)
    # # crassidis_UKF_real_world_no_abias(crass_orb,tf=60*60*24)
    #
    # # crassidis_UKF_mine(crass_orb,dt = 10,tf=60*60*8)
    # # crassidis_UKF_mine(crass_orb_1,dt = 1,tf=60*60*8)
    # # crassidis_UKF_real_world(crass_orb,tf=60*60*8)
    #
    # bc_UKF_baseline(orb = myorb, tf = 2*60*60,dt = 1)
    # bc_UKF_baseline(orb = myorb, tf = 2*60*60,dt = 10)
    #
    # bc_crassidis_UKF(orb = myorb, tf = 2*60*60,dt = 1)
    # bc_crassidis_UKF(orb = myorb, tf = 2*60*60,dt = 10)

    ##case with bias on mtq and prop torque.
