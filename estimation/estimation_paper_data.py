#estimation results for paper
from sat_ADCS_estimation import *
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
def crassidis_UKF_real_world(orb=None,t0 = 0, tf = 60*60*8,dt=10,close = False):
    np.random.seed(1)
    tlim00 = 60*30
    tlim0 = 2*60*60
    tlim1 = np.round(tf*0.25)
    tlim2 = np.round(tf*0.5)
    tlim3 = np.round(tf*0.75)

    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mtq = True,include_mtqbias = True,estimate_mtq_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mtq = True,include_mtqbias = True,estimate_mtq_bias = False,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole=False)

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

    angcov = (50*math.pi/180.0)**2.0
    if close:
        angcov = (0.5*math.pi/180)**2.0
        estimate[8] = 0
        estimate[3:7] = q0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*angcov,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
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
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))
    bdotgain = 1e10
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

        nB2 = norm(orbt.B)**2.0
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = 0*est_state[7:10]
        elif t<tlim0:
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/dt
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = 0*est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
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
            # ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            ud = np.cross(Bbody,-kw*est_sat.J@w_err-ka*q_err[1:]*np.sign(q_err[0]) - est_sat.dist_torque(est_state,est.os_vecs))/nB2
            offset_vec = 0*est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
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
        dist_torq_hist[ind,:] = real_sat.last_dist_torq.copy()
        est_dist_torq_hist[ind,:] = est_sat.last_dist_torq.copy()
        act_torq_hist[ind,:] = real_sat.last_act_torq.copy()
        est_act_torq_hist[ind,:] = est_sat.last_act_torq.copy()
        sens_hist[ind,:] = sens.copy()
        eclipse_hist[ind] = orbt.in_eclipse()

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
    if close:
        base_title = "close_"+base_title
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
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_err_plot")


    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
    sim.orb_hist = orb_hist
    sim.cov_hist = cov_hist
    sim.cov_est0 = cov_estimate.copy()
    sim.int_cov = int_cov.copy()
    sim.t_hist = t_hist
    # sim.real_sat = real_sat
    # sim.est_sat = est_sat
    # sim.est_sat0 = est_sat0
    # sim.est = est
    sim.metrics = metrics
    sim.comp_time = comp_time_hist
    sim.dist_torq_hist = dist_torq_hist
    sim.est_dist_torq_hist = est_dist_torq_hist
    sim.act_torq_hist = act_torq_hist
    sim.est_act_torq_hist = est_act_torq_hist
    sim.sens_hist = sens_hist
    sim.eclipse_hist = eclipse_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#Crassidis "real_world"
def crassidis_UKF_real_world_no_dipole(orb=None,t0 = 0, tf = 60*60*8,dt=10,close = False):
    np.random.seed(1)
    tlim00 = 60*30
    tlim0 = 2*60*60
    tlim1 = np.round(tf*0.25)
    tlim2 = np.round(tf*0.5)
    tlim3 = np.round(tf*0.75)

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

    angcov = (50*math.pi/180.0)**2.0
    if close:
        angcov = (0.5*math.pi/180)**2.0
        estimate[8] = 0
        estimate[3:7] = q0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*angcov,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
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
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))
    bdotgain = 1e10
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

        nB2 = norm(orbt.B)**2.0
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = 0*est_state[7:10]
        elif t<tlim0:
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/dt
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = 0*est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
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
            offset_vec = 0*est_state[7:10]#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
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
        dist_torq_hist[ind,:] = real_sat.last_dist_torq.copy()
        est_dist_torq_hist[ind,:] = est_sat.last_dist_torq.copy()
        act_torq_hist[ind,:] = real_sat.last_act_torq.copy()
        est_act_torq_hist[ind,:] = est_sat.last_act_torq.copy()
        sens_hist[ind,:] = sens.copy()
        eclipse_hist[ind] = orbt.in_eclipse()

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
    if close:
        base_title = "close_"+base_title
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
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_err_plot")


    #generate statistics
    metrics = find_metrics(t_hist,angdiff)

    #save data and plots
    sim = simsave()
    sim.state_hist = state_hist
    sim.est_state_hist = est_state_hist
    sim.control_hist = control_hist
    sim.orb_hist = orb_hist
    sim.cov_hist = cov_hist
    sim.cov_est0 = cov_estimate.copy()
    sim.int_cov = int_cov.copy()
    sim.t_hist = t_hist
    # sim.real_sat = real_sat
    # sim.est_sat = est_sat
    # sim.est_sat0 = est_sat0
    # sim.est = est
    sim.metrics = metrics
    sim.comp_time = comp_time_hist
    sim.dist_torq_hist = dist_torq_hist
    sim.est_dist_torq_hist = est_dist_torq_hist
    sim.act_torq_hist = act_torq_hist
    sim.est_act_torq_hist = est_act_torq_hist
    sim.sens_hist = sens_hist
    sim.eclipse_hist = eclipse_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)
    # except:
    #     print("DIDNT SAVE")
    #     breakpoint()

#my controller on the Crassidis test
def crassidis_UKF_mine_no_dipole(orb=None,t0 = 0, tf = 60*60*8,dt=10,extra_tag = None,mrperrcov = 0,werrcovmult = 0,kap = 0,mtm_scale=1e0,scaled_UT = False,al = 1,bet = 2,ang_werr_mult = 1,useSR = False,est_ctrl_cov = None,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0,close = False):
    np.random.seed(1)
    tlim00 = 10000#60*50
    tlim0 = 20000#np.round(tf/10)
    tlim1 = 30000#np.round(tf*0.2)
    tlim2 = np.round(tf*0.5)
    tlim3 = np.round(tf*0.75)

    #real_sat
    # mtm_scale = 1e3
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole = False,mtm_scale =  mtm_scale,jmult = 1,mtq_std = est_ctrl_cov)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = False,include_mtq = True,estimate_mtq_bias = True,estimate_mtm_bias=True,include_mtmbias=True,estimate_sun_bias=True,estimate_dipole = False,mtm_scale = mtm_scale,jmult = 1,mtq_std = est_ctrl_cov)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10+6+3)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    if close:
        estimate[0:3] = w0
        estimate[3:7] = q0
    else:
        estimate[14] = 20.1*(math.pi/180.0)/3600.0


    angcov = (50*math.pi/180.0)**2.0
    avcov = av_init_cov_est
    if close:
        avcov = ((0.5*math.pi/180)/(60))**2.0
        angcov = (0.5*math.pi/180)**2.0
    cov_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*5**2.0,np.eye(3)*(1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.1*0.3)**2.0)
    # alt_cov_estimate = block_diag(np.eye(3)*(math.pi/180.0/10)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,0.25*np.eye(3)*5**2.0,np.eye(3)*(0.5*1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.5*0.1*0.3)**2.0)

    int_cov =  dt*block_diag(np.block([[np.eye(3)*1e-16*werrcovmult,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
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
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))
    bdotgain = 1e10
    kw = 1000
    ka = 10
    real_vecs = os_local_vecs(orbt,state[3:7])
    sens = real_sat.sensor_values(state,real_vecs)

    # est.initialize_estimate(sens,[np.array([0,1,2]),np.array([6,7,8])],[orbt.B*est.sat.attitude_sensors[0].scale,est_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)],np.array([3,4,5]),orbt)


    mag_max = [j.max for j in est_sat.actuators]
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
        # pred_dyn_state = extra.mean1
        # psens = extra[1]
        # sens1 = extra[2]
        tt1 = time.process_time()
        est_state = est.use_state.val
        # est_vecs = os_local_vecs(orbt,est_state[3:7])
        # est_sens = est.sat.sensor_values(est_state,est_vecs)


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
        # est_vecs = os_local_vecs(orbt,est_state[3:7])
        unbiased_sens = sens-est_state[10:]
        print('real state ',state)
        print('est  state ',est_state[0:7])
        # print('pred state ',pred_dyn_state[0:7])
        # print('real dip ', real_sat.disturbances[2].main_param)
        # print('est dip ', est_state[-3:])
        # print('real abias ',real_abias)
        # print('est  abias ',est_state[7:10])
        # print('pred abias ',pred_dyn_state[7:10])
        # print('real sbias ',real_sbias)
        # print('est  sbias ',est_state[10:])
        # print('pred sbias ',pred_dyn_state[10:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print('est state')
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi)-np.pi))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi) - np.pi))
        # print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi,4*norm(np.arctan(autovar[3:6]/2.0))*180.0/math.pi)
        # sbvec_est = unbiased_sens[6:9]
        # bbvec_est = unbiased_sens[0:3]
        # srvec_real = real_sat.attitude_sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)
        # brvec_real = est.sat.attitude_sensors[0].scale*orbt.B
        # print('ang to S/B est ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(two_vec_to_quat(srvec_real,brvec_real,sbvec_est,bbvec_est),state[3:7]),-1,1)**2.0 ))
        print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)
        # print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)

        print('ab  ',norm(est_state[7:10]-real_abias))
        # print((est_state[7:10]-real_abias))
        print(autovar[6:9])

        print('gb ',norm(est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # print((est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        print(autovar[12:15]*180.0/math.pi)

        print('mb (x1e8/scale) ',norm(est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        # print((est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        print(autovar[9:12]*1e8/est.sat.attitude_sensors[0].scale)

        print('sb (x1e4) ',norm(est_state[16:19]-real_sbias[6:9])*1e4)
        # print((est_state[16:19]-real_sbias[6:9])*1e4)
        print(autovar[15:18]*1e4)

        # errvec = est_state-real_full_state
        # errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),5),errvec[7:]])
        # print('err vec ',errvec )
        # mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
        # print('mahalanobis_dist ',np.sqrt(mahalanobis_dist2))
        # print('prob ',chi2.pdf(mahalanobis_dist2,18))
        # cd = chi2.cdf(mahalanobis_dist2,18)
        # print('cdf ',cd)
        # print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))
        # print(est.wts_m)
        # print(est.wts_c)
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

        nB2 = norm(orbt.B)**2.0
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])
        elif t<tlim0:
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/dt
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
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
            # ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
            ud = np.cross(Bbody,-kw*est_sat.J@w_err-ka*q_err[1:]*np.sign(q_err[0]) - est_sat.dist_torque(est_state,est.os_vecs))/nB2
            # ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
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
        dist_torq_hist[ind,:] = real_sat.last_dist_torq.copy()
        est_dist_torq_hist[ind,:] = est_sat.last_dist_torq.copy()
        act_torq_hist[ind,:] = real_sat.last_act_torq.copy()
        est_act_torq_hist[ind,:] = est_sat.last_act_torq.copy()
        sens_hist[ind,:] = sens.copy()
        eclipse_hist[ind] = orbt.in_eclipse()

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
    if close:
        base_title = "close_"+base_title
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
    # plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_err_plot")

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
    sim.dist_torq_hist = dist_torq_hist
    sim.est_dist_torq_hist = est_dist_torq_hist
    sim.act_torq_hist = act_torq_hist
    sim.est_act_torq_hist = est_act_torq_hist
    sim.sens_hist = sens_hist
    sim.eclipse_hist = eclipse_hist
    # try:

    labels = ["title","al","kap","bet","dt","werrcovmult","mrpcov","scaled_UT","mtmscale","SR","ctrl cov est"]+["covest0","intcov"]+["conv time","tc","last 100 ang err mean","last 100 ang err max","last 100 av err mean","last 100 av err max"]
    info = [base_title,est.al,est.kap,est.bet,dt,werrcovmult,mrperrcov,scaled_UT,mtm_scale,useSR,est_ctrl_cov]+[np.diag(cov_estimate.copy()),int_cov.copy()]+[metrics.time_to_conv,metrics.tc_est,np.mean(angdiff[-100:]),np.amax(angdiff[-100:]),np.mean(matrix_row_norm(avdiff)[-100:]),np.amax(matrix_row_norm(avdiff)[-100:])]
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

#my controller on the Crassidis test
def crassidis_UKF_mine(orb=None,t0 = 0, tf = 60*60*8,dt=10,extra_tag = None,mrperrcov = 0,werrcovmult = 0,kap = 0,mtm_scale=1e0,scaled_UT = False,al = 1,bet = 2,ang_werr_mult = 1,useSR = False,est_ctrl_cov = None,av_init_cov_est = (20*(math.pi/180.0)/3600.0)**2.0,close = False):
    np.random.seed(1)
    tlim00 = 60*30
    tlim0 = 2*60*60
    tlim1 = np.round(tf*0.25)
    tlim2 = np.round(tf*0.5)
    tlim3 = np.round(tf*0.75)
    #real_sat
    # mtm_scale = 1e3
    real_sat = create_Crassidis_UKF_sat(real=True,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mtq = True,estimate_mtm_bias=False,include_mtmbias=True,estimate_sun_bias=False,estimate_dipole = False,mtm_scale =  mtm_scale,jmult = 1,mtq_std = est_ctrl_cov)
    est_sat = create_Crassidis_UKF_sat(real=False,include_sun = True,use_gg = True,use_drag = True, use_SRP = True,use_dipole = True,include_mtq = True,estimate_mtq_bias = True,estimate_mtm_bias=True,include_mtmbias=True,estimate_sun_bias=True,estimate_dipole = True,mtm_scale = mtm_scale,jmult = 1,mtq_std = est_ctrl_cov)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    if orb is None:
        # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
        os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7200,7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    q0 = two_vec_to_quat(-orb.states[orb.times[0]].R,orb.states[orb.times[0]].V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V

    estimate = np.zeros(10+6+3+3)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))

    if close:
        estimate[0:3] = w0
        estimate[3:7] = q0
    else:
        estimate[14] = 20.1*(math.pi/180.0)/3600.0

    angcov = (50*math.pi/180.0)**2.0
    avcov = av_init_cov_est
    if close:
        avcov = ((0.5*math.pi/180)/(60))**2.0
        angcov = (0.5*math.pi/180)**2.0
    cov_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*5**2.0,np.eye(3)*(1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.1*0.3)**2.0,np.eye(3)*(1)**2.0)
    # alt_cov_estimate = block_diag(np.eye(3)*(math.pi/180.0/10)**2.0,np.eye(3)*(50*math.pi/180.0)**2.0,0.25*np.eye(3)*5**2.0,np.eye(3)*(0.5*1e-9*mtm_scale)**2.0,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0,np.eye(3)*(0.5*0.1*0.3)**2.0)


    int_cov =  dt*block_diag(np.block([[np.eye(3)*1e-16*werrcovmult,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.diagflat([np.diag(j.std**2.0) for j in est_sat.disturbances if j.time_varying and j.estimated_param]))
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
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))
    bdotgain = 1e10
    kw = 1000
    ka = 10
    real_vecs = os_local_vecs(orbt,state[3:7])
    sens = real_sat.sensor_values(state,real_vecs)

    # est.initialize_estimate(sens,[np.array([0,1,2]),np.array([6,7,8])],[orbt.B*est.sat.attitude_sensors[0].scale,est_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)],np.array([3,4,5]),orbt)


    mag_max = [j.max for j in est_sat.actuators]
    while t<tf:
        print("======================================")
        print(t,ind)

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        real_sbias = np.concatenate([j.bias for j in real_sat.attitude_sensors if j.has_bias])
        real_abias = np.concatenate([j.bias for j in real_sat.actuators if j.has_bias])
        real_dist = np.concatenate([j.main_param for j in real_sat.disturbances if j.time_varying])
        # print(sens)

        #run estimator
        tt0 = time.process_time()
        aa,extra = est.update(control,sens,orbt)
        # pred_dyn_state = extra.mean1
        # psens = extra[1]
        # sens1 = extra[2]
        tt1 = time.process_time()
        est_state = est.use_state.val
        # est_vecs = os_local_vecs(orbt,est_state[3:7])
        # est_sens = est.sat.sensor_values(est_state,est_vecs)


        real_full_state = np.concatenate([state.copy(),real_abias.copy(),real_sbias.copy(),real_dist.copy()])
        # print(est_sens)

        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        # est_vecs = os_local_vecs(orbt,est_state[3:7])
        unbiased_sens = sens-est_state[10:-3]
        print('real state ',state)
        print('est  state ',est_state[0:7])
        # print('pred state ',pred_dyn_state[0:7])
        # print('real dip ', real_sat.disturbances[2].main_param)
        # print('est dip ', est_state[-3:])
        # print('real abias ',real_abias)
        # print('est  abias ',est_state[7:10])
        # print('pred abias ',pred_dyn_state[7:10])
        # print('real sbias ',real_sbias)
        # print('est  sbias ',est_state[10:])
        # # print('pred sbias ',pred_dyn_state[10:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print('est state')
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi)-np.pi))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi) - np.pi))
        # print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi,4*norm(np.arctan(autovar[3:6]/2.0))*180.0/math.pi)
        # sbvec_est = unbiased_sens[6:9]
        # bbvec_est = unbiased_sens[0:3]
        # srvec_real = real_sat.attitude_sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)
        # brvec_real = est.sat.attitude_sensors[0].scale*orbt.B
        # print('ang to S/B est ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(two_vec_to_quat(srvec_real,brvec_real,sbvec_est,bbvec_est),state[3:7]),-1,1)**2.0 ))
        print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)
        # print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)

        print('ab  ',norm(est_state[7:10]-real_abias))
        # print((est_state[7:10]-real_abias))
        print(autovar[6:9])

        print('gb ',norm(est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # print((est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        print(autovar[12:15]*180.0/math.pi)

        print('mb (x1e8/scale) ',norm(est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        # print((est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        print(autovar[9:12]*1e8/est.sat.attitude_sensors[0].scale)

        print('sb (x1e4) ',norm(est_state[16:19]-real_sbias[6:9])*1e4)
        print((est_state[16:19]-real_sbias[6:9])*1e4)
        print(autovar[15:18]*1e4)

        # errvec = est_state-real_full_state
        # errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),5),errvec[7:]])
        # print('err vec ',errvec )
        # mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
        # print('mahalanobis_dist ',np.sqrt(mahalanobis_dist2))
        # print('prob ',chi2.pdf(mahalanobis_dist2,18))
        # cd = chi2.cdf(mahalanobis_dist2,18)
        # print('cdf ',cd)
        # print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))
        # print(est.wts_m)
        # print(est.wts_c)
        #control law

        nB2 = norm(orbt.B)**2.0
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err = w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])
        elif t<tlim0:
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/dt
            # offset_vec = est_sat.disturbances[2].main_param#offset_vec = np.zeros(3)
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
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
            # ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
            ud = np.cross(Bbody,-kw*est_sat.J@w_err-ka*q_err[1:]*np.sign(q_err[0]) - est_sat.dist_torque(est_state,est.os_vecs))/nB2
            # ud = -kw*w_err-ka*q_err[1:]*np.sign(q_err[0])
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])#-np.cross(Bbody,est_sat.disturbances[2].main_param)# + np.concatenate([j.bias for j in est_sat.actuators])
            # print(norm(-kw*w_err),norm(-ka*q_err[1:]*np.sign(q_err[0])),norm(ud))
            # print(norm(offset_vec))
        control = limit(ud-offset_vec,mag_max)
        print('ctrl ', control)
        print(orbt.in_eclipse())
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

    base_title = "mine_real_world_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    if close:
        base_title = "close_"+base_title
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

    plot_the_thing(est_state_hist[:,19:22],title = "Est Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Est Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dipole_plot")
    plot_the_thing(state_hist[:,19:22],title = "Dist Dipole",xlabel='Time (s)',norm = True,ylabel = 'Dist Dipole (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_plot")
    plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Dist Dipole Error",xlabel='Time (s)',ylabel = 'Dist Dipole Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/dipole_err_plot")

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
    sim.metrics = metrics
    sim.comp_time = comp_time_hist
    sim.dist_torq_hist = dist_torq_hist
    sim.est_dist_torq_hist = est_dist_torq_hist
    sim.act_torq_hist = act_torq_hist
    sim.est_act_torq_hist = est_act_torq_hist
    sim.sens_hist = sens_hist
    sim.eclipse_hist = eclipse_hist
    # try:

    labels = ["title","al","kap","bet","dt","werrcovmult","mrpcov","scaled_UT","mtmscale","SR","ctrl cov est"]+["covest0","intcov"]+["conv time","tc","last 100 ang err mean","last 100 ang err max","last 100 av err mean","last 100 av err max"]
    info = [base_title,est.al,est.kap,est.bet,dt,werrcovmult,mrperrcov,scaled_UT,mtm_scale,useSR,est_ctrl_cov]+[np.diag(cov_estimate.copy()),int_cov.copy()]+[metrics.time_to_conv,metrics.tc_est,np.mean(angdiff[-100:]),np.amax(angdiff[-100:]),np.mean(matrix_row_norm(avdiff)[-100:]),np.amax(matrix_row_norm(avdiff)[-100:])]
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

#my base case
def bc_UKF_baseline(orb = None, t0 = 0, tf = 60*90,dt = 1,alt_title = None,rand = False,mtmscale = 1e4,abias_missing=False,anoise_missing = False,dist_missing = False,close = False,werrcov = 1e-17,mrperrcov = 1e-12,care_about_eclipse = False):
    if not rand:
        np.random.seed(1)
    tlim00 = np.round(min([5*60,0.05*tf]))
    tlim0 = np.round(min([15*60,0.1*tf]))
    tlim1 = np.round(tf*0.25)
    tlim2 = np.round(tf*0.5)
    tlim3 = np.round(tf*0.75)

    #
    #real_sat
    real_sat = create_BC_sat(real=True,use_SRP = True,use_dipole = False,include_mtqbias = True,rand=rand,mtm_scale = mtmscale,care_about_eclipse = care_about_eclipse)
    w0 = random_n_unit_vec(3)*1.0*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    if orb is None:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,use_SRP = not dist_missing,use_dipole = False,use_gg = not dist_missing,use_drag =  not dist_missing,include_mtqbias = not abias_missing,estimate_mtq_bias = not abias_missing,mtm_scale = mtmscale,include_mtq_noise = not anoise_missing,care_about_eclipse = care_about_eclipse)
    estimate = np.zeros(19-3*abias_missing)
    estimate[3] = 1

    angcov = 100#10
    avcov = (math.pi/180.0)**2.0

    if close:
        estimate[0:3] = w0
        estimate[3:7] = q0
        avcov = ((0.5*math.pi/180)/(60))**2.0
        angcov = (0.5*math.pi/180)**2.0
    if abias_missing:
        cov_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*(1e-7*mtmscale)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0)
    else:
        cov_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(1e-7*mtmscale)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0)


    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    # werrcov = 1e-17#1e-20#0#1e-20#0#1e-20
    # mrperrcov = 1e-12#1e-10#16
    # if anoise_missing:
    #     werrcov = 1e-13#1e-20#0#1e-20#0#1e-20
    #     mrperrcov = 1e-8#1e-10#16
    if abias_missing:
        int_cov =  dt*block_diag(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    else:
        int_cov =  dt*block_diag(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]))


    mtq_max = [j.max for j in est_sat.actuators]
    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False,sunsensors_during_eclipse = True)

    est.use_cross_term = True
    est.al = 1.0#1e-3#e-3#1#1e-1#al# 1e-1
    est.kap =  3 - 21 + 3*(abias_missing+anoise_missing)#15#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),19))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))

    est_sat0 = copy.deepcopy(est_sat)

    bdotgain = 1e6
    kw = 1000#5
    ka = 50#0.001
    ee = 0.005
    bb = 0.15

    while t<tf:

        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)

        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val
        print(t)
        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print(autovar[3:6])
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi,4*norm(np.arctan(autovar[3:6]/2.0))*180.0/math.pi)
        wa,va = np.linalg.eigh(est.use_state.cov[3:6,3:6])
        # print(va)
        # print(wa)
        # breakpoint()
        print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)

        #control law
        nB2 = norm(orbt.B)**2.0
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        w_err =w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])#np.zeros(3)
        elif t<tlim0:
            #bdot
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/(dt*mtmscale)
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
            # ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
            # ud = np.cross(Bbody,-est_sat.invJ@(ka*q_err[1:]*np.sign(q_err[0])*ee*ee + ee*saturate(kw*w_err/bb)*bb) - est_sat.dist_torque(est_state,est.os_vecs))/nB2#np.cross(Bbody,-kw*est_sat.J@w_err-ka*q_err[1:]*np.sign(q_err[0]) - est_sat.dist_torque(est_state,est.os_vecs))/nB2
            ud = np.cross(Bbody,-ka*q_err[1:]*np.sign(q_err[0])*ee*ee - ee*saturate(kw*est_sat.J@w_err/bb)*bb - est_sat.dist_torque(est_state,est.os_vecs))/nB2
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])
            # print(q_err[1:]*np.sign(q_err[0]))
            # print(q_err)
            # print(np.cross(Bbody,-est_sat.invJ@(ka*q_err[1:]*np.sign(q_err[0])*ee*ee))/nB2)
            # print(np.cross(Bbody,-est_sat.invJ@(ee*saturate(kw*w_err/bb)*bb)/nB2))
            # print(np.cross(Bbody,-est_sat.dist_torque(est_state,est.os_vecs))/nB2)
            # print(-offset_vec)
            print('quat',state[3:7])#,norm(q_err[1:]),norm(np.cross(q_err[1:],est.os_vecs['b']))/norm(est.os_vecs['b']))

        control = limit(ud,mtq_max)-offset_vec
        print('ctrl',control)
        print('av',state[0:3]*180.0/math.pi,norm(state[0:3])*180.0/math.pi)#,(180.0/math.pi)*norm(np.cross(state[0:3],est.os_vecs['b']))/norm(est.os_vecs['b']))
        print(orbt.in_eclipse())
        print("B real", real_vecs["b"])
        print("B est", Bbody)
        print("B should meas",real_vecs["b"]*real_sat.sensors[0].scale + np.hstack([j.bias for j in real_sat.sensors[0:3]]))
        print("B meas", sens[0:3])
        print("B est meas", Bbody*est_sat.sensors[0].scale + est_state[10:13])
        print("rot ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(Bbody,normalize(quat_mult(quat_inv(q),state[3:7])[1:])))/norm(Bbody)))
        print(va[:,-1],np.sqrt(wa[-1]))
        print("cov ax ang to Bbody", (180.0/np.pi)*np.arccos(np.abs(np.dot(Bbody,va[:,-1]))/norm(Bbody)))

        print(real_sat.sensors[0].bias[0],est.sat.sensors[0].bias[0]-real_sat.sensors[0].bias[0])

        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        # breakpoint()
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)
        dist_torq_hist[ind,:] = real_sat.last_dist_torq.copy()
        est_dist_torq_hist[ind,:] = est_sat.last_dist_torq.copy()
        act_torq_hist[ind,:] = real_sat.last_act_torq.copy()
        est_act_torq_hist[ind,:] = est_sat.last_act_torq.copy()
        sens_hist[ind,:] = sens.copy()
        eclipse_hist[ind] = orbt.in_eclipse()

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

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
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    abreal = state_hist[:,7:10]
    if not abias_missing:
        abest = est_state_hist[:,7:10]
        abdiff = abest-abreal
        gbiasdiff = (180/np.pi)*(est_state_hist[:,13:16]-state_hist[:,13:16])
    else:
        gbiasdiff = (180/np.pi)*(est_state_hist[:,10:13]-state_hist[:,13:16])
    if alt_title is None:
        base_title = "baseline_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        base_title = alt_title+"_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)

    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")
    plot_the_thing(abreal,title = "MTQ Bias",xlabel='Time (s)',norm = True,ylabel = 'MTQ Bias (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/abias_plot")

    if not abias_missing:
        plot_the_thing(abest,title = "Est MTQ Bias",xlabel='Time (s)',norm = True,ylabel = 'MTQ Magic Bias (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_abias_plot")
        plot_the_thing(abdiff,title = "MTQ Bias Error",xlabel='Time (s)',ylabel = 'MTQ Bias Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/aberr_plot")
        plot_the_thing(np.log10(matrix_row_norm(abdiff)),title = "Log MTQ Bias Error",xlabel='Time (s)',ylabel = 'Log MTQ Bias Error (log Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logab_plot")

        plot_the_thing(est_state_hist[:,13:16]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
        plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
        plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
        plot_the_thing(np.log10(matrix_row_norm(gbiasdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_loggb_plot")

        plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
        plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
        plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")
        plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,10:13]-state_hist[:,10:13])),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logmb_plot")

        plot_the_thing(est_state_hist[:,16:19],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
        plot_the_thing(state_hist[:,16:19],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
        plot_the_thing(est_state_hist[:,16:19]-state_hist[:,16:19],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")
        plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,16:19]-state_hist[:,16:19])),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logsb_plot")
    else:

        plot_the_thing(est_state_hist[:,10:13]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
        plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
        plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
        plot_the_thing(np.log10(matrix_row_norm(gbiasdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_loggb_plot")

        plot_the_thing(est_state_hist[:,7:10],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
        plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
        plot_the_thing(est_state_hist[:,7:10]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")
        plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,7:10]-state_hist[:,10:13])),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logmb_plot")

        plot_the_thing(est_state_hist[:,13:16],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
        plot_the_thing(state_hist[:,16:19],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
        plot_the_thing(est_state_hist[:,13:16]-state_hist[:,16:19],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")
        plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,13:16]-state_hist[:,16:19])),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logsb_plot")

    plot_the_thing(control_hist,title = "Control",xlabel='Time (s)',norm = True,ylabel = 'Control (N)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ctrl_plot")
    plot_the_thing(np.log10(angdiff),title = "Log Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (log deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logang_plot")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "Log AV Error",xlabel='Time (s)',ylabel = 'AV Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logav_plot")

    #generate statistics
    metrics = find_metrics(t_hist,angdiff)
    labels = ["title","al","kap","bet","dt","werrcov","mrpcov","mtmscale"]+["covest0","intcov"]+["conv time","tc","last 100 ang err mean","last 100 ang err max","last 100 av err mean","last 100 av err max"]
    info = [base_title,est.al,est.kap,est.bet,dt,werrcov,mrperrcov,mtmscale]+[np.diag(cov_estimate.copy()),int_cov.copy()]+[metrics.time_to_conv,metrics.tc_est,np.mean(angdiff[-100:]),np.amax(angdiff[-100:]),np.mean(matrix_row_norm(avdiff)[-100:]),np.amax(matrix_row_norm(avdiff)[-100:])]
    with open("paper_test_files/"+base_title+"/info", 'w') as f:
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
    sim.dist_torq_hist = dist_torq_hist
    sim.est_dist_torq_hist = est_dist_torq_hist
    sim.act_torq_hist = act_torq_hist
    sim.est_act_torq_hist = est_act_torq_hist
    sim.sens_hist = sens_hist
    sim.eclipse_hist = eclipse_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)


def bc_UKF_prop(orb = None, t0 = 0, tf = 60*90,dt = 1,alt_title = None,rand = False,mtmscale = 1e0,close = False,prop_missing = False,werrcov = 1e-17, mrperrcov = 1e-12):
    if not rand:
        np.random.seed(1)
    tlim00 = np.round(min([5*60,0.05*tf]))
    tlim0 = np.round(min([15*60,0.1*tf]))
    tlim1 = np.round(tf*0.25)
    tlim2 = np.round(tf*0.5)
    tlim3 = np.round(tf*0.75)

    #
    #real_sat
    real_sat = create_BC_sat(real=True,use_SRP = True,use_dipole = False,include_mtqbias = True,rand=rand,mtm_scale = mtmscale,use_prop = True)
    w0 = random_n_unit_vec(3)*1.0*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    if orb is None:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,use_SRP = True,use_dipole = False,include_mtqbias = True,estimate_mtq_bias =True,mtm_scale = mtmscale,use_prop = not prop_missing,estimate_prop_torq = not prop_missing)
    estimate = np.zeros(22-3*prop_missing)
    estimate[3] = 1
    angcov = 10
    avcov = (math.pi/180.0)**2.0

    if close:
        estimate[0:3] = w0
        estimate[3:7] = q0
        avcov = ((0.5*math.pi/180)/(60))**2.0
        angcov = (0.5*math.pi/180)**2.0
    if prop_missing:
        cov_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(1e-7*mtmscale)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0)
    else:
        cov_estimate = block_diag(np.eye(3)*avcov,np.eye(3)*angcov,np.eye(3)*0.1**2.0,np.eye(3)*(1e-7*mtmscale)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0,np.eye(3)*(1e-6)**2.0)


    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*dt*block_diag(np.block([[np.eye(3)*1e-14,],[1e-6*np.eye(3),1e-6*np.eye(3)]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    # werrcov = 1e-17#1e-20#0#1e-20#0#1e-20
    # mrperrcov = 1e-12#1e-10#16
    int_cov =  dt*block_diag(np.block([[np.eye(3)*werrcov,np.zeros((3,3))],[np.zeros((3,3)),dt*np.eye(3)*mrperrcov]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.actuators if j.has_bias and j.estimated_bias]),np.diagflat([j.bias_std_rate**2.0*j.scale**2.0  for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.diagflat([np.diag(j.std**2.0) for j in est_sat.disturbances if j.time_varying and j.estimated_param]))


    mtq_max = [j.max for j in est_sat.actuators]
    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = dt,quat_as_vec = False)

    est.use_cross_term = True
    est.al = 1.0#1e-3#e-3#1#1e-1#al# 1e-1
    est.kap =  3 - 21 + 3*prop_missing#15#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.include_int_noise_separately = False
    est.include_sens_noise_separately = False
    est.scale_nonseparate_adds = False
    est.included_int_noise_where = 2
    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),22))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    comp_time_hist =  np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_dist_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    est_act_torq_hist = np.nan*np.zeros((int((tf-t0)/dt),3))
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    eclipse_hist = np.nan*np.zeros(int((tf-t0)/dt))

    est_sat0 = copy.deepcopy(est_sat)

    bdotgain = 1e6
    kw = 100#5
    ka = 25#0.001
    ee = 0.005
    bb = 100#1#0.15
    while t<tf:
        print("+++++++++++++++++++++++++++++")
        print(t)
        #simulate sensors
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)


        real_sbias = np.concatenate([j.bias for j in real_sat.attitude_sensors if j.has_bias])
        real_abias = np.concatenate([j.bias for j in real_sat.actuators if j.has_bias])
        real_dist =  np.concatenate([j.main_param for j in real_sat.disturbances])



        #run estimator
        tt0 = time.process_time()
        est.update(control,sens,orbt)
        tt1 = time.process_time()
        est_state = est.use_state.val



        real_full_state = np.concatenate([state.copy(),real_abias.copy(),real_sbias.copy(),real_dist.copy()])
        if prop_missing:
            unbiased_sens = sens-est_state[10:]
        else:
            unbiased_sens = sens-est_state[10:-3]
        print('real state ',state)
        print('est  state ',est_state[0:7])
        # autovar = np.sqrt(np.diagonal(est.use_state.cov))

        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        # print('mrp ',(180/np.pi)*norm((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi)-np.pi))
        # print((180/np.pi)*((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi) - np.pi))
        # print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi,4*norm(np.arctan(autovar[3:6]/2.0))*180.0/math.pi)

        print('av ',norm(state[0:3]-est_state[0:3])*180.0/math.pi)
        print('av est ',est_state[0:3]*180.0/math.pi,norm(est_state[0:3])*180.0/math.pi)
        # print(autovar[0:3]*180.0/math.pi)

        # print('ab  ',norm(est_state[7:10]-real_abias))
        # # print((est_state[7:10]-real_abias))
        # # print(autovar[6:9])
        #
        # print('gb ',norm(est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # # print((est_state[13:16]-real_sbias[3:6])*180.0/math.pi)
        # # print(autovar[12:15]*180.0/math.pi)
        #
        # print('mb (x1e8/scale) ',norm(est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        # # print((est_state[10:13]-real_sbias[0:3])*1e8/est.sat.attitude_sensors[0].scale)
        # # print(autovar[9:12]*1e8/est.sat.attitude_sensors[0].scale)
        #
        # print('sb (x1e4) ',norm(est_state[16:19]-real_sbias[6:9])*1e4)
        # # print((est_state[16:19]-real_sbias[6:9])*1e4)
        # # print(autovar[15:18]*1e4)
        #
        # if not prop_missing:
        #     print('prop ',norm(est_state[19:]-real_dist))
        #     # print((est_state[16:19]-real_sbias[6:9])*1e4)
        #     # print(autovar[18:])



        #control law
        nB2 = norm(orbt.B)**2.0
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]

        # print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(q,state[3:7]),-1,1)**2.0 ))
        # print('av ',norm(state[0:3]-w)*180.0/math.pi)
        # print(sens)
        # print(orbt.in_eclipse())
        wdes = np.zeros(3)
        w_err =w-wdes
        Bbody = rot_mat(q).T@orbt.B
        if t<tlim00:
            ud = np.zeros(3)
            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])
        elif t<tlim0:
            #bdot
            ud = -bdotgain*(sens[0:3]-prev_sens[0:3])/(dt*mtmscale)
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
            # ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
            ud = np.cross(Bbody,-ka*q_err[1:]*np.sign(q_err[0])*ee*ee - ee*saturate(kw*est_sat.J@w_err/bb)*bb - est_sat.dist_torque(est_state,est.os_vecs))/nB2
            # print(est_sat.dist_torque(est_state,est.os_vecs))
            # print(est.sat.dist_torque(est_state,est.os_vecs))
            # print(est_sat.last_dist_torq)
            # print(est.sat.last_dist_torq)

            offset_vec = np.concatenate([j.bias for j in est_sat.actuators])
            print("p ",-ka*q_err[1:]*np.sign(q_err[0])*ee*ee)
            print("d ",-ee*saturate(kw*est_sat.J@w_err/bb)*bb)
            print("dist ",-est_sat.dist_torque(est_state,est.os_vecs))
        control = limit(ud,mtq_max)-offset_vec
        print("ctrl: ",control)
        print("est torq act: ",np.cross(control,Bbody))
        print("est torq all: ",np.cross(control,Bbody) + est_sat.dist_torque(est_state,est.os_vecs))
        print("est acc: ",est.sat.invJ@np.cross(control,Bbody))
        print("est acc all: ",est.sat.invJ@(np.cross(control,Bbody) + est_sat.dist_torque(est_state,est.os_vecs)))
        print("normalized Bbody: ",normalize(Bbody))
        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators if j.has_bias]+[j.bias*j.scale for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        t_hist[ind] = t
        comp_time_hist[ind] = (tt1-tt0)
        dist_torq_hist[ind,:] = real_sat.last_dist_torq.copy()
        est_dist_torq_hist[ind,:] = est_sat.last_dist_torq.copy()
        act_torq_hist[ind,:] = real_sat.last_act_torq.copy()
        est_act_torq_hist[ind,:] = est_sat.last_act_torq.copy()
        sens_hist[ind,:] = sens.copy()
        eclipse_hist[ind] = orbt.in_eclipse()

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        prev_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-2, atol=1e-6)#,jac = ivp_jac)
        real_sat.dynamics(state,control,orbt,verbose = False,save_dist_torq = True,save_act_torq = True,update_actuator_noise = False)
        # breakpoint()
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
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_vec3(quatdiff[j,:],0)/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    abreal = state_hist[:,7:10]
    abest = est_state_hist[:,7:10]
    abdiff = abest-abreal
    gbiasdiff = (180/np.pi)*(est_state_hist[:,13:16]-state_hist[:,13:16])
    if alt_title is None:
        base_title = "bcprop_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        base_title = alt_title+"_bcprop_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("paper_test_files/"+base_title)

    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "AV",xlabel='Time (s)',norm = True,ylabel = 'AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/av_plot")
    plot_the_thing(state_hist[:,3:7],title = "Quat",xlabel='Time (s)',ylabel = 'Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/quat_plot")

    plot_the_thing(angdiff,title = "Angular Error",xlabel='Time (s)',ylabel = 'Angular Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/ang_plot")
    plot_the_thing(avdiff,title = "AV Error",xlabel='Time (s)',ylabel = 'AV Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/averr_plot")

    plot_the_thing(mrpdiff,title = "MRP Error",xlabel='Time (s)',ylabel = 'MRP Error (deg)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mrp_plot")
    plot_the_thing(est_state_hist[:,3:7],title = "Est Quat",xlabel='Time (s)',ylabel = 'Est Quat',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_quat_plot")
    plot_the_thing(est_state_hist[:,0:3]*180.0/math.pi,title = "Est AV",xlabel='Time (s)',norm = True,ylabel = 'Est AV (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_av_plot")

    plot_the_thing(abest,title = "Est MTQ Bias",xlabel='Time (s)',norm = True,ylabel = 'MTQ Magic Bias (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_abias_plot")
    plot_the_thing(abreal,title = "MTQ Bias",xlabel='Time (s)',norm = True,ylabel = 'MTQ Bias (Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/abias_plot")
    plot_the_thing(abdiff,title = "MTQ Bias Error",xlabel='Time (s)',ylabel = 'MTQ Bias Error (Am2)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/aberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(abdiff)),title = "Log MTQ Bias Error",xlabel='Time (s)',ylabel = 'Log MTQ Bias Error (log Am2)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logab_plot")

    plot_the_thing(est_state_hist[:,13:16]*180.0/math.pi,title = "Est Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_gbias_plot")
    plot_the_thing(state_hist[:,13:16]*180.0/math.pi,title = "Gyro Bias",xlabel='Time (s)',norm = True,ylabel = 'Gyro Bias (deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gbias_plot")
    plot_the_thing(gbiasdiff,title = "Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (deg/s)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/gberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(gbiasdiff)),title = "Log Gyro Bias Error",xlabel='Time (s)',ylabel = 'Gyro Bias Error (log deg/s)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_loggb_plot")

    plot_the_thing(est_state_hist[:,10:13],title = "Est MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'Est MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_mbias_plot")
    plot_the_thing(state_hist[:,10:13],title = "MTM Bias",xlabel='Time (s)',norm = True,ylabel = 'MTM Bias (scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mbias_plot")
    plot_the_thing(est_state_hist[:,10:13]-state_hist[:,10:13],title = "MTM Bias Error",xlabel='Time (s)',ylabel = 'MTM Bias Error (scaled nT)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/mberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,10:13]-state_hist[:,10:13])),title = "Log MTM Bias Error",xlabel='Time (s)',ylabel = 'Log MTM Bias Error (log scaled nT)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logmb_plot")

    plot_the_thing(est_state_hist[:,16:19],title = "Est Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Est Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_sbias_plot")
    plot_the_thing(state_hist[:,16:19],title = "Sun Bias",xlabel='Time (s)',norm = True,ylabel = 'Sun Bias ()',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sbias_plot")
    plot_the_thing(est_state_hist[:,16:19]-state_hist[:,16:19],title = "Sun Bias Error",xlabel='Time (s)',ylabel = 'Sun Bias Error ()', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/sberr_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,16:19]-state_hist[:,16:19])),title = "Log Sun Bias Error",xlabel='Time (s)',ylabel = 'Log Sun Bias Error (log ())',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logsb_plot")

    plot_the_thing(est_dist_torq_hist,title = 'Estimated Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/est_dist_torq_plot")
    plot_the_thing(dist_torq_hist,title = 'Dist Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Dist Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/dist_torq_plot")
    plot_the_thing(est_dist_torq_hist-dist_torq_hist,title = 'Dist Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'Dist Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/dist_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_dist_torq_hist-dist_torq_hist)),title = 'Log Dist Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log Dist Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/_log_dist_torq_err_plot")

    plot_the_thing(est_act_torq_hist,title = 'Estimated act Torq',xlabel = 'Time (s)',norm = True, ylabel = 'Est act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/est_act_torq_plot")
    plot_the_thing(act_torq_hist,title = 'act Torq',xlabel = 'Time (s)',norm = True, ylabel = 'act Torq (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/act_torq_plot")
    plot_the_thing(est_act_torq_hist-act_torq_hist,title = 'act Torq Error',xlabel = 'Time (s)',norm = True, ylabel = 'act Torq Err (Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/act_torq_err_plot")
    plot_the_thing(np.log10(matrix_row_norm(est_act_torq_hist-act_torq_hist)),title = 'Log act Torq Error',xlabel = 'Time (s)',norm = False, ylabel = 'Log act Torq Err (log Nm)', xdata = np.array(t_hist),save = True, plot_now = False, save_name = "paper_test_files/"+base_title+"/_log_act_torq_err_plot")

    plot_the_thing(state_hist[:,19:22],title = "Prop Torq",xlabel='Time (s)',norm = True,ylabel = 'Prop Torq (Nm)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/prop_plot")

    if not prop_missing:
        plot_the_thing(est_state_hist[:,19:22],title = "Est Prop Torq",xlabel='Time (s)',norm = True,ylabel = 'Est Prop Torq (Nm)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/est_prop_plot")
        plot_the_thing(est_state_hist[:,19:22]-state_hist[:,19:22],title = "Prop Torq Error",xlabel='Time (s)',ylabel = 'Prop Torq Error (Nm)', norm=True,xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/properr_plot")
        plot_the_thing(np.log10(matrix_row_norm(est_state_hist[:,19:22]-state_hist[:,19:22])),title = "Log Prop Torq Error",xlabel='Time (s)',ylabel = 'Log Prop Torq Error (log Nm)',xdata = np.array(t_hist),save =True,plot_now = False, save_name = "paper_test_files/"+base_title+"/_logprop_plot")

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
    sim.dist_torq_hist = dist_torq_hist
    sim.est_dist_torq_hist = est_dist_torq_hist
    sim.act_torq_hist = act_torq_hist
    sim.est_act_torq_hist = est_act_torq_hist
    sim.sens_hist = sens_hist
    sim.eclipse_hist = eclipse_hist
    # try:
    with open("paper_test_files/"+base_title+"/data", "wb") as fp:   #Pickling
        pickle.dump(sim, fp)

#crassidis on my base case
def bc_crassidis_UKF(orb = None, t0 = 0, tf = 60*90,dt = 1,alt_title = None,rand = False,close = False):
    if not rand:
        np.random.seed(1)
    tlim00 = 60*30
    tlim0 = 5*60
    tlim1 = 20*60
    tlim2 = 50*60
    tlim3 = 70*60

    #
    #real_sat
    real_sat = create_BC_sat(real=True,use_SRP = True,use_dipole = False,include_mtqbias = False,rand=rand)
    w0 = random_n_unit_vec(3)*1.0*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    if orb is None:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,use_SRP = True,use_dipole = False,include_mtqbias = False)
    estimate = np.zeros(10)
    estimate[3] = 1

    angcov = (50*math.pi/180.0)**2.0
    avcov = np.nan

    if close:
        estimate[0:3] = w0
        estimate[3:7] = q0
        angcov = (0.5*math.pi/180)**2.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*angcov,np.eye(3)*(20*(math.pi/180.0)/3600.0)**2.0)
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
    if alt_title is None:
        base_title = "crassidis_baseline_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        base_title = alt_title+"_crassidis_"+str(dt)+"_"+time.strftime("%Y%m%d-%H%M%S")
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
    my_tf = 60*60*12
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

    longrun_crass = 60*60*24
    shortrun_crass =  60*60*12
    myrun = 60*60*6
    myrun_short = 60*60*3
    ##replicate his results
    # crassidis_UKF_attitude_errors_replication(crass_orb,tf = shortrun_crass)
    # crassidis_UKF_attNbias_errors_replication(crass_orb, tf = shortrun_crass)
    #compare his to mine on "real world" situation but with his sensors and setup
    #without dipole
    # crassidis_UKF_real_world_no_dipole(crass_orb,tf=longrun_crass)
    # crassidis_UKF_real_world_no_dipole(crass_orb,dt = 1,tf=shortrun_crass)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=longrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -18,useSR = True)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 1,tf=shortrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -18,useSR = True)
    # #with
    # crassidis_UKF_real_world(crass_orb,tf=longrun_crass)
    # crassidis_UKF_real_world(crass_orb,dt = 1,tf=shortrun_crass)
    # crassidis_UKF_mine(crass_orb,dt = 10,tf=longrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -21,useSR = True)
    # crassidis_UKF_mine(crass_orb,dt = 1,tf=shortrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -21,useSR = True)
    # #
    # # # #when starting close to truth -- no attitude or AV error.
    # crassidis_UKF_real_world_no_dipole(crass_orb,tf=longrun_crass,close = True)
    # crassidis_UKF_real_world_no_dipole(crass_orb,dt = 1,tf=shortrun_crass,close = True)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 10,tf=longrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -18,useSR = True,close = True)
    # crassidis_UKF_mine_no_dipole(crass_orb,dt = 1,tf=shortrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -18,useSR = True,close = True)
    #
    # crassidis_UKF_real_world(crass_orb,tf=longrun_crass,close = True)
    # crassidis_UKF_real_world(crass_orb,dt = 1,tf=shortrun_crass,close = True)
    # crassidis_UKF_mine(crass_orb,dt = 10,tf=longrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -18,useSR = True,close = True)
    # crassidis_UKF_mine(crass_orb,dt = 1,tf=shortrun_crass,mrperrcov = 1e-12,werrcovmult = 1e-1,kap = -18,useSR = True,close = True)
    #
    # #mine, his in BC, cubesat.
    # bc_UKF_baseline(orb = myorb,tf= myrun_short)
    # # # #bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 0,mrperrcov = 0)
    # # # #bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 1e-12,mrperrcov = 1e-12)
    # # # bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 1e-17,mrperrcov = 0)
    # # # #bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 0,mrperrcov = 1e-12)
    # # # bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 0,mrperrcov = 1e-16)
    # # bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 1e-17,mrperrcov = 1e-16)
    # # bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 1e-12,mrperrcov = 0)
    # # bc_UKF_baseline(orb = myorb,tf=myrun_short,werrcov = 1e-12,mrperrcov = 1e-16)
    # #bc_crassidis_UKF(orb = myorb,tf=myrun)
    #
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,close = True,alt_title = "close_baseline")
    # # #bc_crassidis_UKF(orb = myorb,tf=myrun,close = True)
    #
    # #actuator noise on/off
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "anoise_missing",anoise_missing = True)
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "anoise_there",anoise_missing = False)
    #
    # # #actuator bias on/off
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "abias_missing",abias_missing = True)
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "abias_there",abias_missing = False)
    #
    # #disturbanceds on/off
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "dist_missing",dist_missing = True)
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "dist_there",dist_missing = False)
    #
    # #BC propulsion
    # bc_UKF_prop(orb = myorb,tf=myrun_short,alt_title = "prop_there",prop_missing = False,)
    # bc_UKF_prop(orb = myorb,tf=myrun_short,alt_title = "prop_missing",prop_missing = True)
    #
    #
    # # # close versions
    # #actuator noise on/off
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "anoise_missing_close",anoise_missing = True,close = True)
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "anoise_there_close",anoise_missing = False,close = True)
    #
    # # #actuator bias on/off
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "abias_missing_close",abias_missing = True,close = True)
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "abias_there_close",abias_missing = False,close = True)
    #
    # #disturbanceds on/off
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "dist_missing_close",dist_missing = True,close = True)
    # bc_UKF_baseline(orb = myorb,tf=myrun_short,alt_title = "dist_there_close",dist_missing = False,close = True)

    #BC propulsion
    # bc_UKF_prop(orb = myorb,tf=myrun_short,alt_title = "prop_missing_close",prop_missing = True,close = True)
    # bc_UKF_prop(orb = myorb,tf=myrun_short,alt_title = "prop_there_close",prop_missing = False,close = True)

    #other tets
    bc_UKF_baseline(orb = myorb,tf=myrun_short,close = False,alt_title = "eclipse",care_about_eclipse = True)
    #random trials in BC world
