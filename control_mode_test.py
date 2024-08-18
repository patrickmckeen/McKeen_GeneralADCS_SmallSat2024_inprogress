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


np.random.seed(1)
avcov = ((0.5*math.pi/180)/(60))**2.0
angcov = (0.5*math.pi/180)**2.0
werrcov = 1e-17
mrperrcov = 1e-12

wie_q0 = normalize(np.array([0.153,0.685,0.695,0.153]))#zeroquat#normalize(np.array([0.153,0.685,0.695,0.153]))
wie_w0 = np.array([0.01,0.01,0.001])#np.array([0.53,0.53,0.053])#/(180.0/math.pi)

wie_base_sat_w_GPS = create_Wie_sat_w_GPS(real=True,rand=False)
wie_base_est_sat_w_GPS = create_Wie_sat_w_GPS( real = False, rand=False)
wie_disturbed_sat_w_GPS = create_Wie_sat_w_GPS(    real=True,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,include_mtmbias = True)
wie_disturbed_est_sat_w_GPS = create_Wie_sat_w_GPS(real=False,rand=False,use_gg = True, use_drag = True, use_dipole = True, use_SRP = True, estimate_dipole = True,dipole_mag_max=50,include_magic_noise = True,include_magicbias = True,estimate_magic_bias = True,include_mtmbias = True,estimate_mtm_bias = True)

dt = 1
planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+120*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR},{0.2:(PointingGoalVectorMode.PROVIDED_MRP,np.zeros(3))},{0.2:np.zeros(3)})
planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+5*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+120*sec2cent:GovernorMode.PLAN_AND_TRACK_LQR}, {0.2:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),0.22+220*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3)),0.22+250*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),0.22+400*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),0.22+700*sec2cent:(PointingGoalVectorMode.NO_GOAL,np.zeros(3)),0.22+1000*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})
planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+10*sec2cent:GovernorMode.BDOT_WITH_EKF,0.22+50*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC}, {0.22:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})
planner_goals = Goals({0.2:GovernorMode.NO_CONTROL,0.22+200*sec2cent:GovernorMode.PLAN_AND_TRACK_MPC}, {0.22:(PointingGoalVectorMode.NADIR,np.zeros(3)),0.22+1500*sec2cent:(PointingGoalVectorMode.ZENITH,np.zeros(3)),0.22+3000*sec2cent:(PointingGoalVectorMode.NADIR,np.zeros(3))}, {0.2:unitvecs[0]})

plannertest = ["planner", wie_base_est_sat_w_GPS,wie_base_sat_w_GPS,True,lovera_w0_vslow,lovera_q0,wie_base_cov0_estimate,True,planner_goals,1200,lovera_orb_file1,""]

tests = [plannertest]

mpc_dt = 1
mpc_ang_limit = 10
mpc_angwt_low = 1e2#1e4
mpc_angwt_high = 1e4#1e12
mpc_avwt = 1e8#1e2
mpc_avangwt = 1e7#1e2
mpc_extrawt = 0
mpc_uwt_from_plan = 1e-3#1e-6
mpc_uwt_from_prev = 0
mpc_lqrwt_mult = 0.0#1.0
mpc_extra_tests = 0
mpc_tol = 1e-15
mpc_Nplot = 0
mpc_gain_info = [mpc_dt,mpc_ang_limit,mpc_angwt_low,mpc_angwt_high,mpc_avwt,mpc_avangwt,mpc_extrawt,mpc_uwt_from_plan,mpc_uwt_from_prev,mpc_lqrwt_mult,mpc_extra_tests,mpc_tol,mpc_Nplot]#[1,10,100,1e6,1,0,1e-6,0]


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
    est.use_cross_term = True
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


    mpc = TrajectoryMPC(mpc_gain_info,est.sat)



    wa = unitvecs[2]*1
    wb = np.zeros(3)
    qa = unitvecs4[0]
    qb = unitvecs4[0]
    diff = mpc.reduced_state_err(np.concatenate([wa,qa]),np.concatenate([wb,qb]),0)
    wc = diff[0:3]
    mrpc = diff[3:]
    assert wc == unitvecs[2]*1
    assert mrpc == np.zeros(3)


    wa = np.zeros(3)
    wb = unitvecs[1]*1
    qa = unitvecs4[0]
    qb = unitvecs4[0]
    diff = mpc.reduced_state_err(np.concatenate([wa,qa]),np.concatenate([wb,qb]),0)
    wc = diff[0:3]
    mrpc = diff[3:]
    assert wc == -unitvecs[1]*1
    assert mrpc == np.zeros(3)


    wa = unitvecs[1]*1
    wb = np.zeros(3)
    qa = unitvecs4[0]
    qb = unitvecs4[1]
    diff = mpc.reduced_state_err(np.concatenate([wa,qa]),np.concatenate([wb,qb]),0)
    wc = diff[0:3]
    mrpc = diff[3:]
    assert wc == unitvecs[1]*1
    assert mrpc == 2*unitvecs[1]




    wa = unitvecs[0]*0.05 + unitvecs[2]*1
    wb = -unitvecs[0]*0.03 - unitvecs[1]*0.3
    qa = unitvecs4[0]
    qb = normalize(np.array([0,0,1,0]))
    diff = mpc.reduced_state_err(np.concatenate([wa,qa]),np.concatenate([wb,qb]),0)
    wc = diff[0:3]
    mrpc = diff[3:]
    assert wc == 0
    assert mrpc == 1


    wa = unitvecs[0]*0.05 + unitvecs[2]*1
    wb = -unitvecs[0]*0.03 - unitvecs[1]*0.3
    qa = normalize(np.array([1,0,1,0]))
    qb = normalize(np.array([0,0,1,0]))
    diff = mpc.reduced_state_err(np.concatenate([wa,qa]),np.concatenate([wb,qb]),0)
    wc = diff[0:3]
    mrpc = diff[3:]
    assert wc == 0
    assert mrpc == 1

#     planner_settings = PlannerSettings(est_sat,tvlqr_len = 100,tvlqr_overlap = 40,dt_tp = 1,precalculation_time = 10,traj_overlap = 50,debug_plot_on = False,bdot_on = 0,
#                 include_gg = True,include_resdipole = False, include_prop = False, include_drag = False, include_srp = False, include_gendist = False)
#     planner_settings.maxIter = 2000
#     planner_settings.maxIter2 = 2000
#     planner_settings.gradTol = 1.0e-6#1e-9#1e-09
#     planner_settings.costTol =      1.0e-4#1e-6#0.0000001
#     planner_settings.ilqrCostTol =  1.0e-2#1e-4#0.000001
#     planner_settings.zCountLim = 5
#     planner_settings.maxOuterIter = 25#70#50#20#20#25#25#1#15#diop
#     planner_settings.maxIlqrIter = 40#350#0#100#300#diop0#50#150#50#25#25#1#30
#     planner_settings.maxOuterIter2 = 25#70#50#20#20#25#25#1#15#diop
#     planner_settings.maxIlqrIter2 = 40
#     planner_settings.default_traj_length = 500
#     planner_settings.penInit = 1.0
#     planner_settings.penMax = 1.0e30
#     planner_settings.penInit2 = 1.0e5
#     planner_settings.penScale2 = 10
#     planner_settings.penMax2 = 1.0e30
#     planner_settings.maxLsIter = 20
#
#     planner_settings.mtq_control_weight = 1.0
#     planner_settings.rw_control_weight = 0.001
#     planner_settings.magic_control_weight = 0.0001
#     planner_settings.rw_AM_weight = 0.1
#     planner_settings.rw_stic_weight = 0.01
#
#     planner_settings.angle_weight = 1.0#0.1#200.0
#     planner_settings.angvel_weight = 1.0#0*10.0#0.1#0.01#0*10.0#.01#01#0.000001
#     planner_settings.u_weight = 1.0#1e-1#1e-1 #0*0.0001#0.0#0.0000001
#     planner_settings.u_with_mag_weight = 0.0
#     planner_settings.av_with_mag_weight = 0.0#0.1
#     planner_settings.ang_av_weight = 0.1#0*10.0#0*100.0
#     planner_settings.angle_weight_N = 10.0#2000.0
#     planner_settings.angvel_weight_N = 10.0#0*10.0#0.1#0*10.0
#     planner_settings.av_with_mag_weight_N = 0.0
#     planner_settings.ang_av_weight_N = 1.0#0*100.0
#
#     planner_settings.angle_weight2 = 1.0#0.1#100.0
#     planner_settings.angvel_weight2 = 1.0#1e-2#1# 1e-1
#     planner_settings.u_weight2 = 1.0
#     planner_settings.u_with_mag_weight2 = 0.0
#     planner_settings.av_with_mag_weight2 = 0.0
#     planner_settings.ang_av_weight2 = 0.0#0*0.2#0.1#0.0001*0
#     planner_settings.angle_weight_N2 = 10.0#1000.0
#     planner_settings.angvel_weight_N2 = 10.0#1.0#0*1.0#0.0
#     planner_settings.av_with_mag_weight_N2 = 0.0
#     planner_settings.ang_av_weight_N2 = 0.0#1.0#0*0.2
#
#     planner_settings.angle_weight_tvlqr = 1.0#30#0.1#10.0
#     planner_settings.angvel_weight_tvlqr = 100.0#0.1#0.01#0#0.001#0.01#0.01#0*0.01#1#$5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0.01#0.00001
#     planner_settings.u_weight_tvlqr = 1.0 #1e-2#0.001#0.00001#0.001#0.00000001#0.3#5#0.1#0.000000001#0.0000000001
#     planner_settings.u_with_mag_weight_tvlqr = 0.0#0.01
#     planner_settings.av_with_mag_weight_tvlqr = 0.0#0.01#0.0#1.0#5.0#0#0.005#0.01
#     planner_settings.ang_av_weight_tvlqr = 0.0#0.02#0*2#0.05#0.01#0.5#0.5#0#10#0.01#0.1#0*0.1#0*10#1000#-0.1#0.0001#0.01
#     planner_settings.angle_weight_N_tvlqr = 10.0#1*500#1#500#200#30#20.0
#     planner_settings.angvel_weight_N_tvlqr = 100.0#$1.0*10#0.1*10#0.1#0.1#0#0.01#1#1#0*0.1#0*0.01#1#5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0*0.00001
#     planner_settings.av_with_mag_weight_N_tvlqr = 0.0#1.0
#     planner_settings.ang_av_weight_N_tvlqr = 0.0#0.2#0.1#0*2#5.0#0#0.05*100#0*0.01#0.01#0.5#0.5#0#10#0.01#0.1#0.01#0.0
#
#
#     planner_settings.regScale = 1.6#1.6#1.6#1.6##2#1.6#4#1.6#2#1.6#1.8#2.0#1.8#5#10#1e3#1.8#2#1.6#1.6#2#1.6#2.0#1.6#3#1.6#5#2#5#1.6#5#1.8
#     planner_settings.regMax = 1.0e20
#     planner_settings.regMax2 = 1.0e20
#     planner_settings.regMin = 1.0e-8#16
#     planner_settings.regBump = 10.0#5#1e-3#1e-2#10.0#1e-1#2.0#10.0#1.0#10.0#1.0 #1.0#10.0#$0.01#0.1#2.0#1e-2#0.1#1.0#0.1#10#1e-10#2.0#1.0#1.0#1.0#10.0#100#0.1#10#1.0 #1.0#1.0#50#10#20.0#100.0
#     planner_settings.regMinCond = 2 #0 means that regmin is basically ignored, and regulaization goes up and down without bounds, case 1 means regularization is always equal to or greater than regmin, case 2 means if the regularization falls below regmin then it clamps to 0
#     planner_settings.regMinCond2 = 2
#     planner_settings.regBumpRandAddRatio = 0.0#1e-20#1e-16#1e-3#4e-3#*1e-4
#     # planner_settings.useEVmagic = 0;#1 #use the eigendecomposition rather than simple regularization
#     # planner_settings.SPDEVreg = 1;#0#1 #regularize/add even if matrix is SPD
#     # planner_settings.SPDEVregAll = 0;#0 reg SPD matrix by adding rho*identity matrix (otherwise do the EV magic reg)
#     # planner_settings.rhoEVregTest = 1;#1 #test if reset is needed (in EV magic case) by comparing to a multiple of rho (otherwise compare to regmin)
#     planner_settings.useDynamicsHess = 0
#     # planner_settings.EVregTestpreabs = 1;#0 #complete the reset test before absolute value is taken
#     # planner_settings.EVaddreg = 0;#0 #do EV magic by adding a value to the eigs that are too small (otherwise clamp to a minimum value)
#     # planner_settings.EVregIsRho = 1; #1 #clamp to or add rho (otherwise regmin)
#     planner_settings.dt_tp = 10.0
#     planner_settings.dt_tvlqr = 1.0
#     planner_settings.useConstraintHess = 0
#
#     planner_settings.control_limit_scale = 0.3
#     planner_settings.rho = 0.0#1e-10#0.01#1.0#0.1#1.0#0.1#0.001#1.0#0.01#1.0
#     planner_settings.wmax = 1.0/60.0
#     planner_settings.considerVectorInTVLQR = 0
#     planner_settings.useRawControlCost = True
#     planner_settings.whichAngCostFunc = 2#2 seems best.0.1#0*2#5.0#0#0.05*100#0*0.01#0.01#0.5#0.5#0#10#0.01#0.1#0.01#0.0
#     planner_settings.bdotgain = 10000000/(planner_settings.dt_tp**2)
#
#     adcsys = ADCS(est_sat,orbit_estimator,est,control_laws,use_planner = True,planner = None,planner_settings = planner_settings,goals=goals,prop_schedule=None,dt = 1,control_dt = 1,tracking_LQR_formulation = 0)
#
#     state0 = np.zeros(real_sat.state_len)
#     state0[0:3] = w0
#     state0[3:7] = q0
#     run_sim(orb_file,state0,copy.deepcopy(real_sat),adcsys,tf=tf,dt = dt,alt_title = title,rand=False)
# except Exception as ae:
#     if isinstance(ae, KeyboardInterrupt):
#         raise
#     else:
#         type, value, tb = sys.exc_info()
#         traceback.print_exc()
#         last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
#         frame = last_frame().tb_frame
#         ns = dict(frame.f_globals)
#         ns.update(frame.f_locals)
#         code.interact(local=ns)
#         pass
