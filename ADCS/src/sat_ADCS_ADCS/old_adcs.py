import math
import time
import numpy as np
import matplotlib.pyplot as plt
from skyfield import api
from helpers import *
# import flight_adcs.trajectory_planner.TrajPlan_Remake_Unix.src.build.tplaunch as tplaunch
import trajectory_planner.src.build.tplaunch as tplaunch
import trajectory_planner.src.build.pysat as pysat

from trajectory import Trajectory
from orbit import Orbit
from orbital_state import Orbital_State
from estimator import estimated_nparray
from sensors import *
import copy
#
# class PlannerSettings:
#     def __init__(self,sat,dt_tvlqr = 1,tvlqr_len = None,tvlqr_overlap = 0,dt_tp = None,precalculation_time = 100,default_traj_length = 1000,traj_overlap = 10,debug_plot_on = False,bdot_on = 1,
#             include_gg = False,include_resdipole = False, include_prop = False, include_drag = False, include_srp = False, include_gendist = False):
#         self.precalculation_time = precalculation_time
#         #TODO: big one--this needs to be redone and cleaned up for different planners, etc.
#
#         self.dt_tvlqr = dt_tvlqr
#         if tvlqr_len is None:
#             tvlqr_len = default_traj_length
#         self.tvlqr_len = tvlqr_len
#         if dt_tp is None:
#             dt_tp = 10*dt_tvlqr
#         self.tvlqr_overlap = tvlqr_overlap
#         self.dt_tp = dt_tp
#         self.default_traj_length = default_traj_length
#         self.traj_overlap = traj_overlap
#         self.debug_plot_on = debug_plot_on
#         self.bdot_on = bdot_on
#
#
#         #TRAJECTORY PLANNER SETTINGS
#         #Set qSettings
#         self.bdotgain = 10000000#1000000000.
#         self.gyrogainH = 0
#         self.gyrogainL = 0
#         #dampgain = -0.5#0.02#-0.5# -0.0002
#         self.dampgainH = -2000.0#-0.5#0.02#-0.5# -0.0002
#         self.dampgainL = -1000.0##0.02#-0.5# -0.0002
#         self.velgainH =  -50.0#0.01#0.02#-0.1#-0.0001
#         self.velgainL =  -200.0#0.01#-0.001#0.02#-0.1#-0.0001
#         self.quatgainH =  -2.0#0.7#0*-1.0#1#0.1#0.1#0.0005#0.05#0.005#0.0005#0.01#0.00001
#         self.quatgainL =  -0.001#-0.0005#0*-1.0#1#0.1#.001#0.0005#0.05#0.005#0.0005#0.01#0.00001x
#         self.HLangleLimit = 10.0*math.pi/180
#         self.Nslew = 0
#         self.randvalH = 0.001#0.001#2.0#0.5
#         self.randvalL = 0*0.00001
#         self.umaxmultH = 1.5#1.5#3.0
#         self.umaxmultL = 1.5
#
#         '''THIS IS OLD, values are wrong, but the idea is correct-->due to units and constraints, the added cost per component per timestep can vary in magnitude.
#         with a velocity magnitude limit of 0.006 rad/s (~0.34 deg/s), a time step with the satellite spinning at max rate will add ~2e-5 * swpoint to the cost due to velocity
#         if the satellite is pointing 180 degrees in the wrong direction (based on vector pointing), the cost added in one timestep would be ~5e0 * sv1
#         if the satellite is actuating all 3 MTQs at the limit of 0.15 Am^2, this will add ~3e-2 * su
#         lagrange multipliers and penalties in their maximum (and with the added 100 multiplier on the omega^2 term) will each add costs per time step of roughly 1e9 to 1e12 if they are off by 1x their limit (u=2*umax, for example). Note that lagrange multiplier and penalty do not start at their maximum. The cost added per component per timestep does not need to compete with those, or be on the same scale. In fact, we should probably watch out for cost approaching the same scale, as that could mean that constraint enforcement may not overpower an infeasible but very optimal trajectory
#
#         These differences should be taken into account when setting the values'''
#
#         self.whichAngCostFunc = 0
#         self.considerVectorInTVLQR = 0
#         self.useRawControlCost = True
#         #0 is (1-dot(ECIvec.T*rotMat(q).T*satvec))
#         #1 is 0.5*(1-dot(ECIvec.T*rotMat(q).T*satvec))^2
#         #2 is acos(dot(ECIvec.T*rotMat(q).T*satvec)
#         #3 is 0.5*acos(dot(ECIvec.T*rotMat(q).T*satvec)^2
#
#         self.quaternionTo3VecMode = 0  #0 is 2*qv*sign(q0)/(1+|q0|), 1 is 2*qv/(1+q0), 2 is qv/q0
#
#         self.mtq_control_weight = 0.0001
#         self.rw_control_weight = 0.001
#         self.magic_control_weight = 0.0001
#         self.rw_AM_weight = 0.1
#         self.rw_stic_weight = 0.01
#
#         self.angle_weight = 10#0.1#200.0
#         self.angvel_weight = 100#0*10.0#0.1#0.01#0*10.0#.01#01#0.000001
#         self.u_weight = 1#1e-1#1e-1 #0*0.0001#0.0#0.0000001
#         self.u_with_mag_weight = 0.0
#         self.av_with_mag_weight = 0#0.1
#         self.ang_av_weight = 0*10.0#0*100.0
#         self.angle_weight_N = 100#2000.0
#         self.angvel_weight_N = 100#0*10.0#0.1#0*10.0
#         self.av_with_mag_weight_N = 0.0
#         self.ang_av_weight_N = 0*100.0
#
#         self.angle_weight2 = 10#0.1#100.0
#         self.angvel_weight2 = 0.1#0*1.0#0.0
#         self.u_weight2 = 0.1#1e-2#1# 1e-1
#         self.u_with_mag_weight2 = 0.0
#         self.av_with_mag_weight2 = 0.0
#         self.ang_av_weight2 = 0*0.2#0*0.2#0.1#0.0001*0
#         self.angle_weight_N2 = 1000#1000.0
#         self.angvel_weight_N2 = 1.0#1.0#0*1.0#0.0
#         self.av_with_mag_weight_N2 = 0
#         self.ang_av_weight_N2 = 0*0.2#1.0#0*0.2
#
#         self.angle_weight_tvlqr = 10#30#0.1#10.0
#         self.angvel_weight_tvlqr = 100.0#0.1#0.01#0#0.001#0.01#0.01#0*0.01#1#$5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0.01#0.00001
#         self.u_weight_tvlqr = 0.01#1e-2#0.001#0.00001#0.001#0.00000001#0.3#5#0.1#0.000000001#0.0000000001
#         self.u_with_mag_weight_tvlqr = 0.0
#         self.av_with_mag_weight_tvlqr = 0#5.0#0#0.005#0.01
#         self.ang_av_weight_tvlqr = 1#0*2#0.05#0.01#0.5#0.5#0#10#0.01#0.1#0*0.1#0*10#1000#-0.1#0.0001#0.01
#         self.angle_weight_N_tvlqr = 100#1*500#1#500#200#30#20.0
#         self.angvel_weight_N_tvlqr = 100#$1.0*10#0.1*10#0.1#0.1#0#0.01#1#1#0*0.1#0*0.01#1#5#0.01#0.001#0.5#0.1#0#1#0*0.000001#1e-1#0*0.0000001#0*0.00001
#         self.av_with_mag_weight_N_tvlqr = 0
#         self.ang_av_weight_N_tvlqr = 1#0*2#5.0#0#0.05*100#0*0.01#0.01#0.5#0.5#0#10#0.01#0.1#0.01#0.0
#
#         self.sun_limit_angle = 20*3.14/180.0# 0.000000001#10*3.14/180.0 #RADIANS
#         self.camera_axis = np.array([[1,0,0]]).T
#
#         #Set forwardPassSettings
#         self.maxLsIter = 20#25#30#30#20#20#45
#         self.beta1 = 1e-10#1e-5#1e-2#1e-5#1e-10#1e-10#1e-10#1e-10#1e-10#1e-10#10#1e-10#1e-8#1e-2#1e-10#1e-4#1e-4#1e-4#2e-1
#         self.beta2 = 20#10#5#2reg0 #20#50#20#100#50#50#25#25#15#25#.030.0 #5.0
#
#         self.regScale = 1.6#1.6#1.6##2#1.6#4#1.6#2#1.6#1.8#2.0#1.8#5#10#1e3#1.8#2#1.6#1.6#2#1.6#2.0#1.6#3#1.6#5#2#5#1.6#5#1.8
#         self.regMax = 1e10
#         self.regMax2 = 1e12
#         self.regMin = 1e-10#16
#         self.regBump = 10#5#1e-3#1e-2#10.0#1e-1#2.0#10.0#1.0#10.0#1.0 #1.0#10.0#$0.01#0.1#2.0#1e-2#0.1#1.0#0.1#10#1e-10#2.0#1.0#1.0#1.0#10.0#100#0.1#10#1.0 #1.0#1.0#50#10#20.0#100.0
#         self.regMinCond = 1 #0 means that regmin is basically ignored, and regulaization goes up and down without bounds, case 1 means regularization is always equal to or greater than regmin, case 2 means if the regularization falls below regmin then it clamps to 0
#         self.regMinCond2 = 1
#         self.regBumpRandAddRatio = 0#1e-20#1e-16#1e-3#4e-3#*1e-4
#         self.useEVmagic = 0;#1 #use the eigendecomposition rather than simple regularization
#         self.SPDEVreg = 1;#0#1 #regularize/add even if matrix is SPD
#         self.SPDEVregAll = 0;#0 reg SPD matrix by adding rho*identity matrix (otherwise do the EV magic reg)
#         self.rhoEVregTest = 1;#1 #test if reset is needed (in EV magic case) by comparing to a multiple of rho (otherwise compare to regmin)
#         self.useDynamicsHess = 0 #1 #multiple of rho used in reset test
#         self.EVregTestpreabs = 1;#0 #complete the reset test before absolute value is taken
#         self.EVaddreg = 0;#0 #do EV magic by adding a value to the eigs that are too small (otherwise clamp to a minimum value)
#         self.EVregIsRho = 1; #1 #clamp to or add rho (otherwise regmin)
#         self.EVrhoAdd = 0;#0 #if adding a value to the eigs, this determines if the reg value added is added to the values less than rho (True) or regmin (false)
#         self.useConstraintHess = 1 #take the absolute value of the eigenvalues before testing and adding to them or clamping them (but after reset test)
#
#         self.control_limit_scale = 0.6
#         self.umax = self.control_limit_scale*np.vstack([np.array(sat.MTQ_max).reshape((sat.number_MTQ,1)),np.array(sat.RW_torq_max).reshape((sat.number_RW,1))])
#         #TODO: add RW saturation, other constraints.
#         self.xmax = 10*np.ones((sat.state_len,1))
#         self.eps = 2.22044604925031e-16
#         self.satAlignVector = [0,0,1]
#         self.wmax = 0.02#0.005 #rad/sec
#         #Set backwardPassSettings#
#         self.mu = 1.0
#         self.rho = 0.0#1e-10#0.01#1.0#0.1#1.0#0.1#0.001#1.0#0.01#1.0
#         self.drho = 1.0
#         #Set alilqrSettings
#         self.regInit = self.regMin#regMin#0#0.0#regMin#0.0#0.0#2*regMin
#
#         self.maxOuterIter = 25#70#50#20#20#25#25#1#15#diop
#         self.maxIlqrIter = 250#350#0#100#300#diop0#50#150#50#25#25#1#30
#         self.maxOuterIter2 = 14#70#50#20#20#25#25#1#15#diop
#         self.maxIlqrIter2 = 200#350#0#100#300#diop0#50#150#50#25#25#1#30
#
#         self.maxIter = 4500
#         self.maxIter2 = 3500
#         self.gradTol = 1e-7#1e-9#1e-09
#         self.costTol =      1e-9#1e-6#0.0000001
#         self.ilqrCostTol =  1e-8#1e-4#0.000001
#         self.maxCost = 1e10
#
#         self.cmax = 0.002
#         self.zCountLim = 20#14#20#10#30#10#45
#         self.penInit = 1# 1.0#0.1#1.0#1.0#10#1.0#1.0#40.0#1.0#0.1#0.1#1#0.01#0.01#10.0#0.1#1#2.5#1e2#5e3#5.0#5#0.5#1.0F
#         self.penInit2 = 1# 1.0#0.1#1.0#1.0#10#1.0#1.0#40.0#1.0#0.1#0.1#1#0.01#0.01#10.0#0.1#1#2.5#1e2#5e3#5.0#5#0.5#1.0F
#         self.penMax = 1e10
#         self.penScale = 10#5#10#10# 3#10#10#4#4#10#100#10#100.0#10.0
#
#         self.lagMultInit = 0.0
#         self.lagMultMax = 1e10
#         self.lagMultMax2 = 1e10
#
#
#         self.useACOSConstraint = 0
#         self.useExtraAVConstraint = 0
#
#         self.plan_for_aero = include_drag
#         self.plan_for_prop = include_prop
#         self.plan_for_srp = include_srp
#         self.plan_for_gg = include_gg
#         self.plan_for_gendist = include_gendist
#         self.plan_for_resdipole = include_resdipole
#         self.srp_coeff = np.zeros((3,))
#         self.drag_coeff = np.zeros((3,))
#         self.coeff_N = 0
#         self.res_dipole = np.array(sat.res_dipole).reshape((3,1))
#         self.prop_torque = np.array(sat.simple_prop_torq_val).reshape((3,1))
#         self.gendist_torq = np.array(sat.gendist_torq).reshape((3,1))
#         self.J_est = sat.J
#
#     def lineSearchSettings(self):
#         return (self.maxLsIter,self.beta1,self.beta2)
#     def auglagSettings(self):
#         return (self.lagMultInit,self.lagMultMax,self.penInit,self.penMax,self.penScale)
#     def breakSettings(self):
#         return (self.maxOuterIter,self.maxIlqrIter,self.maxIter,self.gradTol,self.ilqrCostTol,self.costTol,self.zCountLim,self.cmax,self.maxCost,self.xmax)
#     def regSettings(self):
#         return (self.regInit,self.regMin,self.regMax,self.regScale,self.regBump,self.regMinCond,self.regBumpRandAddRatio,self.useEVmagic,self.SPDEVreg,self.SPDEVregAll,self.rhoEVregTest,self.EVregTestpreabs,self.EVaddreg,self.EVregIsRho,self.EVrhoAdd,self.useDynamicsHess,self.useConstraintHess)
#
#     def lineSearchSettings2(self):
#         return (self.maxLsIter,self.beta1,self.beta2)
#     def auglagSettings2(self):
#         return (self.lagMultInit,self.lagMultMax2,self.penInit2,self.penMax,self.penScale)
#     def breakSettings2(self):
#         return (self.maxOuterIter2,self.maxIlqrIter2,self.maxIter2,self.gradTol,self.ilqrCostTol,self.costTol,self.zCountLim,self.cmax,self.maxCost,self.xmax)
#     def regSettings2(self):
#         return (self.regInit,self.regMin,self.regMax2,self.regScale,self.regBump,self.regMinCond2,0*self.regBumpRandAddRatio,0,self.SPDEVreg,self.SPDEVregAll,self.rhoEVregTest,self.EVregTestpreabs,self.EVaddreg,self.EVregIsRho,self.EVrhoAdd,self.useDynamicsHess,self.useConstraintHess)
#
#     def highSettings(self):
#         return (self.gyrogainH,self.dampgainH,self.velgainH,self.quatgainH,self.randvalH,self.umaxmultH)
#     def lowSettings(self):
#         return (self.gyrogainL,self.dampgainL,self.velgainL,self.quatgainL,self.randvalL,self.umaxmultL)
#
#     def systemSettings(self):
#         return (self.J_est,self.dt_tp,self.dt_tvlqr,self.eps,self.tvlqr_len,self.tvlqr_overlap)
#     def mainAlilqrSettings(self):
#         return (self.lineSearchSettings(),self.auglagSettings(),self.breakSettings(),self.regSettings())
#     def secondAlilqrSettings(self):
#         return (self.lineSearchSettings2(),self.auglagSettings2(),self.breakSettings2(),self.regSettings2())
#     def initTrajSettings(self):
#         return (self.bdotgain,self.HLangleLimit,self.highSettings(),self.lowSettings())
#     def optMainCostSettings(self):
#         return (self.angle_weight,self.angvel_weight,self.u_weight,self.u_with_mag_weight,self.av_with_mag_weight,self.ang_av_weight,self.angle_weight_N,self.angvel_weight_N,self.av_with_mag_weight_N,self.ang_av_weight_N,self.whichAngCostFunc,self.useRawControlCost)
#     def optSecondCostSettings(self):
#         return (self.angle_weight2,self.angvel_weight2,self.u_weight2,self.u_with_mag_weight2,self.av_with_mag_weight2,self.ang_av_weight2,self.angle_weight_N2,self.angvel_weight_N2,self.av_with_mag_weight_N2,self.ang_av_weight_N2,self.whichAngCostFunc,self.useRawControlCost)
#     def optTVLQRCostSettings(self,tracking_LQR_formulation):
#         return (self.angle_weight_tvlqr,self.angvel_weight_tvlqr,self.u_weight_tvlqr,self.u_with_mag_weight_tvlqr,self.av_with_mag_weight_tvlqr,self.ang_av_weight_tvlqr,self.angle_weight_N_tvlqr,self.angvel_weight_N_tvlqr,self.av_with_mag_weight_N_tvlqr,self.ang_av_weight_N_tvlqr,self.considerVectorInTVLQR,self.useRawControlCost,tracking_LQR_formulation)
#     def planner_disturbance_settings(self):
#         return ((self.plan_for_aero,self.plan_for_prop,self.plan_for_srp,self.plan_for_gg,self.plan_for_resdipole,self.plan_for_gendist),self.srp_coeff,self.drag_coeff,self.coeff_N,self.prop_torque,self.gendist_torq,self.res_dipole)

#
# class Goals:
#     def __init__(self,control_mode_dict=None,pointing_dict=None,sat_vec_dict=None,default_mode = None,default_point_mode = None, default_point_vec = None, default_sat_vec = None):
#         # print(control_mode_dict)
#         if control_mode_dict is None:
#             control_mode_dict = {}
#         if pointing_dict is None:
#             pointing_dict = {}
#         if sat_vec_dict is None:
#             sat_vec_dict = {}
#         self.control_modes = copy.deepcopy(dict(control_mode_dict))
#         self.point_goal = copy.deepcopy(dict(pointing_dict))
#         self.sat_vecs = copy.deepcopy(dict(sat_vec_dict))
#         if default_mode is None:
#             self.default_control_mode = GovernorMode.SIMPLE_BDOT
#         else:
#             self.default_control_mode = default_mode
#
#         if default_point_mode is None:
#             self.default_pointing_goal_mode = PointingGoalVectorMode.NADIR
#         else:
#             self.default_pointing_goal_mode = default_point_mode
#
#         if default_point_vec is None:
#             self.default_pointing_goal_vector = np.array([[0, 0, -1]]).T
#         else:
#             self.default_pointing_goal_vector = default_point_vec
#
#         if default_mode is None:
#             self.default_satellite_pointing_vector = np.array([[1,0,0]]).T
#         else:
#             self.default_satellite_pointing_vector = default_sat_vec

    # def get_pointing_info(self,orbit,for_TP=False,quatmode = 0):
    #     if isinstance(orbit,Orbital_State):
    #         t = orbit.J2000
    #         key = sorted(self.point_goal.keys())
    #         key_array = np.array(key)
    #         times_list = np.where(t>=key_array)[0]
    #         if len(times_list) == 0:
    #             mode = self.default_pointing_goal_mode
    #             vec = self.default_pointing_goal_vector
    #         else:
    #             tmp = self.point_goal[key[times_list[-1]]]
    #             # breakpoint()
    #             if isinstance(tmp,PointingGoalVectorMode):
    #                 mode = tmp
    #                 vec = np.zeros((3,1))
    #             else:
    #                 (mode,vec) = tmp
    #         pg = pointing_goal_vec_finder_times(mode, vec, orbit.J2000, orbit.R, orbit.V, orbit.S,quatmode)
    #
    #         key = sorted(self.sat_vecs.keys())
    #         key_array = np.array(key)
    #         times_list = np.where(t>=key_array)[0]
    #         if len(times_list) == 0:
    #             sv = self.default_satellite_pointing_vector
    #         else:
    #             sv = self.sat_vecs[key[times_list[-1]]]
    #
    #         return pg,sv
    #     elif isinstance(orbit,Orbit):
    #         if not for_TP:
    #             results = [(self.get_pointing_info(orbit.states[j],quatmode=quatmode)) for j in orbit.times]
    #             pg = [j[0] for j in results]
    #             sv = [j[1] for j in results]
    #         else:
    #             key = sorted(self.point_goal.keys())
    #             key_array = np.array(key)
    #             orbkeys = orbit.times
    #             times_list_list = [np.where(orbit.states[j].J2000>=key_array)[0] for j in orbkeys]
    #             point_goals_list = [self.point_goal[key[j[-1]]] if len(j)>0 else (self.default_pointing_goal_mode,self.default_pointing_goal_vector) for j in times_list_list]
    #             modes = [j[0] for j in point_goals_list]
    #             vecs = [j[1] for j in point_goals_list]
    #
    #             eci_vecs = [pointing_goal_vec_finder_times(modes[j], vecs[j], orbit.states[orbkeys[j]].J2000, orbit.states[orbkeys[j]].R, orbit.states[orbkeys[j]].V, orbit.states[orbkeys[j]].S) for j in range(len(times_list_list))]
    #             full_orientation_commands = [j == PointingGoalVectorMode.PROVIDED_MRP for j in modes]
    #             if any(full_orientation_commands):
    #                 pg = [mrp_to_quat(eci_vecs[j]) if full_orientation_commands[j] else np.vstack([np.nan*np.ones((1,1)),eci_vecs[j]]) for j in range(len(eci_vecs))]
    #
    #             else:
    #                 pg = eci_vecs
    #             sv = [(self.get_pointing_info(orbit.states[j],quatmode=quatmode))[1] for j in orbit.states.keys()]
    #             # breakpoint()
    #         # t = orbit.times
    #         # pg_res = [self.point_goal[np.where(np.ndarray.item(j)>=sorted(self.point_goal.keys()))[0][-1]] for j in t]
    #         # sv = [self.sat_vecs[np.where(j>=sorted(self.sat_vecs.keys()))[0][-1]] for j in t]
    #         # pg = [pointing_goal_vec_finder_times(pg_res[j][0], pg_res[j][1], orbit.states[orbit.times[j]].J2000, orbit.states[orbit.times[j]].R, orbit.states[orbit.times[j]].V, orbit.states[orbit.times[j]].S) for j in range(len(orbit.times))]
    #         return pg,sv
    #     else:
    #         raise ValueError("must be orbit or orbital state")

    # def get_control_mode(self,t):
    #     if isinstance(t,Orbital_State):
    #         t = t.J2000
    #     key = sorted(self.control_modes.keys())
    #     key_array = np.array(key)
    #     times_list = np.where(t>=key_array)[0]
    #     # print(times_list)
    #     # print(np.where(t>=key_array))
    #     # print(self.control_modes)
    #     if len(times_list) == 0:
    #         return self.default_control_mode
    #     else:
    #         return self.control_modes[key[times_list[-1]]]


    # def get_next_goal_change_time(self,t):
    #     #returns Nan if no goal in future
    #     if isinstance(t,Orbital_State):
    #         t = t.J2000
    #     key = sorted(self.control_modes.keys())
    #     key_array = np.array(key)
    #     times_list = np.where(t<key_array)[0]
    #     # print(times_list)
    #     # print(np.where(t>=key_array))
    #     # print(self.control_modes)
    #     if len(times_list) == 0:
    #         return np.nan
    #     else:
    #         return key[times_list[0]]


#
# class ADCS:
#     def __init__(self, estimator, orbital_estimator, controller,goals = None,planner = None,planner_settings = None, use_plan = False,prop_schedule = None):
#         """
#         Initialize the governor.
#         """
#         self.estimator = estimator
#         self.orbital_estimator = orbital_estimator
#         controller.reset_sat(estimator.sat,controller.control_wt,controller.control_diff_from_plan_wt)
#         self.controller = controller
#         if goals is None:
#             self.goals = Goals()
#         elif isinstance(goals,Goals):
#             self.goals = goals
#         elif isinstance(goals,list):
#             self.goals = Goals(*list)
#         self.use_plan = use_plan
#         if planner_settings is None:
#             planner_settings = PlannerSettings(self.estimator.sat,self.controller.update_period)
#         self.planner_settings = planner_settings
#         if planner is None and self.use_plan:
#             self.update_planner_settings()
#             self.planner.setVerbosity(True)
#         if planner is not None:
#             planner.setVerbosity(True)
#             self.planner = planner
#
#         self.prop_dict = {}
#         if prop_schedule is None:
#             prop_schedule = ([],[])
#             self.update_prop_dict(prop_schedule)
#         self.prop = False
#         # self.prop_torque = np.zeros((3,1))
#         # self.prop_torque_saved = self.prop_torque
#         # self.reinit_Ekf_prop_cov = False
#         self.current_trajectory = Trajectory()
#         self.next_trajectory = Trajectory()
#         self.J2000 = 0
#         self.mode = np.nan
#
#         # self.current_trajectory_state = np.nan*np.ones((self.estimator.sat.state_len,1))
#         # self.next_trajectory_state = np.nan*np.ones((self.estimator.sat.state_len,1))
#         # self.current_trajectory_control = np.nan*np.ones((self.estimator.sat.control_len,1))
#         # self.next_trajectory_control = np.nan*np.ones((self.estimator.sat.control_len,1))
#         self.prec_ready = False
#         self.estimation = np.nan*np.ones((self.estimator.sat.state_len,1))
#         self.estimation_prev = np.nan*np.ones((self.estimator.sat.state_len,1))
#         self.orbit_estimation_prev = Orbital_State(0,np.array([[0,0,1]]).T,np.zeros((3,1)))#np.nan*np.ones((6,1))
#         self.orbit_estimation = Orbital_State(0,np.array([[0,0,1]]).T,np.zeros((3,1)))#np.nan*np.ones((6,1))
#         self.planned_state = np.nan*np.ones((self.estimator.sat.state_len,1))
#         self.planned_next_state = np.nan*np.ones((self.estimator.sat.state_len,1))
#
#         self.planned_control = np.nan*np.ones((self.estimator.sat.control_len,1))
#         if self.controller.tracking_LQR_formulation==1:
#             self.planned_gain = np.nan*np.ones((self.estimator.sat.control_len,self.estimator.sat.state_len+1))
#             self.planned_ctg = np.nan*np.ones((self.estimator.sat.state_len+1,self.estimator.sat.state_len+1))
#             self.planned_next_gain = np.nan*np.ones((self.estimator.sat.control_len,self.estimator.sat.state_len+1))
#             self.planned_next_ctg = np.nan*np.ones((self.estimator.sat.state_len+1,self.estimator.sat.state_len+1))
#         elif self.controller.tracking_LQR_formulation==0:
#             self.planned_gain = np.nan*np.ones((self.estimator.sat.control_len,self.estimator.sat.state_len-1))
#             self.planned_ctg = np.nan*np.ones((self.estimator.sat.state_len-1,self.estimator.sat.state_len-1))
#             self.planned_next_gain = np.nan*np.ones((self.estimator.sat.control_len,self.estimator.sat.state_len-1))
#             self.planned_next_ctg = np.nan*np.ones((self.estimator.sat.state_len-1,self.estimator.sat.state_len-1))
#         elif self.controller.tracking_LQR_formulation==2:
#             self.planned_gain = np.nan*np.ones((self.estimator.sat.control_len,self.estimator.sat.state_len+2))
#             self.planned_ctg = np.nan*np.ones((self.estimator.sat.state_len+2,self.estimator.sat.state_len+2))
#             self.planned_next_gain = np.nan*np.ones((self.estimator.sat.control_len,self.estimator.sat.state_len+2))
#             self.planned_next_ctg = np.nan*np.ones((self.estimator.sat.state_len+2,self.estimator.sat.state_len+2))
#         self.point_goal = np.nan*np.ones((3,1))
#         self.sat_vec = np.nan*np.ones((3,1))
#         self.B_body = np.nan*np.ones((3,1))
#         self.dB_body = np.nan*np.ones((3,1))
#         self.command = np.zeros((self.estimator.sat.control_len,1))
#         self.time_step =  gcd_sample_time([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr, self.estimator.update_period,self.controller.update_period,self.orbital_estimator.update_period])
#         self.addl_info = {}

    # def update_prop_dict(self,prop_schedule):
        #
        # prop_on_times = np.copy(prop_schedule[0])
        # prop_off_times = np.copy(prop_schedule[1])
        # if any([j in prop_off_times for j in prop_on_times]):
        #     raise ValueError("Prop cannot be turned on and off at the same timepoint.")
        # if len(prop_on_times)>0 or len(prop_off_times)>0:
        #     prop_dict = {j : 1 for j in prop_on_times}.update({j : -1 for j in prop_off_times})
        #
        #
        #     if min(prop_dict.keys())<self.J2000:
        #         raise ValueError("schedule includes times that have already passed")
        #
        #
        #     dict_old = self.prop_dict
        #     dict_old_cut = {k:v for k,v in dict_old.items() if k > self.J2000}
        #
        #     prop_dict = {k:v for k,v in prop_dict.items() if k not in dict_old_cut.keys()} #exclude repeats
        #     combined_dict = {**prop_dict,**dict_old_cut}
        #     times = sorted(combined_dict.keys())
        #     combined_dict = {k:combined_dict[k] for k in times} #make sure keys and values are sorted
        #
        #     prop_switches = [combined_dict[j] for j in times]
        #     prop_status = np.cumsum(np.array(prop_switches))
        #     prop_status = prop_status.flatten().tolist()
        #     if not (np.all(np.logical_or(prop_status==0,prop_status==1)) or (np.all(np.logical_or(prop_status==0,prop_status==-1)))):
        #         raise ValueError("There are multiple instances of turning the prop on (or off) in a row.")
        #     if min(prop_status) == -1 and not self.prop:
        #         raise ValueError("The prop schedule and current prop status indicates that an already-off propulsion will be turned off.")
        #     if min(prop_status) == 0 and self.prop:
        #         raise ValueError("The prop schedule and current prop status indicates that an already-on propulsion will be turned on.")
        #     self.prop_dict = combined_dict


    # def orbit_est_update(self,sense,orbital_truth):
    #     sense = np.copy(sense)
    #     if orbital_truth is not None:
    #         orbital_truth = orbital_truth.copy()
    #     GPS_sensor = [j for j in self.estimator.sat.sensors if isinstance(j,GPS)][0] #TODO: currently assumes just one GPS sensor
    #     GPS_list = sense[GPS_sensor.sensor_output_range[0]:GPS_sensor.sensor_output_range[1],:]
    #     orbit_estimation = self.orbital_estimator.update(GPS_list,self.J2000,orbital_truth)
    #
    #     # orbit_estimation.B = 1e-4*np.array([[0,1,0]]).T
    #     self.orbit_estimation_prev = self.orbit_estimation.copy()
    #     self.orbit_estimation = orbit_estimation

    # def est_update(self,sense,state_truth):
    #     sense = np.copy(sense)
    #     if state_truth is not None:
    #         state_truth = np.copy(state_truth)
    #     in_eclipse = self.orbit_estimation.in_eclipse()
    #     what_sensors = [True if not (isinstance(j,GPS) or (isinstance(j,SunSensor) and in_eclipse)) else False for j in self.estimator.sat.sensors]
    #     # print(self.command)
    #     estimation = self.estimator.update(sense,self.command,self.orbit_estimation,what_sensors,self.prop,truth = state_truth)
    #     # if est_prop:
    #     #     self.prop_torque = self.EkfModule.get_prop_torq()
    #     self.estimation_prev = np.copy(self.estimation)
    #     self.estimation = estimation
    #     mtm3 = [j for j in self.estimator.sat.sensors if isinstance(j,ThreeAxisMTM)]
    #     if len(mtm3) == 0:
    #         self.dB_body = np.nan*np.ones((3,1))
    #     elif len(mtm3) == 1:
    #         mtm3a = mtm3[0]
    #         bnew = sense[mtm3a.sensor_output_range[0]:mtm3a.sensor_output_range[1],:].reshape((3,1))
    #         if not np.any(np.isnan(self.B_body)):
    #             self.dB_body = (bnew - self.B_body.reshape((3,1)))/self.estimator.update_period
    #         else:
    #             self.dB_body = np.zeros((3,1))
    #         self.B_body = bnew
    #     else: #if more than one MTM, combine them? TODO: improve this or simplify it
    #         bnew = np.mean([sense[j.sensor_output_range[0]:j.sensor_output_range[1],:].reshape((3,1)) for j in mtm3])
    #         if not np.any(np.isnan(self.B_body)):
    #             self.dB_body = (bnew - self.B_body.reshape((3,1)))/self.estimator.update_period
    #         else:
    #             self.dB_body = np.zeros((3,1))
    #         self.B_body = bnew

    # def controller_update(self):
    #     self.point_goal,self.sat_vec = self.goals.get_pointing_info(self.orbit_estimation,quatmode = self.controller.quaternionTo3VecMode)
    #
    #     ctrl_mode = self.goals.get_control_mode(self.J2000)
    #     self.mode = ctrl_mode
    #     # print(self.orbit_estimation)
    #     osp1 = self.orbit_estimation.orbit_rk4(self.controller.update_period,calc_env_vecs = False)
    #
    #     if ctrl_mode in [GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.PLAN_OPEN_LOOP]:
    #         ps = self.current_trajectory.state_nearest_to_time(self.J2000)
    #         pc = self.current_trajectory.control_nearest_to_time(self.J2000)
    #         pg = self.current_trajectory.gain_nearest_to_time(self.J2000)
    #         pm = self.current_trajectory.ctg_nearest_to_time(self.J2000)
    #
    #         ctrl_goal = (ps,pc,pg,pm,self.point_goal,self.sat_vec)
    #         self.planned_state = ps
    #         self.planned_control = pc
    #         self.planned_gain = pg
    #         self.planned_ctg = pm
    #         # print(self.J2000,osp1.J2000,self.orbit_estimation.J2000, cent2sec*(osp1.J2000-self.J2000), cent2sec*(self.orbit_estimation.J2000-self.J2000))
    #         psp1 = self.current_trajectory.state_nearest_to_time(osp1.J2000)
    #         pcp1 = self.current_trajectory.control_nearest_to_time(osp1.J2000)
    #         pgp1 = self.current_trajectory.gain_nearest_to_time(osp1.J2000)
    #         pmp1 = self.current_trajectory.ctg_nearest_to_time(osp1.J2000)
    #         # breakpoint()
    #         self.planned_next_state = psp1
    #         self.planned_next_ctg = pmp1
    #         self.planned_next_gain = pgp1
    #         goalp1, svp1 = self.goals.get_pointing_info(osp1,quatmode = self.controller.quaternionTo3VecMode)
    #         ctrl_goalp1 = (psp1,pcp1,pgp1,pmp1,goalp1, svp1)
    #     else:
    #         goalp1, svp1 = self.goals.get_pointing_info(osp1,quatmode = self.controller.quaternionTo3VecMode)
    #         ctrl_goal = ([],[],[],[],self.point_goal,self.sat_vec)
    #         ctrl_goalp1 = ([],[],[],[],goalp1,svp1)
    #         if ctrl_mode in QuaternionModeList:
    #             self.planned_state = np.vstack([np.zeros((3,1)),vec3_to_quat(ctrl_goal[4],self.controller.quaternionTo3VecMode)])
    #             self.planned_next_state = np.vstack([np.zeros((3,1)),vec3_to_quat(goalp1,self.controller.quaternionTo3VecMode)])
    #
    #     new_command,addl_info = self.controller.update(ctrl_mode,ctrl_goal,ctrl_goalp1,self.estimation,self.orbit_estimation,osp1,prop_on = self.prop,dB_body = self.dB_body,fakeTF=False)
    #     self.prev_command = self.command
    #     self.command = new_command
    #     self.addl_info["controller"] = addl_info
    #
    # def update(self,t,sense,state_truth = None,orbital_truth = None):
    #     self.addl_info.clear()
    #     self.J2000 = t
    #     sense = np.copy(sense)
    #     if state_truth is not None:
    #         state_truth = np.copy(state_truth)
    #     if orbital_truth is not None:
    #         orbital_truth = orbital_truth.copy()
    #     # print(t,self.orbital_estimator.prev_os.J2000,self.orbital_estimator.update_period)
    #     if (t-self.orbital_estimator.current_state_estimate.J2000)*cent2sec>self.orbital_estimator.update_period - time_eps:
    #         self.orbit_est_update(sense,orbital_truth)
    #
    #     if (t-self.estimator.prev_os.J2000)*cent2sec>self.estimator.update_period - time_eps:
    #         # print(self.orbit_estimation.R)
    #         self.est_update(sense,state_truth)
    #
    #     if self.use_plan:
    #         if self.prec_ready:
    #             print('testing switch')
    #             if t >= (self.next_trajectory.min_time() - time_eps/cent2sec): #switch to precalculated traj?
    #                 print('should switch??')
    #                 # breakpoint()
    #                 self.switch_to_precalculated_trajectory(True, self.estimation) #True for verbose
    #                 # print("Bstate: ",self.orbit_estimation.B.T)
    #         # print('checking precalc need')
    #         prec_need,x_prec = self.check_for_precalculate_and_do()
    #         if prec_need:
    #             # print('need to precalc')
    #             # x_prec = self.precalculate()
    #             print('precalc done')
    #             # print("================")
    #             # for j in range(len(self.prec_t)):
    #             #     print(int(cent2sec*(self.prec_t[j]-0.22)),np.round(self.prec_X[j].T,3),np.round(self.prec_U[j].T,2))
    #             # print("================")
    #
    #
    #             # print("B precalc: ",np.vstack([self.tp_orbit.states[self.tp_orbit.times[j]].B.T for j in range(10)]))
    #             # print([(-0.22+self.tp_orbit.times[j])*cent2sec for j in range(10)])
    #             self.planner_settings.res_dipole = np.array(self.estimator.sat.res_dipole).reshape((3,1))
    #             self.planner_settings.prop_torque = np.array(self.estimator.sat.simple_prop_torq_val).reshape((3,1))
    #             self.planner_settings.gendist_torq = np.array(self.estimator.sat.gendist_torq).reshape((3,1))
    #             self.update_planner_settings()
    #             lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr = self.start_trajectory_planning(x_prec)
    #             Uset_lqr[self.estimator.sat.number_MTQ:,:] *= self.csat.read_magrw_torq_mult()
    #             # print(lqr_times-0.22)
    #             # print(self.tp_sat_vec)
    #             planned_orbit = self.tp_orbit.new_orbit_from_times(lqr_times)
    #             pg,sv = self.goals.get_pointing_info(planned_orbit,quatmode = self.controller.quaternionTo3VecMode)
    #             sv = np.hstack([j.reshape((3,1)) for j in sv])
    #             print(sv.shape)
    #             pg = np.hstack([j.reshape((3,1)) for j in pg])
    #             pointing = rot_mat_list(Xset_lqr[3:7,:],sv,transpose = False)
    #             # pointing_goal = np.hstack([j.reshape((3,1)) for j in pg])
    #             print(pg)
    #             # print(pg - -1*matrix_row_normalize(np.squeeze(np.array(planned_orbit.get_vecs()[0]))).T)
    #             print([(180.0/math.pi)*math.acos(np.clip(pointing[:,j].reshape((3,1)).T@pg[:,j].reshape((3,1)),-1,1)) for j in range(lqr_times.size)])
    #             print([(180.0/math.pi)*math.acos(np.clip(2.0*((Xset_lqr[3:7,j].T@mrp_to_quat(pg[:,j].reshape((3,1))).reshape((4,1))).item())**2.0 - 1,-1,1)) for j in range(lqr_times.size)])
    #
    #             Kmult = np.eye(self.estimator.sat.control_len)
    #             Kmult[self.estimator.sat.number_MTQ:,:] *= self.csat.read_magrw_torq_mult()
    #             if self.controller.tracking_LQR_formulation == 1:
    #                 Kset_lqr = [Kmult@Kset_lqr[:,j].reshape((self.estimator.sat.control_len,self.estimator.sat.state_len+1)) for j in range(Kset_lqr.shape[1])]
    #                 Sset_lqr = [Sset_lqr[:,j].reshape((self.estimator.sat.state_len+1,self.estimator.sat.state_len+1)) for j in range(Sset_lqr.shape[1])]
    #             elif self.controller.tracking_LQR_formulation == 0:
    #                 Kset_lqr = [Kmult@Kset_lqr[:,j].reshape((self.estimator.sat.control_len,self.estimator.sat.state_len-1)) for j in range(Kset_lqr.shape[1])]
    #                 Sset_lqr = [Sset_lqr[:,j].reshape((self.estimator.sat.state_len-1,self.estimator.sat.state_len-1)) for j in range(Sset_lqr.shape[1])]
    #             elif self.controller.tracking_LQR_formulation == 2:
    #                 Kset_lqr = [Kmult@Kset_lqr[:,j].reshape((self.estimator.sat.control_len,self.estimator.sat.state_len+2)) for j in range(Kset_lqr.shape[1])]
    #                 Sset_lqr = [Sset_lqr[:,j].reshape((self.estimator.sat.state_len+2,self.estimator.sat.state_len+2)) for j in range(Sset_lqr.shape[1])]
    #             else:
    #                 raise ValueError("wrong value for lqr tracking formulation option")
    #             # print(Kset_lqr[0])
    #             print(matrix_row_norm(Xset_lqr[0:3,:].T).T)
    #             print("K: ",Kset_lqr[-1])
    #             print("S: ",Sset_lqr[-1])
    #             print("S: ",Sset_lqr[-2])
    #             # breakpoint()
    #
    #             self.next_trajectory = Trajectory(lqr_times,Xset_lqr,Uset_lqr,Kset_lqr,Sset_lqr)
    #             self.prec_ready = True #TODO should this be in a thread????
    #
    #     if (t-self.controller.prev_os.J2000)*cent2sec>(self.controller.update_period - time_eps):
    #         self.controller_update()
    #     return self.command,self.addl_info
    #
    # def update_planner_settings(self):
    #     csat = pysat.Satellite()
    #     csat.change_Jcom(self.estimator.sat.J)
    #     for k in range(self.estimator.sat.number_MTQ):
    #         csat.add_MTQ(self.estimator.sat.MTQ_axes[k],self.planner_settings.control_limit_scale*self.estimator.sat.MTQ_max[k],self.planner_settings.mtq_control_weight)
    #     for k in range(self.estimator.sat.number_RW):
    #         csat.add_RW(self.estimator.sat.RW_z_axis[k],self.estimator.sat.RW_J[k],self.planner_settings.control_limit_scale*self.estimator.sat.RW_torq_max[k],0.8*self.estimator.sat.RW_saturation[k],self.planner_settings.rw_control_weight,self.planner_settings.rw_AM_weight,0.05*self.estimator.sat.RW_saturation[k],self.planner_settings.rw_stic_weight,0.4*self.estimator.sat.RW_saturation[k])
    #     for k in range(self.estimator.sat.number_magic):
    #         csat.add_magic(self.estimator.sat.magic_axes[k],self.planner_settings.control_limit_scale*self.estimator.sat.magic_max[k],self.planner_settings.magic_control_weight)
    #     if self.planner_settings.wmax>0:
    #         csat.set_AV_constraint(self.planner_settings.wmax)
    #     if self.planner_settings.sun_limit_angle>0:
    #         csat.add_sunpoint_constraint(self.planner_settings.camera_axis,self.planner_settings.sun_limit_angle,0)
    #     if self.planner_settings.plan_for_gg:
    #         csat.add_gg_torq()
    #     if self.planner_settings.plan_for_aero:
    #         csat.add_aero_torq(self.planner_settings.drag_coeff,self.planner_settings.coeff_N)
    #     if self.planner_settings.plan_for_srp:
    #         csat.add_srp_torq(self.planner_settings.srp_coeff,self.planner_settings.coeff_N)
    #     if self.planner_settings.plan_for_resdipole:
    #         csat.add_resdipole_torq(self.planner_settings.res_dipole)
    #     if self.planner_settings.plan_for_prop:
    #         csat.add_prop_torq(self.planner_settings.prop_torque)
    #         # breakpoint()
    #     if self.planner_settings.plan_for_gendist:
    #         csat.add_gendist_torq(self.planner_settings.gendist_torq)
    #
    #     planner = tplaunch.Planner(csat,self.planner_settings.systemSettings(),
    #                                 self.planner_settings.mainAlilqrSettings(),
    #                                 self.planner_settings.secondAlilqrSettings(),
    #                                 self.planner_settings.initTrajSettings(),
    #                                 self.planner_settings.optMainCostSettings(),
    #                                 self.planner_settings.optSecondCostSettings(),
    #                                 self.planner_settings.optTVLQRCostSettings(self.controller.tracking_LQR_formulation))
    #     planner.setquaternionTo3VecMode(self.planner_settings.quaternionTo3VecMode)
    #     self.planner = planner
    #     self.csat = csat
    #     self.planner.setVerbosity(True)

    # def prop_off(self):
    #     """
    #     This function turns propulsion off for the TP, attitude EKF, and controller.
    #     """
    #     self.prop = False
    #
    # def prop_on(self):
    #     """
    #     This function turns propulsion on for the TP, attitude EKF, and controller.
    #
    #     Note that if plan_prop_on is still False, the TP won't plan with prop. This just *enables*
    #     modules to use propulsion, but doesn't force them to.
    #     """
    # #     self.prop = True
    #
    # def precalculate_for_trajectory_planner(self, t_start_planning, t_end_planning):
    #     """
    #     This function precalculates r_ECI, v_ECI, B_ECI, pointing_goal, and x_est over the timespan
    #     t_start_planning to t_end_planning, based on the current state estimate and pointing goal mode.
    #
    #     Parameters
    #     ---------------
    #         x_est: 7 x 1 np array, represents current estimated state
    #         rv_est: 6 x 1 np array, represents current estimate of r_ECI and v_ECI from orbit Ekf
    #         t_current: tuple, of format (decimal_hours, day, month, year), current time at start of precalculation
    #         t_start_planning: tuple, of format (decimal_hours, day, month, year), traj start time
    #         t_end_planning: tuple, of format (decimal_hours, day, month, year), traj end time
    #
    #     Returns
    #     --------------
    #         point_ECI: 8 x N np array, array of ECI pointing goal vectors to feed into TP (w/ times)
    #         sat_vec: 8 x N np array, array of satellite body pointing vecs to feed into TP (w/ times)
    #         B_ECI: 8 x N np array, array of magfield vecs to feed into TP (w/ times)
    #         r_ECI: 8 x N np array, array of orbital position vecs to feed into TP (w/ times)
    #         v_ECI: 8 x N np array, array of orbital velocity vecs to feed into TP (w/ times)
    #         sun_ECI: 8 x N np array, array of sun vecs rel. to s/c pos to feed into TP (w/ times)
    #         x_trajstart: 7 x 1 np array, [w, q]^T est. state at start time of trajectory
    #     """
    #     print('starting to precaculate')
    #     #We need to find an appropriate dt for r_ECI, v_ECI, which is the largest common sample time of
    #     #the TP sample time, TVLQR sample time, and controller sample time (for precalculation)
    #     #TODO: give precalculation its own sample time?
    #     gcd_dt_precalc = gcd_sample_time([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr, self.estimator.update_period,self.controller.update_period,self.orbital_estimator.update_period])
    #     max_dt = max([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr, self.estimator.update_period,self.controller.update_period,self.orbital_estimator.update_period])
    #     #j2000_current = J2000calc(t_current[0], t_current[1], t_current[2], t_current[3])
    #     #j2000_start_planning = J2000calc(t_start_planning[0], t_start_planning[1], t_start_planning[2], t_start_planning[3])
    #     #j2000_end_planning = J2000calc(t_end_planning[0], t_end_planning[1], t_end_planning[2], t_end_planning[3])
    #
    #     j2000_start_precalc = self.J2000#j2000_update(j2000_current, -1*max_dt)
    #     j2000_end_precalc = t_end_planning + 1.0*max_dt/cent2sec
    #     # print(t_end_planning)
    #
    #     # print((t_end_planning-self.J2000)*cent2sec,gcd_dt_precalc)
    #     orbit_prec = Orbit(self.orbit_estimation_prev,j2000_end_precalc,gcd_dt_precalc)
    #     self.orbit_projected = orbit_prec
    #
    #     gcd_dt_tp = gcd_sample_time([self.planner_settings.dt_tp, self.planner_settings.dt_tvlqr])
    #     tp_orbit = orbit_prec.get_range(t_start_planning,t_end_planning,dt = gcd_dt_tp)
    #     point_ECI,sat_vec = self.goals.get_pointing_info(tp_orbit,for_TP=True,quatmode = self.controller.quaternionTo3VecMode)
    #     self.tp_orbit = tp_orbit
    #     self.tp_point_goal = point_ECI
    #     self.tp_sat_vec = sat_vec
    #
    #     prev_t = self.orbit_estimation_prev.J2000 #assumes no prop switch in the time between the previous estimation and now.
    #     prop_prec_t,prop_prec_p = self.propulsion_finder(orbit_prec,prev_t,self.prop)
    #     ind_before_tp = np.where(np.array(prop_prec_t)<=t_start_planning)[0][-1]
    #     t_before_tp = prop_prec_t[ind_before_tp]
    #     p_before_tp = prop_prec_p[ind_before_tp]
    #     prop_tp_t,prop_tp_p = self.propulsion_finder(tp_orbit,t_before_tp,p_before_tp)
    #
    #     self.tp_prop = prop_tp_p
    #
    #     # print('orbit generated')
    #     # self.prop_projected = prop_prec
    #
    #     #Next, calculate x_trajstart
    #     #t = t_current
    #     x_prec = self.estimation
    #     x_prec_prev = self.estimation_prev
    #     estimated = False
    #     self.prec_t = []
    #     self.prec_X = []
    #     self.prec_U = []
    #     #Update x_prec if we are currently running a trajectory
    #     if not self.current_trajectory.is_empty():
    #         if self.current_trajectory.time_in_span(t_start_planning):
    #             #t_start_planning is in current trajectory
    #             x_prec = self.current_trajectory.state_nearest_to_time(t_start_planning)
    #             print('using best point in current trajectory')
    #             estimated = True
    #         elif (self.prec_ready and self.next_trajectory.time_in_span(t_start_planning)):
    #             #t_start_planning is in the next planned trajectory??? Not sure this state can arise
    #             x_prec = self.next_trajectory.state_nearest_to_time(t_start_planning)
    #             estimated = True
    #
    #     if not estimated:
    #         #TODO: could error if it searches for a future trajectory that doesn't exist?? This could happen if precalc was longer than trajectories
    #         #Want to go from j2000t = j2000_current to j2000_start_planning - 1
    #         t = j2000_start_precalc
    #         while t<t_start_planning:
    #             ctrl_mode = self.goals.get_control_mode(t)
    #             # breakpoint()
    #             if ctrl_mode in [GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.PLAN_OPEN_LOOP] and self.current_trajectory.time_in_span(t):
    #                 #go to end of that trajectory and run whatever comes next.
    #                 x_prec = self.current_trajectory.last_state()
    #                 x_prec_prev = self.current_trajectory.penultimate_state()
    #                 tmax = self.current_trajectory.max_time()
    #                 if (tmax-t_start_planning)*cent2sec > -time_eps:
    #                     t = t_start_planning
    #                     x_prec = self.current_trajectory.state_nearest_to_time(t)
    #                     x_prec_prev = self.current_trajectory.state_nearest_to_time(t-self.time_step/cent2sec)
    #                 else:
    #                     t = tmax
    #
    #             elif ctrl_mode in [GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.PLAN_OPEN_LOOP] and self.prec_ready and self.next_trajectory.time_in_span(t):
    #                 #go to end of that trajectory and run whatever comes next.
    #                 x_prec = self.next_trajectory.last_state()
    #                 x_prec_prev = self.next_trajectory.penultimate_state()
    #                 tmax = self.next_trajectory.max_time()
    #                 if (tmax-t_start_planning)*cent2sec > -time_eps:
    #                     t = t_start_planning
    #                     x_prec = self.next_trajectory.state_nearest_to_time(t)
    #                     x_prec_prev = self.next_trajectory.state_nearest_to_time(t-self.time_step/cent2sec)
    #                 else:
    #                     t = tmax
    #             else:
    #                 #If in plan and track but traj doesn't exist, do bdot
    #                 if ctrl_mode in [GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.PLAN_OPEN_LOOP]:
    #                     if self.estimator.sat.control_len>self.sat.number_MTQ:
    #                         ctrl_mode = GovernorMode.RWBDOT_WITH_EKF
    #                     else:
    #                         ctrl_mode = GovernorMode.BDOT_WITH_EKF
    #                 #Get B_ECI, r_ECI at t, t-1, and t+1
    #                 os = orbit_prec.get_os(t)
    #                 osp1 = orbit_prec.get_os(t+self.time_step/cent2sec)
    #                 osm1 = orbit_prec.get_os(t-self.time_step/cent2sec)
    #                 #breakpoint()
    #                     #dB_body = dB_body#np.atleast_2d(np.array(np.matrix(((rot.T@Bt_ECI -rot_mat(x_prec_prev[-4:]).T@Bt_ECI)/self.sample_time)).A1)).reshape(3)
    #                 #Get control vector by simulating whatever mode we're in (not PLAN_AND_TRACK).
    #                 pg,sv = self.goals.get_pointing_info(os,quatmode = self.controller.quaternionTo3VecMode)
    #                 pgp1,svp1 = self.goals.get_pointing_info(osp1,quatmode = self.controller.quaternionTo3VecMode)
    #                 ctrl_goal = ([],[],[],[],pg,sv)
    #                 ctrl_goalp1 = ([],[],[],[],pgp1,svp1)
    #
    #                 prop_inds = np.where([j<t for j in prop_prec_t])[0]
    #                 if len(prop_inds) == 0:
    #                     raise ValueError("There should always be a time in the list less than this...")
    #                 prop_t = prop_prec_p[max(prop_inds)]
    #                 u_prec = self.controller.update(ctrl_mode,ctrl_goal,ctrl_goalp1,x_prec,os,osp1,prop_on = prop_t,fakeTF=True)
    #                 #breakpoint()
    #                 #Now, propagate attitude forward using rk4 + normalize quaternion
    #                 #Increment t
    #                 dt = min(abs(t_start_planning-t)*cent2sec,self.time_step)
    #                 x_prec_prev = x_prec
    #                 prop_j = prop_prec_p[np.where(np.array(prop_prec_t)<=t)[0][-1]]
    #                 use_prop_j = prop_j and self.planner_settings.plan_for_prop
    #                 self.prec_t += [t]
    #                 self.prec_X += [x_prec]
    #                 self.prec_U += [u_prec]
    #                 x_prec = self.estimator.sat.rk4(x_prec, u_prec, dt, os,osp1,use_prop = use_prop_j,use_gg = self.planner_settings.plan_for_gg,use_srp = self.planner_settings.plan_for_srp,use_drag = self.planner_settings.plan_for_aero,use_resdipole = self.planner_settings.plan_for_resdipole,use_gen_dist = self.planner_settings.plan_for_gendist).reshape((self.estimator.sat.state_len,1))
    #                 #TODO: add disturbance estimation here?
    #                 t += dt/cent2sec
    #
    #     #Finally, calculate point_ECI, sat_vec and get matrix inputs ready for TP
    #     #Initialize matrix inputs for TP to all zeros
    #     # duration = math.ceil((j2000_end_planning-j2000_start_planning)/gcd_dt_altro)+1
    #     # tp_times = sorted(list(set([t_start_planning] + np.arange(t_start_planning,j2000_end_precalc,gcd_dt_tp).tolist() + [j2000_end_precalc])))
    #
    #     #Get pointing goal vec, sat pointing vec, r_ECI, v_ECI, sun_ECI, B_ECI for t=j2000_start_planning...j2000_end_planning + 1
    #     # t_tp = t_start_planning
    #     #while j2000_delta(j2000_end_planning, vec_j2000) < 0.5:
    #     #[prop_prec_p[np.where(np.array(prop_prec_t)<j)] for j in tp_orbit.times if tp_orbit.time_in_span(prop_prec_t[j])]
    #     #TODO adjust x_prec for systems with RWs by using their estimated state.
    #     return x_prec

    # def propulsion_finder(self,orbit,prev_t,prev_p):
    #     #prev meaning right before start of this orbit.
    #     prop_prec_t = sorted(orbit.times)
    #     prop_prec_p = [prev_p]
    #     for j in prop_prec_t[1:]:
    #         prop_switches = [self.prop_dict[t] for t in self.prop_dict.keys() if (j>=t and prev_t<t)]
    #         if prop_prec_p[-1]:
    #             #prop is on in previous time step
    #             if sum(prop_switches) == -1:
    #                 prop_j = False
    #             elif sum(prop_switches) == 0:
    #                 prop_j = True
    #             else:
    #                 raise ValueError("seems prop should be turned on when already on or turned on/off an uneven number of times in this interval (like on 3 times and off once, etc)")
    #         else:
    #             #prop is off in previous time step
    #             if sum(prop_switches) == 1:
    #                 prop_j = True
    #             elif sum(prop_switches) == 0:
    #                 prop_j = False
    #             else:
    #                 raise ValueError("seems prop should be turned off when already off or turned on/off an uneven number of times in this interval (like on 3 times and off once, etc)")
    #         prop_prec_p += [prop_j]
    #         prev_t = j
    # #     return prop_prec_t,prop_prec_p
    #
    # def check_for_precalculate_and_do(self):
    #     """
    #     This function checks if we are ready to precalculate given the current time tuple.
    #
    #     Parameters
    #     ------------
    #         current_time: tuple #TODO change to J2000
    #             time in centuries since January 1, 2000 (in UTC time)
    #
    #     Returns
    #     ---------
    #         precalculate: boolean
    #             True if we should start precalculating, False otherwise
    #     """
    #     # current_j2000 = self.J2000
    #     future_j2000 = self.J2000 + (self.planner_settings.precalculation_time)/cent2sec
    #     # ctrl_mode_now = self.goals.get_control_mode(current_j2000)
    #     ctrl_mode_future = self.goals.get_control_mode(future_j2000)
    #
    #     # in_plan_and_track = (ctrl_mode_now == GovernorMode.PLAN_AND_TRACK_LQR or ctrl_mode_now == GovernorMode.PLAN_AND_TRACK_MPC) #TODO should this be if *any* time between then are planned?
    #     currently_planning = self.prec_ready
    #     if self.current_trajectory.is_empty():
    #         need_new_traj = (ctrl_mode_future in [GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.PLAN_OPEN_LOOP])
    #         j2000_start_planning = self.goals.get_next_goal_change_time(self.J2000)
    #     else:
    #         need_new_traj = (ctrl_mode_future in [GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.PLAN_OPEN_LOOP]) and (future_j2000+(self.planner_settings.traj_overlap+time_eps)/cent2sec) >= self.current_trajectory.max_time()
    #         j2000_start_planning = self.current_trajectory.max_time()-self.planner_settings.traj_overlap/cent2sec
    #
    #     need_to_precalculate = not currently_planning and need_new_traj
    #     # self.prev_future_j2000 =  future_j2000
    #     if not need_to_precalculate:
    #         return need_to_precalculate,[]
    #     #Get start and end of planning
    #     """
    #     cntrl_switch, _, keys, _ = self.get_cntrl_switch(current_time)
    #     use_ind = np.where(cntrl_switch)[0][0]
    #     t_start_planning = self.control_change_times[keys[use_ind]]
    #     t_end_planning = self.control_change_times[keys[use_ind+1]]
    #     """
    #     j2000 = self.J2000
    #     j2000_end_planning_1 = j2000_start_planning + self.planner_settings.default_traj_length*1.0/cent2sec
    #     j2000_end_planning_2 = self.goals.get_next_goal_change_time(j2000_start_planning)
    #     j2000_end_planning = min(j2000_end_planning_1,j2000_end_planning_2)
    #     # print("0000000000000\n"+str((j2000_end_planning - j2000_start_planning)*cent2sec))
    #     j2000_end_planning += self.planner_settings.traj_overlap*1.0/cent2sec  #Create a little bit of overlap between trajectories
    #
    #     # print("0000000000000\n"+str((j2000_end_planning - j2000_start_planning)*cent2sec))
    #     #Call main precalculate fn to get tp inputs.
    #     x_prec = self.precalculate_for_trajectory_planner(j2000_start_planning, j2000_end_planning)
    #     #(bdot_on, plot_on) = self.tp_settings
    #     #TODO: Figure out if we need to do something with plan_for_prop here
    #     #Calculate N and t_final planning (different from t_end due to trajectory overlap)
    #     #Reshape x0 for planner and return.
    #     x_prec = np.array(x_prec).reshape((self.estimator.sat.state_len,1))
    #     return need_to_precalculate,x_prec
    #     # return x_prec

    # def switch_to_precalculated_trajectory(self, verbose=False, x_estimate=None):
    #     """
    #     This function switches an already-precalculated trajectory to be the "current" trajectory
    #     """
    #     if x_estimate is None:
    #         x_estimate = np.zeros((self.estimator.sat.state_len,1))
    #     # print(self.next_trajectory.gains)
    #     self.current_trajectory = self.next_trajectory.copy()
    #
    #     self.prec_ready = False
    #     ################################
    #     if verbose:
    #         print("\n\n********************\n********************\n"+
    #                         "estimated traj start: "+str(self.current_trajectory.first_state())+
    #                       "\n   actual traj start: "+str(x_estimate.T)+
    #                         "\n********************\n********************\n\n")
    #         # breakpoint()
    #
    # def call_trajOpt_python_debug(self, planner, vecs_w_time, N, t_start, t_end, x0, bdotOn,axes):
    #     (traj_init,vecs,costSettings,_) = planner.prepareForAlilqr(vecs_w_time, N, t_start, t_end, x0, bdotOn)
    #     alilqrOut = alilqr_python_for_debug_plot(planner,traj_init,vecs,costSettings,self.planner_settings.mainAlilqrSettings(),axes)
    #     #(optOut,mu,lastGrad) = alilqrOut
    #     (_, _,_,lqr_opt) = planner.cleanUpAfterAlilqr(vecs_w_time, N, t_start, t_end, alilqrOut)
    #     #(Xset, Uset, Kset, lamset) = main_opt
    #     (Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,lqr_times) = lqr_opt
    #     return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr
    #
    # @staticmethod
    # def call_trajOpt_cpp(planner, vecs_w_time, N, t_start, t_end, x0, bdotOn):
    #
    #     #Not getting success, lastGrad, main_opt
    #     print(x0.T)
    #     (_, _,_,lqr_opt, traj_init) = planner.trajOpt(vecs_w_time, N, t_start, t_end, x0, bdotOn)
    #     #(Xset, Uset, Kset, lamset) = main_opt
    #     (Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr,lqr_times) = lqr_opt
    #     # print(lqr_times)
    #     # print(lqr_times.shape)
    #     # (Xset, Uset, _) = traj_init
    #     #print(f"TP QUEUE SIZE IS = {planner_queue.qsize()}")
    #     return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr
    #
    # def live_debug_plot(self, fig, axes, planner_args):
    #     (planner, vecsPy, testN,j1, j2, x0, bdotOn) = planner_args
    #     lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr = self.call_trajOpt_python_debug(planner, vecsPy, N, t_start_planning, t_end_planning, x0, bdotOn,axes)
    #     return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr
    #     #time.sleep(1)
    #     #(Xset, Uset, Kset, Sset) = read_from_planner(comm_q)
    #     #return Xset, Uset, Kset, Sset
    #
    # def start_trajectory_planning(self, x0, debug_plot_on=None):
    #     bdotOn = self.planner_settings.bdot_on
    #     if debug_plot_on is None:
    #         debug_plot_on=self.planner_settings.debug_plot_on
    #     planner = self.planner
    #     #TODO: eliminate the vector time stuff and just use J2000. needs changing in the C++ code, too, though
    #     # time_tuples = [j2000_to_tuple(j) for j in self.tp_orbit.times]
    #     # time_vecs = np.array([[self.tp_orbit.times[j],time_tuples[j][0],time_tuples[j][1],time_tuples[j][2],time_tuples[j][3]] for j in range(len(self.tp_orbit.times))]).T
    #     # print([np.vstack([time_vecs,np.squeeze(np.array(k)).T]) for k in self.tp_orbit.get_vecs()])
    #     # vecsPy = tuple([np.vstack([time_vecs,np.squeeze(np.array(k)).T]) for k in self.tp_orbit.get_vecs()]+[np.vstack([time_vecs,np.squeeze(self.tp_sat_vec).T])]+[np.vstack([time_vecs,np.squeeze(self.tp_point_goal).T])])
    #     # vecsPy = tuple([j.T for j in vecsP])
    #     print("%%%%%%%%%%%%%%%%",(np.array(self.tp_orbit.times).min()-0.22)*cent2sec)
    #     vecsPy = tuple([np.copy(np.array(self.tp_orbit.times), order='C')]+[np.copy(np.squeeze(np.array(k)).T, order='C') for k in self.tp_orbit.get_vecs()] + [np.copy(np.squeeze(self.tp_sat_vec).T, order='C')]+[np.copy(np.squeeze(self.tp_point_goal).T, order='C')]+[np.copy(np.array(self.tp_prop), order='C')])
    #     N = len(self.tp_orbit.times)
    #     t_start_planning = self.tp_orbit.min_time()
    #     t_end_planning = self.tp_orbit.max_time()
    #     if debug_plot_on:
    #         #This will pause sim
    #         planner_args = (self.planner, vecsPy, N, t_start_planning, t_end_planning, x0, bdotOn)
    #         fig = plt.figure(2)
    #         ax = fig.add_subplot(812)
    #         ax2 = fig.add_subplot(813)
    #         ax3 = fig.add_subplot(814)
    #         ax4 = fig.add_subplot(815)
    #         ax5 = fig.add_subplot(816)
    #         ax6 = fig.add_subplot(817)
    #         ax7 = fig.add_subplot(818)
    #         axes = (ax,ax2,ax3,ax4,ax5,ax6,ax7)
    #         lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr = self.live_debug_plot(fig, axes, planner_args)
    #         return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr
    #     else:
    #         lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr = self.call_trajOpt_cpp(planner, vecsPy, N, t_start_planning, t_end_planning, x0, bdotOn)
    #
    #         return lqr_times,Xset_lqr, Uset_lqr, Kset_lqr, Sset_lqr
