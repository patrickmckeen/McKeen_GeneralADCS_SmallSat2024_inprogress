

from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import types
from scipy.optimize import minimize_scalar


class Controller:

    def __init__(self,sat,prev_os = None,update_period = 1,\
            control_modes = control_modes,
            bdot_gain = 10000000000,
            counter_dipole = False,
            counter_bias = False,
            pd_w_gain = 0.1,
            pd_q_gain = 10,
            fobj_tol = 1e-8,#1e-8,
            MPC_tol = 1e-8,#1e-10,
            constraint_replacement_val = 1e-10,
            control_wt = None,
            MPC_use_hess = True,
            MPC_step_cycle = 12,
            MPC_barrier_cycle = 12,
            beta0 = 10,
            beta_ratio = 10,
            MPC_use_solve = False,
            MPC_eigval_lim = 1e-2,
            MPC_line_search_cycle = 20,
            MPC_line_search_alpha = 0.5,
            c1 = 1e-4,
            stop_spin_limit = 1*math.pi/180,
            include_gg = True, include_prop = False, include_drag = False, include_srp = False, include_resdipole = False, include_gendist = False,
            MPC_use_ctg = False,
            MPC_remove_B = False,
            ctrl_wt_base = 1e-16,
            ctrl_diff_from_plan_wt_base = 1e-6,
            control_diff_from_plan_wt = None,
            MPC_plot = False,
            MPC_use_integral_term = False,
            MPC_use_av_integral_term = False,
            MPC_integral_gain = 0.01,
            MPC_av_integral_gain = 0.01,
            MPC_integral_decay_rate = 0.3,
            MPC_av_integral_decay_rate = 0.3,
            MPC_weight_control_change_from_prev = True,
            MPC_weight_control_change_from_plan = False,
            tracking_LQR_formulation = 0,
            RW_control_constraint_fudge_scale = 0.95,
            quaternionTo3VecMode = 0
            # counter_dipole = False,
            # counter_bias = False
            ): #do Ctrl+Alt+[ (cmd+opt+[), or editor:fold-current-row in the command palette to fold this whole function in Atom
        # if sat is None:
        #     self.sat = Satellite()
        # else:
        #     self.sat = sat
        if prev_os is None:
            self.prev_os =  Orbital_State(0,np.array([[0,0,1]]).T,np.array([[0,0,0]]).T)
        else:
            self.prev_os = prev_os
        self.update_period = update_period
        self.control_modes = control_modes

        # self.bdot_gain = bdot_gain
        self.ctrl_wt_base = ctrl_wt_base
        self.ctrl_diff_from_plan_wt_base = ctrl_diff_from_plan_wt_base
        self.pd_w_gain = pd_w_gain
        self.pd_q_gain = pd_q_gain
        self.fobj_tol = fobj_tol
        self.constraint_replacement_val = constraint_replacement_val

        self.quaternionTo3VecMode = quaternionTo3VecMode #0 is 2*qv*sign(q0)/(1+|q0|), 1 is 2*qv/(1+q0), 2 is qv/q0


        self.MPC_use_ctg = MPC_use_ctg
        self.sat = sat
        self.MPC_tol = MPC_tol
        if control_diff_from_plan_wt is None:
            control_diff_from_plan_wt = self.ctrl_diff_from_plan_wt_base*np.eye(self.sat.control_len)
        elif isinstance(control_diff_from_plan_wt,NumberTypes):
            control_diff_from_plan_wt = control_diff_from_plan_wt*np.eye(self.sat.control_len)
        if control_wt is None:
            control_wt = self.ctrl_wt_base*np.eye(self.sat.control_len)
        elif isinstance(control_wt,NumberTypes):
            control_wt = control_wt*np.eye(self.sat.control_len)
        self.control_wt = control_wt
        self.control_diff_from_plan_wt = control_diff_from_plan_wt
        self.MPC_use_hess = MPC_use_hess
        self.MPC_use_solve = MPC_use_solve
        self.MPC_step_cycle = MPC_step_cycle
        self.MPC_barrier_cycle = MPC_barrier_cycle
        self.MPC_remove_B = MPC_remove_B
        self.prev_mpc_score= np.nan
        self.beta0 = beta0
        self.beta_ratio = beta_ratio
        self.MPC_line_search_cycle = MPC_line_search_cycle
        self.MPC_line_search_alpha = MPC_line_search_alpha
        self.MPC_plot = MPC_plot
        self.c1 = c1
        self.MPC_eigval_lim = MPC_eigval_lim
        self.stop_spin_limit = stop_spin_limit
        self.MPC_use_integral_term = MPC_use_integral_term
        self.MPC_use_av_integral_term = MPC_use_av_integral_term
        self.MPC_integral_value = zeroquat
        self.MPC_av_integral_value = np.zeros((3,1))
        self.MPC_integral_gain = MPC_integral_gain
        self.MPC_av_integral_gain = MPC_av_integral_gain
        self.MPC_integral_decay_rate = MPC_integral_decay_rate
        self.MPC_av_integral_decay_rate = MPC_av_integral_decay_rate
        self.MPC_weight_control_change_from_prev = MPC_weight_control_change_from_prev
        self.MPC_weight_control_change_from_plan = MPC_weight_control_change_from_plan
        self.prev_control = np.nan*np.ones((self.sat.control_len,))
        self.tracking_LQR_formulation = tracking_LQR_formulation
        self.RW_control_constraint_fudge_scale = RW_control_constraint_fudge_scale

        self.include_gg = bool(include_gg)
        self.include_drag = bool(include_drag)
        self.include_srp = bool(include_srp)
        self.include_prop = bool(include_prop)
        self.include_gendist = bool(include_gendist)
        self.include_resdipole = bool(include_resdipole)
        self.counter_dipole = bool(counter_dipole)
        self.counter_bias = bool(counter_bias)
        #
        # self.orbit_B_field_avg = np.nan
        # self.orbit_B_field_avg_calc = False
        # self.orbit_B_field_avg_calc_time = 0
        # self.orbit_B_field_avg_num_periods = 2.0
        # self.orbit_B_field_avg_dt_sec = 60.0
        # self.orbit_B_field_stale_time_sec = 10.0*24.0*60.0*60.0 #10 days?
        # self.lovera_pd_w_gain = 50# 1.0
        # self.lovera_pd_q_gain = 50#2.5#4.0
        # self.lovera_eps = 0.01
        # self.lovera_beta = 0.15#0.0002
        # self.lovera_gamma0_mat = np.zeros((3,3))
        # self.lovera_gamma0_count = np.nan


        # self.wie_pd_w_gain = np.eye(3)*200*20
        # self.wie_pd_q_gain = np.eye(3)*0.5*20

        # self.wisniewski_Lambda_q = np.diagflat([0.002,0.002,0.002])
        # self.wisniewski_Lambda_s = np.diagflat([0.003,0.003,0.003])
        # self.prevb = 0
        # self.addl_info = {} # for pulling data out of control processes in simulation. Cleared every update.


    # def reset_sat(self,sat,control_wt=None,control_diff_from_plan_wt=None):
    #     self.sat = sat
    #     if control_wt is None:
    #         control_wt = self.ctrl_wt_base*np.eye(self.sat.control_len)
    #     elif isinstance(control_wt,NumberTypes):
    #         control_wt = control_wt*np.eye(self.sat.control_len)
    #     if control_diff_from_plan_wt is None:
    #         control_diff_from_plan_wt = self.ctrl_diff_from_plan_wt_base*np.eye(self.sat.control_len)
    #     elif isinstance(control_diff_from_plan_wt,NumberTypes):
    #         control_diff_from_plan_wt = control_diff_from_plan_wt*np.eye(self.sat.control_len)
    #     self.control_wt = control_wt
    #     self.control_diff_from_plan_wt = control_diff_from_plan_wt


    def Xupdate(self,ctrl_mode,ctrl_goal,next_ctrl_goal,state,os,osp1,prop_on = False,is_fake=False):
        self.prev_mpc_score = np.nan
        self.addl_info.clear()
        (plan_state,plan_control,plan_gain,plan_ctg,_,_) = ctrl_goal
        state = np.copy(state)
        if isinstance(plan_control,np.ndarray):
            plan_control = np.copy(plan_control)
        if isinstance(plan_gain,np.ndarray):
            plan_gain = np.copy(plan_gain)
        if isinstance(plan_ctg,np.ndarray):
            plan_ctg = np.copy(plan_ctg)
        if isinstance(plan_state,np.ndarray):
            plan_state = np.copy(plan_state)
            q_goal = plan_state[3:7].reshape((4,1))
            q_c = state[3:7].reshape((4,1))
            quaterr = quat_mult(quat_inv(q_c),q_goal)
            # self.MPC_integral_value = quat_mult(slerp(self.MPC_integral_value,zeroquat,self.MPC_integral_decay_rate),quaterr)
            # self.MPC_av_integral_value = self.MPC_av_integral_value*(1-self.MPC_av_integral_decay_rate) + (plan_state[:3,:].reshape((3,1)) - state[:3,:].reshape((3,1)))
        # if dB_body is None:
        #     dB_body = np.array([[0,0,0]]).T
        # if ctrl_mode == GovernorMode.NO_CONTROL:
        #     u = self.no_control()
        # elif ctrl_mode == GovernorMode.SIMPLE_BDOT:
        #     u = self.bdot(dB_body)
        # elif ctrl_mode == GovernorMode.BDOT_WITH_EKF:
        #     u = self.bdot_with_ekf(state,os,is_fake)
        elif ctrl_mode == GovernorMode.RWBDOT_WITH_EKF:
            u = self.rwbdot_with_ekf(state,os,is_fake)
        # elif ctrl_mode == GovernorMode.SIMPLE_MAG_PD: #TODO: not tested
        #     goal_state = np.vstack([np.zeros((3,1)),vec3_to_quat(ctrl_goal[4],self.quaternionTo3VecMode)]).reshape((7,1))
        #     u = self.simple_mag_pd(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.TRAJ_MAG_PD: #TODO: not tested
        #     goal_state = ctrl_goal[0]
        #     u = self.simple_mag_pd(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.WISNIEWSKI_SLIDING: #TODO: not tested
        #     # breakpoint()
        #     qwgoal = quat_mult(vec3_to_quat(next_ctrl_goal[4],self.quaternionTo3VecMode),quat_inv(vec3_to_quat(ctrl_goal[4],self.quaternionTo3VecMode)))
        #     qwgoal *= np.sign(qwgoal[0])
        #     # wgoal = 2.0*rot_mat(state[3:7]).T@qwgoal[1:]*(1-norm(qwgoal[1:])**2/3/qwgoal[0]**2.0)/qwgoal[0]/self.update_period
        #     wgoal = rot_mat(state[3:7]).T@quat_log(qwgoal)/self.update_period #rot_mat(state[3:7]).T@
        #     # breakpoint()
        #     goal_state = np.vstack([wgoal,vec3_to_quat(ctrl_goal[4],self.quaternionTo3VecMode)]).reshape((7,1))
        #     u = self.wisniewski_sliding(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.LOVERA_MAG_PD: #TODO: not tested
        #     goal_state = np.vstack([np.zeros((3,1)),vec3_to_quat(ctrl_goal[4],self.quaternionTo3VecMode)]).reshape((7,1))
        #     u = self.lovera_mag_pd(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.WIE_MAGIC_PD: #TODO: not tested
        #     goal_state = np.vstack([np.zeros((3,1)),vec3_to_quat(ctrl_goal[4],self.quaternionTo3VecMode)]).reshape((7,1))
        #     u = self.wie_magic_pd(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.LOVERA_MAG_PD_QUATSET_B: #TODO: not tested
        #     goal_state = [np.vstack([np.zeros((3,1)),ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     u = self.lovera_mag_pd_quatset_perpB(state,os,goal_state,is_fake)

        # elif ctrl_mode == GovernorMode.LOVERA_MAG_PD_QUATSET_LYAP: #TODO: not tested
        #     goal_state = [np.vstack([np.zeros((3,1)),ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     u = self.lovera_mag_pd_quatset_minLyap(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.WISNIEWSKI_SLIDING_QUATSET_B: #TODO: not tested
        #
        #     goal_state = [np.vstack([np.zeros((3,1)),ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     next_goal_state = [np.vstack([np.zeros((3,1)),next_ctrl_goal[4]]).reshape((6,1)),next_ctrl_goal[5]]
        #     u = normalize(goal_state[0][3:,:]).reshape((3,1))
        #     v = normalize(goal_state[1]).reshape((3,1))
        #
        #     un = normalize(next_goal_state[0][3:,:]).reshape((3,1))
        #     vn = normalize(next_goal_state[1]).reshape((3,1))
        #
        #     uxun = np.cross(u,un)
        #     vxvn = np.cross(v,vn)
        #     wu = np.zeros((3,1))
        #     wv = np.zeros((3,1))
        #     if norm(uxun) > 0:
        #         wu = (uxun*math.asin(norm(uxun))/norm(uxun))/self.update_period
        #     if norm(vxvn) > 0:
        #         wv = (vxvn*math.asin(norm(vxvn))/norm(vxvn))/self.update_period
        #     wgoal = rot_mat(state[3:7]).T@wu - wv
        #     goal_state = [np.vstack([wgoal,ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     u = self.wisniewski_sliding_quatset_perpB(state,os,goal_state,is_fake)

        #
        # elif ctrl_mode == GovernorMode.WISNIEWSKI_SLIDING_QUATSET_ANG: #TODO: not tested
        #
        #     goal_state = [np.vstack([np.zeros((3,1)),ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     next_goal_state = [np.vstack([np.zeros((3,1)),next_ctrl_goal[4]]).reshape((6,1)),next_ctrl_goal[5]]
        #     u = normalize(goal_state[0][3:,:]).reshape((3,1))
        #     v = normalize(goal_state[1]).reshape((3,1))
        #
        #     un = normalize(next_goal_state[0][3:,:]).reshape((3,1))
        #     vn = normalize(next_goal_state[1]).reshape((3,1))
        #
        #     uxun = np.cross(u,un)
        #     vxvn = np.cross(v,vn)
        #     wu = np.zeros((3,1))
        #     wv = np.zeros((3,1))
        #     if norm(uxun) > 0:
        #         wu = (uxun*math.asin(norm(uxun))/norm(uxun))/self.update_period
        #     if norm(vxvn) > 0:
        #         wv = (vxvn*math.asin(norm(vxvn))/norm(vxvn))/self.update_period
        #
        #     goal_state = [np.vstack([rot_mat(state[3:7]).T@wu,ctrl_goal[4]]).reshape((6,1)),np.vstack([wv,ctrl_goal[5]]).reshape((6,1))]
        #     u = self.wisniewski_sliding_quatset_minang(state,os,goal_state,is_fake)

        # elif ctrl_mode == GovernorMode.WISNIEWSKI_SLIDING_QUATSET_LYAP: #TODO: not tested
        #
        #     goal_state = [np.vstack([np.zeros((3,1)),ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     next_goal_state = [np.vstack([np.zeros((3,1)),next_ctrl_goal[4]]).reshape((6,1)),next_ctrl_goal[5]]
        #     u = normalize(goal_state[0][3:,:]).reshape((3,1))
        #     v = normalize(goal_state[1]).reshape((3,1))
        #
        #     un = normalize(next_goal_state[0][3:,:]).reshape((3,1))
        #     vn = normalize(next_goal_state[1]).reshape((3,1))
        #
        #     uxun = np.cross(u,un)
        #     vxvn = np.cross(v,vn)
        #     wu = np.zeros((3,1))
        #     wv = np.zeros((3,1))
        #     if norm(uxun) > 0:
        #         wu = (uxun*math.asin(norm(uxun))/norm(uxun))/self.update_period
        #     if norm(vxvn) > 0:
        #         wv = (vxvn*math.asin(norm(vxvn))/norm(vxvn))/self.update_period
        #     wgoal = rot_mat(state[3:7]).T@wu - wv
        #     # goal_state = [np.vstack([wgoal,ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     goal_state = [np.vstack([rot_mat(state[3:7]).T@wu,ctrl_goal[4]]).reshape((6,1)),np.vstack([wv,ctrl_goal[5]]).reshape((6,1))]
        #
        #     u = self.wisniewski_sliding_quatset_minLyap(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.LOVERA_MAG_PD_QUATSET_ANG: #TODO: not tested
        #     goal_state = [np.vstack([np.zeros((3,1)),ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     u = self.lovera_mag_pd_quatset_minang(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.WIE_MAGIC_PD_QUATSET_ANG: #TODO: not tested
        #     goal_state = [np.vstack([np.zeros((3,1)),ctrl_goal[4]]).reshape((6,1)),ctrl_goal[5]]
        #     u = self.wie_magic_pd_quatset_minang(state,os,goal_state,is_fake)
        # elif ctrl_mode == GovernorMode.PLAN_AND_TRACK_LQR: #TODO: not tested
        #
        #     # if self.MPC_use_integral_term:
        #     #     plan_state[3:7] = quat_mult(plan_state[3:7],quat_power(self.MPC_integral_value,self.MPC_integral_gain))
        #     # if self.MPC_use_av_integral_term:
        #     #     plan_state[:3,:] += self.MPC_av_integral_value*self.MPC_av_integral_gain
        #     # # (plan_state,plan_control,plan_gain,plan_ctg,_,_) = ctrl_goal
        #     # (next_state,next_control,next_gain,next_ctg,_,_) = next_ctrl_goal
        #     u = self.traj_lqr(plan_state,plan_control,plan_gain,plan_ctg,state,os,osp1,is_fake)
        elif ctrl_mode == GovernorMode.PLAN_AND_TRACK_MPC: #TODO: note tested
            (next_state,next_control,next_gain,next_ctg,_,_) = next_ctrl_goal
            inputstate = next_state
            # next_state = np.copy(next_state)
            if isinstance(next_control,np.ndarray):
                next_control = np.copy(next_control)
            if isinstance(next_gain,np.ndarray):
                next_gain = np.copy(next_gain)
            if isinstance(next_ctg,np.ndarray):
                next_ctg = np.copy(next_ctg)
            if isinstance(next_state,np.ndarray):
                next_state = np.copy(next_state)
            # if self.MPC_use_integral_term:
            #     next_state[3:7] = quat_mult(next_state[3:7],quat_power(self.MPC_integral_value,self.MPC_integral_gain))
            # if self.MPC_use_av_integral_term:
            #     next_state[:3,:] += self.MPC_av_integral_value*self.MPC_av_integral_gain
            # print("current:\n",state.T)
                # breakpoint()
            # print("goal:\n",next_state.T)
            if norm(state[0:3]) > self.stop_spin_limit:
                if self.sat.number_RW==0:
                    u0 = self.bdot_with_ekf(state, os, True)
                else:
                    u0 = self.rwbdot_with_ekf(state, os, True)
            else:
                u0 = self.traj_lqr(plan_state,plan_control,plan_gain,plan_ctg,state,os,osp1,is_fake)
            u0 *= 0.99 #keep it from being too close to boundaries initially
            u = self.traj_mpc(u0,next_state,next_gain,next_ctg,state,os,osp1,plan_control,is_fake,prop_on)
        # elif ctrl_mode == GovernorMode.PLAN_OPEN_LOOP:
        #     u = plan_control
        else:
            raise ValueError("invalid control mode")
        # if not is_fake:
        #     self.prev_os = os
        #     self.prev_control = u
        if np.any(np.isnan(u)):
            raise ValueError("nan control")
        return u.reshape((self.sat.control_len,1)),self.addl_info
        #TODO: add more control modes, especially those with RWs


    # def orbit_B_field_avg_test_and_fix(self,os):
    #     # if isnan(self.lovera_gamma0_count):
    #     #     self.lovera_gamma0_mat = os.
    #     # self.lovera_gamma0_mat =
    #     if ((os.J2000 > self.orbit_B_field_avg_calc_time + sec2cent*self.orbit_B_field_stale_time_sec) or not self.orbit_B_field_avg_calc) : #TODO--in flight software, this can't be done in the control call. Need to have th4e adcs unit seeing in advance if the Lovera PD will ever be used and then calculating this stuff. Also may need to recacalculated so often?
    #         orb_energy = 0.5*(norm(os.V))**2.0 - mu_e/norm(os.R)#vis-viva in km^s/s^2
    #         sma = -0.5*mu_e/orb_energy #semi-major axis in km
    #         if sma < 0:
    #             sma = sma*-1.0 #this really shouldn't occur for our test and application cases....
    #         orb_period = 2.0*math.pi*math.sqrt(sma**3.0/mu_e)
    #         orbit_pred = Orbit(os,os.J2000 + self.orbit_B_field_avg_num_periods*orb_period*sec2cent,self.orbit_B_field_avg_dt_sec)
    #         orb_vecs = orbit_pred.get_vecs()
    #         orb_b = orb_vecs[2]
    #         self.orbit_B_field_avg = sum(orb_b)/len(orb_b)
    #         self.orbit_B_field_avg_calc = True
    #         self.orbit_B_field_avg_calc_time = os.J2000


    # def minang_quat(self, state,os, goal_state,is_fake):
    #     u = normalize(goal_state[0][3:])
    #     v = normalize(goal_state[1])
    #     th = np.arccos(np.dot(u,v))
    #     ct2 = np.cos(th/2.0)
    #     xq = np.concatenate([ct2,0.5*np.cross(v,u)/ct2])
    #     yq = np.concatenate([[0],0.5*(v+u)/ct2])
    #     q0 = state[3:7]
    #     qg = normalize(xq*np.dot(q0,xq)+yq*np.dot(q0,yq)) # from my quaternion pointing note
    #     return np.concatenate([goal_state[0][0:3],qg])

    #
    # def reduced_state_err(self,state,desired):
    #     state = state.reshape((self.sat.state_len,1))
    #     desired = desired.reshape((self.sat.state_len,1))
    #     q = state[3:7].reshape((4,1))
    #     w = state[0:3].reshape((3,1))
    #     extra = state[7:,:].reshape((self.sat.state_len-7,1))
    #     q_desired = desired[3:7]
    #     w_desired = desired[0:3]
    #     extra_desired = desired[7:,:].reshape((self.sat.state_len-7,1))
    #     q_desired = normalize(q_desired).reshape((4,1))
    #     # q = q*np.sign(q_desired[0])*np.sign(q[0])
    #     q_err = quat_mult(quat_inv(q_desired),q)#np.vstack([np.array([np.ndarray.item(q.T@q_desired)]),-q_desired[1:]*q[0]+q[1:]*q_desired[0]-np.cross(q_desired[1:],q[1:])])
    #     q_err = normalize(q_err).reshape((4,1))
    #     # return np.vstack([w-w_desired,q_err[1:].reshape((3,1)),extra-extra_desired]).reshape((self.sat.state_len-1,1))
    #     # return np.vstack([w-w_desired,q_err[1:].reshape((3,1))/np.ndarray.item(q_err[0]),extra-extra_desired]).reshape((self.sat.state_len-1,1))
    #
    #     return np.vstack([w-w_desired,quat_to_vec3(q_err,self.quaternionTo3VecMode),extra-extra_desired]).reshape((self.sat.state_len-1,1))
    #
    # def reduced_state_err_jac(self,state,desired):
    #     state = state.reshape((self.sat.state_len,1))
    #     desired = desired.reshape((self.sat.state_len,1))
    #     q = state[3:7].reshape((4,1))
    #     w = state[0:3].reshape((3,1))
    #     # extra = state[7:,:].reshape((self.sat.state_len-7,1))
    #     q_desired = desired[3:7]
    #     w_desired = desired[0:3]
    #     # extra_desired = desired[7:,:].reshape((self.sat.state_len-7,1))
    #     q_desired = normalize(q_desired).reshape((4,1))
    #     sq = 1#np.sign(q[0])
    #     # q = q*np.sign(q_desired[0])*sq
    #     q_err = quat_mult(quat_inv(q_desired),q)
    #     se = 1#np.sign(q_err[0])
    #     nq_err = normalize(q_err*se).reshape((4,1))
    #     hlen = self.sat.state_len-7
    #     dwe__dw = np.eye(3)
    #     dwe__dq = np.zeros((3,4))
    #     dwe__dh = np.zeros((3,hlen))
    #     dve__dw = np.zeros((3,3))
    #     dqe__dq = np.vstack([q_desired.T,np.hstack([-q_desired[1:],np.eye(3)*q_desired[0]-skewsym(q_desired[1:])])])#*sq*np.sign(q_desired[0])
    #     dnqe__dqe = quat_norm_jac(q_err)
    #     dve__dnqe = quat_to_vec3_deriv(q_err,self.quaternionTo3VecMode)
    #     dve__dq = dve__dnqe@dnqe__dqe@dqe__dq
    #     dve__dh = np.zeros((3,hlen))
    #     dhe__dw = np.zeros((hlen,3))
    #     dhe__dq = np.zeros((hlen,4))
    #     dhe__dh = np.eye(hlen)
    #     return np.block([[dwe__dw,dwe__dq,dwe__dh],[dve__dw,dve__dq,dve__dh],[dhe__dw,dhe__dq,dhe__dh]])
    # def reduced_state_err_hess(self,state,desired):
    #     state = state.reshape((self.sat.state_len,1))
    #     desired = desired.reshape((self.sat.state_len,1))
    #     q = state[3:7].reshape((4,1))
    #     # extra = state[7:,:].reshape((self.sat.state_len-7,1))
    #     q_desired = desired[3:7]
    #     # extra_desired = desired[7:,:].reshape((self.sat.state_len-7,1))
    #     q_desired = normalize(q_desired).reshape((4,1))
    #     sq = 1#np.sign(np.copy(q[0]))
    #     # q = q*np.sign(q_desired[0])*sq
    #     q_err = quat_mult(quat_inv(q_desired),q)
    #     nq_err = normalize(q_err).reshape((4,1))
    #
    #     dqe__dq = np.vstack([q_desired.T,np.hstack([-q_desired[1:],np.eye(3)*q_desired[0]-skewsym(q_desired[1:])])])#*np.sign(q_desired[0])
    #     dnqe__dqe = quat_norm_jac(q_err)
    #     dve__dnqe = quat_to_vec3_deriv(q_err,self.quaternionTo3VecMode)
    #     dve__dq = dve__dnqe@dnqe__dqe@dqe__dq
    #
    #     ddve__dqednqei = [x@dnqe__dqe for x in quat_to_vec3_deriv2(nq_err,self.quaternionTo3VecMode)]
    #     ddnqe__dqedqei = quat_norm_hess(q_err)
    #     ddve__dqedqei = multi_matrix_chain_rule_vector([ddve__dqednqei],[dnqe__dqe],1,[0])
    #     ddve__dqdqei = [(dve__dnqe@x+y)@dqe__dq for x,y in zip(ddnqe__dqedqei,ddve__dqedqei)]
    #
    #
    #     ddve__dqdqi = multi_matrix_chain_rule_vector([ddve__dqdqei],[dqe__dq],1,[0])
    #     output = [np.zeros((self.sat.state_len-1,self.sat.state_len)) for j in range(self.sat.state_len)]
    #     for k in range(4):
    #         dderr__dqdqi = output[3+k]
    #         dderr__dqdqi[3:6,3:7] = ddve__dqdqi[k]
    #         output[3+k] = dderr__dqdqi
    #     return output

    def control_constraints(self,state,os,control_des = None,torq_des = None,alph = None):
        if alph == None:
            alph = self.RW_control_constraint_fudge_scale
        if control_des is not None and torq_des is not None:
            warnings.warn('both should not be provided. Using torq_des')
            control_des = None
        l,u = self.sat.control_bounds()
        if self.sat.number_RW>0:
            l_rw = l[self.sat.number_MTQ:self.sat.number_MTQ+self.sat.number_RW,:]
            u_rw = u[self.sat.number_MTQ:self.sat.number_MTQ+self.sat.number_RW,:]
            if control_des is not None:
                control_lim = self.sat.apply_control_bounds(control_des)
                mtq_lim, rw_lim,mag_lim = self.sat.sort_control_vec(control_lim)
                Bbody = rot_mat(state[3:7]).T@os.B
                torq_des = self.sat.magic_ax_mat@mag_lim + self.sat.RW_ax_mat@rw_lim - skewsym(Bbody)@self.sat.MTQ_ax_mat@mtq_lim
            RW_h = state[7:,:]
            w = state[:3,:]
            ang_mom = self.sat.J@w + self.sat.RW_ax_mat@RW_h

            l[self.sat.number_MTQ:self.sat.number_MTQ+self.sat.number_RW,:] = np.array([max(l_rw[j,0],
                                                                                        -((alph*self.sat.RW_saturation[j] - RW_h[j,0])/self.update_period + self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@(-np.cross(w,ang_mom) + torq_des )).item())/(1+self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@self.sat.RW_z_axis[j]).item())
                                                                                        ) for j in range(self.sat.number_RW)]).reshape((self.sat.number_RW,1))
            u[self.sat.number_MTQ:self.sat.number_MTQ+self.sat.number_RW,:] = np.array([min(u_rw[j,0],
                                                                                        -((-alph*self.sat.RW_saturation[j] - RW_h[j,0])/self.update_period + self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@(-np.cross(w,ang_mom) + torq_des )).item())/(1+self.sat.RW_J[j]*(self.sat.RW_z_axis[j].T@self.sat.invJ_noRW@self.sat.RW_z_axis[j]).item())
                                                                                        ) for j in range(self.sat.number_RW)]).reshape((self.sat.number_RW,1))
        return l,u

    def control_from_des(self,state,os,control_des,is_fake):
        # keep total RW torque (inc. bias, if relevant) parallel to desired total RW torque
        # keep total MTQ moment (inc. bias and residual dipole, if relevant) parallel to desired total MTQ moment
        # total RW torque and total MTQ moment, compared to their desired values, should be scaled by the same scalar.
        # respect torque limits, moment limits, and RW saturation limits
        # should preserve total torque direction. does NOT consider the effect of the usage/AM costs between various RW, MTQ, etc vary and this is part of the motivation for the desired value


        l,u = self.control_constraints(state,os,control_des = control_des)

        xadj = np.zeros((self.sat.control_len,1))
        mtq_des,_,_ = self.sat.sort_control_vec(control_des)
        if (not is_fake) and self.counter_dipole and self.sat.number_MTQ>0:
            if mtq_des.size>0 and not np.all(mtq_des==0):
                if self.sat.MTQ_ax_mat_rank<3:
                    warnings.warn('not full rank of MTQ, cannot correct for dipole. Ignoring dipole correction.')
                elif self.sat.MTQ_ax_mat_rank>3:
                    raise ValueError("this shouldn't be possible")
                elif self.sat.number_MTQ==3:
                    offset_vec_mtq = np.linalg.solve(self.sat.MTQ_ax_mat,self.sat.res_dipole.reshape((3,1)))
                    xadj[0:3] = offset_vec_mtq;
                else:
                    # offset_vec = np.zeros((self.sat.control_len,1))
                    offset_vec_mtq = pinv_LIrows(self.sat.MTQ_ax_mat)@self.sat.res_dipole.reshape((3,1))
                    Zmat = self.sat.MTQ_ax_mat_ns
                    xadj[0:self.sat.number_MTQ,:] = (np.eye(self.sat.number_MTQ)-Zmat@pinv_LIcols(Zmat))@offset_vec_mtq
        u_ref0 = control_des

        if (not is_fake) and self.counter_bias:
            bias = np.array([[ self.sat.act_noise[j].bias  if  self.sat.act_noise[j].has_bias else 0 for j in range(self.sat.control_len)]]).T
            xadj += bias

        scale = 1

        upper_lim = u+xadj#np.fmax(u-xadj,u)
        lower_lim = l+xadj#np.fmax(l-xadj,l)
        # upper_test = (u_ref0) - upper_lim
        # lower_test = (u_ref0) - lower_lim


        upper_list = np.fmin(np.where(u_ref0>0,upper_lim/u_ref0,np.inf),np.where(u_ref0<0,lower_lim/u_ref0,np.inf));
        lower_list = np.fmax(np.where(u_ref0>0,lower_lim/u_ref0,0.0),np.where(u_ref0<0,upper_lim/u_ref0,0.0));

        scale_max = np.nanmin(upper_list)
        scale_min = np.nanmax(lower_list)
        # if np.isnan(scale_max):
        #     scale_max = np.inf
        # if np.isnan(scale_min):
        #     scale_min = np.ninf
        if scale_min>scale_max:
            breakpoint()
            raise ValueError('no viable solution!')
        if scale_max>=1 and scale_min<=1:
            scale = 1.0
        else:
            if scale_min > 1.0:
                scale = scale_min
            else:
                scale = scale_max
        u_ref = u_ref0*scale;
        out = u_ref - xadj
        return out

    def control_from_wdotdes(self,state,os,wdotdes,is_fake,require_parallel = False):
        # counter_dipole = self.counter_dipole
        # counter_bias = self.counter_bias

        RW_h = state[7:,:]
        w = state[0:3]
        q = state[3:7]


        Bbody = rot_mat(q).T@os.B
        ang_mom = self.sat.J@w + self.sat.RW_ax_mat@RW_h

        mtqA = np.zeros((3,0))
        if self.sat.number_MTQ>0:
            mtqA = -skewsym(Bbody)@self.sat.MTQ_ax_mat# np.hstack([-skewsym(Bbody)@i for i in self.sat.MTQ_axes])
        rwA = np.zeros((3,0))
        rwA0 = np.zeros((3,0))
        if self.sat.number_RW>0:
            rwA0 = self.sat.RW_ax_mat# np.hstack([i for i in self.sat.RW_z_axis])
            rwA = (np.eye(3)+0.5*self.update_period*(skewsym(w+self.update_period*wdotdes/3)))@rwA0
        magA = np.zeros((3,0))
        if self.sat.number_magic>0:
            magA = self.sat.magic_ax_mat#np.hstack([i for i in self.sat.magic_axes])
        Amat = np.hstack([mtqA,rwA,magA])
        Amat0 = np.hstack([mtqA,rwA0,magA])
        A_mod = (np.eye(3)-0.5*self.update_period*skewsym(ang_mom)@self.sat.invJ_noRW)
        A_add = 0.5*skewsym(w)@np.hstack([mtqA,0*rwA0,magA])
        # A_addl = np.hstack([np.zeros((3,self.sat.number_MTQ)),A_addl_rw,np.zeros((3,self.sat.number_magic))])
        # A_addl = np.hstack([np.zeros((3,self.sat.number_MTQ)),A_addl_rw,np.zeros((3,self.sat.number_magic))]) - 0.5*self.update_period*skewsym(w)@(np.eye(3)-self.sat.RW_ax_mat@np.diagflat(self.sat.RW_J)@self.sat.RW_ax_mat.T)@Amat
        # b = self.sat.J_noRW@wdotdes
        # b_addl = np.cross(w,self.sat.J@w + 0.5*self.sat.RW_ax_mat@RW_h)
        # ang_mom_dot_des = self.sat.J@wdotdes + self.sat.RW_ax_mat@(-np.diagflat(self.sat.RW_J)@self.sat.RW_ax_mat.T@wdotdes)
        b = self.sat.J_noRW@wdotdes +  np.cross(w + 0.5*wdotdes*self.update_period,ang_mom) +  0.5*self.update_period*np.cross(w + wdotdes*self.update_period/3,self.sat.J_noRW@wdotdes)
        # b_addl += 0.5*self.update_period*(np.cross(wdotdes,ang_mom) + np.cross(w,ang_mom_dot_des) + self.update_period*(np.cross(wdotdes,ang_mom_dot_des)))
        b_addl = np.zeros((3,1))
        if (not is_fake) and self.counter_dipole:
            b_addl += A_mod@np.cross(Bbody,self.sat.res_dipole)
        bias = np.zeros((self.sat.control_len,1))
        if (not is_fake) and self.counter_bias:
            bias = np.array([[ self.sat.act_noise[j].bias  if  self.sat.act_noise[j].has_bias else 0 for j in range(self.sat.control_len)]]).T
            # mtq_bias,rw_bias,mag_bias = self.sat.sort_control_vec(bias)
            # b_addl -= mtqA@mtq_bias#sum([-np.cross(Bbody,self.sat.MTQ_axes[j]*self.sat.act_noise[j].bias) for j in range(self.sat.number_MTQ) if  self.sat.act_noise[j].has_bias])
            # b_addl -= rwA@rw_bias#sum([self.sat.RW_z_axis[j]*self.sat.act_noise[self.sat.number_MTQ+j].bias for j in range(self.sat.number_RW) if  self.sat.act_noise[self.sat.number_MTQ+j].has_bias])
            # b_addl -= magA@mag_bias#sum([self.sat.magic_axes[j]*self.sat.act_noise[self.sat.number_MTQ+self.sat.number_RW+j].bias for j in range(self.sat.number_magic) if  self.sat.act_noise[self.sat.number_MTQ+self.sat.number_RW+j].has_bias])
        try:
            l0,u0 = self.control_constraints(state,os,torq_des = b)
            u_ref = scipy.optimize.lsq_linear(Amat,b.squeeze(),bounds = (l0.squeeze(),u0.squeeze()))
            poss_ctrl = u_ref.x.reshape((self.sat.control_len,1))
            poss_torq = Amat@poss_ctrl
            l,u = self.control_constraints(state,os,torq_des = poss_torq)
            u_ref2 = scipy.optimize.lsq_linear(A_mod@Amat0 + A_add,((A_mod@Amat0+A_add)@(poss_ctrl-bias) + b_addl).squeeze(),bounds = (l.squeeze(),u.squeeze()))
        except:
            breakpoint()
        out = u_ref2.x.reshape((self.sat.control_len,1))
        if require_parallel:
            raise ValueError("not yet implemented")
        return out












    #
    #
    #
    #
    #
    #
    #
    # def traj_lqr(self,plan_state,plan_control,plan_gain,plan_ctg,state,os,osp1,is_fake):
    #     err = self.reduced_state_err(state,plan_state)
    #     if self.tracking_LQR_formulation==0:
    #         u = plan_control - plan_gain@err
    #     elif self.tracking_LQR_formulation==1:
    #         u = plan_control - plan_gain@np.concatenate([state,[1]])
    #     elif self.tracking_LQR_formulation==2:
    #         u = plan_control - plan_gain@np.concatenate([err,self.sat.gen_dist_torq()])#err #TODO better use the torque here--it should be the difference between the disturbing torque being experied/estimated and those that were planned for (if GG was in planner but not controller, GG should be taken out of dist torq est)
    #     else:
    #         raise ValueError("need correct number of LQR tracking formulation")
    #     return self.control_from_des(state,os,u,is_fake)
    #
    #

























    def traj_mpc(self,u0,next_state,next_gain,next_ctg,state,os,osp1,u_plan,is_fake,prop_on = False):

        use_prop = self.include_prop and prop_on
        use_srp = self.include_srp #and not (os.in_eclipse() and osp1.in_eclipse())#taken care of in dynamcis
        use_resdipole = self.include_resdipole #and not is_fake
        use_gen_dist = self.include_gendist #and not is_fake
        osmid = os.average(osp1)
        dist_torq = self.sat.gen_dist_torq()#np.zeros((3,1))
        rmat_ECI2B = rot_mat(state[3:7]).T

        nbb = (rot_mat(state[3:7]).T@normalize(osmid.B)).reshape((3,1))
        u0 = u0.reshape((self.sat.control_len,1))


        if self.MPC_remove_B:
            u0[0:self.estimator.sat.number_MTQ] = u0[0:self.estimator.sat.number_MTQ] - sat.estimator.sat.MTQ_ax_mat.T@nbb*np.ndarray.item(u0.T@sat.estimator.sat.MTQ_ax_mat@nbb)
            u0[0:self.estimator.sat.number_MTQ] = 0.99*limit_vec(u0[0:self.estimator.sat.number_MTQ],self.sat.mtqrw_max)
        unew = u0


        # print(np.round(next_gain.T@next_gain,1))
        # print(np.round(next_ctg,1))
        # next_gain *= 0
        # next_ctg *= 0
        # gain_mat = next_gain.T@next_gain
        # if self.MPC_use_ctg:
        #     gain_mat = next_ctg
        # self.sat.dynamics(state,unew, os, add_torq=np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,print_things = True)
        xp10 = self.sat.rk4(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid)
        # addl_wt_err = np.zeros((self.sat.state_len-1,self.sat.state_len-1))
        angerr = 2*math.acos(np.clip(np.ndarray.item(xp10[3:7].reshape((4,1)).T@next_state[3:7].reshape((4,1))),-1,1))*180/math.pi
        angerr = min(angerr,360-angerr)
        if angerr > 10:
            addl_wt_bare = 0*np.block([[np.eye(3)*1e3,np.zeros((3,self.sat.state_len-3))],[np.zeros((self.sat.state_len-3,self.sat.state_len))]])
            # addl_wt_err = np.block([[np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,self.sat.state_len-7))],[np.zeros((3,3)),np.eye(3)*1e12,np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]])#
            addl_wt_err = 0*np.block([[0*np.eye(3)*1e6-gain_mat[0:3,0:3],-np.eye(3)*1e2-gain_mat[0:3,3:6],np.zeros((3,self.sat.state_len-7))],[-np.eye(3)*1e2-gain_mat[3:6,0:3],np.eye(3)*1e10-gain_mat[3:6,3:6],np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]])
            addl_wt_err = np.block([[np.eye(3)*1e1,0*-np.eye(3),np.zeros((3,self.sat.state_len-7))],[0*-np.eye(3),np.eye(3)*1e4,np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]])# - gain_mat

        else:
            addl_wt_bare = 0*np.block([[np.eye(3)*1e0,np.zeros((3,self.sat.state_len-3))],[np.zeros((self.sat.state_len-3,self.sat.state_len))]])
            addl_wt_err = np.block([[np.eye(3)*1e1,0*-np.eye(3),np.zeros((3,self.sat.state_len-7))],[0*-np.eye(3),np.eye(3)*1e4,np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]])# - gain_mat
        # addl_wt_bare = np.block([[np.eye(3)*1e3,np.zeros((3,self.sat.state_len-3))],[np.zeros((self.sat.state_len-3,self.sat.state_len))]])
        # meaneig =  0.5*(np.trace(gain_mat)/(self.sat.state_len-1))
        # ratio = 0.5
        # addl_wt_err = np.block([[np.eye(3)*meaneig*2*(1-ratio),np.zeros((3,self.sat.state_len-4))],[np.zeros((3,3)),np.eye(3)*meaneig*2*ratio,np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]])#*np.eye(self.sat.state_len-1)# np.block([[np.eye(3)*1e2,np.eye(3)*1e8,np.zeros((3,self.sat.state_len-7))],[np.eye(3)*1e8,np.eye(3)*1e8,np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]]) #np.block([[-gain_mat[0:3,0:3],-gain_mat[0:3,3:6],np.zeros((3,self.sat.state_len-7))],[-gain_mat[3:6,0:3],np.eye(3)*1e12,np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]])
        # print(np.round(gain_mat + addl_wt_err,0))
        # addl_wt_err = np.block([[np.eye(3)*1e2,0*np.eye(3)*1e2,np.zeros((3,self.sat.state_len-7))],[np.eye(3)*1e2*0,np.eye(3)*1e8,np.zeros((3,self.sat.state_len-7))],[np.zeros((self.sat.state_len-7,self.sat.state_len-1))]])
        u0 = np.copy(unew)
        fnew = self.mpc_obj_func(unew,next_gain,next_ctg,xp10,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare)
        print(fnew,2*math.acos(np.ndarray.item(np.clip(next_state[3:7].reshape((4,1)).T@xp10[3:7].reshape((4,1)),-1,1)))*180/math.pi,(180.0/math.pi)*norm(next_state[0:3]-xp10[0:3]),unew.T,norm(unew),self.reduced_state_err(xp10,next_state).T)
        steps = {0:(-1,(unew,fnew),{0:(unew,fnew)})}
        self.steps = steps
        for d in range(self.MPC_barrier_cycle):
            # print("d ",d)
            self.newstep = {}
            t = self.beta0*(self.beta_ratio**d)
            if np.abs((2*self.sat.number_MTQ + 4*self.sat.number_RW)/t) <= self.fobj_tol:
                print('breaking due to small change',d)
                break
            for i in range(self.MPC_step_cycle):
                # print(i)
                #first guess
                xp1 = self.sat.rk4(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid)

                lb = self.mpc_lim_func(unew,xp1)
                # print(next_gain.T@next_gain)
                f0 = self.mpc_obj_func(unew,next_gain,next_ctg,xp1,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare)
                f = f0*t + lb
                self.newnewstep = [(unew,f0)]
                # breakpoint()
                [dxp1__dx,dxp1__du,dxp1__dtorq] = self.sat.rk4JacobiansWithTorq(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist = use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole =use_resdipole,mid_orbital_state = osmid)
                daddlwte__dui = [np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
                daddlwte__dxi = [np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.state_len)] #-1 to account for the quaternion order reduction
                daddlwtb__dui = [np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
                daddlwtb__dxi = [np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.state_len)] #-1 to account for the quaternion order reduction
                dlb__du,dlb__dx = self.mpc_lim_jac(unew,xp1)
                df0__du,df0__dx = self.mpc_obj_jac(unew,next_gain,next_ctg,xp1,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,daddlwte__dui,daddlwte__dxi,addl_wt_bare,daddlwtb__dui,daddlwtb__dxi)
                df__du = t*df0__du + dlb__du
                df__dx = t*df0__dx + dlb__dx
                fjac = df__du + (df__dx.T@dxp1__du).T
                f0jac = df0__du + (df0__dx.T@dxp1__du).T
                lbjac = dlb__du + (dlb__dx.T@dxp1__du).T

                if not self.MPC_use_hess:
                    fhess = np.eye(self.sat.control_len)
                    tmax = 0
                    mmax = 0
                    if self.sat.number_RW >0:
                        tmax = max(self.sat.RW_torq_max)
                    if self.sat.number_MTQ>0:
                        mmax = max(self.sat.MTQ_max)
                    u_step = -normalize(fjac)*2*max(tmax,mmax)
                else:
                    [_,ddxp1__dudui,_] = self.sat.rk4_u_Hessians(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist = use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole =use_resdipole,mid_orbital_state = osmid)
                    ddaddlwte__duidui = [[np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.control_len)] for k in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
                    ddaddlwte__dxidxi = [[np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.state_len)] for k in  range(self.sat.state_len)]#-1 to account for the quaternion order reduction
                    ddaddlwte__duidxi = [[np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.state_len)] for k in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
                    ddaddlwtb__duidui = [[np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.control_len)] for k in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
                    ddaddlwtb__dxidxi = [[np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.state_len)] for k in  range(self.sat.state_len)]#-1 to account for the quaternion order reduction
                    ddaddlwtb__duidxi = [[np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.state_len)] for k in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
                    ddlb__dudu, ddlb__dxdx, ddlb__dudx = self.mpc_lim_hess(unew,xp1)
                    ddf0__dudu, ddf0__dxdx, ddf0__dudx = self.mpc_obj_hess(unew,next_gain,next_ctg,xp1,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,daddlwte__dui,daddlwte__dxi,ddaddlwte__duidui,ddaddlwte__dxidxi,ddaddlwte__duidxi,addl_wt_bare,daddlwtb__dui,daddlwtb__dxi,ddaddlwtb__duidui,ddaddlwtb__dxidxi,ddaddlwtb__duidxi)
                    ddf__dudu = ddf0__dudu*t + ddlb__dudu
                    ddf__dxdx = ddf0__dxdx*t + ddlb__dxdx
                    ddf__dudx = ddf0__dudx*t + ddlb__dudx
                    f0hess = ddf0__dudu+ddf0__dudx@dxp1__du + dxp1__du.T@(ddf0__dudx.T+ddf0__dxdx@dxp1__du) + np.hstack([(df0__dx.T@j).T for j in ddxp1__dudui])
                    lbhess = ddlb__dudu+ddlb__dudx@dxp1__du + dxp1__du.T@(ddlb__dudx.T+ddlb__dxdx@dxp1__du) + np.hstack([(dlb__dx.T@j).T for j in ddxp1__dudui])
                    fhess = ddf__dudu+ddf__dudx@dxp1__du + dxp1__du.T@(ddf__dudx.T+ddf__dxdx@dxp1__du) + np.hstack([(df__dx.T@j).T for j in ddxp1__dudui])

                    assert np.allclose(fhess,fhess.T)
                    if self.MPC_use_solve:
                        u_step = -scipy.linalg.solve(fhess,fjac,assume_a='sym')
                    else:
                        w,v = np.linalg.eig(fhess)
                        # print(np.abs(np.real(w)),math.sqrt(t)*self.MPC_eigval_lim)
                        iw2 = [1/j if j>math.sqrt(t)*self.MPC_eigval_lim else 1/(math.sqrt(t)*self.MPC_eigval_lim) for j in np.abs(np.real(w))]
                        # w2 = [j if j>self.MPC_eigval_lim else self.MPC_eigval_lim for j in np.abs(w)]
                        v = np.real(v)
                        u_step = -(v@np.diagflat(iw2)@v.T@fjac).reshape((self.sat.control_len,1))

                if self.MPC_remove_B:
                    u_step[0:self.estimator.sat.number_MTQ] = u_step[0:self.estimator.sat.number_MTQ] - sat.estimator.sat.MTQ_ax_mat.T@nbb*np.ndarray.item(u_step.T@sat.estimator.sat.MTQ_ax_mat@nbb)
                exp_change = fjac.T@u_step
                best_change_est = fjac.T@u_step + 0.5*u_step.T@fhess@u_step

                # if (best_change_est>0):
                #     breakpoint()
                exp_change *= -np.sign(best_change_est)
                u_step *= -np.sign(best_change_est)

                rk4_info = (state,os,osp1,osmid,np.zeros((3,1)),use_prop,use_gen_dist,self.include_gg, use_srp,self.include_drag,use_resdipole)
                score_info = (next_gain,next_state,next_ctg,u_plan,addl_wt_err,addl_wt_bare,t)
                uprev = unew
                unew = self.linesearch(f,unew,u_step,exp_change,rk4_info,score_info,dist_torq)
                xnew = self.sat.rk4(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid)
                fnew = self.mpc_obj_func(unew,next_gain,next_ctg,xnew,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare)
                self.newstep[i] = (i,(unew,fnew),self.newnewstep)
                if norm(uprev.reshape((self.sat.control_len,1))-unew.reshape((self.sat.control_len,1))) < self.MPC_tol:
                    print('breaking due to small step ',i)
                    break
                if d == 0:
                    u1 = unew
            xnew = self.sat.rk4(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid)
            fnew = self.mpc_obj_func(unew,next_gain,next_ctg,xnew,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare)
            print(fnew,2*math.acos(np.ndarray.item(np.clip(next_state[3:7].reshape((4,1)).T@xnew[3:7].reshape((4,1)),-1,1)))*180/math.pi,(180.0/math.pi)*norm(next_state[0:3]-xnew[0:3]),unew.T,norm(unew))#,self.reduced_state_err(xnew,next_state).T)
            self.steps[d+1] = (d,(unew,fnew),self.newstep)
        if False: #optimal
            print('optimal opt test')

            wn = state[0:3].reshape((3,1))
            d = quat_mult(quat_inv(state[3:7]),next_state[3:7])
            d0 = d[0]
            dv = d[1:].reshape((3,1))
            avwt = 1
            p1 = next_state[0:3].reshape((3,1))-wn
            p2 =  np.sign(d0)*(1/4)*dv
            tq = p2/avwt + p1 #-self.sat.J@np.linalg.solve((1/48)*(wn@dv.T + dv@wn.T) + np.eye(3)*(avwt+(d0 + np.ndarray.item(dv.reshape((3,1)).T@wn.reshape((3,1)))/3)/48),((1/8)*d.T@np.vstack([-wn.T,skewsym(wn)/3 + np.eye(3)*(2-norm(wn)**2/6)-wn@wn.T/3]) + avwt*(next_state[0:3].reshape((3,1))-wn).T).T).reshape((3,1))
                 # = np.linalg.solve(,next_state[0:3].reshape((3,1))-wn +
            minu = np.vstack([limit_vec(np.cross(nbb,tq-self.sat.gen_dist_torq().reshape((3,1)))/norm(os.B),self.sat.MTQ_max),np.zeros((self.sat.control_len-self.sat.number_MTQ,1))])
            minx = self.sat.rk4(state,minu,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid)
            minf = self.mpc_obj_func(minu,next_gain,next_ctg,minx,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare)
            minang = 2*math.acos(np.ndarray.item(np.clip(next_state[3:7].reshape((4,1)).T@minx[3:7].reshape((4,1)),-1,1)))*180/math.pi
            print(minf,minang,(180.0/math.pi)*norm(next_state[0:3]-minx[0:3]),minu.T,norm(minu))
        if False: #Optimal
            testfun = lambda u : self.mpc_obj_func(u.reshape((self.sat.control_len,1)),next_gain,next_ctg, self.sat.rk4(state,u.reshape((self.sat.control_len,1)),self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid),next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare)
            # testjac = lambda ux : self.mpc_obj_jac(u.reshape((self.sat.control_len,1)),next_gain,next_ctg, self.sat.rk4(state,u.reshape((self.sat.control_len,1)),self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid),next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare,daddlwte__dui,daddlwte__dxi,addl_wt_bare,daddlwtb__dui,daddlwtb__dxi)

            lb,ub = self.sat.control_bounds()
            bnd = scipy.optimize.Bounds(lb, ub)
            res = scipy.optimize.minimize(testfun, u0, args=(), jac=None, hess=None, hessp=None, bounds=bnd)
            optx = self.sat.rk4(state,res.x,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist =use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole = use_resdipole,mid_orbital_state = osmid)

            print('optimum: ',res.fun,2*math.acos(np.ndarray.item(np.clip(next_state[3:7].reshape((4,1)).T@optx[3:7].reshape((4,1)),-1,1)))*180/math.pi,(180.0/math.pi)*norm(next_state[0:3]-optx[0:3]),res.x,norm(res.x))
        print("Ctrl:")
        # print(state.T)
        print(xnew.T)
        [dxnew__dx,dxnew__du,dxnew__dtorq] = self.sat.rk4JacobiansWithTorq(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist = use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole =use_resdipole,mid_orbital_state = osmid)

        lb = self.mpc_lim_func(unew,xnew)

        dlb__du,dlb__dx = self.mpc_lim_jac(unew,xnew)
        df0__du,df0__dx = self.mpc_obj_jac(unew,next_gain,next_ctg,xnew,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,daddlwte__dui,daddlwte__dxi,addl_wt_bare,daddlwtb__dui,daddlwtb__dxi)
        df__du = t*df0__du + dlb__du
        df__dx = t*df0__dx + dlb__dx
        fjac = df__du + (df__dx.T@dxp1__du).T
        f0jac = df0__du + (df0__dx.T@dxp1__du).T
        lbjac = dlb__du + (dlb__dx.T@dxp1__du).T
        if self.MPC_use_hess:
            [_,xnewhess,_] = self.sat.rk4_u_Hessians(state,unew,self.update_period,os,osp1,add_torq = np.zeros((3,1)),use_prop = use_prop,use_gen_dist = use_gen_dist,use_gg = self.include_gg, use_srp = use_srp, use_drag = self.include_drag,use_resdipole =use_resdipole,mid_orbital_state = osmid)
            ddlb__dudu, ddlb__dxdx, ddlb__dudx = self.mpc_lim_hess(unew,xnew)
            ddf0__dudu, ddf0__dxdx, ddf0__dudx = self.mpc_obj_hess(unew,next_gain,next_ctg,xnew,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,daddlwte__dui,daddlwte__dxi,ddaddlwte__duidui,ddaddlwte__dxidxi,ddaddlwte__duidxi,addl_wt_bare,daddlwtb__dui,daddlwtb__dxi,ddaddlwtb__duidui,ddaddlwtb__dxidxi,ddaddlwtb__duidxi)
            ddf__dudu = ddf0__dudu*t + ddlb__dudu
            ddf__dxdx = ddf0__dxdx*t + ddlb__dxdx
            ddf__dudx = ddf0__dudx*t + ddlb__dudx
            f0hess = ddf0__dudu+ddf0__dudx@dxnew__du + dxnew__du.T@(ddf0__dudx.T+ddf0__dxdx@dxnew__du) + np.hstack([(df0__dx.T@j).T for j in xnewhess])
            lbhess = ddlb__dudu+ddlb__dudx@dxnew__du + dxnew__du.T@(ddlb__dudx.T+ddlb__dxdx@dxnew__du) + np.hstack([(dlb__dx.T@j).T for j in xnewhess])
            fhess = ddf__dudu+ddf__dudx@dxnew__du + dxnew__du.T@(ddf__dudx.T+ddf__dxdx@dxnew__du) + np.hstack([(df__dx.T@j).T for j in xnewhess])

        #     print(fhess/t)
        #     print(f0hess)
        #     print(np.linalg.eig(fhess/t))
        #     print(np.linalg.eig(f0hess))
        # print(fjac.T/t)
        # print(f0jac.T)
        # print(fnew+lb/t)
        # print(fnew)
        self.prev_mpc_score = fnew
        return unew

    def linesearch(self,fbase,ubase,u_step,exp_change,rk4_info,score_info,dist_torq,use_constraints = True):
        (state,os,osp1,osmid,add_torq,use_prop,use_gen_dist,use_gg, use_srp, use_drag ,use_resdipole) = rk4_info
        (next_gain,next_state,next_ctg,u_plan,addl_wt_err,addl_wt_bare,t)  = score_info
        line_search_step = 0
        if use_constraints:
            ls_mult = self.limit_ustep_with_constraints(ubase,state,u_step,os)
        else:
            ls_mult = 1

        while True:
            # np.max(self.constraints(ubase + u_step*ls_mult,np.zeros((self.sat.state_len,1))))
            u_ls = ubase + u_step*ls_mult
            xp1_ls = self.sat.rk4(state,u_ls,self.update_period,os,osp1,add_torq = add_torq,use_prop = use_prop,use_gen_dist = use_gen_dist,use_gg = use_gg, use_srp = use_srp, use_drag = use_drag,use_resdipole =use_resdipole,mid_orbital_state = osmid)
            if np.any(self.constraints(u_ls,xp1_ls)>0) and use_constraints:
                ls_mult *= self.MPC_line_search_alpha
                continue
            f_ls0 = self.mpc_obj_func(u_ls,next_gain,next_ctg,xp1_ls,next_state,os,osp1,dist_torq,u_plan,addl_wt_err,addl_wt_bare)
            if use_constraints:
                f_ls = self.mpc_lim_func(u_ls,xp1_ls)*use_constraints + t*f_ls0
            else:
                f_ls = t*f_ls0

            self.newnewstep += [(u_ls,f_ls0)]
            sufficient_decrease = bool(f_ls <= (fbase + ls_mult*self.c1*exp_change)) #Wolfe's 1st condition
            if sufficient_decrease:
                return u_ls
            line_search_step += 1
            ls_mult *= self.MPC_line_search_alpha
            if line_search_step > self.MPC_line_search_cycle or norm(u_step*ls_mult) < self.MPC_tol:
                return ubase

    def constraints(self,u,xp1):
        if self.sat.number_RW>0:
            RW_h = xp1[-self.sat.number_RW:,:].reshape((self.sat.number_RW,1))
        else:
            RW_h = np.zeros((0,1))
        l0,u0 = self.sat.control_bounds()
        limit_pos = u - u0
        limit_neg = -u + l0
        RWh_pos = RW_h.reshape((self.sat.number_RW,1)) - np.array(self.sat.RW_saturation).reshape((self.sat.number_RW,1))
        RWh_neg = -RW_h.reshape((self.sat.number_RW,1)) - np.array(self.sat.RW_saturation).reshape((self.sat.number_RW,1))
        constraints = np.vstack([limit_pos,limit_neg,RWh_pos,RWh_neg])

        return constraints
    def constraints_w_jac(self,u,xp1):
        constraints = self.constraints(u,xp1)
        dcon__du = np.block([\
                            [np.eye(self.sat.control_len)],\
                            [-np.eye(self.sat.control_len)],\
                            [np.zeros((2*self.sat.number_RW,self.sat.control_len))]\
                            ])
        dcon__dx = np.block([\
                            [np.zeros((2*self.sat.control_len,self.sat.state_len))],\
                            [np.zeros((self.sat.number_RW,7)), np.eye((self.sat.number_RW))],\
                            [np.zeros((self.sat.number_RW,7)), -np.eye((self.sat.number_RW))]\
                            ])
        return constraints, dcon__du, dcon__dx
    def constraints_w_jac_hess(self,u,xp1):
        constraints = self.constraints(u,xp1)
        dcon__du = np.block([\
                            [np.eye(self.sat.control_len)],\
                            [-np.eye(self.sat.control_len)],\
                            [np.zeros((2*self.sat.number_RW,self.sat.control_len))]\
                            ])
        dcon__dx = np.block([\
                            [np.zeros((2*self.sat.control_len,self.sat.state_len))],\
                            [np.zeros((self.sat.number_RW,7)), np.eye((self.sat.number_RW))],\
                            [np.zeros((self.sat.number_RW,7)), -np.eye((self.sat.number_RW))]\
                            ])
        ddcon__dudu = [np.zeros((self.sat.control_len,self.sat.control_len)) for j in range(constraints.size)]
        ddcon__dxdx = [np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(constraints.size)]
        ddcon__dudx = [np.zeros((self.sat.control_len,self.sat.state_len)) for j in range(constraints.size)]
        return constraints,dcon__du, dcon__dx,ddcon__dudu,ddcon__dxdx,ddcon__dudx

    def limit_ustep_with_constraints(self,u0,x0,ustep,os):
        u_mtq,u_RW,u_mag = self.sat.sort_control_vec(u0+ustep)
        if not self.sat.ignore_RW_angle:
            RW_h0 = x0[7+self.sat.number_RW:7+2*self.sat.number_RW,:].reshape((self.sat.number_RW,1))
            raise ValueError("accounting for RW angle not implemented")
        else:
            RW_h0 = x0[7:7+self.sat.number_RW,:].reshape((self.sat.number_RW,1))
        Bbody = rot_mat(x0[3:7]).T@os.B
        torq = np.cross(self.sat.MTQ_ax_mat@u_mtq,Bbody) + self.sat.RW_ax_mat@u_RW + self.sat.magic_ax_mat@u_mag
        w = x0[0:3]
        ang_mom = self.sat.J@w + self.sat.RW_ax_mat@RW_h0
        constraints,dcon__du,dcon__dxp = self.constraints_w_jac(u0+ustep,np.vstack([np.zeros((7,1)),RW_h0 + self.update_period*(-u_RW-np.diagflat(self.sat.RW_J)@self.sat.RW_ax_mat.T@self.sat.invJ_noRW@(-np.cross(w,ang_mom) + torq ))]))
        du__dustep = np.eye(self.sat.control_len)
        dxp__du = np.zeros((self.sat.state_len,self.sat.control_len))
        dang_mom__du = np.zeros((3,self.sat.control_len))
        dtorq__du = np.hstack([-skewsym(Bbody)@self.sat.MTQ_ax_mat,self.sat.RW_ax_mat,self.sat.magic_ax_mat])
        if self.sat.number_RW>0:
            # breakpoint()
            dxp__du[-self.sat.number_RW:,:] += -np.hstack([np.zeros((self.sat.number_RW,self.sat.number_MTQ)),np.eye(self.sat.number_RW),np.zeros((self.sat.number_RW,self.sat.number_magic))]) - np.diagflat(self.sat.RW_J)@self.sat.RW_ax_mat.T@self.sat.invJ_noRW@(-skewsym(w)@dang_mom__du + dtorq__du)
        dcon__dustep = (dcon__du + dcon__dxp@dxp__du)@du__dustep


        viol = np.maximum(constraints,0)
        dviol = dcon__dustep@ustep
        # dviol = np.maximum(dviol,0)
        dustep_scale = np.hstack([1 - (viol[j,:].item()/(dviol[j,:]).item()) if dviol[j,0]!=0 else 1 for j in range(viol.size)])#scipy.linalg.solve(dcon_dustep,viol)
        # breakpoint()
        dustep_scale = np.where(viol==0, 1, dustep_scale)
        dustep_scale = np.where(dviol==0, 1, dustep_scale)
        dustep_scale = np.maximum(dustep_scale,0)
        mult = np.amax(dustep_scale)
        mult = np.minimum(1,mult)
        if mult<1:
            breakpoint()
        # if np.any(constraints>0):
        #     breakpoint()
        return mult

    def mpc_lim_func(self,u,xp1):
        constraints = self.constraints(u,xp1)
        if np.any(constraints>0):
            breakpoint()
            raise ValueError("constraints should be less than zero.. how did this happen???")
        constraints = np.minimum(constraints,-self.constraint_replacement_val)#np.array([min(j,-self.constraint_replacement_val) for j in constraints])
        phi = -1*np.sum(np.log(-constraints))
        return phi
    def mpc_lim_jac(self,u,xp1):
        constraints,dcon__du,dcon__dx = self.constraints_w_jac(u,xp1)
        if np.any(constraints>0):
            raise ValueError("constraints should be less than zero.. how did this happen???")
        constraints = np.minimum(constraints,-self.constraint_replacement_val)#p.array([min(j,-self.constraint_replacement_val) for j in constraints])
        # phi = -1*sum(math.log(-constraints))
        dphi__du = ((-1/constraints).T@dcon__du).T
        dphi__dx = ((-1/constraints).T@dcon__dx).T
        return dphi__du, dphi__dx
    def mpc_lim_hess(self,u,xp1):
        constraints, dcon__du, dcon__dx,ddcon__dudu,ddcon__dxdx,ddcon__dudx = self.constraints_w_jac_hess(u,xp1)
        if np.any(constraints>0):
            raise ValueError("constraints should be less than zero.. how did this happen???")
        constraints = np.minimum(constraints,-self.constraint_replacement_val)#np.array([min(j,-self.constraint_replacement_val) for j in constraints])
        # phi = -1*sum(math.log(-constraints))
        dphi__du = ((-1/constraints).T@dcon__du).T
        dphi__dx = ((-1/constraints).T@dcon__dx).T
        ddphi__dudu = dcon__du.T@np.diagflat(1/(constraints*constraints))@dcon__du + sum([(-1/np.ndarray.item(constraints[j,:]))*ddcon__dudu[j] for j in range(constraints.size)])
        ddphi__dxdx = dcon__dx.T@np.diagflat(1/(constraints*constraints))@dcon__dx + sum([(-1/np.ndarray.item(constraints[j,:]))*ddcon__dxdx[j] for j in range(constraints.size)])
        ddphi__dudx = dcon__du.T@np.diagflat(1/(constraints*constraints))@dcon__dx  + sum([(-1/np.ndarray.item(constraints[j,:]))*ddcon__dudx[j] for j in range(constraints.size)])
        return ddphi__dudu,ddphi__dxdx,ddphi__dudx

    def mpc_obj_func(self,u,gain_p1,ctg_p1,xp1,xdes,os,osp1,dist_torq = None,u_plan = None,addl_wt_err = None,addl_wt_bare = None):
        if addl_wt_err is None:
            addl_wt_err = np.zeros((self.sat.state_len-1,self.sat.state_len-1)) #-1 to account for the quaternion order reduction
        if addl_wt_bare is None:
            addl_wt_bare = np.zeros((self.sat.state_len,self.sat.state_len)) #-1 to account for the quaternion order reduction
        if dist_torq is None:
            dist_torq = np.zeros((3,1))

        err = self.reduced_state_err(xp1,xdes)
        if self.tracking_LQR_formulation==1:
            aug = np.vstack([xp1,np.ones((1,1))])
            if self.MPC_use_ctg:
                f = 0.5*np.ndarray.item(aug.T@(ctg_p1)@aug + err.T@addl_wt_err@err+ xp1.T@addl_wt_bare@xp1)
            else:
                f = 0.5*np.ndarray.item(aug.T@(gain_p1.T@gain_p1)@aug  + err.T@addl_wt_err@err + xp1.T@addl_wt_bare@xp1)
        elif self.tracking_LQR_formulation==0:
            if self.MPC_use_ctg:
                f = 0.5*np.ndarray.item(err.T@(addl_wt_err + ctg_p1)@err + xp1.T@addl_wt_bare@xp1)
            else:
                f = 0.5*np.ndarray.item(err.T@(addl_wt_err + gain_p1.T@gain_p1)@err + xp1.T@addl_wt_bare@xp1)

        elif self.tracking_LQR_formulation==2:
            errt = np.vstack([err,dist_torq])
            addl_wt_err_t = np.block([[addl_wt_err,np.zeros((self.sat.state_len-1,3))],[np.zeros((3,self.sat.state_len-1+3))]])
            if self.MPC_use_ctg:
                f = 0.5*np.ndarray.item(errt.T@(addl_wt_err_t + ctg_p1)@errt + xp1.T@addl_wt_bare@xp1)
            else:
                f = 0.5*np.ndarray.item(errt.T@(addl_wt_err_t + gain_p1.T@gain_p1)@errt + xp1.T@addl_wt_bare@xp1)

        if self.MPC_weight_control_change_from_prev and not np.any(np.isnan(self.prev_control)):
            du = u-self.prev_control
            f += 0.5*np.ndarray.item(du.T@self.control_wt@du)
        elif not self.MPC_weight_control_change_from_prev:
            f += 0.5*np.ndarray.item(u.T@self.control_wt@u)

        if self.MPC_weight_control_change_from_plan and u_plan is not None:
            du = u-u_plan
            f += 0.5*np.ndarray.item(du.T@self.control_diff_from_plan_wt@du)
        return f
    def mpc_obj_jac(self,u,gain_p1,ctg_p1,xp1,xdes,os,osp1,dist_torq = None,u_plan = None,addl_wt_err = None,daddlwte__dui = None,daddlwte__dxi = None,addl_wt_bare = None,daddlwtb__dui = None,daddlwtb__dxi = None):
        if addl_wt_err is None:
            addl_wt_err = np.zeros((self.sat.state_len-1,self.sat.state_len-1)) #-1 to account for the quaternion order reduction
        if daddlwte__dui is None:
            daddlwte__dui = [np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
        if daddlwte__dxi is None:
            daddlwte__dxi = [np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.state_len)] #-1 to account for the quaternion order reduction

        if addl_wt_bare is None:
            addl_wt_bare = np.zeros((self.sat.state_len,self.sat.state_len)) #-1 to account for the quaternion order reduction
        if daddlwtb__dui is None:
            daddlwtb__dui = [np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
        if daddlwtb__dxi is None:
            daddlwtb__dxi = [np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.state_len)] #-1 to account for the quaternion order reduction
        if dist_torq is None:
            dist_torq = np.zeros((3,1))
        err = self.reduced_state_err(xp1,xdes)
        derr__dx = self.reduced_state_err_jac(xp1,xdes)
        if self.tracking_LQR_formulation==1:
            aug = np.vstack([xp1,np.ones((1,1))])
            df__du = 0.5*np.vstack([err.T@j@err for j in daddlwte__dui]) + 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dui])
            if self.MPC_use_ctg:
                df__dx = derr__dx.T@(addl_wt_err)@err + np.block([[np.eye(self.sat.state_len),np.zeros((self.sat.state_len,1))]])@ctg_p1@aug + addl_wt_bare@xp1 + 0.5*np.vstack([err.T@j@err for j in daddlwte__dxi])+ 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dxi])
            else:
                df__dx = derr__dx.T@(addl_wt_err)@err+ np.block([[np.eye(self.sat.state_len),np.zeros((self.sat.state_len,1))]])@gain_p1.T@gain_p1@aug + addl_wt_bare@xp1 + 0.5*np.vstack([err.T@j@err for j in daddlwte__dxi])+ 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dxi])
        elif self.tracking_LQR_formulation==0:
            df__du = 0.5*np.vstack([err.T@j@err for j in daddlwte__dui]) + 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dui])
            if self.MPC_use_ctg:
                df__dx = derr__dx.T@(addl_wt_err + ctg_p1)@err + addl_wt_bare@xp1 + 0.5*np.vstack([err.T@j@err for j in daddlwte__dxi])+ 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dxi])
            else:
                df__dx = derr__dx.T@(addl_wt_err + gain_p1.T@gain_p1)@err + addl_wt_bare@xp1 + 0.5*np.vstack([err.T@j@err for j in daddlwte__dxi])+ 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dxi])
        elif self.tracking_LQR_formulation==2:
            errt = np.vstack([err,dist_torq])
            addl_wt_err_t = np.block([[addl_wt_err,np.zeros((self.sat.state_len-1,3))],[np.zeros((3,self.sat.state_len-1+3))]])
            derr__dx_t = np.vstack([derr__dx,np.zeros((3,self.sat.state_len))])
            df__du = 0.5*np.vstack([err.T@j@err for j in daddlwte__dui]) + 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dui])
            if self.MPC_use_ctg:
                df__dx = derr__dx_t.T@(addl_wt_err_t + ctg_p1)@errt + addl_wt_bare@xp1 + 0.5*np.vstack([err.T@j@err for j in daddlwte__dxi])+ 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dxi])
            else:
                df__dx = derr__dx_t.T@(addl_wt_err_t + gain_p1.T@gain_p1)@errt + addl_wt_bare@xp1 + 0.5*np.vstack([err.T@j@err for j in daddlwte__dxi])+ 0.5*np.vstack([xp1.T@j@xp1 for j in daddlwtb__dxi])
        if self.MPC_weight_control_change_from_prev and not np.any(np.isnan(self.prev_control)):
            du = u.reshape((self.sat.control_len,1))-self.prev_control.reshape((self.sat.control_len,1))
            df__du += self.control_wt@du.reshape((self.sat.control_len,1))
        elif not self.MPC_weight_control_change_from_prev:
            df__du += self.control_wt@u.reshape((self.sat.control_len,1))

        if self.MPC_weight_control_change_from_plan and u_plan is not None:
            du = u-u_plan
            df__du += self.control_diff_from_plan_wt@du
        return df__du, df__dx
    def mpc_obj_hess(self,u,gain_p1,ctg_p1,xp1,xdes,os,osp1,dist_torq = None,u_plan = None,addl_wt_err = None,daddlwte__dui = None,daddlwte__dxi = None,ddaddlwte__duidui = None,ddaddlwte__dxidxi = None,ddaddlwte__duidxi = None,addl_wt_bare = None,daddlwtb__dui = None,daddlwtb__dxi = None,ddaddlwtb__duidui = None,ddaddlwtb__dxidxi = None,ddaddlwtb__duidxi = None):

        if addl_wt_err is None:
            addl_wt_err = np.zeros((self.sat.state_len-1,self.sat.state_len-1)) #-1 to account for the quaternion order reduction
        if daddlwte__dui is None:
            daddlwte__dui = [np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
        if daddlwte__dxi is None:
            daddlwte__dxi = [np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.state_len)] #-1 to account for the quaternion order reduction
        if ddaddlwte__duidui is None:
            ddaddlwte__duidui = [[np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.control_len)] for k in range(self.sat.control_len)] #-1 to account for the quaternion order reduction
        if ddaddlwte__dxidxi is None:
            ddaddlwte__dxidxi = [[np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.state_len)] for k in  range(self.sat.state_len)]#-1 to account for the quaternion order reduction
        if ddaddlwte__duidxi is None:
            ddaddlwte__duidxi = [[np.zeros((self.sat.state_len-1,self.sat.state_len-1)) for j in range(self.sat.state_len)] for k in range(self.sat.control_len)] #-1 to account for the quaternion order reduction

        if addl_wt_bare is None:
            addl_wt_bare = np.zeros((self.sat.state_len,self.sat.state_len)) # to account for the quaternion order reduction
        if daddlwtb__dui is None:
            daddlwtb__dui = [np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.control_len)] # to account for the quaternion order reduction
        if daddlwtb__dxi is None:
            daddlwtb__dxi = [np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.state_len)] # to account for the quaternion order reduction
        if ddaddlwtb__duidui is None:
            ddaddlwtb__duidui = [[np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.control_len)] for k in range(self.sat.control_len)] # to account for the quaternion order reduction
        if ddaddlwtb__dxidxi is None:
            ddaddlwtb__dxidxi = [[np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.state_len)] for k in  range(self.sat.state_len)]# to account for the quaternion order reduction
        if ddaddlwtb__duidxi is None:
            ddaddlwtb__duidxi = [[np.zeros((self.sat.state_len,self.sat.state_len)) for j in range(self.sat.state_len)] for k in range(self.sat.control_len)] # to account for the quaternion order reduction
        if dist_torq is None:
            dist_torq = np.zeros((3,1))
        err = self.reduced_state_err(xp1,xdes)
        derr__dx = self.reduced_state_err_jac(xp1,xdes)
        dderr__dxdxi = self.reduced_state_err_hess(xp1,xdes)
        if self.tracking_LQR_formulation==1:
            augmat = np.block([[np.eye(self.sat.state_len),np.zeros((self.sat.state_len,1))]])
            aug = np.vstack([xp1,np.ones((1,1))])
            ddf__dudu = 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__duidui]) + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__duidui])
            ddf__dudx = 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__duidxi]) + np.vstack([(derr__dx.T@j@err).T for j in daddlwte__dui]) + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__duidxi]) + np.vstack([(j@xp1).T for j in daddlwtb__dui])
            if self.MPC_use_ctg:
                ddf__dxdx = derr__dx.T@(addl_wt_err)@derr__dx \
                            + augmat@ctg_p1@augmat.T \
                            + 2*np.vstack([(derr__dx.T@(j)@err).T for j in daddlwte__dxi]) \
                            + np.hstack([j.T@(addl_wt_err)@err for j in dderr__dxdxi]) \
                            + 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__dxidxi]) \
                            + (addl_wt_bare) \
                            + 2*np.vstack([(j@xp1).T for j in daddlwtb__dxi]) \
                            + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__dxidxi])
            else:
                ddf__dxdx = derr__dx.T@(addl_wt_err)@derr__dx \
                            + augmat@gain_p1.T@gain_p1@augmat.T \
                            + 2*np.vstack([(derr__dx.T@(j)@err).T for j in daddlwte__dxi]) \
                            + np.hstack([j.T@(addl_wt_err)@err for j in dderr__dxdxi]) \
                            + 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__dxidxi]) \
                            + (addl_wt_bare) \
                            + 2*np.vstack([(j@xp1).T for j in daddlwtb__dxi]) \
                            + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__dxidxi])


        elif self.tracking_LQR_formulation==0:
            ddf__dudu = 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__duidui]) + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__duidui])
            ddf__dudx = 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__duidxi]) + np.vstack([(derr__dx.T@j@err).T for j in daddlwte__dui]) + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__duidxi]) + np.vstack([(j@xp1).T for j in daddlwtb__dui])
            if self.MPC_use_ctg:
                ddf__dxdx = derr__dx.T@(addl_wt_err + ctg_p1)@derr__dx \
                            + 2*np.vstack([(derr__dx.T@(j)@err).T for j in daddlwte__dxi]) \
                            + np.hstack([j.T@(addl_wt_err + ctg_p1)@err for j in dderr__dxdxi]) \
                            + 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__dxidxi]) \
                            + (addl_wt_bare) \
                            + 2*np.vstack([(j@xp1).T for j in daddlwtb__dxi]) \
                            + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__dxidxi])
            else:
                ddf__dxdx = derr__dx.T@(addl_wt_err + gain_p1.T@gain_p1)@derr__dx \
                            + 2*np.vstack([(derr__dx.T@(j)@err).T for j in daddlwte__dxi]) \
                            + np.hstack([j.T@(addl_wt_err + gain_p1.T@gain_p1)@err for j in dderr__dxdxi]) \
                            + 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__dxidxi]) \
                            + (addl_wt_bare) \
                            + 2*np.vstack([(j@xp1).T for j in daddlwtb__dxi]) \
                            + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__dxidxi])
        elif self.tracking_LQR_formulation==2:
            errt = np.vstack([err,dist_torq])
            addl_wt_err_t = np.block([[addl_wt_err,np.zeros((self.sat.state_len-1,3))],[np.zeros((3,self.sat.state_len-1+3))]])
            derr__dx_t = np.vstack([derr__dx,np.zeros((3,self.sat.state_len))])
            ddf__dudu = 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__duidui]) + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__duidui])
            ddf__dudx = 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__duidxi]) + np.vstack([(derr__dx.T@j@err).T for j in daddlwte__dui]) + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__duidxi]) + np.vstack([(j@xp1).T for j in daddlwtb__dui])
            dderr__dxdxi_t = [np.vstack([j,np.zeros((3,self.sat.state_len))]) for j in dderr__dxdxi ]
            if self.MPC_use_ctg:
                ddf__dxdx = derr__dx_t.T@(addl_wt_err_t + ctg_p1)@derr__dx_t \
                            + 2*np.vstack([(derr__dx.T@(j)@err).T for j in daddlwte__dxi]) \
                            + np.hstack([j.T@(addl_wt_err_t + ctg_p1)@errt for j in dderr__dxdxi_t]) \
                            + 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__dxidxi]) \
                            + (addl_wt_bare) \
                            + 2*np.vstack([(j@xp1).T for j in daddlwtb__dxi]) \
                            + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__dxidxi])
            else:
                ddf__dxdx = derr__dx_t.T@(addl_wt_err_t + gain_p1.T@gain_p1)@derr__dx_t \
                            + 2*np.vstack([(derr__dx.T@(j)@err).T for j in daddlwte__dxi]) \
                            + np.hstack([j.T@(addl_wt_err_t + gain_p1.T@gain_p1)@errt for j in dderr__dxdxi_t]) \
                            + 0.5*np.block([[err.T@j@err for j in k] for k in ddaddlwte__dxidxi]) \
                            + (addl_wt_bare) \
                            + 2*np.vstack([(j@xp1).T for j in daddlwtb__dxi]) \
                            + 0.5*np.block([[xp1.T@j@xp1 for j in k] for k in ddaddlwtb__dxidxi])


        if self.MPC_weight_control_change_from_prev and not np.any(np.isnan(self.prev_control)):
            du = u-self.prev_control
            ddf__dudu += self.control_wt
        elif not self.MPC_weight_control_change_from_prev:
            ddf__dudu += self.control_wt

        if self.MPC_weight_control_change_from_plan and u_plan is not None:
            du = u-u_plan
            ddf__dudu += self.control_diff_from_plan_wt
        return ddf__dudu, ddf__dxdx, ddf__dudx
