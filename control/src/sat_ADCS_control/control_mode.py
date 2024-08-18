import numpy as np
import scipy
import random
import pytest
from sat_ADCS_helpers import *
import warnings
# from flight_adcs.flight_utils.sat_helpers import *
# import sim_adcs.sim_helpers as sim_helpers#TODO: eliminate the need for this?
import math
import copy

"""
to update from control folder

cd ../control && \
python3.10 -m build && \
pip3.10 install ./dist/sat_ADCS_control-0.0.1.tar.gz && \
cd ../control
"""


class Goals:
    def __init__(self,control_mode_dict=None,eci_vec_dict=None,sat_vec_dict=None,default_mode = None,default_point_mode = None, default_eci_vec = None, default_sat_vec = None):
        # print(control_mode_dict)
        #TODO filtering on added dictionaries if they contain the same timestep repeated
        self.baseline_default_control_mode = GovernorMode.SIMPLE_BDOT
        self.baseline_default_pointing_goal_mode = PointingGoalVectorMode.NADIR
        self.baseline_default_eci_pointing_vector = np.array([0, 0, -1])
        self.baseline_default_satellite_pointing_vector = np.array([1,0,0])
        if control_mode_dict is None:
            control_mode_dict = {}
        if eci_vec_dict is None:
            eci_vec_dict = {}
        if sat_vec_dict is None:
            sat_vec_dict = {}
        self.control_modes = copy.deepcopy(dict(control_mode_dict))
        self.eci_vecs = copy.deepcopy(dict(eci_vec_dict))
        self.sat_vecs = copy.deepcopy(dict(sat_vec_dict))
        if default_mode is None:
            self.default_control_mode = self.baseline_default_control_mode
        else:
            self.default_control_mode = default_mode

        if default_point_mode is None:
            self.default_pointing_goal_mode = self.baseline_default_pointing_goal_mode
        else:
            self.default_pointing_goal_mode = default_point_mode

        if default_eci_vec is None:
            self.default_eci_pointing_vector = self.baseline_default_eci_pointing_vector
        else:
            self.default_eci_pointing_vector = default_eci_vec

        if default_sat_vec is None:
            self.default_satellite_pointing_vector = self.baseline_default_satellite_pointing_vector
        else:
            self.default_satellite_pointing_vector = default_sat_vec

    def add_goals(self,new_goals): #if there is a discrepancy, updates with the new one.
        #TODO: not tested!
        self.control_modes |=  new_goals.control_modes
        self.eci_vecs |= new_goals.eci_vecs
        self.sat_vecs |= new_goals.sat_vecs
        if (new_goals.default_control_mode is not self.baseline_default_control_mode) or (self.default_control_mode is self.baseline_default_control_mode) :
            self.default_control_mode = new_goals.default_control_mode
        if (new_goals.default_pointing_goal_mode is not self.baseline_default_pointing_goal_mode) or (self.default_pointing_goal_mode is self.baseline_default_pointing_goal_mode) :
            self.default_pointing_goal_mode = new_goals.default_pointing_goal_mode
        if (new_goals.default_eci_pointing_vector is not self.baseline_default_eci_pointing_vector) or (self.default_eci_pointing_vector is self.baseline_default_eci_pointing_vector) :
            self.default_eci_pointing_vector = new_goals.default_eci_pointing_vector
        if (new_goals.default_satellite_pointing_vector is not self.baseline_default_satellite_pointing_vector) or (self.default_satellite_pointing_vector is self.baseline_default_satellite_pointing_vector) :
            self.default_satellite_pointing_vector = new_goals.default_satellite_pointing_vector

    def clear_past_goals(self,j2000_line):
        #TODO: not tested!
        self.control_modes = {x:self.control_modes[x] for x in self.control_modes.keys() if x>=j2000_line}
        self.eci_vecs = {x:self.eci_vecs[x] for x in self.eci_vecs.keys() if x>=j2000_line}
        self.sat_vecs = {x:self.sat_vecs[x] for x in self.sat_vecs.keys() if x>=j2000_line}

    def get_pointing_info(self,orbit,for_TP=False,quatmode = 0):
        if isinstance(orbit,Orbital_State):
            t = orbit.J2000
            key = sorted(self.eci_vecs.keys())
            key_array = np.array(key)
            times_list = np.where(t>=key_array)[0]
            if len(times_list) == 0:
                mode = self.default_pointing_goal_mode
                vec = self.default_eci_pointing_vector
            else:
                res = self.eci_vecs[key[times_list[-1]]]
                if isinstance(res,PointingGoalVectorMode):
                    mode = res
                    vec = np.nan*np.zeros(3)
                else:
                    (mode,vec) = self.eci_vecs[key[times_list[-1]]]
            if mode in fullQuatSpecifiedList:
                qgoal = vec3_to_quat(pointing_goal_vec_finder_times(mode, vec, orbit,quatmode),quatmode)
                return state_goal(goal_q=qgoal)
            else:
                qgoal = None
            pg = pointing_goal_vec_finder_times(mode, vec, orbit,quatmode)

            key = sorted(self.sat_vecs.keys())
            key_array = np.array(key)
            times_list = np.where(t>=key_array)[0]
            if len(times_list) == 0:
                sv = self.default_satellite_pointing_vector
            else:
                sv = self.sat_vecs[key[times_list[-1]]]
            return state_goal(u=pg,v=sv) #pg,sv
        elif isinstance(orbit,Orbit):
            if not for_TP:
                return {j:(self.get_pointing_info(orbit.states[j],quatmode=quatmode)) for j in orbit.times}
            else:
                key = sorted(self.eci_vecs.keys())
                key_array = np.array(key)
                orbkeys = orbit.times
                times_list_list = [np.where(orbit.states[j].J2000>=key_array)[0] for j in orbkeys]
                point_goals_list = [self.eci_vecs[key[j[-1]]] if len(j)>0 else (self.default_pointing_goal_mode,self.default_eci_pointing_vector) for j in times_list_list]
                modes = [j[0] for j in point_goals_list]
                vecs = [j[1] for j in point_goals_list]

                eci_vecs = [pointing_goal_vec_finder_times(modes[j], vecs[j], orbit.states[orbkeys[j]]) for j in range(len(times_list_list))]
                full_orientation_commands = [j in fullQuatSpecifiedList for j in modes]
                sv = [self.get_pointing_info(orbit.states[j],quatmode=quatmode).body_vec for j in orbit.states.keys()]
                if any(full_orientation_commands):
                    pg = [vec3_to_quat(eci_vecs[j],quatmode) if full_orientation_commands[j] else np.vstack([[np.nan],eci_vecs[j]]) for j in range(len(eci_vecs))]
                else:
                    pg = eci_vecs
                return pg,sv
                # breakpoint()
            # t = orbit.times
            # pg_res = [self.point_goal[np.where(np.ndarray.item(j)>=sorted(self.point_goal.keys()))[0][-1]] for j in t]
            # sv = [self.sat_vecs[np.where(j>=sorted(self.sat_vecs.keys()))[0][-1]] for j in t]
            # pg = [pointing_goal_vec_finder_times(pg_res[j][0], pg_res[j][1], orbit.states[orbit.times[j]].J2000, orbit.states[orbit.times[j]].R, orbit.states[orbit.times[j]].V, orbit.states[orbit.times[j]].S) for j in range(len(orbit.times))]

        else:
            raise ValueError("must be orbit or orbital state")

    def get_control_mode(self,t):
        if isinstance(t,Orbital_State):
            t = t.J2000
        key = sorted(self.control_modes.keys())
        key_array = np.array(key)
        times_list = np.where(t>=key_array)[0]
        # print(times_list)
        # print(np.where(t>=key_array))
        # print(self.control_modes)
        if len(times_list) == 0:
            return self.default_control_mode
        else:
            return self.control_modes[key[times_list[-1]]]

    def get_next_ctrlmode_change_time(self,t):
        #returns Nan if no goal in future
        if isinstance(t,Orbital_State):
            t = t.J2000
        key = sorted(self.control_modes.keys())
        key_array = np.array(key)
        times_list = np.where(t<key_array)[0]
        # print(times_list)
        # print(np.where(t>=key_array))
        # print(self.control_modes)
        if len(times_list) == 0:
            return np.nan
        else:
            return key[times_list[0]]

class state_goal:
    def __init__(self,goal_q=None,goal_w=None,goal_extra=None,u=None,v=None):
        if goal_q is None:
            goal_q = np.nan*zeroquat
        if goal_w is None:
            goal_w = np.zeros(3)
        if goal_extra is None:
            goal_extra = np.zeros(0)
        if u is None:
            u = np.nan*np.zeros(3)
        if v is None:
            v = np.nan*np.zeros(3)
        self.state = np.concatenate([goal_w.copy(),goal_q.copy(),goal_extra.copy()])
        self.eci_vec = u.copy()
        self.body_vec = v.copy()

    def copy(self):
        qg = None
        if not np.any(np.isnan(self.state[3:7])):
            qg = self.state[3:7].copy()
        wg = None
        if np.any(np.abs(self.state[0:3])>0):
            wg = self.state[0:3].copy()
        if np.any(np.isnan(self.eci_vec)):
            u = np.zeros(3)
        else:
            u = self.eci_vec.copy()
        if np.any(np.isnan(self.body_vec)):
            v = np.zeros(3)
        else:
            v = self.body_vec.copy()
        if self.state.size>7:
            eg = self.state[7:].copy()
        else:
            eg = None
        return state_goal(qg,wg,eg,u,v)


class Params:
    def __init__(self):
        pass


class ControlMode():
    """
    ----=
    """
    def __init__(self,modename,sat,params,maintain_RW,include_disturbances,calc_av_from_quat,include_rotational_motion):
        """
        Initialize
        """
        self.modename = modename
        self.sat = sat
        self.params = params
        self.maintain_RW = maintain_RW
        self.include_disturbances = include_disturbances
        self.addl_info = {}
        self.calc_av_from_quat = calc_av_from_quat
        self.saved_dist = np.zeros(3)
        self.beta_search_N = 4
        self.include_rotational_motion = include_rotational_motion

        self.reset_sat(self.sat)

        # self.Bbody_mat = np.array([1/self.sat.sensors[j].scale for j in range(len(sens)) if self.mtm_reading_mask[j]])@self.mtm_axes_mat_inv


        # self.mtq_axes_mat = np.stack([self.sat.attitude_sensors[j].axis for j in range(len(self.sat.attitude_sensors)) if self.mtm_mask[j]])
        # if self.MTQ_matrix_rank != 3:
        #     raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.

    def reset_sat(self,sat):
        self.sat = sat

        self.mtq_mask = np.array([isinstance(j,MTQ) for j in self.sat.actuators])
        # self.mtq_max = np.array([j.max for j in self.sat.actuators])
        self.mtq_N = sum(self.mtq_mask)

        self.mtq_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,MTQ) for j in self.sat.actuators]).astype(bool)
        # if sum(self.mtq_mask) != 3:
        #     raise ValueError('This is currently only implemented for exactly 3 MTQs') #TO-DO: include with more, by averaging?
        if np.any(self.mtq_mask):
            # self.MTQ_axes = np.vstack([j.axis.reshape((1,3)) if isinstance(j,MTQ) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.MTQ_axes = np.vstack([j.axis  for j in self.sat.actuators if isinstance(j,MTQ)])
            self.MTQ_matrix_rank = np.linalg.matrix_rank(self.MTQ_axes)
            # breakpoint()
            self.MTQ_matrix_null_space = scipy.linalg.null_space(self.MTQ_axes)
            self.MTQ_matrix_row_space = scipy.linalg.orth(self.MTQ_axes.T)
            if self.MTQ_matrix_rank == 3 and sum(self.mtq_mask) == 3:
                self.MTQ_matrix_inv = np.linalg.inv(self.MTQ_axes)
            elif sum(self.mtq_mask) > 3:
                warnings.warn("There are more than 3 MTQs here. This complicates turning a desired dipole into commands. Using only first 3 MTQs.") #TODO: # fix this
                if np.linalg.matrix_rank(self.MTQ_axes[:,0:3]) < 3:
                    raise ValueError("First 3 MTQs are not linearly independent. This is not invertible. Figuring out a spanning set has not been implemented. Please put a spanning set as the first 3 MTQs.")
                self.MTQ_matrix_inv = np.linalg.inv(self.MTQ_axes[:,0:3])
            else:
                raise ValueError("There are either less than 3 MTQs or there are 3/more but they are not linearly independent. This does not give full magnetic control. A matrix inverse cannot be calculated.")
            self.MTQ_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))#np.stack([self.MTQ_matrix_inv[j,:] if isinstance(j,MTQ) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.MTQ_ctrl_matrix[:,self.mtq_ctrl_mask] = np.eye(self.mtq_N)
        else:
            self.MTQ_axes = np.nan*np.eye(3)
            self.MTQ_matrix_inv = np.nan*np.eye(3)
            self.MTQ_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))#np.stack([self.MTQ_matrix_inv[j,:] if isinstance(j,MTQ) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.MTQ_matrix_rank = 0
            self.MTQ_matrix_null_space = np.eye(3)
            self.MTQ_matrix_row_space = np.zeros((3,0))



        self.magic_mask = np.array([isinstance(j,Magic) for j in self.sat.actuators])
        # self.magic_max = np.array([j.max for j in self.sat.actuators])
        self.magic_N = sum(self.magic_mask)
        self.magic_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,Magic) for j in self.sat.actuators]).astype(bool)

        if np.any(self.magic_mask):

            # self.Magic_axes = np.vstack([j.axis.reshape((1,3)) if isinstance(j,Magic) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.Magic_axes = np.vstack([j.axis  for j in self.sat.actuators if isinstance(j,Magic)])
            # self.Magic_matrix =np.vstack([j.axis.reshape((1,3)) if isinstance(j,Magic) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.Magic_matrix_rank = np.linalg.matrix_rank(self.Magic_axes)
            self.Magic_matrix_null_space = scipy.linalg.null_space(self.Magic_axes)
            self.Magic_matrix_row_space = scipy.linalg.orth(self.Magic_axes.T)
            if self.Magic_matrix_rank == 3 and sum(self.magic_mask) == 3:
                self.Magic_matrix_inv = np.linalg.inv(self.Magic_axes)
            elif sum(self.mtq_mask) > 3:
                warnings.warn("There are more than 3 Magic actuators here. This complicates turning a desired torque into commands. Using only first 3 Magic actuators.") #TODO: # fix this
                if np.linalg.matrix_rank(self.Magic_matrix[:,0:3]) < 3:
                    raise ValueError("First 3 Magic actuators are not linearly independent. This is not invertible. Figuring out a spanning set has not been implemented. Please put a spanning set as the first 3 Magic actuators.")
                self.Magic_matrix_inv = np.linalg.inv(self.Magic_axes[:,0:3])
            else:
                warnings.warn("There are more than 3 Magic actuators here. This complicates turning a desired torque into commands. Using only first 3 Magic actuators.") #TODO: # fix this
                self.Magic_matrix_inv = np.nan*np.ones([3,self.magic_N])
                # raise ValueError("There are either less than 3 Magic actuators or there are 3/more but they are not linearly independent. This does not give full control. A matrix inverse cannot be calculated.")
            self.Magic_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))#np.stack([self.MTQ_matrix_inv[j,:] if isinstance(j,MTQ) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.Magic_ctrl_matrix[:,self.magic_ctrl_mask] = np.eye(self.magic_N)# self.Magic_matrix_inv
        else:
            self.Magic_axes =  np.nan*np.eye(3)
            self.Magic_matrix_inv = np.nan*np.eye(3)
            self.Magic_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))
            self.Magic_matrix_rank = 0
            self.Magic_matrix_null_space = np.eye(3)
            self.Magic_matrix_row_space = np.zeros((3,0))
        # if sum(self.magic_mask) != 3:
        #     raise ValueError('This is currently only implemented for exactly 3 Magics') #TO-DO: include with more, by averaging?



        self.rw_mask = np.array([isinstance(j,RW) for j in self.sat.actuators])
        self.rw_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,RW) for j in self.sat.actuators]).astype(bool)
        self.rw_N = sum(self.rw_mask)


        self.RWjs = np.diagflat(np.array([self.sat.actuators[j].J for j in self.sat.momentum_inds]))
        if self.sat.number_RW > 0:
            # self.RWaxes = np.vstack([j.axis.reshape((1,3)) if isinstance(j,RW) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.RWaxes = np.vstack([j.axis for j in self.sat.actuators if isinstance(j,RW) ])

            self.RW_matrix_rank = np.linalg.matrix_rank(self.RWaxes)
            self.RW_matrix_null_space = scipy.linalg.null_space(self.RWaxes)
            self.RW_matrix_row_space = scipy.linalg.orth(self.RWaxes.T)
            if self.RW_matrix_rank == 3 and sum(self.rw_mask) == 3:
                self.RW_matrix_inv = np.linalg.inv(self.RWaxes)
            elif sum(self.rw_mask) > 3:
                warnings.warn("There are more than 3 RWs here. This complicates turning a desired torque into commands. Using only first 3 RWs.") #TODO: # fix this
                if np.linalg.matrix_rank(self.RWaxes[:,0:3]) < 3:
                    raise ValueError("First 3 RWs are not linearly independent. This is not invertible. Figuring out a spanning set has not been implemented. Please put a spanning set as the first 3 RWs.")
                self.RW_matrix_inv = np.linalg.inv(self.RWaxes[:,0:3])
            else:
                raise ValueError("There are either less than 3 RWs or there are 3/more but they are not linearly independent. This does not give full control. A matrix inverse cannot be calculated.")
            self.RW_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))#np.stack([self.RW_matrix_inv[j,:] if isinstance(j,RW) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
            self.RW_ctrl_matrix[:,self.rw_ctrl_mask] = np.eye(self.rw_N)

        else:
            self.RW_matrix_inv = np.nan*np.eye(3)
            self.RW_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))
            self.RWaxes = np.zeros((0,3))
            self.RW_matrix_rank = 0
            self.RW_matrix_null_space = np.eye(3)
            self.RW_matrix_row_space = np.zeros((3,0))

    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        return np.zeros(self.sat.control_len)


    def quatset_vec_rates_from_goal(self,goal_state,next_goal,dt):
        do_thing = True
        do_thing_next = True
        try:
            u = goal_state.eci_vec
            v = goal_state.body_vec
            if np.any(np.isnan(u)) or np.any(np.isnan(v)) or v is None or u is None:
                do_thing = False
        except:
            do_thing = do_thing
        if not do_thing:
            raise ValueError("vectors in goal_state not correct form. Cannot define quatset.")

        try:
            un = next_goal.eci_vec
            vn = next_goal.body_vec
            if np.any(np.isnan(un)) or np.any(np.isnan(vn)) or vn is None or un is None:
                do_thing_next = False
        except:
            do_thing_next = False
        if not do_thing_next:
            raise ValueError("vectors in next_goal not correct form. Cannot define quatset.")
        wu = np.cross(u,un)/dt #global frame
        wv = np.cross(v,vn)/dt #body frame
        return wu,wv

    def quatset_lyapR_func(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = False,qmult = 1, av_mat_mult = np.eye(3),betadot = 0):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        # qg *= np.sign(np.dot(qg,state[3:7]))
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr = quat_mult(quat_inv(qg),state[3:7])
        # qerr0 = np.dot(qg,state[3:7])
        if not wg_from_next_step:
            w_err = state[0:3]
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] - wg@rot_mat(qerr)
        return 0.5*np.dot(w_err,w_err@self.sat.J@av_mat_mult) + 2*(1-np.dot(goal_state.body_vec,goal_state.eci_vec@rot_mat(state[3:7])))*qmult #assumes av_mat_mult is PD.

    def quatset_lyapR_func_dbeta(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = False,qmult = 1, av_mat_mult = np.eye(3)):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        dqg__dbeta = -x*np.sin(beta)+y*np.cos(beta)
        # qg *= np.sign(np.dot(qg,state[3:7]))
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr = quat_mult(quat_inv(qg),state[3:7])
        # qerr0 = np.dot(qg,state[3:7])
        if not wg_from_next_step:
            w_err = state[0:3]
            dwerr_dbeta = np.zeros(3)
            dwerr_dbetadot = np.zeros(3)
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] + wv@rot_mat(qerr) - wu@rot_mat(state[3:7]) - 2*betadot*goal_state.eci_vec@rot_mat(state[3:7])
            dwerr_dbeta = -2*np.cross(wv,goal_state.body_vec)@rot_mat(qerr)
            dwerr_dbetadot = -2*goal_state.eci_vec@rot_mat(state[3:7])
        return np.dot(dwerr_dbeta,w_err@self.sat.J@av_mat_mult) #assumes av_mat_mult is PD.

    def quatset_lyapR_func_dt(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = False,consider_next_goal = False,qmult = 1, av_mat_mult = np.eye(3),betadot = 0,betaddot = 0):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        # qg *= np.sign(np.dot(qg,state[3:7]))
        # qerr = quat_mult(quat_inv(qg),state[3:7])

        if consider_next_goal:
            xn,yn = self.quatset_xy(next_goal)
            dx = quat_mult(quat_inv(x),xn)
            dy = quat_mult(quat_inv(y),yn)
            dx *= np.sign(dx[0])
            dy *= np.sign(dy[0])
            xdot = rot_exp(quat_log(dx)/dt)
            ydot = rot_exp(quat_log(dy)/dt)
            dqg_prenorm_dt = xdot*np.cos(beta)+ydot*np.sin(beta) + (-x*np.sin(beta)+y*np.cos(beta))*betadot
            dqg__dt = normed_vec_jac(qg,dqg_prenorm_dt)
            # dqg__dt *= np.sign(np.dot(qg,state[3:7]))
        else:
            dqg__dt = (-x*np.sin(beta)+y*np.cos(beta))*betadot
        qerr = quat_mult(quat_inv(qg),state[3:7])
        dqerr_dt = quat_mult(quat_inv(qg),0.5*state[0:3]@Wmat(state[3:7]).T) + quat_mult(quat_inv(dqg__dt),state[3:7]) # +

        # qerr0 = np.dot(qg,state[3:7])
        if not wg_from_next_step:
            w_err = state[0:3]
            dwgdt_overall = np.zeros(3)
            wu = np.zeros(3)
            wv = np.zeros(3)
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            # wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] + wv@rot_mat(qerr) - wu@rot_mat(state[3:7]) - 2*betadot*goal_state.eci_vec@rot_mat(state[3:7])
            wg_ECI = -wv@rot_mat(qg).T + wu + 2*betadot*goal_state.eci_vec

            dwgdt = 2*betadot*np.cross(goal_state.body_vec,wv)@rot_mat(qg).T@rot_mat(state[3:7]) #does not account for changing wv, wu--this would require advanced knowledge of goals for time step after next, too. Which this isn't set up to do yet.
            if consider_next_goal:
                dwgdt += 2*np.cross(quat_mult(quat_inv(qg),dqg__dt)[1:],wv)@rot_mat(qg).T@rot_mat(state[3:7])
            # w_err = state[0:3] - wg@rot_mat(qerr)
            dwgdt_overall = dwgdt@rot_mat(qerr) - wg_ECI@rot_mat(state[3:7])@skewsym(state[0:3]) + 2*betaddot*goal_state.eci_vec@rot_mat(state[3:7]) + 2*betadot*np.cross(wu,goal_state.eci_vec)@rot_mat(state[3:7])
        cmd = self.baseline_actuation(state,qerr,w_err,vecs)
        torq = self.sat.act_torque(state,cmd,vecs,False)
        dw_errdt = (-np.cross(state[0:3],state[0:3]@self.sat.J) + torq + self.sat.dist_torque(state,vecs))@self.sat.invJ - dwgdt_overall

        # return np.dot(dw_errdt,w_err@self.sat.J@av_mat_mult) - 2*(np.dot(goal_state.body_vec,goal_state.ECI_vec@rot_mat(state[3:7])@skewsym(state[0:3])) + np.dot(np.cross(wv,goal_state.body_vec),goal_state.ECI_vec@rot_mat(state[3:7])) + np.dot(goal_state.body_vec,np.cross(wu,goal_state.ECI_vec)@rot_mat(state[3:7])))*qmult #assumes av_mat_mult is PD.
        # print(np.dot(dw_errdt,w_err@self.sat.J@av_mat_mult) - 2*np.dot(np.cross(wv+state[0:3]-wu@rot_mat(state[3:7]),goal_state.body_vec),goal_state.eci_vec@rot_mat(state[3:7]))*qmult)
        return np.dot(dw_errdt,w_err@self.sat.J@av_mat_mult) - 2*np.dot(np.cross(wv+state[0:3]-wu@rot_mat(state[3:7]),goal_state.body_vec),goal_state.eci_vec@rot_mat(state[3:7]))*qmult #assumes av_mat_mult is PD.

    def quatset_lyapR_func_dt_dbeta(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = True,consider_next_goal = False,qmult = 1, av_mat_mult = np.eye(3),betadot = 0,betaddot = 0):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        dqg__dbeta = -x*np.sin(beta)+y*np.cos(beta)
        # qg_prenorm_dbeta *= np.sign(np.dot(qg,state[3:7]))
        # dqg__dbeta = normed_vec_jac(x*np.cos(beta)+y*np.sin(beta),qg_prenorm_dbeta)
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        if consider_next_goal:
            xn,yn = self.quatset_xy(next_goal)
            dx = quat_mult(quat_inv(x),xn)
            dy = quat_mult(quat_inv(y),yn)
            dx *= np.sign(dx[0])
            dy *= np.sign(dy[0])
            xdot = rot_exp(quat_log(dx)/dt)
            ydot = rot_exp(quat_log(dy)/dt)
            dqg_prenorm_dt = xdot*np.cos(beta)+ydot*np.sin(beta) +dqg__dbeta*betadot
            dqg_prenorm_dt_dbeta = -xdot*np.sin(beta)+ydot*np.cos(beta) - qg*betadot
            # dqg_prenorm__dbeta *= np.sign(np.dot(qg,state[3:7]))
            dqg__dt = normed_vec_jac(qg,dqg_prenorm_dt)
            dqg__dt_dbeta =  dqg_prenorm_dt_dbeta@normed_vec_jac(qg) + np.tensordot(dqg__dbeta, dqg_prenorm_dt@normed_vec_hess(qg),([1],[0]))
        else:
            dqg__dt = np.zeros(4)
            dqg__dt_dbeta = np.zeros(4)
        # qg *= np.sign(np.dot(qg,state[3:7])) #AFTER on purpose--so we can use the sign in the derivative.
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr = quat_mult(quat_inv(qg),state[3:7])
        # qerr0 = np.dot(qg,state[3:7])
        # dqerr_dt = quat_mult(quat_inv(qg),0.5*state[0:3]@Wmat(state[3:7]).T) + quat_mult(quat_inv(dqg__dt),state[3:7]) # +
        # dqerr0_dt = np.dot(dqg__dt,state[3:7]) + np.dot(qg,0.5*state[0:3]@Wmat(state[3:7]).T)
        dqerr__dbeta = -quat_mult(quat_inv(qg),quat_mult(dqg__dbeta,quat_inv(qerr)))
        # dqerr0__dbeta = np.dot(dqg__dbeta,state[3:7])
        # dqerr_dt_dbeta = quat_mult(-quat_mult(quat_inv(qg),quat_mult(dqg__dbeta,quat_inv(qg))),0.5*state[0:3]@Wmat(state[3:7]).T) + quat_mult(-quat_mult(quat_inv(dqg__dt),quat_mult(dqg__dt_dbeta,quat_inv(dqg__dt))),state[3:7])
        # dqerr0_dt_dbeta = np.dot(dqg__dt_dbeta,state[3:7]) + np.dot(dqg__dbeta,0.5*state[0:3]@Wmat(state[3:7]).T)

        if not wg_from_next_step:
            w_err = state[0:3]
            dwgdt_overall = np.zeros(3)
            dwgdt_overall_dbeta = np.zeros(3)
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            # wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] + wv@rot_mat(qerr) - wu@rot_mat(state[3:7]) - 2*betadot*goal_state.eci_vec@rot_mat(state[3:7])
            wg_ECI = -wv@rot_mat(qg).T + wu + 2*betadot*goal_state.eci_vec
            wgE_dbeta = 2*np.cross(wv,v)@rot_mat(qg).T

            dwgdt = 2*betadot*np.cross(goal_state.body_vec,wv)@rot_mat(qg).T@rot_mat(state[3:7]) #does not account for changing wv, wu--this would require advanced knowledge of goals for time step after next, too. Which this isn't set up to do yet.
            dwgdt_dbeta = -2*betadot*np.cross(np.cross(goal_state.body_vec,wv),goal_state.body_vec)@rot_mat(qg).T@rot_mat(state[3:7])
            if consider_next_goal:
                dwgdt += 2*np.cross(quat_mult(quat_inv(qg),dqg__dt)[1:],wv)@rot_mat(qg).T@rot_mat(state[3:7])
                dwgdt_dbeta += ( 2*np.cross(-np.cross(goal_state.body_vec,quat_mult(quat_inv(qg),dqg__dt_dbeta)[1:]) + quat_mult(quat_inv(qg),dqg__dt_dbeta)[1:],wv) - 4*np.cross(np.cross(quat_mult(quat_inv(qg),dqg__dt)[1:],wv),goal_state.body_vec))@rot_mat(qg).T@rot_mat(state[3:7])
            # w_err = state[0:3] - wg@rot_mat(qerr)
            dwgdt_overall = dwgdt@rot_mat(qerr) - wg_ECI@rot_mat(state[3:7])@skewsym(state[0:3]) + 2*betaddot*goal_state.eci_vec@rot_mat(state[3:7]) + 2*betadot*np.cross(wu,goal_state.eci_vec)@rot_mat(state[3:7])
            dwgdt_overall_dbeta = dwgdt_dbeta@rot_mat(qerr) - 2*np.cross(dwgdt,goal_state.body_vec)@rot_mat(qg).T@rot_mat(state[3:7]) - wgE_dbeta@rot_mat(state[3:7])@skewsym(state[0:3])

        dwerr_dbeta = -wgE_dbeta@rot_mat(state[3:7])
        cmd = self.baseline_actuation(state,qerr,w_err,vecs)
        cmd_dbeta = dqerr__dbeta@self.baseline_actuation_jac_over_qerr(state,qerr,w_err,vecs) #TODO:add to quatset stuff.
        torq = self.sat.act_torque(state,cmd,vecs,False)
        torq_dcmd = sum([self.sat.actuators[j].dtorq__du(cmd[j],self.sat,state,vecs,update_noise = False) for j in range(len(self.sat.actuators))],np.zeros(3)) #TODO: move to a dact_torq__du function in the satellite class
        torq_dbeta = cmd_dbeta@torq_dcmd
        dw_errdt = (-np.cross(state[0:3],state[0:3]@self.sat.J) + torq + self.sat.dist_torque(state,vecs))@self.sat.invJ - dwgdt_overall
        dw_errdt_dbeta = torq_dbeta@self.sat.invJ - dwgdt_overall_dbeta

        return np.dot(dw_errdt_dbeta,w_err@self.sat.J@av_mat_mult) + np.dot(dw_errdt,dwerr_dbeta@self.sat.J@av_mat_mult) #assumes av_mat_mult is PD.



    def quatset_lyap_func(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = False,qmult = 1, av_mat_mult = np.eye(3),betadot = 0):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        # qg *= np.sign(np.dot(qg,state[3:7]))
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr0 = np.dot(qg,state[3:7])
        s0 = np.sign(qerr0)
        qerr0*=s0
        qerr*=s0
        qg*=s0
        if not wg_from_next_step:
            w_err = state[0:3]
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] - wg@rot_mat(qerr)
        return 0.5*np.dot(w_err,w_err@self.sat.J@av_mat_mult) + 2*(1-qerr0)*qmult #assumes av_mat_mult is PD.


    def quatset_lyap_func_dbeta(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = False,qmult = 1, av_mat_mult = np.eye(3)):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        # qg = normalize(x*np.cos(beta)+y*np.sin(beta))
        qg = x*np.cos(beta)+y*np.sin(beta)
        # dqg_prenorm = -x*np.sin(beta)+y*np.cos(beta)
        # dqg__dbeta = normed_vec_jac(x*np.cos(beta)+y*np.sin(beta),dqg_prenorm)
        dqg__dbeta = -x*np.sin(beta)+y*np.cos(beta)
        # dqg__dbeta *= np.sign(np.dot(qg,state[3:7]))
        # qg *= np.sign(np.dot(qg,state[3:7]))
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr0 = np.dot(qg,state[3:7])
        s0 = np.sign(qerr0)
        qerr0*=s0
        qerr*=s0
        qg*=s0
        dqg__dbeta*=s0
        # dqerr__dbeta = quat_mult(-quat_mult(quat_inv(qg),quat_mult(dqg__dbeta,quat_inv(qg))),state[3:7])
        dqerr__dbeta = -quat_mult(quat_inv(qg),quat_mult(dqg__dbeta,qerr))
        dqerr0__dbeta = np.dot(dqg__dbeta,state[3:7])
        if not wg_from_next_step:
            w_err = state[0:3]
            dwerr_dbeta = np.zeros(3) #this boils down to minang...
            dwerr_dbetadot = np.zeros(3)
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] + wv@rot_mat(qerr) - wu@rot_mat(state[3:7]) - 2*betadot*goal_state.eci_vec@rot_mat(state[3:7])
            dwerr_dbeta = -2*np.cross(wv,goal_state.body_vec)@rot_mat(qerr)
            dwerr_dbetadot = -2*goal_state.eci_vec@rot_mat(state[3:7])

            # nq0 = quat_mult(state[3:7],rot_exp(dt*state[0:3]))
            # nqg = self.minang_quat(nq0,next_goal)
            # dq = quat_mult(nqg,quat_inv(qg))
            # dq *= np.sign(dq[0])
            # wg_ECI = quat_log(dq)/dt
            # w_err = state[0:3] - wg_ECI@rot_mat(state[3:7])
            #
            # u = goal_state.eci_vec
            # nwgE = norm(wg_ECI)
            # ss = np.sin(nwgE*dt)
            # cc = np.cos(nwgE*dt)
            # uwgE = np.dog(u,wg_ECI)
            # tmp = (1+cc)/ss
            # dwgE_dbeta = -np.arccos(uwgE*ss/norm(wgE))*( np.cross(u,wg_ECI)*nwgE + u*tmp*nwgE**2.0 + wg_ECI*uwgE*(2*cc/(dt*nwgE) - tmp ) )/np.sqrt(nwgE**2.0-ss*ss*uwgE**2.0) #from a lot of Math and the Jr -1 (theta) equations in Sola (eq. 184)
            # dwerr_dbeta = np.zeros(3) - dwgE_dbeta@rot_mat(state[3:7])

        return np.dot(dwerr_dbeta,w_err@self.sat.J@av_mat_mult) + 2*(-dqerr0__dbeta)*qmult #assumes av_mat_mult is PosDef


    def quatset_lyap_func_dt(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = False,consider_next_goal = False,qmult = 1, av_mat_mult = np.eye(3),betadot = 0,betaddot = 0):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        if consider_next_goal:
            xn,yn = self.quatset_xy(next_goal)
            dx = quat_mult(quat_inv(x),xn)
            dy = quat_mult(quat_inv(y),yn)
            dx *= np.sign(dx[0])
            dy *= np.sign(dy[0])
            xdot = rot_exp(quat_log(dx)/dt)
            ydot = rot_exp(quat_log(dy)/dt)
            dqg_prenorm_dt = xdot*np.cos(beta)+ydot*np.sin(beta) + (-x*np.sin(beta)+y*np.cos(beta))*betadot
            dqg__dt = normed_vec_jac(qg,dqg_prenorm_dt)
            # dqg__dt *= np.sign(np.dot(qg,state[3:7]))
        else:
            dqg__dt = (-x*np.sin(beta)+y*np.cos(beta))*betadot
        qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr0 = np.dot(qg,state[3:7])
        s0 = np.sign(qerr0)
        qerr0*=s0
        qerr*=s0
        qg*=s0
        dqg__dt*=s0
        dqerr_dt = quat_mult(quat_inv(qg),0.5*state[0:3]@Wmat(state[3:7]).T - quat_mult(dqg__dt,qerr)) # +
        dqerr0_dt = np.dot(dqg__dt,state[3:7]) + np.dot(qg,0.5*state[0:3]@Wmat(state[3:7]).T)

        if not wg_from_next_step:
            w_err = state[0:3]
            dwgdt_overall = np.zeros(3)
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            # wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] + wv@rot_mat(qerr) - wu@rot_mat(state[3:7]) - 2*betadot*goal_state.eci_vec@rot_mat(state[3:7])
            wg_ECI = -wv@rot_mat(qg).T + wu + 2*betadot*goal_state.eci_vec

            dwgdt = 2*betadot*np.cross(goal_state.body_vec,wv)@rot_mat(qg).T@rot_mat(state[3:7]) #does not account for changing wv, wu--this would require advanced knowledge of goals for time step after next, too. Which this isn't set up to do yet.
            if consider_next_goal:
                dwgdt += 2*np.cross(quat_mult(quat_inv(qg),dqg__dt)[1:],wv)@rot_mat(qg).T@rot_mat(state[3:7])
            # w_err = state[0:3] - wg@rot_mat(qerr)
            dwgdt_overall = dwgdt@rot_mat(qerr) - wg_ECI@rot_mat(state[3:7])@skewsym(state[0:3]) + 2*betaddot*goal_state.eci_vec@rot_mat(state[3:7]) + 2*betadot*np.cross(wu,goal_state.eci_vec)@rot_mat(state[3:7])
        cmd = self.baseline_actuation(state,qerr,w_err,vecs)
        torq = self.sat.act_torque(state,cmd,vecs,False)
        dw_errdt = (-np.cross(state[0:3],state[0:3]@self.sat.J) + torq + self.sat.dist_torque(state,vecs))@self.sat.invJ - dwgdt_overall

        return np.dot(dw_errdt,w_err@self.sat.J@av_mat_mult) + 2*(-dqerr0_dt)*qmult #assumes av_mat_mult is PD.


    def quatset_lyap_func_dt_dbeta(self,state,goal_state,beta,vecs,next_goal = None, dt=None,wg_from_next_step = True,consider_next_goal = False,qmult = 1, av_mat_mult = np.eye(3),betadot = 0,betaddot = 0):
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        dqg__dbeta = -x*np.sin(beta)+y*np.cos(beta)
        # qg_prenorm_dbeta *= np.sign(np.dot(qg,state[3:7]))
        # dqg__dbeta = normed_vec_jac(x*np.cos(beta)+y*np.sin(beta),qg_prenorm_dbeta)
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        if consider_next_goal:
            xn,yn = self.quatset_xy(next_goal)
            dx = quat_mult(quat_inv(x),xn)
            dy = quat_mult(quat_inv(y),yn)
            dx *= np.sign(dx[0])
            dy *= np.sign(dy[0])
            xdot = rot_exp(quat_log(dx)/dt)
            ydot = rot_exp(quat_log(dy)/dt)
            dqg_prenorm_dt = xdot*np.cos(beta)+ydot*np.sin(beta) +dqg__dbeta*betadot
            dqg_prenorm_dt_dbeta = -xdot*np.sin(beta)+ydot*np.cos(beta) - qg*betadot
            # dqg_prenorm__dbeta *= np.sign(np.dot(qg,state[3:7]))
            dqg__dt = normed_vec_jac(qg,dqg_prenorm_dt)
            dqg__dt_dbeta =  dqg_prenorm_dt_dbeta@normed_vec_jac(qg) + np.tensordot(dqg__dbeta, dqg_prenorm_dt@normed_vec_hess(qg),([1],[0]))
        else:
            dqg__dt = np.zeros(4)
            dqg__dt_dbeta = np.zeros(4)
        # qg *= np.sign(np.dot(qg,state[3:7])) #AFTER on purpose--so we can use the sign in the derivative.
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr0 = np.dot(qg,state[3:7])
        s0 = np.sign(qerr0)
        qerr0*=s0
        qerr*=s0
        qg*=s0
        dqg__dt*=s0
        dqg__dbeta*=s0
        dqg__dt_dbeta*=s0
        dqerr_dt = quat_mult(quat_inv(qg),(0.5*state[0:3]@Wmat(state[3:7]).T - quat_mult(dqg__dt,qerr))) # +
        dqerr0_dt = np.dot(dqg__dt,state[3:7]) + np.dot(qg,0.5*state[0:3]@Wmat(state[3:7]).T)
        dqerr__dbeta = -quat_mult(quat_inv(qg),quat_mult(dqg__dbeta,quat_inv(qerr)))
        dqerr0__dbeta = np.dot(dqg__dbeta,state[3:7])
        dqerr_dt_dbeta = quat_mult(quat_inv(qg), -quat_mult(dqg__dbeta,dqerr_dt) - quat_mult(dqg__dt_dbeta,qerr) - quat_mult(dqg__dt,dqerr__dbeta))
        dqerr0_dt_dbeta = np.dot(dqg__dt_dbeta,state[3:7]) + np.dot(dqg__dbeta,0.5*state[0:3]@Wmat(state[3:7]).T)

        if not wg_from_next_step:
            w_err = state[0:3]
            dwgdt_overall = np.zeros(3)
            dwgdt_overall_dbeta = np.zeros(3)
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            # wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] + wv@rot_mat(qerr) - wu@rot_mat(state[3:7]) - 2*betadot*goal_state.eci_vec@rot_mat(state[3:7])
            wg_ECI = -wv@rot_mat(qg).T + wu + 2*betadot*goal_state.eci_vec
            wgE_dbeta = 2*np.cross(wv,v)@rot_mat(qg).T

            dwgdt = 2*betadot*np.cross(goal_state.body_vec,wv)@rot_mat(qg).T@rot_mat(state[3:7]) #does not account for changing wv, wu--this would require advanced knowledge of goals for time step after next, too. Which this isn't set up to do yet.
            dwgdt_dbeta = -2*betadot*np.cross(np.cross(goal_state.body_vec,wv),goal_state.body_vec)@rot_mat(qg).T@rot_mat(state[3:7])
            if consider_next_goal:
                dwgdt += 2*np.cross(quat_mult(quat_inv(qg),dqg__dt)[1:],wv)@rot_mat(qg).T@rot_mat(state[3:7])
                dwgdt_dbeta += ( 2*np.cross(-np.cross(goal_state.body_vec,quat_mult(quat_inv(qg),dqg__dt_dbeta)[1:]) + quat_mult(quat_inv(qg),dqg__dt_dbeta)[1:],wv) - 4*np.cross(np.cross(quat_mult(quat_inv(qg),dqg__dt)[1:],wv),goal_state.body_vec))@rot_mat(qg).T@rot_mat(state[3:7])
            # w_err = state[0:3] - wg@rot_mat(qerr)
            dwgdt_overall = dwgdt@rot_mat(qerr) - wg_ECI@rot_mat(state[3:7])@skewsym(state[0:3]) + 2*betaddot*goal_state.eci_vec@rot_mat(state[3:7]) + 2*betadot*np.cross(wu,goal_state.eci_vec)@rot_mat(state[3:7])
            dwgdt_overall_dbeta = dwgdt_dbeta@rot_mat(qerr) - 2*np.cross(dwgdt,goal_state.body_vec)@rot_mat(qg).T@rot_mat(state[3:7]) - wgE_dbeta@rot_mat(state[3:7])@skewsym(state[0:3])

        dwerr_dbeta = -wgE_dbeta@rot_mat(state[3:7])
        cmd = self.baseline_actuation(state,qerr,w_err,vecs)
        cmd_dbeta = dqerr__dbeta@self.baseline_actuation_jac_over_qerr(state,qerr,w_err,vecs) #TODO:add to quatset stuff.
        torq = self.sat.act_torque(state,cmd,vecs,False)
        torq_dcmd = sum([self.sat.actuators[j].dtorq__du(cmd[j],self.sat,state,vecs,update_noise = False) for j in range(len(self.sat.actuators))],np.zeros(3)) #TODO: move to a dact_torq__du function in the satellite class
        torq_dbeta = cmd_dbeta@torq_dcmd
        dw_errdt = (-np.cross(state[0:3],state[0:3]@self.sat.J) + torq + self.sat.dist_torque(state,vecs))@self.sat.invJ - dwgdt_overall
        dw_errdt_dbeta = torq_dbeta@self.sat.invJ - dwgdt_overall_dbeta

        return np.dot(dw_errdt_dbeta,w_err@self.sat.J@av_mat_mult) + np.dot(dw_errdt,dwerr_dbeta@self.sat.J@av_mat_mult) + 2*(-dqerr0_dt_dbeta)*qmult #assumes av_mat_mult is PD.

    def state_err(self,state,desired,next_desired = None,dt = None,print_info=False):
        # state = state.reshape(self.sat.state_len,1)
        # desired = desired.reshape(self.sat.state_len,1)
        if isinstance(desired,state_goal):
            desired = desired.state
        q = state[3:7]
        w = state[0:3]
        extra = state[7:self.sat.state_len]
        q_desired = desired[3:7]
        w_desired = desired[0:3]
        q_desired = normalize(q_desired)
        qerr = quat_mult(quat_inv(q_desired),q)##np.vstack([np.array([np.ndarray.item(q.T@q_desired)]),-q_desired[1:]*q[0]+q[1:]*q_desired[0]-np.cross(q_desired[1:],q[1:])])
        if self.calc_av_from_quat:
            if next_desired is not None:
                if isinstance(next_desired,state_goal):
                    next_desired = next_desired.state
                qwgoal = quat_mult(quat_inv(q_desired),normalize(next_desired[3:7]))
                qwgoal *= np.sign(qwgoal[0])
                w_desired = quat_log(qwgoal)/dt
                # w_desired = quat_log(quat_mult(quat_inv(q_desired),next_desired.state[3:7]))/dt
                desired[0:3] = w_desired
        if print_info:
            # print('     body des ', w_desired)
            # print('     ECI      ', w_desired@rot_mat(q_desired).T)
            # print('     body act ', w_desired@rot_mat(qerr))
            print('     qdes dot ', w_desired@Wmat(q_desired).T)
            print('     qdes     ', q_desired)
            # print('     qerr dot ', (w-w_desired@rot_mat(qerr))@Wmat(qerr).T)
            # print('     qerr     ', qerr)

        extra_desired = desired[7:self.sat.state_len]
        # qerr = qerr*np.sign(np.dot(q,q_desired))
        return np.concatenate([w-w_desired@rot_mat(qerr),qerr])#,extra-extra_desired])

    def av_from_quat_goals(self,state,goal_state,next_goal,dt):
        #generates the angular velocity between 2 successive quaternion goals (this is the body-frame angular velocity WHEN IN THE DESIRED GOAL STATE)
        qwgoal = quat_mult(quat_inv(goal_state.state[3:7]),next_goal.state[3:7])
        qwgoal *= np.sign(qwgoal[0])
        wgoal = quat_log(qwgoal)/dt#@rot_mat(goal_state.state[3:7]).T#@rot_mat(state[3:7])/dt #rot_mat(state[3:7]).T@
        goal_state.state[0:3] = wgoal
        return wgoal

    def mtq_command_maintain_RW(self,mtq_cmd,state,vecs,compensate_bias = True,scaled_lim = 1.0):
        scaled_lim = abs(scaled_lim)
        scaled_lim = max(scaled_lim,1.0/scaled_lim)
        # udes0 = mtq_cmd@self.MTQ_ctrl_matrix
        mtq_udes = mtq_cmd
        # print('ud1',udes)
        #operates differently from magic or from RW because it's interaction with torque is nonlinear. a net magnetic moment (including bias) that is parallel to the goal moment is preferered, even if it's vector distance from the desired isn't minimized
        l,u = self.sat.control_bounds() #currently assumes l = -u; TODO: fix
        l_mtq = l[self.mtq_ctrl_mask]
        u_mtq = u[self.mtq_ctrl_mask]
        if compensate_bias:
            bias = np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        else:
            bias = np.zeros(len(self.sat.actuators))
        mtq_bias = bias[self.mtq_ctrl_mask]
        udes_nozmask = np.abs(mtq_udes)>num_eps
        mult_bounds_noz = np.stack([mtq_bias[udes_nozmask]/mtq_udes[udes_nozmask] + np.abs(u_mtq[udes_nozmask]/mtq_udes[udes_nozmask]),mtq_bias[udes_nozmask]/mtq_udes[udes_nozmask] - np.abs(u_mtq[udes_nozmask]/mtq_udes[udes_nozmask])])
        zval_possible = np.abs(mtq_bias)<=u_mtq

        # breakpoint()
        #outside limits serve to sanity check to make sure it's not wildly off (like, its parallel but 100x the magnitude due to weird geometry--if it is, just clip it ) Baseline is only allow if scaling is between 0 and 1. Clip if too big.

        maxk = min(min(mult_bounds_noz[0,:]),scaled_lim)
        mink = max(max(mult_bounds_noz[1,:]),0) #don't go negative!
        perfect_soln_possible = np.all(zval_possible) and maxk >= mink
        if perfect_soln_possible:
            if maxk>=1:
                if mink<=1:
                    k = 1
                else:
                    k = mink
            else:
                if mink<=1:
                    k = maxk
                else:
                    raise ValueError('this should not be possible')
            mtq_new_cmd = k*mtq_udes - mtq_bias
        # elif np.all(zval_possible):
        #     #noz bounds broken
        #     mtq_udes = k*udes0 - mtq_bias
        #
        # elif min(mult_bounds_noz[0,:]) > max(mult_bounds_noz[1,:]):
        #     #zero broken
        else: #can be imrpoved to also weight this so that being closer to the direction of the desired vector has some benfit.

            mtq_new_cmd = np.clip(mtq_udes-mtq_bias,l_mtq,u_mtq)
            # possible_pts = [np.array([np.sign((k & 2**ind) - 0.5)*u_mtq[ind] for ind in range(sum(self.mtq_ctrl_mask))]) for k in range(0,2**sum(self.mtq_ctrl_mask))]
            # normd = normalize(mtq_udes)
            # pts_dist = np.array([norm((np.eye(len(normd))-np.outer(normd,normd))@(j+mtq_bias)) for j in possible_pts])
            # ind = np.argmin(pts_dist)
            # print(ind)
            # if len(np.array([ind]).flatten()) == 1:
            #     mtq_new_cmd = possible_pts[ind]
            # else:
            #     mtq_new_cmd = possible_pts[ind[0]]





        # else:
        #     l,u = self.sat.control_bounds()
        #     clipped_udes = limit(udes,u)
        # print('ud2',udes)
        clipped_udes = np.zeros(self.sat.control_len)#self.sat.apply_control_bounds(udes)
        clipped_udes[self.mtq_ctrl_mask] = mtq_new_cmd
        # print('cliptest', mtq_new_cmd+mtq_bias,mtq_udes)
        # print((180.0/np.pi)*np.arccos(np.clip(np.abs(np.dot(normalize(mtq_new_cmd+mtq_bias),normalize(mtq_udes))),-1,1)))
        # print(norm((np.eye(len(mtq_udes))-np.outer(normalize(mtq_udes),normalize(mtq_udes)))@(mtq_new_cmd+mtq_bias)),norm(mtq_udes))
        # print('mult',np.dot(normalize(mtq_udes),(mtq_new_cmd+mtq_bias))/norm(mtq_udes))


        # print('ud3',clipped_udes)
        if self.maintain_RW and self.sat.number_RW>0:
            exp_torq = sum([self.sat.actuators[j].torque(clipped_udes[j],self.sat,state,vecs,update_noise = False) for j in range(len(self.sat.actuators)) if isinstance(self.sat.actuators[j],MTQ)],np.zeros(3))
            clipped_udes[self.rw_ctrl_mask] += exp_torq@self.sat.invJ_noRW@self.RWaxes.T@self.RWjs #keeps RW velocities the same.
        return clipped_udes


    def heritage_mtq_command_maintain_RW(self,mtq_cmd,state,vecs,compensate_bias = True):

        udes = mtq_cmd@self.MTQ_ctrl_matrix
        if compensate_bias:
            udes -= np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        clipped_udes = self.sat.apply_control_bounds(udes)
        if self.maintain_RW and self.sat.number_RW>0:
            exp_torq = sum([self.sat.actuators[j].torque(clipped_udes[j],self.sat,state,vecs,update_noise = False) for j in range(len(self.sat.actuators)) if isinstance(self.sat.actuators[j],MTQ)],np.zeros(3))
            clipped_udes[self.rw_ctrl_mask] += exp_torq@self.sat.invJ_noRW@self.RWaxes.T@self.RWjs #keeps RW velocities the same.
        return clipped_udes


    def magic_command_maintain_RW(self,magic_cmd,state,vecs,compensate_bias = True):
        udes = magic_cmd@self.Magic_ctrl_matrix
        if compensate_bias:
            udes -= np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        clipped_udes = self.sat.apply_control_bounds(udes)
        if self.maintain_RW and self.sat.number_RW>0:
            exp_torq = sum([self.sat.actuators[j].torque(clipped_udes[j],self.sat,state,vecs,update_noise = False) for j in range(len(self.sat.actuators)) if isinstance(self.sat.actuators[j],Magic)],np.zeros(3))
            clipped_udes[self.rw_ctrl_mask] += exp_torq@self.sat.invJ_noRW@self.RWaxes.T@self.RWjs #keeps RW velocities the same.
        return clipped_udes

    def quatset_xy(self,goal_state):
        do_thing = True
        try:
            u = goal_state.eci_vec
            v = goal_state.body_vec
            if np.any(np.isnan(u)) or np.any(np.isnan(v)) or v is None or u is None:
                do_minang = do_thing
        except:
            do_thing = do_thing
        if not do_thing:
            raise ValueError("vectors not correct form. Cannot define quatset.")
        th = np.arccos(np.dot(u,v))
        ct2 = np.cos(th/2.0)
        xq = np.concatenate([[ct2],-0.5*np.cross(v,u)/ct2])
        yq = -np.concatenate([[0],0.5*(v+u)/ct2])
        return xq,yq

    def minang_quat(self, q0, goal_state):
        xq,yq = self.quatset_xy(goal_state)
        qg = normalize(xq*np.dot(q0,xq)+yq*np.dot(q0,yq)) # from my quaternion pointing note
        return qg

    def perpB_quat(self, q0, goal_state,Bbody):
        x,y = self.quatset_xy(goal_state)
        vec = Bbody@self.sat.invJ
        dx = np.dot(vec,quat_mult(quat_inv(x),q0)[1:])
        dy = np.dot(vec,quat_mult(quat_inv(y),q0)[1:])
        if dx==0:
            if dy==0:
                warnings.warning("no solution for perpB")
                return normalize(x*np.dot(q0,x)+y*np.dot(q0,y)) #minang quat answer
            else:
                b1 = 0
        elif dy==0:
            b1 = 0.5*np.pi#,-0.5*np.pi
        else:
            b1 = np.arctan(dy/dx)+0.5*np.pi

        qg = normalize(x*np.cos(b1)+y*np.sin(b1)) # from my quaternion pointing note
        qg *= np.sign(np.dot(q0,qg)) #makes sure qerr has positive scalar component
        return qg

    def quatset_pos_to_av_goal(self,state,goal_state,next_goal,dt,qg,metric = "ang"):
        # qerr = quat_mult(quat_inv(qg),state[3:7])
        nq0 = quat_mult(state[3:7],rot_exp(dt*state[0:3]))
        if metric=="ang":
            nqg = self.minang_quat(nq0,next_goal)
        elif metric == "B":
            nqg = self.perpB_err(q0, goal_state,Bbody)

        dq = quat_mult(quat_inv(qg),nqg)
        dq *= np.sign(dq[0])
        wg = quat_log(dq)/dt
        return wg

    def minLyap_err(self,state,goal_state,vecs,next_goal,dt,qmult=1,av_mat_mult = np.eye(3),betadot = 0,betaddot=0):
        wg_next = True
        consider_next = False
        if self.params.lyap_type == 0:
            # N = 5
            # beta0s = np.linspace(0,2*pi,N+1)[:-1]
            # res = [scipy.optimize.minimize_scalar()]
            # for b in beta0s:
            overall_func = lambda b : self.quatset_lyap_func(state,goal_state,b,vecs,next_goal = next_goal, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot)
            # beta = scipy.optimize.minimize_scalar(func).x

            #V=0.5*np.dot(w_err,w_err@self.sat.J@av_mat_mult) + 2*(1-qerr[0])*qmult
            #minimiing V over beta is achieved by maximizing qerr[0], so you get minang.
            # w_err,q_err = self.minang_err(state,goal_state,next_goal,dt)
        elif self.params.lyap_type == 1:
            overall_func = lambda b : self.quatset_lyap_func_dt(state,goal_state,b,vecs,next_goal = next_goal,consider_next_goal = consider_next, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot,betaddot = betaddot)
            # beta = scipy.optimize.minimize_scalar(func).x
        elif self.params.lyap_type == 2:
            func = lambda b : self.quatset_lyap_func(state,goal_state,b,vecs,next_goal = next_goal, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot)
            func_dt = lambda b : self.quatset_lyap_func_dt(state,goal_state,b,vecs,next_goal = next_goal,consider_next_goal = consider_next, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot,betaddot = betaddot)
            overall_func = lambda b : func(b)*np.exp(func_dt(b))
            # beta = scipy.optimize.minimize_scalar(overall_func).x
             #minimize (V/-Vdot) (with vdot negative). if only postive vdot posssible, minimize vdot
        # V = 0.5*self.params.kp_gain*(2-_err[0])
        elif self.params.lyap_type == 3:
            func = lambda b : self.quatset_lyap_func(state,goal_state,b,vecs,next_goal = next_goal, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot)
            func_dt = lambda b : self.quatset_lyap_func_dt(state,goal_state,b,vecs,next_goal = next_goal,consider_next_goal = consider_next, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot,betaddot = betaddot)
            overall_func = lambda b : func(b)*(np.abs(func_dt(b))**np.sign(func_dt(b)))
             #minimize (V/-Vdot) (with vdot negative). if only postive vdot posssible, minimize vdot
        # V = 0.5*self.params.kp_gain*(2-_err[0])
        else:
            raise ValueError("Incorrect lyap type mode")
        results = [scipy.optimize.minimize_scalar(overall_func,method = 'golden',bracket =[(np.pi/self.beta_search_N)*j,(np.pi/self.beta_search_N)*(j+1)]) for j in range(self.beta_search_N)]
        beta = results[np.argmin([j.fun for j in results])].x
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr *= np.sign(qerr[0])
        goal_state.state[3:7] = qg
        if not wg_next:
            w_err = state[0:3]
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] - wg@rot_mat(qerr)
            goal_state.state[0:3] = wg
        return w_err,qerr


    def minLyapR_err(self,state,goal_state,vecs,next_goal,dt,qmult = 1, av_mat_mult = np.eye(3),betadot = 0,betaddot=0):
        wg_next = True
        consider_next = False
        if self.params.lyap_type == 0:
            # N = 5
            # beta0s = np.linspace(0,2*pi,N+1)[:-1]
            # res = [scipy.optimize.minimize_scalar()]
            # for b in beta0s:
            overall_func = lambda b : self.quatset_lyapR_func(state,goal_state,b,vecs,next_goal = next_goal, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot)
            # beta = scipy.optimize.minimize_scalar(func).x

            #V=0.5*np.dot(w_err,w_err@self.sat.J@av_mat_mult) + 2*(1-qerr[0])*qmult
            #minimiing V over beta is achieved by maximizing qerr[0], so you get minang.
            # w_err,q_err = self.minang_err(state,goal_state,next_goal,dt)
        elif self.params.lyap_type == 1:
            overall_func = lambda b : self.quatset_lyapR_func_dt(state,goal_state,b,vecs,next_goal = next_goal,consider_next_goal = consider_next, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot,betaddot = betaddot)
            # beta = scipy.optimize.minimize_scalar(func).x
        elif self.params.lyap_type == 2:
            func = lambda b : self.quatset_lyapR_func(state,goal_state,b,vecs,next_goal = next_goal, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot)
            func_dt = lambda b : self.quatset_lyapR_func_dt(state,goal_state,b,vecs,next_goal = next_goal,consider_next_goal = consider_next, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot,betaddot = betaddot)
            overall_func = lambda b : func(b)*np.exp(func_dt(b))
            # beta = scipy.optimize.minimize_scalar(overall_func).x
             #minimize (V/-Vdot) (with vdot negative). if only postive vdot posssible, minimize vdot
        # V = 0.5*self.params.kp_gain*(2-_err[0])
        elif self.params.lyap_type == 3:
            func = lambda b : self.quatset_lyapR_func(state,goal_state,b,vecs,next_goal = next_goal, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot)
            func_dt = lambda b : self.quatset_lyapR_func_dt(state,goal_state,b,vecs,next_goal = next_goal,consider_next_goal = consider_next, dt=dt,wg_from_next_step = wg_next,qmult = qmult, av_mat_mult = av_mat_mult,betadot = betadot,betaddot = betaddot)
            overall_func = lambda b : func(b)*(np.abs(func_dt(b))**np.sign(func_dt(b)))
            # beta = scipy.optimize.minimize_scalar(overall_func).x
             #minimize (V/-Vdot) (with vdot negative). if only postive vdot posssible, minimize vdot
        # V = 0.5*self.params.kp_gain*(2-_err[0])
        else:
            print(self.params.lyap_type )
            raise ValueError("Incorrect lyapR type mode")
        results = [scipy.optimize.minimize_scalar(overall_func,method = 'golden',bracket =[(np.pi/self.beta_search_N)*j,(np.pi/self.beta_search_N)*(j+1)]) for j in range(self.beta_search_N)]
        beta = results[np.argmin([j.fun for j in results])].x
        x,y = self.quatset_xy(goal_state)
        x = normalize(x)
        y = normalize(y)
        qg = x*np.cos(beta)+y*np.sin(beta)
        qerr = quat_mult(quat_inv(qg),state[3:7])
        qerr *= np.sign(qerr[0])
        goal_state.state[3:7] = qg
        if not wg_next:
            w_err = state[0:3]
        else:
            wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
            wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
            w_err = state[0:3] - wg@rot_mat(qerr)
            goal_state.state[0:3] = wg
        return w_err,qerr

    def minang_err(self,state,goal_state,next_goal,dt,betadot = 0,betaddot=0):
        qg = self.minang_quat(state[3:7],goal_state)
        goal_state.state[3:7] = qg
        qerr = quat_mult(quat_inv(qg),state[3:7])
        if not self.calc_av_from_quat:
            return state[0:3], qerr


        wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
        wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
        w_err = state[0:3] - wg@rot_mat(qerr)
        goal_state.state[0:3] = wg

        # nq0 = quat_mult(state[3:7],rot_exp(dt*state[0:3]))
        # nqg = self.minang_quat(nq0,next_goal)
        # dq = quat_mult(quat_inv(qg),nqg)
        # dq *= np.sign(dq[0])
        # wg = quat_log(dq)/dt
        # goal_state.state[0:3] = wg
        return state[0:3]-wg@rot_mat(qerr), qerr

    def perpB_err(self,state,goal_state,next_goal,dt,Bbody,betadot = 0,betaddot=0):
        qg = self.perpB_quat(state[3:7],goal_state,Bbody)
        goal_state.state[3:7] = qg
        qerr = quat_mult(quat_inv(qg),state[3:7])
        if not self.calc_av_from_quat:
            return state[0:3], qerr

        # qstep = rot_exp(dt*state[0:3])
        # nq0 = quat_mult(state[3:7],qstep)
        # nqg = self.perpB_quat(nq0,next_goal,Bbody@rot_mat(qstep))#assume B_ECI is constant. TODO:fix that?
        # dq = quat_mult(quat_inv(qg),nqg)
        # dq *= np.sign(dq[0])
        # wg = quat_log(dq)/dt
        # goal_state.state[0:3] = wg
        wu,wv = self.quatset_vec_rates_from_goal(goal_state,next_goal,dt)
        wg = -wv + wu@rot_mat(qg) + 2*betadot*goal_state.body_vec
        w_err = state[0:3] - wg@rot_mat(qerr)
        goal_state.state[0:3] = wg

        # wg = self.quatset_pos_to_av_goal(self,state,goal_state,next_goal,dt,qg)
        return state[0:3]-wg@rot_mat(qerr), qerr


    def rankisN_ctrl_adjust(self,control_des,l,u,adj,scaled_lim = 1.0): #*****
        #number of actuators is equal to the rank of the space.
        #TODO:test this!
        scaled_lim = abs(scaled_lim)
        scaled_lim = max(scaled_lim,1.0/scaled_lim)
        # cmd_mtq = mtq_udes - adj[self.mtq_ctrl_mask]
        upper_lim = u+adj#np.fmax(u-xadj,u)
        lower_lim = l+adj#np.fmax(l-xadj,l)
        if np.all(control_des>=lower_lim) and np.all(control_des<=upper_lim): #can match exactly
            return control_des-adj


        #can be parallel?
        upper_list = np.fmin(np.where(control_des>0,upper_lim/control_des,np.inf),np.where(control_des<0,lower_lim/control_des,np.inf));
        lower_list = np.fmax(np.where(control_des>0,lower_lim/control_des,0.0),np.where(control_des<0,upper_lim/control_des,0.0));

        #outside limits serve to sanity check to make sure it's not wildly off (like, its parallel but 100x the magnitude due to weird geometry--if it is, just clip it ) Baseline is only allow if scaling is between 0 and 1. Clip if too big.

        scale_max = min(np.nanmin(upper_list),scaled_lim)
        scale_min = max(np.nanmax(lower_list),0) #don't go negative!
        #TODO: proof

        if scale_min>scale_max:
            #can't keep it parallel--just clip it.
            return np.clip(control_des-adj,l,u)
        if scale_max>=1 and scale_min<=1:
            scale = 1.0
        else:
            if scale_min > 1.0:
                scale = scale_min
            else:
                scale = scale_max
        cmd = control_des*scale - adj
        # scaled_ratio = norm(cmd+adj)/norm(control_des)
        #
        # if scaled_ratio>scaled_lim:# or scaled_ratio <1/scaled_lim:
        #     #is pretty far off, despite being parallel.
        #     return np.clip(control_des-adj,l,u)
        #     #this could be rolled into earlier parts of function--kept separate for clarity for now.

        return cmd


    def general_ctrl_from_torq(self,state,w_err,os,vecs,RW_bias,RW_mom_wt_list,base_torq,sqrt_mtq_wt_list,sqrt_thruster_wt_list,sqrt_RW_wt_list,wdesdot = np.zeros(3)):

        base_torq0 = base_torq.copy()


        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
            base_torq += (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],w_err)@self.sat.J + wdesdot@self.sat.J)

        moddist = self.sat.dist_torque(state,vecs)
        if self.include_disturbances:
            base_torq -= moddist
            self.saved_dist = moddist.copy()

        M = self.RWaxes.T@self.RW_ctrl_matrix + self.Magic_axes.T@self.Magic_ctrl_matrix + skewsym(vecs["b"])@self.MTQ_axes.T@self.MTQ_ctrl_matrix
        #torq = M@u
        sqrtQinv = self.Magic_ctrl_matrix.T@np.diagflat(1/sqrt_thruster_wt_list)@self.Magic_ctrl_matrix + self.RW_ctrl_matrix.T@np.diagflat(1/sqrt_RW_wt_list)@self.RW_ctrl_matrix + self.MTQ_ctrl_matrix.T@np.diagflat(1/sqrt_mtq_wt_list)@self.MTQ_ctrl_matrix
        A = M@sqrtQinv

        #first solve without bounds. minimize 0.5*u.T@Q@u - h_err.T@RW_mom_wt@u subject to base_torq = M@u
        A_pi = np.linalg.pinv(A)
        base_cmd = sqrtQinv@(A_pi@base_torq + (np.eye(self.sat.control_len)-A_pi@A)@sqrtQinv@self.RW_ctrl_matrix@(RW_mom_wt_list*(state[7:self.sat.state_len]-RW_bias)) )


        l0,u0 = self.sat.control_bounds()
        bias = np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])

        if not (np.any(base_cmd-bias>u0) or np.any(base_cmd-bias<l0)): #works based on original goals
            return base_cmd - bias


        #TODO add ther control effort solutions like
        #is there a solution that preserves torque exactly?
        #account fo disturbances and rotational motion more intelligently?

        #TODO add a solution that sees if any combo works exactly. account for disturbances and rotaitonal motion intelligently, and the attempts to despin. plus how to deal with bias.
        #TODO decision tree about things like accounting for bias or not, etc.

        base_cmd = sqrtQinv@(A_pi@base_torq0)# + (np.eye(self.sat.control_len)-A_pi@A)@sqrtQinv@self.RW_ctrl_matrix@(RW_mom_wt_list*(state[7:self.sat.state_len]-RW_bias)) )

        clipped_cmd0 = np.clip(base_cmd,l0,u0)
        new_base_torq = M@base_cmd

        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
            new_base_torq += (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],w_err)@self.sat.J + wdesdot@self.sat.J)

        moddist = self.sat.dist_torque(state,vecs)
        if self.include_disturbances:
            new_base_torq -= moddist
            self.saved_dist = moddist.copy()
        new_base_cmd = sqrtQinv@(A_pi@new_base_torq + (np.eye(self.sat.control_len)-A_pi@A)@sqrtQinv@self.RW_ctrl_matrix@(RW_mom_wt_list*(state[7:self.sat.state_len]-RW_bias)) )

        if not (np.any(new_base_cmd-bias>u0) or np.any(new_base_cmd-bias<l0)): #clipped initial
            return new_base_cmd - bias

        clipped_cmd = np.clip(new_base_cmd,l0,u0)
        return clipped_cmd - bias




    def add_RW_mtq_desat(self,state,os,vecs,RW_bias,c_gain,base_torq):
        base_rw_cmd = np.linalg.pinv(self.RWaxes)@base_torq
        rw_despin = c_gain*(state[7:self.sat.state_len]-RW_bias)
        u_mtq =  np.linalg.pinv(skewsym(vecs["b"])@self.MTQ_axes)@self.RWaxes@rw_despin
        rw_mtq_correction =  np.linalg.pinv(self.RWaxes)@(skewsym(vecs["b"])@self.MTQ_axes@u_mtq  - self.RWaxes@rw_despin)
        u_rw = base_rw_cmd + rw_despin + rw_mtq_correction

        bias = np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])

        l0,u0 = self.sat.control_bounds()
        base_cmd = self.RW_ctrl_matrix.T@u_rw + self.MTQ_ctrl_matrix.T@u_mtq - bias

        if not (np.any(base_cmd>u0) or np.any(base_cmd<l0)): #works based on original goals
            return u_mtq,u_rw

        #TODO add ther control effort solutions like
        #is there a solution by varying u_mtq, correction, & despin?
        #is there a solution that preserves torque exactly?
        #account fo disturbances and rotational motion more intelligently?
        #Try and do weightsings?? Quadratic solution to constant desat??

        cmd0 = self.RW_ctrl_matrix.T@base_rw_cmd - bias
        clipped_cmd0 = np.clip(self.RW_ctrl_matrix.T@base_rw_cmd - bias,l0,u0)

        base_rw_cmd = self.RW_ctrl_matrix@clipped_cmd0
        u_mtq =  np.linalg.pinv(skewsym(vecs["b"])@self.MTQ_axes)@self.RWaxes@rw_despin
        rw_mtq_correction =  np.linalg.pinv(self.RWaxes)@(skewsym(vecs["b"])@self.MTQ_axes@u_mtq  - self.RWaxes@rw_despin)
        u_rw = base_rw_cmd + rw_despin + rw_mtq_correction

        base_cmd = self.RW_ctrl_matrix.T@u_rw + self.MTQ_ctrl_matrix.T@u_mtq - bias

        if not (np.any(base_cmd>u0) or np.any(base_cmd<l0)):
            return u_mtq,u_rw

        clipped_cmd = np.clip(self.RW_ctrl_matrix.T@u_rw + self.MTQ_ctrl_matrix.T@u_mtq,l0,u0)
        return self.MTQ_ctrl_matrix@clipped_cmd,self.RW_ctrl_matrix@clipped_cmd


    #
    def control_from_wdotdes(self,state,os,wdotdes,is_fake,compensate_bias = True,require_parallel = False): #*****TODO: untested
        RW_h = state[7:self.sat.state_len]
        w = state[0:3]
        q = state[3:7]
        dt = 1


        Bbody = rot_mat(q).T@os.B
        # breakpoint()
        ang_mom = w@self.sat.J + RW_h@self.RWaxes
        breakpoint()

        raise ValueError('do not use control_from_wdotdes')
        #
        # # mtqA = np.zeros((3,0))
        # # if self.sat.number_MTQ>0:
        # mtqA = -skewsym(Bbody)@self.MTQ_ctrl_matrix.T# np.hstack([-skewsym(Bbody)@i for i in self.sat.MTQ_axes])
        # # rwA = np.zeros((3,0))
        # # rwA0 = np.zeros((3,0))
        # # if self.sat.number_RW>0:
        # rwA0 = self.RW_ctrl_matrix.T# np.hstack([i for i in self.sat.RW_z_axis])
        # rwA = (np.eye(3)+self.include_rotational_motion*0.5*dt*(skewsym(w+dt*wdotdes/3)))@rwA0
        # # magA = np.zeros((3,0))
        # # if self.sat.number_magic>0:
        # magA = self.Magic_ctrl_matrix.T#np.hstack([i for i in self.sat.magic_axes])
        # Amat = mtqA+rwA+magA
        # Amat0 = mtqA+rwA0+magA
        # A_mod = np.eye(3)+self.include_rotational_motion*0.5*dt*self.sat.invJ_noRW@skewsym(ang_mom)
        # A_add = 0.5*skewsym(w)@(mtqA+magA)
        # # A_addl = np.hstack([np.zeros((3,self.sat.number_MTQ)),A_addl_rw,np.zeros((3,self.sat.number_magic))])
        # # A_addl = np.hstack([np.zeros((3,self.sat.number_MTQ)),A_addl_rw,np.zeros((3,self.sat.number_magic))]) - 0.5*dt*skewsym(w)@(np.eye(3)-self.sat.RWax_mat@np.diagflat(self.sat.RW_J)@self.sat.RWax_mat.T)@Amat
        # # b = self.sat.J_noRW@wdotdes
        # # b_addl = np.cross(w,self.sat.J@w + 0.5*self.sat.RWax_mat@RW_h)
        # # ang_mom_dot_des = self.sat.J@wdotdes + self.sat.RWax_mat@(-np.diagflat(self.sat.RW_J)@self.sat.RWax_mat.T@wdotdes)
        # b = wdotdes@self.sat.J_noRW +  self.include_rotational_motion*np.cross(w + 0.5*wdotdes*dt,ang_mom) +  self.include_rotational_motion*0.5*dt*np.cross(w + wdotdes*dt/3,wdotdes@self.sat.J_noRW)
        # # b_addl += 0.5*dt*(np.cross(wdotdes,ang_mom) + np.cross(w,ang_mom_dot_des) + dt*(np.cross(wdotdes,ang_mom_dot_des)))
        # b_addl = np.zeros(3)
        # if (not is_fake) and self.include_disturbances:
        #     b_addl += self.sat.dist_torque(state,os_local_vecs(os,q))@A_mod
        # bias = np.zeros(self.sat.control_len)
        # if (not is_fake) and compensate_bias:
        #     bias =  np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        #     # mtq_bias,rw_bias,mag_bias = self.sat.sort_control_vec(bias)
        #     # b_addl -= mtqA@mtq_bias#sum([-np.cross(Bbody,self.sat.MTQ_axes[j]*self.sat.act_noise[j].bias) for j in range(self.sat.number_MTQ) if  self.sat.act_noise[j].has_bias])
        #     # b_addl -= rwA@rw_bias#sum([self.sat.RW_z_axis[j]*self.sat.act_noise[self.sat.number_MTQ+j].bias for j in range(self.sat.number_RW) if  self.sat.act_noise[self.sat.number_MTQ+j].has_bias])
        #     # b_addl -= magA@mag_bias#sum([self.sat.magic_axes[j]*self.sat.act_noise[self.sat.number_MTQ+self.sat.number_RW+j].bias for j in range(self.sat.number_magic) if  self.sat.act_noise[self.sat.number_MTQ+self.sat.number_RW+j].has_bias])
        # try:
        #     l0,u0 = self.sat.control_bounds()  #TODO: include RW saturation limits.
        #     # l0,u0 = self.control_constraints(state,os,torq_des = b)
        #     u_ref = scipy.optimize.lsq_linear(Amat,b + b_addl,bounds = (l0.squeeze(),u0.squeeze()))
        #     poss_ctrl = u_ref.x.reshape(self.sat.control_len)
        #     out = poss_ctrl - bias
        #     # poss_torq = Amat@poss_ctrl
        #     # l,u = self.control_constraints(state,os,torq_des = poss_torq)
        #     # u_ref2 = scipy.optimize.lsq_linear(A_mod@Amat0 + A_add,((A_mod@Amat0+A_add)@(poss_ctrl-bias) + b_addl).squeeze(),bounds = (l.squeeze(),u.squeeze()))
        #     # out = u_ref2.x.reshape(self.sat.control_len)
        # except:
        #     breakpoint()
        # if require_parallel:
        #     raise ValueError("not yet implemented")
        # return out











    def control_from_des(self,state,os,control_des,is_fake,compensate_bias = True,compensate_dipole = False,scale_lim = 2): #*****
        # keep total RW/magic torque (inc. bias, if relevant) parallel to desired total RW/magic torque
        # keep total MTQ moment (inc. bias and residual dipole, if relevant) parallel to desired total MTQ moment
        # total RW/magic torque and total MTQ moment, compared to their desired values, should be scaled by the same scalar.
        # respect torque limits, moment limits, and RW saturation limits
        # should preserve total torque direction. does NOT consider the effect of the usage/AM costs between various RW, MTQ, etc vary and this is part of the motivation for the desired value



        mtq_udes = control_des[self.mtq_ctrl_mask]
        magic_udes = control_des[self.magic_ctrl_mask]
        rw_udes = control_des[self.rw_ctrl_mask]

        Nmtq = sum(self.mtq_ctrl_mask)
        Nmag = sum(self.magic_ctrl_mask)
        Nrw = sum(self.rw_ctrl_mask)

                                                # print('ud1',udes)
                                                #operates differently from magic or from RW because it's interaction with torque is nonlinear. a net magnetic moment (including bias) that is parallel to the goal moment is preferered, even if it's vector distance from the desired isn't minimized
        l,u = self.sat.control_bounds() #currently assumes l = -u; TODO: fix
        l_mtq = l[self.mtq_ctrl_mask]
        u_mtq = u[self.mtq_ctrl_mask]
        l_rw = l[self.rw_ctrl_mask]
        u_rw = u[self.rw_ctrl_mask]
        l_magic = l[self.magic_ctrl_mask]
        u_magic = u[self.magic_ctrl_mask]


        if compensate_bias:
            bias = np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        else:
            bias = np.zeros(len(self.sat.actuators))

        res_dipole = np.zeros(3)
        if compensate_dipole:
            dipole_mask = np.array([isinstance(j,Dipole_Disturbance) for j in self.sat.disturbances])
            if (not is_fake) and sum(self.mtq_mask)>0 and np.any(dipole_mask):
                res_dipole = np.sum([j.main_param for j in self.sat.disturbances[dipole_mask]],initial=np.zeros(3))

        #TODO: simplify this code structure--there is redundancy
        if Nrw == 0: #no reaction wheels TODO: add them
            cmd_rw = np.zeros(0)
            if Nmag == 0: #only MTQs
                cmd_mag = np.zeros(0)
                if not compensate_dipole:
                    if self.MTQ_matrix_rank == Nmtq:
                        cmd = self.rankisN_ctrl_adjust(control_des,l,u,bias)
                    # elif self.MTQ_matrix_rank > Nmtq: #not possible
                    else:
                         #self.MTQ_matrix_rank < Nmtq
                         #columns are not linearly independent.
                         if np.all([control_des-bias]<=u) and np.all([control_des-bias]>=l):
                             #can match exactly
                             cmd = control_des-bias
                         else:
                             #can't match exactly
                             #possible to exactly match the magnetic dipole?
                             closest_point = np.linalg.pinv(self.MTQ_axes)@self.MTQ_axes@(control_des-bias) #TODO: move this to matrix inverse calucation at initiation!
                             exact_match_possible = np.all([np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.abs(np.dot(self.MTQ_matrix_row_space[:,j],closest_point)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             # exact_match_possible = np.all([np.all(np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.dot(self.MTQ_matrix_row_space[:,j],bias-control_des)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             #formula above is from intersectino of axis-algined bounding box and a plane, generalized to a line or subspace. Basically, for each independent vector normal to the null space, project the center of the "acceptable box" onto that vector, as well as the corners most aligned with the vector. If the distance of the center from the plane/line/subspace is less than the distance (in projection) from the center to the corner, then the line/plane/subspace passes through the "acceptable box" ALONG THAT VECTOR--must be true for all of these such vectors.
                             if exact_match_possible:
                                 cmd = closest_point
                                 #find it!
                             else:

                                # no exact match is possible. Try to find one that has the dipole parallel to goal.
                                #sanity check to make sure it's not wildly off (like, its parallel but 100x the magnitude due to weird geometry--if it is, just clip it )
                                cmd = closest_point
                else:
                    if self.MTQ_matrix_rank == Nmtq:
                        if self.MTQ_matrix_rank<3:
                            warnings.warn('not full rank of MTQ, cannot correct for dipole. Ignoring dipole correction.')
                        else:
                            control_des[self.mtq_ctrl_mask] -= res_dipole@self.MTQ_matrix_inv
                        cmd = self.rankisN_ctrl_adjust(control_des,l,u,bias)
                    else:
                         #self.MTQ_matrix_rank < Nmtq
                         #columns are not linearly independent.
                         if self.MTQ_matrix_rank<3:
                             warnings.warn('not full rank of MTQ, cannot correct for dipole. Ignoring dipole correction.')
                         else:
                             control_des[self.mtq_ctrl_mask] -= res_dipole@self.MTQ_matrix_inv
                         if np.all([control_des-bias]<=u) and np.all([control_des-bias]>=l):
                             #can match exactly
                             cmd = control_des-bias
                         else:
                             #can't match exactly
                             #possible to exactly match the magnetic dipole?
                             closest_point = np.linalg.pinv(self.MTQ_axes)@self.MTQ_axes@(control_des-bias) #TODO: move this to matrix inverse calucation at initiation!
                             exact_match_possible = np.all([np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.abs(np.dot(self.MTQ_matrix_row_space[:,j],closest_point)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             # exact_match_possible = np.all([np.all(np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.dot(self.MTQ_matrix_row_space[:,j],bias-control_des)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             #formula above is from intersectino of axis-algined bounding box and a plane, generalized to a line or subspace. Basically, for each independent vector normal to the null space, project the center of the "acceptable box" onto that vector, as well as the corners most aligned with the vector. If the distance of the center from the plane/line/subspace is less than the distance (in projection) from the center to the corner, then the line/plane/subspace passes through the "acceptable box" ALONG THAT VECTOR--must be true for all of these such vectors.
                             if exact_match_possible:
                                 cmd = closest_point
                                 #find it!
                             else:

                                # no exact match is possible. Try to find one that has the dipole parallel to goal.
                                #sanity check to make sure it's not wildly off (like, its parallel but 100x the magnitude due to weird geometry--if it is, just clip it )
                                cmd = closest_point
            else:
                if not compensate_dipole:
                    if self.MTQ_matrix_rank == Nmtq:
                        cmd = self.rankisN_ctrl_adjust(control_des,l,u,bias)
                    # elif self.MTQ_matrix_rank > Nmtq: #not possible
                    else:
                         #self.MTQ_matrix_rank < Nmtq
                         #columns are not linearly independent.
                         if np.all([control_des-bias]<=u) and np.all([control_des-bias]>=l):
                             #can match exactly
                             cmd = control_des-bias
                         else:
                             #can't match exactly
                             #possible to exactly match the magnetic dipole?
                             closest_point = np.linalg.pinv(self.MTQ_axes)@self.MTQ_axes@(control_des-bias) #TODO: move this to matrix inverse calucation at initiation!
                             exact_match_possible = np.all([np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.abs(np.dot(self.MTQ_matrix_row_space[:,j],closest_point)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             # exact_match_possible = np.all([np.all(np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.dot(self.MTQ_matrix_row_space[:,j],bias-control_des)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             #formula above is from intersectino of axis-algined bounding box and a plane, generalized to a line or subspace. Basically, for each independent vector normal to the null space, project the center of the "acceptable box" onto that vector, as well as the corners most aligned with the vector. If the distance of the center from the plane/line/subspace is less than the distance (in projection) from the center to the corner, then the line/plane/subspace passes through the "acceptable box" ALONG THAT VECTOR--must be true for all of these such vectors.
                             if exact_match_possible:
                                 cmd = closest_point
                                 #find it!
                             else:

                                # no exact match is possible. Try to find one that has the dipole parallel to goal.
                                #sanity check to make sure it's not wildly off (like, its parallel but 100x the magnitude due to weird geometry--if it is, just clip it )
                                cmd = closest_point
                else:
                    if self.MTQ_matrix_rank == Nmtq:
                        if self.MTQ_matrix_rank<3:
                            warnings.warn('not full rank of MTQ, cannot correct for dipole. Ignoring dipole correction.')
                        else:
                            control_des[self.mtq_ctrl_mask] -= res_dipole@self.MTQ_matrix_inv
                        cmd = self.rankisN_ctrl_adjust(control_des,l,u,bias)
                    else:
                         #self.MTQ_matrix_rank < Nmtq
                         #columns are not linearly independent.
                         if self.MTQ_matrix_rank<3:
                             warnings.warn('not full rank of MTQ, cannot correct for dipole. Ignoring dipole correction.')
                         else:
                             control_des[self.mtq_ctrl_mask] -= res_dipole@self.MTQ_matrix_inv
                         if np.all([control_des-bias]<=u) and np.all([control_des-bias]>=l):
                             #can match exactly
                             cmd = control_des-bias
                         else:
                             #can't match exactly
                             #possible to exactly match the magnetic dipole?
                             closest_point = np.linalg.pinv(self.MTQ_axes)@self.MTQ_axes@(control_des-bias) #TODO: move this to matrix inverse calucation at initiation!
                             exact_match_possible = np.all([np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.abs(np.dot(self.MTQ_matrix_row_space[:,j],closest_point)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             # exact_match_possible = np.all([np.all(np.dot(u,np.abs(self.MTQ_matrix_row_space[:,j]))>=np.dot(self.MTQ_matrix_row_space[:,j],bias-control_des)) for j in range(self.MTQ_matrix_rank)]) #TODO: test this!!!
                             #formula above is from intersectino of axis-algined bounding box and a plane, generalized to a line or subspace. Basically, for each independent vector normal to the null space, project the center of the "acceptable box" onto that vector, as well as the corners most aligned with the vector. If the distance of the center from the plane/line/subspace is less than the distance (in projection) from the center to the corner, then the line/plane/subspace passes through the "acceptable box" ALONG THAT VECTOR--must be true for all of these such vectors.
                             if exact_match_possible:
                                 cmd = closest_point
                                 #find it!
                             else:

                                # no exact match is possible. Try to find one that has the dipole parallel to goal.
                                #sanity check to make sure it's not wildly off (like, its parallel but 100x the magnitude due to weird geometry--if it is, just clip it )
                                cmd = closest_point



        return cmd

    def reduced_state_err(self,state,desired,quatvecmode):#TODO: not tested
        if isinstance(desired,state_goal):
            desired = desired.state
        q = state[3:7]
        w = state[0:3]
        extra = state[7:self.sat.state_len]
        q_desired = desired[3:7]
        w_desired = desired[0:3]
        # extra_desired = desired[7:self.sat.state_len]
        q_desired = normalize(q_desired)
        q_err = quat_mult(quat_inv(q_desired),q)#np.vstack([np.array([np.ndarray.item(q.T@q_desired)]),-q_desired[1:]*q[0]+q[1:]*q_desired[0]-np.cross(q_desired[1:],q[1:])])
        q_err = normalize(q_err)
        # breakpoint()
        return np.concatenate([w-w_desired@rot_mat(q_err),quat_to_vec3(q_err,quatvecmode)])#,extra-extra_desired])
        # return np.concatenate([w-w_desired,quat_to_vec3(q_err,quatvecmode),extra-extra_desired])

    def reduced_state_err_jac(self,state,desired,quatvecmode):#TODO: not tested

        if isinstance(desired,state_goal):
            desired = desired.state
        q = state[3:7]
        q_desired = desired[3:7]
        q_desired = normalize(q_desired)
        qides = quat_inv(q_desired)
        q_err = quat_mult(qides,q)
        nq_err = normalize(q_err)
        hlen = self.sat.state_len-7
        res = np.zeros((self.sat.state_len,self.sat.state_len-1))
        res[0:3,0:3] = np.eye(3)
        if hlen>0:
            res[-hlen:,-hlen:] = np.eye(hlen)
        # breakpoint()
        res[3:7,3:6] = quat_left_mult_matrix(qides).T@quat_norm_jac(q_err)@quat_to_vec3_deriv(nq_err,quatvecmode)
        return res

    def reduced_state_err_hess(self,state,desired,quatvecmode):#TODO: not tested

        if isinstance(desired,state_goal):
            desired = desired.state
        q = state[3:7]
        q_desired = desired[3:7]
        q_desired = normalize(q_desired)
        qides = quat_inv(q_desired)
        q_err = quat_mult(qides,q)
        nq_err = normalize(q_err)

        dqe__dq = quat_left_mult_matrix(qides)
        dnqe__dqe = quat_norm_jac(q_err)
        dve__dnqe = quat_to_vec3_deriv(nq_err,quatvecmode)
        dve__dq = dqe__dq@dnqe__dqe@dve__dnqe

        ddve__dqednqei = [x@dnqe__dqe for x in quat_to_vec3_deriv2(nq_err,quatvecmode)]
        ddnqe__dqedqei = quat_norm_hess(q_err)
        ddve__dqedqei = multi_matrix_chain_rule_vector([dnqe__dqe],[ddve__dqednqei],1,[0])
        ddve__dqdqei = [dqe__dq@(x@dve__dnqe+y) for x,y in zip(ddnqe__dqedqei,ddve__dqedqei)]

        ddve__dqdqi = multi_matrix_chain_rule_vector([dqe__dq],[ddve__dqdqei],1,[0])
        output = [np.zeros((self.sat.state_len-1,self.sat.state_len)) for j in range(self.sat.state_len)]
        for k in range(4):
            dderr__dqdqi = output[3+k]
            dderr__dqdqi[3:6,3:7] = ddve__dqdqi[k]
            output[3+k] = dderr__dqdqi
        return output
