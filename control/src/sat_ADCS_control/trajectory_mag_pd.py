from .control_mode import *
from .lovera_pd import *

class Trajectory_Mag_PD(Lovera):
    def __init__(self,gain_info,sat,maintain_RW = True,include_disturbances=True,calc_av_from_quat = False,include_rotational_motion = True):
        # ModeName = GovernorMode.ΤRAJ_MAG_PD

        # self.gain = gain
        # params = Params()
        super().__init__(gain_info,sat,maintain_RW,include_disturbances,False,"",calc_av_from_quat,include_rotational_motion)
        self.modename = GovernorMode.TRAJ_MAG_PD
        # self.maintain_RW = maintain_RW
        # self.include_disturbances = include_disturbances
        #
        # # self.Bbody_mat = np.array([1/self.sat.sensors[j].scale for j in range(len(sens)) if self.mtm_reading_mask[j]])@self.mtm_axes_mat_inv
        #
        # self.mtq_mask = np.array([isinstance(j,MTQ) for j in self.sat.actuators])
        # self.mtq_max = np.array([j.max for j in self.sat.actuators])
        #
        # self.mtq_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,MTQ) for j in self.sat.actuators]).astype(bool)
        # if sum(self.mtq_mask) != 3:
        #     raise ValueError('This is currently only implemented for exactly 3 MTQs') #TO-DO: include with more, by averaging?
        # # self.MTQ_matrix_inv = np.linalg.inv(np.stack([j.axis if isinstance(j,MTQ) for j in self.sat.actuators]))
        # #
        # # self.MTQ_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))#np.stack([self.MTQ_matrix_inv[j,:] if isinstance(j,MTQ) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
        # # self.MTQ_ctrl_matrix[:,self.mtq_ctrl_mask] = self.MTQ_matrix_inv
        # # self.RWaxes = np.stack([j.axis if isinstance(j,RW) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
        # # self.rw_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,RW) for j in self.sat.actuators]).astype(bool)
        # #
        # #
        # # self.RWjs = self.diagflat(np.array([self.sat.actuators[j].J for j in self.sat.momentum_inds]))
        # # self.RWaxes = np.stack([self.sat.actuators[j].axis for j in self.sat.momentum_inds])
        # # # self.mtq_axes_mat = np.stack([self.sat.attitude_sensors[j].axis for j in range(len(self.sat.attitude_sensors)) if self.mtm_mask[j]])
        # if self.MTQ_matrix_rank != 3:
        #     raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.


    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        """
        This function finds the commanded control input using the Lovera Magnetic PD control law at a specific point
        in a trajectory. Equivalent
        to control mode GovernorMode.TRAJ_MAG_PD.
        """
        q = state[3:7]
        plan_goal = planner_params[0][-1]
        plan_goal_next = planner_params[1][-1]
        vecs = os_local_vecs(os,q)
        err = self.state_err(state,plan_goal,next_desired = plan_goal_next,dt= (cent2sec*(osp1.J2000-os.J2000)))
        q_err = err[3:7]
        q_err *= np.sign(q_err[0])
        w_err = err[0:3]

        return self.baseline_actuation(state,q_err,w_err,vecs)
