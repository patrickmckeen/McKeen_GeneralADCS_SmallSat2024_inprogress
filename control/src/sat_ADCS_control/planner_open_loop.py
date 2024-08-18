from .control_mode import *

class PlannerOpenLoop(ControlMode):
    def __init__(self,gain_info,sat):
        # ModeName = GovernorMode.ΤRAJ_MAG_PD

        # self.gain = gain
        # params = Params()
        ModeName = GovernorMode.PLAN_OPEN_LOOP
        params = Params()
        params.gain = gain
        super().__init__(ModeName,sat,params,False,False,False,False)
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
        This function applies the control input from the original plan, adjusting for bias.
        """

        return planner_params[0][1] - np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
