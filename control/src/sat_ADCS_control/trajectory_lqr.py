from .control_mode import *

class TrajectoryLQR(ControlMode):
    def __init__(self,gain_info,sat,maintain_RW = True,tracking_LQR_formulation=0,include_disturbances=False,calc_av_from_quat = False,include_rotational_motion = False):
        ModeName = GovernorMode.PLAN_AND_TRACK_LQR
        self.tracking_LQR_formulation = tracking_LQR_formulation

        # self.gain = gain
        params = Params()
        super().__init__(ModeName,sat,params,maintain_RW,include_disturbances,calc_av_from_quat,include_rotational_motion)
        if sum(self.mtq_mask) not in [0,3]:
            raise ValueError('This is currently only implemented for exactly 0 or 3 MTQs') #TO-DO: include with more, by averaging?
        if sum(self.mtq_mask) == 3:
            if self.MTQ_matrix_rank not in [0,3]:
                raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.




    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        """
        This function finds the commanded control input using bdot at a specific point
        in a trajectory, based simply on the derivative of the magnetic field. Equivalent
        to control mode GovernorMode.SIMPLE_BDOT.

        Parameters
        ------------
            db_body: np array (3 x 1)
                derivative of the magnetic field in body coordinates, in T
        Returns
        ---------
            u_out: np array (3 x 1)
                magnetic dipole to actuate for bdot, in Am^2 and body coordinates
        """
        q = state[3:7]
        vecs = os_local_vecs(os,q)
        plan_state = planner_params[0][-1]
        plan_control = planner_params[0][1]
        plan_gain = planner_params[0][2]
        planned_torq = planner_params[0][4]

        err = self.reduced_state_err(state,plan_state,quatvecmode=0)
        if self.tracking_LQR_formulation==0:
            u = plan_control - err@plan_gain.T
        elif self.tracking_LQR_formulation==1:
            u = plan_control - np.concatenate([state,[1]])@plan_gain.T
        elif self.tracking_LQR_formulation==2:
            torq_err = self.sat.dist_torque(state,vecs) - planned_torq
            u = plan_control - np.concatenate([err,torq_err])@plan_gain.T#err #TODO better use the torque here--it should be the difference between the disturbing torque being experied/estimated and those that were planned for (if GG was in planner but not controller, GG should be taken out of dist torq est)
        else:
            raise ValueError("need correct number of LQR tracking formulation")
        return self.control_from_des(state,os,u,is_fake)
