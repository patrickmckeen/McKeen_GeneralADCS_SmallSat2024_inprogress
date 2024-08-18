from .control_mode import *

class RW_PD(ControlMode):
    def __init__(self,gain_info,sat,include_disturbances=True,include_rotational_motion=True):
        ModeName = GovernorMode.RW_PID

        params = Params()

        params.w_gain = gain_info[0]
        params.q_gain = gain_info[1]
        params.max_vec = np.array([j.max for j in sat.actuators if isinstance(j,RW)])

        # params.gain = gain
        super().__init__(ModeName,sat,params,False,include_disturbances,False,include_rotational_motion)


    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        """
        This function finds the commanded control input using bdot at a specific point
        in a trajectory, based on the rate of change of the magnetic field, estimated
        using the angular velocity and current magnetic field reading. Equivalent
        to control mode GovernorMode.BDOT_WITH_EKF.

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
        err = self.state_err(state,goal_state,next_desired = next_goal,dt= (cent2sec*(osp1.J2000-os.J2000)))
        q_err = err[3:7]
        q_err *= np.sign(q_err[0])
        w_err = err[0:3]
        base_torq = -1.0*(q_err[1:]@self.params.q_gain + w_err@self.params.w_gain)*self.params.max_vec
        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
            base_torq += (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],w_err)@self.sat.J)

        if self.include_disturbances:
            moddist = self.sat.dist_torque(state,vecs)
            base_torq -= moddist
            self.saved_dist = moddist.copy()

        # return self.control_from_wdotdes()
        return self.RW_ctrl_matrix.T@base_torq - np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        # return self.mtqrw_from_torqdes(self.bdot_gain*(-self.sat.J@w/self.update_period + cross(w,self.sat.J@w + RW_h)),RW_h,Bbody,is_fake)
