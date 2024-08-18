from .control_mode import *

class Magic_Reduced_PD(ControlMode):
    def __init__(self,gain_info,sat,include_disturbances=True):
        ModeName = GovernorMode.Reduced_Magic_PD

        params = Params()

        params.w_gain = gain_info[0]
        params.normal_gain = gain_info[1]
        params.dt = gain_info[2]
        # params.gain = gain
        super().__init__(ModeName,sat,params,True,include_disturbances,False,False)


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

        avec = goal_state.body_vec
        bvec = goal_state.eci_vec
        bvec_b = bvec@rot_mat(q)
        avec_vel_des = 0
        avec_vel_des_dot = 0
        bvec_vel_des = 0
        bvec_vel_des_dot = 0
        bd_prev_inertial = (goal_state.eci_vec - prev_goal.eci_vec)/self.params.dt
        bd_next_inertial = (next_goal.eci_vec - goal_state.eci_vec)/self.params.dt
        dot_bvec_inertial = 0.5*(bd_prev_inertial+bd_next_inertial)
        ddot_bvec_inertial = (bd_next_inertial-bd_prev_inertial)/self.params.dt
        dot_bvec_b =  -np.cross(state[0:3],goal_state.eci_vec) +  dot_bvec_inertial@rot_mat(q)
        # ddot_bvec_b = -np.cross(w_dot,goal_state.eci_vec) - 2*np.cross(state[0:3],dot_bvec_b) - np.cross(state[0:3],np.cross(state[0:3],bvec_b)) + ddot_bvec_inertial@rot_mat(q)
        errvec = np.cross(bvec_b,avec)
        errang = norm(errvec)
        n_errvec = normalize(errvec)
        w_desired = avec_vel_des*avec + bvec_vel_des*bvec_b + np.cross(bvec_b,dot_bvec_inertial@rot_mat(q))

        # n_errvec_dot = np.cross(dot_bvec_b,avec)/np.sin(-errang) + mydot(avec,dot_bvec_b)*n_errvec*np.cos(errang)/np.sin(-errang)**2.0

        w_desired_dot = avec_vel_des_dot*avec + bvec_vel_des_dot*bvec_b + bvec_vel_des_dot*dot_bvec_b \
                        - np.cross(state[0:3],np.cross(bvec_b,dot_bvec_inertial@rot_mat(q))) + np.cross(bvec_b,ddot_bvec_inertial@rot_mat(q))


        w_err = state[0:3]-w_desired


        base_torq = -1.0*(errvec@self.params.normal_gain + w_err@self.params.w_gain)
        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
            base_torq += (np.cross(state[0:3],ang_mom) + w_desired_dot@self.sat.J)

        if self.include_disturbances:
            moddist = self.sat.dist_torque(state,vecs)
            base_torq -= moddist
            self.saved_dist = moddist.copy()

        return self.control_from_wdotdes(state,os,base_torq@self.sat.invJ,is_fake)
        # return self.mtqrw_from_torqdes(self.bdot_gain*(-self.sat.J@w/self.update_period + cross(w,self.sat.J@w + RW_h)),RW_h,Bbody,is_fake)
