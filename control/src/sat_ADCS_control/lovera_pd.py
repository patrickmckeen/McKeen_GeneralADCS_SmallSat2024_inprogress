from .control_mode import *

class Lovera(ControlMode):
    def __init__(self,gain_info,sat,maintain_RW = True,include_disturbances=False,quatset=False,quatset_type = "B",calc_av_from_quat = False,include_rotational_motion = False):
        if not quatset:
            ModeName = GovernorMode.LOVERA_MAG_PD
        else:
            if quatset_type == "B":
                ModeName = GovernorMode.LOVERA_MAG_PD_QUATSET_B
            elif quatset_type == "Ang":
                ModeName = GovernorMode.LOVERA_MAG_PD_QUATSET_ANG
            elif quatset_type[:-1] == "Lyap":
                ModeName = GovernorMode.LOVERA_MAG_PD_QUATSET_LYAP
            elif quatset_type[:-1] == "LyapRmat":
                ModeName = GovernorMode.LOVERA_MAG_PD_QUATSET_LYAPR
            # elif quatset_type == "Bnow":
            #     ModeName = GovernorMode.LOVERA_MAG_PD_QUATSET_Bnow
            else:
                raise ValueError("incorrect quatset class of Lovera")

        # self.gain = gain
        params = Params()
        params.gain_eps = gain_info[0]
        params.kp_gain = gain_info[1]
        params.kv_gain = gain_info[2]
        if quatset and quatset_type[:-1][:4] == "Lyap":
            params.lyap_type = int(quatset_type[-1]) #0,1,2
        super().__init__(ModeName,sat,params,maintain_RW,include_disturbances,calc_av_from_quat,include_rotational_motion)
        # self.maintain_RW = maintain_RW
        # self.include_disturbances = include_disturbances
        #
        # # self.Bbody_mat = np.array([1/self.sat.sensors[j].scale for j in range(len(sens)) if self.mtm_reading_mask[j]])@self.mtm_axes_mat_inv
        #
        # self.mtq_mask = np.array([isinstance(j,MTQ) for j in self.sat.actuators])
        # self.mtq_max = np.array([j.max for j in self.sat.actuators])
        #
        # self.mtq_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,MTQ) for j in self.sat.actuators]).astype(bool)
        if sum(self.mtq_mask) != 3:
            raise ValueError('This is currently only implemented for exactly 3 MTQs') #TO-DO: include with more, by averaging?
        # self.MTQ_matrix_inv = np.linalg.inv(np.stack([j.axis if isinstance(j,MTQ) for j in self.sat.actuators]))
        #
        #
        # self.RWaxes = np.stack([self.sat.actuators[j].axis for j in self.sat.momentum_inds])
        # # self.mtq_axes_mat = np.stack([self.sat.attitude_sensors[j].axis for j in range(len(self.sat.attitude_sensors)) if self.mtm_mask[j]])
        if self.MTQ_matrix_rank != 3:
            raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.


    def baseline_actuation(self,state,qerr,werr,vecs,other_err=None):
        base_torq = -(self.params.gain_eps*self.params.gain_eps*self.params.kp_gain*qerr[1:] + self.params.kv_gain*self.params.gain_eps*werr)@self.sat.invJ

        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
            base_torq += (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],werr)@self.sat.J)


        # print((self.params.gain_eps*self.params.gain_eps*self.params.kp_gain*qerr[1:])@self.sat.invJ,norm((self.params.gain_eps*self.params.gain_eps*self.params.kp_gain*qerr[1:])@self.sat.invJ))
        # print((self.params.kv_gain*self.params.gain_eps*werr)@self.sat.invJ,norm((self.params.kv_gain*self.params.gain_eps*werr)@self.sat.invJ))
        if self.include_disturbances:
            moddist = self.sat.dist_torque(state,vecs)
            base_torq -= moddist
            self.saved_dist = moddist.copy()
        mtq_cmd = np.cross(vecs["b"],base_torq)/(norm(vecs["b"])**2.0)
        u = self.heritage_mtq_command_maintain_RW(mtq_cmd,state,vecs)
        return u


    def baseline_actuation_jac_over_qerr(self,state,qerr,werr,vecs,other_err=None):
        jac = -(self.params.gain_eps*self.params.gain_eps*self.params.kp_gain*quat2vec_mat.T)@self.sat.invJ
        mtq_cmd_jac = -jac@skewsym(vecs["b"])/(norm(vecs["b"])**2.0)
        # u =self.mtq_command_maintain_RW(mtq_cmd,state,vecs)#TODO: add this.
        return mtq_cmd_jac

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
        if self.modename is GovernorMode.LOVERA_MAG_PD:
            # if self.calc_av_from_quat:
            #     wgoal = self.av_from_quat_goals(state,goal_state,next_goal,(cent2sec*(osp1.J2000-os.J2000)))
            err = self.state_err(state,goal_state,next_desired = next_goal,dt= (cent2sec*(osp1.J2000-os.J2000)))
            q_err = err[3:7]
            q_err *= np.sign(q_err[0])
            w_err = err[0:3]
        else:
            if self.modename is GovernorMode.LOVERA_MAG_PD_QUATSET_B:
                w_err,q_err = self.perpB_err(state,goal_state,next_goal,(osp1.J2000-os.J2000)*cent2sec,vecs["b"])
            elif self.modename is GovernorMode.LOVERA_MAG_PD_QUATSET_ANG:
                w_err,q_err = self.minang_err(state,goal_state,next_goal,(osp1.J2000-os.J2000)*cent2sec)
            elif self.modename is GovernorMode.LOVERA_MAG_PD_QUATSET_LYAP:
                w_err,q_err = self.minLyap_err(state,goal_state,vecs,next_goal,(osp1.J2000-os.J2000)*cent2sec,qmult = self.params.gain_eps*self.params.gain_eps*self.params.kp_gain,av_mat_mult = np.eye(3))
            elif self.modename is GovernorMode.LOVERA_MAG_PD_QUATSET_LYAPR:
                w_err,q_err = self.minLyapR_err(state,goal_state,vecs,next_goal,(osp1.J2000-os.J2000)*cent2sec,qmult = self.params.gain_eps*self.params.gain_eps*self.params.kp_gain,av_mat_mult = np.eye(3))
            # elif self.modename is GovernorMode.LOVERA_MAG_PD_QUATSET_Bnow:
            #     pass
            else:
                raise ValueError("Incorrect mode for Lovera")

        return self.baseline_actuation(state,q_err,w_err,vecs)
        # base_torq = -(self.params.gain_eps*self.params.gain_eps*self.params.kp_gain*q_err[1:] + self.params.kv_gain*self.params.gain_eps*w_err)@self.sat.invJ
        # if self.include_disturbances:
        #     base_torq -= self.sat.dist_torque(state,vecs)
        # mtq_cmd = np.cross(vecs["b"],base_torq)/(norm(vecs["b"])**2.0)
        # u = self.mtq_command_maintain_RW(mtq_cmd,state,vecs)
        # self.addl_info["desired_torq"] = base_torq
        # #TODO deal with case where bdot is being run and mtqs are very close to boundary--basically add control mode that keeps wheels at some ang_mom when other torques are happening.
        # return u
