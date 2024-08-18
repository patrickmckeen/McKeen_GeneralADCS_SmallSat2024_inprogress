from .control_mode import *

class Magic_PD(ControlMode):
    def __init__(self,gain_info,sat,maintain_RW = True,include_disturbances=False,quatset=False,quatset_type = "Ang",calc_av_from_quat = False,include_rotational_motion = False):
        if not quatset:
            ModeName = GovernorMode.WIE_MAGIC_PD
        else:
            if quatset_type == "Ang":
                ModeName = GovernorMode.WIE_MAGIC_PD_QUATSET_ANG
            elif quatset_type[:-1] == "Lyap":
                ModeName = GovernorMode.WIE_MAGIC_PD_QUATSET_LYAP
            elif quatset_type[:-1] == "LyapRmat":
                ModeName = GovernorMode.WIE_MAGIC_PD_QUATSET_LYAPR
            else:
                print(quatset_type)
                print(quatset_type[:-1])
                raise ValueError("incorrect quatset class of Magic thruster control (Wie)")

                # self.gain = gain
        params = Params()
        params.w_gain = gain_info[0]
        params.q_gain = gain_info[1]
        params.Tc_vec = np.array([j.max for j in sat.actuators if isinstance(j,Magic)])
        if quatset and quatset_type[:-1][:4] == "Lyap":
            # print('qtype',quatset_type[-1])
            params.lyap_type = int(quatset_type[-1]) #0,1,2
        super().__init__(ModeName,sat,params,maintain_RW,include_disturbances,calc_av_from_quat,include_rotational_motion)
        if sum(self.magic_mask) != 3:
            raise ValueError('This is currently only implemented for exactly 3 Magic thrusters')
        if np.linalg.matrix_rank(self.Magic_matrix) != 3:
            raise ValueError('Magic thruster axes need full rank')

    def baseline_actuation(self,state,qerr,werr,vecs,other_err=None):
        base_torq = -1.0*(qerr[1:]@self.params.q_gain + werr@self.params.w_gain)*self.params.Tc_vec
        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
            base_torq += (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],werr)@self.sat.J)

        if self.include_disturbances:
            moddist = self.sat.dist_torque(state,vecs)
            base_torq -= moddist
            self.saved_dist = moddist.copy()
        u = self.magic_command_maintain_RW(base_torq,state,vecs)
        self.addl_info["desired_torq"] = base_torq
        return u

    def baseline_actuation_jac_over_qerr(self,state,qerr,werr,vecs,other_err=None):
        jac = -1.0*(quat2vec_mat.T*self.params.q_gain)*self.params.Tc_vec
        # u = self.magic_command_maintain_RW(base_torq,state,vecs) #TODO: add this.
        return jac


    def find_actuation(self, state, os, osp1,goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        q = state[3:7]
        vecs = os_local_vecs(os,q)
        if self.modename is GovernorMode.WIE_MAGIC_PD:
            # if self.calc_av_from_quat:
            #     wgoal = self.av_from_quat_goals(state,goal_state,next_goal,(cent2sec*(osp1.J2000-os.J2000)))
            err = self.state_err(state,goal_state,next_desired = next_goal,dt= (cent2sec*(osp1.J2000-os.J2000)))
            q_err = err[3:7]
            q_err *= np.sign(q_err[0])
            w_err = err[0:3]
        else:
            if self.modename is GovernorMode.WIE_MAGIC_PD_QUATSET_ANG:
                w_err,q_err = self.minang_err(state,goal_state,next_goal,(osp1.J2000-os.J2000)*cent2sec)
            elif self.modename is GovernorMode.WIE_MAGIC_PD_QUATSET_LYAP:
                w_err,q_err = self.minLyap_err(state,goal_state,vecs,next_goal,(osp1.J2000-os.J2000)*cent2sec,qmult = np.mean(np.diag(self.params.q_gain))*np.mean(self.params.Tc_vec),av_mat_mult = np.eye(3))
            elif self.modename is GovernorMode.WIE_MAGIC_PD_QUATSET_LYAPR:
                w_err,q_err = self.minLyapR_err(state,goal_state,vecs,next_goal,(osp1.J2000-os.J2000)*cent2sec,qmult = np.mean(np.diag(self.params.q_gain))*np.mean(self.params.Tc_vec),av_mat_mult = np.eye(3))
            else:
                raise ValueError("Incorrect mode for Wie Magic")
        return self.baseline_actuation(state,q_err,w_err,vecs)
        # # print(kw*w_err/self.lovera_beta)
        # # print(limit_vec(kw*w_err/self.lovera_beta,1.0))
        # # udes = -1.0*self.sat.invJ@(self.lovera_eps**2.0 * kp * q_err[1:,:] + self.lovera_eps*self.lovera_beta * limit_vec(kw*w_err/self.lovera_beta,1.0))
        # base_torq = -1.0*(q_err[1:]@self.params.q_gain + w_err@self.params.w_gain)*self.params.Tc_vec
        # print((180/np.pi)*(4.0*(quat_to_mrp(state[3:7])/2.0)))
        # print((180/np.pi)*(4.0*(quat_to_mrp(q_err)/2.0)))
        # print('q',q_err[1:],q_err[1:]@self.params.q_gain*self.params.Tc_vec,norm(q_err[1:]@self.params.q_gain*self.params.Tc_vec))
        # print('w',w_err,w_err@self.params.w_gain*self.params.Tc_vec,norm(w_err@self.params.w_gain*self.params.Tc_vec))
        # print(state[0:3])
        # # print(self.params.Tc_vec)
        # if self.include_disturbances:
        #     base_torq -= self.sat.dist_torque(state,vecs)
        # u = self.magic_command_maintain_RW(base_torq,state,vecs)
        # self.addl_info["desired_torq"] = base_torq
        # return u
