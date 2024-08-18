from .control_mode import *

class WisniewskiTwisting(ControlMode):
    def __init__(self,gain_info,sat,maintain_RW = True,include_disturbances=False,quatset=False,quatset_type = "B",calc_av_from_quat = False,include_rotational_motion = True):
        if not quatset:
            ModeName = GovernorMode.WISNIEWSKI_TWISTING
        else:
            if quatset_type == "B":
                ModeName = GovernorMode.WISNIEWSKI_TWISTING_QUATSET_B
            elif quatset_type == "Ang":
                ModeName = GovernorMode.WISNIEWSKI_TWISTING_QUATSET_ANG
            elif quatset_type[:-1] == "Lyap":
                ModeName = GovernorMode.WISNIEWSKI_TWISTING_QUATSET_LYAP
            elif quatset_type == "S":
                ModeName = GovernorMode.WISNIEWSKI_TWISTING_QUATSET_MINS
            elif quatset_type[:-1] == "LyapRmat":
                ModeName = GovernorMode.WISNIEWSKI_TWISTING_QUATSET_LYAPR
            elif quatset_type == "BS":
                ModeName = GovernorMode.WISNIEWSKI_TWISTING_QUATSET_BS
            else:
                raise ValueError("incorrect quatset class of WisniewskiTwisting")

        # self.gain = gain
        params = Params()
        params.Lambda_q = gain_info[0]
        params.Lambda_s = gain_info[1]
        if len(gain_info)>2:
            params.Lambda_All = gain_info[2]
        else:
            params.Lambda_All = np.eye(3)
        if len(gain_info)>3:
            params.Lambda_q2 = gain_info[3]
        else:
            params.Lambda_q2 = np.zeros((3,3))
        if len(gain_info)>4:
            params.qexp = gain_info[4]
        else:
            params.qexp = 1.0
        if len(gain_info)>5:
            params.sexp = gain_info[5]
        else:
            params.sexp = 0.5
        if len(gain_info)>6:
            params.const_int = gain_info[6]
        else:
            params.const_int = 0
        if len(gain_info)>7:
            params.q0exp_1 = gain_info[7]
        else:
            params.q0exp_1 = 0
        if len(gain_info)>8:
            params.q0exp_2 = gain_info[8]
        else:
            params.q0exp_2 = 0
        params.inv_Lambda_All = np.linalg.inv(params.Lambda_All)
        if quatset and quatset_type[:-1][:4] == "Lyap":
            params.lyap_type = int(quatset_type[-1]) #0,1,2
        super().__init__(ModeName,sat,params,maintain_RW,include_disturbances,calc_av_from_quat,include_rotational_motion)

        self.s = np.nan*np.ones(3)
        self.s1 = np.nan*np.ones(3)
        self.s2 = np.nan*np.ones(3)
        self.s3 = np.nan*np.ones(3)
        self.int_term = np.zeros(3)
        self.s_ECI = np.nan*np.ones(3)
        self.s1_ECI = np.nan*np.ones(3)
        self.s2_ECI = np.nan*np.ones(3)
        self.s3_ECI = np.nan*np.ones(3)
        self.s1d = np.nan*np.ones(3)
        self.s2d = np.nan*np.ones(3)
        self.s3d = np.nan*np.ones(3)
        self.s1d_2o = np.nan*np.ones(3)
        self.s2d_2o = np.nan*np.ones(3)
        self.s3d_2o = np.nan*np.ones(3)
        self.torq = np.nan*np.ones(3)
        self.wd = np.nan*np.ones(3)
        self.wd_ECI = np.nan*np.ones(3)
        self.we = np.nan*np.ones(3)
        self.we_ECI = np.nan*np.ones(3)
        self.quaterr = np.nan*np.ones(4)

        self.prev_s = np.nan*np.ones(3)
        self.prev_s1 = np.nan*np.ones(3)
        self.prev_s2 = np.nan*np.ones(3)
        self.prev_s3 = np.nan*np.ones(3)
        self.prev_s_ECI = np.nan*np.ones(3)
        self.prev_s1_ECI = np.nan*np.ones(3)
        self.prev_s2_ECI = np.nan*np.ones(3)
        self.prev_s3_ECI = np.nan*np.ones(3)
        self.prev_s1d = np.nan*np.ones(3)
        self.prev_s2d = np.nan*np.ones(3)
        self.prev_s3d = np.nan*np.ones(3)
        self.prev_s1d_2o = np.nan*np.ones(3)
        self.prev_s2d_2o = np.nan*np.ones(3)
        self.prev_s3d_2o = np.nan*np.ones(3)
        self.prev_torq = np.nan*np.ones(3)
        self.prev_wd = np.nan*np.ones(3)
        self.prev_wd_ECI = np.nan*np.ones(3)
        self.prev_we = np.nan*np.ones(3)
        self.prev_we_ECI = np.nan*np.ones(3)
        self.prev_quaterr = np.nan*np.ones(4)
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
        # self.MTQ_ctrl_matrix = np.zeros((3,sum([j.input_len for j in self.sat.actuators])))#np.stack([self.MTQ_matrix_inv[j,:] if isinstance(j,MTQ) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
        # self.MTQ_ctrl_matrix[:,self.mtq_ctrl_mask] = self.MTQ_matrix_inv
        # self.RW_matrix = np.stack([j.axis if isinstance(j,RW) else np.zeros((j.input_len,3)) for j in self.sat.actuators])
        # self.rw_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,RW) for j in self.sat.actuators]).astype(bool)
        #
        #
        # self.RWjs = self.diagflat(np.array([self.sat.actuators[j].J for j in self.sat.momentum_inds]))
        # self.RWaxes = np.stack([self.sat.actuators[j].axis for j in self.sat.momentum_inds])
        # # self.mtq_axes_mat = np.stack([self.sat.attitude_sensors[j].axis for j in range(len(self.sat.attitude_sensors)) if self.mtm_mask[j]])
        if self.MTQ_matrix_rank != 3:
            raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.


    def scalc(self,qerr,werr,state,save = False,do_print=False):
        s = werr@self.sat.J
        if do_print:
            print('s1',s@self.params.Lambda_All,np.dot(s@self.params.Lambda_All,s@self.params.Lambda_All))
            print('s1wb',state[0:3]@self.sat.J@self.params.Lambda_All,np.dot(state[0:3]@self.sat.J@self.params.Lambda_All,state[0:3]@self.sat.J@self.params.Lambda_All))
            print('s1wd',-(state[0:3]-werr)@self.sat.J@self.params.Lambda_All,np.dot(-(state[0:3]-werr)@self.sat.J@self.params.Lambda_All,-(state[0:3]-werr)@self.sat.J@self.params.Lambda_All))
        s += (qerr[0]**self.params.q0exp_1)*(norm(qerr[1:])**(self.params.qexp-1.0))*qerr[1:]@self.params.Lambda_q
        if do_print:
            tt = (qerr[0]**self.params.q0exp_1)*(norm(qerr[1:])**(self.params.qexp-1.0))*qerr[1:]@self.params.Lambda_q@self.params.Lambda_All
            print('s2',tt,np.dot(tt,tt))
            # print((qerr[0]**self.params.q0exp_1)*(norm(qerr)**(self.params.qexp-1.0))*qerr[1:],(norm(qerr)**(self.params.qexp-1.0))*qerr[1:],qerr[1:])
        s += (qerr[0]**self.params.q0exp_2)*qerr[1:]@self.params.Lambda_q2
        if do_print:
            tt = (qerr[0]**self.params.q0exp_2)*qerr[1:]@self.params.Lambda_q2@self.params.Lambda_All
            print('s3',tt,np.dot(tt,tt))
        s = s@self.params.Lambda_All
        if do_print:
            print(s)

        # s1 = state[0:3]@self.sat.J@self.params.Lambda_All
        # s2 = -(state[0:3]-werr)@self.sat.J@self.params.Lambda_All
        # s3 = qerr[1:]@self.params.Lambda_q@self.params.Lambda_All
        # if save:
        #     self.s = s
        #     self.s1 = s1
        #     self.s2 = s2
        #     self.s3 = s3
        #     self.s_ECI = s@rot_mat(state[3:7]).T
        #     self.s1_ECI = s1@rot_mat(state[3:7]).T
        #     self.s2_ECI = s2@rot_mat(state[3:7]).T
        #     self.s3_ECI = s3@rot_mat(state[3:7]).T
        return s


    def sdotcalc_noact(self,qerr,werr,state,vecs,save = False,do_print=False):

        sdot = 0.5*( \
            -self.params.q0exp_1*(norm(qerr[1:])**2.0)*np.dot(qerr[1:],werr)*qerr[1:] \
            + (qerr[0]**2)*(self.params.qexp-1)*np.dot(qerr[1:],werr)*qerr[1:] \
            + qerr[0]*(norm(qerr[1:])**2.0)*(qerr[0]*werr + np.cross(qerr[1:],werr)) \
            )@self.params.Lambda_q@self.params.Lambda_All*(qerr[0]**(self.params.q0exp_1-1.0))*(norm(qerr[1:])**(self.params.qexp-3.0))
        if do_print:
            tt=  (qerr[0]**self.params.q0exp_1)*(norm(qerr)**(self.params.qexp-1.0))*qerr[1:]@self.params.Lambda_q@self.params.Lambda_All
            ttd = 0.5*( \
                -self.params.q0exp_1*(norm(qerr[1:])**2.0)*np.dot(qerr[1:],werr)*qerr[1:] \
                + qerr[0]*qerr[0]*(self.params.qexp-1)*np.dot(qerr[1:],werr)*qerr[1:] \
                + qerr[0]*(norm(qerr[1:])**2.0)*(qerr[0]*werr + np.cross(qerr[1:],werr)) \
                )@self.params.Lambda_q@self.params.Lambda_All*(qerr[0]**(self.params.q0exp_1-1.0))*(norm(qerr[1:])**(self.params.qexp-3.0))
            print('sd2',ttd,2*np.dot(tt,ttd))
        sdot += 0.5*( \
                -self.params.q0exp_2*np.dot(qerr[1:],werr)*qerr[1:] \
                + qerr[0]*(qerr[0]*werr + np.cross(qerr[1:],werr)) \
                )@self.params.Lambda_q2@self.params.Lambda_All*(qerr[0]**(self.params.q0exp_2-1.0))
        if do_print:
            tt=  (qerr[0]**self.params.q0exp_2)*qerr[1:]@self.params.Lambda_q2@self.params.Lambda_All
            ttd = 0.5*( \
                -self.params.q0exp_2*np.dot(qerr[1:],werr)*qerr[1:] \
                + qerr[0]*(qerr[0]*werr + np.cross(qerr[1:],werr)) \
                )@self.params.Lambda_q2@self.params.Lambda_All*(qerr[0]**(self.params.q0exp_2-1.0))
            print('sd3',ttd,2*np.dot(tt,ttd))
        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
                # print(state[7:self.sat.state_len]@self.RWaxes)
            sdot -= (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],werr)@self.sat.J)@self.params.Lambda_All
        moddist = np.zeros(3)
        if self.include_disturbances:
            moddist = self.sat.dist_torque(state,vecs,details=True)
            # print('est dist list')
            # print(self.sat.last_dist_list)
            sdot += moddist@self.params.Lambda_All
            if save:
                self.saved_dist = moddist.copy()
        if do_print:
            tt= werr@self.sat.J@self.params.Lambda_All
            ttwb = state[0:3]@self.sat.J@self.params.Lambda_All
            ttwd = -(state[0:3]-werr)@self.sat.J@self.params.Lambda_All
            ttwbd =  -(np.cross(state[0:3],ang_mom))@self.params.Lambda_All + (self.sat.last_act_torq+moddist)@self.params.Lambda_All
            ttwdd =  -np.cross(state[0:3],werr)@self.sat.J@self.params.Lambda_All
            ttd = -(np.cross(state[0:3],ang_mom) + np.cross(state[0:3],werr)@self.sat.J)@self.params.Lambda_All + (self.sat.last_act_torq+moddist)@self.params.Lambda_All

            print('sd1',ttd,2*np.dot(tt,ttd))
            print('sdwb1',ttwbd,2*np.dot(ttwb,ttwbd))
            print('sdwd1',ttwdd,2*np.dot(ttwd,ttwdd))
        return sdot


    def baseline_actuation(self,state,qerr,werr,vecs,other_err=None,dt = 1,save = False):
        s = self.scalc(qerr,werr,state,save = save)
        base_torq = -1.0*self.sdotcalc_noact(qerr,werr,state,vecs,save)
        # base_torq -= s@self.params.Lambda_s
        # print('base torq eq ', base_torq,norm(base_torq))
        # print((norm(s)**self.params.sexp)*normalize(s)@self.params.Lambda_s,self.params.Lambda_s,norm(s))
        base_torq -= (norm(s)**self.params.sexp)*normalize(s)@self.params.Lambda_s#np.sign(s)*np.sqrt(np.absolute(s))@self.params.Lambda_s#@self.params.Lambda_All

        # print('base torq with s ', base_torq,norm(base_torq))
        base_torq += self.int_term
        # print('base torq with int ', base_torq,norm(base_torq))
        # base_torq -= np.sign(s)*np.sqrt(np.absolute(s))@self.params.Lambda_s
        # print(s,np.sign(s))
        # print(np.absolute(s),np.sqrt(np.absolute(s)))
        # print(np.sign(s)*np.sqrt(np.absolute(s)),np.sign(s)*np.sqrt(np.absolute(s))@self.params.Lambda_s)
        # udes = base_torq
        # udes = s*np.dot(s,base_torq@self.params.inv_Lambda_All)/(norm(s)**2.0)
        # print('base torq all ', base_torq,norm(base_torq))
        udes = s*np.dot(s,base_torq)/(norm(s)**2.0)

        # if self.include_disturbances:
        #     udes -= self.sat.dist_torque(state,vecs)
        # print('udes ', udes,norm(udes))
        mtq_cmd = np.cross(vecs["b"],udes)/(norm(vecs["b"])**2.0)
        u = self.mtq_command_maintain_RW(mtq_cmd,state,vecs)
        u_noB = self.mtq_command_maintain_RW(mtq_cmd,state,vecs,compensate_bias = False)

        ang_mom = state[0:3]@self.sat.J
        if self.sat.number_RW > 0:
            ang_mom += state[7:self.sat.state_len]@self.RWaxes
        return u

    def baseline_actuation_jac_over_qerr(self,state,qerr,werr,vecs,other_err=None):
        ang_mom = state[0:3]@self.sat.J
        if self.sat.number_RW > 0:
            ang_mom += state[7:self.sat.state_len]@self.RWaxes
        s = werr@self.sat.J + qerr[1:]@self.params.Lambda_q
        sjac = quat2vec_mat.T@self.params.Lambda_q
        base_torq = np.cross(state[0:3],ang_mom) + np.cross(state[0:3],werr)@self.sat.J - 0.5*((werr@Wmat(qerr).T)[1:])@self.params.Lambda_q#(q_err[0]*w_err-np.cross(w_err,q_err[1:]))
        base_torq_jac = np.zeros((4,3))
        base_torq_jac[0,:] = 0.5*w
        base_torq_jac[1:,:] = 0.5*skewsym(w)
        base_torq_jac = -0.5*((werr@Wmat(qerr).T)[1:])@self.params.Lambda_q
        if self.include_disturbances:
            base_torq -= self.sat.dist_torque(state,vecs)
        base_torq -= s@self.params.Lambda_s
        base_torq_jac -= sjac@self.params.Lambda_s
        jac = base_torq_jac@np.outer(s,s)/(norm(s)**2.0) + sjac@(np.eye(3)*np.dot(s,base_torq) + np.outer(base_torq,s) -  2*np.outer(s,s)*np.dot(s,base_torq)/(norm(s)))/(norm(s)**2.0)
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
        dt = (cent2sec*(osp1.J2000-os.J2000))
        if self.modename is GovernorMode.WISNIEWSKI_TWISTING:
            # if self.calc_av_from_quat:
            #    # wgoal = self.av_from_quat_goals(state,goal_state,next_goal,(cent2sec*(osp1.J2000-os.J2000)))
            err = self.state_err(state,goal_state,next_desired = next_goal,dt= dt,print_info=True)
            # print(goal_state.state[0:3])
            q_err = err[3:7]
            q_err *= np.sign(q_err[0])
            w_err = err[0:3]
            # print(q_err)
            # print('zen ',normalize(vecs["r"]))
            # print('ram ',normalize(vecs["v"]))
            # print('nrm ',normalize(np.cross(vecs["r"],vecs["v"])))
            # print('wd  ',state[0:3]-w_err)
            # print('wdn ',normalize(state[0:3]-w_err))
            # print('wd deg/orb ',norm(state[0:3]-w_err)*(180.0/math.pi)*90*60)
        else:
            if self.modename is GovernorMode.WISNIEWSKI_TWISTING_QUATSET_B:
                w_err,q_err = self.perpB_err(state,goal_state,next_goal,dt,vecs["b"])
            elif self.modename is GovernorMode.WISNIEWSKI_TWISTING_QUATSET_ANG:
                w_err,q_err = self.minang_err(state,goal_state,next_goal,dt)
            elif self.modename is GovernorMode.WISNIEWSKI_TWISTING_QUATSET_LYAP:
                w_err,q_err = self.minLyap_err(state,goal_state,vecs,next_goal,dt,qmult = np.mean(np.diag(self.params.Lambda_q@self.params.Lambda_s)),av_mat_mult = np.eye(3))
            elif self.modename is GovernorMode.WISNIEWSKI_TWISTING_QUATSET_LYAPR:
                w_err,q_err = self.minLyapR_err(state,goal_state,vecs,next_goal,dt,qmult = np.mean(np.diag(self.params.Lambda_q@self.params.Lambda_s)),av_mat_mult = np.eye(3))
            elif self.modename is GovernorMode.WISNIEWSKI_TWISTING_QUATSET_BS:
                w_err,q_err = self.minBS_err(state,goal_state,vecs,next_goal,(osp1.J2000-os.J2000)*cent2sec,qmult = np.mean(np.diag(self.params.Lambda_q@self.params.Lambda_s)),av_mat_mult = np.eye(3))
            # elif self.modename is GovernorMode.WISNIEWSKI_TWISTING_QUATSET_MINS:
            #     pass
            else:
                raise ValueError("Incorrect mode for Wisniewksi twisting")

        if not is_fake:
            self.int_term -= self.params.const_int*normalize(self.scalc(q_err,w_err,state,save=False))*dt
            # print('int term ', self.int_term)

        res =  self.baseline_actuation(state,q_err,w_err,vecs,save = False)

        # self.prev_wd = state[0:3]-w_err
        # self.prev_wd_ECI = (state[0:3]-w_err)@rot_mat(state[3:7]).T
        #
        # self.prev_we = w_err
        # self.prev_we_ECI = w_err@rot_mat(state[3:7]).T
        return res
        # ang_mom = state[0:3]@self.sat.J
        # if self.sat.number_RW > 0:
        #     ang_mom += state[7:self.sat.state_len]@self.RWaxes
        # # vecs = os_local_vecs(os,q)
        # print(q_err)
        # # print(wgoal)
        # # print(w_err)
        # # print(0.5*((w_err@Wmat(q_err).T)[0]))
        # s = w_err@self.sat.J + q_err[1:]@self.params.Lambda_q
        # # print(np.dot(w_err@self.sat.J,q_err[1:]@self.params.Lambda_q))
        # base_torq = np.cross(state[0:3],ang_mom) + np.cross(state[0:3],w_err)@self.sat.J - 0.5*((w_err@Wmat(q_err).T)[1:])@self.params.Lambda_q#(q_err[0]*w_err-np.cross(w_err,q_err[1:]))
        # print('base torq ',base_torq)#, np.cross(state[0:3],ang_mom),np.cross(state[0:3],goal_state.state[0:3])@self.sat.J, -0.5*((w_err@Wmat(q_err).T)[1:])@self.params.Lambda_q)
        # # print(np.dot((base_torq-s@self.params.Lambda_s)@self.sat.invJ,w_err))
        # # print(np.dot((base_torq-s@self.params.Lambda_s),q_err[1:]@self.params.Lambda_q))
        # # print(np.dot((base_torq-s@self.params.Lambda_s)@self.sat.invJ,q_err[1:]@self.params.Lambda_q))
        # if self.include_disturbances:
        #     base_torq -= self.sat.dist_torque(state,vecs)
        # # statedot = self.sat.dynamics_core(state[0:7],np.zeros(3),os,False,False,False,False)
        # # texp = statedot[0:3]@self.sat.J
        # # sdotexp = w_err@self.sat.J_noRW + 0.5*statedot[4:7]@self.params.Lambda_q
        # # print(base_torq,texp)
        # # print(statedot)
        # # breakpoint()
        #
        # # print(base_torq,-self.sat.dist_torque(state,vecs))
        # base_torq -= s@self.params.Lambda_s
        # # print(base_torq,-s@self.params.Lambda_s)
        # # breakpoint()
        # udes = s*np.dot(s,base_torq)/(norm(s)**2.0)
        # self.addl_info["desired_torq"] = udes
        # # print(udes)
        # mtq_cmd = np.cross(vecs["b"],base_torq)/(norm(vecs["b"])**2.0)
        # # print(np.dot((np.cross(mtq_cmd,vecs["b"]) + self.sat.dist_torque(state,vecs) -  np.cross(state[0:3],ang_mom) )@self.sat.invJ,w_err))
        # # print(np.dot((np.cross(mtq_cmd,vecs["b"]) + self.sat.dist_torque(state,vecs) -  np.cross(state[0:3],ang_mom)),q_err[1:]@self.params.Lambda_q))
        # # print(np.dot((np.cross(mtq_cmd,vecs["b"]) + self.sat.dist_torque(state,vecs) -  np.cross(state[0:3],ang_mom))@self.sat.invJ,q_err[1:]@self.params.Lambda_q))
        # # print(mtq_cmd)
        # # print(mtq_cmd-np.array([j.bias for j in self.sat.actuators]).flatten())
        # u = self.mtq_command_maintain_RW(mtq_cmd,state,vecs)
        # # print(u)
        #
        # #TODO deal with case where bdot is being run and mtqs are very close to boundary--basically add control mode that keeps wheels at some ang_mom when other torques are happening.
        # return u
