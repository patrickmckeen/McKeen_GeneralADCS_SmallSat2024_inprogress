from .control_mode import *

class WisniewskiTwisting2Magic(ControlMode):
    def __init__(self,gain_info,sat,maintain_RW = True,include_disturbances=False,quatset=False,quatset_type = "B",calc_av_from_quat = False,include_rotational_motion = True):
        if not quatset:
            ModeName = GovernorMode.WISNIEWSKI_TWISTING2_MAGIC
        else:
            # if quatset_type == "B":
            #     ModeName = GovernorMode.WISNIEWSKI_SLIDING_QUATSET_B
            # elif quatset_type == "Ang":
            #     ModeName = GovernorMode.WISNIEWSKI_SLIDING_QUATSET_ANG
            # elif quatset_type[:-1] == "Lyap":
            #     ModeName = GovernorMode.WISNIEWSKI_SLIDING_QUATSET_LYAP
            # elif quatset_type == "S":
            #     ModeName = GovernorMode.WISNIEWSKI_SLIDING_QUATSET_MINS
            # elif quatset_type[:-1] == "LyapRmat":
            #     ModeName = GovernorMode.WISNIEWSKI_SLIDING_QUATSET_LYAPR
            # else:
            raise ValueError("incorrect quatset class of WisniewskiTwisting2Magic")

        # self.gain = gain
        params = Params()
        params.Lambda_q = gain_info[0]
        params.Lambda_s = gain_info[1]
        if len(gain_info)>2:
            params.Lambda_All = gain_info[2]
        else:
            params.Lambda_All = np.eye(3)
        params.inv_Lambda_All = np.linalg.inv(params.Lambda_All)
        if quatset and quatset_type[:-1][:4] == "Lyap":
            params.lyap_type = int(quatset_type[-1]) #0,1,2
        super().__init__(ModeName,sat,params,maintain_RW,include_disturbances,calc_av_from_quat,include_rotational_motion)

        self.s = np.nan*np.ones(3)
        self.s1 = np.nan*np.ones(3)
        self.s2 = np.nan*np.ones(3)
        self.s3 = np.nan*np.ones(3)
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
        if sum(self.magic_mask) != 3:
            raise ValueError('This is currently only implemented for exactly 3 Magic') #TO-DO: include with more, by averaging?
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
        if np.linalg.matrix_rank(self.Magic_matrix) != 3:
            raise ValueError('Magic axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.


    def scalc(self,qerr,werr,state,save = False):
        s = (qerr[1:]@self.params.Lambda_q + werr@self.sat.J)@self.params.Lambda_All

        s1 = state[0:3]@self.sat.J@self.params.Lambda_All
        s2 = -(state[0:3]-werr)@self.sat.J@self.params.Lambda_All
        s3 = qerr[1:]@self.params.Lambda_q@self.params.Lambda_All
        if save:
            self.s = s
            self.s1 = s1
            self.s2 = s2
            self.s3 = s3
            self.s_ECI = s@rot_mat(state[3:7]).T
            self.s1_ECI = s1@rot_mat(state[3:7]).T
            self.s2_ECI = s2@rot_mat(state[3:7]).T
            self.s3_ECI = s3@rot_mat(state[3:7]).T
        return s

    def basetorq_calc(self,qerr,werr,state,vecs,other_err,save = False):
        base_torq = -0.5*(qerr[0]*werr + np.cross(qerr[1:],werr))@self.params.Lambda_q#@self.params.Lambda_All
        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
                # print(state[7:self.sat.state_len]@self.RWaxes)
            base_torq += (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],werr)@self.sat.J)#@self.params.Lambda_All
        moddist = np.zeros(3)
        if self.include_disturbances:
            moddist = self.sat.dist_torque(state,vecs)
            base_torq -= moddist
            self.saved_dist = moddist.copy()


        return base_torq

    def baseline_actuation(self,state,qerr,werr,vecs,other_err=None,dt = 1,save = False):
        # qerr *= -np.sign(np.dot(qerr[1:],werr@self.sat.J))
        if save:
            self.prev_s = self.s
            self.prev_s1 = self.s1
            self.prev_s2 = self.s2
            self.prev_s3 = self.s3
            self.prev_s_ECI = self.s_ECI
            self.prev_s1_ECI = self.s1_ECI
            self.prev_s2_ECI = self.s2_ECI
            self.prev_s3_ECI = self.s3_ECI
            self.prev_s1d = self.s1d
            self.prev_s2d = self.s2d
            self.prev_s3d = self.s3d
            self.prev_s1d_2o = self.s1d_2o
            self.prev_s2d_2o = self.s2d_2o
            self.prev_s3d_2o = self.s3d_2o
            self.prev_torq = self.torq
            self.prev_wd = self.wd
            self.prev_wd_ECI = self.wd_ECI
            self.prev_we = self.we
            self.prev_we_ECI = self.we_ECI

            self.prev_quaterr = self.quaterr

            self.we = werr
            self.quaterr = qerr.copy()
            self.wd = state[0:3]-werr
            self.we_ECI = werr@rot_mat(state[3:7]).T
            self.wd_ECI = (state[0:3]-werr)@rot_mat(state[3:7]).T
        s = self.scalc(qerr,werr,state,save = save)
        base_torq = self.basetorq_calc(qerr,werr,state,vecs,other_err,save = save)
        # print('sint',s)
        # print(qerr[1:]@self.params.Lambda_q)
        # print(werr@self.sat.J)
        # base_torq = -0.5*((werr@Wmat(qerr).T)[1:])@self.params.Lambda_q + np.cross(state[0:3],ang_mom) + np.cross(state[0:3],werr)@self.sat.J
        # print(self.include_rotational_motion)
        # print(qerr)

        # print(base_torq)
        # print(np.dot(s,-0.5*((werr@Wmat(qerr).T)[1:])@self.params.Lambda_q + 0.5*quat_mult(qerr,state[0:3])@quat2vec_mat.T@self.params.Lambda_q ))
        # print((base_torq - np.cross(state[0:3],ang_mom)) + 0.5*((werr@Wmat(qerr).T)[1:])@self.params.Lambda_q)

        base_torq -= np.sign(s)*np.sqrt(np.absolute(s))@self.params.Lambda_s#@self.params.Lambda_All
        # udes = base_torq
        # udes = s*np.dot(s,base_torq@self.params.inv_Lambda_All)/(norm(s)**2.0)
        udes = s*np.dot(s,base_torq)/(norm(s)**2.0)


        # print('   udes ',udes,norm(udes))
        # print('   udes0 ',base_torq,norm(base_torq))
        # print('   wddot approx ',((state[0:3]-werr)-self.prev_wd)/dt)
        # print('   wddot est ',(np.cross(state[0:3],werr)))
        # print('   wddotECI approx ',((state[0:3]-werr)-self.prev_wd_ECI@rot_mat(state[3:7]))/dt - np.cross(state[0:3],self.prev_wd_ECI@rot_mat(state[3:7])))
        # print('   wddotn approx ',normalize(((state[0:3]-werr)-self.prev_wd)/dt))
        # print('   wddotn est ',normalize(np.cross(state[0:3],werr)))
        # print('   wddotnECI approx ',normalize(((state[0:3]-werr)-self.prev_wd_ECI@rot_mat(state[3:7]))/dt - np.cross(state[0:3],self.prev_wd_ECI@rot_mat(state[3:7]))))



        # if self.include_disturbances:
        #     udes -= self.sat.dist_torque(state,vecs)
        # mtq_cmd = np.cross(vecs["b"],udes)/(norm(vecs["b"])**2.0)
        u = self.magic_command_maintain_RW(udes,state,vecs)
        u_noB = self.magic_command_maintain_RW(udes,state,vecs,compensate_bias = False)

        ang_mom = state[0:3]@self.sat.J
        if self.sat.number_RW > 0:
            ang_mom += state[7:self.sat.state_len]@self.RWaxes
        s1d = (-np.cross(state[0:3],ang_mom) + u_noB + self.saved_dist)@self.params.Lambda_All
        s2d = (-np.cross(state[0:3],werr)@self.sat.J)@self.params.Lambda_All
        s3d =  0.5*(qerr[0]*werr + np.cross(qerr[1:],werr))@self.params.Lambda_q@self.params.Lambda_All

        s1dd = (-np.cross(s1d@self.params.inv_Lambda_All@self.sat.invJ,ang_mom) - np.cross(state[0:3],s1d@self.params.inv_Lambda_All) - np.cross(state[0:3],self.saved_dist))@self.params.Lambda_All
        s2dd = ((-np.cross(s1d@self.params.inv_Lambda_All@self.sat.invJ,werr)-np.cross(state[0:3],(s1d+s2d)@self.params.inv_Lambda_All@self.sat.invJ))@self.sat.J)@self.params.Lambda_All
        s3dd =  0.5*(-0.5*np.dot(werr,werr)*qerr[1:] + qerr[0]*(s1d+s2d)@self.params.inv_Lambda_All@self.sat.invJ + np.cross(qerr[1:],(s1d+s2d)@self.params.inv_Lambda_All@self.sat.invJ))@self.params.Lambda_q@self.params.Lambda_All

        if save:
            self.s1d = s1d
            self.s2d = s2d
            self.s3d = s3d
            self.s1d_2o = s1d + 0.5*dt*s1dd
            self.s2d_2o = s2d + 0.5*dt*s2dd
            self.s3d_2o = s3d + 0.5*dt*s3dd
            self.torq = u_noB + self.saved_dist

        # print('   we approx ',    (werr-self.prev_we)/dt)
        # print('   we est ',       -np.cross(state[0:3],ang_mom)@self.sat.invJ - np.cross(state[0:3],werr) + (self.prev_torq)@self.sat.invJ)
        # print('   we est3 ',       -np.cross(state[0:3],ang_mom)@self.sat.invJ + np.cross(state[0:3],werr) + (self.prev_torq)@self.sat.invJ)
        # print('   we ECI approx ',    (werr-self.prev_we_ECI@rot_mat(state[3:7]))/dt + np.cross(state[0:3],self.prev_we_ECI@rot_mat(state[3:7])))
        # print(mtq_cmd)
        # print(u)
        # print(u_noB)
        # print(dt)
        print(dt)
        dels1 = (self.s1-self.prev_s1)/dt
        dels2 = (self.s2-self.prev_s2)/dt
        dels3 = (self.s3-self.prev_s3)/dt
        print(' s1d approx ',    dels1)
        print('  s1d est   ',       self.prev_s1d)
        # print('  s1d est   ',       self.prev_s1d_2o)
        # print('  s1d est   ',       self.s1d)
        # print(' s1d appECI ',    (self.s1-self.prev_s1_ECI@rot_mat(state[3:7]))/dt)
        err1 = dels1 - self.prev_s1d
        err2 = dels2 - self.prev_s2d
        err3 = dels3 - self.prev_s3d
        print(' s1d err    ',   err1, norm(err1)/norm(dels1),(180.0/np.pi)*np.arccos(np.dot(normalize(dels1),normalize(err1))))
        # print('   s1  ',       s1)
        # print('   s1 est from prev ',       self.prev_s1 + dt*self.prev_s1d)
        # print('   prev s1 ',       self.prev_s1)
        # print('   s1d ECI approx ',    (s1-self.prev_s1_ECI@rot_mat(state[3:7]))/dt + np.cross(state[0:3],self.prev_s1_ECI@rot_mat(state[3:7])))

        print('    s2d approx ',    dels2)
        print('       s2d est ',       self.prev_s2d)
        # print('       s2d est ',       self.prev_s2d_2o)
        # print('       s2d est ',       self.s2d)
        # print('    s2d appECI ',    (self.s2-self.prev_s2_ECI@rot_mat(state[3:7]))/dt + np.cross(state[0:3],self.prev_s2_ECI@rot_mat(state[3:7])))
        print('    s2d err    ',   err2, norm(err2)/norm(dels2),(180.0/np.pi)*np.arccos(np.dot(normalize(dels2),normalize(err2))))
        #
        print('  s3d approx ',   dels3)
        print('     s3d est ',      self.prev_s3d)
        # print('     s3d est ',      self.prev_s3d_2o)
        # print('     s3d est ',      self.s3d)
        # print('  s3d appECI ',    (self.s3-self.prev_s3_ECI@rot_mat(state[3:7]))/dt + np.cross(state[0:3],self.prev_s3_ECI@rot_mat(state[3:7])))
        # print(' s3d appECI2 ',    (self.s3-self.prev_s3_ECI@rot_mat(state[3:7]))/dt)
        print('  s3d err    ',   err3, norm(err3)/norm(dels3),(180.0/np.pi)*np.arccos(np.dot(normalize(dels3),normalize(err3))))

        #
        print('     sd approx ',(self.s1+self.s2+self.s3-self.prev_s1-self.prev_s2-self.prev_s3)/dt)
        print('        sd est ',self.prev_s1d+self.prev_s2d+self.prev_s3d)
        # print('        sd est ',self.prev_s1d_2o+self.prev_s2d_2o+self.prev_s3d_2o)
        # print('        sd est ',self.s1d+self.s2d+self.s3d)
        print('        sd err ',(self.s1+self.s2+self.s3-self.prev_s1-self.prev_s2-self.prev_s3)/dt - (self.prev_s1d+self.prev_s2d+self.prev_s3d))
        # print('        s      ',self.s1+self.s2+self.s3)
        print('        s      ',self.s)
        print('   pred s      ',self.prev_s1+self.prev_s2+self.prev_s3 + dt*(self.prev_s1d+self.prev_s2d+self.prev_s3d))
        # print('   prev s      ',self.prev_s1+self.prev_s2+self.prev_s3)
        print('   prev s      ',self.prev_s)
        print(rot_exp(self.prev_we))
        print(rot_exp(0.5*(self.prev_we+self.we)*dt)+(dt*dt/24)*np.concatenate([[0],np.cross(self.we,self.prev_we)]))
        print(quat_mult(quat_inv(self.prev_quaterr),self.quaterr))
        # print(' pred torq ',-np.cross(vecs["b"],u_noB))
        # print(self.prev_s-self.prev_s1-self.prev_s2-self.prev_s3)

        # print('   sdot approx ',    (s-self.prev_s)/dt)
        # print('   sdot est ',       -base_torq + (-np.cross(vecs["b"],u)+moddist)@self.params.Lambda_All)
        # print('   sdot2 est ',      0.5*(qerr[0]*werr + np.cross(qerr[1:],werr))@self.params.Lambda_q - np.cross(state[0:3],ang_mom)@self.params.Lambda_All - np.cross(state[0:3],(werr-state[0:3])@self.sat.J@self.params.Lambda_All) + (-np.cross(vecs["b"],u)+moddist)@self.params.Lambda_All)
        # print('   sdot3 est ',      0.5*(qerr[0]*werr + np.cross(qerr[1:],werr))@self.params.Lambda_q - np.cross(state[0:3],ang_mom)@self.params.Lambda_All + np.cross(state[0:3],werr)@self.sat.J@self.params.Lambda_All + (-np.cross(vecs["b"],u)+moddist)@self.params.Lambda_All)
        # print('   sdotECI approx ', (s-self.prev_s_ECI@rot_mat(state[3:7]) - np.cross(state[0:3],self.prev_s_ECI@rot_mat(state[3:7])))/dt)
        # print('   sdotn approx ',   normalize((s-self.prev_s)/dt))
        # print('   sdotn est ',      normalize(-base_torq + (-np.cross(vecs["b"],u)+moddist)@self.params.Lambda_All))
        # print('   sdotn2 est ',     normalize(0.5*(qerr[0]*werr + np.cross(qerr[1:],werr))@self.params.Lambda_q - np.cross(state[0:3],ang_mom)@self.params.Lambda_All - np.cross(state[0:3],(werr-state[0:3])@self.sat.J@self.params.Lambda_All) + (-np.cross(vecs["b"],u)+moddist)@self.params.Lambda_All))
        # print('   sdotnECI approx ',normalize((s-self.prev_s_ECI@rot_mat(state[3:7]))/dt - np.cross(state[0:3],self.prev_s_ECI@rot_mat(state[3:7]))))

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
        # u =self.mtq_command_maintain_RW(mtq_cmd,state,vecs)#TODO: add this.
        return jac



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
        if self.modename is GovernorMode.WISNIEWSKI_TWISTING2_MAGIC:
            # if self.calc_av_from_quat:
            #    # wgoal = self.av_from_quat_goals(state,goal_state,next_goal,(cent2sec*(osp1.J2000-os.J2000)))
            err = self.state_err(state,goal_state,next_desired = next_goal,dt= (cent2sec*(osp1.J2000-os.J2000)),print_info=True)
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
            # if self.modename is GovernorMode.WISNIEWSKI_SLIDING_QUATSET_B:
            #     w_err,q_err = self.perpB_err(state,goal_state,next_goal,(osp1.J2000-os.J2000)*cent2sec,vecs["b"])
            # elif self.modename is GovernorMode.WISNIEWSKI_SLIDING_QUATSET_ANG:
            #     w_err,q_err = self.minang_err(state,goal_state,next_goal,(osp1.J2000-os.J2000)*cent2sec)
            # elif self.modename is GovernorMode.WISNIEWSKI_SLIDING_QUATSET_LYAP:
            #     w_err,q_err = self.minLyap_err(state,goal_state,vecs,next_goal,(osp1.J2000-os.J2000)*cent2sec,qmult = np.mean(np.diag(self.params.Lambda_q@self.params.Lambda_s)),av_mat_mult = np.eye(3))
            # elif self.modename is GovernorMode.WISNIEWSKI_SLIDING_QUATSET_LYAPR:
            #     w_err,q_err = self.minLyapR_err(state,goal_state,vecs,next_goal,(osp1.J2000-os.J2000)*cent2sec,qmult = np.mean(np.diag(self.params.Lambda_q@self.params.Lambda_s)),av_mat_mult = np.eye(3))
            # # elif self.modename is GovernorMode.WISNIEWSKI_SLIDING_QUATSET_MINS:
            # #     pass
            # else:
            raise ValueError("Incorrect mode for WisniewksiTwisting2Magic")

        res =  self.baseline_actuation(state,q_err,w_err,vecs,save = not is_fake)

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
