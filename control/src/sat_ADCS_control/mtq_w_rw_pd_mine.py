from .control_mode import *

class MTQ_W_RW_PD_MINE(ControlMode):
    def __init__(self,gain_info,sat,include_disturbances=True):
        ModeName = GovernorMode.MTQ_W_RW_PD_MINE

        params = Params()

        params.w_gain = gain_info[0]
        params.q_gain = gain_info[1]

        if len(gain_info) < 3:
            params.c_gain = 0.005
        else:
            params.c_gain = gain_info[2]

        if len(gain_info) < 4:
            params.RW_bias_h = np.array([0.05*j.max_h for j in sat.actuators if isinstance(j,RW)])
        else:
            params.RW_bias_h = gain_info[3] #TODO: make bias adaptive to move more quickly through stiction, etc.
        if len(gain_info) < 5:
            params.integral_gain = np.eye(3)*0 #TODO: not tested
        else:
            params.integral_gain = gain_info[4]
        # params.gain = gain
        super().__init__(ModeName,sat,params,False,include_disturbances,False,True)
        self.RW_J_matrix = np.diagflat(np.array([self.sat.actuators[j].J for j in self.sat.momentum_inds]))
        self.integral_term = np.zeros(3)





        if np.linalg.matrix_rank(self.RWaxes) != 3:
            raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.


    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        """
        """
        q = state[3:7]
        vecs = os_local_vecs(os,q)
        err = self.state_err(state,goal_state,next_desired = next_goal,dt= (cent2sec*(osp1.J2000-os.J2000)))
        q_err = err[3:7]
        q_err *= np.sign(q_err[0])
        w_err = err[0:3]



        base_torq = -1.0*(quat_to_vec3(q_err,0)@self.params.q_gain + (self.integral_term@self.params.integral_gain + w_err)@self.params.w_gain)

        return self.general_ctrl_from_torq(state,w_err,os,vecs,self.params.RW_bias_h,np.ones(self.rw_N)*0.005,base_torq,self.ones(self.n_MTQ)*1e3,self.ones(self.magic_N)*1e-8,self.ones(self.rw_N),wdesdot = np.zeros(3))

        print('base_torq0',base_torq)
        # print('exp wdot',base_torq@self.sat.invJ*180.0/math.pi,norm(base_torq@self.sat.invJ*180.0/math.pi))
        if self.include_rotational_motion:
            ang_mom = state[0:3]@self.sat.J
            if self.sat.number_RW > 0:
                ang_mom += state[7:self.sat.state_len]@self.RWaxes
            base_torq += (np.cross(state[0:3],ang_mom) + np.cross(state[0:3],w_err)@self.sat.J) #TODO: add case of goal AV changing over time--w_ref_dot or whatever

        moddist = self.sat.dist_torque(state,vecs)
        if self.include_disturbances:
            base_torq -= moddist
            self.saved_dist = moddist.copy()

        print('base_torq',base_torq)

        bias = np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])


        w_err_dot = (moddist + base_torq - np.cross(state[0:3],ang_mom) - np.cross(state[0:3],w_err)@self.sat.J)@self.sat.invJ

        self.integral_term += (cent2sec*(osp1.J2000-os.J2000))*(quat_to_vec3(q_err,0)@self.params.q_gain + w_err_dot@self.sat.J)


        u_mtq,u_rw = self.add_RW_mtq_desat(state,os,vecs,self.params.RW_bias_h,self.params.c_gain,base_torq)
        base_cmd = self.RW_ctrl_matrix.T@u_rw + self.MTQ_ctrl_matrix.T@u_mtq - bias
        return base_cmd

        # base_rw = np.linalg.pinv(self.RWaxes)@base_torq
        #
        # l0,u0 = self.sat.control_bounds()
        #
        # rw_despin = self.params.c_gain*(state[7:self.sat.state_len]-self.params.RW_bias_h)
        # u_mtq =  np.linalg.pinv(skewsym(vecs["b"])@self.MTQ_axes)@self.RWaxes@rw_despin
        # rw_mtq_correction =  np.linalg.pinv(self.RWaxes)@(skewsym(vecs["b"])@self.MTQ_axes@u_mtq  - self.RWaxes@rw_despin)
        #
        #
        # u_rw = base_rw + rw_despin + rw_mtq_correction
        # bias = np.concatenate([j.bias*j.has_bias for j in self.sat.actuators])
        # base_cmd = self.RW_ctrl_matrix.T@u_rw + self.MTQ_ctrl_matrix.T@u_mtq - bias
        # print(np.dot(rw_despin + rw_mtq_correction,(state[7:self.sat.state_len]-self.params.RW_bias_h)))
        #
        # print(l0)
        # print(u0)
        #
        # print('base_cmd',base_cmd)
        # print('base_cmd no bias',base_cmd + bias)
        # print(base_cmd>u0)
        # print(base_cmd<l0)
        # if not (np.any(base_cmd>u0) or np.any(base_cmd<l0)):
        #     return base_cmd
        #
        #
        #
        #
        # cmd0 = self.RW_ctrl_matrix.T@base_rw - bias
        #
        #
        # print('cmd0',cmd0)
        # print(cmd0>u0)
        # print(cmd0<l0)
        # if (np.any(cmd0>u0) or np.any(cmd0<l0)):
        #     print('doing backup 1')
        #     return np.clip(self.RW_ctrl_matrix.T@base_rw - bias,l0,u0)
        # else:
        #     print('doing backup 2')
        #     des = self.RW_ctrl_matrix.T@(rw_despin + rw_mtq_correction) + self.MTQ_ctrl_matrix.T@u_mtq
        #     return self.rankisN_ctrl_adjust(des,l0,u0,-self.RW_ctrl_matrix.T@base_rw + bias)


        #TODO: include RW saturation limits.
