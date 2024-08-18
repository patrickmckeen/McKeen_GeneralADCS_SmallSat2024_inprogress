from .control_mode import *

class Magic_PD(ControlMode):
    def __init__(self,sat,gain):
        ModeName = GovernorMode.SIMPLE_MAG_PD
        self.gain = gain
        super().__init__(ModeName,sat)

    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):

        err = self.state_err(state,goal_state)
        q_err = err[3:7,:]
        w_err = err[0:3,:]
        kw = self.pd_w_gain
        ka = self.pd_q_gain
        nB2 = norm(os.B)
        Bbody = rot_mat(q).T@os.B
        u_mtq = -kw*cross(Bbody, self.sat.J@w_err)/nB2-ka*cross(Bbody,q_err[1:,:]*np.sign(np.ndarray.item(q_err[0,0])))/nB2

        udes = np.vstack([u_mtq,np.zeros((self.sat.number_RW+self.sat.number_magic,1))])

        u = self.control_from_des(state,os,udes,False)
        return u
