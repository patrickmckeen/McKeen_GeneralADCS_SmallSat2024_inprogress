from .control_mode import *
from scipy.optimize import minimize,Bounds,LinearConstraint

class Trajectory2MPC(ControlMode):
    def __init__(self,gain_info,sat,maintain_RW = True,include_disturbances=True,calc_av_from_quat = False,include_rotational_motion = False):
        ModeName = GovernorMode.PLAN_AND_TRACK_MPC
        # self.tracking_LQR_formulation = tracking_LQR_formulation
        if len(gain_info)==0:
            gain_info = [1]
        self.dt = gain_info[0]
        # self.gain = gain
        params = Params()
        super().__init__(ModeName,sat,params,False,include_disturbances,False,False)


    def scoring_func(self,u,rk4func,weight,next_plan,nextrk4func):
        xkp1 = rk4func(u)
        xkp2 = nextrk4func(x,u)
        xerr = self.reduced_state_err(xkp1,next_plan,quatvecmode=0)
        # breakpoint()
        return xerr@weight@xerr


    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        """
        This function finds the commanded control input.

        """
        q = state[3:7]
        vecs = os_local_vecs(os,q)
        plan_state = planner_params[0][-1]
        plan_control = planner_params[0][1]
        plan_gain = planner_params[0][2]
        planned_torq = planner_params[0][4]

        next_plan_state = planner_params[1][-1]
        next_plan_control = planner_params[1][1]
        next_plan_gain = planner_params[1][2]
        next_plan_ctg = planner_params[1][3]
        next_planned_torq = planner_params[1][4]

        err = self.reduced_state_err(state,plan_state,quatvecmode=0)

        u = plan_control - err@plan_gain.T

        # breakpoint()
        next_state = lambda u, self=self,os = os,osp1=osp1,: self.sat.rk4(state[:self.sat.state_len],u[:self.sat.conrol_len],self.dt,os,osp1,verbose=False,quat_as_vec = True,save_info = False)
        next_next_state = lambda x,u, self=self,os = osp1,osp1=osp1,: self.sat.rk4(x,u[self.sat.conrol_len:],self.dt,osp1,osp1,verbose=False,quat_as_vec = True,save_info = False)
        # func = lambda u, self=self,next_state=next_state,next_plan_gain = next_plan_gain, next_plan_state = next_plan_state: self.scoring_func(u,next_state,next_plan_gain.T@next_plan_gain,next_plan_state)
        func = lambda u, self=self,next_state=next_state,next_next_state = next_next_state,next_plan_gain = next_plan_gain, next_plan_state = next_plan_state: self.scoring_func(u,next_state,next_plan_ctg,next_plan_state,next_next_state)

        lb,ub = self.sat.control_bounds()
        if self.sat.number_RW>0:
            pass #add constraints on h
        bounds = Bounds(lb=np.concatenate([lb,lb]),ub=np.concatenate([ub,ub]))
        res = minimize(func,u,bounds=bounds)#,constraints=y)
        return res.x[:self.sat.control_len]
