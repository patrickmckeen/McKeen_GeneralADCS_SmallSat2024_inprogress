from .control_mode import *

class NoControl(ControlMode):
    def __init__(self,sat):
        ModeName = GovernorMode.NO_CONTROL
        pars = Params()
        super().__init__(ModeName,sat,pars,False,False,False,False)

    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        return np.zeros(self.sat.control_len)
