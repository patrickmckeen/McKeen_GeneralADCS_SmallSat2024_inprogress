from .control_mode import *

class MagicBdot(ControlMode):
    def __init__(self,sat,gain=1):
        ModeName = GovernorMode.MAGIC_BDOT_WITH_EKF

        params = Params()
        params.gain = gain
        super().__init__(ModeName,sat,params,True,True,False,False)

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
        return self.control_from_wdotdes(state,os,-self.params.gain*state[0:3],is_fake)
        # return self.mtqrw_from_torqdes(self.bdot_gain*(-self.sat.J@w/self.update_period + cross(w,self.sat.J@w + RW_h)),RW_h,Bbody,is_fake)
