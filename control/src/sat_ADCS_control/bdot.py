from .control_mode import *

class Bdot(ControlMode):
    def __init__(self,gain,sat,maintain_RW = True):
        ModeName = GovernorMode.SIMPLE_BDOT
        # self.gain = gain
        params = Params()
        params.gain = gain
        super().__init__(ModeName,sat,params,maintain_RW,False,False,False)
        # self.maintain_RW = maintain_RW

        self.mtm_mask = np.array([isinstance(j,MTM) for j in self.sat.attitude_sensors]+[False for j in range(self.sat.number_RW)])
        self.mtm_reading_mask = np.concatenate([np.ones(j.output_length)*isinstance(j,MTM) for j in self.sat.attitude_sensors]+[[False for j in range(self.sat.number_RW)]]).astype(bool)
        if sum(self.mtm_mask) != 3:
            raise ValueError('This is currently only implemented for exactly 3 MTMs') #TO-DO: include with more, by averaging?
        if np.any(self.mtm_mask):
            self.mtm_axes_mat = np.stack([self.sat.attitude_sensors[j].axis/self.sat.attitude_sensors[j].scale for j in range(len(self.sat.attitude_sensors)) if self.mtm_mask[j]]).T
        else:
            self.mtm_axes_mat = np.nan
        if np.linalg.matrix_rank(self.mtm_axes_mat) != 3:
            raise ValueError('MTM axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.
        self.mtm_axes_mat_inv = np.linalg.inv(self.mtm_axes_mat)
        self.mtm_read_mat = np.zeros((sum([j.output_length for j in self.sat.attitude_sensors])+self.sat.number_RW,3))
        self.mtm_read_mat[self.mtm_reading_mask,:] = self.mtm_axes_mat

        # self.Bbody_mat = np.array([1/self.sat.sensors[j].scale for j in range(len(sens)) if self.mtm_reading_mask[j]])@self.mtm_axes_mat_inv

        # self.mtq_mask = np.array([isinstance(j,MTQ) for j in self.sat.actuators])
        # self.mtq_max = np.array([j.max for j in self.sat.actuators])
        #
        # self.mtq_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,MTQ) for j in self.sat.actuators]).astype(bool)
        if sum(self.mtq_mask) != 3:
            raise ValueError('This is currently only implemented for exactly 3 MTQs') #TO-DO: include with more, by averaging?
        # self.MTQ_matrix_inv = np.linalg.inv(np.stack([j.axis if isinstance(j,MTQ) for j in self.sat.actuators]))
        #
    # # self.mtq_axes_mat = np.stack([self.sat.attitude_sensors[j].axis for j in range(len(self.sat.attitude_sensors)) if self.mtm_mask[j]])
        if self.MTQ_matrix_rank != 3:
            raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.
        self.prev_B_body = np.nan*np.ones(3)
        self.prev_J2000 = np.nan

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
        #Get settings
        B_body = sens@self.mtm_read_mat
        if np.any(np.isnan(self.prev_B_body)):
            u = np.zeros(3)
        else:
            bdotbase = -self.params.gain*(B_body-self.prev_B_body)/((os.J2000-self.prev_J2000)*cent2sec)
            vecs = os_local_vecs(os,state[3:7])
            vecs["b"] = B_body
            u = self.mtq_command_maintain_RW(bdotbase,state,vecs)
        if not is_fake:
            self.prev_B_body = B_body
            self.prev_J2000 = os.J2000
        #TODO deal with case where bdot is being run and mtqs are very close to boundary--basically add control mode that keeps wheels at some ang_mom when other torques are happening.
        return u
