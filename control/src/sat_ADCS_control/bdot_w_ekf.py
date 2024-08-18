from .control_mode import *

class BdotEKF(ControlMode):
    def __init__(self,gain,sat,maintain_RW = True,include_disturbances=False):
        ModeName = GovernorMode.BDOT_WITH_EKF
        # self.gain = gain
        params = Params()
        params.gain = gain
        super().__init__(ModeName,sat,params,maintain_RW,include_disturbances,False,False)
        # self.maintain_RW = maintain_RW
        # self.include_disturbances = include_disturbances
        #
        # # self.Bbody_mat = np.array([1/self.sat.sensors[j].scale for j in range(len(sens)) if self.mtm_reading_mask[j]])@self.mtm_axes_mat_inv
        #
        # self.mtq_mask = np.array([isinstance(j,MTQ) for j in self.sat.actuators])
        # self.mtq_max = np.array([j.max for j in self.sat.actuators])

        # self.mtq_ctrl_mask = np.concatenate([np.ones(j.input_len)*isinstance(j,MTQ) for j in self.sat.actuators]).astype(bool)
        if sum(self.mtq_mask) != 3:
            raise ValueError('This is currently only implemented for exactly 3 MTQs') #TO-DO: include with more, by averaging?
        # self.MTQ_matrix_inv = np.linalg.inv(np.stack([j.axis if isinstance(j,MTQ) for j in self.sat.actuators]))

        #
        # self.RWjs = self.diagflat(np.array([self.sat.actuators[j].J for j in self.sat.momentum_inds]))
        # self.RWaxes = np.stack([self.sat.actuators[j].axis for j in self.sat.momentum_inds])
        # # self.mtq_axes_mat = np.stack([self.sat.attitude_sensors[j].axis for j in range(len(self.sat.attitude_sensors)) if self.mtm_mask[j]])
        if self.MTQ_matrix_rank != 3:
            raise ValueError('MTQ axes need full rank') #TO-DO: solution for less. Should still be able to do SOMETHING.

    def find_actuation(self, state, os, osp1,  goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
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
        w = state[0:3]
        q = state[3:7]
        vecs = os_local_vecs(os,q)
        base_torq = -self.params.gain*w/norm(vecs["b"])**2.0
        print('base_torq',base_torq)
        if self.include_disturbances:
            moddist = self.sat.dist_torque(state,vecs)/norm(vecs["b"])**2.0
            base_torq -= moddist
            self.saved_dist = moddist.copy()
        bdotbase = np.cross(vecs["b"], base_torq)
        print('dist',self.saved_dist)
        print('bdotbase',bdotbase)
        u = self.mtq_command_maintain_RW(bdotbase,state,vecs)
        print('exp cmd torq',-np.cross(vecs["b"], bdotbase))
        print('exp net torq',self.saved_dist-np.cross(vecs["b"], bdotbase))
        # print('exp al',-np.cross(vecs["b"], bdotbase)@self.sat.invJ)
        print('pred wn',((self.saved_dist-np.cross(vecs["b"], bdotbase))@self.sat.invJ + state[0:3])*180.0/math.pi,norm(((self.saved_dist-np.cross(vecs["b"], bdotbase))@self.sat.invJ + state[0:3])*180.0/math.pi))
        # print('u',u)

        #TODO deal with case where bdot is being run and mtqs are very close to boundary--basically add control mode that keeps wheels at some ang_mom when other torques are happening.
        return u
