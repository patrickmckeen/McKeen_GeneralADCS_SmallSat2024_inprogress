from .attitude_estimator import *
import copy


class Crassidis_UKF(Estimator):
    def __init__(self, sat,estimate,cov_estimate,integration_cov,sample_time = 1,use_cross_term = False):
        super().__init__(sat,estimate,cov_estimate,integration_cov,sample_time,use_cross_term,False)
        # self.al = 1e-3
        # self.kap = 0
        # self.bet = 2.0#-1.0#2.0
        # self.include_int_noise_separately = True
        # self.neglect_low_sun = False
        self.vec_mode = 6
        self.lam = 1
        self.gyro_mask = np.array([isinstance(j,Gyro) for j in self.sat.attitude_sensors]+[False for j in range(self.sat.number_RW)])
        self.gyro_reading_mask = np.concatenate([np.ones(j.output_length)*isinstance(j,Gyro) for j in self.sat.attitude_sensors]+[[False for j in range(self.sat.number_RW)]]).astype(bool)
        if sum(self.gyro_mask) != 3:
            raise ValueError('This is currently only implemented for exactly 3 gyroscopes')
        self.gyro_axes_mat_inv = np.stack([self.sat.attitude_sensors[j].axis for j in range(len(self.sat.attitude_sensors)) if self.gyro_mask[j]])
        if np.linalg.matrix_rank(self.gyro_axes_mat_inv) != 3:
            raise ValueError('This is currently only implemented for satellites with gyroscopes that can measure all 3 axes')

    def reunite_states(self,dynstate,rest_state,quatref):
        quatdiff = quat_mult(quat_inv(quatref),dynstate[3:7])
        v3diff = quat_to_vec3(quatdiff,self.vec_mode)
        return np.concatenate([dynstate[0:3],v3diff,rest_state])

    def new_post_state(self,pre_rest_state,post_dynstate,quatref):
        post_state = self.reunite_states(post_dynstate,pre_rest_state,quatref)
        s0len = np.zeros(np.size(post_state) + 1)
        s0len[3:7] = quatref
        full_state = self.add_to_state(s0len,post_state)#these are backwards on purpose
        return post_state,full_state

    def update_core(self, control_vec, sensors_in, os,which_sensors):
        control_vec = np.copy(control_vec)
        os = os.copy()
        which_sensors = [which_sensors[j] and not self.gyro_mask[j] for j in range(len(which_sensors))]
        #take out values
        state0 = self.use_state.val.copy()
        quat0 = state0[3:7].copy()
        #find repeated orbital state once
        mid_os = self.prev_os.average(os)
        sens_vec_len = sum([self.sat.attitude_sensors[j].output_length for j in range(len(self.sat.attitude_sensors)) if which_sensors[j]])
        #generate sigma points of augmemted state--state itself, including actuator bias values, disturbance values, sensor bias values; sensor noise , control noise to use, snesor noise to use, possibly integration noise to use

        state_cov = self.use_state.cov.copy()[3:,3:] + self.use_state.int_cov.copy()[3:,3:]
        L = np.size(state_cov,0)
        lam = self.lam#1#3#3-L#self.al**2.0*(k+L)-L
        mat = np.linalg.cholesky(state_cov)
        # w,v = np.linalg.eig(state_cov)
        # srw = [math.sqrt(j) for j in np.abs(np.real(w))]
        # mat = np.real(v)@np.diagflat(srw)
        offsets = (L+lam)**0.5*np.hstack([mat,-mat]).T

        states = np.vstack([state0,self.add_to_state(state0,np.hstack([np.zeros((np.size(offsets,0),3)),offsets]))])

        wts_m = np.array([lam/(L+lam)]+[0.5/(L+lam) for j in range(2*L)])
        wts_c = np.array([lam/(L+lam) ]+[0.5/(L+lam) for j in range(2*L)])

        sigma_state_len = len(state0) - 4
        post_pts = np.nan*np.ones((2*L+1,sigma_state_len+3))
        post_sens = np.nan*np.ones((2*L+1,sens_vec_len))
        satj = copy.deepcopy(self.sat) #TODO: this needs to be deepcopy, right?
        whichj = which_sensors.copy()
        gyros_in = np.array([sensors_in[j] for j in range(len(self.gyro_reading_mask)) if self.gyro_reading_mask[j]])
        # breakpoint()
        for j in range(2*L+1): #TODO vectorize
            statej = states[j]
            self.sat_match(satj,statej)
            wj = (gyros_in-np.concatenate([satj.attitude_sensors[j].bias for j in range(len(self.sat.attitude_sensors)) if self.gyro_mask[j]]))@self.gyro_axes_mat_inv
            qj = quat_mult(statej[3:7],rot_exp(self.update_period*wj)) #TODO, check this.
            # breakpoint()
            post_dyn_state_j = np.concatenate([wj,qj,sensors_in[-self.sat.number_RW:]])

            if j == 0:
                post_quat = post_dyn_state_j[3:7]#can happen before integration noise is added because j=0 has 0 integration noise
            statej,full_statej = self.new_post_state(statej[self.sat.state_len:],post_dyn_state_j,post_quat)
            post_pts[j,:] = statej.copy()

            self.sat_match(satj,full_statej)
            vecsj = os_local_vecs(os,full_statej[3:7])
            sensj = satj.sensor_values(full_statej[0:self.sat.state_len],vecsj,which_sensors)

            post_sens[j,:] = sensj

        state1 = np.dot(wts_m,post_pts)
        sens1 = np.dot(wts_m,post_sens)
        pts_diff = post_pts - state1
        sens_diff = post_sens - sens1
        cov1 = sum([wts_c[j]*np.outer(pts_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)]) + self.use_state.int_cov
        # breakpoint()
        covyy = sum([wts_c[j]*np.outer(sens_diff[j,:],sens_diff[j,:]) for j in range(2*L+1)],0*np.eye(sens_vec_len)) + self.sat.sensor_cov(which_sensors)#sum([wts_c[i]*(post_sens[j,:]-sens1)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        covyx = sum([wts_c[j]*np.outer(sens_diff[j,:],pts_diff[j,3:]) for j in range(2*L+1)],np.zeros((sens_vec_len,sigma_state_len)))#sum([wts_c[i]*(state_pts_err[i]-x1_cut_red)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        try:
            # Kk = np.linalg.inv(covyy)@covyx
            Kk = scipy.linalg.solve(covyy,covyx,assume_a='pos')
        except:
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')


        state2 = state1 + np.concatenate([np.zeros(3),(sensors_in[which_sensors]-sens1)@Kk])
        cov2 = cov1 - block_diag(np.zeros((3,3)),Kk.T@covyy@Kk)
        cov2 = 0.5*(cov2 + cov2.T)
        # breakpoint()
        dvec3 = state2[3:6]
        dquat = vec3_to_quat(dvec3,self.vec_mode)
        quat = quat_mult(post_quat,dquat)
        state2 = np.concatenate([state2[0:3],quat,state2[6:self.sat.state_len-1],state2[self.sat.state_len-1:]])
        self.os_vecs = os_local_vecs(os,state2[3:7])

        return estimated_nparray(state2,cov2),[]
