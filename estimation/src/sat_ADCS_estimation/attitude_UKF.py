from .attitude_estimator import *
import copy
import scipy



class UKF(Estimator):
    def __init__(self, sat,estimate,cov_estimate,integration_cov,sample_time = 1,use_cross_term = False,quat_as_vec = False):
        super().__init__(sat,estimate,cov_estimate,integration_cov,sample_time,use_cross_term,quat_as_vec)
        self.al = 1.0#1e-3
        self.kap = 1.0
        self.bet = 2.0#-1.0#2.0
        self.include_int_noise_separately = False
        self.include_sens_noise_separately = False
        self.included_int_noise_where = 2
        self.neglect_low_sun = False
        self.scale_nonseparate_adds = False
        self.scale = 1
        self.sepscale = 1
        self.eigs = False
        self.scaled_UT=False
        self.UT_scale = 0.1
        # self.vec_mode = 6


    def make_pts_and_wts(self,pt0,which_sensors):

        state_cov = self.use_state.cov.copy() + self.use_state.int_cov.copy()*(not self.include_int_noise_separately)*(self.included_int_noise_where==0)
        int_cov = self.use_state.int_cov.copy()*self.include_int_noise_separately
        control_cov = self.sat.control_cov()
        use_control = np.size(control_cov)>0 and not np.all(control_cov==0)
        control_cov *= use_control
        sens_cov = self.sat.sensor_cov(which_sensors)*self.include_sens_noise_separately

        include_cov = [True,self.include_sens_noise_separately,use_control,self.include_int_noise_separately]
        covs = [state_cov,sens_cov,control_cov,int_cov]
        zeros = [pt0,sens_cov[0,:]*0,control_cov[0,:]*0,int_cov[0,:]*0]

        L = np.sum([include_cov[j]*np.size(covs[j],0) for j in range(4)])

        lam = self.al**2.0*(self.kap+L)-L#3#3-L#self.al**2.0*(k+L)-L

        if self.scaled_UT:
            lam = self.kap

        # lam = self.al**2.0*(self.kap)-L#3#3-L#self.al**2.0*(k+L)-L
        offsets = [0,0,0,0]
        pts = [zeros]
        self.scale = L+lam
        if self.scale_nonseparate_adds:
            self.sepscale = self.scale
        else:
            if (not self.include_int_noise_separately)*(self.included_int_noise_where==0):
                state_cov = self.use_state.cov.copy() + self.use_state.int_cov.copy()/self.scale


        for j in range(4):
            if include_cov[j]:
                if self.eigs:
                    w,v = np.linalg.eig(covs[j]*self.scale)
                    srw = np.diagflat(np.sqrt(np.abs(np.real(w))))#[math.sqrt(j) for j in np.abs(np.real(w))]
                    v = np.real(v)
                    mat = v@srw@np.linalg.inv(v)
                    # (U,D,perm) = scipy.linalg.ldl(covs[j]*self.scale,lower=False)
                    # D = np.diag(np.sqrt(np.diag(D[perm])))
                    # U = U[perm,:]
                    # # U /= np.diag(U)[:,None]
                    # mat = U@D
                    # mat = np.sqrt
                    # # mat = np.real(v@srw@v.T)
                else:
                    # (Lmat,D,perm) = scipy.linalg.ldl(covs[j]*self.scale)
                    # D = np.diag(np.sqrt(np.diag(D)))
                    # Lmat = Lmat#[perm,:]
                    # mat = (Lmat@D)
                    # breakpoint()
                    # U /= np.diag(U)[:,None]
                    try:
                        mat = np.linalg.cholesky(self.scale*covs[j])
                    except:
                        breakpoint()
                    # mat = scipy.linalg.cholesky(covs[j])
                # offsets[j] = (L+lam)**0.5*np.hstack([mat,-mat]).T
                offsets[j] = np.hstack([mat,-mat]).T
                if self.scaled_UT:
                    offsets[j] = self.al*offsets[j]
                if j == 0:
                    states = self.add_to_state(pt0,offsets[0])
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in states]
                else:
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in offsets[j]]

        if (1==self.included_int_noise_where) and not self.include_int_noise_separately:
            if self.eigs:
                w,v = np.linalg.eig(self.use_state.int_cov.copy()*self.sepscale)
                srw = np.diagflat(np.sqrt(np.abs(np.real(w))))#[math.sqrt(j) for j in np.abs(np.real(w))]
                v = np.real(v)
                mat = v@srw
            else:
                mat = np.linalg.cholesky(self.sepscale*self.use_state.int_cov.copy())
            extra_cov = np.hstack([mat,-mat]).T
            for j in range(len(pts[1:1+2*state_cov.shape[0]])):
                pts[j][3] = extra_cov[j,:]

        wts_m = np.array([lam/(L+lam)]+[0.5/(L+lam) for j in range(2*L)])
        wts_c = np.array([lam/(L+lam) + (1.0-self.al**2.0 + self.bet)]+[0.5/(L+lam) for j in range(2*L)])
        self.wts_m = wts_m
        self.wts_c = wts_c

        # breakpoint()
        if self.scaled_UT:

            wts_m = np.array([(lam/(L+lam))/self.al**2.0 + 1/self.al**2.0 - 1]+[(0.5/(L+lam))/self.al**2.0 for j in range(2*L)])
            wts_c = np.array([(lam/(L+lam))/self.al**2.0 + 1/self.al**2.0 - 1 + (1.0-self.al**2.0 + self.bet)]+[(0.5/(L+lam))/self.al**2.0 for j in range(2*L)])
            # wts_m = wts_m/self.UT_scale**2.0
            # wts_c = wts_c/self.UT_scale**2.0
            # wts_m[0] += (1-1/self.UT_scale**2.0)
            # wts_c[0] +=  (1-1/self.UT_scale**2.0) + (1-self.UT_scale**2.0)
        return L,pts,wts_m,wts_c,np.vstack([pt0,states]+[pt0]*(2*L-states.shape[0]))

    def reunite_states(self,dynstate,rest_state,quatref):
        if self.quat_as_vec:
            return np.concatenate([dynstate,rest_state])
        else:
            quatdiff = quat_mult(quat_inv(quatref),dynstate[3:7])
            v3diff = quat_to_vec3(quatdiff,self.vec_mode)
            return np.concatenate([dynstate[0:3],v3diff,rest_state])

    def new_post_state(self,pre_rest_state,post_dynstate,int_err,quatref):
        post_dyn_state_w_int_err = self.add_to_state(post_dynstate,int_err[0:self.sat.state_len - 1 + self.quat_as_vec])
        post_state = self.reunite_states(post_dyn_state_w_int_err,pre_rest_state+int_err[self.sat.state_len - 1 + self.quat_as_vec:],quatref)
        s0len = np.zeros(np.size(post_state) + 1 - self.quat_as_vec)
        s0len[3:7] = quatref
        full_state = self.add_to_state(s0len,post_state)#these are backwards on purpose
        return post_state,full_state

    def update_core(self, control_vec, sensors_in, os,which_sensors):
        control_vec = np.copy(control_vec)
        os = os.copy()

        #take out values
        state0 = self.use_state.val.copy()
        quat0 = state0[3:7].copy()
        #find repeated orbital state once
        mid_os = self.prev_os.average(os)
        mid_os = [self.prev_os.average(os,CG5_c[j]) for j in range(5)]
        dyn_state0 = self.sat.rk4(state0[0:self.sat.state_len],control_vec,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = False)
        vecs0 = os_local_vecs(os,dyn_state0[3:7])
        for j in range(len(self.sat.sensors)): #ignore sun sensors that should be in shadow
            which_sensors[j] &= not (isinstance(self.sat.sensors[j],SunSensor) and self.sat.sensors[j].clean_reading(dyn_state0,vecs0)<1e-10)


        sens_vec_len = sum([self.sat.sensors[j].output_length for j in range(len(self.sat.sensors)) if which_sensors[j]])
        #generate sigma points of augmemted state--state itself, including actuator bias values, disturbance values, sensor bias values; sensor noise , control noise to use, snesor noise to use, possibly integration noise to use.
        L,pts,wts_m,wts_c,sig0 = self.make_pts_and_wts(state0,which_sensors)
        sigma_state_len = len(state0) - 1 + self.quat_as_vec
        post_pts = np.nan*np.ones((2*L+1,sigma_state_len))
        post_sens = np.nan*np.ones((2*L+1,sens_vec_len))
        satj = copy.deepcopy(self.sat) #TODO: this needs to be deepcopy, right?
        whichj = which_sensors.copy()

        extra_obj = extra()
        extra_obj.cov0 = self.use_state.cov.copy()
        extra_obj.sig0 = sig0
        extra_obj.mean0 = sig0[0,:].copy()
        for j in range(2*L+1): #TODO vectorize
            [full_pre_statej,sens_noise_j,control_noise_j,int_noise_extra_j] = pts[j]

            self.sat_match(satj,full_pre_statej)
            post_dyn_state_j = satj.rk4(full_pre_statej[0:self.sat.state_len],control_vec + control_noise_j,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = False)

            if j == 0:
                post_quat = post_dyn_state_j[3:7]#can happen before integration noise is added because j=0 has 0 integration noise
            post_statej,post_full_statej = self.new_post_state(full_pre_statej[self.sat.state_len:],post_dyn_state_j,int_noise_extra_j,post_quat)
            post_pts[j,:] = post_statej.copy()

            self.sat_match(satj,post_full_statej)
            vecsj = os_local_vecs(os,post_full_statej[3:7])
            sensj = satj.sensor_values(post_full_statej[0:self.sat.state_len],vecsj,which_sensors) + sens_noise_j

            if self.neglect_low_sun:
                bad_sunj = [isinstance(self.sat.attitude_sensors[j],SunSensor) and self.sat.attitude_sensors[j].clean_reading(full_statej[0:self.sat.state_len],vecsj)<1e-10 for j in range(len(self.sat.attitude_sensors))]
                sensj[bad_sunj] = np.nan
            post_sens[j,:] = sensj.copy()

        # breakpoint()
        state1 = np.dot(wts_m,post_pts)
        dquat1 = vec3_to_quat(state1[3:6],self.vec_mode)
        quat1 = quat_mult(post_quat,dquat1)
        pred_dyn_state = np.concatenate([state1[0:3],quat1,state1[6:self.sat.state_len-1],state1[self.sat.state_len-1:]])
        psens = self.sat.sensor_values(pred_dyn_state,os_local_vecs(os,pred_dyn_state[3:7]))
        sens1 = np.dot(wts_m,post_sens)
        extra_obj.mean1 = pred_dyn_state.copy()
        # breakpoint()
        pts_diff = post_pts - state1
        sens_diff = post_sens - sens1
        cov1 = sum([wts_c[j]*np.outer(pts_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)])
        if not self.include_int_noise_separately and self.included_int_noise_where == 2:
            cov1 += self.use_state.int_cov*self.sepscale

        covyy = sum([wts_c[j]*np.outer(sens_diff[j,:],sens_diff[j,:]) for j in range(2*L+1)],0*np.eye(sens_vec_len))#sum([wts_c[i]*(post_sens[j,:]-sens1)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        covyx = sum([wts_c[j]*np.outer(sens_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)],np.zeros((sens_vec_len,sigma_state_len)))#sum([wts_c[i]*(state_pts_err[i]-x1_cut_red)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        if not self.include_sens_noise_separately:
            covyy += self.sat.sensor_cov(which_sensors)*self.sepscale
        # breakpoint()

        try:
            # Kk = np.linalg.inv(covyy)@covyx
            Kk = scipy.linalg.solve(covyy,covyx)
        except:
            # breakpoint()
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')


        extra_obj.senscov = covyy.copy()
        # extra_obj.sens_state = state1 +  (sensors_in[which_sensors]-sens1)@covyx
        extra_obj.sens1 = sens1.copy()
        self.sat_match(satj,pred_dyn_state)
        vecsj = os_local_vecs(os,pred_dyn_state[3:7])
        extra_obj.sens_of_state1 = satj.sensor_values(pred_dyn_state[0:self.sat.state_len],vecsj,which_sensors).copy()

        extra_obj.sens_sig = post_sens.copy()


        state2 = state1 + (sensors_in[which_sensors]-sens1)@Kk
        cov2 = cov1 - Kk.T@covyy@Kk
        cov2 = 0.5*(cov2 + cov2.T)

        if not self.quat_as_vec:
            dvec3 = state2[3:6]
            dquat = vec3_to_quat(dvec3,self.vec_mode)
            quat = quat_mult(post_quat,dquat)
            state2 = np.concatenate([state2[0:3],quat,state2[6:self.sat.state_len-1],state2[self.sat.state_len-1:]])
        else:
            state20 = np.copy(state2)
            state2[3:7] = normalize(state2[3:7])
            norm_jac = state_norm_jac(state20)
            cov2 = norm_jac.T@cov2@norm_jac


        self.sat_match(satj,state2)
        vecsj = os_local_vecs(os,state2[3:7])
        extra_obj.sens_of_state2 = satj.sensor_values(state2[0:self.sat.state_len],vecsj,which_sensors).copy()

        extra_obj.cov1 = cov1
        extra_obj.cov2 = cov2
        tmp = np.zeros(state2.shape[0])
        tmp[3:7] = post_quat
        extra_obj.sig1 = self.add_to_state(tmp,post_pts).copy()
        extra_obj.mean2 = state2.copy()
        extra_obj.sens_state = self.add_to_state(tmp,state1 +  (sensors_in[which_sensors]-sens1)@covyx)



        return estimated_nparray(state2,cov2),extra_obj#(pred_dyn_state,psens,sens1)
