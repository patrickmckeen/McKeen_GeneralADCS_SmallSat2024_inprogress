from .attitude_estimator import *
import copy
import scipy
from choldate import cholupdate, choldowndate
import copy


class SRUK_copy(Estimator):
    def __init__(self, sat,estimate,cov_estimate,integration_cov,sample_time = 1,use_cross_term = False,quat_as_vec = False):
        super().__init__(sat,estimate,cov_estimate,integration_cov,sample_time,use_cross_term,quat_as_vec)
        self.al = 1.0#1e-3
        self.kap = 0.0
        self.bet = 2.0#-1.0#2.0
        self.include_int_noise_separately = False
        self.include_sens_noise_separately = False
        self.scale_nonseparate_adds = False
        self.sepscale = 1
        try:
            self.srcov = np.linalg.cholesky(self.use_state.cov).T
        except:
            w,v = np.linalg.eig(self.use_state.cov)
            srw = np.diagflat(np.sqrt(np.abs(np.real(w))))#[math.sqrt(j) for j in np.abs(np.real(w))]
            v = np.real(v)
            self.srcov  = (v@srw@v.T).T#@np.linalg.inv(v)
        try:
            self.sric = np.linalg.cholesky(self.use_state.int_cov).T
        except:
            w,v = np.linalg.eig(self.use_state.int_cov)
            srw = np.diagflat(np.sqrt(np.abs(np.real(w))))#[math.sqrt(j) for j in np.abs(np.real(w))]
            v = np.real(v)
            self.sric  = (v@srw@v.T).T#@np.linalg.inv(v)
        # self.vec_mode = 6

    def weighted_cholupdate(self,mat,vec,wt):
        vec = vec.copy()
        s = np.array(np.shape(vec))
        if len(s)>2:
            raise ValueError('can only handle vectors and matrices')
        if not np.any(s==mat.shape[0]):
            raise ValueError('not the right shape')
        if len(s)==2 and not np.any(s==1):
            rr = s[0]
            if s[1]!=mat.shape[0]:
                vec = vec.T.copy()
                rr = s[1]
            for j in range(rr):
                mat = self.weighted_cholupdate(mat.copy(),vec[j,:].copy(),wt)
        else:
            # mat = self.mycholupdate(mat,(np.abs(wt)**0.5)*vec,np.sign(wt))
            if wt>=0:
                cholupdate(mat,(wt**0.5)*vec)
            else:
                choldowndate(mat,((-wt)**0.5)*vec)
        return mat

    #
    # def mycholupdate(self,mat, vec,sign):
    #     vec = np.ravel(vec)
    #     n = np.size(vec)
    #     for k in range(n):
    #         r = math.sqrt(mat[k, k]**2 + sign*vec[k]**2)
    #         c = r / mat[k, k]
    #         s = vec[k] / mat[k, k]
    #         mat[k, k] = r
    #         mat[k,k+1:n] = (mat[k,k+1:n] + sign*s*vec[k+1:n])/c
    #         vec[k+1:n] = c*vec[k+1:n] - s*mat[k, k+1:n]
    #     return mat


    def make_pts_and_wts(self,pt0,which_sensors):

        state_srcov = self.srcov
        int_srcov = self.sric*self.include_int_noise_separately
        control_srcov = self.sat.control_srcov()
        use_control = np.size(control_srcov)>0 and not np.all(control_srcov==0)
        control_srcov *= use_control
        sens_srcov = self.sat.sensor_srcov(which_sensors)*self.include_sens_noise_separately

        include_srcov = [True,self.include_sens_noise_separately,use_control,self.include_int_noise_separately]
        srcovs = [state_srcov,sens_srcov,control_srcov,int_srcov]
        zeros = [pt0,sens_srcov[0,:]*0,control_srcov[0,:]*0,int_srcov[0,:]*0]

        L = np.sum([include_srcov[j]*np.size(srcovs[j],0) for j in range(4)])

        lam = self.al**2.0*(self.kap+L)-L#3#3-L#self.al**2.0*(k+L)-L

        # lam = self.al**2.0*(self.kap)-L#3#3-L#self.al**2.0*(k+L)-L
        offsets = [0,0,0,0]
        pts = [zeros]
        scale = math.sqrt(L+lam)
        if self.scale_nonseparate_adds:
            self.sepscale = scale


        for j in range(4):
            if include_srcov[j]:
                mat = scale*srcovs[j]
                offsets[j] = np.vstack([mat,-mat])
                if j == 0:
                    states = self.add_to_state(pt0,offsets[0])
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in states]
                else:
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in offsets[j]]

        wts_m = np.array([lam/(L+lam)]+[0.5/(L+lam) for j in range(2*L)])
        wts_c = np.array([lam/(L+lam) + (1.0 - self.al**2.0 + self.bet)]+[0.5/(L+lam) for j in range(2*L)])
        self.wts_m = wts_m
        self.wts_c = wts_c
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
        # breakpoint()
        state0 = self.use_state.val.copy()
        quat0 = state0[3:7].copy()
        #find repeated orbital state once
        mid_os = self.prev_os.average(os)
        # mid_os = [self.prev_os.average(os,CG5_c[j]) for j in range(5)]
        dyn_state0 = self.sat.rk4(state0[0:self.sat.state_len],control_vec,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = True)
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
        extra_obj.cov0 = (self.srcov.T@self.srcov).copy()
        extra_obj.sig0 = sig0
        extra_obj.mean0 = sig0[0,:].copy()
        extra_obj.full_sig = copy.deepcopy(pts)
        post_full =  np.nan*np.ones((2*L+1,sigma_state_len+1))
        for j in range(2*L+1): #TODO vectorize
            [full_pre_statej,sens_noise_j,control_noise_j,int_noise_extra_obj_j] = pts[j]

            self.sat_match(satj,full_pre_statej)
            post_dyn_state_j = satj.rk4(full_pre_statej[0:self.sat.state_len],control_vec + control_noise_j,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = True)

            if j == 0:
                post_quat = post_dyn_state_j[3:7]#can happen before integration noise is added because j=0 has 0 integration noise
            post_statej,post_full_statej = self.new_post_state(full_pre_statej[self.sat.state_len:],post_dyn_state_j,int_noise_extra_obj_j,post_quat)
            post_pts[j,:] = post_statej.copy()
            post_full[j,:] = post_full_statej.copy()

            self.sat_match(satj,post_full_statej)
            vecsj = os_local_vecs(os,post_full_statej[3:7])
            sensj = satj.sensor_values(post_full_statej[0:self.sat.state_len],vecsj,which_sensors) + sens_noise_j

            post_sens[j,:] = sensj.copy()

        # breakpoint()
        state1 = np.dot(wts_m,post_pts)
        dquat1 = vec3_to_quat(state1[3:6],self.vec_mode)
        quat1 = quat_mult(post_quat,dquat1)
        pred_dyn_state = np.concatenate([state1[0:3],quat1,state1[6:self.sat.state_len-1],state1[self.sat.state_len-1:]])
        psens = self.sat.sensor_values(pred_dyn_state,os_local_vecs(os,pred_dyn_state[3:7]))
        sens1 = np.dot(wts_m,post_sens)
        extra_obj.post_full = post_full.copy()
        # breakpoint()
        # breakpoint()
        pts_diff = post_pts - state1
        sens_diff = post_sens - sens1
        if not self.include_int_noise_separately:
            srcov1 = np.linalg.qr(np.hstack([pts_diff[1:,:].T*np.sqrt(wts_c[1:]),self.sric*self.sepscale]).T,mode = 'r')
        else:
            srcov1 = np.linalg.qr((pts_diff[1:,].T*np.sqrt(wts_c[1:])).T,mode = 'r')
        srcov1 = self.weighted_cholupdate(srcov1.copy(),pts_diff[0,:].copy(),wts_c[0])



        if not self.include_int_noise_separately:
            srcov_sens = np.linalg.qr(np.hstack([sens_diff[1:,:].T*np.sqrt(wts_c[1:]),self.sat.sensor_srcov(which_sensors)*self.sepscale]).T,mode = 'r')
        else:
            srcov_sens = np.linalg.qr((sens_diff[1:,].T*np.sqrt(wts_c[1:])).T,mode = 'r')
        srcov_sens = self.weighted_cholupdate(srcov_sens.copy(),sens_diff[0,:].copy(),wts_c[0])
        # srcov_sens = srcov_sens.T

        extra_obj.senscov = (srcov_sens.T@srcov_sens).copy()
        extra_obj.sens1 = sens1.copy()
        self.sat_match(satj,pred_dyn_state)
        vecsj = os_local_vecs(os,pred_dyn_state[3:7])
        extra_obj.sens_of_state1 = satj.sensor_values(pred_dyn_state[0:self.sat.state_len],vecsj,which_sensors).copy()

        extra_obj.sens_sig = post_sens.copy()


        covyx = sum([wts_c[j]*np.outer(sens_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)],np.zeros((sens_vec_len,sigma_state_len)))#sum([wts_c[i]*(state_pts_err[i]-x1_cut_red)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        # covyx_test = covyx.copy()
        try:

            # Kk = np.linalg.inv(covyy)@covyx
            # Kk = np.linalg.solve(srcov_sens,np.linalg.solve(srcov_sens.T,covyx))
            # Kk = np.linalg.solve(srcov_sens@srcov_sens.T,srcov_sens.T@np.linalg.solve(srcov_sens.T@srcov_sens,srcov_sens@covyx))
            # Kk = np.linalg.lstsq(np.linalg.lstsq(covyx.T,srcov_sens.T)[0],srcov_sens)
            # Kk = np.linalg.lstsq(np.linalg.lstsq(covyx.T,srcov_sens.T)[0],srcov_sens)

            # Pyyinv@Kk = Pyx
            # Kk = np.linalg.lstsq(srcov_sens,np.linalg.lstsq(srcov_sens.T,covyx)[0])[0]
            # Kk = np.linalg.lstsq(srcov_sens.T,np.linalg.lstsq(srcov_sens,covyx)[0])[0]
            # Kk = np.linalg.solve(srcov_sens,np.linalg.solve(srcov_sens.T,covyx))
            # Kk = np.linalg.solve(srcov_sens.T,np.linalg.solve(srcov_sens,covyx))
            # Kk = np.linalg.lstsq(srcov_sens@srcov_sens.T,srcov_sens.T@np.linalg.lstsq(srcov_sens.T@srcov_sens,srcov_sens@covyx)[0])[0]
            # Kk = np.linalg.lstsq(srcov_sens.T@srcov_sens,srcov_sens.T@np.linalg.lstsq(srcov_sens@srcov_sens.T,srcov_sens@covyx)[0])[0]
            # Kk = np.linalg.solve(srcov_sens@srcov_sens.T,srcov_sens.T@np.linalg.solve(srcov_sens.T@srcov_sens,srcov_sens@covyx))
            # Kk = np.linalg.solve(srcov_sens.T@srcov_sens,srcov_sens.T@np.linalg.solve(srcov_sens@srcov_sens.T,srcov_sens@covyx)) #kinda works!
            Kk = scipy.linalg.solve_triangular(srcov_sens,scipy.linalg.solve_triangular(srcov_sens,covyx,trans = 'T'))

        except:
            breakpoint()
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')

        # breakpoint()
        state2 = state1 + (sensors_in[which_sensors]-sens1)@Kk
        extra_obj.mean1 = pred_dyn_state.copy()
        srcov2 = srcov1.copy()
        U = (srcov_sens@Kk).copy()
        srcov2 = self.weighted_cholupdate(srcov2.copy(),U.copy(),-1)
        self.srcov = srcov2.copy()
        cov2 = srcov2.T@srcov2
        # print(np.diag(cov2-srcov2.T@srcov2))

        # cov1 = sum([wts_c[j]*np.outer(pts_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)])
        # if not self.include_int_noise_separately:
        #     cov1 += self.use_state.int_cov
        # covyy = sum([wts_c[j]*np.outer(sens_diff[j,:],sens_diff[j,:]) for j in range(2*L+1)],0*np.eye(sens_vec_len))#sum([wts_c[i]*(post_sens[j,:]-sens1)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        # if not self.include_sens_noise_separately:
        #     covyy += self.sat.sensor_cov(which_sensors)
        # Kk_test = scipy.linalg.solve(covyy,covyx_test
        # cov2_test = cov1 - Kk_test.T@covyy@Kk_test
        # print(np.amax(np.abs(np.diag(cov1-srcov1.T@srcov1))),np.amax(np.abs(np.diag(covyy-srcov_sens.T@srcov_sens))),np.amax(np.abs(np.diag(cov2_test-srcov2.T@srcov2))))

        if not self.quat_as_vec:
            dvec3 = state2[3:6]
            dquat = vec3_to_quat(dvec3,self.vec_mode)
            quat = quat_mult(post_quat,dquat)
            state2 = np.concatenate([state2[0:3],quat,state2[6:self.sat.state_len-1],state2[self.sat.state_len-1:]])
            # breakpoint()
        else:
            state20 = np.copy(state2)
            state2[3:7] = normalize(state2[3:7])
            norm_jac = state_norm_jac(state20)
            cov2 = norm_jac.T@cov2@norm_jac


        self.sat_match(satj,state2)
        vecsj = os_local_vecs(os,state2[3:7])
        extra_obj.sens_of_state2 = satj.sensor_values(state2[0:self.sat.state_len],vecsj,which_sensors).copy()

        extra_obj.cov1 = srcov1.T@srcov1
        extra_obj.cov2 = cov2
        tmp = np.zeros(state2.shape[0])
        tmp[3:7] = post_quat
        extra_obj.sig1 = self.add_to_state(tmp,post_pts).copy()
        extra_obj.mean2 = state2.copy()
        extra_obj.sens_state = self.add_to_state(tmp,state1 +  (sensors_in[which_sensors]-sens1)@covyx)
        return estimated_nparray(state2,cov2),extra_obj
