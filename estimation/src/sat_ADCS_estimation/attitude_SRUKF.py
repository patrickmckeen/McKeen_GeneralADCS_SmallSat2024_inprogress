from .attitude_estimator import *
import copy
import scipy
from choldate import cholupdate, choldowndate
import copy


class SRUKF(Estimator):
    def __init__(self, j2000,sat,estimate,cov_estimate,integration_cov,sample_time = 1,use_cross_term = False,quat_as_vec = False,adj_kap_by_L = False,sunsensors_during_eclipse = False,verbose = False):
        super().__init__(j2000,sat,estimate,cov_estimate,integration_cov,sample_time,use_cross_term,quat_as_vec,sunsensors_during_eclipse = sunsensors_during_eclipse,verbose = verbose)
        self.al = 1.0#1e-3
        self.kap = 0.0
        self.bet = 2.0#-1.0#2.0
        self.include_int_noise_separately = False
        self.include_sens_noise_separately = False
        self.use_estimated_int_noise = False
        self.estimated_int_noise_scale = 1.0
        self.scale_nonseparate_adds = False
        self.estimated_int_noise_other_scale = 0.0
        self.sepscale = 1
        self.adj_kap_by_L = adj_kap_by_L #use if kappa is something like 3-L (negative linear with L) and sometimes L gets small (like if sensors aren't used at some times)
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
        # vec0 = vec.copy()
        # mat0 = mat.copy()
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
                # mr_mat = mat.copy()
                # mr_vec = vec.copy()
                mat = self.weighted_cholupdate(mat,vec[j,:],wt)
                # if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
                #     breakpoint()
                #     raise np.linalg.LinAlgError('NAN IN date')
        else:
            # mat = self.mycholupdate(mat,(np.abs(wt)**0.5)*vec,np.sign(wt))
            if wt>=0:
                cholupdate(mat,(wt**0.5)*vec)
            else:
                choldowndate(mat,((-wt)**0.5)*vec)
        return mat


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

        state_srcov = self.srcov.copy()
        # print('cov1',np.diag(self.srcov))
        # if self.adj_kap_by_L:
        #     rem_inds = [self.sat.state_len+self.sat.act_bias_len + j for j in range(len(which_sensors)) if (not which_sensors[j])*self.sat.attitude_sensors[j].estimated_bias  ]
        #     use_inds = np.array([j for j in range(int(state_srcov.shape[0])) if j not in rem_inds])
        #     state_srcov[:,rem_inds] = 0
        # breakpoint()
        int_srcov = self.sric*self.include_int_noise_separately
        control_srcov = self.sat.control_srcov()
        use_control = np.size(control_srcov)>0 and not np.all(control_srcov==0)
        control_srcov *= use_control
        # print(contrl_srcov)
        sens_srcov = self.sat.sensor_srcov(which_sensors)*self.include_sens_noise_separately

        include_srcov = [True,self.include_sens_noise_separately,use_control,self.include_int_noise_separately]
        srcovs = [state_srcov,sens_srcov,control_srcov,int_srcov]
        zeros = [pt0,sens_srcov[0,:]*0,control_srcov[0,:]*0,int_srcov[0,:]*0]

        L = int(np.sum([include_srcov[j]*np.size(srcovs[j],0) for j in range(4)]))

        Lk_adj = int(self.adj_kap_by_L*sum([self.sat.attitude_sensors[j].output_length*(not which_sensors[j])*self.sat.attitude_sensors[j].estimated_bias for j in range(len(which_sensors))]))
        L -= Lk_adj

        lam = self.al**2.0*(self.kap+L+Lk_adj )-L#3#3-L#self.al**2.0*(k+L)-L
        # print(L,Lk_adj,lam)


        # lam = self.al**2.0*(self.kap)-L#3#3-L#self.al**2.0*(k+L)-L
        offsets = [0,0,0,0]
        pts = [zeros]
        scale = math.sqrt(L+lam)
        if self.scale_nonseparate_adds:
            self.sepscale = scale

        # print(pt0)
        for j in range(4):
            if include_srcov[j]:
                mat = scale*srcovs[j]
                offsets[j] = np.vstack([mat,-mat])
                if j == 0:
                    states = self.add_to_state(pt0,offsets[0])
                    # print(states)
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in states]
                else:
                    pts += [zeros[:j]+[k]+zeros[j+1:] for k in offsets[j]]

        wts_m = np.array([lam/(L+lam)]+[0.5/(L+lam) for j in range(2*L)])
        wts_c = np.array([lam/(L+lam) + (1.0 - self.al**2.0 + self.bet)]+[0.5/(L+lam) for j in range(2*L)])
        self.wts_m = wts_m
        self.wts_c = wts_c
        # breakpoint()
        return L,pts,wts_m,wts_c,np.vstack([pt0,states]+[pt0]*(2*L-states.shape[0]))

    def reunite_states(self,dynstate,rest_state,quatref):
        if self.quat_as_vec:
            return np.concatenate([dynstate,rest_state])
        else:
            quatdiff = quat_mult(quat_inv(quatref),dynstate[3:7])
            v3diff = quat_to_vec3(quatdiff,self.vec_mode)
            return np.concatenate([dynstate[0:3],v3diff,dynstate[7:],rest_state])

    def new_post_state(self,pre_rest_state,post_dynstate,int_err,quatref):
        post_dyn_state_w_int_err = self.add_to_state(post_dynstate,int_err[0:self.sat.state_len - 1 + self.quat_as_vec])
        post_state = self.reunite_states(post_dyn_state_w_int_err,pre_rest_state+int_err[self.sat.state_len - 1 + self.quat_as_vec:],quatref)
        s0len = np.zeros(np.size(post_state) + 1 - self.quat_as_vec)
        s0len[3:7] = quatref
        full_state = self.add_to_state(s0len,post_state)#these are backwards on purpose
        # breakpoint()
        return post_state,full_state

    def update_core(self, control_vec, sensors_in, os,which_sensors):
        # control_vec = np.copy(control_vec)
        # os = os.copy()

        #take out values
        # breakpoint()
        which_sensors = which_sensors.copy()
        which_sensors_plus = which_sensors + [True]*len(self.sat.momentum_inds)

        state0 = self.use_state.val.copy()
        # print(state0)
        quat0 = state0[3:7]
        #find repeated orbital state once
        mid_os = self.prev_os.average(os)
        # print("&&&&&&&MIDOS&&&&&&&&",mid_os.R)
        # mid_os = [self.prev_os.average(os,CG5_c[j]) for j in range(5)]
        dyn_state0 = self.sat.noiseless_rk4(state0[0:self.sat.state_len],control_vec,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = True)
        vecs0 = os_local_vecs(os,dyn_state0[3:7])
        for j in range(len(self.sat.attitude_sensors)): #ignore sun sensors that should be in shadow
            which_sensors[j] &= not (isinstance(self.sat.attitude_sensors[j],SunSensor) and self.sat.attitude_sensors[j].clean_reading(dyn_state0,vecs0)<1e-10)


        sens_vec_len = sum([self.sat.attitude_sensors[j].output_length for j in range(len(self.sat.attitude_sensors)) if which_sensors[j]]) + sum([1 for j in self.sat.momentum_inds])
        #generate sigma points of augmemted state--state itself, including actuator bias values, disturbance values, sensor bias values; sensor noise , control noise to use, snesor noise to use, possibly integration noise to use.
        L,pts,wts_m,wts_c,sig0 = self.make_pts_and_wts(state0,which_sensors)

        if self.verbose:
            print('cov0',0.5*np.log10(np.diag(self.srcov.T@self.srcov)))
            print('state0',state0)
        sigma_state_len = len(state0) - 1 + self.quat_as_vec
        post_pts = np.nan*np.ones((2*L+1,sigma_state_len))
        post_sens = np.nan*np.ones((2*L+1,sens_vec_len))
        satj = self.sat#copy.deepcopy(self.sat) #TODO: this needs to be deepcopy, right?
        whichj = which_sensors

        extra_obj = extra()
        post_full =  np.nan*np.ones((2*L+1,sigma_state_len+1))
        for j in range(2*L+1): #TODO vectorize
            [full_pre_statej,sens_noise_j,control_noise_j,int_noise_extra_obj_j] = pts[j]
            # print(control_noise_j)
            # print(sens_noise_j)
            # print(control_vec,self.update_period)

            self.sat_match(satj,full_pre_statej)
            # print(satj.last_dist_torq)
            # print(satj.last_act_torq)
            # print(post_dyn_state_j)

            if j == 0:
                if self.use_estimated_int_noise and not self.estimated_int_noise_scale==0:
                    post_dyn_state_j,err_est = satj.noiseless_rk4(full_pre_statej[0:self.sat.state_len],control_vec + control_noise_j,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = True,save_info = False,give_err_est = True)
                    int_err_est = self.sric*self.sepscale
                    err_est_reduced = err_est
                    # err_est_reduced = np.zeros(err_est.size-1)
                    # err_est_reduced[0:3] = err_est[0:3]
                    # err_est_reduced[6:] = err_est[7:]
                    # err_est_reduced[3:6] = np.abs(quat_to_vec3(err_est[3:7],self.vec_mode))#@quat_to_vec3_deriv(post_dyn_state_j[3:7],0)
                    # int_err_est[:self.sat.state_len-1,:] = 0
                    # int_err_est[:,:self.sat.state_len-1] = 0
                    int_err_est[:self.sat.state_len-1,:self.sat.state_len-1] = int_err_est[:self.sat.state_len-1,:self.sat.state_len-1]*self.estimated_int_noise_other_scale + np.diagflat(err_est_reduced*self.estimated_int_noise_scale)
                    # print(np.log10(err_est_reduced*self.estimated_int_noise_scale))
                    # print(np.log10(np.diag(self.sric*self.sepscale)))
                else:
                    post_dyn_state_j = satj.noiseless_rk4(full_pre_statej[0:self.sat.state_len],control_vec + control_noise_j,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = True,save_info = False)

                post_quat = post_dyn_state_j[3:7]#can happen before integration noise is added because j=0 has 0 integration noise
            else:
                post_dyn_state_j = satj.noiseless_rk4(full_pre_statej[0:self.sat.state_len],control_vec + control_noise_j,self.update_period,self.prev_os,os,mid_orbital_state = mid_os,quat_as_vec = True,save_info = False)
            post_statej,post_full_statej = self.new_post_state(full_pre_statej[self.sat.state_len:],post_dyn_state_j,int_noise_extra_obj_j,post_quat)

            if self.verbose:
                print('******',j)
                print(full_pre_statej)
                print(post_full_statej)
            post_pts[j,:] = post_statej#
            post_full[j,:] = post_full_statej#.copy()

            self.sat_match(satj,post_full_statej)
            vecsj = os_local_vecs(os,post_full_statej[3:7])
            sensj = satj.noiseless_sensor_values(post_full_statej[0:self.sat.state_len],vecsj,which_sensors) + sens_noise_j
            post_sens[j,:] = sensj#.copy()

        # breakpoint()
        state1 = np.dot(wts_m,post_pts)
        dquat1 = vec3_to_quat(state1[3:6],self.vec_mode)
        quat1 = quat_mult(post_quat,dquat1)
        pred_dyn_state = np.concatenate([state1[0:3],quat1,state1[6:self.sat.state_len-1],state1[self.sat.state_len-1:]])

        self.sat_match(satj,pred_dyn_state)
        psens = satj.noiseless_sensor_values(pred_dyn_state,os_local_vecs(os,pred_dyn_state[3:7]))
        sens1 = np.dot(wts_m,post_sens)
        # print(sens1)
        # print(sensors_in[which_sensors])
        # extra_obj.post_full = post_full.copy()
        # breakpoint()
        # breakpoint()
        pts_diff = post_pts - state1
        sens_diff = post_sens - sens1
        if not self.include_int_noise_separately:
            if self.use_estimated_int_noise:
                srcov1 = np.linalg.qr(np.hstack([pts_diff[1:,:].T*np.sqrt(wts_c[1:]),int_err_est]).T,mode = 'r')
            else:
                srcov1 = np.linalg.qr(np.hstack([pts_diff[1:,:].T*np.sqrt(wts_c[1:]),self.sric*self.sepscale]).T,mode = 'r')
        else:
            srcov1 = np.linalg.qr((pts_diff[1:,].T*np.sqrt(wts_c[1:])).T,mode = 'r')

        if np.any(np.isnan(srcov1)) or np.any(np.isinf(srcov1)):
            # breakpoint()
            raise np.linalg.LinAlgError('NAN IN COV')
        srcov1 = self.weighted_cholupdate(srcov1,pts_diff[0,:],wts_c[0])

        if np.any(np.isnan(srcov1)) or np.any(np.isinf(srcov1)):
            # breakpoint()
            raise np.linalg.LinAlgError('NAN IN COV')

        if self.verbose:
            print('cov1',0.5*np.log10(np.diag(srcov1.T@srcov1)))
            print('state1',pred_dyn_state)
            print('sens1',sens1)
        # breakpoint()



        if not self.include_sens_noise_separately:
            srcov_sens = np.linalg.qr(np.hstack([sens_diff[1:,:].T*np.sqrt(wts_c[1:]),self.sat.sensor_srcov(which_sensors)*self.sepscale]).T,mode = 'r')
        else:
            srcov_sens = np.linalg.qr((sens_diff[1:,].T*np.sqrt(wts_c[1:])).T,mode = 'r')
        srcov_sens = self.weighted_cholupdate(srcov_sens,sens_diff[0,:],wts_c[0])
        # print(srcov_sens)
        # srcov_sens = srcov_sens.T

        # extra_obj.senscov = (srcov_sens.T@srcov_sens).copy()
        # print(np.diagonal(srcov_sens.T@srcov_sens).copy())
        # extra_obj.sens1 = sens1.copy()
        # self.sat_match(satj,pred_dyn_state)
        vecsj = os_local_vecs(os,pred_dyn_state[3:7])
        # extra_obj.sens_of_state1 = satj.sensor_values(pred_dyn_state[0:self.sat.state_len],vecsj,which_sensors).copy()
        #
        # extra_obj.sens_sig = post_sens.copy()


        covyx = sum([wts_c[j]*np.outer(sens_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)],np.zeros((sens_vec_len,sigma_state_len)))#sum([wts_c[i]*(state_pts_err[i]-x1_cut_red)@(sens_pts[i]-sens1).T for i in range(2*L+1)])
        # covyx_test = covyx.copy()
        try:

            # Kk = np.linalg.inv(covyy)@covyx
            # Kk = np.linalg.solve(srcov_sens,np.linalg.solve(srcov_sens.T,covyx))
            # Kk = np.linalg.solve(srcov_sens@srcov_sens.T,srcov_sens.T@np.linalg.solve(srcov_sens.T@srcov_sens,srcov_sens@covyx))
            # Kk = np.linalg.lstsq(np.linalg.lstsq(covyx.T,srcov_sens.T)[0],srcov_sens)
            # Kk = np.linalg.lstsq(np.linalg.lstsq(covyx.T,srcov_sens.T)[0],srcov_sens)

            # Pyyinv@Kk = Pyx
            Kk = np.linalg.lstsq(srcov_sens,np.linalg.lstsq(srcov_sens.T,covyx,rcond=None)[0],rcond=None)[0]
            # Kk = scipy.linalg.solve_triangular(srcov_sens,scipy.linalg.solve_triangular(srcov_sens,covyx,trans = 'T'))
            # Kk = scipy.linalg.solve_triangular(srcov_sens,scipy.linalg.solve_triangular(srcov_sens,covyx,trans = 'T'))



            # Kk = np.linalg.solve(srcov_sens,np.linalg.solve(srcov_sens.T,covyx))
            # Kk = np.linalg.solve(srcov_sens.T,np.linalg.solve(srcov_sens,covyx))
            # Kk = np.linalg.lstsq(srcov_sens.T,np.linalg.lstsq(srcov_sens,covyx)[0])[0]
            # Kk = np.linalg.lstsq(srcov_sens@srcov_sens.T,srcov_sens.T@np.linalg.lstsq(srcov_sens.T@srcov_sens,srcov_sens@covyx)[0])[0]
            # Kk = np.linalg.lstsq(srcov_sens.T@srcov_sens,srcov_sens.T@np.linalg.lstsq(srcov_sens@srcov_sens.T,srcov_sens@covyx)[0])[0]
            # Kk = np.linalg.solve(srcov_sens@srcov_sens.T,srcov_sens.T@np.linalg.solve(srcov_sens.T@srcov_sens,srcov_sens@covyx))
            # Kk = np.linalg.solve(srcov_sens.T@srcov_sens,srcov_sens.T@np.linalg.solve(srcov_sens@srcov_sens.T,srcov_sens@covyx)) #kinda works!


        except:
            breakpoint()
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')

        # breakpoint()
        if self.verbose:
            # breakpoint()
            print('real sens',sensors_in[which_sensors_plus])
        state2 = state1 + (sensors_in[which_sensors_plus]-sens1)@Kk
        # extra_obj.mean1 = pred_dyn_state.copy()
        srcov2 = srcov1.copy()
        # if np.any(np.isnan(srcov2)) or np.any(np.isinf(srcov2)):
        #     breakpoint()
        #     raise np.linalg.LinAlgError('NAN IN COV')
        U = (srcov_sens@Kk)
        # if np.any(np.isnan(U)) or np.any(np.isinf(U)):
        #     breakpoint()
        #     raise np.linalg.LinAlgError('NAN IN COV')
        srcov2 = self.weighted_cholupdate(srcov2,U,-1)
        if np.any(np.isnan(srcov2)) or np.any(np.isinf(srcov2)):
            warnings.warn('NAN IN COV')
            srcov2 = srcov1.copy()
        self.srcov = srcov2
        # print('cov2',np.diag(self.srcov))
        cov2 = srcov2.T@srcov2
        # if np.any(np.isnan(self.srcov)) or np.any(np.isinf(self.srcov)):
        #     breakpoint()
        #     raise np.linalg.LinAlgError('NAN IN COV')

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

        if self.verbose:
            print('cov2',0.5*np.log10(np.diag(cov2)))
            print('state2',state2)


        self.sat_match(satj,state2)
        # self.os_vecs = os_local_vecs(os,state2[3:7])
        # extra_obj.sens_of_state2 = satj.sensor_values(state2[0:self.sat.state_len],vecsj,which_sensors).copy()
        #
        # extra_obj.cov1 = srcov1.T@srcov1
        # extra_obj.cov2 = cov2
        tmp = np.zeros(state2.shape[0])
        tmp[3:7] = post_quat
        # extra_obj.sig1 = self.add_to_state(tmp,post_pts).copy()
        # extra_obj.mean2 = state2.copy()
        # extra_obj.sens_state = self.add_to_state(tmp,state1 +  (sensors_in[which_sensors]-sens1)@covyx)
        return estimated_nparray(state2,cov2),extra_obj

    def change_vals_from_inds(self,valinds,new_val=None,newcov=None,new_int_cov = None,clearx=True):
        super().change_vals_from_inds(valinds,new_val,newcov,new_int_cov,clearx)
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
