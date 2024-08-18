from .attitude_estimator import *


class EKF(Estimator):

    def update_core(self,control_vec,sensors_in,os,which_sensors):
        scalar_update = False
        predicted_state = self.predictive_step(os,control_vec)
        #remove values not in use
        pv = predicted_state.val
        pc = predicted_state.cov

        vecs = os_local_vecs(os,pv[3:7])
        for j in range(len(self.sat.sensors)): #ignore sun sensors that should be in shadow
            which_sensors[j] &= not (isinstance(self.sat.sensors[j],SunSensor) and self.sat.sensors[j].clean_reading(pv,vecs)<1e-10)
        print(which_sensors)

        state_mod = np.eye(7)
        if not self.quat_as_vec:
            state_mod = block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(pv[3:7],self.vec_mode),self.vec_mode))

        if scalar_update:
            state2,cov2 = self.scalar_update(pc,pv,os)
        else:
            #sensor jacobian
            state_jac,bias_jac = self.sat.sensor_state_jacobian(pv,vecs,which = which_sensors,keep_unused_biases=True)
            lenA = np.size(state_jac,1)
            Hk = np.vstack([state_mod@state_jac,np.zeros((self.sat.act_bias_len,lenA)),bias_jac,np.zeros((self.sat.dist_param_len-(len(self.use)-sum(self.use)),lenA))])
            sens_cov = self.sat.sensor_cov(which_sensors,keep_unused_biases=False)
            lilhk = self.sat.sensor_values(pv,vecs,which = which_sensors)
            try:
                # breakpoint()
                Kk = scipy.linalg.solve((sens_cov+Hk.T@pc@Hk),(pc@Hk).T,assume_a='pos')
            except:
                raise np.linalg.LinAlgError('Matrix is singular. (probably)')

            dstate = (sensors_in[which_sensors]-lilhk)@Kk
            cov2 = pc@(np.eye(len(pv) - 1 + self.quat_as_vec)-Hk@Kk)
            cov2 = 0.5*(cov2 + cov2.T)

            if self.quat_as_vec:
                state2 = pv + dstate
                state20 = np.copy(state2)
                state2[3:7] = normalize(state2[3:7])
                norm_jac = state_norm_jac(state20)
                cov2 = norm_jac.T@cov2@norm_jac
            else:
                state2 = pv.copy()
                state2[0:3] += dstate[0:3]
                state2[7:] += dstate[6:]
                state2[3:7] = quat_mult(state2[3:7],vec3_to_quat(dstate[3:6],self.vec_mode))

        return estimated_nparray(state2,cov2)


    def predictive_step(self,os,control_vec):
        prev_control_vec = np.copy(control_vec)
        os = os.copy()

        #take out values
        state0 = self.use_state.val
        dyn_state0 = state0[0:self.sat.state_len].copy()
        cov0 = self.use_state.cov

        #find repeated orbital state once
        mid_os = self.prev_os.average(os)
        #propagate
        state1 = state0.copy()#non-base state parameters remain constant
        state1[0:self.sat.state_len] = self.sat.rk4(state0[0:self.sat.state_len],control_vec,self.update_period,self.prev_os,os,mid_orbital_state=mid_os)


        #find Jacobians
        A0,B0,dab0,dsb0,ddmp0 = self.sat.rk4Jacobians(dyn_state0,control_vec,self.update_period,self.prev_os,os,mid_orbital_state=mid_os)
        if self.quat_as_vec:
            B,combo = B0,np.vstack([A0,dab0,dsb0,ddmp0])
        else:
            T2 = block_diag(np.eye(3),quat_to_vec3_deriv(state1[3:7],self.vec_mode),np.eye(self.sat.state_len - 7))
            T1 =  block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(state0[3:7],self.vec_mode),self.vec_mode),np.eye(self.sat.state_len - 7))
            B = B0@T2
            combo = np.vstack([T1@A0,dab0,dsb0,ddmp0])@T2

        #make big Jacobians for full state
        BigA = np.eye(np.size(self.use_state,0) - 1 + self.quat_as_vec)
        BigB = np.zeros((self.sat.control_len,np.size(self.use_state,0) - 1 + self.quat_as_vec))
        BigA[:,:self.sat.state_len - 1 + self.quat_as_vec] = combo
        BigB[:,:self.sat.state_len - 1 + self.quat_as_vec] = B

        #covaraince propagation
        cov1 =  self.use_state.int_cov + BigA.T@cov0@BigA + BigB.T@self.sat.control_cov()@BigB
        #sensor biases go through udate even if unused, because biases drift even if you don't sample
        ##let disabled disturbances go for now, because their values and covariances should be 0.
        res =  estimated_nparray(state1,cov1)
        return res

    def scalar_update(self,pc,pv,os):
        pcs = pc.copy()
        pvs = pv.copy()
        bias_ind = 0
        reading_ind = 0
        for j in range(len(self.sat.attitude_sensors)):
            sensj = self.sat.attitude_sensors[j]
            if which_sensors[j]:
                vecsj = os_local_vecs(os,pvs[3:7])
                if sensj.output_length == 1:
                    Hkj = np.zeros(sum(self.use))
                    Hkj[0:self.sat.state_len] = sensj.basestate_jac(pvs,vecsj).flatten()
                    if sensj.has_bias:
                        Hkj[self.len_before_sens_bias+bias_ind] = sensj.bias_jac(pvs,vecsj).item()

                    Kkj = (pcs@Hkj)/(sensj.cov().item()+Hkj@pcs@Hkj)

                    pcs = pcs-pcs@np.outer(Hkj,Kkj)
                    pvs = pvs + (sensors_in[reading_ind]-sensj.reading(pvs,vecsj).item())*Kkj


                else:
                    Hkj = np.zeros((sum(self.use),sensj.output_length))
                    Hkj[0:self.sat.state_len,:] = sensj.basestate_jac(pvs,vecsj)
                    if sensj.has_bias:
                        Hkj[self.len_before_sens_bias+bias_ind:self.len_before_sens_bias+bias_ind+satj.output_length,:] = sensj.bias_jac(pvs,vecsj)

                    try:
                        Kkj = scipy.linalg.solve((sensj.cov()+Hkj.T@pcs@Hkj),(pcs@Hkj).T,assume_a='sym')
                    except:
                        raise np.linalg.LinAlgError('Matrix is singular. (probably)')


                    pcs = pcs-pcs@Hkj@Kkj
                    pvs = pvs + (sensors_in[reading_ind:reading_ind+sensj.output_length]-sensj.reading(pvs,vecsj))@Kkj

                pvs0 = np.copy(pvs)
                pvs[3:7] = normalize(pvs[3:7])
                norm_jac = state_norm_jac(pvs0)
                pcs = 0.5*(pcs + pcs.T)
                pcs = norm_jac.T@pcs@norm_jac

            if sensj.has_bias:
                bias_ind += sensj.output_length
            reading_ind += sensj.output_length
            print(j)
            print(sensj)
            print(np.diagonal(pcs)[0:3])
            print(np.diagonal(pcs)[3:7])
            print(np.diagonal(pcs)[7:10])
            print(np.diagonal(pcs)[10:14])
            breakpoint()
        mom_ind = sum([j.output_length for j in self.sat.sensors])
        vecs = os_local_vecs(os,pvs[3:7])
        for k in self.sat.momentum_inds:
            rwk = self.sat.actuators[k]
            Hkk = np.zeros(sum(self.use))
            Hkk[7+k] = 1

            Kkj = pcs[:,7+k]/(rwk.momentum_measurement_cov()+pcs[7+k,7+k])

            pcs = pcs-pcs@np.outer(Hkj,Kkj)
            pvs = pvs + (sensors_in[mom_ind]-rwk.measure_momentum())@Kkj

            pvs0 = np.copy(pvs)
            pvs[3:7] = normalize(pvs[3:7])
            norm_jac = state_norm_jac(pvs0)
            pcs = 0.5*(pcs + pcs.T)
            pcs = norm_jac.T@pcs@norm_jac
            mom_ind += 1
        return pvs, pcs
