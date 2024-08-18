from .attitude_estimator import *


class EKF_bias_only(Estimator):

    def update_core(self,control_vec,sensors_in,os,which_sensors):
        gyro_inds = [j for j in range(len(sensors_in)) if isinstance(self.sat.sensors[j],Gyro)]
        gyro_input_inds = [self.att_sens_input_inds[j] for j in gyro_inds]
        gyro_valinds = [self.att_sens_bias_valinds[j] for j in range(len(self.sat.att_sens_bias_inds)) if isinstance(self.sat.sensors[self.sat.att_sens_bias_inds[j]],Gyro) and self.sat.sensors[self.sat.att_sens_bias_inds[j]].has_bias]
        gyros_reading = np.concatenate([sensors_in[j] for j in gyro_input_inds]) #TODO what if there are not exactly 3 gyros
        gyro_biases = np.concatenate([self.use_state.val[j] for j in gyro_valinds])
        pw = gyros_reading - gyro_biases

        pq = normalize(self.use_state.val[3:7] + 0.5*pw@Wmat(self.use_state.val[3:7]).T)
        pv = self.use_state.val.copy()[3:]
        pv[3:7] = pq
        # pc = square_mat_mask(pc,total_use[3:])

        pv0 = np.concatenate([pw,pv])

        vecs = os_local_vecs(os,pq)
        for j in range(len(self.sat.sensors)): #ignore sun sensors that should be in shadow
            which_sensors[j] &= not (isinstance(self.sat.sensors[j],SunSensor) and self.sat.sensors[j].clean_reading(pv0,vecs)<1e-10)



        total_use = self.use.copy()
        # for j in range(len(self.sat.att_sens_bias_inds)):
        #     val_inds = self.att_sens_bias_valinds[j]
        #     total_use[val_inds] &= which_sensors[self.sat.att_sens_bias_inds[j]]

        # pv = pv[total_use[3:]]
        # pv0 = pv0[total_use]

        # breakpoint()
        # pb = gyro_biases
        # pr = np.array([self.use_state.val[j] for j in range(10,len(self.use_state.val)) if j not in gyro_valinds])
        F = state_norm_jac(self.use_state.val)[3:,3:]
        pc = F.T@self.use_state.cov[3:,3:]@F + self.use_state.int_cov[3:,3:]


        for j in gyro_inds:
            which_sensors[j] = False

        which_w_bias = np.array([which_sensors[j] for j in self.sat.att_sens_bias_inds])


        #reduced to relevant sensors and disturbances, etc.

        # pv = self.use_state.val.copy()[3:]
        # pv = pv[total_use[3:]]
        # pc = square_mat_mask(pc,total_use[3:])
        #
        # pv0 = np.concatenate([nan3.copy(),pv])
        # pic = square_mat_mask(pic,total_use)

        #
        #sensor jacobian
        state_jac,bias_jac = self.sat.sensor_state_jacobian(pv0,vecs,which = which_sensors,keep_unused_biases=True)
        state_jac = state_jac[3:,:]
        lenA = np.size(state_jac,1)
        # bias_jac0 = np.zeros((self.sat.att_sens_bias_len,lenA))
        # breakpoint()
        # bias_jac0[which_w_bias,:] = bias_jac
        Hk = np.vstack([state_jac,np.zeros((self.sat.act_bias_len,lenA)),bias_jac,np.zeros((self.sat.dist_param_len-(len(self.use)-sum(self.use)),lenA))])
        sens_cov = self.sat.sensor_cov(which_sensors)
        lilhk = self.sat.sensor_values(pv0,vecs,which = which_sensors,keep_unused_biases=False)


        try:
            # print(np.linalg.cond(sens_cov_cut),np.linalg.cond(cov1_cut),np.linalg.cond(Hk_cut@cov1_cut@Hk_cut.T),np.linalg.cond((sens_cov_cut+Hk_cut@cov1_cut@Hk_cut.T)))
            # breakpoint()
            Kk = scipy.linalg.solve((sens_cov+Hk.T@pc@Hk),(pc@Hk).T,assume_a='sym')
            #Kk = (cov1@Hk1.T)@np.linalg.inv(sens_cov1+Hk1@cov1@Hk1.T)
        except:
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')

        state2 = pv + (sensors_in[which_sensors]-lilhk)@Kk


        state20 = np.copy(state2)
        state2[0:4] = normalize(state2[0:4])
        norm_jac = state_norm_jac(np.concatenate([np.zeros(3),state20]))[3:,3:]
        cov2 = pc@(np.eye(len(pv))-Hk@Kk)

        # breakpoint()

        # for j in gyro_valinds:
        #     total_use[j] = True

        cov2 = 0.5*(cov2 + cov2.T)
        cov2 = norm_jac.T@cov2@norm_jac
        rv = self.use_state.val.copy()
        rc = self.use_state.cov.copy()
        rc[3:,3:] = cov2
        rv[3:] = state2
        rv[0:3] = pw
        # rc = square_mat_refill(cov2,total_use,rc)

        return estimated_nparray(rv,rc)
