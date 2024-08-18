import numpy as np
import scipy
import scipy.linalg
import random
import math
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
import pytest
from sat_ADCS_satellite import *
import warnings
from .attitude_estimator import *

class PerfectEstimator(Estimator):
    def update(self,control_vec,sensors_in,os,which_sensors = None,truth = None):
        if truth is None:
            error("can't be None")
        if self.has_sun and os.in_eclipse():
            self.byebye_sun()
        if not self.has_sun and not os.in_eclipse():
            self.hello_sun()
        if self.prev_os.R.all() == 0:
            self.prev_os = os
            self.prev_os_vecs = os_local_vecs(os,self.full_state.val[3:7])
        else:
            self.prev_os_vecs = self.os_vecs.copy()
        if which_sensors is None:
            which_sensors = [True for j in self.sat.attitude_sensors]
        if not self.has_sun and not self.sunsensors_during_eclipse:
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensor) for j in range(len(which_sensors))]
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensorPair) for j in range(len(which_sensors))]
        else: # has sun, ignore readings from sun sensors that are too little--could be albedo, noise, etc.
            which_sensors = [which_sensors[j] and not (isinstance(self.sat.attitude_sensors[j],SunSensor) and ((sensors_in[j] - self.sat.attitude_sensors[j].bias) < self.sat.attitude_sensors[j].std))  for j in range(len(which_sensors))]
        # # print(which_sensors)

        # scalar_update = False
        truth_cov = np.eye(self.sat.state_len)*1e-20
        #remove values not in use
        pv = np.concatenate([truth,self.full_state.val[self.sat.state_len:]])
        pc = scipy.linalg.block_diag(truth_cov, self.full_state.cov[self.sat.state_len-1:,self.sat.state_len-1:])

        vecs = os_local_vecs(os,pv[3:7])
        for j in range(len(self.sat.sensors)): #ignore sun sensors that should be in shadow
            if isinstance(self.sat.sensors[j],SunSensor):
                which_sensors[j] &= not (self.sat.sensors[j].clean_reading(pv,vecs)<1e-10)
        print(which_sensors)

        state_mod = np.eye(7)
        # if not self.quat_as_vec:
        #     state_mod = block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(pv[3:7],self.vec_mode),self.vec_mode))

        # if scalar_update:
        #     state2,cov2 = self.scalar_update(pc,pv,os)
        # else:
            #sensor jacobian
        state_jac,bias_jac = self.sat.sensor_state_jacobian(pv,vecs,which = which_sensors,keep_unused_biases=True)
        lenA = np.size(state_jac,1)
        Hk = np.vstack([state_mod@state_jac,np.zeros((self.sat.act_bias_len,lenA)),bias_jac,np.zeros((self.sat.dist_param_len-(len(self.use)-sum(self.use)),lenA))])
        sens_cov = self.sat.sensor_cov(which_sensors,keep_unused_biases=False)
        lilhk = self.sat.sensor_values(pv,vecs,which = which_sensors)
        zk = sensors_in[which_sensors]

        lilhk = np.concatenate([lilhk,truth])
        addmat = np.eye(len(pv),self.sat.state_len)#np.block([[np.eye(self.sat.state_len)],[np.zeros(len(pv)-self.sat.state_len,self.sat.state_len)]])
        Hk = np.block([[Hk,addmat]])
        sens_cov = scipy.linalg.block_diag(sens_cov,truth_cov)
        zk = np.concatenate([zk,truth])


        try:
            # breakpoint()
            Kk = scipy.linalg.solve((sens_cov+Hk.T@pc@Hk),(pc@Hk).T,assume_a='pos')
        except:
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')

        dstate = (zk-lilhk)@Kk
        cov2 = pc@(np.eye(len(pv))-Hk@Kk)
        cov2 = 0.5*(cov2 + cov2.T)

        # if self.quat_as_vec:
        state2 = pv + dstate
        state20 = np.copy(state2)
        state2[3:7] = normalize(state2[3:7])
        norm_jac = state_norm_jac(state20)
        cov2 = norm_jac.T@cov2@norm_jac
        cov2 = np.delete(cov2,3,0)
        cov2 = np.delete(cov2,3,1)
        # else:
        #     state2 = pv.copy()
        #     state2[0:3] += dstate[0:3]
        #     state2[7:] += dstate[6:]
        #     state2[3:7] = quat_mult(state2[3:7],vec3_to_quat(dstate[3:6],self.vec_mode))

        out = estimated_nparray(state2,cov2)



        self.prev_os = os

        oc = out.cov
        # breakpoint()
        if not self.use_cross_term:
            # p = self.sat.state_len+self.sat.att_sens_bias_len+self.sat.act_bias_len - 1 + self.quat_as_vec
            # oc[0:p,p:] = 0
            # oc[p:,0:p] = 0
            ab0 = self.sat.state_len - 1 + self.quat_as_vec
            ab1 = self.sat.state_len - 1 + self.quat_as_vec + self.sat.act_bias_len
            sb0,sb1 = ab0 + self.sat.att_sens_bias_len,ab1 + self.sat.att_sens_bias_len
            d0 = sb1
            # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 0))
            # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 0))
            # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 1))
            # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 1))
            # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 1))
            # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 0))
            oc[ab0:ab1,sb0:sb1] = 0
            oc[sb0:sb1,ab0:ab1] = 0
            oc[ab0:ab1,d0:] = 0
            oc[d0:,ab0:ab1] = 0
            oc[sb0:sb1,d0:] = 0
            oc[d0:,sb0:sb1] = 0
        # breakpoint()
        self.full_state.set_indices(self.use,out.val,oc,square_mat_sections(self.full_state.int_cov,self.cov_use()),[3])
        self.use_state = self.full_state.pull_indices(self.use,[3])
        self.sat.match_estimate(self.full_state,self.update_period)
        self.os_vecs = os_local_vecs(os,self.full_state.val[3:7])

        return self.full_state.val[0:self.sat.state_len],extra

        if self.has_sun and os.in_eclipse():
            self.byebye_sun()
        if not self.has_sun and not os.in_eclipse():
            self.hello_sun()
        if self.prev_os.R.all() == 0:
            self.prev_os = os
            self.prev_os_vecs = os_local_vecs(os,self.full_state.val[3:7])
        else:
            self.prev_os_vecs = self.os_vecs.copy()
        if which_sensors is None:
            which_sensors = [True for j in self.sat.attitude_sensors]
        if not self.has_sun and not self.sunsensors_during_eclipse:
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensor) for j in range(len(which_sensors))]
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensorPair) for j in range(len(which_sensors))]
        else: # has sun, ignore readings from sun sensors that are too little--could be albedo, noise, etc.
            which_sensors = [which_sensors[j] and not (isinstance(self.sat.attitude_sensors[j],SunSensor) and ((sensors_in[j] - self.sat.attitude_sensors[j].bias) < self.sat.attitude_sensors[j].std))  for j in range(len(which_sensors))]
        # # print(which_sensors)



        if self.prev_os.R.all() == 0:
            self.prev_os = os
        truth_cov = np.eye(self.sat.state_len)*1e-25
        # breakpoint()

        x1 = np.concatenate([truth,self.full_state.val.copy()[self.sat.state_len:]])
        cov1 = self.full_state.cov.copy()+self.full_state.int_cov.copy()

        state_mod = np.eye(7)
        # if not self.quat_as_vec:
        #     state_mod = block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(x1[3:7],self.vec_mode),self.vec_mode))
        # print(len(self.sat.sensors),len(which_sensors))
        numGPS = sum([isinstance(j,GPS) for j in self.sat.sensors])
        # tmp = [self.sat.sensors[j].no_noise_reading(x1[0:self.sat.state_len],os_local_vecs(os,truth[3:7])) for j in range(len(self.sat.sensors)-numGPS) if which_sensors[j] and not isinstance(self.sat.sensors[j],GPS)]
        # print(sensors_in.shape,sensors_in)
        # print(tmp)
        # lilhk = np.concatenate([np.array(sensors_in)]+[self.sat.sensors[j].no_noise_reading(x1[0:self.sat.state_len],os_local_vecs(os,truth[3:7])) for j in range(len(self.sat.sensors)-numGPS) if which_sensors[j] and not isinstance(self.sat.sensors[j],GPS)])
        vecs = os_local_vecs(os,x1[3:7])

        lilhk = self.sat.sensor_values(x1,vecs,which = which_sensors)
        sens_cov = self.sat.sensor_cov(which_sensors)

        state_jac,bias_jac = self.sat.sensor_state_jacobian(x1,vecs,which = which_sensors)
        lenA = np.size(state_jac,1)
        Hk = np.vstack([state_mod@state_jac,np.zeros((self.sat.act_bias_len,lenA)),bias_jac,np.zeros((len(self.use)-sum(self.use),lenA))])
        zk = sensors_in

        if truth is not None and truth_cov is not None:
            lilhk = np.concatenate([lilhk,x1[0:self.sat.state_len]])
            breakpoint()
            Hk = np.block([[Hk,np.block([[np.eye(self.sat.state_len )],[np.zeros((len(x1)-self.sat.state_len ,self.sat.state_len ))]])]])
            sens_cov = scipy.linalg.block_diag(sens_cov,truth_cov)
            zk = np.concatenate([zk,truth])

        try:
            # print(np.linalg.cond(sens_cov_cut),np.linalg.cond(cov1_cut),np.linalg.cond(Hk_cut@cov1_cut@Hk_cut.T),np.linalg.cond((sens_cov_cut+Hk_cut@cov1_cut@Hk_cut.T)))
            Kk = scipy.linalg.solve((sens_cov+Hk.T@cov1@Hk),cov1@Hk,assume_a='pos')
            #Kk = (cov1@Hk1.T)@np.linalg.inv(sens_cov1+Hk1@cov1@Hk1.T)
        except:
            raise np.linalg.LinAlgError('Matrix is singular. (probably)')


        dstate = (sensors_in-lilhk)@Kk
        cov2 = cov1@(np.eye(len(pv) - 1 + self.quat_as_vec)-Hk@Kk)
        cov2 = 0.5*(cov2 + cov2.T)

        if self.quat_as_vec:
            state2 = x1 + dstate
            state20 = np.copy(state2)
            state2[3:7] = normalize(state2[3:7])
            norm_jac = state_norm_jac(state20)
            cov2 = norm_jac.T@cov2@norm_jac
        else:
            state2 = x1.copy()
            state2[0:3] += dstate[0:3]
            state2[7:] += dstate[6:]
            state2[3:7] = quat_mult(state2[3:7],vec3_to_quat(dstate[3:6],self.vec_mode))

        out = estimated_nparray(state2,cov2)
        self.prev_os = os

        oc = out.cov
        if not self.use_cross_term:
            # p = self.sat.state_len+self.sat.att_sens_bias_len+self.sat.act_bias_len - 1 + self.quat_as_vec
            # oc[0:p,p:] = 0
            # oc[p:,0:p] = 0
            ab0 = self.sat.state_len - 1 + self.quat_as_vec
            ab1 = self.sat.state_len - 1 + self.quat_as_vec + self.sat.act_bias_len
            sb0,sb1 = ab0 + self.sat.att_sens_bias_len,ab1 + self.sat.att_sens_bias_len
            d0 = sb1
            # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 0))
            # oc[sb0:sb1,sb0:sb1] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 0))
            # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[ab0:ab1,sb0:sb1],axis = 1))
            # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,sb0:sb1],axis = 1))
            # oc[d0:,d0:] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 1))
            # oc[ab0:ab1,ab0:ab1] += np.diagflat(np.sum(oc[d0:,ab0:ab1],axis = 0))
            oc[ab0:ab1,sb0:sb1] = 0
            oc[sb0:sb1,ab0:ab1] = 0
            oc[ab0:ab1,d0:] = 0
            oc[d0:,ab0:ab1] = 0
            oc[sb0:sb1,d0:] = 0
            oc[d0:,sb0:sb1] = 0
        self.full_state.set_indices(self.use,out.val,oc,square_mat_sections(self.full_state.int_cov,self.cov_use()),[3]*(not self.quat_as_vec))
        self.use_state = self.full_state.pull_indices(self.use,[3]*(not self.quat_as_vec))
        self.sat.match_estimate(self.full_state,self.update_period)
        self.os_vecs = os_local_vecs(os,self.full_state.val[3:7])
        return self.full_state.val[0:self.sat.state_len]

class GaussianExtraEstimator(PerfectEstimator):
    def set_cov(self,cov):
        self.cov = cov

    def update(self, sensors_in, control_vec, os,which_sensors = None,prop_on = False,truth = None):
        err_state = np.random.multivariate_normal(truth.flatten(),self.cov).reshape((self.sat.state_len,1))
        err_state[3:7,:] = normalize(err_state[3:7,:])
        super().update(sensors_in, control_vec, os,[False for j in range(len(self.sat.sensors))],prop_on,err_state,use_truth = True,truth_cov = self.cov)
        self.state_full.val[0:self.sat.state_len,:] = np.copy(err_state)
        self.state_full.val[3:7,:] = normalize(self.state_full.val[3:7,:])
        return self.state_full.val[0:self.sat.state_len,:].reshape((self.sat.state_len,1))

class GaussianEstimator(Estimator):
    def __init__(self,cov,sat=None,sample_time = 1,estimate=None):
        self.reset(cov,sat,sample_time,estimate)

    def reset(self,cov,sat = None,sample_time=1,estimate=None):
        if sat is None:
            sat = Satellite()
            for j in range(len(sat.sensors)):
                if sat.sensors[j].has_bias:
                    jbs = sat.sensors[j].bias.size
                    if not isinstance(sat.sensors[j].bias,estimated_nparray):
                        sat.sensors[j].bias = estimated_nparray(sat.sensors[j].bias,np.diagflat(sat.sensors[j].bias_std_rate*np.ones((jbs,1))),np.diagflat(sat.sensors[j].bias_std_rate**2*np.ones((jbs,1))))

            for j in range(len(sat.act_noise)):
                if sat.act_noise[j].has_bias:
                    sat.act_noise[j].bias = estimated_float(sat.act_noise[j].bias,sat.act_noise[j].bias_std_rate,sat.act_noise[j].bias_std_rate**2)
        else:
            for j in range(len(sat.sensors)):
                if sat.sensors[j].has_bias:
                    jbs = sat.sensors[j].bias.size
                    if not isinstance(sat.sensors[j].bias,estimated_nparray):
                        sat.sensors[j].bias = estimated_nparray(sat.sensors[j].bias,np.diagflat(sat.sensors[j].bias_std_rate*np.ones((jbs,1))),np.diagflat(sat.sensors[j].bias_std_rate**2*np.ones((jbs,1))))
            for j in range(len(sat.act_noise)):
                if sat.act_noise[j].has_bias:
                    jbs = sat.act_noise[j].bias.size
                    if not isinstance(sat.act_noise[j].bias,estimated_float):
                        sat.act_noise[j].bias = estimated_float(sat.act_noise[j].bias.val,sat.act_noise[j].bias_std_rate,sat.act_noise[j].bias_std_rate**2)

        if estimate is None:
            estimate = np.zeros((sat.state_len,1))
            estimate[3:7,:] = np.array([[0.5,0.5,0.5,0.5]]).T
        self.state_full = estimated_nparray(estimate,cov,0*cov)
        self.update_period = sample_time
        self.include_gendist = False
        self.estimate_dist = False

        self.sat = sat
        self.cov = cov

        # extras_len = self.dist_inds()[-1] - self.sensor_bias_inds()[-1][1]
        # self.cross_term_main_extras = np.zeros((self.sat.state_len+self.sensor_bias_inds()[-1][1],extras_len))
        # self.extras_cov = np.zeros((extras_len,extras_len))
        self.prev_os = Orbital_State(0,np.array([[0,0,1]]).T,np.array([[0,0,0]]).T)

    def update(self, sensors_in, control_vec, os,which_sensors = None,prop_on = False,truth = None):
        # super().update(sensors_in, control_vec, os,which_sensors,prop_on,truth,use_truth = True)
        err_state = np.random.multivariate_normal(truth.flatten(),self.cov).reshape((self.sat.state_len,1))
        err_state[3:7,:] = normalize(err_state[3:7,:])
        self.state_full.val = err_state# = estimated_nparray(err_state,self.cov,0*self.cov)
        return self.state_full.val[0:self.sat.state_len,:].reshape((self.sat.state_len,1))
