import numpy as np
import scipy
from scipy.linalg import block_diag
import scipy.linalg
import random
import math
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
import pytest
from sat_ADCS_satellite import *
import warnings

"""
to update from control folder

cd ../estimation && \
python3.10 -m build && \
pip3.10 install ./dist/sat_ADCS_estimation-0.0.1.tar.gz && \
cd ../control
"""


class Variable_Dist_Desc():
    def __init__(self,dist_index,active,estimate_indices,value,j2000_off):
        self.dist_index = dist_index
        self.active = active
        self.estimate_indices = estimate_indices
        self.value = value
        self.j2000_off = j2000_off



class Estimator():
    #TODO: test. Much test.
    #TODO: update parts that call EKF to give correct arguments.--time call in simulation file
    #TODO: better bias1 growth model?
    ##TODO setup to take cov,int_cov,etc from input satellite.

    sat = None

    def __init__(self, j2000,sat,estimate,cov_estimate,integration_cov,sample_time = 1,use_cross_term = False,quat_as_vec = False,vec_mode = 5,sunsensors_during_eclipse = False,verbose = False):#disturbances are turned on by being in the satellite object. actuator and sensor biases, as well as dipole,general, and prop disturbance, are estimated if the satellite values are an estimated object. Otherwise they are just used.

        self.quat_as_vec = quat_as_vec
        if not sat.estimated:
            raise ValueError('sat in estimator should be estimated')
        self.sat = sat
        self.use_cross_term = use_cross_term
        self.prop_list = [j for j in range(len(self.sat.disturbances)) if isinstance(self.sat.disturbances[j],Prop_Disturbance)]
        self.len_before_sens_bias = self.sat.state_len + self.sat.act_bias_len
        self.att_sens_bias_valinds = [self.len_before_sens_bias + np.arange(0,self.sat.attitude_sensors[j].output_length) + sum([self.sat.attitude_sensors[i].output_length for i in self.sat.att_sens_bias_inds if i<j ]) for j in self.sat.att_sens_bias_inds  ]
        self.att_sens_input_inds = [np.arange(0,self.sat.attitude_sensors[j].output_length) + sum([self.sat.attitude_sensors[i].output_length for i in range(len(self.sat.attitude_sensors)) if i<j ]) for j in range(len(self.sat.attitude_sensors))  ]

        # self.att_sens_info = {j:(j,self,self.sat.attitude_sensors[j].output_length,self.sat.attitude_sensors[j].has_bias,)}

        self.reset( j2000,estimate = estimate,cov_estimate=cov_estimate,integration_cov = integration_cov,sample_time = sample_time,use_cross_term =  use_cross_term)

        self.variable_dist_info = {self.sat.dist_param_inds[j]:Variable_Dist_Desc(self.sat.dist_param_inds[j],self.sat.disturbances[self.sat.dist_param_inds[j]].active,np.arange(0,self.sat.disturbances[self.sat.dist_param_inds[j]].main_param.size) + self.len_before_sens_bias + self.sat.att_sens_bias_len+sum([self.sat.disturbances[i].main_param.size for i in self.sat.dist_param_inds[:j] ]),[],np.nan) for j in range(len(self.sat.dist_param_inds))}

        # print([type(j.bias) for j in self.sat.sensors])
        self.prev_os = Orbital_State(0,np.array([1,0,0]),np.array([0,1,0]))
        self.has_sun = True
        self.vec_mode = vec_mode #governs quaternion conversions to length-3 parameterizaitons
        self.sunsensors_during_eclipse = sunsensors_during_eclipse
        self.verbose = verbose


    def reset(self,j2000, estimate,cov_estimate,integration_cov,sample_time = 1,use_cross_term=False):
        #estimate_dist = False,estimate_act_bias = None, estimate_sensor_bias = None, use_cross_term = False):
        """
        Reset the estimator. Also called by __init__
        Inputs:
            init_settings -- tuple with settings in this order: (initial_state_estimate, output_size_ to be set
        Returns:
            nothing
        """
        if len(estimate) != 7+self.sat.number_RW + self.sat.act_bias_len + self.sat.att_sens_bias_len + self.sat.dist_param_len:
            print(len(estimate),7+self.sat.number_RW , self.sat.act_bias_len , self.sat.att_sens_bias_len , self.sat.dist_param_len)
            raise ValueError('estimate length does not match estimates in satellite')

        if not (( self.quat_as_vec and cov_estimate.shape == (len(estimate),len(estimate))) or (not self.quat_as_vec and np.all(cov_estimate.shape == (len(estimate)-1,len(estimate)-1)))):
            print(cov_estimate.shape,len(estimate))
            print( cov_estimate.shape == (len(estimate)-1,len(estimate)-1))
            raise ValueError('cov estimate wrong shape')

        if not (( self.quat_as_vec and integration_cov.shape == (len(estimate),len(estimate))) or  (not self.quat_as_vec and np.all(integration_cov.shape == (len(estimate)-1,len(estimate)-1)))):
            print(integration_cov.shape,len(estimate))
            print( integration_cov.shape == (len(estimate)-1,len(estimate)-1))
            raise ValueError('integration cov wrong shape')

        # sat.match_estimate(estimate)

        self.update_period = sample_time
        self.original_state = estimated_nparray(estimate,cov_estimate,integration_cov)
        self.full_state = self.original_state.copy()
        self.use_state = self.original_state.copy()
        self.sat.match_estimate(self.full_state,self.update_period)
        self.use = np.ones(self.full_state.val.size).astype(bool)
        self.state_len = len(self.use)
        for j in range(len(self.sat.disturbances)):
            if not self.sat.disturbances[j].active:
                self.specific_dist_off(j2000,j,save_info=True,save_from_sat = False)
        # self.os_vecs = None
        # self.prev_os_vecs = None

    def initialize_estimate(self,sensors_in,vec_inds,ECIvecs, gyro_inds,orb,q= None):
        #TODO add vector vhecks to make sure there are at least 2, they all have len 3, etc.
        #make sure to
        bodyvecs = [sensors_in[j] for j in vec_inds]
        gyro_in = sensors_in[gyro_inds]

        if q is None:
            weights = [1/np.array([self.sat.attitude_sensors[j].std**2.0*self.sat.attitude_sensors[j].scale**2.0 + self.full_state.cov[j+self.len_before_sens_bias - 1 + self.quat_as_vec,j+self.len_before_sens_bias - 1 + self.quat_as_vec] for j in k]) for k in vec_inds]
            if not all([np.all(j==j[0]*np.ones(j.size)) for j in weights]):
                raise ValueError("not prepared for inconsistent weights")
            qopt = wahbas_svd(weights,bodyvecs,ECIvecs)
        else:
            qopt = q
        state = self.full_state.val.copy()
        state[3:7] = qopt
        # self.full_state = estimated_nparray(state,self.full_state.cov,self.full_state.int_cov)
        covvecinds = np.concatenate(vec_inds)+self.len_before_sens_bias - 1 + self.quat_as_vec
        covgyroinds = gyro_inds+self.len_before_sens_bias - 1 + self.quat_as_vec
        Qnongyro = np.diag(np.diag(self.full_state.cov)[covvecinds])
        Qgyro = np.diag(np.diag(self.full_state.cov)[covgyroinds])
        Rnongyro = np.diag(np.diag(self.sat.sensor_cov())[np.concatenate(vec_inds)])
        Rgyro = np.diag(np.diag(self.sat.sensor_cov())[gyro_inds])
        snongyro = sensors_in[np.concatenate(vec_inds)]
        sgyro = sensors_in[gyro_inds]
        Qw = self.full_state.cov[0:3,0:3]

        Dvec = self.sat.noiseless_sensor_values(state,os_local_vecs(orb,qopt))[np.concatenate(vec_inds)] - state[covvecinds+1]
        b_nongyro_guess = np.linalg.inv(Rnongyro+Qnongyro)@Qnongyro@(snongyro-Dvec)
        state[covvecinds+1-self.quat_as_vec] = b_nongyro_guess

        b_gyro_guess = np.linalg.inv(Rgyro + Qgyro + Qw)@Qgyro@sgyro
        state[covgyroinds+1-self.quat_as_vec] = b_gyro_guess

        wguess = Qw@np.linalg.inv(Qgyro)@b_gyro_guess

        state[0:3] = wguess
        if q is None:
            self.full_state = estimated_nparray(state,self.full_state.cov,self.full_state.int_cov)
            self.use_state = self.full_state.pull_indices(self.use,[3]*(not self.quat_as_vec))
        return state

    def sat_match(self,sat,state):
        full_statej = self.full_state#.copy()
        full_statej.val[self.use] = state
        sat.match_estimate(full_statej,self.update_period)

    def update(self,control_vec,sensors_in,os,which_sensors = None,truth = None,use_truth = False, truth_cov = None):

        # print('up0',self.use_state.val)
        # print('inside0', self.prev_os.J2000,os.J2000)
        # print(self.prev_os.R,os.R)
        if use_truth:
            if truth_cov is None or np.any(np.isnan(truth_cov)) or np.all(truth_cov==0):
                return truth
            else:
                return np.random.multivariate_normal(truth,truth_cov)
        if self.has_sun and os.in_eclipse():
            self.byebye_sun()
        if not self.has_sun and not os.in_eclipse():
            self.hello_sun()
        if self.prev_os.J2000 == 0:
            self.prev_os = os
            # self.prev_os_vecs = os_local_vecs(os,self.full_state.val[3:7])
        else:
            # self.prev_os_vecs = self.os_vecs.copy()
            pass
        if which_sensors is None:
            which_sensors = [True for j in self.sat.attitude_sensors]
        if not self.has_sun and not self.sunsensors_during_eclipse:
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensor) for j in range(len(which_sensors))]
            which_sensors = [which_sensors[j] and not isinstance(self.sat.attitude_sensors[j],SunSensorPair) for j in range(len(which_sensors))]
        else: # has sun, ignore readings from sun sensors that are too little--could be albedo, noise, etc.
            which_sensors = [which_sensors[j] and not (isinstance(self.sat.attitude_sensors[j],SunSensor) and ((sensors_in[j] - self.sat.attitude_sensors[j].bias) < self.sat.attitude_sensors[j].std))  for j in range(len(which_sensors))]
        # # print(which_sensors)
        # print('moving to core')
        # print('up1',self.use_state.val)
        # print('inside1', self.prev_os.J2000,os.J2000)
        # print(self.prev_os.R,os.R)
        out,extra = self.update_core(control_vec,sensors_in,os,which_sensors)
        # print('done with core')
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
        # self.os_vecs = os_local_vecs(os,self.full_state.val[3:7])
        # print('up2',self.use_state.val)
        # print('inside2', self.prev_os.J2000,os.J2000)
        # print(self.prev_os.R,os.R)

        return self.full_state.val[0:self.sat.state_len],extra

    def cov_use(self):
        if self.quat_as_vec:
            return self.use
        else:
            res = self.use.copy()
            res = np.delete(res,3)
            return res

    def add_to_state(self,state,add):
        add = np.squeeze(add)
        state = np.squeeze(state)
        if add.ndim == 1:
            if self.quat_as_vec:
                result = state+add
                result[3:7] = normalize(result[3:7])
            else:
                result = state.copy()
                result[0:3] = state[0:3] + add[0:3]
                result[7:] = state[7:] + add[6:]
                result[3:7] = quat_mult(state[3:7],vec3_to_quat(add[3:6],self.vec_mode))
        else:
            if self.quat_as_vec:
                result = state+add
                result[3:7] = matrix_row_normalize(result[3:7])
            else:
                result = np.zeros((np.size(add,0),np.size(state,0)))
                result[:,0:3] = state[0:3] + add[:,0:3]
                result[:,7:] = state[7:] + add[:,6:]
                result[:,3:7] = np.vstack([quat_mult(state[3:7],vec3_to_quat(add[j,3:6],self.vec_mode)) for j in range(np.size(add,0))])
        return result


    def subtract_state_off(self,state,add):
        add = np.squeeze(add)
        state = np.squeeze(state)
        if add.ndim == 1:
            if self.quat_as_vec:
                result = add - state
                result[3:7] = normalize(result[3:7])
            else:
                result = state.copy()
                result[0:3] = add[0:3] - state[0:3]
                result[7:] =  add[6:] - state[7:]
                result[3:7] = quat_mult(quat_inv(state[3:7]),vec3_to_quat(add[3:6],self.vec_mode))
        else:
            if self.quat_as_vec:
                result = add - state
                result[:,3:7] = matrix_row_normalize(result[:,3:7])
            else:
                result = np.zeros((np.size(add,0),np.size(state,0)))
                result[:,0:3] = add[:,0:3] - state[0:3]
                result[:,7:] = add[:,6:] - state[7:]
                result[:,3:7] = np.vstack([quat_mult(quat_inv(state[3:7]),vec3_to_quat(add[j,3:6],self.vec_mode)) for j in range(np.size(add,0))])
        return result



    def states_diff(self,state1,state0):
        state1 = np.squeeze(state1)
        state0 = np.squeeze(state0)
        if state1.ndim == 1:
            if self.quat_as_vec:
                result = state1 - state0
                result[3:7] = normalize(result[3:7])
            else:
                result = np.zeros(np.size(state1,0)-1)
                result[0:3] = state1[0:3] - state0[0:3]
                result[6:] =  state1[7:] - state0[7:]
                result[3:6] = quat_to_vec3(quat_mult(quat_inv(state0[3:7]),state1[3:7]),self.vec_mode)
        else:
            if self.quat_as_vec:
                result = state1 - state0
                result[:,3:7] = matrix_row_normalize(result[:,3:7])
            else:
                result = np.zeros((np.size(state1,0),np.size(state1,1)-1))
                result[:,0:3] = state1[:,0:3] - state0[0:3]
                result[:,6:] = state1[:,7:] - state0[7:]
                result[:,3:6] = np.vstack([quat_to_vec3(quat_mult(quat_inv(state0[3:7]),state1[j,3:7]),self.vec_mode) for j in range(np.size(state1,0))])
        return result


    # def fix_post_cov(self,):



    def byebye_sun(self):
        self.sat.srp_dist_off()
        self.has_sun = False

    def hello_sun(self):
        self.sat.srp_dist_on()
        self.has_sun = True

    def prop_on(self,j2000,ind = None):
        if len(self.prop_list)==0:
            warnings.warn('there are no prop distubances to turn off')
            return
        elif ind is None:
            if len(self.prop_list)>1:
                raise ValueError('multiple prop disturbances and no index specified')
            else:
                ind = self.prop_list[0]
        if ind == 'all':
            [self.prop_on(j2000,j) for j in self.prop_list]
            return
        else:
            self.sat.prop_dist_on(ind) #should error if this is not prop
            self.specific_dist_on(j2000,ind,reset_to='saved_w_int_cov_added')

    def prop_off(self,j2000,ind = None):
        if len(self.prop_list)==0:
            warnings.warn('there are no prop distubances to turn off')
            return
        elif ind is None:
            if len(self.prop_list)>1:
                raise ValueError('multiple prop disturbances and no index specified')
            else:
                ind = self.prop_list[0]
        if ind == 'all':
            [self.prop_off(j2000,j) for j in self.prop_list]
            return
        else:
            self.sat.prop_dist_off(ind) #should error if this is not prop
            self.specific_dist_off(j2000,ind,save_info = True)

    def specific_dist_on(self,j2000,ind,reset_to = 'dist_object'):
        self.sat.specific_dist_on(ind) #turn on in satellite
        if ind in self.sat.dist_param_inds:

            self.variable_dist_info[ind].active = True #indicate active
            timeoff = self.variable_dist_info[ind].j2000_off
            self.variable_dist_info[ind].j2000_off = np.nan
            valinds = self.variable_dist_info[ind].estimate_indices #find indices in the full state associated
            self.use[valinds] = True #turn on in use vector

            if reset_to == 'dist_object':
                v =  self.sat.disturbances[ind].val
                c = square_mat_section(self.full_state0.cov.copy(),valinds)
                ic = self.sat.disturbances[ind].std**2.0
            elif reset_to == 'saved':
                saved = self.variable_dist_info[ind].value
                [v,c,ic] =  saved
            elif reset_to == 'saved_w_int_cov_added':
                saved = self.variable_dist_info[ind].value
                [v,c,ic] =  saved
                if not np.isnan(timeoff):
                    c += ic*((j2000-timeoff)*cent2sec)
            elif reset_to == 'zero_val_saved_cov':
                saved = self.variable_dist_info[ind].value
                [v,c,ic] =  saved
                v *= 0
            elif reset_to == 'zero_val_saved_cov_w_int_cov_added':
                saved = self.variable_dist_info[ind].value
                [v,c,ic] =  saved
                v *= 0
                if not np.isnan(timeoff):
                    c += ic*((j2000-timeoff)*cent2sec)
            elif reset_to == 'initial':
                v = self.original_state.val[valinds]
                c = square_mat_section(self.original_state.cov,valinds).copy()
                ic = square_mat_section(self.original_state.int_cov,valinds).copy()
            else:
                raise ValueError('Need different saved option')
            self.change_vals_from_inds(valinds,new_val=v,newcov=c,new_int_cov = ic,clearx=True)

    def specific_dist_off(self,j2000,ind,save_info = False,save_from_sat = True):
        self.sat.specific_dist_off(ind) #turn off in satellite
        if ind in self.sat.dist_param_inds:
            # breakpoint()
            self.variable_dist_info[ind].active = False #indicate not active
            valinds = self.variable_dist_info[ind].estimate_indices #find indices in the full state associated
            self.use[valinds] = False #turn off in use vector
            if np.isnan(self.variable_dist_info[ind].j2000_off):
                self.variable_dist_info[ind].j2000_off = j2000

            if save_info:
                if save_from_sat:
                    v = self.sat.disturbances[ind].main_param
                    ic = self.sat.disturbances[ind].std**2.0
                else:
                    v = self.full_state.val[valinds].copy()
                    ic = square_mat_sections(self.full_state.int_cov.copy(),valinds)
                c = square_mat_sections(self.full_state.cov.copy(),valinds-1*(not self.quat_as_vec))
                self.variable_dist_info[ind].value = [v,c,ic]
            else:
                v = np.zeros(len(valinds))
                c = np.eye(len(valinds))

            #reset use state
            self.use_state = self.full_state.pull_indices(self.use,[3]*(not self.quat_as_vec))
            #set full state to nan so issues can appear, not int_cov as that's stored in teh satellite as std.
            self.change_vals_from_inds(valinds,new_val = v*np.nan,newcov = np.nan*c,new_int_cov = ic,clearx=True)

    def change_vals_from_inds(self,valinds,new_val=None,newcov=None,new_int_cov = None,clearx=True):
        l = np.size(valinds)
        if new_int_cov is None:
            new_int_cov = 0*np.eye(l)#self.full_state0.int_cov.copy()[valinds,valinds]
        if newcov is None:
            newcov = 0*np.eye(l)#self.full_state0.cov.copy()[valinds,valinds]
        if new_val is None:
            new_val = np.zeros(l)#self.full_state0.val.copy()[valinds]
        fs = self.full_state.copy()
        inds_mask = np.zeros(self.full_state.val.size).astype(bool)
        inds_mask[valinds] = True
        fs.set_indices(inds_mask,new_val,newcov,new_int_cov,[3]*(not self.quat_as_vec))
        # fs.val[valinds] = new_val
        # breakpoint()
        # fs.int_cov[min(valinds):max(valinds)+1,min(valinds):max(valinds)+1] = new_int_cov
        if clearx:
            if not self.quat_as_vec:
                cov_inds_mask = inds_mask.copy()
                cov_inds_mask = np.delete(cov_inds_mask,3)
            else:
                cov_inds_mask = inds_mask
            overlap = np.zeros((sum(cov_inds_mask),sum(cov_inds_mask)))
            ii = 0
            for j in range(len(cov_inds_mask)):
                if cov_inds_mask[j]:
                    overlap[ii,:] = fs.cov[j,cov_inds_mask].copy()
                    ii += 1
            fs.cov[cov_inds_mask,:] = 0
            fs.cov[:,cov_inds_mask] = 0
            ii = 0
            for j in range(len(cov_inds_mask)):
                if cov_inds_mask[j]:
                    fs.cov[j,cov_inds_mask] = overlap[ii,:].copy()
                    ii += 1
        # fs.cov[valinds,valinds] = newcov
        self.full_state = estimated_nparray(fs.val,fs.cov,fs.int_cov)
        self.use_state = self.full_state.pull_indices(self.use,[3]*(not self.quat_as_vec))
