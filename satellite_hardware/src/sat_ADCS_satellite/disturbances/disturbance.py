import numpy as np
import scipy
import random
from sat_ADCS_helpers import *
# from flight_adcs.flight_utils.sat_helpers import *
# import sim_adcs.sim_helpers as sim_helpers#TODO: eliminate the need for this?
import math
import pytest


class Disturbance:
    """
    This class describes a disturbance torque.

    Attributes
    ------------
        params -- vary with type
    """
    def __init__(self,params,time_varying = False,std = np.nan,estimate=False,active=True):
        self.time_varying = time_varying
        self.std = std*self.time_varying
        self.main_param = np.nan*np.zeros(0)
        self.set_params(params)
        self.active = active
        self.estimated_param = estimate
        if self.time_varying and np.any(np.isnan(self.std)):
            raise ValueError('standard deviation of a time-varying disturbance should not be nan')
        if self.time_varying and (np.any(np.isnan(self.main_param)) or np.ndim(self.main_param)==0 or np.size(self.main_param)<1):
            raise ValueError('the "main param" (estimatable value) of a timevarying / estimated disturbance should not be nan and should have more than 0 dimensions')

    def turn_on(self):
        self.active = True

    def turn_off(self):
        self.active = False

    def set_params(self,params):
        self.params = params

    def update(self,j2000):
        pass

    def torque(self,sat,vecs):
        return np.zeros((3,1))

    def torque_qjac(self,sat,vecs): #torque jacobian over quaternion
        return np.zeros((4,3))
        #TODO: currently assuming disturbance torques are only modified by q. probably none affected by RWw, actuation, torq, or w, but should consider how they change with RW ang or unbalance

    def torque_qqhess(self,sat,vecs): #hessian of torque elements over quaternion
        return np.zeros((4,4,3))

    def torque_valjac(self,sat,vecs):
        return np.zeros((self.main_param.size,3))

    def torque_valvalhess(self,sat,vecs):
        return np.zeros((self.main_param.size,self.main_param.size,3))

    def torque_qvalhess(self,sat,vecs):
        return np.zeros((4,self.main_param.size,3))
