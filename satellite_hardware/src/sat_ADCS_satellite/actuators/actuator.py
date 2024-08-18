import numpy as np
import scipy
import random
import pytest
from sat_ADCS_helpers import *
import warnings
# from flight_adcs.flight_utils.sat_helpers import *
# import sim_adcs.sim_helpers as sim_helpers#TODO: eliminate the need for this?
import math


class Actuator():
    """
    This class represents an actuator for a satellite. It allows a custom noise model and custom settings for that model. Currently, it is very bare-bones
    but flexible.

    Parameters
    ----------
    Attributes
    ---------
    """
    def __init__(self, axis,max1,J,has_momentum, max_h,momentum,momentum_sens_noise_std, has_bias, bias,bias_std_rate,use_noise,noise_model,noise_settings,input_len,estimate_bias):
        """
        Initialize the set of sensors.
        See class definition.
        """
        # self.actuator_index = np.nan
        # self.actuator_output_range = [np.nan,np.nan]
        self.use_noise = use_noise
        self.noise_settings = noise_settings
        self.has_momentum = has_momentum
        self.axis = normalize(axis)
        self.max = max1
        self.max_h = max_h
        self.momentum_sens_noise_std = momentum_sens_noise_std
        self.J = J #kg*m^2
        self.noise = 0
        self.last_bias_update = 0
        self.update_momentum(momentum) #Nms


        if noise_model is None:
            noise_model = lambda x, noise_settings:(x, noise_settings)
        self.noise_model = noise_model
        self.has_bias = has_bias
        self.bias = np.atleast_1d(np.array(bias))*self.has_bias
        self.bias_std_rate = bias_std_rate*self.has_bias
        self.estimated_bias = estimate_bias*self.has_bias
        self.input_len = input_len
        self.update_noise()

    def clean_torque(self, command,sat,state,vecs):
        """
        clean torque--no bias or noise
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (3), torque generated in body frame
        """
        return np.zeros(3)

    def torque(self, command,sat,state,vecs,update_noise = False,update_bias = False,j2000 = np.nan):
        """
        torque with bias and noise, if relevant
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (3), torque generated in body frame
        """
        if abs(command)>self.max:

            warnings.warn("requested command exceeds actuation limit")
        noisy_command = command + self.use_noise*self.noise
        if update_noise:
            self.update_noise()
        return self.no_noise_torque(noisy_command,sat,state,vecs,update_bias=update_bias,j2000=j2000)

    def no_noise_torque(self, command,sat,state,vecs,update_bias = False,j2000 = np.nan):
        """
        torque with bias, no noise
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (3), torque generated in body frame
        """
        if abs(command)>self.max:
            warnings.warn("requested command exceeds actuation limit")
        biased_command = command + self.has_bias*self.bias
        if update_bias:
            self.update_bias(j2000)
        return self.clean_torque(biased_command,sat,state,vecs)

    def clean_storage_torque(self, command,sat,state,vecs):
        """
        clean torque on momentum storage--no bias or noise
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (1), torque on own momentum storage
        """
        if self.has_momentum:
            return np.zeros((1,))
        else:
            return np.zeros((0,))

    def no_noise_storage_torque(self, command,sat,state,vecs,update_bias = False, j2000 = np.nan):
        """
        baised torque on momentum storage--no noise
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (1), torque on own momentum storage
        """
        biased_command = command + self.has_bias*self.bias
        if update_bias:
            self.update_bias(j2000)
        return self.clean_storage_torque(biased_command,sat,state,vecs)

    def storage_torque(self, command,sat,state,vecs,update_noise = False,update_bias = False, j2000 = np.nan):
        """
        biased and noised torque on momentum storage
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (1), torque on own momentum storage
        """
        if abs(command)>self.max:
            warnings.warn("requested command exceeds actuation limit")
        noisy_command = command + self.use_noise*self.noise
        if update_noise:
                self.update_noise()
        return self.no_noise_storage_torque(noisy_command,sat,state,vecs,update_bias=update_bias,j2000=j2000)

    def dtorq__du(self,command,sat,state,vecs):
        return np.zeros((self.input_len,3))

    def dtorq__dbias(self,command,sat,state,vecs):
        if self.has_bias:
            return self.dtorq__du(command,sat,state,vecs)
        else:
            return np.zeros((0,3))

    def dtorq__dbasestate(self,command,sat,state,vecs):
        return np.zeros((7,3))

    def dtorq__dh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((1,3))
        else:
            return np.zeros((0,3))

    def ddtorq__dudu(self,command,sat,state,vecs):
        return np.zeros((self.input_len,self.input_len,3))

    def ddtorq__dudbias(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddtorq__dudu(command,sat,state,vecs)
        else:
            return np.zeros((self.input_len,0,3))

    def ddtorq__dudbasestate(self,command,sat,state,vecs):
        return np.zeros((self.input_len,7,3))

    def ddtorq__dudh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((self.input_len,1,3))
        else:
            return np.zeros((self.input_len,0,3))

    def ddtorq__dbiasdbias(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddtorq__dudu(command,sat,state,vecs)
        else:
            return np.zeros((0,0,3))

    def ddtorq__dbiasdbasestate(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddtorq__dudbasestate(command,sat,state,vecs)
        else:
            return np.zeros((0,7,3))

    def ddtorq__dbiasdh(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddtorq__dudh(command,sat,state,vecs)
        else:
            if self.has_momentum:
                return np.zeros((0,1,3))
            else:
                return np.zeros((0,0,3))

    def ddtorq__dbasestatedbasestate(self,command,sat,state,vecs):
        return np.zeros((7,7,3))

    def ddtorq__dbasestatedh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((7,1,3))
        else:
            return np.zeros((7,0,3))

    def ddtorq__dhdh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((1,1,3))
        else:
            return np.zeros((0,0,3))

    def dstor_torq__du(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((self.input_len,1))
        else:
            return np.zeros((self.input_len,0))

    def dstor_torq__dbias(self,command,sat,state,vecs):
        if self.has_bias:
            return self.dstor_torq__du(command,sat,state,vecs)
        else:
            if self.has_momentum:
                return np.zeros((0,1))
            else:
                return np.zeros((0,0))

    def dstor_torq__dbasestate(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((7,1))
        else:
            return np.zeros((7,0))

    def dstor_torq__dh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((1,1))
        else:
            return np.zeros((0,0))

    def ddstor_torq__dudu(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((self.input_len,self.input_len,1))
        else:
            return np.zeros((self.input_len,self.input_len,0))

    def ddstor_torq__dudbias(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddstor_torq__dudu(command,sat,state,vecs)
        else:
            if self.has_momentum:
                return np.zeros((self.input_len,0,1))
            else:
                return np.zeros((self.input_len,0,0))

    def ddstor_torq__dudbasestate(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((self.input_len,7,1))
        else:
            return np.zeros((self.input_len,7,0))

    def ddstor_torq__dudh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((self.input_len,1,1))
        else:
            return np.zeros((self.input_len,0,0))

    def ddstor_torq__dbiasdbias(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddstor_torq__dudu(command,sat,state,vecs)
        else:
            if self.has_momentum:
                return np.zeros((0,0,1))
            else:
                return np.zeros((0,0,0))

    def ddstor_torq__dbiasdbasestate(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddstor_torq__dudbasestate(command,sat,state,vecs)
        else:
            if self.has_momentum:
                return np.zeros((0,7,1))
            else:
                return np.zeros((0,7,0))

    def ddstor_torq__dbiasdh(self,command,sat,state,vecs):
        if self.has_bias:
            return self.ddstor_torq__dudh(command,sat,state,vecs)
        else:
            if self.has_momentum:
                return np.zeros((0,1,1))
            else:
                return np.zeros((0,0,0))

    def ddstor_torq__dbasestatedbasestate(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((7,7,1))
        else:
            return np.zeros((7,7,0))

    def ddstor_torq__dbasestatedh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((7,1,1))
        else:
            return np.zeros((7,0,0))

    def ddstor_torq__dhdh(self,command,sat,state,vecs):
        if self.has_momentum:
            return np.zeros((1,1,1))
        else:
            return np.zeros((0,0,0))

    def update_bias(self,j2000):
        if self.has_bias:
            if self.last_bias_update > 0:
                if j2000>self.last_bias_update:
                    self.bias = np.random.normal(self.bias,self.bias_std_rate*math.sqrt((j2000-self.last_bias_update)*cent2sec))
                    self.last_bias_update = j2000
            else:
                self.bias = np.random.normal(self.bias,self.bias_std_rate)
                self.last_bias_update = j2000

    def set_noise(self,noise):
        self.noise = noise

    def set_bias(self,bias):
        self.bias = bias

    def update_noise(self):
        # print(biased_val)
        if self.use_noise:
            noise,new_settings = self.noise_model(self.noise_settings)
            # print(noisy_val)
            # if self.has_bias:
            #     self.bias = np.random.normal(self.bias, self.bias_std_rate)
            #Update noise settings if needed (e.g. if updating bias)
            self.noise_settings = new_settings
            self.noise = noise

    def body_momentum(self):
        if self.has_momentum:
            return self.axis*self.momentum
        else:
            return np.zeros(3)

    def update_momentum(self,h):
        if self.has_momentum:
            if h>self.max_h:
                warnings.warn("angular momentum exceeds saturation limit")
            self.momentum = h
        elif ~np.isnan(h):
            raise ValueError('This actuator does not have momentum')
        else:
            self.momentum = np.nan

    def measure_momentum(self):
        return np.random.normal(self.momentum, self.momentum_sens_noise_std)


    def measure_momentum_noiseless(self):
        return self.momentum

    def momentum_measurement_cov(self):
        if self.has_momentum:
            return self.momentum_sens_noise_std*self.momentum_sens_noise_std
        else:
            return np.zeros((0,0))

    def momentum_measurement_srcov(self):
        if self.has_momentum:
            return self.momentum_sens_noise_std
        else:
            return np.zeros((0,0))

    def control_cov(self): #returns this even if use_noise is false for things like Kalman Filtering -- where you might generate the torque without noise, but need to know the covariance
        return self.std*self.std

    def control_srcov(self): #returns this even if use_noise is false for things like Kalman Filtering -- where you might generate the torque without noise, but need to know the covariance
        return self.std

    # def noise_model(self,x,std):
    #    return (np.random.normal(x, std),std)
