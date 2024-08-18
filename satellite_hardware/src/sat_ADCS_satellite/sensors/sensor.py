import numpy as np
import scipy
import random
import pytest
from sat_ADCS_helpers import *
# from flight_adcs.flight_utils.sat_helpers import *
# import sim_adcs.sim_helpers as sim_helpers#TODO: eliminate the need for this?
import math

class Sensor():
    """
    This class represents a sensor for a satellite. It allows a custom noise model and custom settings for that model. Currently, it is very bare-bones
    but flexible.

    Parameters
    ----------
    init_settings: tuple
        tuple with settings in this order: (noise_model, noise_settings, output_length, sample_time) to be set. Settings are described in "Attributes" section.

    Attributes
    ----------
    output_length: int
        size of output (# of sensor outputs), default is 3
    noise_model : func(truth_val, noise_settings) -> noisy_val
        function that takes in truth value (output_length x 1 np array) and noise_settings (tuple), and outputs noisy sensor value (output_length x 1
        np array) and updated noise_settings (tuple). Default is just to return truth value and same noise_settings (no updates, no noise)
    noise_settings : tuple
        the tuple can contain literally anything, and represents inputs into the noise model. Highly dependent on how the noise model is defined. One common formulation
        would be (std_nose, bias, std_bias_drift)
    noisy_val : numpy array (output_length x 1)
        raw sensor output from most recent sensor update
    sample_time : float
        this represents the sensor sample time in the simulation
    """

    def __init__(self, output_length, sample_time,has_bias, bias,bias_std_rate,use_noise,noise_model,noise_update_model,noise_settings,estimate_bias,scale):
        """
        Initialize the set of sensors.
        See class definition.
        """
        self.output_length = output_length
        self.noise_settings = noise_settings
        if noise_model is None:
            noise_model = lambda x, noise_settings,state,vecs:(x)
        if noise_update_model is None:
            noise_update_model = lambda x, noise_settings,state,vecs,dt:(noise_settings)
        # self.sensor_index = np.nan
        # self.sensor_output_range = [np.nan,np.nan]

        self.noise_model = noise_model
        self.noise_update_model = noise_update_model
        self.sample_time = sample_time
        self.has_bias = has_bias
        self.bias = np.atleast_1d(np.array(bias))*self.has_bias*scale
        self.last_bias_update = 0
        self.last_noise_model_update = 0
        self.use_noise = use_noise
        self.bias_std_rate = bias_std_rate*self.has_bias*scale
        self.attitude_sensor = True
        self.estimated_bias = estimate_bias*self.has_bias*scale
        self.scale = scale

    def clean_reading(self,state,vecs):
        return np.zeros(self.output_length)*self.scale

    def no_noise_reading(self,state,vecs,update_bias = False,j2000 = np.nan):
        base_reading = self.clean_reading(state,vecs)
        biased_reading = base_reading + self.has_bias*self.bias
        if update_bias:
            self.update_bias(j2000)
        return biased_reading

    def reading(self,state,vecs, update_bias = False, j2000 = np.nan):
        biased_reading = self.no_noise_reading(state,vecs,update_bias,j2000)
        noisy_reading = self.add_noise(biased_reading,state,vecs)
        return noisy_reading

    def add_noise(self,biased_val,state,vecs):
        if self.use_noise:
            return self.noise_model(biased_val/self.scale, self.noise_settings,state,vecs)*self.scale
        else:
            return biased_val

    def set_bias(self,bias):
        self.bias = bias/self.scale

    def update_bias(self,j2000):
        if self.has_bias:
            if self.last_bias_update>0:
                if j2000>self.last_bias_update:
                    self.bias = np.random.normal(self.bias,self.bias_std_rate*math.sqrt((j2000-self.last_bias_update)*cent2sec))
                    self.last_bias_update = j2000
            else:
                self.bias = np.random.normal(self.bias,self.bias_std_rate)
                self.last_bias_update = j2000

    def update_noise(self,biased_val,state,vecs,j2000):
        if self.use_noise:
            if self.last_noise_model_update>0:
                if j2000>self.last_noise_model_update:
                    new_settings = self.noise_update_model(biased_val/self.scale,self.noise_settings,state,vecs,j2000-self.last_noise_model_update)
                    self.noise_settings = new_settings
                    self.last_noise_model_update = j2000
            else:
                new_settings = self.noise_update_model(biased_val/self.scale,self.noise_settings,state,vecs,np.nan)
                self.noise_settings = new_settings
                self.last_noise_model_update = j2000

    def basestate_jac(self,x,vecs):
        return np.zeros((7,self.output_length))

    def bias_jac(self,x,vecs):
        if self.has_bias:
            return np.eye(self.output_length)
        else:
            return np.zeros((0,self.output_length))

    def orbitRV_jac(self,x,vecs):
        return np.zeros((6,self.output_length))

    def cov(self): #returns this even if use_noise is false for things like Kalman Filtering -- where you might generate the torque without noise, but need to know the covariance
        # print(self,self.noise_settings)
        if isinstance(self.std,float):
            return np.eye(self.output_length)*self.std**2.0*self.scale*self.scale
        else:
            return np.diagflat(self.std*self.std)*self.scale*self.scale


    def srcov(self): #returns this even if use_noise is false for things like Kalman Filtering -- where you might generate the torque without noise, but need to know the covariance
        # print(self,self.noise_settings)
        if isinstance(self.std,float):
            return np.eye(self.output_length)*self.std*self.scale
        else:
            return np.diagflat(self.std)*self.scale
