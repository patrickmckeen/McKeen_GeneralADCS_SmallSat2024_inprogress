import numpy as np
import scipy
import scipy.linalg
import random
import math
from sat_ADCS_helpers import *
import pytest
from sat_ADCS_satellite import *
from sat_ADCS_orbit import *
import warnings

class OrbitalEstimator():

    def __init__(self, sat,sample_time = 1, state_estimate = None, cov_estimate = None, GPS_rnoise_var_guess=1,GPS_vnoise_var_guess=1,integration_cov = None,GPS_downrange_bias_magnitude = 0):
        """
        Initialize the estimator.
        Inputs:
            init_settings -- tuple with settings to be set
        Returns:
            nothing
        """
        self.reset(sat,sample_time, state_estimate, cov_estimate, GPS_rnoise_var_guess,GPS_vnoise_var_guess,integration_cov ,GPS_downrange_bias_magnitude)

    def update(self, GPS_in, j2000,truth = None,calc_B = True,calc_S = True,calc_all = True):

        estimated_GPS = [j for j in self.sat.sensors if j.has_bias and isinstance(j,GPS)]
        if self.current_state_estimate.J2000 == 0 or np.isnan(self.current_state_estimate.J2000):
            cov = self.current_cov
            self.current_state_estimate = Orbital_State(j2000,GPS_in[0:3],GPS_in[3:6],altrange = self.current_state_estimate.altrange, rhovsalt = self.current_state_estimate.rhovsalt,calc_B = False,calc_S = False,calc_all = False)#,gc = self.current_state_estimate.g_coeffs, hc = self.current_state_estimate.h_coeffs)

            R = self.current_state_estimate.ecef_to_eci(GPS_in[0:3])
            V = self.current_state_estimate.ecef_to_eci(GPS_in[0:3])
            self.current_state_estimate = Orbital_State(j2000,R,V,altrange = self.current_state_estimate.altrange, rhovsalt = self.current_state_estimate.rhovsalt,calc_B = calc_B,calc_S = calc_S,calc_all = calc_all)#,gc = self.current_state_estimate.g_coeffs, hc = self.current_state_estimate.h_coeffs)

            self.prev_os = self.current_state_estimate.copy()
            self.prev_os.J2000 = self.current_state_estimate.J2000-self.update_period*sec2cent
            # breakpoint()
            # self.prev_vec_os = self.current_state_vec_estimate
        elif truth is None:
            #TODO currently assumes all GPS_in (which could be stacks of 6-element info from multiple GPS) are used each time. Could change so some sensors used some of the time, like the attitude estimator
            # breakpoint()

            next_os = self.current_state_estimate.orbit_rk4(self.update_period,True,True,calc_all=False,calc_B=False,calc_S=False)

            # os = Orbital_State(j2000,np.ones(3),np.ones(3),calc_all=False,B=np.ones(3)*np.nan)
            # r_ECI_measured = next_os.ecef_to_eci(GPS_in[0:3])
            # v_ECI_measured = next_os.ecef_to_eci(GPS_in[3:6])
            # if np.all([self.state_estimate_vec() == 0]):
            #     est_vec = np.zeros(6)
            #     est_vec[0:3] = r_ECI_measured
            #     est_vec[3:6] = v_ECI_measured
            # else:
            #     est_vec = self.state_estimate_vec()
            #
            # r_ECI = est_vec[0:3]
            # v_ECI = est_vec[3:6]

            r,v = next_os.R,next_os.V
            dr__dr0,dr__dv0,dv__dr0,dv__dv0 = self.current_state_estimate.orbit_rk4_jacobians(self.update_period, True,True)
            Fk = np.block([[dr__dr0, dr__dv0], [dv__dr0, dv__dv0]])


            sens_bias_len = sum([j.output_length for j in estimated_GPS])
            full_len = self.current_state_vec_estimate.val.size

            x1_full = np.concatenate([r,v,self.current_state_vec_estimate.val[6:]])
            x0_full = self.current_state_vec_estimate.val

            # int_cov_full = self.current_state_vec_estimate.int_cov
            # cov0_full = self.current_state_vec_estimate.cov

            F_full = np.zeros((full_len,full_len))
            F_full[0:6,0:6] = Fk
            F_full[6:sens_bias_len,6:sens_bias_len] = np.eye(sens_bias_len)

            Hk = np.eye(6)#np.block([[np.eye(3), np.zeros([3,3])],[np.zeros([3,3]), np.eye(3)]])
            Hk_full = np.zeros((Hk.shape[0],full_len))
            Hk_full[:,0:6] = Hk

            i = 0
            for j in estimated_GPS:
                dd = j.output_length
                # print(j,dd,p)
                Hk_full[6+i:6+i+6,6+i:6+i+6] = np.eye(dd)
                i+=dd

            #Find x estimate and covariance estimate from predict step
            cc = self.current_cov
            cov1 = Fk@cc@Fk.T + self.current_state_vec_estimate.int_cov
            #Adjust for GPS downrange bias  TODO--why was this included? add back?
            # downrange_adjustment = normalize(v_ECI_measured)*self.GPS_downrange_bias_magnitude #km
            # r_ECI_measured = r_ECI_measured + downrange_adjustment

            zk = np.concatenate([ next_os.ecef_to_eci(GPS_in[i*3:i*3+3]) for i in range(math.floor(GPS_in.size/3))])

            #Form hk and Hk (dhk/dx) and find Kalman gain
            hk = Hk_full@x1_full

            Kk = scipy.linalg.solve((self.sensor_cov+Hk_full@cov1@Hk_full.T),Hk_full@cov1.T,assume_a='sym').T

            xk = x1_full + Kk@(zk-hk)

            self.innovation = zk-hk

            cov  = (np.eye(full_len)-Kk@Hk)@cov1
            cov = 0.5*(cov+cov.T)

            ii = 0
            # print([type(j.bias) for j in self.sat.sensors])

            for j in estimated_GPS:
                ind = j.sensor_index
                print('GPGPGPGPGPGPGPGPGP',j,j.sensor_index,len(estimated_GPS))
                self.sat.sensors[ind].bias = estimated_nparray(xk[6+ii:6+ii+6],covk_full[6+ii:6+ii+6,6+ii:6+ii+6],self.sat.sensors[ind].bias.int_cov)
                ii += 6

            #Set variables and return
            self.current_cov = cov[0:6,0:6]
            self.prev_os = self.current_state_estimate
            self.prev_vec_os = self.current_state_vec_estimate
            self.current_state_estimate = Orbital_State(j2000,xk[0:3],xk[3:6],altrange = self.current_state_estimate.altrange, rhovsalt = self.current_state_estimate.rhovsalt,calc_B = calc_B,calc_S = calc_S,calc_all = calc_all)#,gc = self.current_state_estimate.g_coeffs, hc = self.current_state_estimate.h_coeffs)



        else:
            self.prev_os = self.current_state_estimate
            self.prev_vec_os = self.current_state_vec_estimate
            self.current_state_estimate = truth

        if len(estimated_GPS)>0:
            self.current_state_vec_estimate = estimated_nparray(np.concatenate([self.current_state_estimate.R,self.current_state_estimate.V,np.array([[j.bias.val for j in estimated_GPS]]).T]),cov,self.current_state_vec_estimate.int_cov)
        else:
            self.current_state_vec_estimate = estimated_nparray(np.concatenate([self.current_state_estimate.R,self.current_state_estimate.V]),cov,self.current_state_vec_estimate.int_cov)
        return self.current_state_estimate


    def propagate(self,j2000):
        estimated_GPS = [j for j in self.sat.sensors if j.has_bias and isinstance(j,GPS)]

        next_os = self.current_state_estimate.orbit_rk4((j2000-self.current_state_estimate.J2000)*cent2sec,True,True)

        r,v = next_os.R,next_os.V
        dr__dr0,dr__dv0,dv__dr0,dv__dv0 = self.current_state_estimate.orbit_rk4_jacobians(self.update_period, True,True)
        Fk = np.block([[dr__dr0, dr__dv0], [dv__dr0, dv__dv0]])

        # dr__dr0,dr__dv0,dv__dr0,dv__dv0 = self.current_state_estimate.orbit_rk4_jacobians(self.update_period, True,True)

        # estimated_GPS = [j for j in self.sat.sensors if j.has_bias and isinstance(j,GPS)]

        sens_bias_len = sum([j.output_length for j in estimated_GPS])
        full_len = self.current_state_vec_estimate.val.size

        x1_full = np.concatenate([r,v,self.current_state_vec_estimate.val[6:]])
        # x0_full = self.current_state_vec_estimate.val


        F_full = np.zeros((full_len,full_len))
        F_full[0:6,0:6] = Fk
        F_full[6:sens_bias_len,6:sens_bias_len] = np.eye(sens_bias_len)

        #Find x estimate and covariance estimate from predict step
        cc = self.current_cov
        cov1 = Fk@cc@Fk.T + self.current_state_vec_estimate.int_cov
        #Adjust for GPS downrange bias  TODO--why was this included? add back?
        # downrange_adjustment = normalize(v_ECI_measured)*self.GPS_downrange_bias_magnitude #km
        # r_ECI_measured = r_ECI_measured + downrange_adjustment
        #Set variables and return
        self.current_cov = cov1[0:6,0:6]
        self.prev_os = self.current_state_estimate
        self.prev_vec_os = self.current_state_vec_estimate
        self.current_state_estimate = Orbital_State(j2000,x1_full[0:3],x1_full[3:6],altrange = self.current_state_estimate.altrange, rhovsalt = self.current_state_estimate.rhovsalt,calc_B = True,calc_S = True,calc_all = True)#,gc = self.current_state_estimate.g_coeffs, hc = self.current_state_estimate.h_coeffs)

        if len(estimated_GPS)>0:
            self.current_state_vec_estimate = estimated_nparray(np.concatenate([self.current_state_estimate.R,self.current_state_estimate.V,np.array([[j.bias.val for j in estimated_GPS]]).T]),cov1,self.current_state_vec_estimate.int_cov)
        else:
            self.current_state_vec_estimate = estimated_nparray(np.concatenate([self.current_state_estimate.R,self.current_state_estimate.V]),cov1,self.current_state_vec_estimate.int_cov)
        return self.current_state_estimate


    def state_estimate_vec(self):
        return np.concatenate([self.current_state_estimate.R,self.current_state_estimate.V])

    def reset(self, sat,sample_time = 1, state_estimate = None, cov_estimate = None, GPS_rnoise_var_guess=0,GPS_vnoise_var_guess=0,integration_cov = None,GPS_downrange_bias_magnitude = 0):
        """
        Reset the estimator. Used in case of fault. Also called by the __init__ method.
        Inputs:
            init_settings -- tuple with settings to be set
        Returns:
            nothing
        """

        if sat is None:
            sat = Satellite()
            for j in range(len(sat.sensors)):
                if sat.sensors[j].has_bias and isinstance(sat.sensors[j],GPS):
                    jbs = sat.sensors[j].bias.size
                    if not isinstance(sat.sensors[j].bias,estimated_nparray):
                        sat.sensors[j].bias = estimated_nparray(sat.sensors[j].bias,np.diagflat(sat.sensors[j].bias_std_rate*np.ones(jbs)),np.diagflat(sat.sensors[j].bias_std_rate**2*np.ones(jbs)))
        else:
            for j in range(len(sat.sensors)):
                if sat.sensors[j].has_bias and isinstance(sat.sensors[j],GPS):
                    jbs = sat.sensors[j].bias.size
                    if not isinstance(sat.sensors[j].bias,estimated_nparray):
                        sat.sensors[j].bias = estimated_nparray(sat.sensors[j].bias,np.diagflat(sat.sensors[j].bias_std_rate*np.ones(jbs)),np.diagflat(sat.sensors[j].bias_std_rate**2*np.ones(jbs)))
        self.sat = sat
        if state_estimate is None:
            state_estimate = Orbital_State(0,np.zeros(3),np.zeros(3))
        if cov_estimate is None:
            cov_estimate = np.block([[1e8*np.eye(3),np.zeros((3,3))],[np.zeros((3,3)),1e3*np.eye(3)]])
        if integration_cov is None:
            integration_cov = np.block([[1*np.eye(3),np.zeros((3,3))],[np.zeros((3,3)),1e-1*np.eye(3)]])


        estimated_GPS = [j for j in self.sat.sensors if j.has_bias and isinstance(j,GPS)]
        self.current_state_estimate = state_estimate
        if len(estimated_GPS)>0:
            full_cov_est = np.block([[cov_estimate,np.zeros((6,6*len(estimated_GPS)))],[np.zeros((len(estimated_GPS)*6,6)),scipy.linalg.block_diag(**[j.bias.cov for j in estimated_GPS])]])
            full_int_cov = np.block([[integration_cov,np.zeros((6,6*len(estimated_GPS)))],[np.zeros((len(estimated_GPS)*6,6)),scipy.linalg.block_diag(*[j.bias.int_cov for j in estimated_GPS])]])
            self.current_state_vec_estimate = estimated_nparray(np.concatenate([state_estimate.R,state_estimate.V,np.array([[j.bias.val for j in estimated_GPS]]).T]),full_cov_est,full_int_cov)
        else:
            self.current_state_vec_estimate = estimated_nparray(np.concatenate([state_estimate.R,state_estimate.V]),cov_estimate,integration_cov)
        self.prev_os = self.current_state_estimate
        self.prev_vec_os = self.current_state_vec_estimate
        self.update_period = sample_time
        self.GPS_downrange_bias_magnitude = GPS_downrange_bias_magnitude #km [THIS SHOULD BE UPLINKED!! It is not estimated in this code!!]
        self.current_cov = cov_estimate #Pk
        self.int_cov = integration_cov #Qk
        self.innovation = np.zeros(6) #zk - hk at most recent timestep (measurement residual)
        if isinstance(GPS_rnoise_var_guess,list):
            if len(GPS_rnoise_var_guess) == 1:
                GPS_rnoise_var_guess = GPS_rnoise_var_guess[0]
        if isinstance(GPS_rnoise_var_guess,int):
            self.sensor_cov = np.block([[np.eye(3)*GPS_rnoise_var_guess, np.zeros([3,3])],[np.zeros([3,3]), np.eye(3)*GPS_vnoise_var_guess]]) #Rk
        else:
            self.sensor_cov = scipy.linalg.block_diag(*[np.block([[np.eye(3)*GPS_rnoise_var_guess[j], np.zeros([3,3])],[np.zeros([3,3]), np.eye(3)*GPS_vnoise_var_guess[j]]]) for j in range(len(GPS_rnoise_var_guess))]) #Rk



        # self.last_update = -1e8
        return


class PerfectOrbitalEstimator(OrbitalEstimator):
    def update(self, GPS_in, j2000,truth = None):
        if isinstance(truth,Orbital_State):
            return truth.copy()
        elif truth is not None:
            return np.copy(truth)
