import numpy as np
from sat_ADCS_helpers.helpers import *
import pytest
import warnings
from .actuators import *
from .sensors import *
from .disturbances import *
from sat_ADCS_orbit import *
from scipy.linalg import block_diag
"""
to update from estimation folder

cd ../satellite_hardware && \
python3.10 -m build && \
pip3.10 install ./dist/sat_ADCS_satellite-0.0.1.tar.gz && \
cd ../estimation
"""

class Satellite:

    #TODO: add CMGs, thrusters, flexible components, kinematic chains

    def __init__(self,mass = 1,
                    COM = None,
                    J = None,
                    disturbances = [],
                    sensors = [],
                    actuators = [],estimated = False
                    ):

        self.disturbances = disturbances
        self.actuators = actuators
        self.sensors = sensors
        self.attitude_sensors = [j for j in sensors if j.attitude_sensor]
        self.other_sensors = [j for j in sensors if not j.attitude_sensor and not isinstance(j,GPS)]
        self.orbit_sensors = [j for j in sensors if isinstance(j,GPS)]
        self.last_dist_torq = np.nan*np.ones(3)
        self.last_act_torq = np.nan*np.ones(3)
        self.last_dist_list = [np.nan*np.ones(3) for j in range(100)]
        self.last_act_list = [np.nan*np.ones(3) for j in range(100)]

        if J is None:
            J = np.eye(3)
        self.J = J #kg*m^2 INCLUDES ANGULAR MOMENTUM STORAGE

        #physical properties
        #mass
        if COM is None:
            COM = np.zeros(3)
        self.mass = mass #kg INCLUDES ANGULAR MOMENTUM STORAGE
        self.COM = COM #m INCLUDES ANGULAR MOMENTUM STORAGE
        self.momentum_inds = np.array([j for j in range(len(self.actuators)) if self.actuators[j].has_momentum])
        self.control_len = len(self.actuators)
        self.number_RW = sum([1 for j in self.actuators if isinstance(j,RW)])
        self.state_len = 7+self.number_RW
        self.update_J(J)
        self.estimated = estimated
        if self.estimated:
            self.act_bias_inds = [j for j in range(len(self.actuators)) if self.actuators[j].estimated_bias]
            self.act_bias_len = sum([self.actuators[j].input_len for j in self.act_bias_inds])
            self.att_sens_bias_inds = [j for j in range(len(self.attitude_sensors)) if self.sensors[j].estimated_bias]
            self.att_sens_bias_len = sum([self.attitude_sensors[j].output_length for j in self.att_sens_bias_inds])
            self.dist_param_inds = [j for j in range(len(self.disturbances)) if self.disturbances[j].estimated_param]
            self.dist_param_len = sum([self.disturbances[j].main_param.size for j in self.dist_param_inds])

    ##physical value set methods
    def update_J(self,J=None,com=None):

        if J is None:
            J = self.J_given
        if com is None:
            com = self.COM
        try:
            J = np.array(J).reshape((3,3))
        except:
            raise ValueError("wrong shape of J matrix")
        if not np.all(np.isreal(J)):
            raise ValueError("J matrix has non-real values")
        if not np.all(np.linalg.eigvals(J) >= 0):
            print('eig vals are',np.linalg.eigvals(J))
            raise ValueError("J matrix not PD")
        if not np.allclose(J, J.T,rtol = 1e-05, atol = 1e-08):
            raise ValueError("J matrix not symmetric")
        J = 0.5*(J + J.T)

        self.J_given = J
        self.invJ_given = np.linalg.inv(self.J)
        self.J = J - self.mass*(np.eye(3)*np.dot(com,com) - np.outer(com,com)) #at COM
        self.invJ = np.linalg.inv(self.J) #at COM
        self.J_noRW = self.J - np.sum(np.array([self.actuators[j].J*np.outer(self.actuators[j].axis,self.actuators[j].axis) for j in self.momentum_inds]),axis = 0) #at COM
        self.invJ_noRW = np.linalg.inv(self.J_noRW) #at COM

    def srp_dist_on(self):
        for j in self.disturbances:
            if isinstance(j,SRP_Disturbance):
                j.turn_on()

    def srp_dist_off(self):
        for j in self.disturbances:
            if isinstance(j,SRP_Disturbance):
                j.turn_off()

    def gen_dist_off(self,ind = None):
        if ind is None:
            for j in self.disturbances:
                if isinstance(j,General_Disturbance):
                    j.turn_off()
        else:
            d = self.disturbances[j]
            if not isinstance(d,General_Disturbance):
                raise ValueError('not right index--this is not a general disturbance')
            else:
                d.turn_off()

    def gen_dist_on(self,ind = None):
        if ind is None:
            for j in self.disturbances:
                if isinstance(j,General_Disturbance):
                    j.turn_on()
        else:
            d = self.disturbances[j]
            if not isinstance(d,General_Disturbance):
                raise ValueError('not right index--this is not a general disturbance')
            else:
                d.turn_on()

    def prop_dist_off(self,ind=None):
        if ind is None:
            for j in self.disturbances:
                if isinstance(j,Prop_Disturbance):
                    j.turn_off()
        else:
            d = self.disturbances[ind]
            if not isinstance(d,Prop_Disturbance):
                raise ValueError('not right index--this is not a prop disturbance')
            else:
                d.turn_off()

    def prop_dist_on(self,ind=None):
        if ind is None:
            for j in self.disturbances:
                if isinstance(j,Prop_Disturbance):
                    j.turn_on()
        else:
            d = self.disturbances[ind]
            if not isinstance(d,Prop_Disturbance):
                raise ValueError('not right index--this is not a prop disturbance')
            else:
                d.turn_on()

    def specific_dist_off(self,ind):
        self.disturbances[ind].turn_off()

    def specific_dist_on(self,ind):
        self.disturbances[ind].turn_on()

    def RWhs(self):
        return np.array([self.actuators[j].momentum for j in self.momentum_inds])

    def update_RWhs(self,state_or_RWhs):
        if np.size(state_or_RWhs) == self.state_len:
            RWhs = self.RWhs_from_state(state_or_RWhs)
        else:
            RWhs = state_or_RWhs
        if np.size(RWhs) != self.number_RW:
            raise ValueError("wrong number of RWhs to update")
        [self.actuators[self.momentum_inds[i]].update_momentum(RWhs[i]) for i in range(len(self.momentum_inds))]

    def match_estimate(self,est,dt):
        if self.estimated:
            full_state = est.val
            int_cov = est.int_cov.copy()/dt
            cov = est.cov
            state = full_state[0:self.state_len]
            adj = 0
            if np.shape(int_cov)[0] + 1 == self.state_len+self.act_bias_len+self.att_sens_bias_len+self.dist_param_len:
                adj = -1
            if np.size(full_state) != self.state_len+self.act_bias_len+self.att_sens_bias_len+self.dist_param_len:
                raise ValueError("est is wrong size")
            act_bias = full_state[self.state_len:self.state_len+self.act_bias_len]
            act_bias_ic = int_cov[self.state_len+adj:adj+self.state_len+self.act_bias_len,self.state_len+adj:adj+self.state_len+self.act_bias_len]
            sens_bias = full_state[self.state_len+self.act_bias_len:self.state_len+self.act_bias_len+self.att_sens_bias_len]
            sens_bias_ic = int_cov[self.state_len+self.act_bias_len+adj:adj+self.state_len+self.act_bias_len+self.att_sens_bias_len,self.state_len+self.act_bias_len+adj:adj+self.state_len+self.act_bias_len+self.att_sens_bias_len]
            dist_param = full_state[self.state_len+self.act_bias_len+self.att_sens_bias_len:self.state_len+self.act_bias_len+self.att_sens_bias_len+self.dist_param_len]
            dist_param_ic = int_cov[self.state_len+self.act_bias_len+self.att_sens_bias_len+adj:adj+self.state_len+self.act_bias_len+self.att_sens_bias_len+self.dist_param_len,self.state_len+self.act_bias_len+self.att_sens_bias_len+adj:adj+self.state_len+self.act_bias_len+self.att_sens_bias_len+self.dist_param_len]
            self.update_RWhs(full_state[0:self.state_len])

            #don't want any noise in the estimator throwing it off.
            for j in self.actuators:
                j.use_noise = False
            for j in self.sensors:
                j.use_noise = False
            for j in self.disturbances:
                j.time_varying = False

            ind = 0
            for j in self.act_bias_inds:
                l =  self.actuators[j].input_len
                self.actuators[j].set_bias(act_bias[ind:ind+l])
                self.actuators[j].bias_std_rate = act_bias_ic[ind:ind+l,ind:ind+l]**0.5
                ind += l
            ind = 0
            for j in self.att_sens_bias_inds:
                l = self.attitude_sensors[j].output_length
                self.attitude_sensors[j].bias = sens_bias[ind:ind+l]
                self.attitude_sensors[j].bias_std_rate = sens_bias_ic[ind:ind+l,ind:ind+l]**0.5
                ind += l
            ind = 0
            for j in self.dist_param_inds:
                if self.disturbances[j].active: #don't remove its values just bc its not active when called
                    l = self.disturbances[j].main_param.size
                    self.disturbances[j].main_param = dist_param[ind:ind+l]
                    self.disturbances[j].std = dist_param_ic[ind:ind+l,ind:ind+l]**0.5
                    ind += l


    # def basestate_from_state(self,state):
    #     return state[0:7]
    #
    def RWhs_from_state(self,state):
        return state[7:]

    def dynamics_for_solver(self,t,x,u,orbital_state,next_orbital_state):
        # x = x
        time_frac = t/((next_orbital_state.J2000-orbital_state.J2000)*cent2sec)
        os = orbital_state.average(next_orbital_state,time_frac)
        # x[3:7] = normalize(x[3:7])
        xdot = self.dynamics_core(x,u,os,False,False,False,False,False)
        # print(xdot
        # xdot = xdot.reshape(self.state_len)
        return xdot

    def dynamics_jac_for_solver(self,t,x,u,orbital_state,next_orbital_state):
        # x = x
        time_frac = t/((next_orbital_state.J2000-orbital_state.J2000)*cent2sec)
        # x[3:7] = normalize(x[3:7])
        os = orbital_state.average(next_orbital_state,time_frac)
        [jac,_,_,_] = self.dynJacCore(x,u,os)
        return jac

    def noiseless_dynamics(self,x,u,orbital_state,verbose = False,save_dist_torq = False,save_act_torq = False,update_actuator_noise = False,save_details = False):
        if np.size(u) != self.control_len:
            raise ValueError("wrong control length")
        if np.size(x) != self.state_len:
            raise ValueError("wrong state length")
        # elif np.size(x) == 7:
        #     x=np.concatenate([x,[j.momentum for j in self.actuators if j.has_momentum]])

        return self.dynamics_core(x,u,orbital_state,verbose,save_dist_torq,save_act_torq,False,save_details,use_noise = False)


    def dynamics(self,x,u,orbital_state,verbose = False,save_dist_torq = False,save_act_torq = False,update_actuator_noise = False,save_details = False):
        if np.size(u) != self.control_len:
            raise ValueError("wrong control length")
        if np.size(x) != self.state_len:
            raise ValueError("wrong state length")
        # elif np.size(x) == 7:
        #     x=np.concatenate([x,[j.momentum for j in self.actuators if j.has_momentum]])

        return self.dynamics_core(x,u,orbital_state,verbose,save_dist_torq,save_act_torq,update_actuator_noise,save_details)

    def dynamics_core(self,x,u,orbital_state,verbose,save_dist_torq,save_act_torq,update_actuator_noise,save_details,use_noise = True):

        #state is angular velocity in body frame, quaternion representing body-to-ECI TF, then angles of RWs, then angular rates of RWs.
        #all angles in radians, rates in rad/s

        R = orbital_state.R
        V = orbital_state.V
        B = orbital_state.B
        S = orbital_state.S
        rho = orbital_state.rho

        w = x[0:3]
        q = x[3:7]
        h = x[7:]
        J = self.J
        invJ_noRW = self.invJ_noRW

        rmat_ECI2B = rot_mat(q).T

        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V
        vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"os":orbital_state}

        dist_torq = self.dist_torque(x,vecs,details=(save_details and save_dist_torq))
        if save_dist_torq:
            self.last_dist_torq = dist_torq
        if use_noise:
            act_torq =  self.act_torque(x,u,vecs,update_actuator_noise,details=(save_details and save_act_torq))
        else:
            act_torq =  self.noiseless_act_torque(x,u,vecs,False,details=(save_details and save_act_torq))
        if save_act_torq:
            self.last_act_torq = act_torq

        qdot = 0.5*w@Wmat(q).T
        torq = act_torq + dist_torq
        # print(act_torq)
        # print(dist_torq)
        # print(u)
        # print(self.last_dist_list)
        if self.number_RW==0:
            wdot = (-np.cross(w,w@J) + torq)@invJ_noRW
            return np.concatenate([wdot,qdot])
        else:
            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])
            # H_from_RW = sum([j.body_momentum() for j in self.actuators if j.has_momentum],np.zeros(3))
            u_RW = np.concatenate([self.actuators[j].storage_torque(u[j],self,x,vecs) for j in self.momentum_inds])
            wdot = (-np.cross(w,w@J + h@RWaxes) + torq)@invJ_noRW
            RW_hdot = u_RW-wdot@RWaxes.T@np.diagflat(RWjs) #u_RW-wdot@RWaxes.T@np.diagflat(RWjs)
            if verbose:
                print('wdot',wdot)
                print('hdot',RW_hdot)
                print('comp1',u_RW)
                print('comp2',-wdot@RWaxes.T@np.diagflat(RWjs))
            # RW_hdot = np.array([-(u_RW[j]).item()-self.RW_J[j]*(self.RW_z_axis[j].reshape((3,1)).T@wdot).item() for j in list(range(self.number_RW))]).reshape((self.number_RW,1))
            # RW_w = [(RW_h[j,:]).item()/self.RW_J[j] - (self.RW_z_axis[j].reshape((3,1)).T@w).item() for j in list(range(self.number_RW))]
            return np.concatenate([wdot,qdot,RW_hdot])

    def control_bounds(self):
        maxs = np.array([j.max for j in self.actuators])
        return -maxs,maxs

    def apply_control_bounds(self,control):
        l,u = self.control_bounds()
        return np.clip(control,l,u)

    # def sort_control_vec(self,vec):
    #     vec = vec.reshape((self.control_len,1))
    #     return vec[0:self.number_MTQ,:], vec[self.number_MTQ:self.number_MTQ+self.number_RW,:],vec[self.number_MTQ+self.number_RW:,:]

    def dynamicsJacobians(self,x,u,orbital_state):
        if np.size(u) != self.control_len:
            raise ValueError("wrong control length")
        if np.size(x) != self.state_len:
            raise ValueError("wrong state length")
        # elif np.size(x) == 7:
        #     x=np.concatenate([x,[j.momentum for j in self.actuators if j.has_momentum]])
        return self.dynJacCore(x,u,orbital_state)

    def dynJacCore(self,x,u,orbital_state):

        #state is angular velocity in body frame, quaternion representing body-to-ECI TF, then angles of RWs, then angular rates of RWs.
        #all angles in radians, rates in rad/s
        #add torq is in body frame

        R = orbital_state.R
        V = orbital_state.V
        B = orbital_state.B
        S = orbital_state.S
        rho = orbital_state.rho

        w = x[0:3]
        q = x[3:7]
        RWhs = x[7:]
        J = self.J
        invJ_noRW = self.invJ_noRW

        rmat_ECI2B = rot_mat(q).T

        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V

        dR_B__dq = drotmatTvecdq(q,R)
        dB_B__dq = drotmatTvecdq(q,B)
        dV_B__dq = drotmatTvecdq(q,V)
        dS_B__dq = drotmatTvecdq(q,S)
        vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"os":orbital_state}
        com = self.COM

        # dist_torq = sum([j.torq(self,vecs)*j.active for j in self.disturbances],np.zeros(3))
        ddist_torq__dx,ddist_torq__ddmp = self.dist_torque_jac(x,vecs)
        # ddist_torq__du = np.zeros((self.control_len,3))
        # ddist_torq_dx = np.zeros((self.state_len,3))
        # ddist_torq_dx[0:3,:] = sum([j.torq_qjac(self,vecs)*j.active for j in self.disturbances],np.zeros(3))
        # ddist_torq__dtorq = np.zeros((3,3))
        # act_torq = sum([self.actuators[j].torque(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros(3))
        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
        # dact_torq__dx = np.vstack([dact_torq__dbase,dact_torq__dh])
        # dact_torq__dtorq = np.zeros((3,3))

        # qdot = 0.5*w@Wmat(q).T
        # wdot = (-np.cross(w,w@J + RWhs@RWaxes) + act_torq + dist_torq + add_torq)@invJ_noRW
        # hdot = u_RW-np.diagflat(RWjs)@RWaxes@wdot #u_RW-wdot@RWaxes.T@np.diagflat(RWjs)
        dxdot__dx = np.zeros((self.state_len,self.state_len))
        dxdot__du = np.zeros((self.control_len,self.state_len))
        dxdot__dx[3,4:7] = 0.5*w
        dxdot__dx[4:7,3] = -0.5*w
        dxdot__dx[4:7,4:7] = 0.5*skewsym(w)
        dxdot__dx[0:3,3:7] = 0.5*Wmat(q).T
        dxdot__du[:,0:3] = dact_torq__du@invJ_noRW

        dxdot__dx[:,0:3] += ddist_torq__dx@invJ_noRW
        dxdot__dx[0:7,0:3] += dact_torq__dbase@invJ_noRW
        dxdot__dx[0:3,0:3] += (-skewsym(w@J)+J@skewsym(w))@invJ_noRW
        if self.number_RW>0:
            dact_torq__dh = np.vstack([self.actuators[j].dtorq__dh(u[j],self,x,vecs) for j in range(len(self.actuators))])
            # RWjs = np.array([j.J for j in self.actuators if j.has_momentum])
            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])
            mRWjs = np.diagflat(RWjs)
            # RWaxes = np.vstack([j.axis for j in self.actuators if j.has_momentum])
            # dxdot__dx[7:,0:3] += dact_torq__dh@invJ_noRW
            dxdot__dx[0:3,0:3] += -skewsym(RWhs@RWaxes)@invJ_noRW
            dxdot__dx[7:,0:3] += (dact_torq__dh+np.cross(RWaxes,w))@invJ_noRW
            dxdot__du[:,7:] = block_diag(*[self.actuators[j].dstor_torq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
            dxdot__du[:,7:] -= dxdot__du[:,0:3]@RWaxes.T@mRWjs
            dxdot__dx[0:7,7:] = np.hstack([self.actuators[j].dstor_torq__dbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))])
            dxdot__dx[7:,7:] = np.diagflat([self.actuators[j].dstor_torq__dh(u[j],self,x,vecs) for j in self.momentum_inds])
            dxdot__dx[:,7:] -= dxdot__dx[:,0:3]@RWaxes.T@mRWjs
        if self.estimated:
            dxdot__dab = np.zeros((self.act_bias_len,self.state_len))
            dxdot__dsb = np.zeros((self.att_sens_bias_len,self.state_len))
            dxdot__ddmp = np.zeros((self.dist_param_len,self.state_len))
            if self.act_bias_len>0:
                dact_torq__dab = np.vstack([self.actuators[j].dtorq__dbias(u[j],self,x,vecs) for j in self.act_bias_inds])
            else:
                dact_torq__dab = np.zeros((0,3))

            dxdot__dab[:,0:3] = dact_torq__dab@invJ_noRW

            dxdot__ddmp[:,0:3] = ddist_torq__ddmp@invJ_noRW

            if self.number_RW>0:
                dxdot__dab[:,7:] = block_diag(*[self.actuators[j].dstor_torq__dbias(u[j],self,x,vecs) for j in self.act_bias_inds])
                dxdot__dab[:,7:] -= dxdot__dab[:,0:3]@RWaxes.T@mRWjs
                dxdot__ddmp[:,7:] -= dxdot__ddmp[:,0:3]@RWaxes.T@mRWjs

            return [dxdot__dx,dxdot__du,dxdot__dab,dxdot__dsb,dxdot__ddmp]
        return [dxdot__dx,dxdot__du]

    def dist_torque(self,x,vecs,details=False):
        if not details:
            return sum([j.torque(self,vecs) for j in self.disturbances],np.zeros(3))
        dist_list = [j.torque(self,vecs) for j in self.disturbances]
        self.last_dist_list = dist_list
        return sum(dist_list,np.zeros(3))

    def act_torque(self,x,u,vecs,update_actuator_noise,details=False):
        if not details:
            return sum([self.actuators[j].torque(u[j],self,x,vecs,update_noise = update_actuator_noise) for j in range(len(self.actuators))],np.zeros(3))
        act_list = [self.actuators[j].torque(u[j],self,x,vecs,update_noise = update_actuator_noise) for j in range(len(self.actuators))]
        self.last_act_list = act_list
        return sum(act_list,np.zeros(3))

    def noiseless_act_torque(self,x,u,vecs,update_actuator_noise,details=False):
        if not details:
            return sum([self.actuators[j].no_noise_torque(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros(3))
        act_list = [self.actuators[j].no_noise_torque(u[j],self,x,vecs) for j in range(len(self.actuators))]
        self.last_act_list = act_list
        return sum(act_list,np.zeros(3))



    def dist_torque_jac(self,x,vecs):
        ddist_torq__dx = np.zeros((self.state_len,3))
        ddist_torq__dx[3:7,:] = sum([j.torque_qjac(self,vecs) for j in self.disturbances],np.zeros((4,3)))
        ddist_torq__ddmp = np.zeros((0,3))
        if self.estimated and self.dist_param_len>0:
            ddist_torq__ddmp = np.vstack([self.disturbances[j].torque_valjac(self,vecs) for j in self.dist_param_inds])
        return ddist_torq__dx,ddist_torq__ddmp


    def dist_torque_hess(self,x,vecs):
        dddist_torq__dxdx = np.zeros((self.state_len,self.state_len,3))
        dddist_torq__dxdx[3:7,3:7,:] = sum([j.torque_qqhess(self,vecs) for j in self.disturbances],np.zeros((4,4,3)))
        dddist_torq__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,3))
        dddist_torq__dxddmp = np.zeros((self.state_len,self.dist_param_len,3))
        if self.estimated:
            ind = 0
            for j in self.dist_param_inds:
                l = self.disturbances[j].main_param.size
                dddist_torq__ddmpddmp[ind:ind+l,ind:ind+l,:] = self.disturbances[j].torque_valvalhess(self,vecs)
                dddist_torq__dxddmp[3:7,ind:ind+l,:] = self.disturbances[j].torque_qvalhess(self,vecs)
                ind += l
        return dddist_torq__dxdx,dddist_torq__dxddmp,dddist_torq__ddmpddmp

    def dynamics_Hessians(self,x,u,orbital_state):
        #state is angular velocity in body frame, quaternion representing body-to-ECI TF, then angles of RWs, then angular rates of RWs.
        #all angles in radians, rates in rad/s
        #add torq is in body frame
        if np.size(u) != self.control_len:
            raise ValueError("wrong control length")
        # if np.size(x) != self.state_len:
        #     raise ValueError("wrong state length")
        if np.size(x) != self.state_len:
            raise ValueError("wrong state length")
        # elif np.size(x) == 7:
        #     x=np.concatenate([x,[j.momentum for j in self.actuators if j.has_momentum]])
        w = x[0:3]#.reshape((3,1))
        q = x[3:7]#normalize(x[3:7,:])
        RWhs = x[7:]
        invJ_noRW = self.invJ_noRW
        J = self.J

        R = orbital_state.R
        V = orbital_state.V
        B = orbital_state.B
        S = orbital_state.S
        rho = orbital_state.rho

        rmat_ECI2B = rot_mat(q).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V
        dR_B__dq = drotmatTvecdq(q,R)
        dB_B__dq = drotmatTvecdq(q,B)
        dV_B__dq = drotmatTvecdq(q,V)
        dS_B__dq = drotmatTvecdq(q,S)
        ddR_B__dqdq = ddrotmatTvecdqdq(q,R)
        ddB_B__dqdq = ddrotmatTvecdqdq(q,B)
        ddV_B__dqdq = ddrotmatTvecdqdq(q,V)
        ddS_B__dqdq = ddrotmatTvecdqdq(q,S)
        vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":orbital_state}
        com = self.COM

        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,3)))
        ddact_torq__dbasedbase = sum([self.actuators[j].ddtorq__dbasestatedbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
        ddact_torq__dudu = np.zeros((self.control_len,self.control_len,3))
        ddact_torq__dudbase = np.zeros((self.control_len,7,3))
        for j in range(len(self.actuators)):
            ddact_torq__dudu[j,j,:] = self.actuators[j].ddtorq__dudu(u[j],self,x,vecs)
            ddact_torq__dudbase[j,:,:] = self.actuators[j].ddtorq__dudbasestate(u[j],self,x,vecs)


        ddxdot__dxdx = np.zeros((self.state_len,self.state_len,self.state_len))
        ddxdot__dudu = np.zeros((self.control_len,self.control_len,self.state_len))
        ddxdot__dxdu = np.zeros((self.state_len,self.control_len,self.state_len))

        dddist_torq__dxdx,dddist_torq__dxddmp,dddist_torq__ddmpddmp = self.dist_torque_hess(x,vecs)

        #qdot eqn
        # dxdot__dx[3:7,3:7] = 0.5*np.block([[0,w.T],[-w,skewsym(w)]])
        # dxdot__dx[0:3,3:7] = 0.5*Wmat(q).T
        ddxdot__dxdx[3,0:3,4:7]  = 0.5*np.eye(3)
        ddxdot__dxdx[4:7,0:3,3]  = 0.5*-np.eye(3)#[-x | -skewsym(x)]
        ddxdot__dxdx[4:7,0:3,4:7] = 0.5*-np.cross(np.expand_dims(np.eye(3),0),np.expand_dims(np.eye(3),1))
        ddxdot__dxdx[0:3,3:7,3:7] = np.transpose(ddxdot__dxdx[3:7,0:3,3:7],(1,0,2))
        #wdot eqn w/o h
        # dxdot__du[:,0:3] = dact_torq__du@invJ_noRW
        # dxdot__dtorq[:,0:3] = invJ_noRW
        # dxdot__dx[3:7,0:3] += ddist_torq__dq@invJ_noRW
        # dxdot__dx[0:7,0:3] += dact_torq__dbase@invJ_noRW
        # dxdot__dx[0:3,0:3] += (-skewsym(w@J)+J@skewsym(w))@invJ_noRW

        # ddxdot__dxdm[3:7,:,0:3] = -np.cross(np.expand_dims(dB_B__dq,1),np.expand_dims(np.eye(3),0))@invJ_noRW
        ddxdot__dudu[:,:,0:3] = ddact_torq__dudu@invJ_noRW
        ddxdot__dxdu[0:7,:,0:3] = np.transpose(ddact_torq__dudbase,(1,0,2))@invJ_noRW
        ddxdot__dxdx[:,:,0:3] += dddist_torq__dxdx@invJ_noRW
        # print(ddxdot__dxdx[0:7,0:3,0])
        ddxdot__dxdx[0:7,0:7,0:3] += ddact_torq__dbasedbase@invJ_noRW
        # print(ddxdot__dxdx[0:7,0:3,0])
        JxI = np.cross(np.expand_dims(J,0),np.expand_dims(np.eye(3),1))
        # ddxdot__dxdx[0:3,k,0:3] += (-JxI[:,k,:] + JxI[k,:,:])@invJ_noRW
        ddxdot__dxdx[0:3,0:3,0:3] += (JxI + np.transpose( JxI,(1,0,2)))@invJ_noRW#(np.cross(np.expand_dims(J,0),np.expand_dims(np.eye(3),1)) - np.cross(np.expand_dims(np.eye(3),0),J))@invJ_noRW
        if self.number_RW>0:
            #wdot eqn w/ h
            # dact_torq__dh = np.vstack([self.actuators[j].dtorq__dh(u[j],self,x,vecs) for j in range(len(self.actuators))])
            ddact_torq__dudh = np.zeros((self.control_len,self.number_RW,3))
            ddact_torq__dhdh = np.zeros((self.number_RW,self.number_RW,3))
            ddact_torq__dbasedh =  np.zeros((7,self.number_RW,3))#np.dstack([ self.actuators[j].ddtorq__dbasestatedh(u[j],self,x,vecs) for j in range(len(self.actuators))])
            ind = 0
            for ind in range(self.number_RW):
                j = self.momentum_inds[ind]
                ddact_torq__dudh[j,ind,:] = self.actuators[j].ddtorq__dudh(u[j],self,x,vecs)
                ddact_torq__dhdh[ind,ind,:] = self.actuators[j].ddtorq__dhdh(u[j],self,x,vecs)
                ddact_torq__dbasedh[:,ind,:] = np.squeeze(self.actuators[j].ddtorq__dbasestatedh(u[j],self,x,vecs))

            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])
            # RWjs = np.array([j.J for j in self.actuators if j.has_momentum])
            mRWjs = np.diagflat(RWjs)
            # RWaxes = np.vstack([j.axis for j in self.actuators if j.has_momentum])
            ddxdot__dxdu[7:,:,0:3] += np.transpose(ddact_torq__dudh,(1,0,2))@invJ_noRW
            ddxdot__dxdx[7:,0:7,0:3] += np.transpose(ddact_torq__dbasedh,(1,0,2))@invJ_noRW ###
            ddxdot__dxdx[0:7,7:,0:3] +=  ddact_torq__dbasedh@invJ_noRW

            # dxdot__dx[0:3,0:3] += -skewsym(RWhs@RWaxes)@invJ_noRW
            # dxdot__dx[7:,0:3] += (dact_torq__dh+np.cross(RWaxes,w))@invJ_noRW
            AxI = -np.cross(np.expand_dims(RWaxes,1),np.expand_dims(np.eye(3),0))
            ddxdot__dxdx[7:,0:3,0:3] += -AxI@invJ_noRW#np.cross(RWaxes,invJ_noRW) ####
            ddxdot__dxdx[0:3,7:,0:3] += -np.transpose(AxI,(1,0,2))@invJ_noRW#np.cross(invJ_noRW,RWaxes)
            ddxdot__dxdx[7:,7:,0:3] += (ddact_torq__dhdh)@invJ_noRW
            #hdot eqn
            # dxdot__dtorq[:,7:] = -mRWjs@RWaxes@invJ_noRW
            # dxdot__du[:,7:] = blocxd_diag([self.actuators[j].dstor_torq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
            ind = 0
            for ind in range(self.number_RW):
                j = self.momentum_inds[ind]
                # if self.actuators[j].has_momentum:
                ddxdot__dxdu[0:7,j,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dudbasestate(u[j],self,x,vecs),(1,0,2)))
                ddxdot__dxdu[7+ind,j,7+ind] += np.transpose(self.actuators[j].ddstor_torq__dudh(u[j],self,x,vecs),(1,0,2))
                ddxdot__dudu[j,j,7+ind] = self.actuators[j].ddstor_torq__dudu(u[j],self,x,vecs)
                ddxdot__dxdx[0:7,0:7,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedbasestate(u[j],self,x,vecs))
                ddxdot__dxdx[7+ind,0:7,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dbasestatedh(u[j],self,x,vecs),(1,0,2)))
                ddxdot__dxdx[0:7,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedh(u[j],self,x,vecs))
                ddxdot__dxdx[7+ind,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dhdh(u[j],self,x,vecs))
                # ind += 1
            ddxdot__dxdu[:,:,7:] -= ddxdot__dxdu[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dudu[:,:,7:] -= ddxdot__dudu[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dxdx[:,:,7:] -= ddxdot__dxdx[:,:,0:3]@RWaxes.T@mRWjs

        if self.estimated:
            ddxdot__dxdab = np.zeros((self.state_len,self.act_bias_len,self.state_len))
            ddxdot__dudab = np.zeros((self.control_len,self.act_bias_len,self.state_len))

            ddxdot__dxdsb = np.zeros((self.state_len,self.att_sens_bias_len,self.state_len))
            ddxdot__dudsb = np.zeros((self.control_len,self.att_sens_bias_len,self.state_len))

            ddxdot__dxddmp = np.zeros((self.state_len,self.dist_param_len,self.state_len))
            ddxdot__duddmp = np.zeros((self.control_len,self.dist_param_len,self.state_len))

            ddxdot__dabdab = np.zeros((self.act_bias_len,self.act_bias_len,self.state_len))
            ddxdot__dsbdsb = np.zeros((self.att_sens_bias_len,self.att_sens_bias_len,self.state_len))
            ddxdot__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,self.state_len))

            ddxdot__dabdsb = np.zeros((self.act_bias_len,self.att_sens_bias_len,self.state_len))
            ddxdot__dabddmp = np.zeros((self.act_bias_len,self.dist_param_len,self.state_len))
            ddxdot__dsbddmp = np.zeros((self.att_sens_bias_len,self.dist_param_len,self.state_len))

            ddact_torq__dudab = np.zeros((self.control_len,self.act_bias_len,3))
            ddact_torq__dabdab = np.zeros((self.act_bias_len,self.act_bias_len,3))
            ddact_torq__dbasedab = np.zeros((7,self.act_bias_len,3))
            ind = 0
            for j in range(len(self.act_bias_inds)):
                actind = self.act_bias_inds[j]
                l = self.actuators[actind].input_len
                ddact_torq__dabdab[ind:ind+l,ind:ind+l,:] = self.actuators[actind].ddtorq__dbiasdbias(u[actind],self,x,vecs)
                ddact_torq__dudab[actind,ind:ind+l,:] = self.actuators[actind].ddtorq__dudbias(u[actind],self,x,vecs)
                ddact_torq__dbasedab[:,ind:ind+l,:] = np.transpose(self.actuators[actind].ddtorq__dbiasdbasestate(u[actind],self,x,vecs),(1,0,2))
                ind+=l

            # dddist_torq__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,3))
            # dddist_torq__dxddmp = np.zeros((self.state_len,self.dist_param_len,3))
            # ind = 0
            # for j in self.dist_param_inds:
            #     l = self.disturbances[j].main_param.size
            #     dddist_torq__ddmpddmp[ind:ind+l,ind:ind+l,:] = self.disturbances[j].torque_valvalhess(self,vecs)
            #     dddist_torq__dxddmp[3:7,ind:ind+l,:] = self.disturbances[j].torque_qvalhess(self,vecs)
            #     ind += l

            ddxdot__dabdab[:,:,0:3] = ddact_torq__dabdab@invJ_noRW
            ddxdot__dudab[:,:,0:3] = ddact_torq__dudab@invJ_noRW
            ddxdot__dxdab[0:7,:,0:3] = ddact_torq__dbasedab@invJ_noRW

            ddxdot__ddmpddmp[:,:,0:3] = dddist_torq__ddmpddmp@invJ_noRW
            ddxdot__dxddmp[:,:,0:3] = dddist_torq__dxddmp@invJ_noRW

            # dact_torq__dab = np.vstack([self.actuators[j].dtorq__du(u[j],self,x,vecs) for j in self.act_bias_inds])
            # dxdot__dab[:,0:3] = dact_torq__dab@invJ_noRW
            # ddist_torq__ddmp = np.vstack([self.disturbances[j].torque_valjac(self,vecs) for j in self.dist_param_inds])
            # dxdot__ddmp[:,0:3] = ddist_torq__ddmp@invJ_noRW

            if self.number_RW>0:
                #wdot eqn w/ h
                # dact_torq__dh = np.vstack([self.actuators[j].dtorq__dh(u[j],self,x,vecs) for j in range(len(self.actuators))])
                ddact_torq__dabdh = np.zeros((self.act_bias_len,self.number_RW,3))
                if ind in range(len(self.act_bias_inds)):
                    actind = self.act_bias_inds[ind]
                    if ind in self.momentum_inds:
                        j = np.where(self.momentum_inds==ind)
                        ddact_torq__dabdh[ind,j,:] = self.actuators[actind].ddtorq__dbiasdh(u[actind],self,x,vecs)

                ddxdot__dxdab[7:,:,0:3] += np.transpose(ddact_torq__dabdh,(1,0,2))@invJ_noRW
                if ind in range(len(self.act_bias_inds)):
                    actind = self.act_bias_inds[ind]
                    l = self.actuators[actind].input_len
                    if ind in self.momentum_inds:
                        j = np.where(self.momentum_inds==ind)
                        ddxdot__dxdab[0:7,ind:ind+l,7+j] += np.squeeze(np.transpose(self.actuators[actind].ddstor_torq__dbiasdbasestate(u[actind],self,x,vecs),(1,0,2)))
                        ddxdot__dxdab[7+j,ind:ind+l,7+j] += np.transpose(self.actuators[actind].ddstor_torq__dbiasdh(u[actind],self,x,vecs),(1,0,2))
                        ddxdot__dabdab[ind:ind+l,ind:ind+l,7+j] = self.actuators[actind].ddstor_torq__dbiasdbias(u[actind],self,x,vecs)
                        ddxdot__dudab[actind,ind:ind+l,7+j] = self.actuators[actind].ddstor_torq__dudbias(u[actind],self,x,vecs)

                ddxdot__dxdab[:,:,7:] -= ddxdot__dxdab[:,:,0:3]@RWaxes.T@mRWjs
                ddxdot__dudab[:,:,7:] -= ddxdot__dudab[:,:,0:3]@RWaxes.T@mRWjs
                ddxdot__dabdab[:,:,7:] -= ddxdot__dabdab[:,:,0:3]@RWaxes.T@mRWjs
                ddxdot__dxddmp[:,:,7:] -= ddxdot__dxddmp[:,:,0:3]@RWaxes.T@mRWjs
                ddxdot__ddmpddmp[:,:,7:] -= ddxdot__ddmpddmp[:,:,0:3]@RWaxes.T@mRWjs

                return [[ddxdot__dxdx,ddxdot__dxdu,ddxdot__dxdab,ddxdot__dxdsb,ddxdot__dxddmp],[ddxdot__dxdu.T,ddxdot__dudu,ddxdot__dudab,ddxdot__dudsb,ddxdot__duddmp],[0,0,ddxdot__dabdab,ddxdot__dabdsb,ddxdot__dabddmp],[0,0,0,ddxdot__dsbdsb,ddxdot__dsbddmp],[0,0,0,0,ddxdot__ddmpddmp]]
        return [[ddxdot__dxdx,ddxdot__dxdu],[ddxdot__dxdu.T,ddxdot__dudu]]

    def rk4(self,x,u,dt,orbital_state0,orbital_state1,verbose=False,mid_orbital_state = None,quat_as_vec = True,save_info = True,give_err_est = False):
        x[3:7] = normalize(x[3:7])
        # np.set_printoptions(precision=3)
        if quat_as_vec:
            if mid_orbital_state is None:
                mid_orbital_state = orbital_state0.average(orbital_state1)
            # print(1,orbital_state0.R)
            k1 = self.dynamics(x,u,orbital_state0,verbose = verbose)
            k2_in = x+k1*0.5*dt
            k2_in[3:7] = normalize(k2_in[3:7])
            # print(2,mid_orbital_state.R)
            k2 = self.dynamics(k2_in,u,mid_orbital_state,verbose = verbose)
            k3_in = x+k2*0.5*dt
            k3_in[3:7] = normalize(k3_in[3:7])
            # print(3,mid_orbital_state.R)
            k3 = self.dynamics(k3_in,u,mid_orbital_state,verbose = verbose)
            k4_in = x+k3*dt
            k4_in[3:7] = normalize(k4_in[3:7])
            # print(4,orbital_state1.R)
            k4 = self.dynamics(k4_in,u,orbital_state1,save_act_torq = save_info,save_dist_torq = save_info,save_details = save_info,verbose = verbose)
            out = (x + (dt/6)*(k1+k2*2+k3*2+k4))
            out[3:7] = normalize(out[3:7])
            if verbose:
                print('k1',k1)
                print('k2',k2)
                print('k3',k3)
                print('k4',k4)


            if give_err_est:
                k33_in = x+(2*k2-k1)*dt
                k33_in[3:7] = normalize(k33_in[3:7])
                # print(3,mid_orbital_state.R)
                k33 = self.dynamics(k33_in,u,orbital_state1,verbose = verbose)
                out3 = (x + (dt/6)*(k1+k2*4+k33))
                out3[3:7] = normalize(out3[3:7])
                est_err = np.zeros(out.size-1)
                est_err[0:3] = np.abs(out[0:3]-out3[0:3])
                est_err[6:] = np.abs(out[7:]-out3[7:])
                est_err[3:6] = np.abs(quat_to_vec3(quat_mult(quat_inv(out[3:7]),out3[3:7]),0))
                return out,est_err
            else:
                return out

        else:
            if give_err_est:
                raise ValueError("not implemented for CG5 method yets")
            ki = [np.zeros(x.shape) for j in range(5)]
            F = [np.zeros(3) for j in range(5)]
            if mid_orbital_state is None:
                mid_orbital_state = [orbital_state0.average(orbital_state1,CG5_c[i]) for i in range(5)]

            for j in range(5):
                midstate = x + dt*sum([CG5_a[j,i]*ki[i] for i in range(j)],np.zeros(x.shape))
                # print('********',j)
                # print(midstate)
                if j>0:
                    # print(x[3:7])
                    # print([rot_exp(CG5_a[j,i]*F[i]) for i in range(j)])
                    midstate[3:7] = normalize(quat_mult(x[3:7],*[rot_exp(CG5_a[j,i]*F[i]) for i in range(j)]))
                ki[j] = self.dynamics(midstate,u,mid_orbital_state[j],verbose = verbose)
                F[j] = dt*midstate[0:3]
                # print(j,ki[j])

            out = x + dt*sum([CG5_b[i]*ki[i] for i in range(5)],np.zeros(x.shape))
            out[3:7] = normalize(quat_mult(x[3:7],*[rot_exp(CG5_b[i]*F[i]) for i in range(5)]))# quat_mult(x[3:7],rot_exp(CG5_b[0]*F[0]),rot_exp(CG5_b[1]*F[1]),rot_exp(CG5_b[2]*F[2]),rot_exp(CG5_b[3]*F[3]),rot_exp(CG5_b[3]*F[3]))
            # breakpoint()
            return out


    def noiseless_rk4(self,x,u,dt,orbital_state0,orbital_state1,verbose=False,mid_orbital_state = None,quat_as_vec = True,save_info = False,give_err_est = False):
        x[3:7] = normalize(x[3:7])
        # np.set_printoptions(precision=3)
        if quat_as_vec:
            if mid_orbital_state is None:
                mid_orbital_state = orbital_state0.average(orbital_state1)

            # print(1,orbital_state0.R)
            k1 = self.noiseless_dynamics(x,u,orbital_state0)
            k2_in = x+k1*0.5*dt
            k2_in[3:7] = normalize(k2_in[3:7])
            # print(2,mid_orbital_state.R)
            k2 = self.noiseless_dynamics(k2_in,u,mid_orbital_state)
            k3_in = x+k2*0.5*dt
            k3_in[3:7] = normalize(k3_in[3:7])
            # print(3,mid_orbital_state.R)
            k3 = self.noiseless_dynamics(k3_in,u,mid_orbital_state)
            k4_in = x+k3*dt
            k4_in[3:7] = normalize(k4_in[3:7])
            # print(4,orbital_state1.R)
            k4 = self.noiseless_dynamics(k4_in,u,orbital_state1,save_dist_torq = save_info,save_details = save_info)
            out = (x + (dt/6)*(k1+k2*2+k3*2+k4))
            out[3:7] = normalize(out[3:7])

            if give_err_est:
                k33_in = x+(2*k2-k1)*dt
                k33_in[3:7] = normalize(k33_in[3:7])
                # print(3,mid_orbital_state.R)
                k33 = self.dynamics(k33_in,u,orbital_state1,verbose = verbose)
                out3 = (x + (dt/6)*(k1+k2*4+k33))
                out3[3:7] = normalize(out3[3:7])
                est_err = np.zeros(out.size-1)
                est_err[0:3]=np.abs(out[0:3]-out3[0:3])
                est_err[6:]=np.abs(out[7:]-out3[7:])
                est_err[3:6] = np.abs(quat_to_vec3(quat_mult(quat_inv(out[3:7]),out3[3:7]),0))
                return out,est_err
            else:
                return out

        else:
            if give_err_est:
                raise ValueError("not implemented for CG5 method yets")
            ki = [np.zeros(x.shape) for j in range(5)]
            F = [np.zeros(3) for j in range(5)]
            if mid_orbital_state is None:
                mid_orbital_state = [orbital_state0.average(orbital_state1,CG5_c[i]) for i in range(5)]

            for j in range(5):
                midstate = x + dt*sum([CG5_a[j,i]*ki[i] for i in range(j)],np.zeros(x.shape))
                # print('********',j)
                # print(midstate)
                if j>0:
                    # print(x[3:7])
                    # print([rot_exp(CG5_a[j,i]*F[i]) for i in range(j)])
                    midstate[3:7] = normalize(quat_mult(x[3:7],*[rot_exp(CG5_a[j,i]*F[i]) for i in range(j)]))
                ki[j] = self.noiseless_dynamics(midstate,u,mid_orbital_state[j])
                F[j] = dt*midstate[0:3]
                # print(j,ki[j])

            out = x + dt*sum([CG5_b[i]*ki[i] for i in range(5)],np.zeros(x.shape))
            out[3:7] = normalize(quat_mult(x[3:7],*[rot_exp(CG5_b[i]*F[i]) for i in range(5)]))# quat_mult(x[3:7],rot_exp(CG5_b[0]*F[0]),rot_exp(CG5_b[1]*F[1]),rot_exp(CG5_b[2]*F[2]),rot_exp(CG5_b[3]*F[3]),rot_exp(CG5_b[3]*F[3]))
            # breakpoint()
            return out

    def rk4Jacobians(self,x,u,dt,orbital_state0,orbital_state1,mid_orbital_state = None,quat_as_vec = False):

        if mid_orbital_state is None:
            mid_orbital_state = orbital_state0.average(orbital_state1)
        xj = state_norm_jac(x)
        x[3:7] = normalize(x[3:7])
        # x0_jacobians = [np.eye(self.state_len),np.zeros((self.control_len,self.state_len)),np.zeros((3,self.state_len)),np.zeros((3,self.state_len))]
        if self.estimated:
            x0_jacobians = [xj,np.zeros((self.control_len,self.state_len)),np.zeros((self.act_bias_len,self.state_len)),np.zeros((self.att_sens_bias_len,self.state_len)),np.zeros((self.dist_param_len,self.state_len))]
        else:
            x0_jacobians = [xj,np.zeros((self.control_len,self.state_len))]
        k1 = self.noiseless_dynamics(x,u,orbital_state0)
        dynamics_jac1 = self.dynamicsJacobians(x,u,orbital_state0)
        k1_jacs = self.rk4_xd_jacobians(dynamics_jac1,x0_jacobians)
        x1,dx1__dz1,x1_jacobians = self.rk4_normstep_w_jac(x,xj,dt*0.5,k1,k1_jacs)

        k2 = self.noiseless_dynamics(x1,u,mid_orbital_state)
        dynamics_jac2 = self.dynamicsJacobians(x1,u,mid_orbital_state)
        k2_jacs = self.rk4_xd_jacobians(dynamics_jac2,x1_jacobians)
        x2,dx2__dz2,x2_jacobians = self.rk4_normstep_w_jac(x,xj,dt*0.5,k2,k2_jacs)

        k3 = self.noiseless_dynamics(x2,u,mid_orbital_state)
        dynamics_jac3 = self.dynamicsJacobians(x2,u,mid_orbital_state)
        k3_jacs = self.rk4_xd_jacobians(dynamics_jac3,x2_jacobians)
        x3,dx3__dz3,x3_jacobians = self.rk4_normstep_w_jac(x,xj,dt,k3,k3_jacs)

        k4 = self.noiseless_dynamics(x3,u,orbital_state1)
        dynamics_jac4 = self.dynamicsJacobians(x3,u,orbital_state1)
        k4_jacs = self.rk4_xd_jacobians(dynamics_jac4,x3_jacobians)
        x4,dx4__dz4,x4_jacobians = self.rk4_normstep_w_jac(x,xj,[dt/6,dt/3,dt/3,dt/6],[k1,k2,k3,k4],[k1_jacs,k2_jacs,k3_jacs,k4_jacs])
        return x4_jacobians

    def rk4_normstep_w_jac(self,x0,x0jac,weights,vals,valjacs,includeHess=False,z_u_hessians=None):
        if isinstance(weights,list):
            z = x0 + sum([weights[j]*vals[j] for j in list(range(len(weights)))])
            z_jac = [sum([weights[j]*valjacs[j][i] for j in list(range(len(weights)))]) for i in range(len(valjacs[0]))]
            z_jac[0] += x0jac
        else:
            z = x0 + weights*vals#sum([weights[j]*vals[j] for j in list(range(len(weights)))]))
            z_jac = [valjacs[i]*weights for i in range(len(valjacs))]
            z_jac[0] += x0jac
        x = np.copy(z)
        x[3:7] = normalize(x[3:7])
        dx__dz = state_norm_jac(z)
        x_jacobians = [j@dx__dz for j in z_jac]
        if includeHess:
            ddx__dzdz = state_norm_hess(z)
            dz__du = z_jac[1]
            if isinstance(weights,list):
                z_uhess = [sum([weights[j]*z_u_hessians[j][i] for j in list(range(len(weights)))]) for i in range(len(z_u_hessians[0]))]
            else:
                z_uhess = [z_u_hessians[i]*weights for i in range(len(z_u_hessians))]
            x_u_hessians = [np.tensordot(z_jac[j],dz__du@ddx__dzdz,([1],[0])) + z_uhess[j]@dx__dz for j in range(len(z_jac))]#self.rk4_norm_u_hessians(dx__dz,ddx__dzdz,z_jac,z_u_hessians)
            return x,x_jacobians,x_u_hessians
        return x,dx__dz,x_jacobians

    def rk4_nonormstep_w_jac(self,x0,x0jac,weights,vals,valjacs,includeHess=False,z_u_hessians=None):
        if isinstance(weights,list):
            z = x0 + sum([weights[j]*vals[j] for j in list(range(len(weights)))])
            z_jac = [sum([weights[j]*valjacs[j][i] for j in list(range(len(weights)))]) for i in range(len(valjacs[0]))]
            z_jac[0] += x0jac
        else:
            z = x0 + weights*vals#sum([weights[j]*vals[j] for j in list(range(len(weights)))]))
            z_jac = [valjacs[i]*weights for i in range(len(valjacs))]
            z_jac[0] += x0jac
        if includeHess:
            dz__du = z_jac[1]
            if isinstance(weights,list):
                z_uhess = [sum([weights[j]*z_u_hessians[j][i] for j in list(range(len(weights)))]) for i in range(len(z_u_hessians[0]))]
            else:
                z_uhess = [z_u_hessians[i]*weights for i in range(len(z_u_hessians))]
            return z,z_jac,z_uhess
        return z,np.eye(self.state_len),z_jac

    def rk4_xd_jacobians(self,dyn_jacobians,x_jacobians):
        return [x_jacobians[j]@dyn_jacobians[0] + (j>0)*dyn_jacobians[j] for j in range(len(dyn_jacobians))]

    def rk4_u_Hessians(self,x,u,dt,orbital_state0,orbital_state1,mid_orbital_state = None):
        xj = state_norm_jac(x)
        # xh = state_norm_hess(x)
        x[3:7] = normalize(x[3:7])
        # x0_jacobians = [np.eye(self.state_len),np.zeros((self.control_len,self.state_len)),np.zeros((3,self.state_len)),np.zeros((3,self.state_len))]
        if self.estimated:
            x0_jacobians = [xj,np.zeros((self.control_len,self.state_len)),np.zeros((self.act_bias_len,self.state_len)),np.zeros((self.att_sens_bias_len,self.state_len)),np.zeros((self.dist_param_len,self.state_len))]
            x0_u_hessians = [np.zeros((self.state_len,self.control_len,self.state_len)),np.zeros((self.control_len,self.control_len,self.state_len)),np.zeros((self.act_bias_len,self.control_len,self.state_len)),np.zeros((self.att_sens_bias_len,self.control_len,self.state_len)),np.zeros((self.dist_param_len,self.control_len,self.state_len))]
        else:
            x0_jacobians = [xj,np.zeros((self.control_len,self.state_len))]
            x0_u_hessians = [np.zeros((self.state_len,self.control_len,self.state_len)),np.zeros((self.control_len,self.control_len,self.state_len))]

        # print(xh)
        # print(xh.shape)
        if mid_orbital_state is None:
            mid_orbital_state = orbital_state0.average(orbital_state1)

        k1,k1_jacs,k1_u_hess = self.dynamics_w_jac_hess(x,x0_jacobians,x0_u_hessians,u,orbital_state0)
        x1,x1_jacobians,x1_u_hessians = self.rk4_normstep_w_jac(x,xj,dt*0.5,k1,k1_jacs,True,k1_u_hess)

        k2,k2_jacs,k2_u_hess = self.dynamics_w_jac_hess(x1,x1_jacobians,x1_u_hessians,u,mid_orbital_state)
        x2,x2_jacobians,x2_u_hessians = self.rk4_normstep_w_jac(x,xj,dt*0.5,k2,k2_jacs,True,k2_u_hess)

        k3,k3_jacs,k3_u_hess = self.dynamics_w_jac_hess(x2,x2_jacobians,x2_u_hessians,u,mid_orbital_state)
        x3,x3_jacobians,x3_u_hessians = self.rk4_normstep_w_jac(x,xj,dt,k3,k3_jacs,True,k3_u_hess)

        k4,k4_jacs,k4_u_hess = self.dynamics_w_jac_hess(x3,x3_jacobians,x3_u_hessians,u,orbital_state1)
        x4,x4_jacobians,x4_u_hessians = self.rk4_normstep_w_jac(x,xj,[dt/6,dt/3,dt/3,dt/6],[k1,k2,k3,k4],[k1_jacs,k2_jacs,k3_jacs,k4_jacs],True,[k1_u_hess,k2_u_hess,k3_u_hess,k4_u_hess])
        return x4_u_hessians

    def dynamics_w_jac_hess(self,x,x_jacobians,x_u_hessians,u,orbital_state):
        k = self.noiseless_dynamics(x,u,orbital_state)
        jac = self.dynamicsJacobians(x,u,orbital_state)
        hess = self.dynamics_Hessians(x,u,orbital_state)
        xd_jac = self.rk4_xd_jacobians(jac,x_jacobians)
        xd_u_hess = self.rk4_xd_u_hessians(jac,hess,x_jacobians,x_u_hessians)
        return k,xd_jac,xd_u_hess

    def rk4_xd_u_hessians(self,dyn_jacobians,dyn_hessians,x_jacobians,x_u_hessians):
        # [xhess,uhess,thess,mhess] = dyn_hessians
        dxd__dx = dyn_jacobians[0] #
        ddxd__dxdx = dyn_hessians[0][0] #
        ddxd__dxdu =  dyn_hessians[0][1] #
        # ddxd__dxdt = xhess[2]
        # ddxd__dxdm = xhess[3]
        # [ddxd__dxdu,ddxd__dudu,ddxd__dudt,ddxd__dudm] = uhess
        # [Au,Bu,Cu] = uhess
        # ddxd__dtdt = thess[2]
        # ddxd__dmdm = mhess[3]
        # [At,Bt,Ct] = thess
        [dx__dx0,dx__du] = x_jacobians[0:2]
        xd_x_hessians = [np.transpose(j,(1,0,2)) for j in dyn_hessians[0]]# [xhess[0],np.transpose(xhess[1],(1,0,2)),np.transpose(xhess[2],(1,0,2)),np.transpose(xhess[3],(1,0,2))]
        # [ddx__dx0du,ddx__dudu,ddx__dtdu,ddx__dmdu] = x_u_hessians
        # xd_u_hessians0 = [0,uhess[1],np.transpose(uhess[2],(1,0,2)), np.transpose(uhess[3],(1,0,2))]
        xd_u_hessians0 = [np.transpose(j,(1,0,2)) for j in dyn_hessians[1]]
        xd_u_hessians0[0] = dyn_hessians[1][0]
        # ddk4__dtorqdu = [multi_matrix_chain_rule_scalar([Cx4,Cu4,Ct4],[dx3__du[i,:],np.eye(self.control_len)[:,i].reshape((self.control_len,1)),np.zeros((3,1))],3,[0,1,2]) \
        #                     + multi_vector_chain_rule([multi_matrix_chain_rule_scalar([Ax4,Au4,At4],[dx3__du[i,:],np.eye(self.control_len)[:,i].reshape((self.control_len,1)),np.zeros((3,1))],3,[0,1,2])],[dx3__dtorq],1) \
        #                     + multi_vector_chain_rule([A4],[ddx3__dtorqdu[i]],1) for i in list(range(self.control_len))]
        # ddxd__dxdu = ddxd__dxdu
        xd_u_hessians = [ np.tensordot(x_jacobians[j],ddxd__dxdu+dx__du@ddxd__dxdx,([1],[0])) + x_u_hessians[j]@dxd__dx + (j>0)*(xd_u_hessians0[j] + dx__du@xd_x_hessians[j] )  for j in range(len(xd_x_hessians))]
        # [multi_matrix_chain_rule_scalar([ddxd__dxdx,ddxd__dxdu[i]],[dx__du[:,i],1],2,[0]) for i in list(range(self.control_len))]
        # xd_u_hessians[0] += 0#np.tensordot(dx__dx0,dx__du@ddxd__dxdx,([1],[0]))#+np.tensordot(dx__dx0,ddxd__dxdu,([1],[0])) # + ddx__dx0du@dxd__dx
        #[multi_vector_chain_rule([ddxd__dxdu[i]],[dx__dx0],1) \
        #+ multi_vector_chain_rule([dxd__dx],[ddx__dx0du[i]],1) for i in list(range(self.control_len))]
        # xd_u_hessians[1] += 0#uhess[1]#ddxd__dudu + dx__du@np.transpose(ddxd__dxdu,(1,0,2))#+np.tensordot(dx__du, dx__du@ddxd__dxdx,([1],[0]))+np.tensordot(dx__du,ddxd__dxdu,([1],[0]))  # + ddx__dudu@dxd__dx
        #[multi_matrix_chain_rule_scalar([Bx,ddxd__dudu[i]],[dx__du[:,i],1],2,[0]) \
        # + multi_vector_chain_rule([ddxd__dxdu[i]],[dx__du],1) \
        # + multi_vector_chain_rule([dxd__dx],[ddx__dudu[i]],1) for i in list(range(self.control_len))]
        # xd_u_hessians[2] += dx__du@np.transpose(ddxd__dxdt,(1,0,2))# + np.transpose(uhess[2],(1,0,2))#np.tensordot(dx__dt, dx__du@ddxd__dxdx,([1],[0]))#+np.tensordot(dx__dt,ddxd__dxdu,([1],[0]))# + ddx__dtdu@dxd__dx
        #[multi_matrix_chain_rule_scalar([ddxd__dxdt,ddxd__dudt[i]],[dx__du[:,i],1],2,[0]) \
        # + multi_vector_chain_rule([ddxd__dxdu[i]],[dx__dt],1) \
        # + multi_vector_chain_rule([dxd__dx],[ddx__dtdu[i]],1) for i in list(range(self.control_len))]
        # xd_u_hessians[3] += dx__du@np.transpose(ddxd__dxdm,(1,0,2))# + np.transpose(uhess[3],(1,0,2))
        return xd_u_hessians#[ddxd__dx0du, ddxd__dudu, ddxd__dtorqdu]

    def update_actuator_noise(self):
        for j in self.actuators:
            j.update_noise()

    def update_actuator_biases(self,j2000):
        for j in self.actuators:
            j.update_bias(j2000)

    def update_sensor_biases(self,j2000):
        for j in self.sensors:
            j.update_bias(j2000)

    def update_disturbances(self,j2000):
        [j.update(j2000) for j in self.disturbances]

    def sensor_values(self,x,vecs,which = None):
        if which is None:
            which = [True for j in self.attitude_sensors]
        else:
            if len(which) != len(self.attitude_sensors):
                raise ValueError("sensor selection list must be length of sensor list")
        return np.concatenate([self.attitude_sensors[j].reading(x,vecs) for j in range(len(self.attitude_sensors)) if which[j]]+[[self.actuators[j].measure_momentum() for j in self.momentum_inds]])


    def GPS_values(self,x,vecs,which = None):
        if which is None:
            which = [True for j in self.orbit_sensors]
        else:
            if len(which) != len(self.orbit_sensors):
                raise ValueError("sensor selection list must be length of sensor list")
        return np.concatenate([self.orbit_sensors[j].reading(x,vecs) for j in range(len(self.orbit_sensors)) if which[j]])

    def noiseless_sensor_values(self,x,vecs,which = None):
        if which is None:
            which = [True for j in self.attitude_sensors]
        else:
            if len(which) != len(self.attitude_sensors):
                raise ValueError("sensor selection list must be length of sensor list")
        return np.concatenate([self.attitude_sensors[j].no_noise_reading(x,vecs) for j in range(len(self.attitude_sensors)) if which[j]]+[[self.actuators[j].measure_momentum_noiseless() for j in self.momentum_inds]])

    def noiseless_GPS_values(self,x,vecs,which = None):
        if which is None:
            which = [True for j in self.orbit_sensors]
        else:
            if len(which) != len(self.orbit_sensors):
                raise ValueError("sensor selection list must be length of sensor list")
        return np.concatenate([self.orbit_sensors[j].no_noise_reading(x,vecs) for j in range(len(self.orbit_sensors)) if which[j]])

    def sensor_state_jacobian(self,x,vecs,which = None,keep_unused_biases=False):
        if which is None:
            which = [True for j in self.attitude_sensors]
        else:
            if len(which) != len(self.attitude_sensors):
                raise ValueError("sensor selection list must be length of sensor list")
        bsjac = np.hstack([self.attitude_sensors[j].basestate_jac(x,vecs) for j in range(len(self.attitude_sensors)) if which[j] ]+[np.zeros((7,self.number_RW))])
        hjac = np.hstack([np.zeros((self.number_RW,np.size(bsjac,1)-self.number_RW)),np.diagflat([1 for j in self.momentum_inds])])
        sensor_bias_jac = block_diag(*[self.attitude_sensors[j].bias_jac(x,vecs) for j in range(len(self.attitude_sensors)) if which[j] ])
        if keep_unused_biases:
            sensor_bias_jac = block_diag(*[self.attitude_sensors[j].bias_jac(x,vecs) if which[j] else [] for j in range(len(self.attitude_sensors))])
        else:
            sensor_bias_jac = block_diag(*[self.attitude_sensors[j].bias_jac(x,vecs) for j in range(len(self.attitude_sensors)) if which[j] ])
        sensor_bias_jac = np.hstack([sensor_bias_jac,np.zeros((sensor_bias_jac.shape[0],self.number_RW))])
        # hjac =  np.vstack([[self.sensors[j].momentum_jacobian(x,os) for j in len(self.sensors) if which[j]])
        return np.vstack([bsjac,hjac]),sensor_bias_jac

    def sensor_cov(self,which = None,keep_unused_biases=False):
        if which is None:
            which = [True for j in self.attitude_sensors]
        else:
            if len(which) != len(self.attitude_sensors):
                raise ValueError("sensor selection list must be length of sensor list")
        if keep_unused_biases:
            bd = np.array(block_diag(*([self.attitude_sensors[j].cov() for j in range(len(self.attitude_sensors))]+[[self.actuators[j].momentum_measurement_cov()] for j in self.momentum_inds])))
        else:
            bd = np.array(block_diag(*([self.attitude_sensors[j].cov() for j in range(len(self.attitude_sensors)) if which[j]]+[[self.actuators[j].momentum_measurement_cov()] for j in self.momentum_inds])))
        return bd

    def sensor_srcov(self,which = None,keep_unused_biases=False):
        if which is None:
            which = [True for j in self.attitude_sensors]
        else:
            if len(which) != len(self.attitude_sensors):
                raise ValueError("sensor selection list must be length of sensor list")
        if keep_unused_biases:
            bd = np.array(block_diag(*([self.attitude_sensors[j].srcov() for j in range(len(self.attitude_sensors))]+[[self.actuators[j].momentum_measurement_srcov()] for j in self.momentum_inds])))
        else:
            bd = np.array(block_diag(*([self.attitude_sensors[j].srcov() for j in range(len(self.attitude_sensors)) if which[j]]+[[self.actuators[j].momentum_measurement_srcov()] for j in self.momentum_inds])))
        return bd

    def control_cov(self):
        bd = np.array(block_diag(*[self.actuators[j].control_cov() for j in range(len(self.actuators)) ]))
        return bd

    def control_srcov(self):
        bd = np.array(block_diag(*[self.actuators[j].control_srcov() for j in range(len(self.actuators)) ]))
        return bd
