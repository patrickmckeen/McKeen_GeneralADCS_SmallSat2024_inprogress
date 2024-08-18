import numpy as np
import math
import pytest
import warnings
from enum import Enum, unique
# import matplotlib.pyplot as plt
from . import data
"""
to update from control folder

cd ../helpers && \
python3.10 -m build && \
pip3.10 install ./dist/sat_ADCS_helpers-0.0.1.tar.gz && \
cd ../ADCS
"""

NumberTypes = (int, float, complex)
# earth_mu = 3.986004418e14 #m^3/s^2
R_e = 6378.1363 #km
cent2sec = 100.0*365.25*24.0*60.0*60.0
sec2cent = 1.0/cent2sec#(100.0*365.25*24.0*60.0*60.0)
mu_e = 398600.4415 #km^3/s^2
J2coeff = 0.1082635854*(10**-2)
J2 = J2coeff*R_e**2*mu_e #km^5/s^2
solar_constant = 1361.0 #W/m^2
c = 299792458.0 #speed of light, m/s
grav_const =  6.6742e-11 #m^3*kg^-1*s^-2
m_earth = 5.9736e+24 #kg
# r_earth = 6361010.0 #m
err_J_var = 0.0#1e-7#0.00005
# stream = pkg_resources.open_text(data, 'de421.bsp')
time_eps = 1.0e-3
num_eps = 1.0e-16
zeroquat = np.array([1.0,0,0,0])

quat2vec_mat = np.block([np.zeros((3,1)),np.eye(3)])

unitvecs = [np.eye(3)[:,j] for j in list(range(3))]
unitvecs4 = [np.eye(4)[:,j] for j in list(range(4))]
noiseless_model = lambda x, noise_settings:(x, noise_settings)


CG5_a = np.zeros((5,5))
CG5_a[1,0] = 0.8177227988124852
CG5_a[2,0] = 0.3199876375476427
CG5_a[2,1] = 0.0659864263556022
CG5_a[3,0] = 0.9214417194464946
CG5_a[3,1] = 0.4997857776773573
CG5_a[3,2] = -0.0969984448371582
CG5_a[4,0] = 0.3552358559023322
CG5_a[4,1] = 0.2390958372307326
CG5_a[4,2] = 0.3918565724203246
CG5_a[4,3] = -0.1092979392113565

CG5_b = [0.1370831520630755, -0.0183698531564020, 0.7397813985370780, -0.1907142565505889, 0.3322195591068374]
CG5_c = [0.0, 0.8177227988124852, 0.3859740639032449, 0.3242290522866937, 0.8768903263420429]


nan3 = np.nan*np.zeros(3)

class estimated_float(float):
    #TODO add value checking
    def __new__(cls,value,cov = None,int_cov = None):
        if cov is None:
            cov = 0
        if int_cov is None:
            int_cov = 0
        return float.__new__(cls,value)
    def __init__(self,value,cov = None,int_cov = None):
        if cov is None:
            cov = 0
        if int_cov is None:
            int_cov = 0
        super().__init__()
        self.cov = cov
        self.int_cov = int_cov
        self.val = value


class estimated_nparray(np.ndarray):
    #TODO add value checking
    def __new__(cls,value,cov = None,int_cov = None):
        if cov is None:
            cov = np.zeros((value.size,value.size))
        if int_cov is None:
            int_cov = np.zeros(cov.shape)
        # print(cls)
        # print(value)
        # print(type(value))
        out = np.ndarray.__new__(cls,value.shape,buffer = value,dtype = value.dtype)
        # print(out)
        # print(type(out))
        return out

    def __init__(self,value,cov = None,int_cov = None):
        if cov is None:
            cov = np.zeros((value.size,value.size))
        if int_cov is None:
            int_cov = np.zeros(cov.shape)
        self.cov = np.copy(cov)
        self.int_cov = np.copy(int_cov)
        self.val = np.copy(value)
        super().__init__()

    def pull_indices(self,inds_mask,cov_missing_inds=[]):
        cov_missing_inds = np.array(cov_missing_inds)
        if cov_missing_inds.size > 0:
            cov_inds_mask = inds_mask.copy()
            cov_inds_mask = np.delete(cov_inds_mask,cov_missing_inds)
        else:
            cov_inds_mask = inds_mask
        return estimated_nparray(self.val[inds_mask],square_mat_sections(self.cov,cov_inds_mask),square_mat_sections(self.int_cov,cov_inds_mask))

    def set_indices(self,inds_mask,val,cov,int_cov,cov_missing_inds=[]):
        cov_missing_inds = np.array(cov_missing_inds)
        if cov_missing_inds.size > 0:
            cov_inds_mask = inds_mask.copy()
            cov_inds_mask = np.delete(cov_inds_mask,cov_missing_inds)
        else:
            cov_inds_mask = inds_mask
        self.val[inds_mask] = val
        self.cov = square_mat_refill(cov,cov_inds_mask,self.cov)
        self.int_cov = square_mat_refill(int_cov,cov_inds_mask,self.int_cov)

    def copy(self):
        return estimated_nparray(self.val.copy(),self.cov.copy(),self.int_cov.copy())




def square_mat_mask(mat,mask):
    tmp = mat[mask,:]
    return tmp[:,mask]

def square_mat_sections(mat,vals):
    tmp = mat[vals,:]
    return tmp[:,vals]

def square_mat_refill(mat,mask,mat_to_fill = None):
    tmp = np.zeros((np.size(mat,0),np.size(mask)))
    tmp[:,mask] = mat
    out = np.zeros((np.size(mask),np.size(mask)))
    out[mask,:] = tmp
    if mat_to_fill is not None:
        out[~mask,:] = mat_to_fill[~mask,:]
        out[:,~mask] = mat_to_fill[:,~mask]
    return out


# def pinv_LIcols(mat):
#     return scipy.linalg.solve(mat.T@mat,mat.T,assume_a='sym')
#
# def pinv_LIrows(mat):
#     return scipy.linalg.solve((mat@mat.T).T,mat,assume_a='sym').T

def limit(u,umax,offset = None):
    #umax should be positive!
    umax = np.fabs(umax)
    if offset is None:
        #bound u to fit within umax but remain parallel to u0
        if np.all(np.fabs(u) < umax): #CHANGED
            return u
        if isinstance(u,np.ndarray):
            ul = abs(u.flatten()/umax)
            ur = max(max(ul),1.0)
        else:
            ul = abs(u/umax)
            ur = max(ul,1.0)
        u = u/ur
        return u
    else:
        #bound u to fit within umax but so that u+offset remains parallel to u0. does NOT take off offset, just considers it when calculating the limit.
        if np.all(np.fabs(u-offset) < umax): #CHANGED
            return u
        if isinstance(u,np.ndarray):
            uln = u.flatten()/(offset-umax)
            ulp = u.flatten()/(offset+umax)
            ul = abs(u.flatten()/umax)
            ur = max(max(ul),1.0)
        else:
            ul = abs(u/umax)
            ur = max(ul,1.0)
        u = u/ur
        return u

def saturate(u,lim = 1.0):
    # u0 = np.copy(u)
    # if isinstance(lim,np.ndarray):
    # lim = lim.flatten()
    # ul = u.flatten().tolist()
    uout = np.minimum(lim,np.fabs(u))*np.sign(u)#uout = np.array([np.sign(ul[j])*min(lim[j],np.fabs(ul[j])) for j in len(ul)]).reshape(u0.shape)
    return uout
    # if np.all(np.fabs(u) < lim): #CHANGED
    #     return u#.reshape(u0.shape)
    # uout = np.array([np.sign(j)*min(lim,np.fabs(j)) for j in u.flatten().tolist()]).reshape(u0.shape)
    # return uout

#
# def limit_scalar(u,umax):
#     # u = float(u)
#     # umax = float(umax)
#     ul = abs(u/umax)
#     ur = max(ul,1.0)
#     u = u/ur
#     return u


def norm(v):
    """
    This function finds the magnitude of a vector.
    Input:
        v -- n x 1 np array representing a vector
    Output:
        v_norm -- double, magnitude of v
    Duplicated from sim_helpers--estimator and components on board satellite must stand on their own.
    """
    return np.linalg.norm(v)
    # v = np.copy(np.array(v))
    # if isinstance(v,np.matrix):
    #     v = v.A1
    # v = v.flatten().tolist()
    # v_norm = 0
    # for elt in v:
    #     v_norm += elt*elt
    # return math.sqrt(v_norm)

def normalize(v):
    """
    This function normalizes a vector.
    Input:
        v -- n x 1 np array representing a vector
    Output:
        v' -- n x 1 np array representing unit vector in same direction as v
    Duplicated from sim_helpers--estimator and components on board satellite must stand on their own.
    """
    #Avoid divide-by-0 if vector is already all 0s
    # v = np.copy(np.array(v))
    # if isinstance(v,np.matrix):
    #     sn = norm(v.A1)
    # elif isinstance(v,np.ndarray):
    #     sn = norm(v.flatten())
    # else:
    sn = norm(v)
    if sn == 0:
        return v
    return v/sn

def matrix_row_normalize(m):
    # m = np.squeeze(np.copy(np.array(m)))
    return m/np.expand_dims(matrix_row_norm(m),axis=1)
    # if len(m.shape) != 2:
    #     raise ValueError("not a 2D matrix")
    #
    # out = np.array([np.array(normalize(m[j,:])).flatten() for j in range(m.shape[0])])
    # return out

def matrix_row_norm(m):
    # m = np.squeeze(np.copy(np.array(m)))
    if len(m.shape) != 2:
        raise ValueError("not a 2D matrix")
    return np.linalg.norm(m, ord=2,axis=1)
    # out = np.array([np.array(norm(m[j,:])).flatten() for j in range(m.shape[0])])
    # return outß

def rot_list(rm_list,veclist,transpose = False):
    if not isinstance(rm_list,list):
        raise ValueError("rmlist must be list")
    if not np.all([j.shape == (3,3) for j in rm_list]):
        raise ValueError("rmlist elements must be 3x3 matrices")
    if veclist.shape[-1] != 3:
        raise ValueError("incorrectly shaped veclist")
    if len(rm_list) != veclist.shape[0]:
        raise ValueError("lists of different lengths")
    if transpose:
        return np.vstack([(rm_list[j].T@veclist[j,:]) for j in range(len(rm_list))])
    else:
        return np.vstack([(rm_list[j]@veclist[j,:]) for j in range(len(rm_list))])

def stateAdd(state,dstate,quat0ind,dquatmode):
    if len(state.shape) != 1:
        raise ValueError("state must be a array with only 1 non-singleton dimension")
    if len(dstate.shape) != 1:
        raise ValueError("dstate must be a array with only 1 non-singleton dimension")
    if dstate.shape[0] != state.shape[0] - 1:
        raise ValueError("dstate must have length equal to one less than the length of the state")
    out = np.zeros(state.shape)
    if quat0ind>0:
        out[0:quat0ind] = state[0:quad0ind] + dstate[0:quad0ind]
    out[quat0ind:quat0ind+4] = quat_mult(stateout[quat0ind:quat0ind+4],vec3_to_quat(dstate[quat0ind:quat0ind+3],dquatmode))
    out[quat0ind+4:] = state[quat0ind+4:] + dstate[quat0ind+3:]
    return out

def vec_norm_jac(v,dv=None):
    # n = norm(v)
    l = v.size
    normv = norm(v)
    if normv > num_eps:
        dndv = v/normv
    else:
        dndv = np.ones(l)
    if dv is None:
        return dndv
    return dv@dndv


def vec_norm_hess(v,dv=None,ddv=None):
    # n = norm(v)
    l = v.size
    normv = norm(v)
    dndv = v/normv
    ddndvdv = np.eye(l)/normv - np.outer(v,v)/normv**3.0# tmp + np.transpose(tmp,(2,0,1)) + np.transpose(tmp,(1,2,0))
    if dv is None:
        if ddv is not None:
            raise ValueError('if jacobian of v is none, hessian must also be none')
        return ddndvdv
    else:
        if ddv is None:
            raise ValueError('if jacobian of v is provided, hessian must also be provided')
        return dv@ddndvdv@dv.T + ddv@dndv


def normed_vec_jac(v,dv=None):
    # n = v/norm(v)
    l = v.size
    normv = norm(v)
    if normv>num_eps:
        dndv = np.eye(l)/normv - np.outer(v,v)/normv**3
    else:
        dndv = np.eye(l)
    if dv is None:
        return dndv
    return dv@dndv

def normed_vec_hess(v,dv=None,ddv=None):
    # n = v/norm(v)
    #dndv = np.eye(l)/normv - np.outer(v,v)/normv**3
    l = v.size
    normv = norm(v)
    dndv = np.eye(l)/normv - np.outer(v,v)/normv**3
    # ddndvdv =  np.eye(l)/normv - np.outer(v,v)/normv**3
    tmp = -np.multiply.outer(dndv,v/normv**2)
    ddndvdv = tmp + np.transpose(tmp,(2,0,1)) + np.transpose(tmp,(1,2,0))
    if dv is None:
        if ddv is not None:
            raise ValueError('if jacobian of v is none, hessian must also be none')
        return ddndvdv
    else:
        if ddv is None:
            raise ValueError('if jacobian of v is provided, hessian must also be provided')
        return np.tensordot(dv,dv@ddndvdv,([1],[0])) + ddv@dndv


def rotz(ang):
    """
    returns 3x3 matrix corresponding to *ang* radian rotation about a z-prop_axis
    """
    c = math.cos(ang)
    s = math.sin(ang)
    return np.block([[c,-s,0],[s,c,0],[0,0,1]])

def rotz_q_p_pi2(ang):
    """
    returns 3x3 matrix corresponding to *pi/2 + ang* radian rotation about a z-prop_axis
    """
    c = -math.sin(ang)
    s = math.cos(ang)
    return np.block([[c,-s,0],[s,c,0],[0,0,1]])


def rotx(ang):
    """
    returns 3x3 matrix corresponding to *ang* radian rotation about an x-prop_axis
    """
    c = math.cos(ang)
    s = math.sin(ang)
    return np.block([[1,0,0],[0,c,-s],[0,s,c]])


def rotx_pi2_m_q(ang):
    """
    returns 3x3 matrix corresponding to *pi/2 - ang* radian rotation about an x-prop_axis
    """
    c = math.sin(ang)
    s = math.cos(ang)
    return np.block([[1,0,0],[0,c,-s],[0,s,c]])



def roty(ang):
    """
    returns 3x3 matrix corresponding to *ang* radian rotation about a y-prop_axis
    """
    c = math.cos(ang)
    s = math.sin(ang)
    return np.block([[c,0,s],[0,1,0],[-s,0,c]])

def trig_sum_solve_or_min(p,a,b):
    #takes 0=p + a*cos(x) +b*sin(x).
    #returns x that solves this or at least minimizes error.
    #returns 2 values in a list. for minimization case, they are the same value

    #overall
    #0 = p + np.sign(a)*math.sqrt(a**2+b**2)*math.cos(x+math.atan(-b/a))

    #deal with edge cases first
    ais0 = (abs(a) < num_eps)
    bis0 = (abs(b) < num_eps)
    pis0 = (abs(p) < num_eps)
    print(ais0,bis0,pis0)
    if ais0 and bis0 and pis0: #why is this case even happening. can be removed a&b is  0 case takes care of it.
        return [0,math.pi]
    if ais0 and bis0:
        return [0,math.pi]
    if bis0 and pis0:
        #0 = a*math.cos(x)
        return [0.5*math.pi,-0.5*math.pi] #expression should still work fine, but this is shortcut
    if ais0 and pis0: #can be removed, ais0 case takes care of it
        #0= b*sin(x).
        return [0,math.pi]
    if ais0:
        # 0=p+b*sin(x)
        #sin(x) = -p/b
        s = math.asin(-p/b)
        return [s,math.pi-s]
    #p=0? no problem. always solvable, expression works fine
    #b=0? no problem, expressions as is work fine


    if abs(p*np.sign(a)/math.sqrt(a**2+b**2))>1.0:
        #must minimize
        x = -math.atan(-b/a)
        #x+math.atan(-b/a) = 0 or pi.
        #choose so that sgn(p) is opposite sign of sgn(a)*math.sqrt(a**2+b**2)*math.cos(x+math.atan(-b/a))
        if p*a > 0.0:
            x += math.pi
        return [x,x]

    #basecase
    #0 = p + np.sign(a)*math.sqrt(a**2+b**2)*math.cos(x+math.atan(-b/a))
    #-p*np.sign(a)/math.sqrt(a**2+b**2)=math.cos(x+math.atan(-b/a))
    #x+math.atan(-b/a) = +/- math.acos(-p*np.sign(a)/math.sqrt(a**2+b**2))
    s = math.atan(-b/a)
    t = math.acos(-p*np.sign(a)/math.sqrt(a**2+b**2))
    return [-s+t,-s-t]


def skewsym(v):
    """
    This function finds the skew symmetric matrix corresponding to a 3 x 1 vector v, such that skew(v)*v2 = v1 x v2 where x is np.cross product
    Inputs:
        v -- (3,) size np array
    Outputs:
        skewsym -- 3 x 3 size np array representing skew symmetric matrix of v
    Duplicated from sim_helpers--estimator and components on board satellite must stand on their own.
    """
    #print("\n",v)
    # v = np.array(v)
    # v = v.flatten().tolist()
    #print(v)
    # v = list(v)
    #print(v)
    #print(v)
    #print(v[0])
    #print(v[1])
    #print(v[2])
    return np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0 ]])


def random_n_unit_vec(n):
    return normalize(np.array([np.random.normal() for j in range(n)]))

# d/dq0 qv/(q0) = -qv/q0^2 = -cly/q0
# d/dqv qv/(q0)  = I/q0
#
# d/dq0  -qv/q0^2 = 2*qv/q0^3 = 2*cly/q0^2
# d/dq0  I/q0 = I*-1/(q0)^2
#
# d/dqi  -qv/q0^2 = (-ei/q0^2)
# d/dqi  I/q0 = 0

def dimcheck1D(arr,axis,dim):
    """swaps array axes to match a dimension along one axis, if possible
    Inputs:
        arr -- array to check
        axis -- index of axis to ceck
        dim -- length axis should be
    Returns:
        arr -- adjusted array, if possible. original array if not
    """
    # arr = np.array(arr)
    shp = arr.shape
    if (shp[axis] != dim):
        if np.any(np.equal(shp, dim)):
            whr = np.array(np.where(np.equal(shp, dim)))
            #print(whr)
            arr = np.swapaxes(arr,axis,whr[0][0])
    return arr


# def gcd_sample_time(block_sample_times, round_exp=4):
#     """
#     This function finds the desired common sample time by finding the GCD of a list of floats corresponding to sample
#     times. Useful for finding a common sample time for TVLQR and the trajectory planner so we know what timestep to
#     calculate the magnetic field at, etc.
#     Inputs:
#         block_sample_times -- list of floats
#         round_exp -- number of places to round to
#     Outputs:
#         sim_sample_time -- max(1e-round_exp, appropriate sim sample time)
#     """
#     min_sample_time = 100000
#     #Find number to round to
#     round_num = 10**-round_exp
#     #Find min GCD of 2 sample times
#     for i in range(len(block_sample_times)):
#         for j in range(i, len(block_sample_times)):
#             if gcd(block_sample_times[i], block_sample_times[j], round_num) < min_sample_time:
#                 min_sample_time = gcd(block_sample_times[i], block_sample_times[j], round_num)
#     #Return the rounded simulation sample time
#     return round(min_sample_time, round_exp)

def reverse_clip_neps(x):
    return np.sign(x)*max(abs(x),num_eps)



def mydot(v,u):
    return np.ndarray.item(v.T@u)



@unique
class PointingGoalVectorMode(Enum):
    """
    This enum exists to provide a more intuitive representation of pointing goal
    modes, and to reduce errors caused by just using numbers in the code rather
    than var names.

    These vectors are all in the ECI frame.
    """
    #These don't require additional vector or position info.

    NO_GOAL = -1 #satellite can do what it wants--useful for before/after a pass
    NADIR = 0
    ZENITH = 1
    RAM = 2
    ANTI_RAM = 3
    POSITIVE_ORBIT_NORMAL = 4
    NEGATIVE_ORBIT_NORMAL = 5
    TOWARDS_SUN = 6
    AWAY_FROM_SUN = 7
    #These require specifying additional vectors and/or positions.
    PROVIDED_ECI_VECTOR = 8 #requires vector
    PROVIDED_ECI_VECTOR_POSITION = 9 #requires position
    PROVIDED_ECEF_GROUND_TARGET = 10 #requires ground pos, points to target even if Earth in the way
    PROVIDED_MRP = 11
    PROVIDED_MRP_WISNIEWSKI = 12

vecNeededlist = [PointingGoalVectorMode.PROVIDED_ECI_VECTOR,PointingGoalVectorMode.PROVIDED_ECI_VECTOR_POSITION,PointingGoalVectorMode.PROVIDED_ECEF_GROUND_TARGET,PointingGoalVectorMode.PROVIDED_MRP,PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI]
fullQuatSpecifiedList = [PointingGoalVectorMode.PROVIDED_MRP,PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI]


@unique
class GovernorMode(Enum):
    """
    This enum exists to provide a more intuitive representation of governor modes,
    and to reduce errors caused by just using numbers in the code rather than var names.
    """
    #Just control modes (as specified by the governor)
    #May want to add the simple magnetic controller here
    NO_CONTROL = 0
    SIMPLE_BDOT = 1
    BDOT_WITH_EKF = 2
    SIMPLE_MAG_PD = 3
    PLAN_AND_TRACK_LQR = 4
    PLAN_AND_TRACK_MPC = 5
    RW_PID = 6
    TRAJ_MAG_PD = 7
    PLAN_OPEN_LOOP = 8
    RWBDOT_WITH_EKF = 9
    LOVERA_MAG_PD = 10
    LOVERA_MAG_PD_QUATSET_B = 11
    LOVERA_MAG_PD_QUATSET_LYAP = 12
    LOVERA_MAG_PD_QUATSET_ANG = 13
    LOVERA_MAG_PD_QUATSET_Bnow = 14
    WISNIEWSKI_SLIDING = 15
    WISNIEWSKI_SLIDING_QUATSET_B = 16
    WISNIEWSKI_SLIDING_QUATSET_ANG = 17
    WISNIEWSKI_SLIDING_QUATSET_LYAP = 18
    WIE_MAGIC_PD = 19
    WIE_MAGIC_PD_QUATSET_ANG = 20
    WIE_MAGIC_PD_QUATSET_LYAP = 21
    WISNIEWSKI_SLIDING_QUATSET_MINS = 22
    LOVERA4_MAG_PD = 23
    LOVERA4_MAG_PD_QUATSET_B = 24
    LOVERA4_MAG_PD_QUATSET_LYAP = 25
    LOVERA4_MAG_PD_QUATSET_ANG = 26
    LOVERA4_MAG_PD_QUATSET_Bnow = 27
    WIE_MAGIC_PD_QUATSET_LYAPR = 28
    WISNIEWSKI_SLIDING_QUATSET_LYAPR = 29
    LOVERA_MAG_PD_QUATSET_LYAPR = 30
    WISNIEWSKI_TWISTING = 31
    WISNIEWSKI_TWISTING_QUATSET_B = 32
    WISNIEWSKI_TWISTING_QUATSET_ANG = 33
    WISNIEWSKI_TWISTING_QUATSET_LYAP = 34
    WISNIEWSKI_TWISTING_QUATSET_MINS = 35
    WISNIEWSKI_TWISTING_QUATSET_LYAPR = 36
    WISNIEWSKI_TWISTING2 = 37
    WISNIEWSKI_TWISTING2_QUATSET_B = 38
    WISNIEWSKI_TWISTING2_QUATSET_ANG = 39
    WISNIEWSKI_TWISTING2_QUATSET_LYAP = 40
    WISNIEWSKI_TWISTING2_QUATSET_MINS = 41
    WISNIEWSKI_TWISTING2_QUATSET_LYAPR = 42
    WISNIEWSKI_SLIDING_MAGIC = 43
    WISNIEWSKI_TWISTING_MAGIC = 44
    WISNIEWSKI_TWISTING2_MAGIC = 45
    WISNIEWSKI_SLIDING_QUATSET_BS = 46
    WISNIEWSKI_TWISTING_QUATSET_BS = 47
    MAGIC_BDOT_WITH_EKF = 48
    MTQ_W_RW_PD = 49
    MTQ_W_RW_PD_MINE = 50



QuaternionModeList = [GovernorMode.RW_PID,GovernorMode.TRAJ_MAG_PD,GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.LOVERA_MAG_PD,GovernorMode.LOVERA4_MAG_PD,GovernorMode.SIMPLE_MAG_PD,GovernorMode.WISNIEWSKI_SLIDING,GovernorMode.WIE_MAGIC_PD,GovernorMode.MTQ_W_RW_PD,GovernorMode.MTQ_W_RW_PD_MINE]
PlannerModeList = [GovernorMode.PLAN_AND_TRACK_LQR,GovernorMode.PLAN_AND_TRACK_MPC,GovernorMode.PLAN_OPEN_LOOP,GovernorMode.TRAJ_MAG_PD]
