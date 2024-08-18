import numpy as np
from sat_ADCS_orbit import *
from sat_ADCS_helpers import *
from sat_ADCS_estimation import *
from sat_ADCS_satellite import *
from .control_mode import *
from .trajectory_mpc import *
from scipy.linalg import block_diag
from scipy.stats import kstest
import math
import numdifftools as nd
from scipy.integrate import odeint, solve_ivp, RK45
# from ascii_graph import Pyasciigraph
# from ascii_graph.colors import *
# from asciichartpy import plot


def test_the_jac():
    sat = create_GPS_BC_sat(real=True,rand=False,care_about_eclipse = False,use_dipole = False)
    # bc_q0 = normalize(np.array([0,0,0,1]))
    # bc_w0 = (np.pi/180.0)*random_n_unit_vec(3)#

    mpc_dt = 1
    mpc_ang_limit = 10
    mpc_angwt_low = 1e4
    mpc_angwt_high = 1e6
    mpc_avwt = 1
    mpc_extrawt = 0
    mpc_uwt_from_plan = 1e-4
    mpc_uwt_from_prev = 0
    mpc_lqrwt_mult = 0.5
    mpc_extra_tests = 0
    mpc_tol = 1e-4
    mpc_gain_info = [mpc_dt,mpc_ang_limit,mpc_angwt_low,mpc_angwt_high,mpc_avwt,mpc_extrawt,mpc_uwt_from_plan,mpc_uwt_from_prev,mpc_lqrwt_mult,mpc_extra_tests,mpc_tol]#[1,10,100,1e6,1,0,1e-6,0]
    # tests = tests_baseline[1:3] + tests_disturbed[1:3] + tests_ctrl[1:3] + tests_genctrl[1:3] + tests_cubesat
    # breakpoint()

    mpc = TrajectoryMPC(mpc_gain_info,sat)


    plan_w = 0.05*random_n_unit_vec(3)
    plan_q = random_n_unit_vec(4)
    next_plan_state = np.concatenate([plan_w,plan_q])

    plan_control = 2*np.random.rand(3) - 1
    control0 = 2*np.random.rand(3) - 1

    t0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    state = np.concatenate([w0,q0])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    osp1 = os.orbit_rk4(1, J2_on=True, rk4_on=True)
    # R = os.R
    # V = os.V
    # B = B_ECI
    # S = os.S
    # rho = os.rho
    # rmat_ECI2B = rot_mat(q0).T
    # R_B = rmat_ECI2B@R
    # B_B = rmat_ECI2B@B
    # S_B = rmat_ECI2B@S
    # V_B = rmat_ECI2B@V
    # dR_B__dq = drotmatTvecdq(q0,R)
    # dB_B__dq = drotmatTvecdq(q0,B)
    # dV_B__dq = drotmatTvecdq(q0,V)
    # dS_B__dq = drotmatTvecdq(q0,S)
    # ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    # ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    # ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    # ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    # vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    # assert np.all(np.isclose(ax/3*(t0),magic.torque(t0,sat,x0,vecs)))
    next_plan_gain = np.random.rand(3,6)
    wt = next_plan_gain.T@next_plan_gain*mpc_lqrwt_mult + scipy.linalg.block_diag(np.eye(3)*mpc_avwt,np.eye(3)*mpc_angwt_low, np.eye(sat.state_len-7)*mpc_extrawt)
    ctrlwt1 = np.eye(sat.control_len)*mpc_uwt_from_plan
    ctrlwt2 = np.eye(sat.control_len)*mpc_uwt_from_prev
    next_state_func = lambda u,os = os,osp1=osp1: sat.rk4(state[:sat.state_len],u,1,os,osp1,verbose=False,quat_as_vec = True,save_info = False)
    func = lambda u, next_state_func=next_state_func,wt=wt, next_plan_state = next_plan_state: mpc.scoring_func(u,next_state_func,wt,ctrlwt1,ctrlwt2,next_plan_state,plan_control)
    next_state_jac_func = lambda u,os = os,osp1=osp1: sat.rk4Jacobians(state[:sat.state_len],u,1,os,osp1,quat_as_vec = True)
    func_jac = lambda u, next_state_func=next_state_func,wt=wt, next_plan_state = next_plan_state: mpc.scoring_func_du(u,next_state_func,next_state_jac_func,wt,ctrlwt1,ctrlwt2,next_plan_state,plan_control).reshape(sat.control_len,)
    next_state_hess_func = lambda u,os = os,osp1=osp1: sat.rk4_u_Hessians(state[:sat.state_len],u,1,os,osp1)
    func_hess = lambda u, next_state_func=next_state_func,wt=wt, next_plan_state = next_plan_state: mpc.scoring_func_dudu(u,next_state_func,next_state_jac_func,next_state_hess_func,wt,ctrlwt1,ctrlwt2,next_plan_state,plan_control)


    Jfun = np.array(nd.Jacobian(func)(control0.flatten().tolist())).reshape(sat.control_len,)
    print(Jfun.T)
    print(func_jac(control0).T)
    assert np.allclose(Jfun, func_jac(control0))

    Hfun = np.array(nd.Hessian(func)(control0).flatten().tolist())
    assert np.allclose(Hfun,func_hess(control0))
