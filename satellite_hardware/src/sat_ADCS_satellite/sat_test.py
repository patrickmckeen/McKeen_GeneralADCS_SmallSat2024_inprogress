from .satellite import Satellite
import numpy as np
from sat_ADCS_orbit import *
from sat_ADCS_helpers import *
from .actuators import *
from .sensors import *
from .disturbances import *
from scipy.linalg import block_diag
from scipy.stats import kstest
import math
import numdifftools as nd
from scipy.integrate import odeint, solve_ivp, RK45
# from ascii_graph import Pyasciigraph
# from ascii_graph.colors import *
from asciichartpy import plot

"""
NEXT TESTS
------


satellite
----------
xxxxxxxbias drift on dynamics
more update_J tests--especially with COM, etc
xxxxxxxmake sure dynamics/angular momentum with COM/off center, etc is correct
xxxxxxxmore RW dynamics/angular momentum tests
new RK4/CG5
noiselss act_torque/dynamics/rk4.

control bounds
apply control bounds
polling non-attitude sensor values, jacobians, covariances, etc
disturbance on/off tests
matching to estimate
scaling of sensors
varying ways to pull out sensor jacobians for bias etc
overall disturbance torque

disturbances
-----------
srp eclipse issues
jacobians w.r.t estimated values
estimated thigngs

actuators
---------
input length
RW sensor things--done?
maximum value warnings
estimated thigngs
input length

sensors
----------
sun sensor solar degratdation
eclipse effects of sunsensors
orbitRV jacobian tests on non GPS
estimated thigngs

"""

def test_J():
    sat = Satellite()
    sat.update_J(np.diagflat([0.1,100,5]))
    assert np.all(sat.J == np.array([[0.1,0,0],[0,100,0],[0,0,5]]))
    assert np.all(sat.invJ == np.array([[10,0,0],[0,0.01,0],[0,0,0.2]]))
    assert np.all(sat.J_noRW == np.array([[0.1,0,0],[0,100,0],[0,0,5]]))
    assert np.all(sat.invJ_noRW == np.array([[10,0,0],[0,0.01,0],[0,0,0.2]]))

def test_J__w_RW():
    Js = [0.001,0.002,0.5]
    acts = [RW(unitvecs[j],0,0.1,Js[j],0,0.1,0,use_noise=False) for j in range(3)]
    sat = Satellite(actuators = acts)
    sat.update_J(np.diagflat([0.1,100,5]))
    assert np.all(sat.J == np.array([[0.1,0,0],[0,100,0],[0,0,5]]))
    assert np.all(sat.invJ == np.array([[10,0,0],[0,0.01,0],[0,0,0.2]]))
    assert np.all(sat.J_noRW == np.array([[0.099,0,0],[0,99.998,0],[0,0,4.5]]))
    assert np.all(sat.invJ_noRW == np.array([[1/0.099,0,0],[0,1/99.998,0],[0,0,2/9]]))

def test_COM_J():
    JA = np.eye(3)
    JB = np.eye(3) + 1*(np.eye(3)*4 - 4*np.outer(unitvecs[0],unitvecs[0]))
    m = 2
    COM = unitvecs[0]
    sat = Satellite(COM = COM, mass = m, J = JA+JB)
    assert np.all(sat.J_given == JA+JB)
    assert np.all(sat.J == 2*np.diagflat([1,2,2]))

def test_update_RWhs_from_state():
    maxt = [0.01,0.05,0.02]
    rwj = [0.001,0.002,0.5]
    maxh = [0.1,0.1,0.1]
    h0 = [0.1,0.0,0.0]
    bias = np.array([-0.001,0.05,0])
    acts = [RW(unitvecs[j],0,maxt[j],rwj[j],h0[j],0.1,0,has_bias = True, bias = bias[j],use_noise = False,bias_std_rate = 0.3) for j in range(3)]
    sat = Satellite(actuators = acts)
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    state = np.concatenate([0.01*unitvecs[0],zeroquat,h0])
    xd = sat.dynamics(state,np.array([0.021,-0.05,0]),os)

    new_h = [np.random.uniform(-1,1) for j in range(3)]
    nh = np.copy(new_h)
    sat.update_RWhs(new_h)
    assert np.all(sat.RWhs() == nh)
    assert np.all([sat.actuators[j].momentum for j in sat.momentum_inds] == nh)

    new_h = [np.random.uniform(-1,1) for j in range(3)]
    nh = np.copy(new_h)
    sat.update_RWhs(np.concatenate([state[0:7],new_h]))
    assert np.all(sat.RWhs() == nh)
    assert np.all([sat.actuators[j].momentum for j in sat.momentum_inds] == nh)

def test__srp():
    es = [0,0.1,0.3]
    ed = [0.5,0.2,0.1]
    ea = [0.5,0.7,0.6]
    ar = [0.1,0.03,10]
    normals = unitvecs
    cents = [np.array([1,0.2,0]),np.array([-0.05,0.1,0.3]),np.array([0.25,-0.01,-0.7])]
    dist = [SRP_Disturbance([[j,ar[j],cents[j],unitvecs[j],ea[j],ed[j],es[j]] for j in range(3)])]
    sat = Satellite(disturbances=dist)
    assert np.all(sat.disturbances[0].eta_s == [0,0.1,0.3])
    assert np.all(sat.disturbances[0].eta_d == [0.5,0.2,0.1])
    assert np.all(sat.disturbances[0].eta_a == [0.5,0.7,0.6])
    assert np.all(sat.disturbances[0].areas == [0.1,0.03,10])
    assert np.all([sat.disturbances[0].normals[j] == unitvecs[j] for j in range(3)])
    assert np.all([sat.disturbances[0].centroids[j] == cents[j] for j in range(3)])

def test__drag():
    drag = Drag_Disturbance([[0,0.1,np.array([1,0.2,0]),unitvecs[0],2],[1,0.03,np.array([-0.05,0.1,0.3]),unitvecs[1],0.1],[2,10,np.array([0.25,-0.01,-0.7]),unitvecs[2],0.3]])
    sat = Satellite(disturbances = [drag])
    assert np.all(sat.disturbances[0].areas == [0.1,0.03,10])
    assert np.all([sat.disturbances[0].centroids[j] == [np.array([1,0.2,0]),np.array([-0.05,0.1,0.3]),np.array([0.25,-0.01,-0.7])][j] for j in range(3)])
    assert np.all([sat.disturbances[0].normals[j] == [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])][j] for j in range(3)])
    assert np.all(sat.disturbances[0].CDs == [2,0.1,0.3])

def test__prop():
    prop = Prop_Disturbance([np.array([1,2,4])])
    sat = Satellite(disturbances = [prop])
    # sat.COM = np.array([-1,1,0.5])
    # sat.update_prop_params(0.4,np.array([3,1,5]),np.array([3,2,1]))
    assert np.all(sat.disturbances[0].main_param == np.array([1,2,4]))
    assert np.all(sat.disturbances[0].torque(sat,[]) == np.array([1,2,4]))

def test__gendist():
    gen = General_Disturbance([np.array([1,2,4])])
    sat = Satellite(disturbances = [gen])
    # sat.COM = np.array([-1,1,0.5])
    # sat.update_prop_params(0.4,np.array([3,1,5]),np.array([3,2,1]))
    assert np.all(sat.disturbances[0].main_param == np.array([1,2,4]))
    assert np.all(sat.disturbances[0].torque(sat,[]) == np.array([1,2,4]))

def test__resdipole():
    dist = Dipole_Disturbance([np.array([0.1,-0.1,0.5]),1],True,0.1)
    sat = Satellite(disturbances=[dist])
    assert np.all(sat.disturbances[0].main_param == np.array([0.1,-0.1,0.5]))
    assert np.all(sat.disturbances[0].std == 0.1*np.eye(3))
    assert np.all(sat.disturbances[0].mag_max == 1)

def test_dynamics_plain():
    sat = Satellite()
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([]),os)
    assert np.all(np.array([0,0,0,0,0.005,0,0]) == xd)
    xd = sat.dynamics(np.array([0.01,0,0,0,0,1,0]),np.array([]),os)
    assert np.all(np.array([0,0,0,0,0,0,-0.005]) == xd)
    sat = Satellite(J=np.diagflat([2,3,5]))
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([]),os)
    assert np.all(np.array([0,0,0,0,0.005,0,0]) == xd)
    qJ = random_n_unit_vec(4)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    J_ECI = R@J_body@R.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = R@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    exp_wd = -R.T@np.linalg.inv(J_ECI)@np.cross(w_ECI,H_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    sat = Satellite(J=J_body)
    state = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    xd = sat.dynamics(state,np.array([]),os)
    print(xd,np.concatenate([exp_wd,exp_qd]))
    print(np.concatenate([exp_wd,exp_qd]) - xd)
    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),xd))

def test_dynamics_MTQ():
    mtqs = [MTQ(j,0,1) for j in unitvecs]
    sat = Satellite(actuators = [mtqs[0]])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([1]),os)
    assert np.all(np.array([0,0,0,0,0.005,0,0]) == xd)
    sat = Satellite(actuators = [mtqs[1]])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([1]),os)
    assert np.all(np.array([0,0,-1e-5,0,0.005,0,0]) == xd)
    sat = Satellite(actuators = [mtqs[2]])
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([1]),os)
    assert np.all(np.array([0,1e-5,0,0,0.005,0,0]) == xd)
    sat = Satellite(actuators = mtqs)
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([1,0,0]),os)
    assert np.all(np.array([0,0,0,0,0.005,0,0]) == xd)
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([0,1,0]),os)
    assert np.all(np.array([0,0,-1e-5,0,0.005,0,0]) == xd)
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([0,0,1]),os)
    assert np.all(np.array([0,1e-5,0,0,0.005,0,0]) == xd)
    qJ = random_n_unit_vec(4)
    m0 = random_n_unit_vec(3)
    B_ECI = 1e-5*random_n_unit_vec(3)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    J_ECI = R@J_body@R.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = R@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    m_ECI = R@m0
    torq_ECI = np.cross(B_ECI,m_ECI)
    exp_wd = -R.T@np.linalg.inv(J_ECI)@(np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    sat = Satellite(J=J_body,actuators=mtqs)
    state = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    xd = sat.dynamics(state,m0,os)
    print(xd,np.concatenate([exp_wd,exp_qd]))
    print(np.concatenate([exp_wd,exp_qd]) - xd)
    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),xd))

def test_MTQ_torque():
    mtqs = [MTQ(j,0,1) for j in unitvecs]
    sat = Satellite()
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    x0 = np.array([0.01,0,0,1,0,0,0])
    q0 = x0[3:7]
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho}
    for i in range(3):
        assert np.all(mtqs[i].torque(0,sat,x0,vecs) == np.zeros(3))
        assert np.all(mtqs[i].torque(1,sat,x0,vecs) == 1e-5*np.cross(unitvecs[i],unitvecs[0]))
    qJ = random_n_unit_vec(4)
    m0 = random_n_unit_vec(3)
    B_ECI = 1e-5*random_n_unit_vec(3)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    J_ECI = R@J_body@R.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = R@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    m_ECI = R@m0
    torq_ECI = np.cross(B_ECI,m_ECI)
    exp_wd = -R.T@np.linalg.inv(J_ECI)@(np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    sat = Satellite(J=J_body,actuators=mtqs)
    state = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)

    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho}
    exp_torq = [np.cross(i,B_B) for i in unitvecs]
    for i in range(3):
        assert np.all(mtqs[i].torque(1,sat,x0,vecs) == exp_torq[i])

def test_MTQ_setup():
    ax = random_n_unit_vec(3)*3
    max_moment = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    mtq = MTQ(ax,std,max_moment,has_bias = True, bias = bias,use_noise = True,bias_std_rate = bsr)
    assert np.all(np.isclose(ax/3,mtq.axis))
    assert np.all(bias==mtq.bias)
    assert mtq.has_bias
    assert mtq.use_noise
    assert ~mtq.has_momentum
    assert np.isnan(mtq.J)
    assert mtq.max == max_moment
    assert np.isnan(mtq.momentum)
    assert np.isnan(mtq.max_h)
    assert mtq.noise_settings == std
    assert mtq.std == std
    assert mtq.bias_std_rate == bsr
    assert np.isnan(mtq.momentum_sens_noise_std)
    # self.noise_model = noise_model TODO--how to check....

def test_MTQ_torque_etc_clean():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    mtq = MTQ(ax,std,max_moment,has_bias = False,use_noise = False)

    m0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[mtq])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose(np.cross(ax/3*(m0),B_B),mtq.torque(m0,sat,x0,vecs)))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    ufun = lambda c: mtq.torque(c,sat,x0,vecs)
    xfun = lambda c: mtq.torque(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    hfun = lambda c: mtq.torque(m0,sat,x0,vecs)
    bfun = lambda c: MTQ(ax,std,max_moment,has_bias = False,use_noise = False).torque(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = False,use_noise = False)]),x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(m0)).T
    Jbfun = np.array(nd.Jacobian(bfun)(20000)).T
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T

    assert np.allclose(Jxfun, mtq.dtorq__dbasestate(m0,sat,x0,vecs))
    assert np.allclose(Jufun, mtq.dtorq__du(m0,sat,x0,vecs))
    assert np.allclose(Jbfun, mtq.dtorq__dbias(m0,sat,x0,vecs))
    assert np.allclose(Jhfun, mtq.dtorq__dh(m0,sat,x0,vecs))


    for j in unitvecs:
        fun_hj = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = False,use_noise = False).torque(c[0],Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = False,use_noise = False)]),np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]),vecsxfun(np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]))),j).item()

        ufunjju = lambda c: np.dot(mtq.dtorq__du(c,sat,x0,vecs),j).item()
        ufunjjb = lambda c: np.dot(mtq.dtorq__dbias(c,sat,x0,vecs),j)
        ufunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(c,sat,x0,vecs),j)
        ufunjjh = lambda c: np.dot(mtq.dtorq__dh(c,sat,x0,vecs),j)

        xfunjju = lambda c: np.dot(mtq.dtorq__du(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjb = lambda c: np.dot(mtq.dtorq__dbias(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjh = lambda c: np.dot(mtq.dtorq__dh(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)

        hfunjju = lambda c: np.dot(mtq.dtorq__du(m0,sat,x0,vecs),j).item()
        hfunjjb = lambda c: np.dot(mtq.dtorq__dbias(m0,sat,x0,vecs),j)
        hfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(m0,sat,x0,vecs),j)
        hfunjjh = lambda c: np.dot(mtq.dtorq__dh(m0,sat,x0,vecs),j)
        #
        bfunjju = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = False, bias = c,use_noise = False).dtorq__du(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = False, bias = c,use_noise = False)]),x0,vecs),j).item()
        # bfunjjb = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbias(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j).item()
        bfunjjx = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = False, bias = c,use_noise = False).dtorq__dbasestate(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = False, bias = c,use_noise = False)]),x0,vecs),j)
        # bfunjjh = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dh(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        # Jxfunjjb = np.aray(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        # Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , np.dot( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jxfunjjb , np.dot( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjx , np.dot( mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jxfunjjh , np.dot( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) , j ))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(m0))
        # Jufunjjb = np.array(nd.Jacobian(ufunjjb)(m0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(m0))
        # Jufunjjh = np.array(nd.Jacobian(ufunjjh)(m0))
        assert np.allclose( Jufunjju , np.dot( mtq.ddtorq__dudu(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jufunjjb , np.dot( mtq.ddtorq__dudbias(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjx.T , np.dot( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jufunjjh , np.dot( mtq.ddtorq__dudh(m0,sat,x0,vecs) , j ))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(20000))
        # Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(bias))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(20000))
        # # Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(bias))
        assert np.allclose( Jbfunjju , np.dot( mtq.ddtorq__dudbias(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jbfunjjb , np.dot( mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjx.T , np.dot( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jbfunjjh , np.dot( mtq.ddtorq__dbiasdh(m0,sat,x0,vecs) , j ))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
        # Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(500.2))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))
        # Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(500.2))
        assert np.allclose( Jhfunjju , np.dot( mtq.ddtorq__dudh(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jhfunjjb , np.dot( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjx , np.dot( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jhfunjjh , np.dot( mtq.ddtorq__dhdh(m0,sat,x0,vecs) , j ))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[m0],x0,[500.2]]).flatten().tolist()))
        Hguess = np.block([[mtq.ddtorq__dudu(m0,sat,x0,vecs)@j,mtq.ddtorq__dudbias(m0,sat,x0,vecs)@j,mtq.ddtorq__dudbasestate(m0,sat,x0,vecs)@j,mtq.ddtorq__dudh(m0,sat,x0,vecs)@j],\
                            [(mtq.ddtorq__dudbias(m0,sat,x0,vecs)@j).T,mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs)@j,mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs)@j,mtq.ddtorq__dbiasdh(m0,sat,x0,vecs)@j],\
                            [(mtq.ddtorq__dudbasestate(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs)@j).T,mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs)@j,mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs)@j],\
                            [(mtq.ddtorq__dudh(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbiasdh(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs)@j).T,mtq.ddtorq__dhdh(m0,sat,x0,vecs)@j]    ])

        assert np.allclose(Hfun[8,:],0)
        assert np.allclose(Hfun[:,8],0)
        assert np.allclose(Hfun[0:8,0:8],Hguess)


    assert np.all(np.isclose( mtq.dtorq__du(m0,sat,x0,vecs) , np.cross(ax/3,B_B) ))
    assert np.all(np.isclose( mtq.dtorq__dbias(m0,sat,x0,vecs) , np.cross(ax/3,B_B) ))
    assert np.all(np.isclose( mtq.dtorq__dbasestate(m0,sat,x0,vecs) , np.vstack([np.zeros((3,3)),np.cross(ax/3*(m0),drotmatTvecdq(q0,B_ECI))]) ))

    assert np.all(np.isclose( mtq.dtorq__dh(m0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(mtq.dtorq__dh(m0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( mtq.ddtorq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dudu(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dudbias(m0,sat,x0,vecs).shape==(1,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]) ,0) ))
    assert np.all(mtq.ddtorq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dudh(m0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs).shape==(0,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) ,np.zeros((0,7,3)) ))#np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]),0) ))
    assert np.all(mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs).shape==(0,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dbiasdh(m0,sat,x0,vecs).shape==(0,0,3))

    dxdx = np.zeros((7,7,3))
    dxdx[3:7,3:7,:] = np.cross(ax/3*(m0),ddrotmatTvecdqdq(q0,B_ECI))

    assert np.all(np.isclose( mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dhdh(m0,sat,x0,vecs).shape==(0,0,3))

    assert np.all(mtq.storage_torque(m0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(mtq.storage_torque(m0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( mtq.dstor_torq__du(m0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(mtq.dstor_torq__du(m0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbias(m0,sat,x0,vecs) , np.zeros((0,0)) ))
    assert np.all(mtq.dstor_torq__dbias(m0,sat,x0,vecs).shape == (0,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbasestate(m0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(mtq.dstor_torq__dbasestate(m0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( mtq.dstor_torq__dh(m0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(mtq.dstor_torq__dh(m0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dudu(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dudbias(m0,sat,x0,vecs).shape==(1,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dudh(m0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((0,0 ,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs).shape==(0,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs) , np.zeros((0,7,0))))
    assert np.all(mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs).shape==(0,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs).shape==(0,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(mtq.ddstor_torq__dhdh(m0,sat,x0,vecs).shape==(0,0,0))

    ax = unitvecs[1]
    max_moment = 4.51
    std = 0.243
    mtq = MTQ(ax,std,max_moment,has_bias = False, use_noise = False)
    m0 = random_n_unit_vec(3)[0]
    B_ECI = unitvecs[2]
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[mtq])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose((m0)*unitvecs[0],mtq.torque(m0,sat,x0,vecs)))
    assert np.all(np.isclose( mtq.dtorq__du(m0,sat,x0,vecs) ,unitvecs[0] ))
    assert np.all(np.isclose( mtq.dtorq__dbias(m0,sat,x0,vecs) , np.zeros((1,3)) ))
    assert np.all(mtq.dtorq__dbias(m0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( mtq.dtorq__dbasestate(m0,sat,x0,vecs) , np.vstack([np.zeros((3,3)),2*(m0)*1*np.vstack([unitvecs[0],np.outer(unitvecs[1],unitvecs[2]) ]) ]) ))
    assert np.all(np.isclose( mtq.dtorq__dh(m0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(mtq.dtorq__dh(m0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( mtq.ddtorq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dudu(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dudbias(m0,sat,x0,vecs).shape==(1,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , np.expand_dims(np.vstack([np.zeros((3,3)),2*1*np.vstack([unitvecs[0],np.outer(unitvecs[1],unitvecs[2]) ]) ]),0) ))
    assert np.all(mtq.ddtorq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dudh(m0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs).shape==(0,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) ,np.zeros((0,7,3))))
    assert np.all(mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs).shape==(0,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dbiasdh(m0,sat,x0,vecs).shape==(0,0,3))

    dxdx = np.zeros((7,7,3))
    dxdx[3:7,3:7,:] = np.cross(ax*(m0),ddrotmatTvecdqdq(q0,B_ECI))
    assert np.all(np.isclose( mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dhdh(m0,sat,x0,vecs).shape==(0,0,3))


    assert np.all(mtq.storage_torque(m0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(mtq.storage_torque(m0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( mtq.dstor_torq__du(m0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(mtq.dstor_torq__du(m0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbias(m0,sat,x0,vecs) , np.zeros((0,0)) ))
    assert np.all(mtq.dstor_torq__dbias(m0,sat,x0,vecs).shape == (0,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbasestate(m0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(mtq.dstor_torq__dbasestate(m0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( mtq.dstor_torq__dh(m0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(mtq.dstor_torq__dh(m0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dudu(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dudbias(m0,sat,x0,vecs).shape==(1,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dudh(m0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs).shape==(0,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs) , np.zeros((0,7,0))))
    assert np.all(mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs).shape==(0,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs).shape==(0,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(mtq.ddstor_torq__dhdh(m0,sat,x0,vecs).shape==(0,0,0))

def test_MTQ_torque_etc_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    biast = bias.copy()
    bsr = 0.03
    mtq = MTQ(ax,std,max_moment,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)

    axr = random_n_unit_vec(3)*3
    max_torqr = 4.51
    momentumr = -3.1
    max_hr = 3.8
    msnsr = 0.3
    Jr = 0.22
    stdr = 0.243
    biasr = random_n_unit_vec(3)[1]*0.1
    bsrr = 0.03
    rw = RW(axr,stdr,max_torqr,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = biasr,use_noise = True,bias_std_rate = bsrr)

    m0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[mtq,rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose(np.cross(ax/3*(m0+biast),B_B),mtq.torque(m0,sat,x0,vecs)))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    ufun = lambda c: mtq.torque(c,sat,x0,vecs)
    xfun = lambda c: mtq.torque(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    hfun = lambda c: mtq.torque(m0,sat,x0,vecs)
    bfun = lambda c: MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).torque(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(m0)).T
    Jbfun = np.array(nd.Jacobian(bfun)(biast)).T
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T

    assert np.allclose(Jxfun, mtq.dtorq__dbasestate(m0,sat,x0,vecs))
    assert np.allclose(Jufun, mtq.dtorq__du(m0,sat,x0,vecs))
    assert np.allclose(Jbfun, mtq.dtorq__dbias(m0,sat,x0,vecs))
    assert np.allclose(Jhfun, mtq.dtorq__dh(m0,sat,x0,vecs))


    for j in unitvecs:
        fun_hj = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = True, bias = c[1],use_noise = False,bias_std_rate = bsr).torque(c[0],Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c[1],use_noise = False,bias_std_rate = bsr),rw]),np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),vecsxfun(np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]))),j).item()

        ufunjju = lambda c: np.dot(mtq.dtorq__du(c,sat,x0,vecs),j).item()
        ufunjjb = lambda c: np.dot(mtq.dtorq__dbias(c,sat,x0,vecs),j).item()
        ufunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(c,sat,x0,vecs),j)
        ufunjjh = lambda c: np.dot(mtq.dtorq__dh(c,sat,x0,vecs),j)

        xfunjju = lambda c: np.dot(mtq.dtorq__du(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjb = lambda c: np.dot(mtq.dtorq__dbias(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjh = lambda c: np.dot(mtq.dtorq__dh(m0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)

        hfunjju = lambda c: np.dot(mtq.dtorq__du(m0,sat,x0,vecs),j).item()
        hfunjjb = lambda c: np.dot(mtq.dtorq__dbias(m0,sat,x0,vecs),j).item()
        hfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(m0,sat,x0,vecs),j)
        hfunjjh = lambda c: np.dot(mtq.dtorq__dh(m0,sat,x0,vecs),j)

        bfunjju = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__du(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j).item()
        bfunjjb = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbias(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j).item()
        bfunjjx = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbasestate(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
        bfunjjh = lambda c: np.dot( MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dh(m0,Satellite(actuators=[MTQ(ax,std,max_moment,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        # Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , np.dot( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjb , np.dot( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjx , np.dot( mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jxfunjjh , np.dot( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) , j ))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(m0))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(m0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(m0))
        # Jufunjjh = np.array(nd.Jacobian(ufunjjh)(m0))
        assert np.allclose( Jufunjju , np.dot( mtq.ddtorq__dudu(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjb , np.dot( mtq.ddtorq__dudbias(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjx.T , np.dot( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jufunjjh , np.dot( mtq.ddtorq__dudh(m0,sat,x0,vecs) , j ))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(biast))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(biast))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(biast))
        # Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(bias))
        assert np.allclose( Jbfunjju , np.dot( mtq.ddtorq__dudbias(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjb , np.dot( mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjx.T , np.dot( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jbfunjjh , np.dot( mtq.ddtorq__dbiasdh(m0,sat,x0,vecs) , j ))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
        Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(500.2))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))
        # Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(500.2))
        assert np.allclose( Jhfunjju , np.dot( mtq.ddtorq__dudh(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjb , np.dot( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjx , np.dot( mtq.ddtorq__dbiasdh(m0,sat,x0,vecs) , j ))
        # assert np.allclose( Jhfunjjh , np.dot( mtq.ddtorq__dhdh(m0,sat,x0,vecs) , j ))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[m0],[biast],x0,[500.2]]).flatten().tolist()))
        test = np.block([[(mtq.ddtorq__dudh(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbiasdh(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs)@j).T,mtq.ddtorq__dhdh(m0,sat,x0,vecs)@j]    ])
        Hguess = np.block([[mtq.ddtorq__dudu(m0,sat,x0,vecs)@j,mtq.ddtorq__dudbias(m0,sat,x0,vecs)@j,mtq.ddtorq__dudbasestate(m0,sat,x0,vecs)@j,mtq.ddtorq__dudh(m0,sat,x0,vecs)@j],\
                            [mtq.ddtorq__dudbias(m0,sat,x0,vecs)@j,mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs)@j,mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs)@j,mtq.ddtorq__dbiasdh(m0,sat,x0,vecs)@j],\
                            [(mtq.ddtorq__dudbasestate(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs)@j).T,mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs)@j,mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs)@j],\
                            [(mtq.ddtorq__dudh(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbiasdh(m0,sat,x0,vecs)@j).T,(mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs)@j).T,mtq.ddtorq__dhdh(m0,sat,x0,vecs)@j]    ])
        # print(Hfun.shape,Hguess.shape)
        assert np.allclose(Hfun[0:9,0:9],Hguess)
        assert np.allclose(Hfun[0:9,9],0)
        assert np.allclose(Hfun[9,0:9],0)
        assert np.allclose(Hfun[9,9],0)


    assert np.all(np.isclose( mtq.dtorq__du(m0,sat,x0,vecs) , np.cross(ax/3,B_B) ))
    assert np.all(np.isclose( mtq.dtorq__dbias(m0,sat,x0,vecs) , np.cross(ax/3,B_B) ))
    assert np.all(np.isclose( mtq.dtorq__dbasestate(m0,sat,x0,vecs) , np.vstack([np.zeros((3,3)),np.cross(ax/3*(m0+biast),drotmatTvecdq(q0,B_ECI))]) ))

    assert np.all(np.isclose( mtq.dtorq__dh(m0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(mtq.dtorq__dh(m0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( mtq.ddtorq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dudu(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dudbias(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]) ,0) ))
    assert np.all(mtq.ddtorq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dudh(m0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) , np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]),0) ))
    assert np.all(mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dbiasdh(m0,sat,x0,vecs).shape==(1,0,3))

    dxdx = np.zeros((7,7,3))
    dxdx[3:7,3:7,:] = np.cross(ax/3*(m0+biast),ddrotmatTvecdqdq(q0,B_ECI))

    assert np.all(np.isclose( mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dhdh(m0,sat,x0,vecs).shape==(0,0,3))



    assert np.all(mtq.storage_torque(m0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(mtq.storage_torque(m0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( mtq.dstor_torq__du(m0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(mtq.dstor_torq__du(m0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbias(m0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(mtq.dstor_torq__dbias(m0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbasestate(m0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(mtq.dstor_torq__dbasestate(m0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( mtq.dstor_torq__dh(m0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(mtq.dstor_torq__dh(m0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dudu(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dudbias(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dudh(m0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs) , np.zeros((1,7,0))))
    assert np.all(mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs).shape==(1,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(mtq.ddstor_torq__dhdh(m0,sat,x0,vecs).shape==(0,0,0))

    ax = unitvecs[1]
    max_moment = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    mtq = MTQ(ax,std,max_moment,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)
    m0 = random_n_unit_vec(3)[0]
    B_ECI = unitvecs[2]
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[mtq])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose((m0+bias)*unitvecs[0],mtq.torque(m0,sat,x0,vecs)))
    assert np.all(np.isclose( mtq.dtorq__du(m0,sat,x0,vecs) ,unitvecs[0] ))
    assert np.all(np.isclose( mtq.dtorq__dbias(m0,sat,x0,vecs) , unitvecs[0] ))

    assert np.all(np.isclose( mtq.dtorq__dbasestate(m0,sat,x0,vecs) , np.vstack([np.zeros((3,3)),2*(m0+bias)*1*np.vstack([unitvecs[0],np.outer(unitvecs[1],unitvecs[2]) ]) ]) ))
    assert np.all(np.isclose( mtq.dtorq__dh(m0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(mtq.dtorq__dh(m0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( mtq.ddtorq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dudu(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dudbias(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dudbasestate(m0,sat,x0,vecs) , np.expand_dims(np.vstack([np.zeros((3,3)),2*1*np.vstack([unitvecs[0],np.outer(unitvecs[1],unitvecs[2]) ]) ]),0) ))
    assert np.all(mtq.ddtorq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dudh(m0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(mtq.ddtorq__dbiasdbias(m0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs) , np.expand_dims(np.vstack([np.zeros((3,3)),2*1*np.vstack([unitvecs[0],np.outer(unitvecs[1],unitvecs[2]) ]) ]),0) ))
    assert np.all(mtq.ddtorq__dbiasdbasestate(m0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(mtq.ddtorq__dbiasdh(m0,sat,x0,vecs).shape==(1,0,3))

    dxdx = np.zeros((7,7,3))
    dxdx[3:7,3:7,:] = np.cross(ax*(m0+bias),ddrotmatTvecdqdq(q0,B_ECI))
    assert np.all(np.isclose( mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddtorq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(mtq.ddtorq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( mtq.ddtorq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(mtq.ddtorq__dhdh(m0,sat,x0,vecs).shape==(0,0,3))


    assert np.all(mtq.storage_torque(m0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(mtq.storage_torque(m0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( mtq.dstor_torq__du(m0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(mtq.dstor_torq__du(m0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbias(m0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(mtq.dstor_torq__dbias(m0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( mtq.dstor_torq__dbasestate(m0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(mtq.dstor_torq__dbasestate(m0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( mtq.dstor_torq__dh(m0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(mtq.dstor_torq__dh(m0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dudu(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dudu(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbias(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dudbias(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(mtq.ddstor_torq__dudbasestate(m0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dudh(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dudh(m0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdbias(m0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs) , np.zeros((1,7,0))))
    assert np.all(mtq.ddstor_torq__dbiasdbasestate(m0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbiasdh(m0,sat,x0,vecs).shape==(1,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs) , dxdx))
    assert np.all(mtq.ddstor_torq__dbasestatedbasestate(m0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(mtq.ddstor_torq__dbasestatedh(m0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( mtq.ddstor_torq__dhdh(m0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(mtq.ddstor_torq__dhdh(m0,sat,x0,vecs).shape==(0,0,0))


    ax = random_n_unit_vec(3)
    max_m = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    mtq = MTQ(ax,std,max_m,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)
    m0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[mtq])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    N = 1000
    test_torq = mtq.clean_torque(m0+bias,sat,x0,vecs)
    opts = [mtq.torque(m0,sat,x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_torq,j) for j in opts]) #bias is constant
    mtq.update_bias(0.22)

    # torq_exp = magic.clean_torque(t0,sat,x0,vecs)
    torq_drift = [mtq.torque(m0,sat,x0,vecs,update_bias = True,j2000 = mtq.last_bias_update+0.5*sec2cent)-mtq.torque(m0,sat,x0,vecs,update_bias = True,j2000 = mtq.last_bias_update+0.5*sec2cent) for j in range(N)]
    exp_dist = [np.cross(ax,B_B)*np.random.normal(0,bsr*math.sqrt(0.5)) for j in range(N)]
    print(kstest([j[0] for j in torq_drift],[j[0] for j in exp_dist]).statistic)

    ks0 = kstest([j[0] for j in torq_drift],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in torq_drift],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in torq_drift],[j[2] for j in exp_dist])
    ind = 0
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 1
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 2
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))



    mtq.update_bias(0.22)
    dt = 0.8
    oldb = mtq.bias
    test_torq = mtq.clean_torque(m0 + oldb,sat,x0,vecs)
    last = mtq.last_bias_update
    opts = [mtq.torque(m0,sat,x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_torq,j) for j in opts]) #bias is constant
    mtq.update_bias(0.21)
    assert mtq.last_bias_update == last
    assert oldb == mtq.bias


    bias_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = mtq.bias
        assert np.allclose(mtq.torque(m0,sat,x0,vecs), mtq.clean_torque(m0+bk,sat,x0,vecs))
        mtq.update_bias(t)
        assert mtq.last_bias_update == t
        bias_drift += [(mtq.bias-bk).item()]
        assert np.allclose(mtq.torque(m0,sat,x0,vecs), mtq.clean_torque(m0+mtq.bias,sat,x0,vecs))
    exp_dist = [np.random.normal(0,bsr*dt) for j in range(N)]
    print(kstest(bias_drift,exp_dist).statistic)

    ks0 = kstest(bias_drift,exp_dist)
    ind = 0
    data_a = bias_drift
    data_b = exp_dist
    hist = np.histogram([dd for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_MTQ_torque_etc_noise():
    ax = random_n_unit_vec(3)
    max_moment = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    mtq = MTQ(ax,std,max_moment,has_bias = True, bias = bias,use_noise = True,bias_std_rate = bsr)

    m0 = random_n_unit_vec(3)[0]*3
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[mtq])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}
    torq_exp = mtq.clean_torque(m0+bias,sat,x0,vecs)
    N = 1000
    torq_err = [mtq.torque(m0,sat,x0,vecs,update_noise = True)-torq_exp for j in range(N)]

    test_torq = mtq.torque(m0,sat,x0,vecs)
    assert np.allclose(test_torq,[mtq.torque(m0,sat,x0,vecs) for j in range(N)])
    exp_dist = [np.cross(ax*np.random.normal(0,std),B_B) for j in range(N)]


    ks0 = kstest([j[0] for j in torq_err],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in torq_err],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in torq_err],[j[2] for j in exp_dist])
    ind = 0
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    ind = 1
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 2
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    assert mtq.control_cov() == std**2.0
    assert np.all(np.shape(mtq.momentum_measurement_cov()) == (0,0))

def test_magic_setup():
    ax = random_n_unit_vec(3)*3
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    mag = Magic(ax,std,max_torq,has_bias = True, bias = bias,use_noise = True,bias_std_rate = bsr)
    assert np.all(np.isclose(ax/3,mag.axis))
    assert np.all(bias==mag.bias)
    assert mag.has_bias
    assert mag.use_noise
    assert ~mag.has_momentum
    assert np.isnan(mag.J)
    assert mag.max == max_torq
    assert np.isnan(mag.momentum)
    assert np.isnan(mag.max_h)
    assert mag.noise_settings == std
    assert mag.std == std
    assert mag.bias_std_rate == bsr
    assert np.isnan(mag.momentum_sens_noise_std)
    # self.noise_model = noise_model TODO--how to check....

def test_magic_torque_etc_clean():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_torq = 4.51
    std = 0.243
    magic = Magic(ax,std,max_torq,has_bias = False,use_noise = False)

    t0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[magic])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose(ax/3*(t0),magic.torque(t0,sat,x0,vecs)))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    ufun = lambda c: magic.torque(c,sat,x0,vecs)
    xfun = lambda c: magic.torque(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    hfun = lambda c: magic.torque(t0,sat,x0,vecs)
    bfun = lambda c: Magic(ax,std,max_torq,has_bias = False,use_noise = False).torque(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = False,use_noise = False)]),x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(t0)).T
    Jbfun = np.array(nd.Jacobian(bfun)(20000)).T
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T

    assert np.allclose(Jxfun, magic.dtorq__dbasestate(t0,sat,x0,vecs))
    assert np.allclose(Jufun, magic.dtorq__du(t0,sat,x0,vecs))
    assert np.allclose(Jbfun, magic.dtorq__dbias(t0,sat,x0,vecs))
    assert np.allclose(Jhfun, magic.dtorq__dh(t0,sat,x0,vecs))


    for j in unitvecs:
        fun_hj = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = False,use_noise = False).torque(c[0],Satellite(actuators=[Magic(ax,std,max_torq,has_bias = False,use_noise = False)]),np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]),vecsxfun(np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]))),j).item()

        ufunjju = lambda c: np.dot(magic.dtorq__du(c,sat,x0,vecs),j).item()
        ufunjjb = lambda c: np.dot(magic.dtorq__dbias(c,sat,x0,vecs),j)
        ufunjjx = lambda c: np.dot(magic.dtorq__dbasestate(c,sat,x0,vecs),j)
        ufunjjh = lambda c: np.dot(magic.dtorq__dh(c,sat,x0,vecs),j)

        xfunjju = lambda c: np.dot(magic.dtorq__du(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjb = lambda c: np.dot(magic.dtorq__dbias(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjx = lambda c: np.dot(magic.dtorq__dbasestate(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjh = lambda c: np.dot(magic.dtorq__dh(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)

        hfunjju = lambda c: np.dot(magic.dtorq__du(t0,sat,x0,vecs),j).item()
        hfunjjb = lambda c: np.dot(magic.dtorq__dbias(t0,sat,x0,vecs),j)
        hfunjjx = lambda c: np.dot(magic.dtorq__dbasestate(t0,sat,x0,vecs),j)
        hfunjjh = lambda c: np.dot(magic.dtorq__dh(t0,sat,x0,vecs),j)
        #
        bfunjju = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = False, bias = c,use_noise = False).dtorq__du(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = False, bias = c,use_noise = False)]),x0,vecs),j).item()
        # bfunjjb = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbias(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j).item()
        bfunjjx = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = False, bias = c,use_noise = False).dtorq__dbasestate(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = False, bias = c,use_noise = False)]),x0,vecs),j)
        # bfunjjh = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dh(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        # Jxfunjjb = np.aray(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        # Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , np.dot( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jxfunjjb , np.dot( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjx , np.dot( magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jxfunjjh , np.dot( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) , j ))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        # Jufunjjb = np.array(nd.Jacobian(ufunjjb)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        # Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , np.dot( magic.ddtorq__dudu(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jufunjjb , np.dot( magic.ddtorq__dudbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjx.T , np.dot( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jufunjjh , np.dot( magic.ddtorq__dudh(t0,sat,x0,vecs) , j ))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(20000))
        # Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(bias))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(20000))
        # # Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(bias))
        assert np.allclose( Jbfunjju , np.dot( magic.ddtorq__dudbias(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jbfunjjb , np.dot( magic.ddtorq__dbiasdbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjx.T , np.dot( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jbfunjjh , np.dot( magic.ddtorq__dbiasdh(t0,sat,x0,vecs) , j ))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
        # Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(500.2))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))
        # Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(500.2))
        assert np.allclose( Jhfunjju , np.dot( magic.ddtorq__dudh(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jhfunjjb , np.dot( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjx , np.dot( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jhfunjjh , np.dot( magic.ddtorq__dhdh(t0,sat,x0,vecs) , j ))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[t0],x0,[500.2]]).flatten().tolist()))
        Hguess = np.block([[magic.ddtorq__dudu(t0,sat,x0,vecs)@j,magic.ddtorq__dudbias(t0,sat,x0,vecs)@j,magic.ddtorq__dudbasestate(t0,sat,x0,vecs)@j,magic.ddtorq__dudh(t0,sat,x0,vecs)@j],\
                            [(magic.ddtorq__dudbias(t0,sat,x0,vecs)@j).T,magic.ddtorq__dbiasdbias(t0,sat,x0,vecs)@j,magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j,magic.ddtorq__dbiasdh(t0,sat,x0,vecs)@j],\
                            [(magic.ddtorq__dudbasestate(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j).T,magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs)@j,magic.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j],\
                            [(magic.ddtorq__dudh(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbiasdh(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j).T,magic.ddtorq__dhdh(t0,sat,x0,vecs)@j]    ])

        assert np.allclose(Hfun[8,:],0)
        assert np.allclose(Hfun[:,8],0)
        assert np.allclose(Hfun[0:8,0:8],Hguess)


    assert np.all(np.isclose( magic.dtorq__du(t0,sat,x0,vecs) ,ax/3 ))
    assert np.all(np.isclose( magic.dtorq__dbias(t0,sat,x0,vecs) , ax/3))
    assert np.all(np.isclose( magic.dtorq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,3))))

    assert np.all(np.isclose( magic.dtorq__dh(t0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(magic.dtorq__dh(t0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( magic.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,0,3))
    assert np.all(np.isclose( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,3))))
    assert np.all(magic.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( magic.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( magic.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) ,np.zeros((0,7,3)) ))#np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]),0) ))
    assert np.all(magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(0,0,3))

    dxdx = np.zeros((7,7,3))

    assert np.all(np.isclose( magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(magic.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( magic.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dhdh(t0,sat,x0,vecs).shape==(0,0,3))

    assert np.all(magic.storage_torque(t0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(magic.storage_torque(t0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( magic.dstor_torq__du(t0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(magic.dstor_torq__du(t0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( magic.dstor_torq__dbias(t0,sat,x0,vecs) , np.zeros((0,0)) ))
    assert np.all(magic.dstor_torq__dbias(t0,sat,x0,vecs).shape == (0,0))
    assert np.all(np.isclose( magic.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(magic.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( magic.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(magic.dstor_torq__dh(t0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0 ,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((0,7,0))))
    assert np.all(magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(0,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(magic.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(0,0,0))

    # ufun = lambda c: magic.storage_torque(c,sat,x0,vecs)
    # xfun = lambda c: magic.storage_torque(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    # hfun = lambda c: magic.storage_torque(t0,sat,x0,vecs)
    # bfun = lambda c: magic.storage_torque(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs)

    # Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    # Jufun = np.array(nd.Jacobian(ufun)(t0))
    # Jbfun = np.array(nd.Jacobian(bfun)(bias))
    # Jhfun = np.array(nd.Jacobian(hfun)(500.2))
    #
    # assert np.allclose(Jxfun, magic.dstor_torq__dbasestate(t0,sat,x0,vecs))
    # assert np.allclose(Jufun, magic.dstor_torq__du(t0,sat,x0,vecs))
    # assert np.allclose(Jbfun, magic.dstor_torq__dbias(t0,sat,x0,vecs))
    # assert np.allclose(Jhfun, magic.dstor_torq__dh(t0,sat,x0,vecs))
    #
    #
    # for j in unitvecs:
    #     ufunjju = lambda c: np.dot(magic.dstor_torq__du(c,sat,x0,vecs),j)
    #     ufunjjb = lambda c: np.dot(magic.dstor_torq__dbias(c,sat,x0,vecs),j)
    #     ufunjjx = lambda c: np.dot(magic.dstor_torq__dbasestate(c,sat,x0,vecs),j)
    #     ufunjjh = lambda c: np.dot(magic.dstor_torq__dh(c,sat,x0,vecs),j)
    #
    #     xfunjju = lambda c: np.dot(magic.dstor_torq__du(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
    #     xfunjjb = lambda c: np.dot(magic.dstor_torq__dbias(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
    #     xfunjjx = lambda c: np.dot(magic.dstor_torq__dbasestate(v,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
    #     xfunjjh = lambda c: np.dot(magic.dstor_torq__dh(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
    #
    #     hfunjju = lambda c: np.dot(magic.dstor_torq__du(t0,sat,x0,vecs),j)
    #     hfunjjb = lambda c: np.dot(magic.dstor_torq__dbias(t0,sat,x0,vecs),j)
    #     hfunjjx = lambda c: np.dot(magic.dstor_torq__dbasestate(t0,sat,x0,vecs),j)
    #     hfunjjh = lambda c: np.dot(magic.dstor_torq__dh(t0,sat,x0,vecs),j)
    #
    #     bfunjju = lambda c: np.dot(magic.dstor_torq__du(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
    #     bfunjjb = lambda c: np.dot(magic.dstor_torq__dbias(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
    #     bfunjjx = lambda c: np.dot(magic.dstor_torq__dbasestate(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
    #     bfunjjh = lambda c: np.dot(magic.dstor_torq__dh(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
    #
    #     Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
    #     Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
    #     Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
    #     Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
    #     assert np.allclose( Jxfunjju , np.dot( magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jxfunjjb , np.dot( magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jxfunjjx , np.dot( magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jxfunjjh , np.dot( magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) , j ))
    #
    #     Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
    #     Jufunjjb = np.array(nd.Jacobian(ufunjjb)(t0))
    #     Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
    #     Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
    #     assert np.allclose( Jufunjju , np.dot( magic.ddstor_torq__dudu(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jufunjjb , np.dot( magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jufunjjx , np.dot( magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jufunjjh , np.dot( magic.ddstor_torq__dudh(t0,sat,x0,vecs) , j ))
    #
    #     Jbfunjju = np.array(nd.Jacobian(bfunjju)(bias))
    #     Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(bias))
    #     Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(bias))
    #     Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(bias))
    #     assert np.allclose( Jbfunjju , np.dot( magic.ddstor_torq__dudbias(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jbfunjjb , np.dot( magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jbfunjjx , np.dot( magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jbfunjjh , np.dot( magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs) , j ))
    #
    #     Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
    #     Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(500.2))
    #     Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))
    #     Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(500.2))
    #     assert np.allclose( Jhfunjju , np.dot( magic.ddstor_torq__dudh(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jhfunjjb , np.dot( magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jhfunjjx , np.dot( magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs) , j ))
    #     assert np.allclose( Jhfunjjh , np.dot( magic.ddstor_torq__dhdh(t0,sat,x0,vecs) , j ))


    ax = unitvecs[1]
    max_torq = 4.51
    std = 0.243
    magic = Magic(ax,std,max_torq,has_bias = False, use_noise = False)
    t0 = random_n_unit_vec(3)[0]
    B_ECI = unitvecs[2]
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[magic])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose((t0)*unitvecs[1],magic.torque(t0,sat,x0,vecs)))
    assert np.all(np.isclose( magic.dtorq__du(t0,sat,x0,vecs) ,unitvecs[1] ))
    assert np.all(np.isclose( magic.dtorq__dbias(t0,sat,x0,vecs) , np.zeros((1,3)) ))
    assert np.all(magic.dtorq__dbias(t0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( magic.dtorq__dbasestate(t0,sat,x0,vecs) ,np.zeros((7,3))))
    assert np.all(np.isclose( magic.dtorq__dh(t0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(magic.dtorq__dh(t0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( magic.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,0,3))
    assert np.all(np.isclose( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,3)) ))
    assert np.all(magic.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( magic.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( magic.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) ,np.zeros((0,7,3))))
    assert np.all(magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(0,0,3))

    dxdx = np.zeros((7,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(magic.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( magic.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dhdh(t0,sat,x0,vecs).shape==(0,0,3))


    assert np.all(magic.storage_torque(t0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(magic.storage_torque(t0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( magic.dstor_torq__du(t0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(magic.dstor_torq__du(t0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( magic.dstor_torq__dbias(t0,sat,x0,vecs) , np.zeros((0,0)) ))
    assert np.all(magic.dstor_torq__dbias(t0,sat,x0,vecs).shape == (0,0))
    assert np.all(np.isclose( magic.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(magic.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( magic.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(magic.dstor_torq__dh(t0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((0,7,0))))
    assert np.all(magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(0,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(magic.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(0,0,0))

def test_magic_torque_etc_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    biast = bias.copy()
    bsr = 0.03
    magic = Magic(ax,std,max_torq,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)

    axr = random_n_unit_vec(3)*3
    max_torqr = 4.51
    momentumr = -3.1
    max_hr = 3.8
    msnsr = 0.3
    Jr = 0.22
    stdr = 0.243
    biasr = random_n_unit_vec(3)[1]*0.1
    bsrr = 0.03
    rw = RW(axr,stdr,max_torqr,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = biasr,use_noise = True,bias_std_rate = bsrr)

    t0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[magic,rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose(ax/3*(t0+biast),magic.torque(t0,sat,x0,vecs)))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    ufun = lambda c: magic.torque(c,sat,x0,vecs)
    xfun = lambda c: magic.torque(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    hfun = lambda c: magic.torque(t0,sat,x0,vecs)
    bfun = lambda c: Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).torque(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(t0)).T
    Jbfun = np.array(nd.Jacobian(bfun)(biast)).T
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T

    assert np.allclose(Jxfun, magic.dtorq__dbasestate(t0,sat,x0,vecs))
    assert np.allclose(Jufun, magic.dtorq__du(t0,sat,x0,vecs))
    assert np.allclose(Jbfun, magic.dtorq__dbias(t0,sat,x0,vecs))
    assert np.allclose(Jhfun, magic.dtorq__dh(t0,sat,x0,vecs))


    for j in unitvecs:
        fun_hj = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = True, bias = c[1],use_noise = False,bias_std_rate = bsr).torque(c[0],Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c[1],use_noise = False,bias_std_rate = bsr),rw]),np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),vecsxfun(np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]))),j).item()

        ufunjju = lambda c: np.dot(magic.dtorq__du(c,sat,x0,vecs),j).item()
        ufunjjb = lambda c: np.dot(magic.dtorq__dbias(c,sat,x0,vecs),j).item()
        ufunjjx = lambda c: np.dot(magic.dtorq__dbasestate(c,sat,x0,vecs),j)
        ufunjjh = lambda c: np.dot(magic.dtorq__dh(c,sat,x0,vecs),j)

        xfunjju = lambda c: np.dot(magic.dtorq__du(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjb = lambda c: np.dot(magic.dtorq__dbias(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjx = lambda c: np.dot(magic.dtorq__dbasestate(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjh = lambda c: np.dot(magic.dtorq__dh(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)

        hfunjju = lambda c: np.dot(magic.dtorq__du(t0,sat,x0,vecs),j).item()
        hfunjjb = lambda c: np.dot(magic.dtorq__dbias(t0,sat,x0,vecs),j).item()
        hfunjjx = lambda c: np.dot(magic.dtorq__dbasestate(t0,sat,x0,vecs),j)
        hfunjjh = lambda c: np.dot(magic.dtorq__dh(t0,sat,x0,vecs),j)

        bfunjju = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__du(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j).item()
        bfunjjb = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbias(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j).item()
        bfunjjx = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbasestate(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
        bfunjjh = lambda c: np.dot( Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dh(t0,Satellite(actuators=[Magic(ax,std,max_torq,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs),j)
        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        # Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , np.dot( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjb , np.dot( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjx , np.dot( magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jxfunjjh , np.dot( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) , j ))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        # Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , np.dot( magic.ddtorq__dudu(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjb , np.dot( magic.ddtorq__dudbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjx.T , np.dot( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jufunjjh , np.dot( magic.ddtorq__dudh(t0,sat,x0,vecs) , j ))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(biast))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(biast))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(biast))
        # Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(biast))
        assert np.allclose( Jbfunjju , np.dot( magic.ddtorq__dudbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjb , np.dot( magic.ddtorq__dbiasdbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjx.T , np.dot( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jbfunjjh , np.dot( magic.ddtorq__dbiasdh(t0,sat,x0,vecs) , j ))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
        Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(500.2))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))
        # Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(500.2))
        assert np.allclose( Jhfunjju , np.dot( magic.ddtorq__dudh(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjb , np.dot( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjx , np.dot( magic.ddtorq__dbiasdh(t0,sat,x0,vecs) , j ))
        # assert np.allclose( Jhfunjjh , np.dot( magic.ddtorq__dhdh(t0,sat,x0,vecs) , j ))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[t0],[biast],x0,[500.2]]).flatten().tolist()))
        test = np.block([[(magic.ddtorq__dudh(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbiasdh(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j).T,magic.ddtorq__dhdh(t0,sat,x0,vecs)@j]    ])
        Hguess = np.block([[magic.ddtorq__dudu(t0,sat,x0,vecs)@j,magic.ddtorq__dudbias(t0,sat,x0,vecs)@j,magic.ddtorq__dudbasestate(t0,sat,x0,vecs)@j,magic.ddtorq__dudh(t0,sat,x0,vecs)@j],\
                            [magic.ddtorq__dudbias(t0,sat,x0,vecs)@j,magic.ddtorq__dbiasdbias(t0,sat,x0,vecs)@j,magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j,magic.ddtorq__dbiasdh(t0,sat,x0,vecs)@j],\
                            [(magic.ddtorq__dudbasestate(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j).T,magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs)@j,magic.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j],\
                            [(magic.ddtorq__dudh(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbiasdh(t0,sat,x0,vecs)@j).T,(magic.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j).T,magic.ddtorq__dhdh(t0,sat,x0,vecs)@j]    ])
        # print(Hfun.shape,Hguess.shape)
        assert np.allclose(Hfun[0:9,0:9],Hguess)
        assert np.allclose(Hfun[0:9,9],0)
        assert np.allclose(Hfun[9,0:9],0)
        assert np.allclose(Hfun[9,9],0)


    assert np.all(np.isclose( magic.dtorq__du(t0,sat,x0,vecs) , ax/3 ))
    assert np.all(np.isclose( magic.dtorq__dbias(t0,sat,x0,vecs) , ax/3 ))
    assert np.all(np.isclose( magic.dtorq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,3)) ))

    assert np.all(np.isclose( magic.dtorq__dh(t0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(magic.dtorq__dh(t0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( magic.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) ,  np.zeros((1,7,3)) ))
    assert np.all(magic.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( magic.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( magic.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) ,  np.zeros((1,7,3)) ))
    assert np.all(magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(1,0,3))

    dxdx = np.zeros((7,7,3))

    assert np.all(np.isclose( magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(magic.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( magic.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dhdh(t0,sat,x0,vecs).shape==(0,0,3))



    assert np.all(magic.storage_torque(t0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(magic.storage_torque(t0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( magic.dstor_torq__du(t0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(magic.dstor_torq__du(t0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( magic.dstor_torq__dbias(t0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(magic.dstor_torq__dbias(t0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( magic.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(magic.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( magic.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(magic.dstor_torq__dh(t0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((1,7,0))))
    assert np.all(magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(1,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(magic.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(0,0,0))


    ax = unitvecs[1]
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    magic = Magic(ax,std,max_torq,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)
    t0 = random_n_unit_vec(3)[0]
    B_ECI = unitvecs[2]
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[magic])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose((t0+bias)*unitvecs[1],magic.torque(t0,sat,x0,vecs)))
    assert np.all(np.isclose( magic.dtorq__du(t0,sat,x0,vecs) ,unitvecs[1] ))
    assert np.all(np.isclose( magic.dtorq__dbias(t0,sat,x0,vecs) , unitvecs[1] ))

    assert np.all(np.isclose( magic.dtorq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,3)) ))
    assert np.all(np.isclose( magic.dtorq__dh(t0,sat,x0,vecs) , np.zeros((0,3))))
    assert np.all(magic.dtorq__dh(t0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( magic.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dudbasestate(t0,sat,x0,vecs) , np.zeros((1,7,3))  ))
    assert np.all(magic.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( magic.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,0,3))

    assert np.all(np.isclose( magic.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(magic.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((1,7,3))  ))
    assert np.all(magic.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(magic.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(1,0,3))

    dxdx = np.zeros((7,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( magic.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,3)) ))
    assert np.all(magic.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,3))
    assert np.all(np.isclose( magic.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(magic.ddtorq__dhdh(t0,sat,x0,vecs).shape==(0,0,3))


    assert np.all(magic.storage_torque(t0,sat,x0,vecs)  == np.zeros(0))
    assert np.all(magic.storage_torque(t0,sat,x0,vecs).shape == (0,))
    assert np.all(np.isclose( magic.dstor_torq__du(t0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(magic.dstor_torq__du(t0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( magic.dstor_torq__dbias(t0,sat,x0,vecs) , np.zeros((1,0)) ))
    assert np.all(magic.dstor_torq__dbias(t0,sat,x0,vecs).shape == (1,0))
    assert np.all(np.isclose( magic.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,0))))
    assert np.all(magic.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,0))
    assert np.all(np.isclose( magic.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((0,0))))
    assert np.all(magic.dstor_torq__dh(t0,sat,x0,vecs).shape==(0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,0))))
    assert np.all(magic.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,0,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((1,7,0))))
    assert np.all(magic.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,0,0)) ))
    assert np.all(magic.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(1,0,0))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(magic.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,0))
    assert np.all(np.isclose( magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,0,0)) ))
    assert np.all(magic.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,0,0))
    assert np.all(np.isclose( magic.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((0,0,0)) ))
    assert np.all(magic.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(0,0,0))


    ax = random_n_unit_vec(3)
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    magic = Magic(ax,std,max_torq,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)
    t0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[magic])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    N = 1000
    test_torq = magic.clean_torque(t0+bias,sat,x0,vecs)
    opts = [magic.torque(t0,sat,x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_torq,j) for j in opts]) #bias is constant
    magic.update_bias(0.22)

    # torq_exp = magic.clean_torque(t0,sat,x0,vecs)
    torq_drift = [magic.torque(t0,sat,x0,vecs,update_bias = True,j2000 = magic.last_bias_update+0.5*sec2cent)-magic.torque(t0,sat,x0,vecs,update_bias = True,j2000 = magic.last_bias_update+0.5*sec2cent) for j in range(N)]

    exp_dist = [ax*np.random.normal(0,bsr*math.sqrt(0.5)) for j in range(N)]

    ks0 = kstest([j[0] for j in torq_drift],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in torq_drift],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in torq_drift],[j[2] for j in exp_dist])
    ind = 0
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 1
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 2
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))


    magic.update_bias(0.22)
    dt = 0.8
    oldb = magic.bias
    test_torq = magic.clean_torque(t0 + oldb,sat,x0,vecs)
    last = magic.last_bias_update
    opts = [magic.torque(t0,sat,x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_torq,j) for j in opts]) #bias is constant
    magic.update_bias(0.21)
    assert magic.last_bias_update == last
    assert oldb == magic.bias



    bias_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = magic.bias
        assert np.allclose(magic.torque(t0,sat,x0,vecs), magic.clean_torque(t0+bk,sat,x0,vecs))
        magic.update_bias(t)
        assert magic.last_bias_update == t
        bias_drift += [(magic.bias-bk).item()]
        assert np.allclose(magic.torque(t0,sat,x0,vecs), magic.clean_torque(t0+magic.bias,sat,x0,vecs))
    exp_dist = [np.random.normal(0,bsr*dt) for j in range(N)]
    print(kstest(bias_drift,exp_dist).statistic)

    ks0 = kstest(bias_drift,exp_dist)
    ind = 0
    data_a = bias_drift
    data_b = exp_dist
    hist = np.histogram([dd for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_magic_torque_etc_noise():
    ax = random_n_unit_vec(3)
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    magic = Magic(ax,std,max_torq,has_bias = True, bias = bias,use_noise = True,bias_std_rate = bsr)

    t0 = random_n_unit_vec(3)[0]*3
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[magic])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    N = 1000
    test_torq = magic.torque(t0,sat,x0,vecs)
    assert np.allclose(test_torq,[magic.torque(t0,sat,x0,vecs) for j in range(N)])

    torq_exp = magic.clean_torque(t0+bias,sat,x0,vecs)
    torq_err = [magic.torque(t0,sat,x0,vecs,update_noise = True)-torq_exp for j in range(N)]
    exp_dist = [ax*np.random.normal(0,std) for j in range(N)]



    ks0 = kstest([j[0] for j in torq_err],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in torq_err],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in torq_err],[j[2] for j in exp_dist])
    ind = 0
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 1
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 2
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    assert magic.control_cov() == std**2.0
    assert np.all(np.shape(magic.momentum_measurement_cov()) == (0,0))

def test_RW_setup():
    ax = random_n_unit_vec(3)*3
    max_torq = 4.51
    momentum = -3.1
    max_h = 3.8
    msns = 0.3
    J = 0.22
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    rw = RW(ax,std,max_torq,J,momentum,max_h,msns,has_bias = True, bias = bias,use_noise = True,bias_std_rate = bsr)
    assert np.all(np.isclose(ax/3,rw.axis))
    assert np.all(bias==rw.bias)
    assert rw.has_bias
    assert rw.use_noise
    assert rw.has_momentum
    assert rw.J == J
    assert rw.max == max_torq
    assert rw.momentum == momentum
    assert rw.max_h == max_h
    assert rw.noise_settings == std
    assert rw.std == std
    assert rw.bias_std_rate == bsr
    assert rw.momentum_sens_noise_std == msns
    # self.noise_model = noise_model TODO--how to check...

def test_RW_torque_etc_clean():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_torq = 4.51
    std = 0.243
    momentumr = -3.1
    max_hr = 3.8
    msnsr = 0.3
    Jr = 0.22
    stdr = 0.243
    biasr = random_n_unit_vec(3)[1]*0.1
    bsrr = 0.03
    rw = RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False)

    t0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose(ax/3*(t0),rw.torque(t0,sat,x0,vecs)))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    ufun = lambda c: rw.torque(c,sat,x0,vecs)
    xfun = lambda c: rw.torque(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    hfun = lambda c: RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = False,use_noise = False).torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = False,use_noise = False)]),x0,vecs)
    bfun = lambda c: RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False).torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False)]),x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(t0)).T
    Jbfun = np.array(nd.Jacobian(bfun)(20000)).T
    Jhfun = np.array(nd.Jacobian(hfun)(momentumr)).T

    assert np.allclose(Jxfun, rw.dtorq__dbasestate(t0,sat,x0,vecs))
    assert np.allclose(Jufun, rw.dtorq__du(t0,sat,x0,vecs))
    assert np.allclose(Jbfun, rw.dtorq__dbias(t0,sat,x0,vecs))
    assert np.allclose(Jhfun, rw.dtorq__dh(t0,sat,x0,vecs))


    for j in unitvecs:
        fun_hj = lambda c: np.dot( RW(ax,std,max_torq,Jr,c[8],max_hr,msnsr,has_bias = False,use_noise = False).torque(c[0],Satellite(actuators=[RW(ax,std,max_torq,Jr,c[8],max_hr,msnsr,has_bias = False,use_noise = False)]),np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]),vecsxfun(np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]))),j).item()

        ufunjju = lambda c: np.dot(rw.dtorq__du(c,sat,x0,vecs),j).item()
        ufunjjb = lambda c: np.dot(rw.dtorq__dbias(c,sat,x0,vecs),j).item()
        ufunjjx = lambda c: np.dot(rw.dtorq__dbasestate(c,sat,x0,vecs),j)
        ufunjjh = lambda c: np.dot(rw.dtorq__dh(c,sat,x0,vecs),j).item()

        xfunjju = lambda c: np.dot(rw.dtorq__du(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjb = lambda c: np.dot(rw.dtorq__dbias(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjx = lambda c: np.dot(rw.dtorq__dbasestate(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjh = lambda c: np.dot(rw.dtorq__dh(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()

        hfunjju = lambda c: np.dot(RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False).dtorq__du(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False)]),x0,vecs),j).item()
        hfunjjb = lambda c:  np.dot(RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False).dtorq__dbias(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False)]),x0,vecs),j).item()
        hfunjjx = lambda c:  np.dot(RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False).dtorq__dbasestate(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False)]),x0,vecs),j)
        hfunjjh = lambda c:  np.dot(RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False).dtorq__dh(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False)]),x0,vecs),j).item()

        bfunjju = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False).dtorq__du(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = False,use_noise = False)]),x0,vecs),j).item()
        bfunjjb = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False).dtorq__dbias(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs),j).item()
        bfunjjx = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False).dtorq__dbasestate(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs),j)
        bfunjjh = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False).dtorq__dh(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs),j).item()
        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        # Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , np.dot( rw.ddtorq__dudbasestate(t0,sat,x0,vecs),j))
        # assert np.allclose( Jxfunjjb , np.dot( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs))
        assert np.allclose( Jxfunjjx , np.dot( rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs),j))
        assert np.allclose( Jxfunjjh , np.dot( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs),j))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        # Jufunjjb = np.array(nd.Jacobian(ufunjjb)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , np.dot( rw.ddtorq__dudu(t0,sat,x0,vecs),j))
        # assert np.allclose( Jufunjjb , np.dot( rw.ddtorq__dudbias(t0,sat,x0,vecs))
        assert np.allclose( Jufunjjx.T , np.dot( rw.ddtorq__dudbasestate(t0,sat,x0,vecs),j))
        assert np.allclose( Jufunjjh , np.dot( rw.ddtorq__dudh(t0,sat,x0,vecs),j))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(20000))
        # Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(20000))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(20000))
        Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(20000))
        assert np.allclose( Jbfunjju , np.dot( rw.ddtorq__dudbias(t0,sat,x0,vecs),j))
        # assert np.allclose( Jbfunjjb , np.dot( rw.ddtorq__dbiasdbias(t0,sat,x0,vecs))
        assert np.allclose( Jbfunjjx.T , np.dot( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs),j))
        assert np.allclose( Jbfunjjh , np.dot( rw.ddtorq__dbiasdh(t0,sat,x0,vecs),j))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(momentumr))
        # Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(momentumr))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(momentumr))
        Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(momentumr))
        assert np.allclose( Jhfunjju , np.dot( rw.ddtorq__dudh(t0,sat,x0,vecs),j))
        # assert np.allclose( Jhfunjjb , np.dot( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs))
        assert np.allclose( Jhfunjjx , np.dot( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs),j))
        assert np.allclose( Jhfunjjh , np.dot( rw.ddtorq__dhdh(t0,sat,x0,vecs),j))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[t0],x0,[momentumr]]).flatten().tolist()))
        Hguess = np.block([[rw.ddtorq__dudu(t0,sat,x0,vecs)@j,rw.ddtorq__dudbias(t0,sat,x0,vecs)@j,rw.ddtorq__dudbasestate(t0,sat,x0,vecs)@j,rw.ddtorq__dudh(t0,sat,x0,vecs)@j],\
                            [(rw.ddtorq__dudbias(t0,sat,x0,vecs)@j).T,rw.ddtorq__dbiasdbias(t0,sat,x0,vecs)@j,rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j,rw.ddtorq__dbiasdh(t0,sat,x0,vecs)@j],\
                            [(rw.ddtorq__dudbasestate(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j).T,rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs)@j,rw.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j],\
                            [(rw.ddtorq__dudh(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbiasdh(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j).T,rw.ddtorq__dhdh(t0,sat,x0,vecs)@j]    ])

        # assert np.allclose(Hfun[8,:],0)
        # assert np.allclose(Hfun[:,8],0)
        assert np.allclose(Hfun,Hguess)


    assert np.all(np.isclose( rw.dtorq__du(t0,sat,x0,vecs) ,ax/3 ))
    assert np.all(np.isclose( rw.dtorq__dbias(t0,sat,x0,vecs) ,np.zeros((0,3))))
    assert np.all(np.isclose( rw.dtorq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,3))))
    assert np.all(np.isclose( rw.dtorq__dh(t0,sat,x0,vecs) , np.zeros((1,3))))

    assert np.all(np.isclose( rw.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(rw.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,0,3))
    assert np.all(np.isclose( rw.ddtorq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,3))))
    assert np.all(rw.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,1,3))

    assert np.all(np.isclose( rw.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(rw.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) ,np.zeros((0,7,3)) ))#np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]),0) ))
    assert np.all(rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(0,1,3))

    dxdx = np.zeros((7,7,3))

    assert np.all(np.isclose( rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,3)) ))
    assert np.all(rw.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,3))
    assert np.all(np.isclose( rw.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dhdh(t0,sat,x0,vecs).shape==(1,1,3))

    assert np.all(rw.storage_torque(t0,sat,x0,vecs)  == -t0)
    assert np.all(np.isclose( rw.dstor_torq__du(t0,sat,x0,vecs) , -1 ))
    assert np.all(np.isclose( rw.dstor_torq__dbias(t0,sat,x0,vecs) , np.zeros((0,1)) ))
    assert np.all(rw.dstor_torq__dbias(t0,sat,x0,vecs).shape == (0,1))
    assert np.all(np.isclose( rw.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,1))))
    assert np.all(rw.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,1))
    assert np.all(np.isclose( rw.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((1,1))))
    assert np.all(rw.dstor_torq__dh(t0,sat,x0,vecs).shape==(1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,1)) ))
    assert np.all(rw.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,0,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0 ,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((0,7,1))))
    assert np.all(rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(0,1,1))

    dxdx = np.zeros((7,7,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,1)) ))
    assert np.all(rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(1,1,1))

    ufun = lambda c: rw.storage_torque(c,sat,x0,vecs).item()
    xfun = lambda c: rw.storage_torque(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    hfun = lambda c: rw.storage_torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs).item()
    bfun = lambda c: rw.storage_torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs).item()

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist()))
    Jufun = np.array(nd.Jacobian(ufun)(t0))
    Jbfun = np.array(nd.Jacobian(bfun)(20000))
    Jhfun = np.array(nd.Jacobian(hfun)(momentumr))

    assert np.allclose(Jxfun, rw.dstor_torq__dbasestate(t0,sat,x0,vecs))
    assert np.allclose(Jufun, rw.dstor_torq__du(t0,sat,x0,vecs))
    assert np.allclose(Jbfun, rw.dstor_torq__dbias(t0,sat,x0,vecs))
    assert np.allclose(Jhfun, rw.dstor_torq__dh(t0,sat,x0,vecs))


    for j in unitvecs:
        ufunjju = lambda c: rw.dstor_torq__du(c,sat,x0,vecs).item()
        # ufunjjb = lambda c: rw.dstor_torq__dbias(c,sat,x0,vecs)
        ufunjjx = lambda c: rw.dstor_torq__dbasestate(c,sat,x0,vecs)
        ufunjjh = lambda c: rw.dstor_torq__dh(c,sat,x0,vecs).item()

        xfunjju = lambda c: rw.dstor_torq__du(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
        # xfunjjb = lambda c: rw.dstor_torq__dbias(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
        xfunjjx = lambda c: rw.dstor_torq__dbasestate(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
        xfunjjh = lambda c: rw.dstor_torq__dh(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()

        hfunjju = lambda c: rw.dstor_torq__du(t0,sat,x0,vecs).item()
        # hfunjjb = lambda c: rw.dstor_torq__dbias(t0,sat,x0,vecs).item()
        hfunjjx = lambda c: rw.dstor_torq__dbasestate(t0,sat,x0,vecs)
        hfunjjh = lambda c: rw.dstor_torq__dh(t0,sat,x0,vecs).item()

        bfunjju = lambda c: rw.dstor_torq__du(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs).item()
        # bfunjjb = lambda c: rw.dstor_torq__dbias(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs)
        bfunjjx = lambda c: rw.dstor_torq__dbasestate(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs)
        bfunjjh = lambda c: rw.dstor_torq__dh(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False, bias = c,use_noise = False)]),x0,vecs).item()

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        # Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs))
        # assert np.allclose( Jxfunjjb , rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs))
        assert np.allclose( Jxfunjjx , rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs))
        assert np.allclose( Jxfunjjh , rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        # Jufunjjb = np.array(nd.Jacobian(ufunjjb)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , rw.ddstor_torq__dudu(t0,sat,x0,vecs))
        # assert np.allclose( Jufunjjb , rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs))
        assert np.allclose( Jufunjjx , rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs))
        assert np.allclose( Jufunjjh , rw.ddstor_torq__dudh(t0,sat,x0,vecs))
        #
        # Jbfunjju = np.array(nd.Jacobian(bfunjju)(20000))
        # # Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(20000))
        # Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(20000))
        # Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(20000))
        # assert np.allclose( Jbfunjju , rw.ddstor_torq__dudbias(t0,sat,x0,vecs))
        # # assert np.allclose( Jbfunjjb , rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs))
        # assert np.allclose( Jbfunjjx , rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs))
        # assert np.allclose( Jbfunjjh , rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(momentumr))
        # Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(momentumr))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(momentumr))
        Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(momentumr))
        assert np.allclose( Jhfunjju , rw.ddstor_torq__dudh(t0,sat,x0,vecs))
        # assert np.allclose( Jhfunjjb , rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs))
        assert np.allclose( Jhfunjjx , rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs))
        assert np.allclose( Jhfunjjh , rw.ddstor_torq__dhdh(t0,sat,x0,vecs))


    ax = unitvecs[1]
    max_torq = 4.51
    std = 0.243
    momentumr = -3.1
    max_hr = 3.8
    msnsr = 0.3
    Jr = 0.22
    stdr = 0.243
    biasr = random_n_unit_vec(3)[1]*0.1
    bsrr = 0.03
    rw = RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = False,use_noise = False)
    t0 = random_n_unit_vec(3)[0]
    B_ECI = unitvecs[2]
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose((t0)*unitvecs[1],rw.torque(t0,sat,x0,vecs)))
    assert np.all(np.isclose( rw.dtorq__du(t0,sat,x0,vecs) ,unitvecs[1] ))
    assert np.all(np.isclose( rw.dtorq__dbias(t0,sat,x0,vecs) , np.zeros((0,3)) ))
    assert np.all(rw.dtorq__dbias(t0,sat,x0,vecs).shape==(0,3))

    assert np.all(np.isclose( rw.dtorq__dbasestate(t0,sat,x0,vecs) ,np.zeros((7,3))))
    assert np.all(np.isclose( rw.dtorq__dh(t0,sat,x0,vecs) , np.zeros((1,3))))
    assert np.all(rw.dtorq__dh(t0,sat,x0,vecs).shape==(1,3))

    assert np.all(np.isclose( rw.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,3)) ))
    assert np.all(rw.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,0,3))
    assert np.all(np.isclose( rw.ddtorq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,3)) ))
    assert np.all(rw.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,1,3))

    assert np.all(np.isclose( rw.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0,3)) ))
    assert np.all(rw.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) ,np.zeros((0,7,3))))
    assert np.all(rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(0,1,3))

    dxdx = np.zeros((7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,3)) ))
    assert np.all(rw.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,3))
    assert np.all(np.isclose( rw.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dhdh(t0,sat,x0,vecs).shape==(1,1,3))


    assert np.all(rw.storage_torque(t0,sat,x0,vecs)  == -t0)
    assert np.all(np.isclose( rw.dstor_torq__du(t0,sat,x0,vecs) , -1 ))
    assert np.all(np.isclose( rw.dstor_torq__dbias(t0,sat,x0,vecs) , np.zeros((0,1)) ))
    assert np.all(rw.dstor_torq__dbias(t0,sat,x0,vecs).shape == (0,1))
    assert np.all(np.isclose( rw.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,1))))
    assert np.all(rw.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,1))
    assert np.all(np.isclose( rw.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((1,1))))
    assert np.all(rw.dstor_torq__dh(t0,sat,x0,vecs).shape==(1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,0,1)) ))
    assert np.all(rw.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,0,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((0,0,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(0,0,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((0,7,1))))
    assert np.all(rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(0,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((0,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(0,1,1))

    dxdx = np.zeros((7,7,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,1)) ))
    assert np.all(rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(1,1,1))

def test_RW_torque_etc_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bias = bias.copy()
    bsr = 0.03
    momentumr = -3.1
    max_hr = 3.8
    msnsr = 0.3
    Jr = 0.22
    rw = RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True,bias = bias,use_noise = False,bias_std_rate=bsr)


    t0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose(ax/3*(t0+bias),rw.torque(t0,sat,x0,vecs)))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    ufun = lambda c: rw.torque(c,sat,x0,vecs)
    xfun = lambda c: rw.torque(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    hfun = lambda c: RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs)
    bfun = lambda c: RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr),rw]),x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(t0)).T
    Jbfun = np.array(nd.Jacobian(bfun)(bias)).T
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T

    assert np.allclose(Jxfun, rw.dtorq__dbasestate(t0,sat,x0,vecs))
    assert np.allclose(Jufun, rw.dtorq__du(t0,sat,x0,vecs))
    assert np.allclose(Jbfun, rw.dtorq__dbias(t0,sat,x0,vecs))
    assert np.allclose(Jhfun, rw.dtorq__dh(t0,sat,x0,vecs))

    for j in unitvecs:
        fun_hj = lambda c: np.dot( RW(ax,std,max_torq,Jr,c[9],max_hr,msnsr,has_bias = True, bias = c[1],use_noise = False,bias_std_rate = bsr).torque(c[0],Satellite(actuators=[RW(ax,std,max_torq,Jr,c[9],max_hr,msnsr,has_bias = True, bias = c[1],use_noise = False,bias_std_rate = bsr)]),np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),vecsxfun(np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]))),j).item()

        ufunjju = lambda c: np.dot(rw.dtorq__du(c,sat,x0,vecs),j).item()
        ufunjjb = lambda c: np.dot(rw.dtorq__dbias(c,sat,x0,vecs),j).item()
        ufunjjx = lambda c: np.dot(rw.dtorq__dbasestate(c,sat,x0,vecs),j)
        ufunjjh = lambda c: np.dot(rw.dtorq__dh(c,sat,x0,vecs),j)

        xfunjju = lambda c: np.dot(rw.dtorq__du(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjb = lambda c: np.dot(rw.dtorq__dbias(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()
        xfunjjx = lambda c: np.dot(rw.dtorq__dbasestate(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j)
        xfunjjh = lambda c: np.dot(rw.dtorq__dh(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)),j).item()

        hfunjju = lambda c: np.dot(RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dtorq__du(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j).item()
        hfunjjb = lambda c: np.dot(RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dtorq__dbias(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j).item()
        hfunjjx = lambda c: np.dot(RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dtorq__dbasestate(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j)
        hfunjjh = lambda c: np.dot(RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dtorq__dh(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j).item()

        bfunjju = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__du(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j).item()
        bfunjjb = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbias(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j).item()
        bfunjjx = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dbasestate(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j)
        bfunjjh = lambda c: np.dot( RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dtorq__dh(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs),j).item()
        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , np.dot( rw.ddtorq__dudbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjb , np.dot( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjx , np.dot( rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jxfunjjh , np.dot( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs) , j ))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , np.dot( rw.ddtorq__dudu(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjb , np.dot( rw.ddtorq__dudbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjx.T , np.dot( rw.ddtorq__dudbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jufunjjh , np.dot( rw.ddtorq__dudh(t0,sat,x0,vecs) , j ))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(bias))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(bias))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(bias))
        Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(bias))
        assert np.allclose( Jbfunjju , np.dot( rw.ddtorq__dudbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjb , np.dot( rw.ddtorq__dbiasdbias(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjx.T , np.dot( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jbfunjjh , np.dot( rw.ddtorq__dbiasdh(t0,sat,x0,vecs) , j ))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(momentumr))
        Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(momentumr))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(momentumr))
        Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(momentumr))
        assert np.allclose( Jhfunjju , np.dot( rw.ddtorq__dudh(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjb , np.dot( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjx , np.dot( rw.ddtorq__dbiasdh(t0,sat,x0,vecs) , j ))
        assert np.allclose( Jhfunjjh , np.dot( rw.ddtorq__dhdh(t0,sat,x0,vecs) , j ))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[t0],[bias],x0,[momentumr]]).flatten().tolist()))
        test = np.block([[(rw.ddtorq__dudh(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbiasdh(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j).T,rw.ddtorq__dhdh(t0,sat,x0,vecs)@j]    ])
        Hguess = np.block([[rw.ddtorq__dudu(t0,sat,x0,vecs)@j,rw.ddtorq__dudbias(t0,sat,x0,vecs)@j,rw.ddtorq__dudbasestate(t0,sat,x0,vecs)@j,rw.ddtorq__dudh(t0,sat,x0,vecs)@j],\
                            [rw.ddtorq__dudbias(t0,sat,x0,vecs)@j,rw.ddtorq__dbiasdbias(t0,sat,x0,vecs)@j,rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j,rw.ddtorq__dbiasdh(t0,sat,x0,vecs)@j],\
                            [(rw.ddtorq__dudbasestate(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs)@j).T,rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs)@j,rw.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j],\
                            [(rw.ddtorq__dudh(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbiasdh(t0,sat,x0,vecs)@j).T,(rw.ddtorq__dbasestatedh(t0,sat,x0,vecs)@j).T,rw.ddtorq__dhdh(t0,sat,x0,vecs)@j]    ])
        # print(Hfun.shape,Hguess.shape)
        assert np.allclose(Hfun,Hguess)


    assert np.all(np.isclose( rw.dtorq__du(t0,sat,x0,vecs) , ax/3 ))
    assert np.all(np.isclose( rw.dtorq__dbias(t0,sat,x0,vecs) , ax/3 ))
    assert np.all(np.isclose( rw.dtorq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,3)) ))

    assert np.all(np.isclose( rw.dtorq__dh(t0,sat,x0,vecs) , np.zeros((1,3))))
    assert np.all(rw.dtorq__dh(t0,sat,x0,vecs).shape==(1,3))

    assert np.all(np.isclose( rw.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbasestate(t0,sat,x0,vecs) ,  np.zeros((1,7,3)) ))
    assert np.all(rw.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,1,3))

    assert np.all(np.isclose( rw.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) ,  np.zeros((1,7,3)) ))
    assert np.all(rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(1,1,3))

    dxdx = np.zeros((7,7,3))

    assert np.all(np.isclose( rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,3)) ))
    assert np.all(rw.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,3))
    assert np.all(np.isclose( rw.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dhdh(t0,sat,x0,vecs).shape==(1,1,3))

    assert np.all(rw.storage_torque(t0,sat,x0,vecs)  == -t0-bias)
    assert np.all(np.isclose( rw.dstor_torq__du(t0,sat,x0,vecs) , -1 ))
    assert np.all(np.isclose( rw.dstor_torq__dbias(t0,sat,x0,vecs) , -1 ))
    assert np.all(np.isclose( rw.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,1))))
    assert np.all(rw.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,1))
    assert np.all(np.isclose( rw.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((1,1))))
    assert np.all(rw.dstor_torq__dh(t0,sat,x0,vecs).shape==(1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(1,1,1))

    dxdx = np.zeros((7,7,0))

    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,1)) ))
    assert np.all(rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(1,1,1))

    ufun = lambda c: rw.storage_torque(c,sat,x0,vecs).item()
    xfun = lambda c: rw.storage_torque(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    hfun = lambda c: RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).storage_torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()
    bfun = lambda c: RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).storage_torque(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(t0))
    Jbfun = np.array(nd.Jacobian(bfun)(bias))
    Jhfun = np.array(nd.Jacobian(hfun)(momentumr))

    assert np.allclose(Jxfun, rw.dstor_torq__dbasestate(t0,sat,x0,vecs))
    assert np.allclose(Jufun, rw.dstor_torq__du(t0,sat,x0,vecs))
    assert np.allclose(Jbfun, rw.dstor_torq__dbias(t0,sat,x0,vecs))
    assert np.allclose(Jhfun, rw.dstor_torq__dh(t0,sat,x0,vecs))


    for j in unitvecs:
        ufunjju = lambda c: rw.dstor_torq__du(c,sat,x0,vecs).item()
        ufunjjb = lambda c: rw.dstor_torq__dbias(c,sat,x0,vecs).item()
        ufunjjx = lambda c: rw.dstor_torq__dbasestate(c,sat,x0,vecs)
        ufunjjh = lambda c: rw.dstor_torq__dh(c,sat,x0,vecs).item()

        xfunjju = lambda c: rw.dstor_torq__du(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
        xfunjjb = lambda c: rw.dstor_torq__dbias(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
        xfunjjx = lambda c: rw.dstor_torq__dbasestate(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
        xfunjjh = lambda c: rw.dstor_torq__dh(t0,sat,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()

        hfunjju = lambda c: RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dstor_torq__du(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()
        hfunjjb = lambda c: RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dstor_torq__dbias(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()
        hfunjjx = lambda c: RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dstor_torq__dbasestate(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs)
        hfunjjh = lambda c: RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr).dstor_torq__dh(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,c,max_hr,msnsr,has_bias = True, bias = bias,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()

        bfunjju = lambda c: RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dstor_torq__du(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()
        bfunjjb = lambda c: RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dstor_torq__dbias(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()
        bfunjjx = lambda c: RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dstor_torq__dbasestate(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs)
        bfunjjh = lambda c: RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr).dstor_torq__dh(t0,Satellite(actuators=[RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = c,use_noise = False,bias_std_rate = bsr)]),x0,vecs).item()

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs) )
        assert np.allclose( Jxfunjjb , rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) )
        assert np.allclose( Jxfunjjx , rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) )
        assert np.allclose( Jxfunjjh , rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , rw.ddstor_torq__dudu(t0,sat,x0,vecs))
        assert np.allclose( Jufunjjb , rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs) )
        assert np.allclose( Jufunjjx , rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs))
        assert np.allclose( Jufunjjh , rw.ddstor_torq__dudh(t0,sat,x0,vecs))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(bias))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(bias))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(bias))
        Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(bias))
        assert np.allclose( Jbfunjju , rw.ddstor_torq__dudbias(t0,sat,x0,vecs))
        assert np.allclose( Jbfunjjb , rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) )
        assert np.allclose( Jbfunjjx , rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) )
        assert np.allclose( Jbfunjjh , rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs) )

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(momentumr))
        Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(momentumr))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(momentumr))
        Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(momentumr))
        assert np.allclose( Jhfunjju , rw.ddstor_torq__dudh(t0,sat,x0,vecs))
        assert np.allclose( Jhfunjjb , rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) )
        assert np.allclose( Jhfunjjx , rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs) )
        assert np.allclose( Jhfunjjh , rw.ddstor_torq__dhdh(t0,sat,x0,vecs) )


    ax = unitvecs[1]
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    momentumr = -3.1
    max_hr = 3.8
    msnsr = 0.3
    Jr = 0.22
    stdr = 0.243
    biasr = random_n_unit_vec(3)[1]*0.1
    bsrr = 0.03
    rw = RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True,bias = bias,bias_std_rate = bsr,use_noise = False)
    t0 = random_n_unit_vec(3)[0]
    B_ECI = unitvecs[2]
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.all(np.isclose((t0+bias)*unitvecs[1],rw.torque(t0,sat,x0,vecs)))
    assert np.all(np.isclose( rw.dtorq__du(t0,sat,x0,vecs) ,unitvecs[1] ))
    assert np.all(np.isclose( rw.dtorq__dbias(t0,sat,x0,vecs) , unitvecs[1] ))

    assert np.all(np.isclose( rw.dtorq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,3)) ))
    assert np.all(np.isclose( rw.dtorq__dh(t0,sat,x0,vecs) , np.zeros((1,3))))
    assert np.all(rw.dtorq__dh(t0,sat,x0,vecs).shape==(1,3))

    assert np.all(np.isclose( rw.ddtorq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudu(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbasestate(t0,sat,x0,vecs) , np.zeros((1,7,3))  ))
    assert np.all(rw.ddtorq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudh(t0,sat,x0,vecs).shape==(1,1,3))

    assert np.all(np.isclose( rw.ddtorq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((1,7,3))  ))
    assert np.all(rw.ddtorq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdh(t0,sat,x0,vecs).shape==(1,1,3))

    dxdx = np.zeros((7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddtorq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,3)) ))
    assert np.all(rw.ddtorq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,3))
    assert np.all(np.isclose( rw.ddtorq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dhdh(t0,sat,x0,vecs).shape==(1,1,3))


    assert np.all(rw.storage_torque(t0,sat,x0,vecs)  == -(t0+bias))
    assert np.all(np.isclose( rw.dstor_torq__du(t0,sat,x0,vecs) ,-1))
    assert np.all(np.isclose( rw.dstor_torq__dbias(t0,sat,x0,vecs) , -1 ))
    assert np.all(np.isclose( rw.dstor_torq__dbasestate(t0,sat,x0,vecs) , np.zeros((7,1))))
    assert np.all(rw.dstor_torq__dbasestate(t0,sat,x0,vecs).shape == (7,1))
    assert np.all(np.isclose( rw.dstor_torq__dh(t0,sat,x0,vecs) , np.zeros((1,1))))
    assert np.all(rw.dstor_torq__dh(t0,sat,x0,vecs).shape==(1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dudu(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudu(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbias(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudbias(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs) ,np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dudbasestate(t0,sat,x0,vecs).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudh(t0,sat,x0,vecs).shape==(1,1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdbias(t0,sat,x0,vecs).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs) , np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dbiasdbasestate(t0,sat,x0,vecs).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdh(t0,sat,x0,vecs).shape==(1,1,1))

    dxdx = np.zeros((7,7,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs) , dxdx))
    assert np.all(rw.ddstor_torq__dbasestatedbasestate(t0,sat,x0,vecs).shape==(7,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs) ,np.zeros((7,1,1)) ))
    assert np.all(rw.ddstor_torq__dbasestatedh(t0,sat,x0,vecs).shape==(7,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dhdh(t0,sat,x0,vecs) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dhdh(t0,sat,x0,vecs).shape==(1,1,1))



    ax = random_n_unit_vec(3)
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    rw = RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True,bias = bias,bias_std_rate = bsr,use_noise = False)
    t0 = random_n_unit_vec(3)[0]
    B_ECI = random_n_unit_vec(3)
    q0 = zeroquat
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    N = 1000
    test_torq = rw.clean_torque(t0+bias,sat,x0,vecs)
    opts = [rw.torque(t0,sat,x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_torq,j) for j in opts]) #bias is constant

    test_storq = rw.clean_storage_torque(t0+bias,sat,x0,vecs)
    opts = [rw.storage_torque(t0,sat,x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_storq,j) for j in opts]) #bias is constant

    rw.update_bias(0.22)

    # torq_exp = magic.clean_torque(t0,sat,x0,vecs)
    torq_drift = [rw.torque(t0,sat,x0,vecs,update_bias = True,j2000 = rw.last_bias_update+0.5*sec2cent)-rw.torque(t0,sat,x0,vecs,update_bias = True,j2000 = rw.last_bias_update+0.5*sec2cent) for j in range(N)]
    exp_dist = [ax*np.random.normal(0,bsr*math.sqrt(0.5)) for j in range(N)]

    ks0 = kstest([j[0] for j in torq_drift],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in torq_drift],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in torq_drift],[j[2] for j in exp_dist])
    ind = 0
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 1
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 2
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))


    storq_drift = [rw.storage_torque(t0,sat,x0,vecs,update_bias = True,j2000 = rw.last_bias_update+0.5*sec2cent).item()-rw.storage_torque(t0,sat,x0,vecs,update_bias = True,j2000 = rw.last_bias_update+0.5*sec2cent).item() for j in range(N)]
    exp_dist = [np.random.normal(0,bsr*math.sqrt(0.5)) for j in range(N)]

    kss = kstest(storq_drift,exp_dist)
    data_a = storq_drift
    data_b = exp_dist
    hist = np.histogram([dd for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert kss.pvalue>0.1 or np.abs(kss.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    rw.update_bias(0.22)
    dt = 0.8
    oldb = rw.bias
    test_torq = rw.clean_torque(t0+oldb,sat,x0,vecs)
    last = rw.last_bias_update
    opts = [rw.torque(t0,sat,x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_torq,j) for j in opts]) #bias is constant
    rw.update_bias(0.21)
    assert rw.last_bias_update == last
    assert oldb == rw.bias

    bias_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = rw.bias
        assert np.allclose(rw.torque(t0,sat,x0,vecs), rw.clean_torque(t0+bk,sat,x0,vecs))
        rw.update_bias(t)
        assert rw.last_bias_update == t
        bias_drift += [(rw.bias-bk).item()]
        assert np.allclose(rw.torque(t0,sat,x0,vecs), rw.clean_torque(t0+rw.bias,sat,x0,vecs))
    exp_dist = [np.random.normal(0,bsr*dt) for j in range(N)]
    print(kstest(bias_drift,exp_dist).statistic)

    ks0 = kstest(bias_drift,exp_dist)
    ind = 0
    data_a = bias_drift
    data_b = exp_dist
    hist = np.histogram([dd for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    rw.update_bias(0.22)
    dt = 0.8
    oldb = rw.bias
    test_storq = rw.clean_storage_torque(t0 + oldb,sat,x0,vecs).item()
    last = rw.last_bias_update
    opts = [rw.storage_torque(t0,sat,x0,vecs).item() for j in range(N)]
    assert np.all([np.allclose(test_storq,j) for j in opts]) #bias is constant
    rw.update_bias(0.21)
    assert rw.last_bias_update == last
    assert oldb == rw.bias

    storque_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = rw.bias
        assert rw.storage_torque(t0,sat,x0,vecs) == rw.clean_storage_torque(t0+bk,sat,x0,vecs)
        rw.update_bias(t)
        assert rw.last_bias_update == t
        storque_drift += [(rw.bias-bk).item()]
        assert rw.storage_torque(t0,sat,x0,vecs) == rw.clean_storage_torque(t0+rw.bias,sat,x0,vecs)
    exp_dist = [np.random.normal(0,bsr*dt) for j in range(N)]
    print(kstest(storque_drift,exp_dist).statistic)
    kss = kstest(storque_drift,exp_dist)
    data_a = storque_drift
    data_b = exp_dist
    hist = np.histogram([dd for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert kss.pvalue>0.1 or np.abs(kss.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_RW_torque_etc_noise():
    ax = random_n_unit_vec(3)
    max_torq = 4.51
    std = 0.243
    bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    momentumr = -3.1
    max_hr = 3.8
    msnsr = 0.3
    Jr = 0.22
    rw = RW(ax,std,max_torq,Jr,momentumr,max_hr,msnsr,has_bias = True, bias = bias,use_noise = True,bias_std_rate = bsr)

    t0 = random_n_unit_vec(3)[0]*3
    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=[rw])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}
    torq_exp = rw.clean_torque(t0+bias,sat,x0,vecs)
    assert np.all(rw.clean_torque(t0+bias,sat,x0,vecs) == rw.no_noise_torque(t0,sat,x0,vecs))
    N = 1000
    torq_err = [rw.torque(t0,sat,x0,vecs,update_noise = True)-torq_exp for j in range(N)]
    exp_dist = [ax*np.random.normal(0,std) for j in range(N)]
    test_torq = rw.torque(t0,sat,x0,vecs)
    assert np.allclose(test_torq,[rw.torque(t0,sat,x0,vecs) for j in range(N)])


    ks0 = kstest([j[0] for j in torq_err],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in torq_err],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in torq_err],[j[2] for j in exp_dist])
    ind = 0
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 1
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 2
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    stor_torq_exp = rw.clean_storage_torque(t0+bias,sat,x0,vecs).item()
    N = 1000
    stor_torq_err = [rw.storage_torque(t0,sat,x0,vecs,update_noise = True).item()-stor_torq_exp for j in range(N)]
    exp_dist = [-np.random.normal(0,std) for j in range(N)]
    kss = kstest(stor_torq_err,exp_dist)
    print(np.abs(kss.statistic))
    ind = 0
    data_a = torq_err
    data_b = exp_dist
    hist = np.histogram([dd for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert kss.pvalue>0.1 or np.abs(kss.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    assert rw.control_cov() == std**2.0
    assert rw.momentum_measurement_cov() == msnsr**2.0

def test_dynamics_RW():
    maxt = [0.01,0.05,0.02]
    rwj = [0.001,0.002,0.5]
    maxh = [0.1,0.1,0.1]
    h0 = [0.1,0.0,0.0]
    bias = np.array([-0.001,0.05,0])
    acts = [RW(unitvecs[j],0,maxt[j],rwj[j],h0[j],0.1,0,has_bias = True, bias = bias[j],use_noise = False,bias_std_rate = 0.3) for j in range(3)]
    sat = Satellite(actuators = acts)
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    state = np.concatenate([0.01*unitvecs[0],zeroquat,h0])
    xd = sat.dynamics(state,np.array([0.021,-0.05,0]),os)
    print(xd)
    print(np.array([0.02/(1-0.001),0,0,0,0.005,0,0,-0.02/(1-0.001),0,0]))
    assert np.allclose(np.array([0.02/(1-0.001),0,0,0,0.005,0,0,-0.02/(1-0.001),0,0]), xd,rtol = 1e-8,atol=1e-8)

    maxt = [0.01,0.05,0.02]
    rwj = [0.001,0.002,0.5]
    maxh = [0.1,0.1,0.1]
    h0 = [0.1,0.0,0.0]
    bias = np.array([-0.001,0.05,0])
    acts = [RW(unitvecs[j],0,maxt[j],rwj[j],0,0.1,0,has_bias = True, bias = bias[j],use_noise = False,bias_std_rate = 0.3) for j in range(3)]
    sat = Satellite(actuators = acts)
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    state = np.concatenate([0.01*unitvecs[0],zeroquat,h0])
    xd = sat.dynamics(state,np.array([0.021,-0.05,0]),os)
    print(xd)
    print(np.array([0.02/(1-0.001),0,0,0,0.005,0,0,-0.02/(1-0.001),0,0]))
    assert np.allclose(np.array([0.02/(1-0.001),0,0,0,0.005,0,0,-0.02/(1-0.001),0,0]), xd,rtol = 1e-8,atol=1e-8)

@pytest.mark.slow
def test_dynamics_etc_controlled_bias():
    bias_mtq = 0.1*random_n_unit_vec(3)
    bias_magic = 0.1*random_n_unit_vec(3)
    bias_rw = 0.1*random_n_unit_vec(3)
    h_rw = 1.0*random_n_unit_vec(3)
    B_ECI = random_n_unit_vec(3)

    mtqs = [MTQ(j,0,1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    magics = [Magic(j,0,1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    rws = [RW(j,0,1,0.1,np.dot(h_rw,j),2,0,has_bias = True, bias = np.dot(bias_rw,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    qJ = random_n_unit_vec(4)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0
    H_body = J_body@w0 + h_rw
    H_ECI = J_ECI@w_ECI + Rm@h_rw
    acts = mtqs+magics+rws
    u = 5*random_n_unit_vec(9)
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    sat = Satellite(J = J_body,actuators = acts,estimated=True)
    state = np.concatenate([w0,q0,h_rw])
    exp_wd = -np.linalg.inv(sat.J_noRW)@np.cross(w0,H_body) + np.linalg.inv(sat.J_noRW)@sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9)],np.zeros(3))
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    exp_hd = sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9) if not acts[j].has_momentum],np.zeros(3)) - sat.J@exp_wd - np.cross(w0,H_body)
    xd = sat.dynamics(state,u,os)
    np.set_printoptions(precision=3)
    assert np.allclose(np.concatenate([exp_wd,exp_qd,exp_hd]),xd)

    ufun = lambda c: sat.dynamics(state,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),os)
    xfun = lambda c: sat.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),u,os)
    hfun = lambda c: Satellite(J = J_body,actuators = mtqs+magics+[RW(unitvecs[j],0,1,0.1,c[j],2,0,has_bias = True, bias = bias_rw[j],use_noise=False,bias_std_rate=0) for j in range(3)],estimated=True).dynamics(np.concatenate([w0,q0,[c[3],c[4],c[5]]]),u,os)
    bfun_mtq = lambda c: [MTQ(unitvecs[j],0,1,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_magic = lambda c: [Magic(unitvecs[i],0,1,has_bias = True, bias = c[i],use_noise=False,bias_std_rate=0,estimate_bias = True) for i in range(3)]
    bfun_rw = lambda c: [RW(unitvecs[j],0,1,0.1,h_rw[j],2,0,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_act = lambda c: bfun_mtq(c[0:3]) + bfun_magic(c[3:6]) + bfun_rw(c[6:9])
    fun_b_sat = lambda cc: Satellite(J = J_body,actuators = bfun_act(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])),estimated=True)
    bfun = lambda cc: fun_b_sat(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])).dynamics(state,u,os)

    # tfun = lambda c: Satellite(J = J_body,actuators = mtqs+magics+[RW(unitvecs[j],0,1,0.1,c[j],2,0,has_bias = True, bias = bias_rw[j],use_noise=False,bias_std_rate=0) for j in range(3)]).dynamics(state,u,os)
    # mfun = lambda c: sat.dynamics(state,u,os,add_m = np.array([c[0],c[1],c[2]]))

    biaslist = np.concatenate([bias_mtq,bias_magic,bias_rw])
    Jxfun = np.array(nd.Jacobian(xfun)(state.flatten().tolist())).T
    Jufun = np.array(nd.Jacobian(ufun)(u)).T
    Jhfun = np.array(nd.Jacobian(hfun)(np.concatenate([np.ones(3),h_rw]))).T
    Jbfun = np.array(nd.Jacobian(bfun)(biaslist)).T

    jacs = sat.dynamicsJacobians(state,u,os)# [dxdot__dx,dxdot__du,dxdot__dtorq,dxdot__dm]
    [[ddxdot__dxdx,ddxdot__dxdu,ddxdot__dxdab,ddxdot__dxdsb,ddxdot__dxddmp],[ddxdot__dudx,ddxdot__dudu,ddxdot__dudab,ddxdot__dudsb,ddxdot__duddmp],[_,_,ddxdot__dabdab,ddxdot__dabdsb,ddxdot__dabddmp],[_,_,_,ddxdot__dsbdsb,ddxdot__dsbddmp],[_,_,_,_,ddxdot__ddmpddmp]] = sat.dynamics_Hessians(state,u,os)

    assert np.allclose(xfun(state),xd)
    assert np.allclose(bfun(biaslist),xd)
    assert np.allclose(ufun(u),xd)
    assert np.allclose(hfun(np.concatenate([np.ones(3),h_rw])),xd)
    assert np.allclose(hfun(np.concatenate([np.zeros(3),h_rw])),xd)

    assert np.allclose(Jxfun, jacs[0])
    assert np.allclose(Jhfun[0:3,:], np.zeros((3,10)))
    assert np.allclose(Jhfun[3:6,:], jacs[0][7:])
    assert np.allclose(Jufun, jacs[1])
    assert np.allclose(Jbfun, jacs[2])
    assert np.allclose(np.zeros((0,10)),jacs[3])
    assert np.allclose(np.zeros((0,10)),jacs[4])

    for j in range(10):
        fun_act = lambda c: mtqs+magics+[RW(unitvecs[j],0,1,0.1,c[j],2,0,has_bias = True, bias = bias_rw[j],use_noise=False,bias_std_rate=0) for j in range(3)]
        fun_sat = lambda c:  Satellite(J = J_body,actuators = fun_act(c),estimated=True)
        fun_hj = lambda c: Satellite(J = J_body,actuators = bfun_act(np.array([c[19],c[20],c[21],c[22],c[23],c[24],c[25],c[26],c[27]])),estimated=True).dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os)[j]
        # fun_bj = lambda c: sat.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os,add_torq = np.array([c[19],c[20],c[21]]),add_m = np.array([c[22],c[23],c[24]]))[j]

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([state,u,bias_mtq,bias_magic,bias_rw]).flatten().tolist()))
        Hguess = np.block([[ddxdot__dxdx[:,:,j],ddxdot__dxdu[:,:,j],ddxdot__dxdab[:,:,j],ddxdot__dxdsb[:,:,j],ddxdot__dxddmp[:,:,j]],
                           [ddxdot__dxdu[:,:,j].T,ddxdot__dudu[:,:,j],ddxdot__dudab[:,:,j],ddxdot__dudsb[:,:,j],ddxdot__duddmp[:,:,j]],
                           [ddxdot__dxdab[:,:,j].T,ddxdot__dudab[:,:,j].T,ddxdot__dabdab[:,:,j],ddxdot__dabdsb[:,:,j],ddxdot__dabddmp[:,:,j]],
                           [ddxdot__dxdsb[:,:,j].T,ddxdot__dudsb[:,:,j].T,ddxdot__dabdsb[:,:,j].T,ddxdot__dsbdsb[:,:,j],ddxdot__dsbddmp[:,:,j]],
                           [ddxdot__dxddmp[:,:,j].T,ddxdot__duddmp[:,:,j].T,ddxdot__dabddmp[:,:,j].T,ddxdot__dsbddmp[:,:,j].T,ddxdot__ddmpddmp[:,:,j]]])
        # print(Hfun.shape,Hguess.shape)
        np.set_printoptions(precision=3)
        assert np.allclose(Hfun,Hguess)

        ufunjju = lambda c: sat.dynamicsJacobians(state,c,os)[1][:,j]
        ufunjjx = lambda c: sat.dynamicsJacobians(state,c,os)[0][:,j]
        ufunjjb = lambda c: sat.dynamicsJacobians(state,c,os)[2][:,j]
        ufunjjs = lambda c: sat.dynamicsJacobians(state,c,os)[3][:,j]
        ufunjjd = lambda c: sat.dynamicsJacobians(state,c,os)[4][:,j]

        xfunjju = lambda c: sat.dynamicsJacobians(c,u,os)[1][:,j]
        xfunjjx = lambda c: sat.dynamicsJacobians(c,u,os)[0][:,j]
        xfunjjb = lambda c: sat.dynamicsJacobians(c,u,os)[2][:,j]
        xfunjjs = lambda c: sat.dynamicsJacobians(c,u,os)[3][:,j]
        xfunjjd = lambda c: sat.dynamicsJacobians(c,u,os)[4][:,j]

        bfunjju = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        bfunjjx = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        bfunjjb = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        bfunjjs = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        bfunjjd = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(state.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(state.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(state.flatten().tolist()))
        # Jxfunjjs = np.array(nd.Jacobian(xfunjjs)(state.flatten().tolist()))
        # Jxfunjjd = np.array(nd.Jacobian(xfunjjd)(state.flatten().tolist()))
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j])
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j].T)
        assert np.allclose( Jxfunjju.T , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jxfunjjb.T , ddxdot__dxdab[:,:,j])
        # assert np.allclose( Jxfunjjs.T , ddxdot__dxdsb[:,:,j])
        # assert np.allclose( Jxfunjjd.T , ddxdot__dxdddmp[:,:,j])

        Jufunjju = np.array(nd.Jacobian(ufunjju)(u.flatten().tolist()))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(u.flatten().tolist()))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(u.flatten().tolist()))
        # Jufunjjs = np.array(nd.Jacobian(ufunjjs)(u.flatten().tolist()))
        # Jufunjjd = np.array(nd.Jacobian(ufunjjd)(u.flatten().tolist()))
        assert np.allclose( Jufunjjx , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jufunjju.T , ddxdot__dudu[:,:,j])
        assert np.allclose( Jufunjju , ddxdot__dudu[:,:,j])
        assert np.allclose( Jufunjjb , ddxdot__dudab[:,:,j])
        # assert np.allclose( Jufunjjs , ddxdot__dudsb[:,:,j])
        # assert np.allclose( Jufunjjd , ddxdot__duddmp[:,:,j])

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(biaslist.flatten().tolist()))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(biaslist.flatten().tolist()))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(biaslist.flatten().tolist()))
        # Jbfunjjs = np.array(nd.Jacobian(bfunjjs)(biaslist.flatten().tolist()))
        # Jbfunjjd = np.array(nd.Jacobian(bfunjjd)(biaslist.flatten().tolist()))
        print(Jbfunjjx)
        print(ddxdot__dxdab[:,:,j])
        print(np.isclose( Jbfunjjx , ddxdot__dxdab[:,:,j]).astype(int))
        print(Hfun[0:10,19:27])
        assert np.allclose( Jbfunjjx , ddxdot__dxdab[:,:,j])
        print(Jbfunjju)
        print(ddxdot__dudab[:,:,j])
        print(np.isclose( Jbfunjju , ddxdot__dudab[:,:,j]).astype(int))
        assert np.allclose( Jbfunjju , ddxdot__dudab[:,:,j])
        assert np.allclose( Jbfunjjb , ddxdot__dabdab[:,:,j])
        assert np.allclose( Jbfunjjb.T , ddxdot__dabdab[:,:,j])
        # assert np.allclose( Jbfunjjs , ddxdot__dabdsb[:,:,j])
        # assert np.allclose( Jbfunjjd , ddxdot__dabddmp[:,:,j])

def test_torque_mag_dist():

    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    q0 =zeroquat
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    res_dipole = Dipole_Disturbance(unitvecs[0],False,0)
    sat = Satellite(disturbances = [res_dipole])
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([]),os)
    assert np.all(np.array([0,0,0,0,0.005,0,0]) == xd)
    assert np.all(0 ==  res_dipole.torque(sat,vecs))
    res_dipole = Dipole_Disturbance(unitvecs[1],False,0)
    sat = Satellite(disturbances = [res_dipole])
    # os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5)
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([]),os)
    assert np.all(unitvecs[2]*-1e-5 ==  res_dipole.torque(sat,vecs))
    assert np.all(np.array([0,0,-1e-5,0,0.005,0,0]) == xd)
    res_dipole = Dipole_Disturbance(unitvecs[2],False,0)
    sat = Satellite(disturbances = [res_dipole])
    xd = sat.dynamics(np.array([0.01,0,0,1,0,0,0]),np.array([]),os)
    assert np.all(unitvecs[1]*1e-5 ==  res_dipole.torque(sat,vecs))
    assert np.all(np.array([0,1e-5,0,0,0.005,0,0]) == xd)
    qJ = random_n_unit_vec(4)
    dip = random_n_unit_vec(3)
    B_ECI = 1e-1*random_n_unit_vec(3)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    dip_ECI = Rm@dip
    torq_ECI = -np.cross(B_ECI,dip_ECI)
    exp_wd = Rm.T@np.linalg.inv(J_ECI)@(-np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    res_dipole = Dipole_Disturbance(dip,False,0)
    sat = Satellite(J=J_body,disturbances=[res_dipole])
    state = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}
    xd = sat.dynamics(state,[],os)
    np.set_printoptions(precision=3)
    print(xd,np.concatenate([exp_wd,exp_qd]))
    print(np.concatenate([exp_wd,exp_qd]) - xd)
    print(Rm.T@np.linalg.inv(J_ECI)@(torq_ECI))
    assert np.allclose(Rm.T@torq_ECI,res_dipole.torque(sat,vecs))
    assert np.allclose(Rm.T@torq_ECI,res_dipole.torque(sat,vecs))
    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),xd))

def test_torque_gen_dist():
    t0 = 3.2*random_n_unit_vec(3)
    gen_torq = General_Disturbance(t0)
    qJ = random_n_unit_vec(4)
    dip = random_n_unit_vec(3)
    B_ECI = 1e-5*random_n_unit_vec(3)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    torq_ECI = Rm@t0
    exp_wd = Rm.T@np.linalg.inv(J_ECI)@(-np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    res_dipole = Dipole_Disturbance(unitvecs[0],False,0)
    sat = Satellite(J=J_body,disturbances=[gen_torq])
    state = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    xd = sat.dynamics(state,[],os)
    print(xd,np.concatenate([exp_wd,exp_qd]))
    print(np.concatenate([exp_wd,exp_qd]) - xd)
    assert np.allclose(Rm.T@torq_ECI,gen_torq.torque(sat,vecs))
    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),xd))

def test_torque_prop_dist():
    t0 = 3.2*random_n_unit_vec(3)
    prop_torq = Prop_Disturbance(t0)
    qJ = random_n_unit_vec(4)
    dip = random_n_unit_vec(3)
    B_ECI = 1e-5*random_n_unit_vec(3)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    torq_ECI = Rm@t0
    exp_wd = Rm.T@np.linalg.inv(J_ECI)@(-np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    res_dipole = Dipole_Disturbance(unitvecs[0],False,0)
    sat = Satellite(J=J_body,disturbances=[prop_torq])
    state = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    xd = sat.dynamics(state,[],os)
    assert np.allclose(Rm.T@torq_ECI,prop_torq.torque(sat,vecs))
    assert np.allclose(t0,prop_torq.torque(sat,vecs))
    print(xd)
    print(np.concatenate([exp_wd,exp_qd]))
    print(xd-np.concatenate([exp_wd,exp_qd]))
    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),xd))

def test_torque_drag_dist():#TODO: more tests

    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),2.2]]
    drag = Drag_Disturbance(face)
    sat = Satellite(disturbances = [drag])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4)
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.allclose(drag.torque(sat,vecs),-0.5*3.4*1.2*2.2*8*8*1e6*unitvecs[2])
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([-0.5*3.4*1.2*2.2*8*8*1e6*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)

    face = [[0,1.2,np.array([1,0,0]),np.array([0,-1,0]),2.2]]
    drag = Drag_Disturbance(face)
    sat = Satellite(disturbances = [drag])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4)
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.allclose(drag.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),2.2],[0,1.2,np.array([-1,0,0]),np.array([0,1,0]),2.2]]
    drag = Drag_Disturbance(face)
    sat = Satellite(disturbances = [drag])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4)
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.allclose(drag.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([1,0,0]),2.2]]
    drag = Drag_Disturbance(face)
    sat = Satellite(disturbances = [drag])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4)
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.allclose(drag.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)


    face = [[0,1.2,np.array([0,1,0]),np.array([0,1,0]),2.2]]
    drag = Drag_Disturbance(face)
    sat = Satellite(disturbances = [drag])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4)
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.allclose(drag.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([1,0,0]),2.2],
            [0,1.2,np.array([-1,0,0]),np.array([-1,0,0]),2.2],
            [0,1.2,np.array([0,1,0]),np.array([0,1,0]),2.2],
            [0,1.2,np.array([0,-1,0]),np.array([0,-1,0]),2.2],
            [0,1.2,np.array([0,0,1]),np.array([0,0,1]),2.2],
            [0,1.2,np.array([0,0,-1]),np.array([0,0,-1]),2.2]]
    drag = Drag_Disturbance(face)
    sat = Satellite(disturbances = [drag])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4)
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.allclose(drag.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([1,0,0]),2.2],
            [0,1.2,np.array([-1,0,0]),np.array([-1,0,0]),2.2],
            [0,1.2,np.array([0,1,0]),np.array([0,1,0]),2.2],
            [0,1.2,np.array([0,-1,0]),np.array([0,-1,0]),2.2],
            [0,1.2,np.array([0,0,1]),np.array([0,0,1]),2.2],
            [0,1.2,np.array([0,0,-1]),np.array([0,0,-1]),2.2]]
    drag = Drag_Disturbance(face)
    sat = Satellite(disturbances = [drag])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4)
    q0 = np.array([1/math.sqrt(2),1/math.sqrt(2),0,0])
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    assert np.allclose(drag.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[-0.5*0.01/math.sqrt(2)],0.005*unitvecs[0]/math.sqrt(2)]) == xd)

def test_torque_srp_dist():#TODO: more tests

    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),1,0,0]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.221,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([0,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    print(srp.torque(sat,vecs))
    print(-solar_constant*1.2/c*unitvecs[2])
    assert np.allclose(srp.torque(sat,vecs),-solar_constant*1.2/c*unitvecs[2])
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([-solar_constant*1.2/c*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)

    face = [[0,1.2,np.array([1,0,0]),np.array([0,-1,0]),1,0,0]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = 1e12*unitvecs[1])
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    assert np.allclose(srp.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)



    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([0,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    assert np.allclose(srp.torque(sat,vecs),-solar_constant*1.2*(0.05+(5/3)*0.25 + 0.7*2)/c*unitvecs[2])
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([-solar_constant*1.2*(0.05+(5/3)*0.25 + 0.7*2)/c*unitvecs[2],[0],0.005*unitvecs[0]]), xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([0,-1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = 1e12*unitvecs[1])
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    assert np.allclose(srp.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),0,0,1]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([1e12,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    cg = 1/math.sqrt(2)
    assert np.allclose(srp.torque(sat,vecs),-solar_constant*(1.2/c)*(unitvecs[2]*(2*cg*cg)))
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([-solar_constant*(1.2/c)*(unitvecs[2]*(2*cg*cg) ),[0],0.005*unitvecs[0]]), xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),0,1,0]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([1e12,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    cg = 1/math.sqrt(2)
    assert np.allclose(srp.torque(sat,vecs),-solar_constant*(1.2/c)*(unitvecs[2]*(2*cg/3 ) + cg*unitvecs[2]/math.sqrt(2)))
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([-solar_constant*(1.2/c)*(unitvecs[2]*(2*cg/3 ) + cg*unitvecs[2]/math.sqrt(2)),[0],0.005*unitvecs[0]]), xd)


    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),1,0,0]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([1e12,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    cg = 1/math.sqrt(2)
    assert np.allclose(srp.torque(sat,vecs),-solar_constant*(1.2/c)*(cg*unitvecs[2]/math.sqrt(2)))
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([-solar_constant*(1.2/c)*( cg*unitvecs[2]/math.sqrt(2)),[0],0.005*unitvecs[0]]), xd)










    face = [[0,1.2,np.array([1,0,0]),np.array([0,1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([1e12,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    cg = 1/math.sqrt(2)
    # F_a = outer(self.eta_a,s_body)
    # F_d = self.eta_d*(2*self.normals/3 + s_body)
    # F_s = 2*self.eta_s*cos_gamma*self.normals
    assert np.allclose(srp.torque(sat,vecs),-solar_constant*(1.2/c)*(unitvecs[2]*(2*0.25*cg/3 + 2*cg*cg*0.7) + cg*(0.05 + 0.25)*unitvecs[2]/math.sqrt(2)))
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([-solar_constant*(1.2/c)*(unitvecs[2]*(2*0.25*cg/3 + 2*cg*cg*0.7) + cg*(0.05 + 0.25)*unitvecs[2]/math.sqrt(2)),[0],0.005*unitvecs[0]]), xd)





    face = [[0,1.2,np.array([1,0,0]),np.array([0,-1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = 1e12*unitvecs[1])
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    assert np.allclose(srp.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.all(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) == xd)



    face = [[0,1.2,np.array([0,0,1]),np.array([0,1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([1e12,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    cg = 1/math.sqrt(2)
    # F_a = outer(self.eta_a,s_body)
    # F_d = self.eta_d*(2*self.normals/3 + s_body)
    # F_s = 2*self.eta_s*cos_gamma*self.normals
    assert np.allclose(srp.torque(sat,vecs),solar_constant*(1.2/c)*(unitvecs[0]*(2*0.25*cg/3 + 2*cg*cg*0.7) + cg*(0.05 + 0.25)*(unitvecs[0]-unitvecs[1])/math.sqrt(2)))
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([solar_constant*(1.2/c)*(unitvecs[0]*(2*0.25*cg/3 + 2*cg*cg*0.7) + cg*(0.05 + 0.25)*(unitvecs[0]-unitvecs[1])/math.sqrt(2)),[0],0.005*unitvecs[0]]) , xd)



    face = [[0,1.2,np.array([0,1,0]),np.array([0,1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([0,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    assert np.allclose(srp.torque(sat,vecs),0*unitvecs[2])
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([0*unitvecs[2],[0],0.005*unitvecs[0]]) , xd)


    face = [[0,1.2,np.array([0,1,0]),np.array([0,1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([1e12,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    cg = 1/math.sqrt(2)
    # F_a = outer(self.eta_a,s_body)
    # F_d = self.eta_d*(2*self.normals/3 + s_body)
    # F_s = 2*self.eta_s*cos_gamma*self.normals
    print(srp.torque(sat,vecs))
    print(solar_constant*(1.2/c)*( cg*(0.05 + 0.25)*unitvecs[2]/math.sqrt(2)))
    assert np.allclose(srp.torque(sat,vecs),solar_constant*(1.2/c)*( cg*(0.05 + 0.25)*unitvecs[2]/math.sqrt(2)))
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([solar_constant*(1.2/c)*( cg*(0.05 + 0.25)*unitvecs[2]/math.sqrt(2)),[0],0.005*unitvecs[0]]) , xd)


    face = [[0,1.2,np.array([0,0,1]),np.array([0,1,0]),0.05,0.25,0.7],[0,1.2,np.array([0,0,-1]),np.array([0,1,0]),0.05,0.25,0.7]]
    srp = SRP_Disturbance(face)
    sat = Satellite(disturbances = [srp])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=np.array([1,0,0])*1e-5,rho = 3.4,S = np.array([1e12,1e12,0]))
    q0 = zeroquat
    w0 = 0.01*unitvecs[0]
    x0 = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = np.array([1,0,0])*1e-5
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    cg = 1/math.sqrt(2)
    # F_a = outer(self.eta_a,s_body)
    # F_d = self.eta_d*(2*self.normals/3 + s_body)
    # F_s = 2*self.eta_s*cos_gamma*self.normals
    assert np.allclose(srp.torque(sat,vecs),0)
    xd = sat.dynamics(x0,np.array([]),os)
    assert np.allclose(np.concatenate([np.zeros(3),[0],0.005*unitvecs[0]]) , xd)

def test_torque_gg_dist():
    Nmass = 5
    masses = [(np.random.uniform(0,3),random_n_unit_vec(3)*np.random.uniform(0,2)) for j in range(Nmass)]
    msphere = 1
    Jsphere = np.eye(3)*1*(2/5)*(0.1)*0.1 #0.1m radius, solid, 1 kg mass.

    gg = GG_Disturbance()

    m0 = sum([j[0] for j in masses]) + msphere #msphere kg sphere at com
    com0 = sum([j[0]*j[1] for j in masses],np.zeros(3))/(m0-msphere)
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    J0 = sum([j[0]*(np.eye(3)*np.dot(j[1],j[1])-np.outer(j[1],j[1])) for j in masses],Jsphere+msphere*(np.eye(3)*np.dot(com0,com0)-np.outer(com0,com0)))
    J = J0#-m0*(np.eye(3)*np.dot(com0,com0) - np.outer(com0,com0))
    # print(J)
    # # print(np.dot(masses[0][1],masses[0][1]))
    # # print(np.eye(3)*np.dot(masses[0][1],masses[0][1]))
    # # print(np.outer(masses[0][1],masses[0][1]))
    # # print(np.eye(3)*np.dot(masses[0][1],masses[0][1])-np.outer(masses[0][1],masses[0][1]))
    # print(sum([j[0]*(np.eye(3)*np.dot(j[1],j[1])-np.outer(j[1],j[1])) for j in masses],np.zeros((3,3))))
    # print(Jsphere+msphere*(np.eye(3)*np.dot(com0,com0)-np.outer(com0,com0)))
    # print(com0)
    # print(m0)
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    w0 = np.zeros(3)
    poss = [Rm@(j[1]-com0) for j in masses]
    spos = np.zeros(3)
    forces = [-mu_e*1e9*j[0]*(os.R*1e3+Rm@j[1])/norm(os.R*1e3+Rm@j[1])**3.0 for j in masses]
    sforce = -mu_e*1e9*msphere*(os.R*1e3+Rm@com0)/norm(os.R*1e3+Rm@com0)**3.0
    torques = [np.cross(poss[j],forces[j]) for j in range(len(masses))]
    torq_ECI = sum(torques,np.zeros(3))
    # torq_ECI = sum([np.cross(Rm@(j[1]-com0)/1000,-mu_e*j[0]*(os.R+Rm@j[1]/1000)/norm(os.R+Rm@j[1]/1000)**3.0) for j in masses],np.zeros(3))*1e6
    sat = Satellite(J=J,disturbances=[gg],mass=m0,COM=com0)
    assert np.all(sat.COM == com0)
    assert np.all(sat.J_given == J0)
    assert sat.mass == m0
    print(sat.J)
    print(sum([j[0]*(np.eye(3)*np.dot(j[1]-com0,j[1]-com0)-np.outer(j[1]-com0,j[1]-com0)) for j in masses],Jsphere))
    print(Jsphere)
    print([j[0]*(np.eye(3)*np.dot(j[1]-com0,j[1]-com0)-np.outer(j[1]-com0,j[1]-com0)) for j in masses])
    assert np.allclose(sat.J, sum([j[0]*(np.eye(3)*np.dot(j[1]-com0,j[1]-com0)-np.outer(j[1]-com0,j[1]-com0)) for j in masses],Jsphere))

    state = np.concatenate([w0,q0])
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}

    xd = sat.dynamics(state,[],os)
    print(masses)
    print(J)
    print(Rm.T@torq_ECI)
    print(gg.torque(sat,vecs))

    print(Rm.T@torq_ECI-gg.torque(sat,vecs))
    print(forces)
    print(poss)
    print('force',sum([-mu_e*1e9*j[0]*(os.R*1e3+Rm@j[1])/norm(os.R*1e3+Rm@j[1])**3.0 for j in masses],np.zeros(3))-mu_e*1e9*msphere*(os.R*1e3+Rm@com0)/norm(os.R*1e3+Rm@com0)**3.0,sum(forces)+sforce)
    print('exp force',-mu_e*1e9*m0*(os.R*1e3+Rm@com0)/norm(os.R*1e3+Rm@com0)**3.0)
    assert np.allclose(Rm.T@torq_ECI,gg.torque(sat,vecs))
    print(xd)
    print(np.linalg.inv(sum([j[0]*(np.eye(3)*np.dot(j[1]-com0,j[1]-com0)-np.outer(j[1]-com0,j[1]-com0)) for j in masses],Jsphere))@Rm.T@torq_ECI)
    assert np.all(np.isclose(np.concatenate([np.linalg.inv(sum([j[0]*(np.eye(3)*np.dot(j[1]-com0,j[1]-com0)-np.outer(j[1]-com0,j[1]-com0)) for j in masses],Jsphere))@Rm.T@torq_ECI,np.zeros(4)]),xd))
    # assert 1==0

def test_prop_dist_update():
    std = 0.2
    max_t = 1.3
    dist = Prop_Disturbance([np.array([0.1,-0.1,0.5]),max_t],True,std)
    dist.last_update = 0.22
    # sat = Satellite(disturbances=[dist])
    assert np.all(dist.main_param == np.array([0.1,-0.1,0.5]))
    assert np.all(dist.std == std*np.eye(3))
    assert np.all(dist.mag_max == max_t)
    dist.update(0.21)
    assert dist.last_update == 0.22
    assert np.all(dist.main_param == np.array([0.1,-0.1,0.5]))
    N = 10000
    dt = 0.7
    tlist = [0.22+(j+1)*dt/cent2sec for j in range(N)]
    torq_list = [dist.update(j).main_param for j in tlist]
    torq_prev_jumps = [torq_list[0]-np.array([0.1,-0.1,0.5])]+[torq_list[j+1]-torq_list[j] for j in range(len(torq_list)-1)]
    mag_list =np.array([norm(j) for j in torq_list])
    assert np.all(mag_list<=max_t+1e-8)
    capped_inds = np.where(~(np.array(mag_list)[1:-1]<max_t))[0]+1 #ignore last one to make test simpler so don't have to deal with edge case where last one or first is capped
    after_capped_inds = capped_inds + 1
    before_capped_inds = capped_inds - 1
    #50% chance that dipole after capped dipole is also capped
    # assert np.abs(sum(~(mag_list[after_capped_inds]<max_m))/len(capped_inds)-0.5)<0.01
    #those that didn't hit rails should follow normal distribution
    # clean_inds = [j for j in range(1,len(dipole_prev_jumps))  if (((j-1) in capped inds) and (j not in capped_inds))]
    clean_jumps = [torq_prev_jumps[j] for j in range(1,len(torq_prev_jumps)) if (((j-1) not in capped_inds) and (j not in capped_inds))]
    exp_dist = [np.random.normal(np.zeros(3),std*dt) for j in range(N)]

    ks0 = kstest([j[0] for j in clean_jumps],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in clean_jumps],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in clean_jumps],[j[2] for j in exp_dist])
    Nc = len(clean_jumps)
    ind = 0
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    ind = 1
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    ind = 2
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    #jumps that did hit rails should have a likelihood of occurence proportional with the previous distance from the rail -- TODO

def test_gen_dist_update():
    std = 0.2
    max_t = 1.3
    dist = General_Disturbance([np.array([0.1,-0.1,0.5]),max_t],True,std)
    dist.last_update = 0.22
    # sat = Satellite(disturbances=[dist])
    assert np.all(dist.main_param == np.array([0.1,-0.1,0.5]))
    assert np.all(dist.std == std*np.eye(3))
    assert np.all(dist.mag_max == max_t)
    dist.update(0.21)
    assert dist.last_update == 0.22
    assert np.all(dist.main_param == np.array([0.1,-0.1,0.5]))
    N = 10000
    dt = 0.7
    tlist = [0.22+(j+1)*dt/cent2sec for j in range(N)]
    torq_list = [dist.update(j).main_param for j in tlist]
    torq_prev_jumps = [torq_list[0]-np.array([0.1,-0.1,0.5])]+[torq_list[j+1]-torq_list[j] for j in range(len(torq_list)-1)]
    mag_list =np.array([norm(j) for j in torq_list])
    assert np.all(mag_list<=max_t+1e-8)
    capped_inds = np.where(~(np.array(mag_list)[1:-1]<max_t))[0]+1 #ignore last one to make test simpler so don't have to deal with edge case where last one or first is capped
    after_capped_inds = capped_inds + 1
    before_capped_inds = capped_inds - 1
    #50% chance that dipole after capped dipole is also capped
    # assert np.abs(sum(~(mag_list[after_capped_inds]<max_m))/len(capped_inds)-0.5)<0.01
    #those that didn't hit rails should follow normal distribution
    # clean_inds = [j for j in range(1,len(dipole_prev_jumps))  if (((j-1) in capped inds) and (j not in capped_inds))]
    clean_jumps = [torq_prev_jumps[j] for j in range(1,len(torq_prev_jumps)) if (((j-1) not in capped_inds) and (j not in capped_inds))]
    exp_dist = [np.random.normal(np.zeros(3),std*dt) for j in range(N)]

    ks0 = kstest([j[0] for j in clean_jumps],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in clean_jumps],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in clean_jumps],[j[2] for j in exp_dist])
    Nc = len(clean_jumps)
    ind = 0
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    ind = 1
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    ind = 2
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    #jumps that did hit rails should have a likelihood of occurence proportional with the previous distance from the rail -- TODO

def test_mag_dist_update():
    std = 0.2
    max_m = 1.3
    dist = Dipole_Disturbance([np.array([0.1,-0.1,0.5]),max_m],True,std)
    dist.last_update = 0.22
    # sat = Satellite(disturbances=[dist])
    assert np.all(dist.main_param == np.array([0.1,-0.1,0.5]))
    assert np.all(dist.std == std*np.eye(3))
    assert np.all(dist.mag_max == max_m)
    dist.update(0.21)
    assert dist.last_update == 0.22
    assert np.all(dist.main_param == np.array([0.1,-0.1,0.5]))
    N = 10000
    dt = 0.7
    tlist = [0.22+(j+1)*dt/cent2sec for j in range(N)]
    dipole_list = [dist.update(j).main_param for j in tlist]
    dipole_prev_jumps = [dipole_list[0]-np.array([0.1,-0.1,0.5])]+[dipole_list[j+1]-dipole_list[j] for j in range(len(dipole_list)-1)]
    mag_list =np.array([norm(j) for j in dipole_list])
    assert np.all(mag_list<=max_m+1e-8)
    capped_inds = np.where(~(np.array(mag_list)[1:-1]<max_m))[0]+1 #ignore last one to make test simpler so don't have to deal with edge case where last one or first is capped
    after_capped_inds = capped_inds + 1
    before_capped_inds = capped_inds - 1
    #50% chance that dipole after capped dipole is also capped
    # assert np.abs(sum(~(mag_list[after_capped_inds]<max_m))/len(capped_inds)-0.5)<0.01
    #those that didn't hit rails should follow normal distribution
    # clean_inds = [j for j in range(1,len(dipole_prev_jumps))  if (((j-1) in capped inds) and (j not in capped_inds))]
    clean_jumps = [dipole_prev_jumps[j] for j in range(1,len(dipole_prev_jumps)) if (((j-1) not in capped_inds) and (j not in capped_inds))]
    exp_dist = [np.random.normal(np.zeros(3),std*dt) for j in range(N)]

    ks0 = kstest([j[0] for j in clean_jumps],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in clean_jumps],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in clean_jumps],[j[2] for j in exp_dist])
    Nc = len(clean_jumps)
    ind = 0
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    ind = 1
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    ind = 2
    data_a = clean_jumps
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]*N/Nc).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
    #jumps that did hit rails should have a likelihood of occurence proportional with the previous distance from the rail -- TODO

def test_sat_dist_update():
    dipole_std = 0.12
    max_m = 1.3
    gen_std = 0.25
    max_t_gen = 0.9
    prop_std = 0.39
    max_t_prop = 1.7

    dists = [Dipole_Disturbance([np.random.uniform(0.2,0.75*max_m)*random_n_unit_vec(3),max_m],True,dipole_std),\
            General_Disturbance([np.random.uniform(0.2,0.75*max_t_gen)*random_n_unit_vec(3),max_t_gen],True,gen_std),\
            Prop_Disturbance([np.random.uniform(0.2,0.75*max_t_prop)*random_n_unit_vec(3),max_t_prop],True,prop_std)]
    for j in dists:
        j.last_update = 0.22
    sat = Satellite(disturbances = dists)
    N = 1001
    dt = np.random.uniform(0.3,1.8)
    tlist = [0.22+(j+1)*dt/cent2sec for j in range(N)]
    dipole_list = []
    gen_torq_list = []
    prop_torq_list = []
    val_lists = [[],[],[]]
    for j in tlist:
        sat.update_disturbances(j)
        val_lists[0] += [sat.disturbances[0].main_param]
        val_lists[1] += [sat.disturbances[1].main_param]
        val_lists[2] += [sat.disturbances[2].main_param]
    prev_jumps = [[val_lists[i][j+1]-val_lists[i][j] for j in range(1,len(val_lists[i])-1)] for i in range(3)]
    mag_lists = [np.array([norm(j) for j in val_lists[i]]) for i in range(3)]
    maxes = [max_m,max_t_gen,max_t_prop]
    stds = [dipole_std,gen_std,prop_std]
    capped_inds = [np.where(~(np.array(mag_lists[i])[1:-1]<maxes[i]))[0]+1 for i in range(3)] #ignore last one to make test simpler so don't have to deal with edge case where last one or first is capped
    after_capped_inds = [j + 1 for j in capped_inds]
    before_capped_inds = [j - 1 for j in capped_inds]
    #50% chance that dipole after capped dipole is also capped
    # assert np.abs(sum(~(mag_list[after_capped_inds]<max_m))/len(capped_inds)-0.5)<0.01
    #those that didn't hit rails should follow normal distribution
    # clean_inds = [j for j in range(1,len(dipole_prev_jumps))  if (((j-1) in capped inds) and (j not in capped_inds))]
    clean_jumps = [[prev_jumps[i][j] for j in range(1,len(prev_jumps[i])) if (((j-1) not in capped_inds[i]) and (j not in capped_inds[i]))] for i in range(3)]
    exp_dist = [[np.random.normal(np.zeros(3),stds[i]*dt) for j in range(N)] for i in range(3)]

    for i in range(3):
        ks0 = kstest([j[0] for j in clean_jumps[i]],[j[0] for j in exp_dist[i]])
        ks1 = kstest([j[1] for j in clean_jumps[i]],[j[1] for j in exp_dist[i]])
        ks2 = kstest([j[2] for j in clean_jumps[i]],[j[2] for j in exp_dist[i]])
        Nc = len(clean_jumps)
        ind = 0
        data_a = clean_jumps[i]
        data_b = exp_dist[i]
        hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]*N/Nc).tolist()
        hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
        ind = 1
        data_a = clean_jumps[i]
        data_b = exp_dist[i]
        hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]*N/Nc).tolist()
        hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
        ind = 2
        data_a = clean_jumps[i]
        data_b = exp_dist[i]
        hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]*N/Nc).tolist()
        hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/Nc)*-0.5*np.log(1e-5/3/(Nc+1))))
        #jumps that did hit rails should have a likelihood of occurence proportional with the previous distance from the rail -- TODO

@pytest.mark.slow
def test_disturbance_torque_jacs_and_hesses():
    #setup each disturbances
    #dipole\
    np.set_printoptions(precision=3)
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,10)
    B_ECI = random_n_unit_vec(3)*np.random.uniform(0.5,2)
    # q0 = zeroquat
    R_ECI = random_n_unit_vec(3)*np.random.uniform(6900,7800)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    os = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,rho = np.random.uniform(1e-15,1e-8),S=S_ECI)
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = Rm.T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dipole = Dipole_Disturbance([random_n_unit_vec(3)*np.random.uniform(0,5)])
    Ndrag = 30
    drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
    while np.any([np.abs(np.pi/2-np.arccos(j))<4.0*np.pi/180.0  for j in np.dot(drag.normals,normalize(V_B))]): #numeric differentiation gets messed up near the corners
        drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
    # drag = Drag_Disturbance([[0,1,unitvecs[0],unitvecs[2],2]])

    gen = General_Disturbance([random_n_unit_vec(3)*np.random.uniform(0,5)])
    gg = GG_Disturbance()
    prop = Prop_Disturbance([random_n_unit_vec(3)*np.random.uniform(0,5)])
    srp_eta_a = [np.random.uniform(0.1,0.9) for j in range(5)]
    srp_eta_d = [np.random.uniform(0.05,0.95-srp_eta_a[j]) for j in range(5)]
    srp_eta_s = [1-srp_eta_a[j]-srp_eta_d[j] for j in range(5)]
    srp = SRP_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),srp_eta_a[j],srp_eta_d[j],srp_eta_s[j]] for j in range(5)])
    while np.any([np.abs(np.pi/2-np.arccos(j))<4.0*np.pi/180.0  for j in np.dot(srp.normals,normalize(S_B-R_B))]): #numeric differentiation gets messed up near the corners
        srp = SRP_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),srp_eta_a[j],srp_eta_d[j],srp_eta_s[j]] for j in range(5)])

    dists = [dipole,drag,gen,gg,prop,srp]

    qJ = random_n_unit_vec(4)
    # qJ = zeroquat
    print(q0)
    J0 = np.diagflat(np.abs(random_n_unit_vec(3))*np.random.uniform(5,20))
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    com = random_n_unit_vec(3)*np.random.uniform(0.1,10)
    m = np.random.uniform(5,50)
    J_body = RJ@J0@RJ.T - m*skewsym(com)@skewsym(com)
    sat = Satellite(mass=m,COM=com,J = J_body,disturbances=dists)
    assert np.allclose(sat.J_given,J_body)
    assert np.allclose(sat.J,RJ@J0@RJ.T)
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@B,"r":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@R,"s":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@S,"v":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),B),"ds":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),S),"dv":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),V),"dr":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),B),"dds":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),R),"os":os}


    for j in range(len(dists)):
        print(j)
        dj = dists[j]
        qfun = lambda c: dj.torque(sat,vecsxfun(c))
        Jqfun = np.array(nd.Jacobian(qfun)(q0.flatten().tolist())).T
        calc_jac = dj.torque_qjac(sat,vecs)
        assert np.allclose(qfun(q0),dj.torque(sat,vecs))
        assert np.allclose(Jqfun, calc_jac)
        for i in unitvecs:
            fun_hi = lambda c: np.dot( dj.torque(sat,vecsxfun(c)),i).item()
            assert np.allclose(fun_hi(q0),np.dot(dj.torque(sat,vecs),i).item())

            Hfun = np.array(nd.Hessian(fun_hi)(q0.flatten().tolist()))
            Hguess = np.dot(dj.torque_qqhess(sat,vecs),i)
            assert np.allclose(Hfun,Hguess)

            Jacifunq = lambda c: np.dot(dj.torque_qjac(sat,vecsxfun(c)),i)
            Hessi_guess = np.array(nd.Jacobian(Jacifunq)(q0))
            assert np.allclose(Hessi_guess,Hguess)


    mp_dipole = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_gen = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_prop = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_list = [mp_dipole,mp_gen,mp_prop]
    fdipole = lambda c : Dipole_Disturbance([np.array([c[0],c[1],c[2]])])
    fgen = lambda c : General_Disturbance([np.array([c[0],c[1],c[2]])])
    fprop = lambda c : Prop_Disturbance([np.array([c[0],c[1],c[2]])])
    fdists = [fdipole,fgen,fprop]
    dists = [fdipole(mp_dipole),fgen(mp_gen),fprop(mp_prop)]

    for j in range(len(dists)):
        print(j)
        dj = dists[j]
        print(dj)
        dfj = fdists[j]
        vfun = lambda c: dfj(c).torque(sat,vecs)
        Jvfun = np.array(nd.Jacobian(vfun)(mp_list[j].flatten().tolist())).T
        calc_jac = dj.torque_valjac(sat,vecs)
        assert np.allclose(vfun(mp_list[j]),dj.torque(sat,vecs))
        assert np.allclose(Jvfun, calc_jac)
        for i in unitvecs:
            fun_hi = lambda c: np.dot( dfj(np.array([c[0],c[1],c[2]])).torque(sat,vecsxfun(np.array([c[3],c[4],c[5],c[6]]))),i).item()
            assert np.allclose(fun_hi(np.concatenate([mp_list[j],q0])),np.dot(dj.torque(sat,vecs),i).item())

            Hfun = np.array(nd.Hessian(fun_hi)(np.concatenate([mp_list[j],q0]).flatten().tolist()))
            Hguessq = np.dot(dj.torque_qvalhess(sat,vecs),i)
            Hguessv = np.dot(dj.torque_valvalhess(sat,vecs),i)
            Hguessqq =  np.dot(dj.torque_qqhess(sat,vecs),i)
            assert np.allclose(Hfun[0:3,0:3],Hguessv)
            assert np.allclose(Hfun[0:3,0:3],Hguessv.T)
            assert np.allclose(Hfun[0:3,3:],Hguessq.T)
            assert np.allclose(Hfun[3:,0:3],Hguessq)
            assert np.allclose(Hfun[3:,3:],Hguessqq)
            assert np.allclose(Hfun[3:,3:],Hguessqq.T)


            Jacqifunq = lambda c: np.dot(dj.torque_qjac(sat,vecsxfun(c)),i)
            Hessi_guess = np.array(nd.Jacobian(Jacqifunq)(q0))
            assert np.allclose(Hessi_guess,Hguessqq)
            assert np.allclose(Hessi_guess.T,Hguessqq)
            Jacvifunq = lambda c: np.dot(dj.torque_valjac(sat,vecsxfun(c)),i)
            Hessi_guess = np.array(nd.Jacobian(Jacvifunq)(q0))
            assert np.allclose(Hessi_guess.T,Hguessq)


            Jacqifunv = lambda c: np.dot(dfj(c).torque_qjac(sat,vecs),i)
            Hessi_guess = np.array(nd.Jacobian(Jacqifunv)(mp_list[j]))
            assert np.allclose(Hessi_guess,Hguessq)
            Jacvifunv = lambda c: np.dot(dfj(c).torque_valjac(sat,vecs),i)
            Hessi_guess = np.array(nd.Jacobian(Jacvifunv)(mp_list[j]))
            assert np.allclose(Hessi_guess,Hguessv)
            assert np.allclose(Hessi_guess.T,Hguessv)

@pytest.mark.slow
def test_dynamics_etc_with_all():

    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    B_ECI = random_n_unit_vec(3)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(69,80)#6900,7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,10)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    u = random_n_unit_vec(9)*np.random.uniform(0.3,2.1)
    os = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,rho = np.random.uniform(1e-12,1e-6),S=S_ECI)
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = Rm.T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    #setup each disturbances
    #dipole

    mp_dipole = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_gen = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_prop = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_list = np.concatenate([mp_dipole,mp_gen,mp_prop])
    dipole = Dipole_Disturbance([mp_dipole],estimate=True)
    Ndrag = 6
    drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
    while np.any([np.abs(np.pi/2-np.arccos(j))<4.0*np.pi/180.0  for j in np.dot(drag.normals,normalize(V_B))]): #numeric differentiation gets messed up near the corners
        drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
    # drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(5)])
    gen = General_Disturbance([mp_gen],estimate=True)
    gg = GG_Disturbance()
    prop = Prop_Disturbance([mp_prop],estimate=True)
    srp_eta_a = [np.random.uniform(0.1,0.9) for j in range(5)]
    srp_eta_d = [np.random.uniform(0.05,0.95-srp_eta_a[j]) for j in range(5)]
    srp_eta_s = [1-srp_eta_a[j]-srp_eta_d[j] for j in range(5)]
    srp = SRP_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),srp_eta_a[j],srp_eta_d[j],srp_eta_s[j]] for j in range(5)])
    dists = [dipole,drag,gen,gg,prop,srp]

    bias_mtq = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    bias_magic = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    bias_rw = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    h_rw = 0.5*random_n_unit_vec(3)*np.random.uniform(0.5,2.5)

    biaslist = np.concatenate([bias_mtq,bias_magic,bias_rw])

    bias_mtm = 0.3*random_n_unit_vec(3)
    bias_gyro = 0.1*random_n_unit_vec(3)
    bias_sun = 30*random_n_unit_vec(9)
    bias_gps = np.concatenate([random_n_unit_vec(3)*60,random_n_unit_vec(3)*1])
    sun_eff = 0.3

    sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun,bias_gps])
    reduced_sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun[0:3],bias_sun[6:9]])

    mtqs = [MTQ(j,0,1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    magics = [Magic(j,0,1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    rws = [RW(j,0,1,0.1,np.dot(h_rw,j),2,0,has_bias = True, bias = np.dot(bias_rw,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]

    mtms = [MTM(j,0,has_bias = True,bias = np.dot(bias_mtm,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    gyros = [Gyro(j,0,has_bias = True,bias = np.dot(bias_gyro,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    suns1 = [SunSensor(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[0:3],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    suns2 = [SunSensor(-j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[3:6],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    suns3 = [SunSensorPair(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[6:],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    gps = GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)

    sens = mtms+gyros+suns1+suns2+suns3+[gps]

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat(np.abs(random_n_unit_vec(3))*np.random.uniform(0.2,3))
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    com = random_n_unit_vec(3)*np.random.uniform(0.01,2)
    m = np.random.uniform(0.5,2)
    J_body_com = RJ@J0@RJ.T
    J_body = J_body_com - m*skewsym(com)@skewsym(com)
    J_ECI_com = Rm@J_body_com@Rm.T
    w0 = 0.05*random_n_unit_vec(3)*np.random.uniform(0.5,2.5)
    w_ECI = Rm@w0
    H_body = J_body_com@w0 + h_rw
    H_ECI = J_ECI_com@w_ECI + Rm@h_rw
    # acts = mtqs+magics+rws

    fun_act = lambda c: mtqs+magics+[RW(unitvecs[j],0,1,0.1,c[j],2,0,has_bias = True, bias = bias_rw[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    fun_sat = lambda c:  Satellite(J = J_body,mass=m,COM=com,actuators = fun_act(c),disturbances = dists,sensors = sens,estimated = True)
    acts = fun_act(h_rw)

    sat = fun_sat(h_rw)# Satellite(mass=m,COM=com,J = J_body,disturbances=dists,actuators = acts)
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@B,"r":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@R,"s":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@S,"v":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),B),"ds":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),S),"dv":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),V),"dr":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),B),"dds":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),R),"os":os}


    #test dynamics
    state = np.concatenate([w0,q0,h_rw])
    exp_torq = sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9)],np.zeros(3)) + sum([j.torque(sat,vecs) for j in dists],np.zeros(3))
    exp_wd = -np.linalg.inv(sat.J_noRW)@np.cross(w0,H_body) + np.linalg.inv(sat.J_noRW)@exp_torq
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    exp_hd = sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9) if not acts[j].has_momentum],np.zeros(3)) + sum([j.torque(sat,vecs) for j in dists],np.zeros(3)) - sat.J@exp_wd - np.cross(w0,H_body)
    xd = sat.dynamics(state,u,os)
    np.set_printoptions(precision=3)
    assert np.allclose(np.concatenate([exp_wd,exp_qd,exp_hd]),xd)

    #test jacobians


    ufun = lambda c: sat.dynamics(state,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),os)
    xfun = lambda c: sat.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),u,os)
    hfun = lambda c: fun_sat(c).dynamics(np.concatenate([w0,q0,[c[3],c[4],c[5]]]),u,os)
    bfun_mtq = lambda c: [MTQ(unitvecs[j],0,1,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_magic = lambda c: [Magic(unitvecs[i],0,1,has_bias = True, bias = c[i],use_noise=False,bias_std_rate=0,estimate_bias = True) for i in range(3)]
    bfun_rw = lambda c: [RW(unitvecs[j],0,1,0.1,h_rw[j],2,0,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_act = lambda c: bfun_mtq(c[0:3]) + bfun_magic(c[3:6]) + bfun_rw(c[6:9])
    fun_b_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = bfun_act(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])),sensors = sens,disturbances = dists,estimated=True)
    bfun = lambda cc: fun_b_sat(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])).dynamics(state,u,os)
    sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                            [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+12],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [GPS(0,has_bias = True,bias = np.array([c[15],c[16],c[17],c[18],c[19],c[20]]),use_noise = False,bias_std_rate = 0,estimate_bias = True)]
    red_sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias =bias_sun[j],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                            [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)]
    fun_s_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = sfun_sens(cc),disturbances = dists,estimated=True)
    fun_red_s_sat = lambda cc :Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = red_sfun_sens(cc),disturbances = dists,estimated=True)
    sfun = lambda c:fun_s_sat(c).dynamics(state,u,os)
    dfun_dist = lambda c: [Dipole_Disturbance([np.array([c[0],c[1],c[2]])],estimate=True),drag,General_Disturbance([np.array([c[3],c[4],c[5]])],estimate=True),gg,Prop_Disturbance([np.array([c[6],c[7],c[8]])],estimate=True),srp]
    fun_d_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = sens,disturbances = dfun_dist(cc),estimated=True)
    dfun = lambda c:fun_d_sat(c).dynamics(state,u,os)

    J_xfun = np.array(nd.Jacobian(xfun)(state.flatten().tolist())).T
    J_ufun = np.array(nd.Jacobian(ufun)(u)).T
    J_hfun = np.array(nd.Jacobian(hfun)(np.concatenate([np.ones(3),h_rw]))).T
    J_bfun = np.array(nd.Jacobian(bfun)(biaslist)).T
    J_sfun = np.array(nd.Jacobian(sfun)(sblist)).T
    J_dfun = np.array(nd.Jacobian(dfun)(mp_list)).T

    assert np.allclose(xfun(state),xd)
    assert np.allclose(ufun(u),xd)
    assert np.allclose(hfun(np.concatenate([np.ones(3),h_rw])),xd)
    assert np.allclose(hfun(np.concatenate([np.zeros(3),h_rw])),xd)
    assert np.allclose(bfun(biaslist),xd)
    assert np.allclose(sfun(sblist),xd)
    assert np.allclose(dfun(mp_list),xd)

    jacs = sat.dynamicsJacobians(state,u,os)# [dxdot__dx,dxdot__du,dxdot__dtorq,dxdot__dm]
    #
    assert np.allclose(J_xfun, jacs[0])
    assert np.allclose(J_hfun[0:3,:], np.zeros((3,10)))
    assert np.allclose(J_hfun[3:6,:], jacs[0][7:])
    assert np.allclose(J_ufun, jacs[1])
    assert np.allclose(J_bfun, jacs[2])
    assert np.allclose(J_sfun[0:6,:], jacs[3][0:6,:])
    assert np.allclose(J_sfun[6:9,:], np.zeros((3,10)))
    assert np.allclose(J_sfun[9:12,:], jacs[3][6:9,:])
    assert np.allclose(J_sfun[12:,:], np.zeros((9,10)))
    assert np.allclose(J_dfun, jacs[4])

    #test Hessians
    [[ddxdot__dxdx,ddxdot__dxdu,ddxdot__dxdab,ddxdot__dxdsb,ddxdot__dxddmp],[ddxdot__dudx,ddxdot__dudu,ddxdot__dudab,ddxdot__dudsb,ddxdot__duddmp],[_,_,ddxdot__dabdab,ddxdot__dabdsb,ddxdot__dabddmp],[_,_,_,ddxdot__dsbdsb,ddxdot__dsbddmp],[_,_,_,_,ddxdot__ddmpddmp]] = sat.dynamics_Hessians(state,u,os)

    for j in range(10):
        fun_hj = lambda c: Satellite(J = J_body,mass = m,COM = com,disturbances = dfun_dist(np.array([ c[40],c[41],c[42],c[43],c[44],c[45],c[46],c[47],c[48] ])),sensors = red_sfun_sens(np.array([c[28],c[29],c[30],c[31],c[32],c[33],c[34],c[35],c[36],c[37],c[38],c[39]])), actuators = \
            bfun_act(np.array([c[19],c[20],c[21],c[22],c[23],c[24],c[25],c[26],c[27]])),estimated=True).dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os)[j]
        # fun_bj = lambda c: sat.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os,add_torq = np.array([c[19],c[20],c[21]]),add_m = np.array([c[22],c[23],c[24]]))[j]

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([state,u,biaslist,reduced_sblist,mp_list]).flatten().tolist()))
        Hguess = np.block([[ddxdot__dxdx[:,:,j],ddxdot__dxdu[:,:,j],ddxdot__dxdab[:,:,j],ddxdot__dxdsb[:,:,j],ddxdot__dxddmp[:,:,j]],
                           [ddxdot__dxdu[:,:,j].T,ddxdot__dudu[:,:,j],ddxdot__dudab[:,:,j],ddxdot__dudsb[:,:,j],ddxdot__duddmp[:,:,j]],
                           [ddxdot__dxdab[:,:,j].T,ddxdot__dudab[:,:,j].T,ddxdot__dabdab[:,:,j],ddxdot__dabdsb[:,:,j],ddxdot__dabddmp[:,:,j]],
                           [ddxdot__dxdsb[:,:,j].T,ddxdot__dudsb[:,:,j].T,ddxdot__dabdsb[:,:,j].T,ddxdot__dsbdsb[:,:,j],ddxdot__dsbddmp[:,:,j]],
                           [ddxdot__dxddmp[:,:,j].T,ddxdot__duddmp[:,:,j].T,ddxdot__dabddmp[:,:,j].T,ddxdot__dsbddmp[:,:,j].T,ddxdot__ddmpddmp[:,:,j]]])
        # print(Hfun.shape,Hguess.shape
        np.set_printoptions(precision=3)
        assert np.allclose(Hfun,Hfun.T)
        assert np.allclose(Hguess,Hguess.T)
        assert np.allclose(Hfun,Hguess)



        ufunjju = lambda c: sat.dynamicsJacobians(state,c,os)[1][:,j]
        ufunjjx = lambda c: sat.dynamicsJacobians(state,c,os)[0][:,j]
        ufunjjb = lambda c: sat.dynamicsJacobians(state,c,os)[2][:,j]
        ufunjjs = lambda c: sat.dynamicsJacobians(state,c,os)[3][:,j]
        ufunjjd = lambda c: sat.dynamicsJacobians(state,c,os)[4][:,j]

        xfunjju = lambda c: sat.dynamicsJacobians(c,u,os)[1][:,j]
        xfunjjx = lambda c: sat.dynamicsJacobians(c,u,os)[0][:,j]
        xfunjjb = lambda c: sat.dynamicsJacobians(c,u,os)[2][:,j]
        xfunjjs = lambda c: sat.dynamicsJacobians(c,u,os)[3][:,j]
        xfunjjd = lambda c: sat.dynamicsJacobians(c,u,os)[4][:,j]

        bfunjju = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        bfunjjx = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        bfunjjb = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        bfunjjs = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        bfunjjd = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        sfunjju = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        sfunjjx = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        sfunjjb = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        sfunjjs = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        sfunjjd = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        dfunjju = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        dfunjjx = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        dfunjjb = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        dfunjjs = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        dfunjjd = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(state.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(state.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(state.flatten().tolist()))
        Jxfunjjs = np.array(nd.Jacobian(xfunjjs)(state.flatten().tolist()))
        Jxfunjjd = np.array(nd.Jacobian(xfunjjd)(state.flatten().tolist()))
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j])
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j].T)
        assert np.allclose( Jxfunjju.T , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jxfunjjb.T , ddxdot__dxdab[:,:,j])
        assert np.allclose( Jxfunjjs.T , ddxdot__dxdsb[:,:,j])
        assert np.allclose( Jxfunjjd.T , ddxdot__dxddmp[:,:,j])

        Jufunjju = np.array(nd.Jacobian(ufunjju)(u.flatten().tolist()))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(u.flatten().tolist()))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(u.flatten().tolist()))
        Jufunjjs = np.array(nd.Jacobian(ufunjjs)(u.flatten().tolist()))
        Jufunjjd = np.array(nd.Jacobian(ufunjjd)(u.flatten().tolist()))
        assert np.allclose( Jufunjjx , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jufunjju.T , ddxdot__dudu[:,:,j])
        assert np.allclose( Jufunjju.T , ddxdot__dudu[:,:,j].T)
        assert np.allclose( Jufunjjb.T , ddxdot__dudab[:,:,j])
        assert np.allclose( Jufunjjs.T , ddxdot__dudsb[:,:,j])
        assert np.allclose( Jufunjjd.T , ddxdot__duddmp[:,:,j])

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(biaslist.flatten().tolist()))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(biaslist.flatten().tolist()))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(biaslist.flatten().tolist()))
        Jbfunjjs = np.array(nd.Jacobian(bfunjjs)(biaslist.flatten().tolist()))
        Jbfunjjd = np.array(nd.Jacobian(bfunjjd)(biaslist.flatten().tolist()))
        assert np.allclose( Jbfunjjx , ddxdot__dxdab[:,:,j])
        assert np.allclose( Jbfunjju , ddxdot__dudab[:,:,j])
        assert np.allclose( Jbfunjjb , ddxdot__dabdab[:,:,j])
        assert np.allclose( Jbfunjjb.T , ddxdot__dabdab[:,:,j])
        assert np.allclose( Jbfunjjs.T , ddxdot__dabdsb[:,:,j])
        assert np.allclose( Jbfunjjd.T , ddxdot__dabddmp[:,:,j])


        Jsfunjju = np.array(nd.Jacobian(sfunjju)(reduced_sblist.flatten().tolist()))
        Jsfunjjx = np.array(nd.Jacobian(sfunjjx)(reduced_sblist.flatten().tolist()))
        Jsfunjjb = np.array(nd.Jacobian(sfunjjb)(reduced_sblist.flatten().tolist()))
        Jsfunjjs = np.array(nd.Jacobian(sfunjjs)(reduced_sblist.flatten().tolist()))
        Jsfunjjd = np.array(nd.Jacobian(sfunjjd)(reduced_sblist.flatten().tolist()))
        assert np.allclose( Jsfunjjx , ddxdot__dxdsb[:,:,j])
        assert np.allclose( Jsfunjju , ddxdot__dudsb[:,:,j])
        assert np.allclose( Jsfunjjb , ddxdot__dabdsb[:,:,j])
        assert np.allclose( Jsfunjjs , ddxdot__dsbdsb[:,:,j])
        assert np.allclose( Jsfunjjs.T , ddxdot__dsbdsb[:,:,j])
        assert np.allclose( Jsfunjjd.T , ddxdot__dsbddmp[:,:,j])

        Jdfunjju = np.array(nd.Jacobian(dfunjju)(mp_list.flatten().tolist()))
        Jdfunjjx = np.array(nd.Jacobian(dfunjjx)(mp_list.flatten().tolist()))
        Jdfunjjb = np.array(nd.Jacobian(dfunjjb)(mp_list.flatten().tolist()))
        Jdfunjjs = np.array(nd.Jacobian(dfunjjs)(mp_list.flatten().tolist()))
        Jdfunjjd = np.array(nd.Jacobian(dfunjjd)(mp_list.flatten().tolist()))
        assert np.allclose( Jdfunjjx , ddxdot__dxddmp[:,:,j])
        assert np.allclose( Jdfunjju , ddxdot__duddmp[:,:,j])
        assert np.allclose( Jdfunjjb , ddxdot__dabddmp[:,:,j])
        assert np.allclose( Jdfunjjs , ddxdot__dsbddmp[:,:,j])
        assert np.allclose( Jdfunjjd.T , ddxdot__ddmpddmp[:,:,j])
        assert np.allclose( Jdfunjjd , ddxdot__ddmpddmp[:,:,j])

@pytest.mark.slow
def test_disturbances_on_off():

    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    B_ECI = random_n_unit_vec(3)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(69,80)#6900,7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,10)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    u = random_n_unit_vec(9)*np.random.uniform(0.3,2.1)
    os = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,rho = np.random.uniform(1e-12,1e-6),S=S_ECI)
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = Rm.T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    #setup each disturbances
    mp_dipole = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_gen = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_prop = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_list = np.concatenate([mp_dipole,mp_gen,mp_prop])
    dipole = Dipole_Disturbance([mp_dipole],estimate=True)
    Ndrag = 6
    drag_faces = [[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)]
    drag = Drag_Disturbance(drag_faces)
    while np.any([np.abs(np.pi/2-np.arccos(j))<4.0*np.pi/180.0  for j in np.dot(drag.normals,normalize(V_B))]): #numeric differentiation gets messed up near the corners
        drag_faces = [[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)]
        drag = Drag_Disturbance(drag_faces)
    # drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(5)])
    gen = General_Disturbance([mp_gen],estimate=True)
    gg = GG_Disturbance()
    prop = Prop_Disturbance([mp_prop],estimate=True)
    srp_eta_a = [np.random.uniform(0.1,0.9) for j in range(5)]
    srp_eta_d = [np.random.uniform(0.05,0.95-srp_eta_a[j]) for j in range(5)]
    srp_eta_s = [1-srp_eta_a[j]-srp_eta_d[j] for j in range(5)]
    srp_faces = [[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),srp_eta_a[j],srp_eta_d[j],srp_eta_s[j]] for j in range(5)]
    srp = SRP_Disturbance(srp_faces)
    dists = [dipole,drag,gen,gg,prop,srp]


    bias_mtq = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    bias_magic = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    bias_rw = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    h_rw = 0.5*random_n_unit_vec(3)*np.random.uniform(0.5,2.5)
    biaslist = np.concatenate([bias_mtq,bias_magic,bias_rw])

    mtqs = [MTQ(j,0,1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    magics = [Magic(j,0,1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    rws = [RW(j,0,1,0.03,np.dot(h_rw,j),2,0,has_bias = True, bias = np.dot(bias_rw,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]

    bias_mtm = 0.3*random_n_unit_vec(3)
    bias_gyro = 0.1*random_n_unit_vec(3)
    bias_sun = 30*random_n_unit_vec(9)
    bias_gps = np.concatenate([random_n_unit_vec(3)*60,random_n_unit_vec(3)*1])
    sun_eff = 0.3
    sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun,bias_gps])
    reduced_sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun[0:3],bias_sun[6:9]])
    mtms = [MTM(j,0,has_bias = True,bias = np.dot(bias_mtm,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    gyros = [Gyro(j,0,has_bias = True,bias = np.dot(bias_gyro,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    suns1 = [SunSensor(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[0:3],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    suns2 = [SunSensor(-j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[3:6],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    suns3 = [SunSensorPair(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[6:],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    gps = GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)

    sens = mtms+gyros+suns1+suns2+suns3+[gps]

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat(np.abs(random_n_unit_vec(3))*np.random.uniform(0.2,3))
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    com = random_n_unit_vec(3)*np.random.uniform(0.01,2)
    m = np.random.uniform(0.5,2)
    J_body_com = RJ@J0@RJ.T
    J_body = J_body_com - m*skewsym(com)@skewsym(com)
    J_ECI_com = Rm@J_body_com@Rm.T
    w0 = 0.05*random_n_unit_vec(3)*np.random.uniform(0.5,2.5)
    w_ECI = Rm@w0
    H_body = J_body_com@w0 + h_rw
    H_ECI = J_ECI_com@w_ECI + Rm@h_rw
    acts = mtqs+magics+rws

    # fun_act = lambda c: mtqs+magics+[RW(unitvecs[j],0,1,0.1,c[j],2,0,has_bias = True, bias = bias_rw[j],use_noise=False,bias_std_rate=0) for j in range(3)]
    # fun_sat = lambda c:  Satellite(J = J_body,mass=m,COM=com,actuators = fun_act(c),disturbances = dists)
    # acts = fun_act(h_rw)
    #
    sat = Satellite(J = J_body,mass=m,COM=com,actuators = acts,disturbances = dists,sensors = sens,estimated=True)
    # sat = fun_sat(h_rw)# Satellite(mass=m,COM=com,J = J_body,disturbances=dists,actuators = acts)
    Nswitch = 10
    active_array = np.ones(6).astype(bool)
    for j in range(Nswitch):
        for i in range(6):
            turnon = bool(np.random.randint(0,2))
            if turnon:
                sat.disturbances[i].turn_on()
            else:
                sat.disturbances[i].turn_off()
            active_array[i] = turnon
    assert not np.any(np.logical_xor(active_array,[j.active for j in sat.disturbances]))

    srp_ind = np.where([isinstance(j,SRP_Disturbance) for j in sat.disturbances])[0]
    gen_ind = np.where([isinstance(j,General_Disturbance) for j in sat.disturbances])[0]
    prop_ind = np.where([isinstance(j,Prop_Disturbance) for j in sat.disturbances])[0]
    for j in range(Nswitch):
        turnon_srp = bool(np.random.randint(0,2))
        turnon_prop = bool(np.random.randint(0,2))
        turnon_gen = bool(np.random.randint(0,2))
        if turnon_srp:
            sat.srp_dist_on()
        else:
            sat.srp_dist_off()
        active_array[srp_ind] = turnon_srp
        if turnon_gen:
            sat.gen_dist_on()
        else:
            sat.gen_dist_off()
        active_array[gen_ind] = turnon_gen
        if turnon_prop:
            sat.prop_dist_on()
        else:
            sat.prop_dist_off()
        active_array[prop_ind] = turnon_prop
    assert not np.any( np.logical_xor(active_array,[j.active for j in sat.disturbances]))

    # fun_sat = lambda c:  Satellite(J = J_body,mass=m,COM=com,actuators = acts,disturbances = sat.disturbances,sensors = sens)
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@B,"r":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@R,"s":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@S,"v":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),B),"ds":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),S),"dv":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),V),"dr":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),B),"dds":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),R),"os":os}


    #test dynamics
    state = np.concatenate([w0,q0,h_rw])
    exp_torq = sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9)],np.zeros(3)) + sum([j.torque(sat,vecs) for j in dists],np.zeros(3))
    exp_wd = -np.linalg.inv(sat.J_noRW)@np.cross(w0,H_body) + np.linalg.inv(sat.J_noRW)@exp_torq
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    exp_hd = sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9) if not acts[j].has_momentum],np.zeros(3)) + sum([j.torque(sat,vecs) for j in dists],np.zeros(3)) - sat.J@exp_wd - np.cross(w0,H_body)
    xd = sat.dynamics(state,u,os)
    np.set_printoptions(precision=3)
    assert np.allclose(np.concatenate([exp_wd,exp_qd,exp_hd]),xd)

    #test jacobians
    ufun = lambda c: sat.dynamics(state,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),os)
    xfun = lambda c: sat.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),u,os)
    bfun_mtq = lambda c: [MTQ(unitvecs[j],0,1,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_magic = lambda c: [Magic(unitvecs[i],0,1,has_bias = True, bias = c[i],use_noise=False,bias_std_rate=0,estimate_bias = True) for i in range(3)]
    bfun_rw = lambda c: [RW(unitvecs[j],0,1,0.03,h_rw[j],2,0,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_act = lambda c: bfun_mtq(c[0:3]) + bfun_magic(c[3:6]) + bfun_rw(c[6:9])
    fun_b_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = bfun_act(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])),sensors = sens,disturbances = sat.disturbances,estimated=True)
    bfun = lambda cc: fun_b_sat(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])).dynamics(state,u,os)
    sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                            [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+12],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [GPS(0,has_bias = True,bias = np.array([c[12],c[13],c[14],c[15],c[16],c[17]]),use_noise = False,bias_std_rate = 0,estimate_bias = True)]
    red_sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias =bias_sun[j],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                            [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)]
    fun_s_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = sfun_sens(cc),disturbances = sat.disturbances,estimated=True)
    fun_red_s_sat = lambda cc :Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = red_sfun_sens(cc),disturbances = sat.disturbances,estimated=True)
    sfun = lambda c:fun_s_sat(c).dynamics(state,u,os)

    dfun_dist = lambda c: [Dipole_Disturbance([np.array([c[0],c[1],c[2]])],estimate=True,active=sat.disturbances[0].active), \
                            Drag_Disturbance(drag_faces,active=sat.disturbances[1].active), \
                            General_Disturbance([np.array([c[3],c[4],c[5]])],estimate=True,active=sat.disturbances[2].active), \
                            GG_Disturbance(active=sat.disturbances[3].active), \
                            Prop_Disturbance([np.array([c[6],c[7],c[8]])],estimate=True,active=sat.disturbances[4].active), \
                            SRP_Disturbance(srp_faces,active=sat.disturbances[5].active)]
    fun_d_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = sens,disturbances = dfun_dist(cc),estimated=True)
    dfun = lambda c:fun_d_sat(c).dynamics(state,u,os)

    J_xfun = np.array(nd.Jacobian(xfun)(state.flatten().tolist())).T
    J_ufun = np.array(nd.Jacobian(ufun)(u)).T
    J_bfun = np.array(nd.Jacobian(bfun)(biaslist)).T
    J_sfun = np.array(nd.Jacobian(sfun)(sblist)).T
    J_dfun = np.array(nd.Jacobian(dfun)(mp_list)).T

    assert np.allclose(xfun(state),xd)
    assert np.allclose(ufun(u),xd)
    assert np.allclose(bfun(biaslist),xd)
    assert np.allclose(sfun(sblist),xd)
    print(dfun(mp_list))
    print([j.active for j in fun_d_sat(mp_list).disturbances])
    t =  sat.dynamics(state,u,os)
    print(xd)
    print([j.active for j in sat.disturbances])
    # print(np.allclose(dfun(mp_list),xd))
    print("--------------------------")
    assert np.allclose(dfun(mp_list),xd)

    jacs = sat.dynamicsJacobians(state,u,os)# [dxdot__dx,dxdot__du,dxdot__dtorq,dxdot__dm]
    #
    assert np.allclose(J_xfun, jacs[0])
    assert np.allclose(J_ufun, jacs[1])
    assert np.allclose(J_bfun, jacs[2])
    assert np.allclose(J_sfun[0:6,:], jacs[3][0:6,:])
    assert np.allclose(J_sfun[6:9,:], np.zeros((3,10)))
    assert np.allclose(J_sfun[9:12,:], jacs[3][6:9,:])
    assert np.allclose(J_sfun[12:,:], np.zeros((9,10)))
    assert np.allclose(J_dfun, jacs[4])

    #test Hessians
    [[ddxdot__dxdx,ddxdot__dxdu,ddxdot__dxdab,ddxdot__dxdsb,ddxdot__dxddmp],[ddxdot__dudx,ddxdot__dudu,ddxdot__dudab,ddxdot__dudsb,ddxdot__duddmp],[_,_,ddxdot__dabdab,ddxdot__dabdsb,ddxdot__dabddmp],[_,_,_,ddxdot__dsbdsb,ddxdot__dsbddmp],[_,_,_,_,ddxdot__ddmpddmp]] = sat.dynamics_Hessians(state,u,os)

    for j in range(10):
        fun_hj = lambda c: Satellite(J = J_body,mass = m,COM = com,disturbances = dfun_dist(np.array([ c[40],c[41],c[42],c[43],c[44],c[45],c[46],c[47],c[48], ])),sensors = red_sfun_sens(np.array([c[28],c[29],c[30],c[31],c[32],c[33],c[34],c[35],c[36],c[37],c[38],c[39]])), actuators = \
            bfun_act(np.array([c[19],c[20],c[21],c[22],c[23],c[24],c[25],c[26],c[27]])),estimated=True).dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os)[j]
        # fun_bj = lambda c: sat.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os,add_torq = np.array([c[19],c[20],c[21]]),add_m = np.array([c[22],c[23],c[24]]))[j]

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([state,u,biaslist,reduced_sblist,mp_list]).flatten().tolist()))
        Hguess = np.block([[ddxdot__dxdx[:,:,j],ddxdot__dxdu[:,:,j],ddxdot__dxdab[:,:,j],ddxdot__dxdsb[:,:,j],ddxdot__dxddmp[:,:,j]],
                           [ddxdot__dxdu[:,:,j].T,ddxdot__dudu[:,:,j],ddxdot__dudab[:,:,j],ddxdot__dudsb[:,:,j],ddxdot__duddmp[:,:,j]],
                           [ddxdot__dxdab[:,:,j].T,ddxdot__dudab[:,:,j].T,ddxdot__dabdab[:,:,j],ddxdot__dabdsb[:,:,j],ddxdot__dabddmp[:,:,j]],
                           [ddxdot__dxdsb[:,:,j].T,ddxdot__dudsb[:,:,j].T,ddxdot__dabdsb[:,:,j].T,ddxdot__dsbdsb[:,:,j],ddxdot__dsbddmp[:,:,j]],
                           [ddxdot__dxddmp[:,:,j].T,ddxdot__duddmp[:,:,j].T,ddxdot__dabddmp[:,:,j].T,ddxdot__dsbddmp[:,:,j].T,ddxdot__ddmpddmp[:,:,j]]])
        # print(Hfun.shape,Hguess.shape
        np.set_printoptions(precision=3)
        assert np.allclose(Hfun,Hfun.T)
        assert np.allclose(Hguess,Hguess.T)
        assert np.allclose(Hfun,Hguess)


        ufunjju = lambda c: sat.dynamicsJacobians(state,c,os)[1][:,j]
        ufunjjx = lambda c: sat.dynamicsJacobians(state,c,os)[0][:,j]
        ufunjjb = lambda c: sat.dynamicsJacobians(state,c,os)[2][:,j]
        ufunjjs = lambda c: sat.dynamicsJacobians(state,c,os)[3][:,j]
        ufunjjd = lambda c: sat.dynamicsJacobians(state,c,os)[4][:,j]

        xfunjju = lambda c: sat.dynamicsJacobians(c,u,os)[1][:,j]
        xfunjjx = lambda c: sat.dynamicsJacobians(c,u,os)[0][:,j]
        xfunjjb = lambda c: sat.dynamicsJacobians(c,u,os)[2][:,j]
        xfunjjs = lambda c: sat.dynamicsJacobians(c,u,os)[3][:,j]
        xfunjjd = lambda c: sat.dynamicsJacobians(c,u,os)[4][:,j]

        bfunjju = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        bfunjjx = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        bfunjjb = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        bfunjjs = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        bfunjjd = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        sfunjju = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        sfunjjx = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        sfunjjb = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        sfunjjs = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        sfunjjd = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        dfunjju = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        dfunjjx = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        dfunjjb = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        dfunjjs = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        dfunjjd = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(state.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(state.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(state.flatten().tolist()))
        Jxfunjjs = np.array(nd.Jacobian(xfunjjs)(state.flatten().tolist()))
        Jxfunjjd = np.array(nd.Jacobian(xfunjjd)(state.flatten().tolist()))
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j])
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j].T)
        assert np.allclose( Jxfunjju.T , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jxfunjjb.T , ddxdot__dxdab[:,:,j])
        assert np.allclose( Jxfunjjs.T , ddxdot__dxdsb[:,:,j])
        assert np.allclose( Jxfunjjd.T , ddxdot__dxddmp[:,:,j])

        Jufunjju = np.array(nd.Jacobian(ufunjju)(u.flatten().tolist()))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(u.flatten().tolist()))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(u.flatten().tolist()))
        Jufunjjs = np.array(nd.Jacobian(ufunjjs)(u.flatten().tolist()))
        Jufunjjd = np.array(nd.Jacobian(ufunjjd)(u.flatten().tolist()))
        assert np.allclose( Jufunjjx , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jufunjju.T , ddxdot__dudu[:,:,j])
        assert np.allclose( Jufunjju.T , ddxdot__dudu[:,:,j].T)
        assert np.allclose( Jufunjjb.T , ddxdot__dudab[:,:,j])
        assert np.allclose( Jufunjjs.T , ddxdot__dudsb[:,:,j])
        assert np.allclose( Jufunjjd.T , ddxdot__duddmp[:,:,j])

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(biaslist.flatten().tolist()))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(biaslist.flatten().tolist()))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(biaslist.flatten().tolist()))
        Jbfunjjs = np.array(nd.Jacobian(bfunjjs)(biaslist.flatten().tolist()))
        Jbfunjjd = np.array(nd.Jacobian(bfunjjd)(biaslist.flatten().tolist()))
        assert np.allclose( Jbfunjjx , ddxdot__dxdab[:,:,j])
        assert np.allclose( Jbfunjju , ddxdot__dudab[:,:,j])
        assert np.allclose( Jbfunjjb , ddxdot__dabdab[:,:,j])
        assert np.allclose( Jbfunjjb.T , ddxdot__dabdab[:,:,j])
        assert np.allclose( Jbfunjjs.T , ddxdot__dabdsb[:,:,j])
        assert np.allclose( Jbfunjjd.T , ddxdot__dabddmp[:,:,j])


        Jsfunjju = np.array(nd.Jacobian(sfunjju)(reduced_sblist.flatten().tolist()))
        Jsfunjjx = np.array(nd.Jacobian(sfunjjx)(reduced_sblist.flatten().tolist()))
        Jsfunjjb = np.array(nd.Jacobian(sfunjjb)(reduced_sblist.flatten().tolist()))
        Jsfunjjs = np.array(nd.Jacobian(sfunjjs)(reduced_sblist.flatten().tolist()))
        Jsfunjjd = np.array(nd.Jacobian(sfunjjd)(reduced_sblist.flatten().tolist()))
        assert np.allclose( Jsfunjjx , ddxdot__dxdsb[:,:,j])
        assert np.allclose( Jsfunjju , ddxdot__dudsb[:,:,j])
        assert np.allclose( Jsfunjjb , ddxdot__dabdsb[:,:,j])
        assert np.allclose( Jsfunjjs , ddxdot__dsbdsb[:,:,j])
        assert np.allclose( Jsfunjjs.T , ddxdot__dsbdsb[:,:,j])
        assert np.allclose( Jsfunjjd.T , ddxdot__dsbddmp[:,:,j])

        Jdfunjju = np.array(nd.Jacobian(dfunjju)(mp_list.flatten().tolist()))
        Jdfunjjx = np.array(nd.Jacobian(dfunjjx)(mp_list.flatten().tolist()))
        Jdfunjjb = np.array(nd.Jacobian(dfunjjb)(mp_list.flatten().tolist()))
        Jdfunjjs = np.array(nd.Jacobian(dfunjjs)(mp_list.flatten().tolist()))
        Jdfunjjd = np.array(nd.Jacobian(dfunjjd)(mp_list.flatten().tolist()))
        assert np.allclose( Jdfunjjx , ddxdot__dxddmp[:,:,j])
        assert np.allclose( Jdfunjju , ddxdot__duddmp[:,:,j])
        assert np.allclose( Jdfunjjb , ddxdot__dabddmp[:,:,j])
        assert np.allclose( Jdfunjjs , ddxdot__dsbddmp[:,:,j])
        assert np.allclose( Jdfunjjd.T , ddxdot__ddmpddmp[:,:,j])
        assert np.allclose( Jdfunjjd , ddxdot__ddmpddmp[:,:,j])

@pytest.mark.slow
def test_dynamics_etc_controlled_noise():

    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    B_ECI = random_n_unit_vec(3)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(69,80)#6900,7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,10)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    u = random_n_unit_vec(9)*np.random.uniform(0.3,8.1)
    os = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,rho = np.random.uniform(1e-12,1e-6),S=S_ECI)
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = Rm.T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    #setup each disturbances
    #dipole

    mp_dipole = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_gen = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_prop = random_n_unit_vec(3)*np.random.uniform(0,5)
    mp_list = np.concatenate([mp_dipole,mp_gen,mp_prop])
    dipole = Dipole_Disturbance([mp_dipole],estimate=True)
    Ndrag = 6
    drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
    while np.any([np.abs(np.pi/2-np.arccos(j))<4.0*np.pi/180.0  for j in np.dot(drag.normals,normalize(V_B))]): #numeric differentiation gets messed up near the corners
        drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
    # drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(5)])
    gen = General_Disturbance([mp_gen],estimate=True)
    gg = GG_Disturbance()
    prop = Prop_Disturbance([mp_prop],estimate=True)
    srp_eta_a = [np.random.uniform(0.1,0.9) for j in range(5)]
    srp_eta_d = [np.random.uniform(0.05,0.95-srp_eta_a[j]) for j in range(5)]
    srp_eta_s = [1-srp_eta_a[j]-srp_eta_d[j] for j in range(5)]
    srp = SRP_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),srp_eta_a[j],srp_eta_d[j],srp_eta_s[j]] for j in range(5)])
    dists = [dipole,drag,gen,gg,prop,srp]

    bias_mtq = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    bias_magic = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    bias_rw = random_n_unit_vec(3)*np.random.uniform(0.01,0.5)
    h_rw = 0.5*random_n_unit_vec(3)*np.random.uniform(0.5,2.5)

    mtq_std = 0.01
    magic_std = 0.03
    rw_std = 0.02

    biaslist = np.concatenate([bias_mtq,bias_magic,bias_rw])

    bias_mtm = 0.3*random_n_unit_vec(3)
    bias_gyro = 0.1*random_n_unit_vec(3)
    bias_sun = 30*random_n_unit_vec(9)
    bias_gps = np.concatenate([random_n_unit_vec(3)*60,random_n_unit_vec(3)*1])
    sun_eff = 0.3

    sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun,bias_gps])
    reduced_sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun[0:3],bias_sun[6:9]])

    mtqs = [MTQ(j,mtq_std,1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=True,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    magics = [Magic(j,magic_std,1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=True,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    rws = [RW(j,rw_std,1,0.1,np.dot(h_rw,j),2,0,has_bias = True, bias = np.dot(bias_rw,j),use_noise=True,bias_std_rate=0,estimate_bias = True) for j in unitvecs]

    mtqs_no_noise = [MTQ(j,0,1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    magics_no_noise = [Magic(j,0,1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]
    rws_no_noise = [RW(j,0,1,0.1,np.dot(h_rw,j),2,0,has_bias = True, bias = np.dot(bias_rw,j),use_noise=False,bias_std_rate=0,estimate_bias = True) for j in unitvecs]


    mtms = [MTM(j,0,has_bias = True,bias = np.dot(bias_mtm,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    gyros = [Gyro(j,0,has_bias = True,bias = np.dot(bias_gyro,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    suns1 = [SunSensor(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[0:3],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    suns2 = [SunSensor(-j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[3:6],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    suns3 = [SunSensorPair(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[6:],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
    gps = GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)

    sens = mtms+gyros+suns1+suns2+suns3+[gps]

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat(np.abs(random_n_unit_vec(3))*np.random.uniform(0.2,3))
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    com = random_n_unit_vec(3)*np.random.uniform(0.01,2)
    m = np.random.uniform(0.5,2)
    J_body_com = RJ@J0@RJ.T
    J_body = J_body_com - m*skewsym(com)@skewsym(com)
    J_ECI_com = Rm@J_body_com@Rm.T
    w0 = 0.05*random_n_unit_vec(3)*np.random.uniform(0.5,2.5)
    w_ECI = Rm@w0
    H_body = J_body_com@w0 + h_rw
    H_ECI = J_ECI_com@w_ECI + Rm@h_rw
    acts = mtqs+magics+rws
    acts_no_noise = mtqs_no_noise+magics_no_noise+rws_no_noise

    sat =  Satellite(mass=m,COM=com,J = J_body,disturbances=dists,actuators = acts,sensors = sens,estimated=True)
    sat_no_noise =  Satellite(mass=m,COM=com,J = J_body,disturbances=dists,actuators = acts_no_noise,sensors = sens,estimated=True)
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@B,"r":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@R,"s":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@S,"v":rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),B),"ds":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),S),"dv":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),V),"dr":drotmatTvecdq(np.array([c[0],c[1],c[2],c[3]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),B),"dds":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[0],c[1],c[2],c[3]]),R),"os":os}

    sat.update_actuator_noise()
    #test dynamics
    state = np.concatenate([w0,q0,h_rw])
    exp_torq = sum([acts[j].no_noise_torque(u[j],sat,state,vecs) for j in range(9)],np.zeros(3)) + sum([j.torque(sat,vecs) for j in dists],np.zeros(3))
    exp_stor_torq = sum([acts[j].no_noise_torque(u[j],sat,state,vecs) for j in range(9) if acts[j].has_momentum],np.zeros(3))
    exp_wd = - np.linalg.inv(sat.J_noRW)@np.cross(w0,H_body) + np.linalg.inv(sat.J_noRW)@exp_torq
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    exp_hd = -exp_stor_torq - 0.1*(exp_wd)
    print(exp_torq)
    print(exp_stor_torq)

    xd = sat.dynamics(state,u,os)
    exp_xd = np.concatenate([exp_wd,exp_qd,exp_hd])
    print(xd)
    print(exp_xd)
    print(sat_no_noise.dynamics(state,u,os))
    np.set_printoptions(precision=3)

    N = 1000
    xd_err = [sat.dynamics(state,u,os,update_actuator_noise=True)-exp_xd for j in range(N)]
    command_noise = [np.concatenate([[np.random.normal(0,mtq_std) for j in range(3)],[np.random.normal(0,magic_std) for j in range(3)],[np.random.normal(0,rw_std) for j in range(3)]]) for i in range(N)]
    exp_torq_noise = [sum([sat.actuators[j].clean_torque(command_noise[i][j],sat,state,vecs) for j in range(len(sat.actuators))]) for i in range(N)]
    exp_h_torq_noise = [sum([sat.actuators[j].clean_torque(command_noise[i][j],sat,state,vecs) if not sat.actuators[j].has_momentum else np.zeros(3)for j in range(len(sat.actuators))] ,np.zeros(3)) for i in range(N)]
    # exp_h_torq_noise = [sum([exp_stor_torq_noise[j][i]*sat.actuators[j].axis for j in range(len(sat.actuators))],np.zeros(3)) for i in range(N)]
    exp_wd_noise = [ np.linalg.inv(sat.J_noRW)@j for j in exp_torq_noise]
    exp_hd_noise = [exp_h_torq_noise[j] - sat.J@exp_wd_noise[j] for j in range(N)]
    exp_dist = [np.concatenate([exp_wd_noise[j],exp_qd,exp_hd_noise[j]]) for j in range(N)]
    test_xd = sat.dynamics(state,u,os)
    assert np.allclose(test_xd,[sat.dynamics(state,u,os) for j in range(N)])

    ksvec = [kstest([j[i] for j in xd_err],[j[i] for j in exp_dist]) for i in range(len(xd))]
    for j in range(3):
        print(j,np.sqrt((1/N)*-0.5*np.log(1e-5/9/(N+1))))
        print(np.mean([i[j] for i in xd_err]),np.mean([i[j] for i in exp_dist]),xd[j])
        print(np.std([i[j] for i in xd_err]),np.std([i[j] for i in exp_dist]),0)
        print(np.mean([i[j+7] for i in xd_err]),np.mean([i[j+7] for i in exp_dist]),xd[j+7])
        print(np.std([i[j+7] for i in xd_err]),np.std([i[j+7] for i in exp_dist]),0)
        ind = j
        data_a = xd_err
        data_b = exp_dist
        hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ksvec[j].pvalue>0.1 or np.abs(ksvec[j].statistic)<(np.sqrt((1/N)*-0.5*np.log(1e-5/9/(N+1))))
        ind = j+7
        hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ksvec[7+j].pvalue>0.1 or np.abs(ksvec[7+j].statistic)<(np.sqrt((1/N)*-0.5*np.log(1e-5/9/(N+1))))
    assert np.allclose(np.mean([i[3:7] for i in exp_dist],axis = 0),exp_xd[3:7])
    assert np.allclose(np.std([i[3:7] for i in exp_dist],axis = 0),np.zeros(4))


    #test jacobians


    ufun = lambda c: sat_no_noise.dynamics(state,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),os)
    xfun = lambda c: sat_no_noise.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),u,os)
    bfun_mtq = lambda c: [MTQ(unitvecs[j],0,1,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_magic = lambda c: [Magic(unitvecs[i],0,1,has_bias = True, bias = c[i],use_noise=False,bias_std_rate=0,estimate_bias = True) for i in range(3)]
    bfun_rw = lambda c: [RW(unitvecs[j],0,1,0.1,h_rw[j],2,0,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
    bfun_act = lambda c: bfun_mtq(c[0:3]) + bfun_magic(c[3:6]) + bfun_rw(c[6:9])
    fun_b_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = bfun_act(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])),sensors = sens,disturbances = dists,estimated=True)
    bfun = lambda cc: fun_b_sat(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])).dynamics(state,u,os)
    sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                            [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+12],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [GPS(0,has_bias = True,bias = np.array([c[12],c[13],c[14],c[15],c[16],c[17]]),use_noise = False,bias_std_rate = 0,estimate_bias = True)]
    red_sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias =bias_sun[j],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                            [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                            [GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)]
    fun_s_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts_no_noise,sensors = sfun_sens(cc),disturbances = dists,estimated=True)
    fun_red_s_sat = lambda cc :Satellite(J = J_body,mass = m,COM = com,actuators = acts_no_noise,sensors = red_sfun_sens(cc),disturbances = dists,estimated=True)
    sfun = lambda c:fun_s_sat(c).dynamics(state,u,os)
    dfun_dist = lambda c: [Dipole_Disturbance([np.array([c[0],c[1],c[2]])],estimate=True),drag,General_Disturbance([np.array([c[3],c[4],c[5]])],estimate=True),gg,Prop_Disturbance([np.array([c[6],c[7],c[8]])],estimate=True),srp]
    fun_d_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts_no_noise,sensors = sens,disturbances = dfun_dist(cc),estimated=True)
    dfun = lambda c:fun_d_sat(c).dynamics(state,u,os)

    J_xfun = np.array(nd.Jacobian(xfun)(state.flatten().tolist())).T
    J_ufun = np.array(nd.Jacobian(ufun)(u)).T
    J_bfun = np.array(nd.Jacobian(bfun)(biaslist)).T
    J_sfun = np.array(nd.Jacobian(sfun)(sblist)).T
    J_dfun = np.array(nd.Jacobian(dfun)(mp_list)).T

    assert np.allclose(xfun(state),exp_xd)
    assert np.allclose(ufun(u),exp_xd)
    assert np.allclose(bfun(biaslist),exp_xd)
    assert np.allclose(sfun(sblist),exp_xd)
    assert np.allclose(dfun(mp_list),exp_xd)

    jacs = sat.dynamicsJacobians(state,u,os)# [dxdot__dx,dxdot__du,dxdot__dtorq,dxdot__dm]
    #
    assert np.allclose(J_xfun, jacs[0])
    assert np.allclose(J_ufun, jacs[1])
    assert np.allclose(J_bfun, jacs[2])
    assert np.allclose(J_sfun[0:6,:], jacs[3][0:6,:])
    assert np.allclose(J_sfun[6:9,:], np.zeros((3,10)))
    assert np.allclose(J_sfun[9:12,:], jacs[3][6:9,:])
    assert np.allclose(J_sfun[12:,:], np.zeros((9,10)))
    assert np.allclose(J_dfun, jacs[4])

    #test Hessians
    [[ddxdot__dxdx,ddxdot__dxdu,ddxdot__dxdab,ddxdot__dxdsb,ddxdot__dxddmp],[ddxdot__dudx,ddxdot__dudu,ddxdot__dudab,ddxdot__dudsb,ddxdot__duddmp],[_,_,ddxdot__dabdab,ddxdot__dabdsb,ddxdot__dabddmp],[_,_,_,ddxdot__dsbdsb,ddxdot__dsbddmp],[_,_,_,_,ddxdot__ddmpddmp]] = sat.dynamics_Hessians(state,u,os)

    for j in range(10):

        fun_hj = lambda c: Satellite(J = J_body,mass = m,COM = com,disturbances = dfun_dist(np.array([ c[40],c[41],c[42],c[43],c[44],c[45],c[46],c[47],c[48], ])),sensors = red_sfun_sens(np.array([c[28],c[29],c[30],c[31],c[32],c[33],c[34],c[35],c[36],c[37],c[38],c[39]])), actuators = \
            bfun_act(np.array([c[19],c[20],c[21],c[22],c[23],c[24],c[25],c[26],c[27]])),estimated=True).dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os)[j]
        # fun_bj = lambda c: sat.dynamics(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os,add_torq = np.array([c[19],c[20],c[21]]),add_m = np.array([c[22],c[23],c[24]]))[j]

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([state,u,biaslist,reduced_sblist,mp_list]).flatten().tolist()))
        Hguess = np.block([[ddxdot__dxdx[:,:,j],ddxdot__dxdu[:,:,j],ddxdot__dxdab[:,:,j],ddxdot__dxdsb[:,:,j],ddxdot__dxddmp[:,:,j]],
                           [ddxdot__dxdu[:,:,j].T,ddxdot__dudu[:,:,j],ddxdot__dudab[:,:,j],ddxdot__dudsb[:,:,j],ddxdot__duddmp[:,:,j]],
                           [ddxdot__dxdab[:,:,j].T,ddxdot__dudab[:,:,j].T,ddxdot__dabdab[:,:,j],ddxdot__dabdsb[:,:,j],ddxdot__dabddmp[:,:,j]],
                           [ddxdot__dxdsb[:,:,j].T,ddxdot__dudsb[:,:,j].T,ddxdot__dabdsb[:,:,j].T,ddxdot__dsbdsb[:,:,j],ddxdot__dsbddmp[:,:,j]],
                           [ddxdot__dxddmp[:,:,j].T,ddxdot__duddmp[:,:,j].T,ddxdot__dabddmp[:,:,j].T,ddxdot__dsbddmp[:,:,j].T,ddxdot__ddmpddmp[:,:,j]]])
        # print(Hfun.shape,Hguess.shape
        np.set_printoptions(precision=3)
        assert np.allclose(Hfun,Hfun.T)
        assert np.allclose(Hguess,Hguess.T)
        assert np.allclose(Hfun,Hguess)



        ufunjju = lambda c: sat.dynamicsJacobians(state,c,os)[1][:,j]
        ufunjjx = lambda c: sat.dynamicsJacobians(state,c,os)[0][:,j]
        ufunjjb = lambda c: sat.dynamicsJacobians(state,c,os)[2][:,j]
        ufunjjs = lambda c: sat.dynamicsJacobians(state,c,os)[3][:,j]
        ufunjjd = lambda c: sat.dynamicsJacobians(state,c,os)[4][:,j]

        xfunjju = lambda c: sat.dynamicsJacobians(c,u,os)[1][:,j]
        xfunjjx = lambda c: sat.dynamicsJacobians(c,u,os)[0][:,j]
        xfunjjb = lambda c: sat.dynamicsJacobians(c,u,os)[2][:,j]
        xfunjjs = lambda c: sat.dynamicsJacobians(c,u,os)[3][:,j]
        xfunjjd = lambda c: sat.dynamicsJacobians(c,u,os)[4][:,j]

        bfunjju = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        bfunjjx = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        bfunjjb = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        bfunjjs = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        bfunjjd = lambda c: fun_b_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        sfunjju = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        sfunjjx = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        sfunjjb = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        sfunjjs = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        sfunjjd = lambda c: fun_red_s_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        dfunjju = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[1][:,j]
        dfunjjx = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[0][:,j]
        dfunjjb = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[2][:,j]
        dfunjjs = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[3][:,j]
        dfunjjd = lambda c: fun_d_sat(c).dynamicsJacobians(state,u,os)[4][:,j]

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(state.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(state.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(state.flatten().tolist()))
        Jxfunjjs = np.array(nd.Jacobian(xfunjjs)(state.flatten().tolist()))
        Jxfunjjd = np.array(nd.Jacobian(xfunjjd)(state.flatten().tolist()))
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j])
        assert np.allclose( Jxfunjjx.T , ddxdot__dxdx[:,:,j].T)
        assert np.allclose( Jxfunjju.T , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jxfunjjb.T , ddxdot__dxdab[:,:,j])
        assert np.allclose( Jxfunjjs.T , ddxdot__dxdsb[:,:,j])
        assert np.allclose( Jxfunjjd.T , ddxdot__dxddmp[:,:,j])

        Jufunjju = np.array(nd.Jacobian(ufunjju)(u.flatten().tolist()))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(u.flatten().tolist()))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(u.flatten().tolist()))
        Jufunjjs = np.array(nd.Jacobian(ufunjjs)(u.flatten().tolist()))
        Jufunjjd = np.array(nd.Jacobian(ufunjjd)(u.flatten().tolist()))
        assert np.allclose( Jufunjjx , ddxdot__dxdu[:,:,j])
        assert np.allclose( Jufunjju.T , ddxdot__dudu[:,:,j])
        assert np.allclose( Jufunjju.T , ddxdot__dudu[:,:,j].T)
        assert np.allclose( Jufunjjb.T , ddxdot__dudab[:,:,j])
        assert np.allclose( Jufunjjs.T , ddxdot__dudsb[:,:,j])
        assert np.allclose( Jufunjjd.T , ddxdot__duddmp[:,:,j])

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(biaslist.flatten().tolist()))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(biaslist.flatten().tolist()))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(biaslist.flatten().tolist()))
        Jbfunjjs = np.array(nd.Jacobian(bfunjjs)(biaslist.flatten().tolist()))
        Jbfunjjd = np.array(nd.Jacobian(bfunjjd)(biaslist.flatten().tolist()))
        assert np.allclose( Jbfunjjx , ddxdot__dxdab[:,:,j])
        assert np.allclose( Jbfunjju , ddxdot__dudab[:,:,j])
        assert np.allclose( Jbfunjjb , ddxdot__dabdab[:,:,j])
        assert np.allclose( Jbfunjjb.T , ddxdot__dabdab[:,:,j])
        assert np.allclose( Jbfunjjs.T , ddxdot__dabdsb[:,:,j])
        assert np.allclose( Jbfunjjd.T , ddxdot__dabddmp[:,:,j])


        Jsfunjju = np.array(nd.Jacobian(sfunjju)(reduced_sblist.flatten().tolist()))
        Jsfunjjx = np.array(nd.Jacobian(sfunjjx)(reduced_sblist.flatten().tolist()))
        Jsfunjjb = np.array(nd.Jacobian(sfunjjb)(reduced_sblist.flatten().tolist()))
        Jsfunjjs = np.array(nd.Jacobian(sfunjjs)(reduced_sblist.flatten().tolist()))
        Jsfunjjd = np.array(nd.Jacobian(sfunjjd)(reduced_sblist.flatten().tolist()))
        assert np.allclose( Jsfunjjx , ddxdot__dxdsb[:,:,j])
        assert np.allclose( Jsfunjju , ddxdot__dudsb[:,:,j])
        assert np.allclose( Jsfunjjb , ddxdot__dabdsb[:,:,j])
        assert np.allclose( Jsfunjjs , ddxdot__dsbdsb[:,:,j])
        assert np.allclose( Jsfunjjs.T , ddxdot__dsbdsb[:,:,j])
        assert np.allclose( Jsfunjjd.T , ddxdot__dsbddmp[:,:,j])

        Jdfunjju = np.array(nd.Jacobian(dfunjju)(mp_list.flatten().tolist()))
        Jdfunjjx = np.array(nd.Jacobian(dfunjjx)(mp_list.flatten().tolist()))
        Jdfunjjb = np.array(nd.Jacobian(dfunjjb)(mp_list.flatten().tolist()))
        Jdfunjjs = np.array(nd.Jacobian(dfunjjs)(mp_list.flatten().tolist()))
        Jdfunjjd = np.array(nd.Jacobian(dfunjjd)(mp_list.flatten().tolist()))
        assert np.allclose( Jdfunjjx , ddxdot__dxddmp[:,:,j])
        assert np.allclose( Jdfunjju , ddxdot__duddmp[:,:,j])
        assert np.allclose( Jdfunjjb , ddxdot__dabddmp[:,:,j])
        assert np.allclose( Jdfunjjs , ddxdot__dsbddmp[:,:,j])
        assert np.allclose( Jdfunjjd.T , ddxdot__ddmpddmp[:,:,j])
        assert np.allclose( Jdfunjjd , ddxdot__ddmpddmp[:,:,j])

def test_rk4():
    sat = Satellite()
    state = np.concatenate([0.01*unitvecs[0],zeroquat])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    xkp1 = sat.rk4(state,np.array([]),1,os,os)
    print(xkp1)
    print(np.array([0.01,0,0,0.9999875,0.00499998,0,0]))
    assert np.allclose(np.array([0.01,0,0,0.9999875,0.00499998,0,0]), xkp1,rtol = 1e-8,atol=1e-8)
    sat = Satellite()
    state = np.concatenate([0.01*unitvecs[0],zeroquat])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    xkp1 = sat.rk4(state,np.array([]),1,os,os,quat_as_vec = False)
    print(xkp1)
    print(np.array([0.01,0,0,0.9999875,0.00499998,0,0]))
    assert np.allclose(np.array([0.01,0,0,0.9999875,0.00499998,0,0]), xkp1,rtol = 1e-8,atol=1e-8)
    # print('1')
    # sat = Satellite()
    # state = np.concatenate([0.1*unitvecs[0],zeroquat])
    # os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    # xkp1 = sat.rk4(state,np.array([]),1,os,os)
    # print(xkp1)
    # print(np.array([0.1,0,0,np.cos(0.1/2),0.0499998,0,0]))
    # print(np.array([0.1,0,0,np.cos(0.1/2),0.0499998,0,0])-xkp1)
    # assert np.allclose(np.array([0.1,0,0,np.cos(0.1/2),0.0499998,0,0]), xkp1,rtol = 1e-8,atol=1e-8)
    print('2')
    sat = Satellite()
    state = np.concatenate([0.1*unitvecs[0],zeroquat])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    xkp1 = sat.rk4(state,np.array([]),1,os,os,quat_as_vec = False)
    print(xkp1)
    print(np.array([0.1,0,0,np.cos(0.1/2),np.sin(0.1/2),0,0]))
    print(np.array([0.1,0,0,np.cos(0.1/2),np.sin(0.1/2),0,0])-xkp1)
    print(np.arcsin(xkp1[4])*2-0.1)
    assert np.allclose(np.array([0.1,0,0,np.cos(0.1/2),np.sin(0.1/2),0,0]), xkp1,rtol = 1e-8,atol=1e-8)
    print('2')
    sat = Satellite()
    state = np.concatenate([1.0*unitvecs[0],zeroquat])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    xkp1 = sat.rk4(state,np.array([]),1,os,os,quat_as_vec = False)
    print(xkp1)
    print(np.array([1,0,0,np.cos(1/2),np.sin(1/2),0,0]))
    print(np.array([1,0,0,np.cos(1/2),np.sin(1/2),0,0])-xkp1)
    print(np.arcsin(xkp1[4])*2-1)
    assert np.allclose(np.array([1,0,0,np.cos(1/2),np.sin(1/2),0,0]), xkp1,rtol = 1e-8,atol=1e-8)

def test_rk4_angular_momentum_noRW():
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,math.sqrt(mu_e/7000),0]))

    xv = 0.01
    zv = 0.001

    state0 = np.array([xv,0,zv,1,0,0,0])
    time = 0.01*(math.pi*2*math.sqrt(7000**3/mu_e))

    sat = Satellite(J = 0.01*np.diagflat([1,2,4]))
    dt = 1.0
    t = 0
    xk = state0
    H0 = rot_mat(state0[3:7])@sat.J@state0[0:3]
    E0 = 0.5*state0[0:3].T@sat.J@state0[0:3]
    iter = 0
    while t<time and iter<1e4:
        xkp1 = sat.rk4(xk,np.array([]),dt,os,os)
        Hkp1 = rot_mat(xkp1[3:7])@sat.J@xkp1[0:3]
        Ekp1 = 0.5*xkp1[0:3].T@sat.J@xkp1[0:3]
        assert np.allclose(Hkp1,H0)
        assert np.allclose(Ekp1,E0)
        xk = xkp1
        t += dt
        iter += 1
    assert t>time

    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,math.sqrt(mu_e/7000),0]))
    w0 = random_n_unit_vec(3)*np.random.uniform(0.05,0.2)
    q0 = random_n_unit_vec(4)
    state0 = np.concatenate([w0,q0])
    time = 0.01*(math.pi*2*math.sqrt(7000**3/mu_e))

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat([2,3,5])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    Rm = rot_mat(q0)
    com = random_n_unit_vec(3)*np.random.uniform(0.01,2)
    m = np.random.uniform(0.5,2)
    J_body_com = RJ@J0@RJ.T
    J_body = J_body_com - m*skewsym(com)@skewsym(com)
    J_ECI_com = Rm@J_body_com@Rm.T
    w_ECI = Rm@w0
    H_ECI = J_ECI_com@w_ECI

    sat = Satellite(J = J_body, COM=com,mass=m)
    dt = 0.9
    t = 0
    xk = state0
    E0 = 0.5*state0[0:3].T@sat.J@state0[0:3]
    iter = 0
    while t<time and iter<1e4:
        xkp1 = sat.rk4(xk,np.array([]),dt,os,os)
        Hkp1 = rot_mat(xkp1[3:7])@sat.J@xkp1[0:3]
        Ekp1 = 0.5*xkp1[0:3].T@sat.J@xkp1[0:3]
        assert np.allclose(Hkp1,H_ECI,atol=1e-6,rtol=1e-4)
        assert np.allclose(Ekp1,E0)
        xk = xkp1
        t += dt
        iter += 1
    assert t>time

    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,math.sqrt(mu_e/7000),0]))
    w0 = 0.01*unitvecs[0]
    q0 = zeroquat
    Rm = rot_mat(q0)
    state0 = np.concatenate([w0,q0])
    time = 0.01*(math.pi*2*math.sqrt(7000**3/mu_e))

    J0 = np.diagflat([2,3,5])
    qJ = zeroquat
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    com = unitvecs[0]*np.random.uniform(-2,2)
    m = np.random.uniform(0.5,2)
    J_body_com = RJ@J0@RJ.T
    J_body = J_body_com - m*skewsym(com)@skewsym(com)
    J_ECI_com = Rm@J_body_com@Rm.T
    w_ECI = Rm@w0
    H_ECI = J_ECI_com@w_ECI
    acts = [Magic(j,0,3.0,False,0,0,False) for j in unitvecs]

    sat = Satellite(J = J_body, COM=com,mass=m,actuators = acts)
    dt = 0.9
    t = 0
    xk = state0
    iter = 0
    E0 = 0.5*state0[0:3].T@sat.J@state0[0:3]
    umag = 0.01
    Ek = E0
    while t<time and iter<1e4:
        xkp1 = sat.rk4(xk,np.array(umag*unitvecs[0]),dt,os,os)
        Hkp1 = rot_mat(xkp1[3:7])@sat.J@xkp1[0:3]
        Ekp1 = 0.5*xkp1[0:3].T@sat.J@xkp1[0:3]
        assert np.allclose(Hkp1,H_ECI+(t+dt)*unitvecs[0]*umag)
        assert np.allclose(Ekp1,Ek+umag*dt*(0.5*(xkp1[0]-xk[0]) + xk[0]))
        xk = xkp1
        Ek = Ekp1
        t += dt
        iter += 1
    assert t>time


    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,math.sqrt(mu_e/7000),0]))

    xv = 0.01
    zv = 0.001

    state0 = np.array([xv,0,zv,1,0,0,0])
    time = 0.01*(math.pi*2*math.sqrt(7000**3/mu_e))

    sat = Satellite(J = 0.01*np.diagflat([1,2,4]))
    dt = 1.0
    t = 0
    xk = state0
    H0 = rot_mat(state0[3:7])@sat.J@state0[0:3]
    E0 = 0.5*state0[0:3].T@sat.J@state0[0:3]
    iter = 0
    while t<time and iter<1e4:
        xkp1 = sat.rk4(xk,np.array([]),dt,os,os,quat_as_vec = False)
        Hkp1 = rot_mat(xkp1[3:7])@sat.J@xkp1[0:3]
        Ekp1 = 0.5*xkp1[0:3].T@sat.J@xkp1[0:3]
        print(Hkp1)
        print(H0)
        print(Ekp1)
        print(E0)
        assert np.allclose(Hkp1,H0)
        assert np.allclose(Ekp1,E0)
        xk = xkp1
        t += dt
        iter += 1
    assert t>time

    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,math.sqrt(mu_e/7000),0]))
    w0 = random_n_unit_vec(3)*np.random.uniform(0.05,0.2)
    q0 = random_n_unit_vec(4)
    state0 = np.concatenate([w0,q0])
    time = 0.01*(math.pi*2*math.sqrt(7000**3/mu_e))

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat([2,3,5])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    Rm = rot_mat(q0)
    com = random_n_unit_vec(3)*np.random.uniform(0.01,2)
    m = np.random.uniform(0.5,2)
    J_body_com = RJ@J0@RJ.T
    J_body = J_body_com - m*skewsym(com)@skewsym(com)
    J_ECI_com = Rm@J_body_com@Rm.T
    w_ECI = Rm@w0
    H_ECI = J_ECI_com@w_ECI

    sat = Satellite(J = J_body, COM=com,mass=m)
    dt = 0.9
    t = 0
    xk = state0
    E0 = 0.5*state0[0:3].T@sat.J@state0[0:3]
    iter = 0
    while t<time and iter<1e4:
        xkp1 = sat.rk4(xk,np.array([]),dt,os,os,quat_as_vec= False)
        Hkp1 = rot_mat(xkp1[3:7])@sat.J@xkp1[0:3]
        Ekp1 = 0.5*xkp1[0:3].T@sat.J@xkp1[0:3]
        assert np.allclose(Hkp1,H_ECI,atol=1e-6,rtol=1e-4)
        assert np.allclose(Ekp1,E0)
        xk = xkp1
        t += dt
        iter += 1
    assert t>time

    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,math.sqrt(mu_e/7000),0]))
    w0 = 0.01*unitvecs[0]
    q0 = zeroquat
    Rm = rot_mat(q0)
    state0 = np.concatenate([w0,q0])
    time = 0.01*(math.pi*2*math.sqrt(7000**3/mu_e))

    J0 = np.diagflat([2,3,5])
    qJ = zeroquat
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    com = unitvecs[0]*np.random.uniform(-2,2)
    m = np.random.uniform(0.5,2)
    J_body_com = RJ@J0@RJ.T
    J_body = J_body_com - m*skewsym(com)@skewsym(com)
    J_ECI_com = Rm@J_body_com@Rm.T
    w_ECI = Rm@w0
    H_ECI = J_ECI_com@w_ECI
    acts = [Magic(j,0,3.0,False,0,0,False) for j in unitvecs]

    sat = Satellite(J = J_body, COM=com,mass=m,actuators = acts)
    dt = 0.9
    t = 0
    xk = state0
    iter = 0
    E0 = 0.5*state0[0:3].T@sat.J@state0[0:3]
    umag = 0.01
    Ek = E0
    while t<time and iter<1e4:
        xkp1 = sat.rk4(xk,np.array(umag*unitvecs[0]),dt,os,os,quat_as_vec= False)
        Hkp1 = rot_mat(xkp1[3:7])@sat.J@xkp1[0:3]
        Ekp1 = 0.5*xkp1[0:3].T@sat.J@xkp1[0:3]
        assert np.allclose(Hkp1,H_ECI+(t+dt)*unitvecs[0]*umag)
        assert np.allclose(Ekp1,Ek+umag*dt*(0.5*(xkp1[0]-xk[0]) + xk[0]))
        xk = xkp1
        Ek = Ekp1
        t += dt
        iter += 1
    assert t>time


@pytest.mark.slow
def test_rk4_etc_with_all():
    dt = 1.2#5.0
    full_test = False

    for qav in [True,False]:

        q0 = random_n_unit_vec(4)
        Rm = rot_mat(q0)
        R_ECI = random_n_unit_vec(3)*np.random.uniform(6900,7800)
        V_ECI = random_n_unit_vec(3)*np.random.uniform(6,10)
        u = random_n_unit_vec(9)*np.random.uniform(0.3,2.1)
        os = Orbital_State(0.22,R_ECI,V_ECI)
        osp1 = os.orbit_rk4(dt, J2_on=True, rk4_on=True)
        osp1.J2000 = 0.22+dt*sec2cent
        V_B = Rm.T@V_ECI
        #setup each disturbances
        #dipole



        mp_dipole = random_n_unit_vec(3)*np.random.uniform(0,5)
        mp_gen = random_n_unit_vec(3)*np.random.uniform(0,5)
        mp_gen2 = random_n_unit_vec(3)*np.random.uniform(0,5)
        mp_prop = random_n_unit_vec(3)*np.random.uniform(0,5)
        mp_list = np.concatenate([mp_dipole,mp_gen,mp_prop,mp_gen2])
        dipole = Dipole_Disturbance([mp_dipole],estimate=True)
        Ndrag = 6
        drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
        while np.any([np.abs(np.pi/2-np.arccos(j))<4.0*np.pi/180.0  for j in np.dot(drag.normals,normalize(V_B))]): #numeric differentiation gets messed up near the corners
            drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(Ndrag)])
        # drag = Drag_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3),np.random.uniform(1.5,2.5)] for j in range(5)])
        gen = General_Disturbance([mp_gen],estimate=True)
        gen2 = General_Disturbance([mp_gen2],estimate=True)
        gg = GG_Disturbance()
        prop = Prop_Disturbance([mp_prop],estimate=True)
        srp_eta_a = [np.random.uniform(0.1,0.9) for j in range(5)]
        srp_eta_d = [np.random.uniform(0.05,0.95-srp_eta_a[j]) for j in range(5)]
        srp_eta_s = [1-srp_eta_a[j]-srp_eta_d[j] for j in range(5)]
        srp = SRP_Disturbance([[j,np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),random_n_unit_vec(3)*np.random.uniform(0.1,2),srp_eta_a[j],srp_eta_d[j],srp_eta_s[j]] for j in range(5)])
        dists = [dipole,drag,gen,gg,prop,srp,gen2]

        bias_mtq = random_n_unit_vec(3)*np.random.uniform(0.01,1.0)
        bias_magic = random_n_unit_vec(3)*np.random.uniform(0.01,1.0)
        bias_rw = random_n_unit_vec(3)*np.random.uniform(0.01,1.0)
        h_rw = 0.1*random_n_unit_vec(3)*np.random.uniform(0.5,2.5)

        mtq_std = 0.01
        magic_std = 0.03
        rw_std = 0.02

        biaslist = np.concatenate([bias_mtq,bias_magic,bias_rw])

        bias_mtm = 0.3*random_n_unit_vec(3)
        bias_gyro = 0.1*random_n_unit_vec(3)
        bias_sun = 30*random_n_unit_vec(9)
        bias_gps = np.concatenate([random_n_unit_vec(3)*60,random_n_unit_vec(3)*1])
        sun_eff = 0.3

        sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun,bias_gps])
        reduced_sblist = np.concatenate([bias_mtm,bias_gyro,bias_sun[0:3],bias_sun[6:9]])



        mtqs = [MTQ(j,0,1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=False,bias_std_rate=0,estimate_bias =True) for j in unitvecs]
        magics = [Magic(j,0,1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=False,bias_std_rate=0,estimate_bias =True) for j in unitvecs]
        rws = [RW(j,0,1,0.1,np.dot(h_rw,j),2,0,has_bias = True, bias = np.dot(bias_rw,j),use_noise=False,bias_std_rate=0,estimate_bias =True) for j in unitvecs]


        mtms = [MTM(j,0,has_bias = True,bias = np.dot(bias_mtm,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
        gyros = [Gyro(j,0,has_bias = True,bias = np.dot(bias_gyro,j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
        suns1 = [SunSensor(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[0:3],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
        suns2 = [SunSensor(-j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[3:6],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
        suns3 = [SunSensorPair(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[6:],j),use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in unitvecs]
        gps = GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)

        sens = mtms+gyros+suns1+suns2+suns3+[gps]

        qJ = random_n_unit_vec(4)
        J0 = np.diagflat([np.abs(np.random.uniform(2,10)),np.abs(np.random.uniform(3,9)),np.abs(np.random.uniform(2,14))])
        RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
        com = random_n_unit_vec(3)*np.random.uniform(0.01,2)
        m = np.random.uniform(0.5,2)
        J_body_com = RJ@J0@RJ.T
        J_body = J_body_com - m*skewsym(com)@skewsym(com)
        J_ECI_com = Rm@J_body_com@Rm.T
        w0 = (2.0*np.pi/180.0)*random_n_unit_vec(3)*np.random.uniform(0.5,2.0)
        w_ECI = Rm@w0
        H_body = J_body_com@w0 + h_rw
        H_ECI = J_ECI_com@w_ECI + Rm@h_rw
        acts = mtqs+magics+rws
        np.set_printoptions(precision=4)
        sat =  Satellite(mass=m,COM=com,J = J_body,disturbances=dists,actuators = acts,sensors = sens,estimated=True)


        # fun_act = lambda c: mtqs+magics+[RW(unitvecs[j],0,1,0.1,c[j],2,0,has_bias = True, bias = bias_rw[j],use_noise=False,bias_std_rate=0) for j in range(3)]
        # fun_sat = lambda c:  Satellite(J = J_body,mass=m,COM=com,actuators = fun_act(c),disturbances = dists,estimated=True)
        # acts = fun_act(h_rw)
        #
        # sat = fun_sat(h_rw)
        #test dynamics
        state = np.concatenate([w0,q0,h_rw])
        xp1 = sat.rk4(state,u,dt,os,osp1,quat_as_vec = qav)

        if full_test:

            ivp_jac = lambda tt,xx : sat.dynamics_jac_for_solver(tt,xx, u, os,osp1)
            # print('x0',np.concatenate([state,sat.RWhs()]))
            out = solve_ivp(sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(u, os,osp1),rtol = 1e-5)#,jac = ivp_jac)
            xp1_test = out.y[:,-1].flatten()
            xp1_test[3:7] = normalize(xp1_test[3:7])
            print('x0',np.concatenate([state,sat.RWhs()]))
            print('rk4',xp1)
            print('test',xp1_test)
            print('diff',xp1-xp1_test)
            print('reldiff',(xp1-xp1_test)/xp1_test)
            # print('dyn',sat.dynamics(state,u,os))
            assert np.allclose(xp1,xp1_test,atol = 0,rtol = 5e-2)

            ivp_jac = lambda tt,xx : sat.dynamics_jac_for_solver(tt,xx, u, os,osp1)
            # print('x0',np.concatenate([state,sat.RWhs()]))
            out = solve_ivp(sat.dynamics_for_solver, (0, 0.1*dt), state, method="RK45", args=(u, os,osp1),rtol = 1e-5)#,jac = ivp_jac)
            xp01_test = out.y[:,-1].flatten()
            xp01_test[3:7] = normalize(xp01_test[3:7])
            xp01 = sat.rk4(state,u,0.1*dt,os,osp1)

            print('x0',np.concatenate([state,sat.RWhs()]))
            print('rk4',xp1)
            print('test',xp1_test)
            print('diff',xp1-xp1_test)
            print('reldiff',(xp1-xp1_test)/xp1_test)
            # print('dyn',sat.dynamics(state,u,os))
            assert np.allclose(xp01,xp01_test,atol = 0,rtol = 1e-3)

        #test jacobians

        ufun = lambda c: sat.rk4(state,np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8]]),dt,os,osp1)
        xfun = lambda c: sat.rk4(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),u,dt,os,osp1)
        bfun_mtq = lambda c: [MTQ(unitvecs[j],0,1,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
        bfun_magic = lambda c: [Magic(unitvecs[i],0,1,has_bias = True, bias = c[i],use_noise=False,bias_std_rate=0,estimate_bias = True) for i in range(3)]
        bfun_rw = lambda c: [RW(unitvecs[j],0,1,0.1,h_rw[j],2,0,has_bias = True, bias = c[j],use_noise=False,bias_std_rate=0,estimate_bias = True) for j in range(3)]
        bfun_act = lambda c: bfun_mtq(c[0:3]) + bfun_magic(c[3:6]) + bfun_rw(c[6:9])
        fun_b_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = bfun_act(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])),sensors = sens,disturbances = dists,estimated=True)
        bfun = lambda cc: fun_b_sat(np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]])).rk4(state,u,dt,os,osp1)
        sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                                [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+12],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [GPS(0,has_bias = True,bias = np.array([c[12],c[13],c[14],c[15],c[16],c[17]]),use_noise = False,bias_std_rate = 0,estimate_bias = True)]
        red_sfun_sens = lambda c : [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias =bias_sun[j],use_noise = False,bias_std_rate = 0) for j in range(3)] +\
                                [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [SunSensorPair(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0,estimate_bias = True) for j in range(3)] +\
                                [GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0,estimate_bias = True)]
        fun_s_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = sfun_sens(cc),disturbances = dists,estimated=True)
        fun_red_s_sat = lambda cc :Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = red_sfun_sens(cc),disturbances = dists,estimated=True)
        sfun = lambda c:fun_s_sat(c).rk4(state,u,dt,os,osp1)
        dfun_dist = lambda c: [Dipole_Disturbance([np.array([c[0],c[1],c[2]])],estimate=True),drag,General_Disturbance([np.array([c[3],c[4],c[5]])],estimate=True),gg,Prop_Disturbance([np.array([c[6],c[7],c[8]])],estimate=True),srp,General_Disturbance([np.array([c[9],c[10],c[11]])],estimate=True)]
        fun_d_sat = lambda cc: Satellite(J = J_body,mass = m,COM = com,actuators = acts,sensors = sens,disturbances = dfun_dist(cc),estimated=True)
        dfun = lambda c:fun_d_sat(c).rk4(state,u,dt,os,osp1)

        J_xfun = np.array(nd.Jacobian(xfun)(state.flatten().tolist())).T
        J_ufun = np.array(nd.Jacobian(ufun)(u)).T
        J_bfun = np.array(nd.Jacobian(bfun)(biaslist)).T
        J_sfun = np.array(nd.Jacobian(sfun)(sblist)).T
        J_dfun = np.array(nd.Jacobian(dfun)(mp_list)).T

        assert np.allclose(xfun(state),xp1)
        assert np.allclose(ufun(u),xp1)
        assert np.allclose(bfun(biaslist),xp1)
        assert np.allclose(sfun(sblist),xp1)
        assert np.allclose(dfun(mp_list),xp1)

        jacs = sat.rk4Jacobians(state,u,dt,os,osp1)# [dxdot__dx,dxdot__du,dxdot__dtorq,dxdot__dm]
        #
        assert np.allclose(J_xfun, jacs[0])
        assert np.allclose(J_ufun, jacs[1])
        assert np.allclose(J_bfun, jacs[2])
        assert np.allclose(J_sfun[0:6,:], jacs[3][0:6,:])
        assert np.allclose(J_sfun[6:9,:], np.zeros((3,10)))
        assert np.allclose(J_sfun[9:12,:], jacs[3][6:9,:])
        assert np.allclose(J_sfun[12:,:], np.zeros((9,10)))
        assert np.allclose(J_dfun, jacs[4])

        #test Hessians
        [ddx__dxdu,ddx__dudu,ddx__dudab,ddx__dudsb,ddx__duddmp] = sat.rk4_u_Hessians(state,u,dt,os,osp1)

        for j in range(10):
            print(j)
            if full_test:
                fun_hj = lambda c: Satellite(J = J_body,mass = m,COM = com,disturbances = dfun_dist(np.array([ c[40],c[41],c[42],c[43],c[44],c[45],c[46],c[47],c[48],c[49],c[50],c[51], ])),sensors = red_sfun_sens(np.array([c[28],c[29],c[30],c[31],c[32],c[33],c[34],c[35],c[36],c[37],c[38],c[39]])), actuators = \
                    bfun_act(np.array([c[19],c[20],c[21],c[22],c[23],c[24],c[25],c[26],c[27]])),estimated=True).rk4(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),dt,os,osp1)[j]
                # fun_bj = lambda c: sat.rk4(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),np.array([c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18]]),os,add_torq = np.array([c[19],c[20],c[21]]),add_m = np.array([c[22],c[23],c[24]]))[j]

                Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([state,u,biaslist,reduced_sblist,mp_list]).flatten().tolist()))
                # Hguess = np.block([[ddxdot__dxdx[:,:,j],ddxdot__dxdu[:,:,j],ddxdot__dxdab[:,:,j],ddxdot__dxdsb[:,:,j],ddxdot__dxddmp[:,:,j]],
                #                    [ddxdot__dxdu[:,:,j].T,ddxdot__dudu[:,:,j],ddxdot__dudab[:,:,j],ddxdot__dudsb[:,:,j],ddxdot__duddmp[:,:,j]],
                #                    [ddxdot__dxdab[:,:,j].T,ddxdot__dudab[:,:,j].T,ddxdot__dabdab[:,:,j],ddxdot__dabdsb[:,:,j],ddxdot__dabddmp[:,:,j]],
                #                    [ddxdot__dxdsb[:,:,j].T,ddxdot__dudsb[:,:,j].T,ddxdot__dabdsb[:,:,j].T,ddxdot__dsbdsb[:,:,j],ddxdot__dsbddmp[:,:,j]],
                #                    [ddxdot__dxddmp[:,:,j].T,ddxdot__duddmp[:,:,j].T,ddxdot__dabddmp[:,:,j].T,ddxdot__dsbddmp[:,:,j].T,ddxdot__ddmpddmp[:,:,j]]])
                # print(Hfun.shape,Hguess.shape
                np.set_printoptions(precision=3)
                assert np.allclose(ddx__dxdu[:,:,j],Hfun[0:10,10:19])
                assert np.allclose(ddx__dxdu[:,:,j],Hfun[10:19,0:10].T)
                assert np.allclose(ddx__dudu[:,:,j],Hfun[10:19,10:19])
                assert np.allclose(ddx__dudu[:,:,j],Hfun[10:19,10:19].T)
                assert np.allclose(ddx__dudab[:,:,j],Hfun[10:19,19:28].T)
                assert np.allclose(ddx__dudab[:,:,j],Hfun[19:28,10:19])
                assert np.allclose(ddx__dudsb[:,:,j],Hfun[10:19,28:37].T)
                assert np.allclose(ddx__dudsb[:,:,j],Hfun[28:37,10:19])
                assert np.allclose(ddx__duddmp[:,:,j],Hfun[10:19,37:49].T)
                assert np.allclose(ddx__duddmp[:,:,j],Hfun[37:49,10:19])



            ufunjju = lambda c: sat.rk4Jacobians(state,c,dt,os,osp1)[1][:,j]
            ufunjjx = lambda c: sat.rk4Jacobians(state,c,dt,os,osp1)[0][:,j]
            ufunjjb = lambda c: sat.rk4Jacobians(state,c,dt,os,osp1)[2][:,j]
            ufunjjs = lambda c: sat.rk4Jacobians(state,c,dt,os,osp1)[3][:,j]
            ufunjjd = lambda c: sat.rk4Jacobians(state,c,dt,os,osp1)[4][:,j]

            xfunjju = lambda c: sat.rk4Jacobians(c,u,dt,os,osp1)[1][:,j]

            bfunjju = lambda c: fun_b_sat(c).rk4Jacobians(state,u,dt,os,osp1)[1][:,j]

            sfunjju = lambda c: fun_red_s_sat(c).rk4Jacobians(state,u,dt,os,osp1)[1][:,j]

            dfunjju = lambda c: fun_d_sat(c).rk4Jacobians(state,u,dt,os,osp1)[1][:,j]

            Jxfunjju = np.array(nd.Jacobian(xfunjju)(state.flatten().tolist()))
            assert np.allclose( Jxfunjju.T , ddx__dxdu[:,:,j])

            Jufunjju = np.array(nd.Jacobian(ufunjju)(u.flatten().tolist()))
            Jufunjjx = np.array(nd.Jacobian(ufunjjx)(u.flatten().tolist()))
            Jufunjjb = np.array(nd.Jacobian(ufunjjb)(u.flatten().tolist()))
            Jufunjjs = np.array(nd.Jacobian(ufunjjs)(u.flatten().tolist()))
            Jufunjjd = np.array(nd.Jacobian(ufunjjd)(u.flatten().tolist()))
            assert np.allclose( Jufunjjx , ddx__dxdu[:,:,j])
            assert np.allclose( Jufunjju.T , ddx__dudu[:,:,j])
            assert np.allclose( Jufunjju , ddx__dudu[:,:,j])
            assert np.allclose( Jufunjjb , ddx__dudab[:,:,j])
            assert np.allclose( Jufunjjs , ddx__dudsb[:,:,j])
            assert np.allclose( Jufunjjd , ddx__duddmp[:,:,j])

            Jbfunjju = np.array(nd.Jacobian(bfunjju)(biaslist.flatten().tolist()))
            assert np.allclose( Jbfunjju , ddx__dudab[:,:,j])


            Jsfunjju = np.array(nd.Jacobian(sfunjju)(reduced_sblist.flatten().tolist()))
            assert np.allclose( Jsfunjju.T , ddx__dudsb[:,:,j])

            Jdfunjju = np.array(nd.Jacobian(dfunjju)(mp_list.flatten().tolist()))
            print( Jdfunjju )
            print( ddx__duddmp[:,:,j])
            print( np.isclose( Jdfunjju.T , ddx__duddmp[:,:,j]).astype(int))
            assert np.allclose( Jdfunjju.T , ddx__duddmp[:,:,j])

def test_mtm_reading_etc_clean():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    sc = 1e3
    mtm = MTM(ax,std,has_bias = False,use_noise = False,scale = sc)
    assert np.all(mtm.axis == normalize(ax))
    assert np.all(mtm.axis == normalize(ax))
    assert mtm.bias  == np.zeros(1)
    assert mtm.bias_std_rate == np.zeros(1)
    assert mtm.sample_time == 0.1
    assert not mtm.has_bias
    assert not mtm.use_noise
    assert 0==mtm.last_bias_update
    assert 0.243==mtm.std
    assert sc == mtm.scale

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[mtm])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

    assert np.isclose(np.dot(ax/3,B_B)*sc,mtm.reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: mtm.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: MTM(ax,std,has_bias = False,use_noise = False,bias = c,scale = sc).reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(20000)

    assert np.isclose(bfun(20000), np.dot(ax/3,B_B)*sc)
    assert np.isclose( xfun(x0) , np.dot(ax/3,B_B)*sc)

    assert np.allclose(Jxfun.T, mtm.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, mtm.bias_jac(x0,vecs))

    assert np.all(np.isclose( mtm.bias_jac(x0,vecs) , np.dot(ax/3,B_B)*sc ))
    assert np.all(np.isclose( mtm.basestate_jac(x0,vecs) , np.vstack([np.zeros((3,1)),np.dot(drotmatTvecdq(q0,B_ECI),np.expand_dims(ax/3,1))*sc]) ))

def test_mtm_reading_etc_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 7.3
    bsr = 0.001
    sc = 1e3
    mtm = MTM(ax,std,has_bias = True,bias = bias,use_noise = False,bias_std_rate = bsr,scale = sc)
    assert np.all(mtm.axis == normalize(ax))
    assert np.all(mtm.axis == normalize(ax))
    assert mtm.bias  == bias
    assert mtm.bias_std_rate == 0.001
    assert mtm.sample_time == 0.1
    assert mtm.has_bias
    assert not mtm.use_noise
    assert 0==mtm.last_bias_update
    assert 0.243==mtm.std
    assert sc == mtm.scale

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[mtm])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

    assert np.isclose(np.dot(ax/3,B_B)*sc + bias,mtm.reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: mtm.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: MTM(ax,std,has_bias = True,use_noise = False,bias = c,bias_std_rate=bsr,scale = sc).reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    assert np.isclose(bfun(bias), np.dot(ax/3,B_B)*sc+bias)
    assert np.isclose( xfun(x0) , np.dot(ax/3,B_B)*sc+bias)

    assert np.allclose(Jxfun.T, mtm.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, mtm.bias_jac(x0,vecs))

    assert np.all(np.isclose( mtm.bias_jac(x0,vecs) , 1 ))
    assert np.all(np.isclose( mtm.basestate_jac(x0,vecs) , np.vstack([np.zeros((3,1)),np.dot(drotmatTvecdq(q0,B_ECI),np.expand_dims(ax/3,1))*sc]) ))

    N = 1000
    dt = 0.8
    test_read = mtm.clean_reading(x0,vecs) + bias
    opts = [mtm.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    mtm.update_bias(0.22)

    reading_drift = [mtm.reading(x0,vecs,update_bias = True,j2000 = mtm.last_bias_update+dt*sec2cent).item()-mtm.reading(x0,vecs,update_bias = True,j2000 = mtm.last_bias_update+dt*sec2cent).item() for j in range(N)]
    exp_dist = [np.random.normal(0,sc*bsr*math.sqrt(dt)) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    mtm.update_bias(0.22)
    dt = 0.8
    oldb = mtm.bias
    test_read = mtm.clean_reading(x0,vecs) + oldb
    last = mtm.last_bias_update
    opts = [mtm.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    mtm.update_bias(0.21)
    assert mtm.last_bias_update == last
    assert oldb == mtm.bias

    reading_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = mtm.bias
        assert mtm.reading(x0,vecs) == mtm.clean_reading(x0,vecs)+bk
        mtm.update_bias(t)
        assert mtm.last_bias_update == t
        reading_drift += [(mtm.bias-bk).item()]
        assert mtm.reading(x0,vecs) == mtm.clean_reading(x0,vecs)+mtm.bias
    exp_dist = [np.random.normal(0,bsr*dt*sc) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_mtm_reading_etc_noise():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 7.3
    bsr = 0.001
    sc = 1e3
    mtm = MTM(ax,std,has_bias = True,bias = bias,use_noise = True,bias_std_rate = bsr,scale = sc)
    assert np.all(mtm.axis == normalize(ax))
    assert mtm.bias  == bias
    assert mtm.bias_std_rate == 0.001
    assert mtm.sample_time == 0.1
    assert mtm.has_bias
    assert mtm.use_noise
    assert 0==mtm.last_bias_update
    assert 0.243==mtm.std
    assert sc == mtm.scale

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[mtm])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

    assert np.isclose(np.dot(ax/3,B_B)*sc + bias,mtm.no_noise_reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: mtm.no_noise_reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: MTM(ax,std,has_bias = True,use_noise = True,bias = c,bias_std_rate=bsr,scale = sc).no_noise_reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    assert np.isclose(bfun(bias), np.dot(ax/3,B_B)*sc+bias)
    assert np.isclose( xfun(x0) , np.dot(ax/3,B_B)*sc+bias)

    assert np.allclose(Jxfun.T, mtm.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, mtm.bias_jac(x0,vecs))

    assert np.allclose(mtm.orbitRV_jac(x0,vecs),np.zeros((6,1)))

    assert np.all(np.isclose( mtm.bias_jac(x0,vecs) , 1 ))
    assert np.all(np.isclose( mtm.basestate_jac(x0,vecs) , np.vstack([np.zeros((3,1)),np.dot(drotmatTvecdq(q0,B_ECI),np.expand_dims(ax/3,1))*sc]) ))

    N = 1000
    sens_exp = mtm.no_noise_reading(x0,vecs)
    sens_err = [(mtm.reading(x0,vecs)-sens_exp).item() for j in range(N)]
    exp_dist = [np.random.normal(0,std*sc) for j in range(N)]

    ks0 = kstest(sens_err,exp_dist)
    data_a = sens_err
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    assert mtm.cov() == std**2.0*sc*sc

def test_gps_reading_etc_clean():
    gps = GPS(np.ones(6), has_bias = False,use_noise = False)
    assert gps.sample_time == 0.1
    assert 0==gps.last_bias_update
    assert not gps.has_bias
    assert np.all(gps.bias  == np.zeros(6))
    assert np.all(gps.bias_std_rate  == np.zeros(6))
    assert not gps.use_noise
    assert np.all(np.ones(6)==gps.std)
    # assert np.all(gps.prev_r_noisy == np.zeros(3))
    # assert np.all(gps.prev_r == np.zeros(3))
    # assert np.all(gps.r == np.zeros(3))
    # assert np.all(gps.r_noisy == np.zeros(3))
    # assert gps.prev_t==-1
    # assert gps.t==0

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[gps])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"os":os}
    out = gps.reading(x0,vecs)
    assert np.allclose(gps.reading(x0,vecs),np.concatenate([os.ECEF,os.eci_to_ecef(V)]))

    assert gps.sample_time == 0.1
    assert 0==gps.last_bias_update
    assert not gps.has_bias
    assert np.all(gps.bias  == np.zeros(6))
    assert np.all(gps.bias_std_rate  == np.zeros(6))
    assert not gps.use_noise
    assert np.all(np.ones(6)==gps.std)
    # assert np.all(gps.prev_r_noisy == out[0:3])
    # assert np.all(gps.prev_r == np.zeros(3))
    # assert np.all(gps.r == np.zeros(3))
    # assert np.all(gps.r_noisy == np.zeros(3))
    # assert gps.prev_t==-1
    # assert gps.t==0



    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[gps])
    x0 = np.concatenate([w0,q0])
    Rr = random_n_unit_vec(3)*np.random.uniform(6800,9000)
    Vv = random_n_unit_vec(3)*np.random.uniform(4,20)
    os = Orbital_State(0.22,Rr,Vv)
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"os":os}
    assert np.allclose(gps.reading(x0,vecs),np.concatenate([os.ECEF,os.eci_to_ecef(os.V)]))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R),"os":os}

    xfun = lambda c: gps.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    bfun = lambda c: GPS(np.ones(6),has_bias = False,use_noise = False,bias = c).reading(x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Jbfun = np.array(nd.Jacobian(bfun)(20000)).T

    assert np.allclose(bfun(20000) , np.concatenate([os.ECEF,os.eci_to_ecef(os.V)]))
    assert np.allclose(xfun(x0) , np.concatenate([os.ECEF,os.eci_to_ecef(os.V)]))

    assert np.allclose(Jxfun, gps.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, gps.bias_jac(x0,vecs))

    assert np.all(np.isclose( gps.bias_jac(x0,vecs) , np.zeros((0,6))))
    assert np.all(np.isclose( gps.basestate_jac(x0,vecs) , np.zeros((7,6)) ))

def test_gps_reading_etc_bias():

    bias = np.random.uniform(1,4)*random_n_unit_vec(6)
    bsr = np.abs(np.random.uniform(0.001,0.1)*random_n_unit_vec(6))

    gps = GPS(np.ones(6), has_bias = True,bias = bias,use_noise = False,bias_std_rate = bsr)
    assert gps.sample_time == 0.1
    assert 0==gps.last_bias_update
    assert gps.has_bias
    assert np.all(gps.bias  == bias)
    assert np.all(gps.bias_std_rate  == bsr)
    assert not gps.use_noise
    assert np.all(np.ones(6)==gps.std)
    # assert np.all(gps.prev_r_noisy == np.zeros(3))
    # assert np.all(gps.prev_r == np.zeros(3))
    # assert np.all(gps.r == np.zeros(3))
    # assert np.all(gps.r_noisy == np.zeros(3))
    # assert gps.prev_t==-1
    # assert gps.t==0

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[gps])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"os":os}
    out = gps.reading(x0,vecs)
    assert np.allclose(gps.reading(x0,vecs),np.concatenate([os.ECEF,os.eci_to_ecef(V)])+bias)

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R),"os":os}

    xfun = lambda c: gps.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    bfun = lambda c: GPS(np.ones(6),has_bias = True,use_noise = False,bias = c,bias_std_rate=bsr).reading(x0,vecs)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist()))
    Jbfun = np.array(nd.Jacobian(bfun)(bias.flatten().tolist()))

    assert np.allclose(bfun(bias), np.concatenate([os.ECEF,os.eci_to_ecef(V)])+bias)
    assert np.allclose( xfun(x0) , np.concatenate([os.ECEF,os.eci_to_ecef(V)])+bias)

    assert np.allclose(Jxfun.T, gps.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, gps.bias_jac(x0,vecs))

    assert np.all(np.isclose( gps.bias_jac(x0,vecs) , np.eye(6) ))
    assert np.all(np.isclose( gps.basestate_jac(x0,vecs) , np.zeros((7,6))))#np.vstack([np.zeros((3,1)),np.dot(drotmatTvecdq(q0,B_ECI),np.expand_dims(ax/3,1))]) ))

    N = 1000
    dt = 0.8
    test_read = gps.clean_reading(x0,vecs) + bias
    opts = [gps.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    gps.update_bias(0.22)

    reading_drift = [gps.reading(x0,vecs,update_bias = True,j2000 = gps.last_bias_update+dt*sec2cent)-gps.reading(x0,vecs,update_bias = True,j2000 = gps.last_bias_update+dt*sec2cent) for j in range(N)]
    exp_dist = [np.random.normal(0,bsr*math.sqrt(dt)) for j in range(N)]

    for j in range(6):
        ksj = kstest([i[j] for i in reading_drift],[i[j] for i in exp_dist])
        print(j,ksj.statistic)
        data_a = reading_drift
        data_b = exp_dist
        hist = np.histogram([dd[j] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum([dd[j]<ee for dd in data_b]) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ksj.pvalue>0.1 or np.abs(ksj.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    gps.update_bias(0.22)
    dt = 0.8
    oldb = gps.bias
    test_read = gps.clean_reading(x0,vecs) + oldb
    last = gps.last_bias_update
    opts = [gps.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    gps.update_bias(0.21)
    assert gps.last_bias_update == last
    assert np.allclose(oldb , gps.bias)

    reading_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = gps.bias
        assert np.allclose(gps.reading(x0,vecs) , gps.clean_reading(x0,vecs)+bk)
        gps.update_bias(t)
        assert gps.last_bias_update == t
        reading_drift += [gps.bias-bk]
        assert  np.allclose(gps.reading(x0,vecs) , gps.clean_reading(x0,vecs)+gps.bias)
    exp_dist = [np.array([np.random.normal(0,bsr[i]*dt) for i in range(6)]) for j in range(N)]

    for j in range(6):
        ksj = kstest([i[j] for i in reading_drift],[i[j] for i in exp_dist])
        print(j,ksj.statistic)
        data_a = reading_drift
        data_b = exp_dist
        hist = np.histogram([dd[j] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum([dd[j]<ee for dd in data_b]) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ksj.pvalue>0.1 or np.abs(ksj.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_gps_reading_etc_noise():

    bias = np.random.uniform(1,4)*random_n_unit_vec(6)
    bsr = np.abs(np.random.uniform(0.001,0.1)*random_n_unit_vec(6))
    std = np.abs(np.random.uniform(0.001,0.1)*random_n_unit_vec(6))


    gps = GPS(std, has_bias = True,bias = bias,use_noise = True,bias_std_rate = bsr)
    assert gps.sample_time == 0.1
    assert 0==gps.last_bias_update
    assert 0 == gps.last_noise_model_update
    assert gps.has_bias
    assert np.all(gps.bias  == bias)
    assert np.all(gps.bias_std_rate  == bsr)
    assert gps.use_noise
    assert np.all(std==gps.std)
    # assert np.all(gps.prev_r_noisy == np.zeros(3))
    # assert np.all(gps.prev_r == np.zeros(3))
    # assert np.all(gps.r == np.zeros(3))
    # assert np.all(gps.r_noisy == np.zeros(3))
    # assert gps.prev_t==-1
    # assert gps.t==0

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[gps])
    x0 = np.concatenate([w0,q0])
    R_ECI = 7000*unitvecs[0]#random_n_unit_vec(3)*np.random.uniform(6800,7500)
    V_ECI = 8*unitvecs[1]#random_n_unit_vec(3)*np.random.uniform(5,12)
    os = Orbital_State(0.22,R_ECI,V_ECI)
    R = R_ECI
    V = V_ECI
    B = os.B
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"os":os}
    out = gps.reading(x0,vecs)
    assert np.allclose(gps.no_noise_reading(x0,vecs),np.concatenate([os.ECEF,os.eci_to_ecef(V)])+bias)

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R),"os":os}

    Rfun = lambda c: np.array([c[0],c[1],c[2]])
    Vfun = lambda c: np.array([c[3],c[4],c[5]])

    orbfun = lambda c:  Orbital_State(0.22,Rfun(c),Vfun(c))

    orbvecs_fun = lambda c: {"b":B_B,"r":rmat_ECI2B@Rfun(c),"s":S_B,"v":rmat_ECI2B@Vfun(c),"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":drotmatTvecdq(q0,Vfun(c)),"dr": drotmatTvecdq(q0,Rfun(c)),"os":orbfun(c)}

    xfun = lambda c: gps.no_noise_reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c))
    bfun = lambda c: GPS(std,has_bias = True,use_noise = True,bias = c,bias_std_rate=bsr).no_noise_reading(x0,vecs)
    rvfun = lambda c: GPS(std,has_bias = True,use_noise = True,bias = bias,bias_std_rate=bsr).no_noise_reading(x0,orbvecs_fun(c))
    rv = np.concatenate([R_ECI,V_ECI])

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist()))
    Jbfun = np.array(nd.Jacobian(bfun)(bias.flatten().tolist()))
    Jrvfun = np.array(nd.Jacobian(rvfun)(rv.flatten().tolist()))

    assert np.allclose(bfun(bias), np.concatenate([os.ECEF,os.eci_to_ecef(V)])+bias)
    assert np.allclose( xfun(x0) , np.concatenate([os.ECEF,os.eci_to_ecef(V)])+bias)

    np.set_printoptions(precision=3)
    assert np.allclose(rvfun(rv), np.concatenate([os.ECEF,os.eci_to_ecef(V)])+bias)

    assert np.allclose(Jxfun.T, gps.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, gps.bias_jac(x0,vecs))
    assert np.allclose(Jrvfun,gps.orbitRV_jac(x0,vecs))

    assert np.allclose(gps.orbitRV_jac(x0,vecs),np.block([[framelib.itrs.rotation_at(os.sf_pos.t),0*np.eye(3)],[0*np.eye(3),framelib.itrs.rotation_at(os.sf_pos.t)]]))

    assert np.all(np.isclose( gps.bias_jac(x0,vecs) , np.eye(6) ))
    assert np.all(np.isclose( gps.basestate_jac(x0,vecs) , np.zeros((7,6))))#np.vstack([np.zeros((3,1)),np.dot(drotmatTvecdq(q0,B_ECI),np.expand_dims(ax/3,1))]) ))

    N = 1000

    sens_exp = gps.no_noise_reading(x0,vecs)
    sens_err = [gps.reading(x0,vecs)-sens_exp for j in range(N)]
    exp_dist = [np.random.normal(0,std) for j in range(N)]

    for j in range(6):
        ksj = kstest([i[j] for i in sens_err],[i[j] for i in exp_dist])
        print(j,ksj.statistic)
        data_a = sens_err
        data_b = exp_dist
        hist = np.histogram([dd[j] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum([dd[j]<ee for dd in data_b]) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        assert ksj.pvalue>0.1 or np.abs(ksj.statistic)<(np.sqrt((1/N)*-0.5*np.log(1e-5/6.0)))

    assert np.all(gps.cov() == np.diagflat(std**2.0))

def test_gyro_reading_etc_clean():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    gyro = Gyro(ax,std,has_bias = False,use_noise = False)
    assert np.all(gyro.axis == normalize(ax))
    assert np.all(gyro.axis == normalize(ax))
    assert gyro.bias  == np.zeros(1)
    assert gyro.bias_std_rate  == np.zeros(1)
    assert gyro.sample_time == 0.1
    assert not gyro.has_bias
    assert not gyro.use_noise
    assert 0==gyro.last_bias_update
    assert 0.243==gyro.std

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[gyro])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

    assert np.isclose(np.dot(ax/3,w0),gyro.reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: gyro.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: Gyro(ax,std,has_bias = False,use_noise = False,bias = c).reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(20000)

    assert np.isclose(bfun(20000) , np.dot(ax/3,w0))
    assert  np.isclose(xfun(x0) , np.dot(ax/3,w0))

    assert np.allclose(Jxfun.T, gyro.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, gyro.bias_jac(x0,vecs))

    assert np.all(np.isclose( gyro.bias_jac(x0,vecs) , np.dot(ax/3,w0) ))
    print( gyro.basestate_jac(x0,vecs) )
    print( np.expand_dims(np.concatenate([ax/3,np.zeros(4)]),0))
    assert np.all(np.isclose( gyro.basestate_jac(x0,vecs) , np.expand_dims(np.concatenate([ax/3,np.zeros(4)]),0).T ))

def test_gyro_reading_etc_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 0.073
    bsr = 0.001
    gyro = Gyro(ax,std,has_bias = True,bias = bias,use_noise = False,bias_std_rate = bsr)
    assert np.all(gyro.axis == normalize(ax))
    assert np.all(gyro.axis == normalize(ax))
    assert gyro.bias  == bias
    assert gyro.bias_std_rate == 0.001
    assert gyro.sample_time == 0.1
    assert gyro.has_bias
    assert not gyro.use_noise
    assert 0==gyro.last_bias_update
    assert 0.243==gyro.std

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[gyro])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

    assert np.isclose(np.dot(ax/3,w0) + bias,gyro.reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: gyro.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: Gyro(ax,std,has_bias = True,use_noise = False,bias = c,bias_std_rate=bsr).reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    assert np.isclose(bfun(bias), np.dot(ax/3,w0)+bias)
    assert np.isclose( xfun(x0) , np.dot(ax/3,w0)+bias)

    assert np.allclose(Jxfun.T, gyro.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, gyro.bias_jac(x0,vecs))

    assert np.all(np.isclose( gyro.bias_jac(x0,vecs) , 1 ))
    assert np.all(np.isclose( gyro.basestate_jac(x0,vecs) , np.expand_dims(np.concatenate([ax/3,np.zeros(4)]),1) ))

    N = 1000
    dt = 0.8
    test_read = gyro.clean_reading(x0,vecs) + bias
    opts = [gyro.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    gyro.update_bias(0.22)

    reading_drift = [gyro.reading(x0,vecs,update_bias = True,j2000 = gyro.last_bias_update+dt*sec2cent).item()-gyro.reading(x0,vecs,update_bias = True,j2000 = gyro.last_bias_update+dt*sec2cent).item() for j in range(N)]
    exp_dist = [np.random.normal(0,bsr*math.sqrt(dt)) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    gyro.update_bias(0.22)
    dt = 0.8
    oldb = gyro.bias
    test_read = gyro.clean_reading(x0,vecs) + oldb
    last = gyro.last_bias_update
    opts = [gyro.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    gyro.update_bias(0.21)
    assert gyro.last_bias_update == last
    assert oldb == gyro.bias

    reading_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = gyro.bias
        assert gyro.reading(x0,vecs) == gyro.clean_reading(x0,vecs)+bk
        gyro.update_bias(t)
        assert gyro.last_bias_update == t
        reading_drift += [(gyro.bias-bk).item()]
        assert gyro.reading(x0,vecs) == gyro.clean_reading(x0,vecs)+gyro.bias
    exp_dist = [np.random.normal(0,bsr*dt) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_gyro_reading_etc_noise():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 0.073
    bsr = 0.001
    gyro = Gyro(ax,std,has_bias = True,bias = bias,use_noise = True,bias_std_rate = bsr)
    assert np.all(gyro.axis == normalize(ax))
    assert gyro.bias  == bias
    assert gyro.bias_std_rate == 0.001
    assert gyro.sample_time == 0.1
    assert gyro.has_bias
    assert gyro.use_noise
    assert 0==gyro.last_bias_update
    assert 0.243==gyro.std

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[gyro])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

    assert np.isclose(np.dot(ax/3,w0) + bias,gyro.no_noise_reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: gyro.no_noise_reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: Gyro(ax,std,has_bias = True,use_noise = True,bias = c,bias_std_rate=bsr).no_noise_reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    assert np.isclose(bfun(bias), np.dot(ax/3,w0)+bias)
    assert np.isclose( xfun(x0) , np.dot(ax/3,w0)+bias)
    assert np.allclose(gyro.orbitRV_jac(x0,vecs),np.zeros((6,1)))

    assert np.allclose(Jxfun.T, gyro.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, gyro.bias_jac(x0,vecs))

    assert np.all(np.isclose( gyro.bias_jac(x0,vecs) , 1 ))
    assert np.all(np.isclose( gyro.basestate_jac(x0,vecs) ,  np.expand_dims(np.concatenate([ax/3,np.zeros(4)]),1) ))

    N = 1000
    sens_exp = gyro.no_noise_reading(x0,vecs)
    sens_err = [(gyro.reading(x0,vecs)-sens_exp).item() for j in range(N)]
    exp_dist = [np.random.normal(0,std) for j in range(N)]

    ks0 = kstest(sens_err,exp_dist)
    data_a = sens_err
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    assert gyro.cov() == std**2.0

def test_sun_reading_etc_clean():
    for t in range(10):
        ax = random_n_unit_vec(3)*3
        ax = ax.copy()
        max_moment = 4.51
        std = 0.243
        sun = SunSensor(ax,std,0.3,has_bias = False,use_noise = False)
        assert np.all(sun.axis == normalize(ax))
        assert np.all(sun.axis == normalize(ax))
        assert sun.bias  == np.zeros(1)
        assert sun.bias_std_rate  == np.zeros(1)
        assert sun.sample_time == 0.1
        assert not sun.has_bias
        assert not sun.use_noise
        assert 0==sun.last_bias_update
        assert 0.243==sun.std
        assert 0 == sun.degradation_value
        assert 0.3 == sun.efficiency

        B_ECI = random_n_unit_vec(3)
        q0 = random_n_unit_vec(4)
        R = rot_mat(q0)
        w0 = 0.05*random_n_unit_vec(3)
        sat = Satellite(sensors=[sun])
        x0 = np.concatenate([w0,q0])
        os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
        R = os.R
        V = os.V
        B = B_ECI
        S = os.S
        rho = os.rho
        rmat_ECI2B = rot_mat(q0).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V
        dR_B__dq = drotmatTvecdq(q0,R)
        dB_B__dq = drotmatTvecdq(q0,B)
        dV_B__dq = drotmatTvecdq(q0,V)
        dS_B__dq = drotmatTvecdq(q0,S)
        nSB = normalize(S_B-R_B)
        vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

        exp = max(0,np.dot(ax/3,nSB)*0.3)
        assert np.isclose(exp,sun.reading(x0,vecs))

        vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                    "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

        xfun = lambda c: sun.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
        bfun = lambda c: SunSensor(ax,std,0.3,has_bias = False,use_noise = False,bias = c).reading(x0,vecs).item()

        Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
        Jbfun = nd.Jacobian(bfun)(20000)

        assert np.isclose(bfun(20000) , exp)
        assert  np.isclose(xfun(x0) ,exp)

        assert np.allclose(Jxfun.T, sun.basestate_jac(x0,vecs))
        assert np.allclose(Jbfun, sun.bias_jac(x0,vecs))

        assert np.all(np.isclose( sun.bias_jac(x0,vecs) , 1))

def test_sun_reading_etc_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 0.073
    bsr = 0.001
    sun = SunSensor(ax,std,0.3,has_bias = True,bias = bias,use_noise = False,bias_std_rate = bsr)
    assert np.all(sun.axis == normalize(ax))
    assert np.all(sun.axis == normalize(ax))
    assert sun.bias  == bias
    assert sun.bias_std_rate == 0.001
    assert sun.sample_time == 0.1
    assert sun.has_bias
    assert not sun.use_noise
    assert 0==sun.last_bias_update
    assert 0.243==sun.std

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[sun])
    x0 = np.concatenate([w0,q0])
    R_ECI = random_n_unit_vec(3)*np.random.uniform(6800,7500)
    os = Orbital_State(0.22,R_ECI,np.array([0,8,0]),B=B_ECI)
    R = R_ECI
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}


    nSB = normalize(S_B-R_B)
    exp = max(0,np.dot(ax/3,nSB)*0.3)
    print(exp+bias,sun.reading(x0,vecs))
    assert np.isclose(exp+bias,sun.reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: sun.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: SunSensor(ax,std,0.3,has_bias = True,use_noise = False,bias = c,bias_std_rate=bsr).reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    assert np.isclose(bfun(bias) , exp+bias)
    assert  np.isclose(xfun(x0) ,exp+bias)

    assert np.allclose(Jxfun.T, sun.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, sun.bias_jac(x0,vecs))

    assert np.all(np.isclose( sun.bias_jac(x0,vecs) , 1 ))

    N = 1000
    dt = 0.8
    test_read = sun.clean_reading(x0,vecs) + bias
    opts = [sun.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    sun.update_bias(0.22)

    reading_drift = [sun.reading(x0,vecs,update_bias = True,j2000 = sun.last_bias_update+dt*sec2cent).item()-sun.reading(x0,vecs,update_bias = True,j2000 = sun.last_bias_update+dt*sec2cent).item() for j in range(N)]
    exp_dist = [np.random.normal(0,bsr*math.sqrt(dt)) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    sun.update_bias(0.22)
    dt = 0.8
    oldb = sun.bias
    test_read = sun.clean_reading(x0,vecs) + oldb
    last = sun.last_bias_update
    opts = [sun.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    sun.update_bias(0.21)
    assert sun.last_bias_update == last
    assert oldb == sun.bias

    reading_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = sun.bias
        assert sun.reading(x0,vecs) == sun.clean_reading(x0,vecs)+bk
        sun.update_bias(t)
        assert sun.last_bias_update == t
        reading_drift += [(sun.bias-bk).item()]
        assert sun.reading(x0,vecs) == sun.clean_reading(x0,vecs)+sun.bias
    exp_dist = [np.random.normal(0,bsr*dt) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_sun_reading_etc_noise():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 0.073
    bsr = 0.001
    sun = SunSensor(ax,std,0.2,has_bias = True,bias = bias,use_noise = True,bias_std_rate = bsr)
    assert np.all(sun.axis == normalize(ax))
    assert sun.bias  == bias
    assert sun.bias_std_rate == 0.001
    assert sun.sample_time == 0.1
    assert sun.has_bias
    assert sun.use_noise
    assert 0==sun.last_bias_update
    assert 0.243==sun.std
    assert 0.2==sun.efficiency

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[sun])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}


    nSB = normalize(S_B-R_B)
    exp = max(0,np.dot(ax/3,nSB)*0.2)
    print(exp+bias,sun.no_noise_reading(x0,vecs))
    assert np.isclose(exp+bias,sun.no_noise_reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: sun.no_noise_reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: SunSensor(ax,std,0.2,has_bias = True,use_noise = True,bias = c,bias_std_rate=bsr).no_noise_reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    assert np.isclose(bfun(bias), exp+bias)
    assert np.isclose( xfun(x0) , exp+bias)

    assert np.allclose(Jxfun.T, sun.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, sun.bias_jac(x0,vecs))

    assert np.all(np.isclose( sun.bias_jac(x0,vecs) , 1 ))

    assert np.allclose(sun.orbitRV_jac(x0,vecs),np.zeros((6,1)))

    N = 1000
    sens_exp = sun.no_noise_reading(x0,vecs).item()
    sens_err = [sun.reading(x0,vecs).item()-sens_exp for j in range(N)]
    exp_dist = [np.random.normal(0,std) for j in range(N)]
    exp_dist = [np.maximum(0,j+sens_exp)-sens_exp for j in exp_dist] #can't be negative.

    ks0 = kstest(sens_err,exp_dist)
    data_a = sens_err
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    assert sun.cov() == std**2.0



def test_sunpair_reading_etc_clean():
    for t in range(10):
        ax = random_n_unit_vec(3)*3
        ax = ax.copy()
        max_moment = 4.51
        std = 0.243
        sun = SunSensorPair(ax,std,[0.3,0.2],has_bias = False,use_noise = False)
        assert np.all(sun.axis == normalize(ax))
        assert np.all(sun.axis == normalize(ax))
        assert sun.bias  == np.zeros(1)
        assert sun.bias_std_rate  == np.zeros(1)
        assert sun.sample_time == 0.1
        assert not sun.has_bias
        assert not sun.use_noise
        assert 0==sun.last_bias_update
        assert 0.243==sun.std
        assert 0 == sun.degradation_value[0]
        assert 0 == sun.degradation_value[1]
        assert 0.3 == sun.efficiency[0]
        assert 0.2 == sun.efficiency[1]

        B_ECI = random_n_unit_vec(3)
        q0 = random_n_unit_vec(4)
        R = rot_mat(q0)
        w0 = 0.05*random_n_unit_vec(3)
        sat = Satellite(sensors=[sun])
        x0 = np.concatenate([w0,q0])
        os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
        R = os.R
        V = os.V
        B = B_ECI
        S = os.S
        rho = os.rho
        rmat_ECI2B = rot_mat(q0).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V
        dR_B__dq = drotmatTvecdq(q0,R)
        dB_B__dq = drotmatTvecdq(q0,B)
        dV_B__dq = drotmatTvecdq(q0,V)
        dS_B__dq = drotmatTvecdq(q0,S)
        nSB = normalize(S_B-R_B)
        vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}

        exp = np.dot(ax/3,nSB).item()
        exp = exp*(0.3*(exp>0) + 0.2*(exp<0))
        assert np.isclose(exp,sun.reading(x0,vecs))

        vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                    "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

        xfun = lambda c: sun.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
        bfun = lambda c: SunSensorPair(ax,std,[0.3,0.2],has_bias = False,use_noise = False,bias = c).reading(x0,vecs).item()

        Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
        Jbfun = nd.Jacobian(bfun)(20000)

        assert np.isclose(bfun(20000) , exp)
        assert  np.isclose(xfun(x0) ,exp)

        assert np.allclose(Jxfun.T, sun.basestate_jac(x0,vecs))
        assert np.allclose(Jbfun, sun.bias_jac(x0,vecs))

        assert np.all(np.isclose( sun.bias_jac(x0,vecs) , 1))

def test_sunpair_reading_etc_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 0.073
    bsr = 0.001
    sun = SunSensorPair(ax,std,[0.3,0.2],has_bias = True,bias = bias,use_noise = False,bias_std_rate = bsr)
    assert np.all(sun.axis == normalize(ax))
    assert np.all(sun.axis == normalize(ax))
    assert sun.bias  == bias
    assert sun.bias_std_rate == 0.001
    assert sun.sample_time == 0.1
    assert sun.has_bias
    assert not sun.use_noise
    assert 0==sun.last_bias_update
    assert 0.243==sun.std

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[sun])
    x0 = np.concatenate([w0,q0])
    R_ECI = random_n_unit_vec(3)*np.random.uniform(6800,7500)
    os = Orbital_State(0.22,R_ECI,np.array([0,8,0]),B=B_ECI)
    R = R_ECI
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}


    nSB = normalize(S_B-R_B)
    exp = np.dot(ax/3,nSB).item()
    exp = exp*(0.3*(exp>0) + 0.2*(exp<0))
    print(exp+bias,sun.reading(x0,vecs))
    assert np.isclose(exp+bias,sun.reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: sun.reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: SunSensorPair(ax,std,[0.3,0.2],has_bias = True,use_noise = False,bias = c,bias_std_rate=bsr).reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    print(bias,bfun(bias),exp,exp+bias)

    assert np.isclose(bfun(bias) , exp+bias)
    assert  np.isclose(xfun(x0) ,exp+bias)

    assert np.allclose(Jxfun.T, sun.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, sun.bias_jac(x0,vecs))

    assert np.all(np.isclose( sun.bias_jac(x0,vecs) , 1 ))

    N = 1000
    dt = 0.8
    test_read = sun.clean_reading(x0,vecs) + bias
    opts = [sun.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    sun.update_bias(0.22)

    reading_drift = [-sun.reading(x0,vecs,update_bias = True,j2000 = sun.last_bias_update+dt*sec2cent).item()+sun.reading(x0,vecs,update_bias = True,j2000 = sun.last_bias_update+dt*sec2cent).item() for j in range(N)]
    exp_dist = [np.random.normal(0,bsr*math.sqrt(dt)) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    sun.update_bias(0.22)
    dt = 0.8
    oldb = sun.bias
    test_read = sun.clean_reading(x0,vecs) + oldb
    last = sun.last_bias_update
    opts = [sun.reading(x0,vecs) for j in range(N)]
    assert np.all([np.allclose(test_read,j) for j in opts]) #bias is constant
    sun.update_bias(0.21)
    assert sun.last_bias_update == last
    assert oldb == sun.bias

    reading_drift = []
    tlist = [last+(j+1)*dt*sec2cent for j in range(N)]
    for t in tlist:
        bk = sun.bias
        assert sun.reading(x0,vecs) == sun.clean_reading(x0,vecs)+bk
        sun.update_bias(t)
        assert sun.last_bias_update == t
        reading_drift += [(sun.bias-bk).item()]
        assert sun.reading(x0,vecs) == sun.clean_reading(x0,vecs)+sun.bias
    exp_dist = [np.random.normal(0,bsr*dt) for j in range(N)]
    print(kstest(reading_drift,exp_dist).statistic)

    ks0 = kstest(reading_drift,exp_dist)
    data_a = reading_drift
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

def test_sunpair_reading_etc_noise():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_moment = 4.51
    std = 0.243
    bias = 0.073
    bsr = 0.001
    sun = SunSensorPair(ax,std,[0.2,0.1],has_bias = True,bias = bias,use_noise = True,bias_std_rate = bsr)
    assert np.all(sun.axis == normalize(ax))
    assert sun.bias  == bias
    assert sun.bias_std_rate == 0.001
    assert sun.sample_time == 0.1
    assert sun.has_bias
    assert sun.use_noise
    assert 0==sun.last_bias_update
    assert 0.243==sun.std
    assert 0.2==sun.efficiency[0]
    assert 0.1==sun.efficiency[1]

    B_ECI = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(sensors=[sun])
    x0 = np.concatenate([w0,q0])
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),B=B_ECI)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq}


    nSB = normalize(S_B-R_B)
    exp = np.dot(ax/3,nSB).item()
    exp = exp*(0.2*(exp>0) + 0.1*(exp<0))
    print(exp+bias,sun.no_noise_reading(x0,vecs))
    assert np.isclose(exp+bias,sun.no_noise_reading(x0,vecs))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}

    xfun = lambda c: sun.no_noise_reading(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]),vecsxfun(c)).item()
    bfun = lambda c: SunSensorPair(ax,std,[0.2,0.1],has_bias = True,use_noise = True,bias = c,bias_std_rate=bsr).no_noise_reading(x0,vecs).item()

    Jxfun = nd.Jacobian(xfun)(x0.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(bias)

    print(bias, bfun(bias), exp,exp+bias)
    assert np.isclose(bfun(bias), exp+bias)
    assert np.isclose( xfun(x0) , exp+bias)

    assert np.allclose(Jxfun.T, sun.basestate_jac(x0,vecs))
    assert np.allclose(Jbfun, sun.bias_jac(x0,vecs))

    assert np.all(np.isclose( sun.bias_jac(x0,vecs) , 1 ))

    assert np.allclose(sun.orbitRV_jac(x0,vecs),np.zeros((6,1)))

    N = 1000
    sens_exp = sun.no_noise_reading(x0,vecs).item()
    sens_err = [sun.reading(x0,vecs).item()-sens_exp for j in range(N)]
    exp_dist = [np.random.normal(0,std) for j in range(N)]
    # exp_dist = [np.maximum(0,j+sens_exp)-sens_exp for j in exp_dist] #can't be negative.

    ks0 = kstest(sens_err,exp_dist)
    data_a = sens_err
    data_b = exp_dist
    hist = np.histogram(data_a,bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd<ee for dd in data_b]) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))

    assert sun.cov() == std**2.0


def test_sat_sensors_no_noise():
    bias_mtq = 0.2*random_n_unit_vec(3)
    bias_magic = 0.2*random_n_unit_vec(3)
    bias_rw = 0.2*random_n_unit_vec(3)
    h_rw = 1.0*random_n_unit_vec(3)
    B_ECI = random_n_unit_vec(3)

    bias_mtm = 0.3*random_n_unit_vec(3)
    bias_gyro = 0.1*random_n_unit_vec(3)
    bias_sun = 30*random_n_unit_vec(9)
    bias_gps = np.concatenate([random_n_unit_vec(3)*60,random_n_unit_vec(3)*1])
    sun_eff = 0.3

    mtqs = [MTQ(j,0,1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=False,bias_std_rate=0) for j in unitvecs]
    magics = [Magic(j,0,1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=False,bias_std_rate=0) for j in unitvecs]
    rws = [RW(j,0,1,0.1,np.dot(h_rw,j),2,0,has_bias = True, bias = np.dot(bias_rw,j),use_noise=False,bias_std_rate=0) for j in unitvecs]

    mtms = [MTM(j,0,has_bias = True,bias = np.dot(bias_mtm,j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,0,has_bias = True,bias = np.dot(bias_gyro,j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    suns1 = [SunSensor(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[0:3],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    suns2 = [SunSensor(-j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[3:6],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    suns3 = [SunSensorPair(j,0,sun_eff,has_bias = True,bias = np.dot(bias_sun[6:],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    gps = GPS(0,has_bias = True,bias = bias_gps,use_noise = False,bias_std_rate = 0)

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0
    H_body = J_body@w0 + h_rw
    H_ECI = J_ECI@w_ECI + Rm@h_rw
    acts = mtqs+magics+rws
    sensors = mtms+gyros+suns1+suns2+suns3+[gps]
    u = 5*random_n_unit_vec(9)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(6900,7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,15)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    os = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,S=S_ECI)
    R = R_ECI
    V = V_ECI
    B = B_ECI
    S = S_ECI
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R),"os":os}

    sat = Satellite(J = J_body,actuators = acts,sensors=sensors)
    state = np.concatenate([w0,q0,h_rw])
    exp_wd = -np.linalg.inv(sat.J_noRW)@np.cross(w0,H_body) + np.linalg.inv(sat.J_noRW)@sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9)],np.zeros(3))
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    exp_hd = sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9) if not acts[j].has_momentum],np.zeros(3)) - sat.J@exp_wd - np.cross(w0,H_body)
    xd = sat.dynamics(state,u,os)
    np.set_printoptions(precision=3)
    assert np.allclose(np.concatenate([exp_wd,exp_qd,exp_hd]),xd)

    sensed = sat.sensor_values(state,vecs)
    assert np.allclose(sensed,np.concatenate([[j.reading(state,vecs)] for j in sensors[:-1]]+[np.atleast_2d(h_rw).T]).flatten())
    assert np.allclose(sensed,np.concatenate([[j.no_noise_reading(state,vecs)] for j in sensors[:-1]]+[np.atleast_2d(h_rw).T]).flatten())
    assert np.allclose(sensed,np.concatenate([[j.clean_reading(state,vecs) + j.bias] for j in sensors[:-1]]+[np.atleast_2d(h_rw).T]).flatten())
    assert np.allclose(sensed-np.concatenate([[j.clean_reading(state,vecs) ]for j in sensors[:-1]]+[h_rw]).flatten(),np.concatenate([bias_mtm,bias_gyro,bias_sun,np.zeros(3)]))

    for k in range(10):
        which = [np.random.uniform(0,1)<0.5 for j in range(15)]
        assert np.allclose(sat.sensor_values(state,vecs,which),np.concatenate([[sensors[j].reading(state,vecs)] for j in range(len(sensors[:-1])) if which[j]]+[np.atleast_2d(h_rw).T]).flatten())

    state_jac,bias_jac = sat.sensor_state_jacobian(state,vecs)
    assert np.allclose(state_jac,np.block([[j.basestate_jac(state,vecs) for j in sensors[:-1]]+[np.zeros((7,3))],[np.zeros((3,15)),np.eye(3)]]))
    assert np.allclose(bias_jac,np.hstack([np.eye(15),np.zeros((15,3))]))


    for k in range(10):
        which = [np.random.uniform(0,1)<0.5 for j in range(15)]
        state_jacw,bias_jacw = sat.sensor_state_jacobian(state,vecs,which)
        assert np.allclose(state_jacw,np.block([[sensors[j].basestate_jac(state,vecs) for j in range(len(sensors[:-1])) if which[j]]+[np.zeros((7,3))],[np.zeros((3,sum(which))),np.eye(3)]]))
        assert np.allclose(bias_jacw,np.hstack([np.eye(sum(which)),np.zeros((sum(which),3))]))


    fun_act = lambda c: mtqs+magics+[RW(unitvecs[j],0,1,0.1,c[j],2,0,has_bias = True, bias = bias_rw[j],use_noise=False,bias_std_rate=0) for j in range(3)]

    fun_mtms = lambda c: [MTM(unitvecs[j],0,has_bias = True,bias = c[j],use_noise = False,bias_std_rate = 0) for j in range(3)]
    fun_gyros = lambda c: [Gyro(unitvecs[j],0,has_bias = True,bias = c[j+3],use_noise = False,bias_std_rate = 0) for j in range(3)]
    fun_suns1 = lambda c: [SunSensor(unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+6],use_noise = False,bias_std_rate = 0) for j in range(3)]
    fun_suns2 = lambda c: [SunSensor(-unitvecs[j],0,sun_eff,has_bias = True,bias = c[j+9],use_noise = False,bias_std_rate = 0) for j in range(3)]
    fun_suns3 = lambda c: [SunSensorPair(unitvecs[j],0,[sun_eff,sun_eff],has_bias = True,bias = c[j+12],use_noise = False,bias_std_rate = 0) for j in range(3)]
    fun_gps =  lambda c: GPS(0,has_bias = True,bias = np.array([c[12],c[13],c[14],c[15],c[16],c[17]]),use_noise = False,bias_std_rate = 0)
    sens_fun = lambda c: fun_mtms(c)+fun_gyros(c)+fun_suns1(c)+fun_suns2(c)+fun_suns3(c)+[fun_gps(c)]
    fun_sat_act = lambda c:  Satellite(J = J_body,actuators = fun_act(c),sensors=sensors)
    fun_sat_sens = lambda c: Satellite(J=J_body,actuators = acts,sensors = sens_fun(c))

    sens_fun_state = lambda c :  fun_sat_act(np.array([c[7],c[8],c[9]])).sensor_values(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),vecsxfun(c))
    sens_fun_bias = lambda c: fun_sat_sens(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],c[20]])).sensor_values(state,vecs)

    # stateh = np.concatenate([state,h_rw])
    sensbias = np.concatenate([bias_mtm,bias_gyro,bias_sun,bias_gps])
    assert np.allclose(sens_fun_state(state),sensed)
    assert np.allclose(sens_fun_bias(sensbias),sensed)
    J_xfun = np.array(nd.Jacobian(sens_fun_state)(state.flatten().tolist()))
    J_bfun = np.array(nd.Jacobian(sens_fun_bias)(sensbias.flatten().tolist()))

    assert np.allclose(J_xfun.T,state_jac)
    assert np.allclose(J_bfun[:,:-6].T,bias_jac)


    for k in range(3):
        which = np.array([np.random.uniform(0,1)<0.5 for j in range(15)])
        state_jacw,bias_jacw = sat.sensor_state_jacobian(state,vecs,which)
        sensedw = sat.sensor_values(state,vecs,which)

        sens_fun_statew = lambda c :  fun_sat_act(np.array([c[7],c[8],c[9]])).sensor_values(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]]),vecsxfun(c),which)
        sens_fun_biasw = lambda c: fun_sat_sens(np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],c[20]])).sensor_values(state,vecs,which)

        assert np.allclose(sens_fun_statew(state),sensedw)
        assert np.allclose(sens_fun_biasw(sensbias),sensedw)
        J_xfunw = np.array(nd.Jacobian(sens_fun_statew)(state.flatten().tolist()))
        J_bfunw = np.array(nd.Jacobian(sens_fun_biasw)(sensbias.flatten().tolist()))

        assert np.allclose(J_xfunw.T,state_jacw)
        J_bfunw2 = J_bfunw[:,:-6]
        assert np.allclose(J_bfunw2[:,which].T,bias_jacw)

        assert np.allclose(J_xfunw.T,np.block([[sensors[j].basestate_jac(state,vecs) for j in range(len(sensors[:-1])) if which[j]]+[np.zeros((7,3))],[np.zeros((3,sum(which))),np.eye(3)]]))
        assert np.allclose(J_bfunw2[:,which].T,np.hstack([np.eye(sum(which)),np.zeros((sum(which),3))]))

def test_sat_sensing_noise_cov():
    bias_mtq = 0.2*random_n_unit_vec(3)
    bias_magic = 0.2*random_n_unit_vec(3)
    bias_rw = 0.2*random_n_unit_vec(3)
    h_rw = 1.0*random_n_unit_vec(3)

    std_mtq = np.abs(0.01*random_n_unit_vec(3))
    std_magic = np.abs(0.03*random_n_unit_vec(3))
    std_rw = np.abs(0.02*random_n_unit_vec(3))
    msns = np.abs(0.001*random_n_unit_vec(3))
    B_ECI = random_n_unit_vec(3)

    bias_mtm = 0.3*random_n_unit_vec(3)
    bias_gyro = 0.1*random_n_unit_vec(3)
    bias_sun = 30*random_n_unit_vec(9)
    bias_gps = np.concatenate([random_n_unit_vec(3)*60,random_n_unit_vec(3)*1])
    sun_eff = 0.3

    std_mtm = np.abs(0.01*random_n_unit_vec(3))
    std_gyro = np.abs(0.01*random_n_unit_vec(3))
    # std_sun = np.abs(1*random_n_unit_vec(6))
    std_sun = np.abs(2*random_n_unit_vec(9))
    std_gps = np.diagflat(np.concatenate([np.abs(random_n_unit_vec(3)*3),np.abs(random_n_unit_vec(3)*1)]))

    mtqs = [MTQ(j,np.dot(std_mtq,j),1,has_bias = True, bias = np.dot(bias_mtq,j),use_noise=True,bias_std_rate=0) for j in unitvecs]
    magics = [Magic(j,np.dot(std_magic,j),1,has_bias = True, bias = np.dot(bias_magic,j),use_noise=True,bias_std_rate=0) for j in unitvecs]
    rws = [RW(j,np.dot(std_rw,j),1,0.1,np.dot(h_rw,j),2,np.dot(msns,j),has_bias = True, bias = np.dot(bias_rw,j),use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtms = [MTM(j,np.dot(std_mtm,j),has_bias = True,bias = np.dot(bias_mtm,j),use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,np.dot(std_gyro,j),has_bias = True,bias = np.dot(bias_gyro,j),use_noise = True,bias_std_rate = 0) for j in unitvecs]
    suns1 = [SunSensor(j,np.dot(std_sun[0:3],j),sun_eff,has_bias = True,bias = np.dot(bias_sun[0:3],j),use_noise = False,bias_std_rate = 0) for j in unitvecs]
    suns2 = [SunSensor(-j,np.dot(std_sun[3:6],j),sun_eff,has_bias = True,bias = np.dot(bias_sun[3:6],j),use_noise = True,bias_std_rate = 0) for j in unitvecs]
    suns3 = [SunSensorPair(j,np.dot(std_sun[6:9],j),sun_eff,has_bias = True,bias = np.dot(bias_sun[6:],j),use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gps = GPS(std_gps,has_bias = True,bias = bias_gps,use_noise = True,bias_std_rate = 0)

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0
    H_body = J_body@w0 + h_rw
    H_ECI = J_ECI@w_ECI + Rm@h_rw
    acts = mtqs+magics+rws
    sensors = mtms+gyros+suns1+suns2+suns3+[gps]
    u = 5*random_n_unit_vec(9)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(6900,7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,15)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    os = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,S=S_ECI)
    R = R_ECI
    V = V_ECI
    B = B_ECI
    S = S_ECI
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R),"os":os}

    sat = Satellite(J = J_body,actuators = acts,sensors=sensors)
    state = np.concatenate([w0,q0,h_rw])
    exp_wd = -np.linalg.inv(sat.J_noRW)@np.cross(w0,H_body) + np.linalg.inv(sat.J_noRW)@sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9)],np.zeros(3))
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    exp_hd = sum([acts[j].torque(u[j],sat,state,vecs) for j in range(9) if not acts[j].has_momentum],np.zeros(3)) - sat.J@exp_wd - np.cross(w0,H_body)
    xd = sat.dynamics(state,u,os)
    np.set_printoptions(precision=3)
    assert np.allclose(np.concatenate([exp_wd,exp_qd,exp_hd]),xd)

    # sensed = sat.sensor_values(state,vecs)
    # assert np.allclose(sensed,np.concatenate([[j.reading(state,vecs)] for j in sensors[:-1]]+[h_rw]))
    # assert np.allclose(sensed,np.concatenate([[j.no_noise_reading(state,vecs)] for j in sensors[:-1]]+[h_rw]))
    # assert np.allclose(sensed,np.concatenate([[j.clean_reading(state,vecs) + j.bias] for j in sensors[:-1]]+[h_rw]))
    # assert np.allclose(sensed-np.concatenate([[j.clean_reading(state,vecs) ]for j in sensors[:-1]]+[h_rw]),np.concatenate([bias_mtm,bias_gyro,bias_sun,np.zeros(3)]))
    #
    # for k in range(10):
    #     which = [np.random.uniform(0,1)<0.5 for j in range(12)]
    #     assert np.allclose(sat.sensor_values(state,vecs,which),np.concatenate([[sensors[j].reading(state,vecs)] for j in range(len(sensors[:-1])) if which[j]]+[h_rw]))

    state_jac,bias_jac = sat.sensor_state_jacobian(state,vecs)
    assert np.allclose(state_jac,np.block([[j.basestate_jac(state,vecs) for j in sensors[:-1]]+[np.zeros((7,3))],[np.zeros((3,15)),np.eye(3)]]))
    assert np.allclose(bias_jac,np.hstack([np.eye(15),np.zeros((15,3))]))

    scov = sat.sensor_cov()
    assert np.allclose(scov,np.diagflat(np.concatenate([std_mtm**2.0,std_gyro**2.0,std_sun**2.0,msns**2.0])))
    assert np.allclose(scov,block_diag(*([sensors[j].cov() for j in range(len(sensors[:-1]))]+[j.momentum_measurement_cov() for j in acts])))
    acov = sat.control_cov()
    assert np.allclose(acov,np.diagflat(np.concatenate([std_mtq**2.0,std_magic**2.0,std_rw**2.0])))
    assert np.allclose(acov,block_diag(*[acts[j].control_cov() for j in range(len(acts))]))


    for k in range(10):
        which = [np.random.uniform(0,1)<0.5 for j in range(15)]
        state_jacw,bias_jacw = sat.sensor_state_jacobian(state,vecs,which)
        assert np.allclose(state_jacw,np.block([[sensors[j].basestate_jac(state,vecs) for j in range(len(sensors[:-1])) if which[j]]+[np.zeros((7,3))],[np.zeros((3,sum(which))),np.eye(3)]]))
        # assert np.allclose(state_jacw,np.vstack([np.hstack([sensors[j].basestate_jac(state,vecs) for j in range(len(sensors[:-1])) if which[j]]+[np.atleast_2d(h_rw)]),np.zeros((3,12))]))
        assert np.allclose(bias_jacw,np.hstack([np.eye(sum(which)),np.zeros((sum(which),3))]))
        scovw = sat.sensor_cov(which)
        assert np.allclose(scovw,block_diag(*[sensors[j].cov() for j in range(len(sensors[:-1])) if which[j]],np.diagflat(msns**2.0)))


    N = 1000
    exp_read = np.concatenate([[j.no_noise_reading(state,vecs)] for j in sensors[:-1]]+[np.atleast_2d(h_rw).T]).flatten()
    sens_err = [sat.sensor_values(state,vecs)-exp_read for j in range(N)]
    sens_noise = [np.random.normal(0,np.concatenate([std_mtm,std_gyro,std_sun[3:9]])) for j in range(N)]
    sens_noise = [np.concatenate([j[0:6],np.maximum(0,j[6:9]+exp_read[9:12])-exp_read[9:12],j[9:12]]) for j in sens_noise]
    rwsens_noise = [np.random.normal(0,msns) for j in range(N)]
    # breakpoint()
    assert np.all([np.all(j[6:9]==0) for j in sens_err])

    data_av = [np.concatenate([j[0:6],j[9:18]]) for j in sens_err]
    data_bv = [np.concatenate([sens_noise[j],rwsens_noise[j]]) for j in range(len(sens_noise))]

    ksvec = [kstest([j[i] for j in data_av],[j[i] for j in data_bv]) for i in range(15)]
    for j in range(15):
        print(j)
        ind = j
        data_a = data_av
        data_b =data_bv
        hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
        graph_data = [hist_a,hist_b]
        print(plot(graph_data,{'height':20}))
        print([k[j] for k in data_av])
        assert ksvec[j].pvalue>0.1 or np.abs(ksvec[j].statistic)<(np.sqrt((1/N)*-0.5*np.log(1e-5/2/(N+1))))
