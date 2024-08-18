# satellite testing
# from satellite import Satellite
import numpy as np
from .chain_helpers import *
from .frames_helpers import *
from .helpers import *
from .plot_helpers import *
from .quat_helpers import *
from numpy.testing import assert_allclose
import math
# from orbital_state import Orbital_State
# from orbit import Orbit
import numdifftools as nd

"""
many tests to add

#estimated stuff!
#pull from square matrices, mask, sections, refill


# pinv_LIcols
# pinv_LIrows
# saturate
# rot_mat
# rot_mat_list
# rot_list
# stateAdd
# Wmat
# rot_z
# trig_sum_solve_or_min
# quaternion_trig_combo
# quat_power
# rot_exp
# quat_log
# quat_ang_diff_rad
# quat_ang_diff_deg
# quat_to_vec3
# quat_to_vec3_deriv
# quat_to_vec3_deriv2

# dimcheck1D
# reverse_clip_neps
# GovernorMode
# PointingGoalVectorMode
# pointing_goal_vec_finder_times
# two_vec_to_quat
"""



def test_limit():
    assert np.all(limit(np.array([1,2,3]),2) == np.array([2/3,4/3,2]))
    assert np.all(limit(np.array([1,2,2]),2) == np.array([1,2,2]))
    assert np.all(limit(-5,2) == -2)

def test_normalize():
    assert np.all(normalize(np.array([1,2,3])) == np.array([1/math.sqrt(14),2/math.sqrt(14),3/math.sqrt(14)]))

def test_norm():
    vs = [[1,2,3,0],[1,2,3],[1,-2,3],np.array([1,2,3]),np.array([[1,2,3]]),np.array([[1,2,3,0]]),np.array([[1,2,3,0]]).T,np.array([[1],[2],[3]]),np.matrix([1,2,3])]
    assert np.all([norm(v) == math.sqrt(14) for v in vs])

def test_matrix_row_normalize():
    vs = np.array([[1,2,3.7,0],[5,5,32,-10],[100,0,0,0]])
    assert np.all(matrix_row_normalize(vs) == np.array([[1/math.sqrt(5+3.7**2),2/math.sqrt(5+3.7**2),3.7/math.sqrt(5+3.7**2),0],[5/math.sqrt(150+32**2),5/math.sqrt(150+32**2),32/math.sqrt(150+32**2),-10/math.sqrt(150+32**2)],[1,0,0,0]]))


def test_matrix_row_norm():
    vs = np.array([[1,2,3.7,0],[5,5,32,-10],[100,0,0,0]])
    assert np.all(matrix_row_norm(vs) == np.array([math.sqrt(5+3.7**2),math.sqrt(150+32**2),100]))


def test_rot_mat():
    #TODO
    pass

def test_drotmatTvecdq():
    v = random_n_unit_vec(3)
    q = random_n_unit_vec(4)
    v = unitvecs[1]
    q = np.concatenate([[1],unitvecs[1]])/np.sqrt(2)
    fun = lambda c: rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@v

    Jfun = nd.Jacobian(fun)(q.flatten().tolist())
    Jfuntest = np.array(Jfun)
    print(Jfuntest)
    print(drotmatTvecdq(q,v))
    assert np.allclose(Jfuntest.T,drotmatTvecdq(q,v))

def test_ddrotmatTvecdqdq():
    v = random_n_unit_vec(3)
    q = random_n_unit_vec(4)
    v = unitvecs[1]
    q = np.concatenate([[1],unitvecs[1]])/np.sqrt(2)
    print(q)
    fun = lambda c: rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@v

    Jfun = nd.Jacobian(fun)(q.flatten().tolist())
    for i in range(3):
        print(i)
        ev = np.zeros(3)
        ev[i] = 1
        fun = lambda c: ev@rot_mat(np.array([c[0],c[1],c[2],c[3]])).T@v
        Hfun = nd.Hessian(fun)(q.flatten().tolist())
        Hfuntest = np.array(Hfun)
        # print(ddrotmatTvecdqdq(q,v))
        print(fun(q))
        print(Hfuntest)
        print(ddrotmatTvecdqdq(q,v)@ev)
        # print(Hfuntest-ddrotmatTvecdqdq(q,v)@ev)
        print('hi')
        print(Hfuntest)
        print(ddrotmatTvecdqdq(q,v))
        assert np.allclose(Hfuntest,ddrotmatTvecdqdq(q,v)@ev)

def test__ddmrp__dqdq():
    q = random_n_unit_vec(4)
    fun = lambda c: quat_to_mrp(np.array([c[0],c[1],c[2],c[3]]))

    Jfun = nd.Jacobian(fun)(q.flatten().tolist())
    Jfuntest = np.array(Jfun)
    for i in range(3):
        ev = np.zeros(3)
        ev[i] = 1
        fun = lambda c: ev.T@quat_to_mrp(np.array([c[0],c[1],c[2],c[3]]))
        Hfun = nd.Hessian(fun)(q.flatten().tolist())
        Hfuntest = np.array(Hfun)
        print(i)
        print(Hfuntest)
        print(ddmrp__dqdq(q)@ev)
        print(ddmrp__dqdq(q))
        assert np.allclose(Hfuntest,ddmrp__dqdq(q)@ev)


def test__dmrp__dq():
    q = random_n_unit_vec(4)
    fun = lambda c: quat_to_mrp(np.array([c[0],c[1],c[2],c[3]]))

    Jfun = nd.Jacobian(fun)(q.flatten().tolist())
    Jfuntest = np.array(Jfun)
    assert np.allclose(Jfuntest.T,dmrp__dq(q))


def test__ddcly__dqdq():
    q = random_n_unit_vec(4)
    fun = lambda c: quat_to_cayley(np.array([c[0],c[1],c[2],c[3]]))

    Jfun = nd.Jacobian(fun)(q.flatten().tolist())
    Jfuntest = np.array(Jfun)
    for i in range(3):
        ev = np.zeros(3)
        ev[i] = 1
        fun = lambda c: ev.T@quat_to_cayley(np.array([c[0],c[1],c[2],c[3]]))
        Hfun = nd.Hessian(fun)(q.flatten().tolist())
        Hfuntest = np.array(Hfun)
        print(i)
        print(Hfuntest)
        print(ddcly__dqdq(q)@ev)
        print(ddcly__dqdq(q))
        assert np.allclose(Hfuntest,ddcly__dqdq(q)@ev)


def test__dcly__dq():
    q = random_n_unit_vec(4)
    fun = lambda c: quat_to_cayley(np.array([c[0],c[1],c[2],c[3]]))

    Jfun = nd.Jacobian(fun)(q.flatten().tolist())
    Jfuntest = np.array(Jfun)
    assert np.allclose(Jfuntest.T,dcly__dq(q))

def test_Wmat():
    #TODO
    pass

def test_quat_mult():
    pass

def test_rotz():
    #TODO
    pass

def test_state_norm_jac():
    for k in range(1):
        q0 = np.array([np.random.normal() for j in range(4)])
        dmag = 1e-4
        dq = np.array([dmag*np.random.normal() for j in range(4)])
        q1 = q0+dq
        x0 = np.zeros(7)
        x0[3:7] = q0
        dx = np.zeros(7)
        dx[3:7] = dq
        x1 = np.zeros(7)
        x1[3:7] = q1
        f0 = np.zeros(7)
        f0[3:7] = normalize(q0)
        f1 = np.zeros(7)
        f1[3:7] = normalize(q1)
        dfdx = state_norm_jac(x0)
        df = dfdx@dx


        print('k',k)
        usenumdiffonly = True
        if not usenumdiffonly:
            err0 = f1-f0
            err1 = f1-f0-df

            assert np.all(np.abs(err1)<=np.abs(err0))
            #TODO improve this check
            assert np.allclose(err1*0,err1,rtol = dmag, atol = dmag**2)
        fun = lambda c: np.hstack([c[0:3],normalize(c[3:7]).flatten()])
        Jfun = nd.Jacobian(fun)(x0.flatten())
        print(Jfun)
        print(dfdx)
        print(np.abs(Jfun-dfdx))
        assert np.allclose(Jfun,dfdx)

def test_state_norm_hess():
    q0 = np.array([np.random.normal() for j in range(4)])
    dmag = 1e-4
    dq = np.array([dmag*np.random.normal() for j in range(4)])
    q1 = q0+dq
    x0 = np.zeros(7)
    x0[3:7] = q0
    dx = np.zeros(7)
    dx[3:7] = dq
    x1 = np.zeros(7)
    x1[3:7] = q1

    f0 = state_norm_jac(x0)
    f1 = state_norm_jac(x1)
    dfdx = state_norm_hess(x0)
    df = dfdx@dx#multi_matrix_chain_rule_scalar([dfdx],[dx],1,[0])

    err0 = f1-f0
    err1 = f1-f0-df
    # assert np.all(np.abs(err1)<=np.abs(err0))
    #TODO improve this check
    # assert np.allclose(err1*0,err1,rtol = dmag, atol = dmag**2)


    q0 = np.array([np.random.normal() for j in range(4)])
    dmag = 1e-4
    dq = np.array([dmag*np.random.normal() for j in range(4)])
    q1 = q0+dq
    x0 = np.zeros(7)
    x0[3:7] = q0
    dx = np.zeros(7)
    dx[3:7] = dq
    x1 = np.zeros(7)
    x1[3:7] = q1
    f0 = np.zeros(7)
    f0[3:7] = normalize(q0)
    f1 = np.zeros(7)
    f1[3:7] = normalize(q1)
    dfdx = state_norm_jac(x0)
    df = dfdx@dx
    ddfdxdx = state_norm_hess(x0)
    ddf = 0.5*dx@(dx@ddfdxdx)#multi_matrix_chain_rule_scalar([ddfdxdx],[dx],1,[0])@dx

    err0 = f1-f0
    err1 = f1-f0-df
    err2 = f1-f0-df-ddf
    # assert np.all(np.abs(err2)<=np.abs(err1))
    # #TODO improve this check
    # assert np.allclose(err2*0,err2,rtol = dmag**2, atol = dmag**3)
    for i in range(7):
        ev = np.zeros(7)
        ev[i] = 1
        fun = lambda c: (ev.T@np.hstack([np.array([c[0],c[1],c[2]]),normalize(np.array([c[3],c[4],c[5],c[6]]))])).item()
        jfun = nd.Hessian(fun)(x0.flatten().tolist())
        assert np.allclose(jfun,np.vstack([j[i] for j in ddfdxdx]))




def test_normed_vec_hess():
    v0 = np.array([np.random.normal() for j in range(3)])
    for i in unitvecs:
        fun = lambda c: np.dot(i,normalize(np.array([c[0],c[1],c[2]]))).item()
        hfun = nd.Hessian(fun)(v0.flatten().tolist())
        assert np.allclose(hfun,np.dot(normed_vec_hess(v0),i))


def test_normed_vec_jac():
    v0 = np.array([np.random.normal() for j in range(3)])
    fun = lambda c: normalize(np.array([c[0],c[1],c[2]]))
    jfun = nd.Jacobian(fun)(v0.flatten().tolist())
    assert np.allclose(jfun,normed_vec_jac(v0))


def test_vec_norm_jac():
    v0 = np.array([np.random.normal() for j in range(3)])
    fun = lambda c: norm(np.array([c[0],c[1],c[2]]))
    jfun = nd.Jacobian(fun)(v0.flatten().tolist())
    assert np.allclose(jfun,vec_norm_jac(v0))


def test_vec_norm_hess():
    v0 = np.array([np.random.normal() for j in range(3)])
    fun = lambda c: norm(np.array([c[0],c[1],c[2]]))
    hfun = nd.Hessian(fun)(v0.flatten().tolist())
    assert np.allclose(hfun,vec_norm_hess(v0))

def test_mv_chain_rule():
    N = 5
    i = 3
    j = 10
    k = 4

    dfdg = [np.array([[np.random.normal() for jj in range(j)] for ii in range(i)]) for n in range(N)]
    dgdu = [np.array([[np.random.normal() for kk in range(k)] for jj in range(j)]) for n in range(N)]

    assert np.allclose(sum([dfdg[l]@dgdu[l] for l in range(N)]),multi_vector_chain_rule(dfdg,dgdu,N),rtol = 1e-8,atol=1e-8)


def test_skewsym2():
    vs = np.array([1,2,3])#,np.array([[1,2,3]]),np.array([[1],[2],[3]]),np.matrix([1,2,3])]
    assert np.all(skewsym(vs) == np.array([[0,-3,2],[3,0,-1],[-2,1,0]]))

def test_slerp():
    # Test case 1
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([0, 1, 0, 0])
    t = 0.5
    expected_result = np.array([0.70710678, 0.70710678, 0, 0])
    result = slerp(q1, q2, t)
    assert np.allclose(result, expected_result)

    # Test case 2
    q1 = np.array([0.70710678, 0.70710678, 0, 0])
    q2 = np.array([0, 0.70710678, 0.70710678, 0])
    t = 0.25
    expected_result = np.array([0.65328148, 0.65328148, 0.27059805, 0.27059805])
    result = slerp(q1, q2, t)
    assert np.allclose(result, expected_result)

    # Test case 3
    q1 = np.array([0.70710678, 0.70710678, 0, 0])
    q2 = np.array([0, 0.70710678, 0.70710678, 0])
    t = 0.75
    expected_result = np.array([-0.27059805, -0.27059805, 0.65328148, 0.65328148])
    result = slerp(q1, q2, t)
    assert np.allclose(result,expected_result)

def test_quat_mult():
    # Test case 1
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([0, 1, 0, 0])
    expected_result = np.array([0, 1, 0, 0])
    result = quat_mult(q1, q2)
    assert np.allclose(result, expected_result)

    # Test case 2
    q1 = np.array([0.70710678, 0, 0.70710678, 0])
    q2 = np.array([0, 0.70710678, 0, 0.70710678])
    expected_result = np.array([0, 1.0, 0, 0])
    result = quat_mult(q1, q2)
    assert np.allclose(result, expected_result)

    # Test case 3
    q1 = np.array([0.70710678, 0, 0.70710678, 0])
    q2 = np.array([0, 0, 0.70710678, 0.70710678])
    expected_result = np.array([-0.5, 0.5, 0.5, 0.5])
    result = quat_mult(q1, q2)
    assert np.allclose(result, expected_result)


def test_skewsym():
    # Test case 1
    v = np.array([1, 2, 3])
    expected_result = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    result = skewsym(v)
    assert np.allclose(result, expected_result)

    # Test case 2
    v = np.array([0, 0, 0])
    expected_result = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = skewsym(v)
    assert np.allclose(result, expected_result)

    # Test case 3
    v = np.array([1, 0, 0])
    expected_result = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    result = skewsym(v)
    assert np.allclose(result, expected_result)

def test_random_n_unit_vec():
    # Test case 1
    n = 3
    result = random_n_unit_vec(n)
    assert np.isclose(np.linalg.norm(result), 1.0)

    # Test case 2
    n = 4
    result = random_n_unit_vec(n)
    assert np.isclose(np.linalg.norm(result), 1.0)

    # Test case 3
    n = 5
    result = random_n_unit_vec(n)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_quat_inv():
    # Test case 1
    q = np.array([1, 2, 3, 4])/np.sqrt(30)
    expected_result = np.array([ 1,-2,-3,-4])/np.sqrt(30)
    result = quat_inv(q)
    assert np.allclose(result, expected_result)

    # Test case 2
    q = np.array([0, 0, 0, 1])
    expected_result = np.array([0, 0, 0, -1])
    result = quat_inv(q)
    assert np.allclose(result, expected_result)

    # Test case 3
    q = np.array([1, 0, 0, 0])
    expected_result = np.array([1, 0, 0, 0])
    result = quat_inv(q)
    assert np.allclose(result, expected_result)


def test_axang2quat():
    # Test case 1
    axis = np.array([1, 0, 0])
    angle = np.pi/2
    expected_result = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
    result = axang2quat(angle, axis)
    assert np.allclose(result, expected_result)

    # Test case 2
    axis = np.array([0, 1, 0])
    angle = np.pi/4
    expected_result = np.array([np.cos(np.pi/8.0), 0, np.sin(np.pi/8.0), 0])
    result = axang2quat(angle, axis)
    assert np.allclose(result, expected_result)

    # Test case 3
    axis = np.array([0, 0, 1])
    angle = np.pi/3
    expected_result = np.array([np.sqrt(3)/2,0,0,0.5])
    result = axang2quat(angle, axis)
    assert np.allclose(result, expected_result)


def test_quat_to_cayley():
    q = random_n_unit_vec(4)
    res = quat_to_cayley(q)
    assert np.allclose(res,q[1:]/q[0])


def test_cayley_to_quat():
    for j in range(4):
        q = random_n_unit_vec(4)
        v = quat_to_cayley(q)
        res = cayley_to_quat(v)
        q = q*np.sign(q[0])
        assert np.allclose(res,q)


def test_quat_to_mrp():
    for j in range(4):
        q = random_n_unit_vec(4)
        res = quat_to_mrp(q)
        assert np.allclose(res,2*q[1:]/(1+q[0]))


def test_mrp_to_quat():
    for j in range(4):
        q = random_n_unit_vec(4)
        v = quat_to_mrp(q)
        res = mrp_to_quat(v)
        assert np.allclose(res,q)



#
