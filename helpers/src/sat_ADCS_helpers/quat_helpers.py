import numpy as np
import math
import pytest
from .helpers import *


def two_vec_to_quat(r1,r2,b1,b2):
    #uses Markley's Fast quaternion atittude estimation approach
    #r is in world, b is in body
    r1 = normalize(r1)#.reshape((3,1))
    r2 = normalize(r2)#.reshape((3,1))
    b1 = normalize(b1)#.reshape((3,1))
    b2 = normalize(b2)#.reshape((3,1))
    b3 = normalize(np.cross(b1,b2))
    r3 = normalize(np.cross(r1,r2))

    b3dr3 = np.dot(b3,r3)
    br3list = [b3dr3,b3[0]*r3[0],b3[1]*r3[1],b3[2]*r3[2]]
    rotax = 0
    if b3dr3!=max(br3list):
        r10 = r1
        r20 = r2
        r30 = r3
        rotax = br3list.index(max(br3list))
        r1 = -r1 + 2*unitvecs[rotax-1]*r1[rotax-1]
        r2 = -r2 + 2*unitvecs[rotax-1]*r2[rotax-1]
        r3 = -r3 + 2*unitvecs[rotax-1]*r3[rotax-1]

    b3dr3 = np.dot(b3,r3)
    br3p1 =(1+b3dr3)
    x3 = np.cross(b3,r3)
    al = br3p1*(np.dot(b1,r1)+np.dot(b2,r2)) + np.dot(x3,np.cross(b2,r2)+np.cross(b1,r1))
    bet = np.dot(b3+r3,np.cross(b1,r1)+np.cross(b2,r2))
    gam = math.sqrt(al**2+bet**2)
    # breakpoint()
    if al<0:
        qout = np.concatenate([[bet*br3p1],bet*x3+(gam-al)*(b3+r3)])*0.5/math.sqrt(gam*(gam-al)*br3p1)
    else:
        qout = np.concatenate([[(gam+al)*br3p1],(gam+al)*x3+bet*(b3+r3)])*0.5/math.sqrt(gam*(gam+al)*br3p1)

    if rotax == 0:
        return qout
    else:
        qout = quat_mult(unitvecs4[rotax],qout)
        # breakpoint()
        return qout

def axang2quat(th,v):
    return np.concatenate([[math.cos(th/2)],normalize(v)*math.sin(th/2)])

def wahbas_svd(weightlist,bodylist,ECIlist):
    if not len(weightlist)==len(bodylist) and len(weightlist)==len(ECIlist):
        raise ValueError('wrong list lengths')
    if not all([j.size == 3 for j in bodylist]):
        raise ValueError('wrong body vector lengths')
    if not all([j.size == 3 for j in ECIlist]):
        raise ValueError('wrong ECI vector lengths')
    Bmat = sum([weightlist[j]*np.outer(bodylist[j],ECIlist[j]) for j in range(len(weightlist))],np.zeros((3,3)))
    # zvec = sum([weightlist[j]*np.cross(bodylist[j],ECIlist[j]) for j in range(len(weightlist))],np.zeros(3))
    # S = Bmat+Bmat.T
    # trB = np.trace(Bmat)
    # K = np.block([[trB,zvec],[np.atleast_2d(zvec).T,S]])
    # val,vec = np.linalg.eigh(K)
    u,s,vh = np.linalg.svd(Bmat)
    A = (u@np.diagflat([1,1,np.linalg.det(u)*np.linalg.det(vh.T)])@vh).T
    tr = np.trace(A)
    if tr>=0:
        rr = math.sqrt(1+tr)
        q0 = 0.5*rr
        ss = 0.5/rr
        qmat = ss*(A-A.T)
        qv = np.array([qmat[2,1],qmat[0,2],qmat[1,0]])
        return np.hstack([[q0],qv])
    else:
        dA = np.diag(A)
        inds = np.argsort(dA)
        rr = math.sqrt(1 - np.sum(dA) + 2*dA[inds[-1]])
        ss = 0.5/rr
        q = np.zeros(4)
        q[inds[-1]+1] = 0.5*rr
        qmat = ss*(A-A.T)
        qpmat = ss*(A+A.T)
        q[0] = qmat[inds[0],inds[1]]*(1-2*(inds[0]<inds[1])*(inds[2]!=1))
        q[inds[0]+1] = qpmat[inds[1],inds[2]]
        q[inds[1]+1] = qpmat[inds[0],inds[2]]
        breakpoint()
        return q

def quat_mult(p,q,*extra): #Multi arg not fully tested!
    # breakpoint()
    # p = p.reshape((4,1))
    # q = q.reshape((4,1))
    # print('+++++++++++++++++++++')
    # print(len(extra))
    # print(p)
    # print(q)
    # print(extra)
    if isinstance(extra,np.ndarray):
        return quat_mult(quat_mult(p,q),extra)
    elif isinstance(extra,(list,tuple)):
        if len(extra) > 1:
            if len(extra) == 4 and np.all([isinstance(j,NumberTypes) for j in extra]):
                return quat_mult(quat_mult(p,q),extra[0])
            elif np.all([len(j) == 4 for j in extra]):
                return quat_mult(quat_mult(p,q),extra[0],*extra[1:])
            else:
                raise ValueError('these do not all appear to be quaternions')
        elif len(extra) == 1:
            return quat_mult(quat_mult(p,q),extra[0])
    p0 = p[0]#).item()
    q0 = q[0]#).item()
    pv = p[1:]#.reshape((3,1))
    qv = q[1:]#.reshape((3,1))
    # print(np.concatenate([[p0*q0-np.dot(pv,qv)],p0*qv + q0*pv + np.cross(pv,qv)]))
    # print('=======================')
    return np.concatenate([[p0*q0-np.dot(pv,qv)],p0*qv + q0*pv + np.cross(pv,qv)])

def quat_ang_diff_rad(p,q):
    return math.acos(np.clip(2.0*np.dot(normalize(p),normalize(q))**2.0-1.0,-1,1))

def quat_ang_diff_deg(p,q):
    return quat_ang_diff_rad(p,q)*180.0/math.pi

def quat_inv(q):
    # q = q.reshape((4,1))
    q0 = q[0]
    qv = q[-3:]#.reshape((3,1))
    nq = norm(q)
    if nq > num_eps:
        return np.concatenate([[q0],-qv])/(nq**2.0)#.reshape((4,1))
    # warnings.warn('invalid q')
    # print(q)
    # breakpoint()
    return np.concatenate([[q0],-qv])

def quat_to_vec3(quat,mode):
    # quat = np.copy(quat).reshape((4,1))
    if mode == 6:# --  2xMRP with scalar component of error quat positive
        if quat[0]>0.0:
            quat *= np.sign(quat[0])
        return 2*quat_to_mrp(quat)
    if mode == 5: # --  2xMRP
        return 2*quat_to_mrp(quat)
    if mode == 4: # --  vector component,
        return quat[1:]
    if mode == 3: # --  vector component,with scalar component of error quat positive
        if quat[0]>0.0:
            quat *= np.sign(quat[0])
        return quat[1:]
    if mode == 2: # --  Cayley
        return quat_to_cayley(quat)
    elif mode == 1: # --  MRP
        return quat_to_mrp(quat)
    else: #mode is 0 or something wrong --  MRP with scalar component of error quat positive
        if np.abs(quat[0])>num_eps:
            quat *= np.sign(quat[0])
        return quat_to_mrp(quat)


def vec3_to_quat(v3,mode):
    if mode == 6:
        q = mrp_to_quat(v3/2.0)
        sq = np.sign(q[0])
        if np.abs(sq)>0.0:
            q *= sq
        return q
    if mode == 5:
        return mrp_to_quat(v3/2.0)
    if mode == 4:
        return np.concatenate([[math.sqrt(1.0-norm(v3)**2.0)],v3])
    if mode == 3:
        return np.concatenate([[math.sqrt(1.0-norm(v3)**2.0)],v3])
    if mode == 2:
        return cayley_to_quat(v3)
    elif mode == 1:
        return mrp_to_quat(v3)
    else:
        q = mrp_to_quat(v3)
        sq = np.sign(q[0])
        if np.abs(sq)>0.0:
            q *= sq
        return q

def quat_to_vec3_deriv(quat,mode):
    # quat = quat.reshape((4,1))
    if mode == 6:
        sq = 1
        if quat[0]>0.0:
            sq = np.sign(quat[0])
        return 2*dmrp__dq(quat*sq)*sq
    if mode == 5:
        return 2*dmrp__dq(quat)
    if mode == 4:
        res = np.zeros((4,3))
        res[1:,:] = np.eye(3)
        return res
    if mode == 3:
        res = np.zeros((4,3))
        res[1:,:] = np.eye(3)
        if quat[0]>0.0:
            res *= np.sign(quat[0])
        return res
    if mode == 2:
        return dcly__dq(quat)
    elif mode == 1:
        return dmrp__dq(quat)
    else:
        sq = 1
        if quat[0]>0.0:
            sq = np.sign(quat[0])
        return dmrp__dq(quat*sq)*sq

def quat_to_vec3_deriv2(quat,mode):
    # quat = quat.reshape((4,1))
    if mode == 6:
        sq = 1
        if quat[0]>0.0:
            sq = np.sign(quat[0])
        return 2*ddmrp__dqdq(quat*sq)
    if mode == 5:
        return 2*ddmrp__dqdq(quat)
    if mode == 4:
        return np.zeros((4,4,3))
    if mode == 3:
        return np.zeros((4,4,3))
    if mode == 2:
        return ddcly__dqdq(quat)
    elif mode == 1:
        return ddmrp__dqdq(quat)
    else:
        return ddmrp__dqdq(quat*np.sign(quat[0]))



def vec3_to_quat_deriv2(v3,mode):
    # quat = quat.reshape((4,1))
    if mode == 6:
        return ddq__dmrpdmpr(v3/2.0)*0.5*0.5
    if mode == 5:
        return ddq__dmrpdmrp(v3/2.0)*0.5*0.5
    if mode == 4:
        res = np.zeros((3,3,4))
        q0 = np.sqrt(1-norm(v3)**2.0)
        res[:,:,0] = -(np.eye(3)/q0) - np.outer(v3,v3)/q0**3.0
        return res
    if mode == 3:
        res = np.zeros((3,3,4))
        q0 = np.sqrt(1-norm(v3)**2.0)
        res[:,:,0] = -(np.eye(3)/q0) - np.outer(v3,v3)/q0**3.0
        return res
    if mode == 2:
        return ddq__dclydcly(v3)
    elif mode == 1:
        return ddq__dmrpdmrp(v3)
    else:
        dq = ddq__dmrpdmrp(v3)
        return dq

def vec3_to_quat_deriv(v3,mode):
    # quat = quat.reshape((4,1))
    if mode == 6:
        dq = dq__dmrp(v3/2.0)*0.5
        q = mrp_to_quat(v3/2.0)
        sq = np.sign(q[0])
        if np.abs(sq)>0.0:
            dq *= sq
        return dq
    if mode == 5:
        return dq__dmrp(v3/2.0)*0.5
    if mode == 4:
        return np.vstack([-v3/np.sqrt(1-norm(v3)**2.0),np.eye(3)])
    if mode == 3:
        return np.vstack([-v3/np.sqrt(1-norm(v3)**2.0),np.eye(3)])
    if mode == 2:
        return dq__dcly(v3)
    elif mode == 1:
        return dq__dmrp(v3)
    else:
        dq = dq__dmrp(v3)
        q = mrp_to_quat(v3)
        sq = np.sign(q[0])
        if np.abs(sq)>0.0:
            dq *= sq
        return dq

def mrp_to_quat(mrp):
    #https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    # return (1/np.sqrt(1+norm(mrp)**2))*np.vstack([np.array([1]),mrp]).reshape((4,1))
    thetad2 = 2*math.atan(norm(mrp)*0.5)
    nhat = normalize(mrp)
    costd2 = math.cos(thetad2)
    return np.concatenate([[costd2],nhat*np.abs(math.sin(thetad2))])#.reshape((4,1))

def quat_to_mrp(quat):
    # quat = quat.reshape((4,1))
    #https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    # return quat[1:,:].reshape((3,1))/quat[0,0]
    # quat *= np.sign(quat[0,0])
    return 2*quat[1:]/(1+quat[0])#.reshape((3,1))

def dmrp__dq(quat):
    # sq = np.sign(np.copy(quat[0,0]))
    # quat *= sq
    return np.vstack([-quat_to_mrp(quat),2*np.eye(3)])/(1+quat[0])

def ddmrp__dqdq(quat):
    # sq = np.sign(np.copy(quat[0,0]))
    # quat *= sq
    res = np.zeros((4,4,3))
    res[0,0,:] = quat_to_mrp(quat)
    res[0,1:,:] = -np.eye(3)
    res[1:,0,:] = -np.eye(3)
    res = 2*res/(1+quat[0])**2
    return res# [2*np.hstack([quat_to_mrp(quat),-np.eye(3)])/(1+quat[0])**2] + [np.hstack([-2*j,np.zeros((3,3))])/(1+(quat[0]))**2 for j in unitvecs]


def dq__dmrp(mrp):
    thetad2 = 2*math.atan(norm(mrp)*0.5)
    dthetad2__dmrp = (1/(1+(norm(mrp)*0.5)**2.0))*vec_norm_jac(mrp)
    nhat = normalize(mrp)
    dnat__dmrp = normed_vec_jac(mrp)
    costd2 = math.cos(thetad2)
    return np.hstack([-math.sin(thetad2)*np.expand_dims(dthetad2__dmrp,1),dnat__dmrp*np.abs(math.sin(thetad2)) + np.sign(math.sin(thetad2))*costd2*np.outer(dthetad2__dmrp,nhat)])


def ddq__dmrpdmrp(mrp):
    thetad2 = 2*math.atan(norm(mrp)*0.5)
    dthetad2__dmrp = (1/(1+(norm(mrp)*0.5)**2.0))*vec_norm_jac(mrp)
    ddthetad2__dmrpdmrp
    nhat = normalize(mrp)
    dnat__dmrp = normed_vec_jac(mrp)
    ddnat__dmrpdmrp = normed_vec_hess(mrp)
    costd2 = math.cos(thetad2)
    res = np.zeros((3,3,4))
    res[:,:,0] = -math.sin(thetad2)*ddthetad2__dmrpdmrp - math.cos(thetad2)*np.outer(dthetad2__dmrp,dthetad2__dmrp)
    res[:,:,1:] += ddnat__dmrpdmrp
    #TODO
    # res[:,:,1:] += np.tensordot()
    #
    # , np.sign(math.sin(thetad2))*costd2*np.outer(dthetad2__dmrp,nhat)])

    return
# d/dq0 2qv/(1+q0) = -2*qv/(1+q0)^2 = -mrp/(1+q0)
# d/dqv 2qv/(1+q0) = 2*I/(1+q0)
#
# d/dq0 -2*qv/(1+q0)^2 = d/dq0 -mrp/(1+q0) = -((1+q0)*dmrpdq0 - 1*mrp)/(1+q0)^2 = (-(1+q0)*-mrp/(1+q0) + mrp)/(1+q0)^2 = (2*mrp)/(1+q0)^2
# d/dq0  2*I/(1+q0) = 2*I*-1/(1+q0)^2
#
# d/dqi -2*qv/(1+q0)^2 = d/dqv -mrp/(1+q0) = (-2/(1+q0)^2)*ei
# d/dqi  2*I/(1+q0) = 0

def cayley_to_quat(cly):
    #https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    # return (1/np.sqrt(1+norm(mrp)**2))*np.vstack([np.array([1]),mrp]).reshape((4,1))
    return np.concatenate([[1],cly])/np.sqrt(1+norm(cly)**2)

def quat_to_cayley(quat):
    #https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    # return quat[1:,:].reshape((3,1))/quat[0,0]
    # quat *= np.sign(quat[0,0]
    # quat = np.copy(quat)
    if abs(quat[0])<num_eps:
        quat[0] = num_eps*np.sign(quat[0])
        quat = normalize(quat)
    return quat[1:]/quat[0]

def dcly__dq(quat):
    # sq = np.sign(np.copy(quat[0,0]))
    # quat *= sq
    # quat = np.copy(quat)
    if abs(quat[0])<num_eps:
        quat[0] = num_eps*np.sign(quat[0])
        quat = normalize(quat)
    return np.vstack([-quat_to_cayley(quat),np.eye(3)])/quat[0]

def ddcly__dqdq(quat):
    # sq = np.sign(np.copy(quat[0,0]))
    # quat *= sq
    res = np.zeros((4,4,3))
    res[0,0,:] = 2*quat_to_cayley(quat)
    res[0,1:,:] = -np.eye(3)
    res[1:,0,:] = -np.eye(3)
    res = res/quat[0]**2
    return res
    # return [np.hstack([2*quat_to_cayley(quat),-np.eye(3)])/(quat[0])**2] + [np.hstack([-j,np.zeros((3,3))])/(quat[0])**2 for j in unitvecs]

def dq__dcly(cly):
    return np.hstack([np.zeros(3),np.eye(3)])/np.sqrt(1+norm(cly)**2) + np.outer((1/np.sqrt(1+norm(cly)**2))*cly,np.concatenate([[1],cly]))


def ddq__dclydcly(cly):
    return
    #TODO
    res = np.zeros((3,3,4))
    # res[:,:,0] =
    # return np.hstack([np.zeros(3),np.eye(3)])/np.sqrt(1+norm(cly)**2) + np.outer((1/np.sqrt(1+norm(cly)**2))*cly,np.concatenate([[1],cly]))


def drotmatTvecdq(q,v):
    # q = q.reshape((4,1))
    # v = v.reshape((3,1))
    # q = normalize(q)
    qv = q[1:]
    #dRBdq = 2*np.block([[q[0,0]*B - np.cross(qv,B,axis=0), np.eye(3)*(np.dot(-qv.T,B).item()) - qv@B.T + B@qv.T-q[0,0]*skewsym(B)]])
    return 2*np.vstack([q[0]*v - np.cross(qv,v), np.eye(3)*np.dot(qv,v) - np.outer(qv,v) + np.outer(v,qv) - q[0]*skewsym(v)])

def ddrotmatTvecdqdq(q,v):
    # q = q.reshape((4,1))
    # v = v.reshape((3,1))
    qv = q[1:]
    # q = normalize(q)
    #dRBdq = 2*np.block([[q[0,0]*B - np.cross(qv,B,axis=0), np.eye(3)*(np.dot(-qv.T,B)).item() - qv@B.T + B@qv.T-q[0,0]*skewsym(B)]])
    output = np.zeros((4,4,3))
    # output = [0]*4
    output[0,:,:] = 2*np.vstack([v ,-skewsym(v)])
    output[:,0,:] = 2*np.vstack([v ,-skewsym(v)])
    tmp = 2*np.multiply.outer(np.eye(3),v)
    output[1:,1:,:] += -tmp
    output[1:,1:,:] +=  np.transpose(tmp,(2,0,1))
    output[1:,1:,:] +=  np.transpose(tmp,(1,2,0))

    # for k in range(1,4):
    #     e = unitvecs[k-1]
    #     output[:,:,k] = 2*np.hstack([ np.zeros(3), np.outer(e,v) - np.outer(v,e)])

    return output

def quat_norm_jac(q):
    out = np.eye(4)/norm(q) - np.outer(q,q)/norm(q)**3
    return out

def state_norm_jac(xk):
    l = xk.shape[0]
    q = xk[3:7]
    out = np.eye(l)
    out[3:7,3:7] = quat_norm_jac(q)#np.eye(4)/norm(q) - np.outer(q,q)/norm(q)**3
    return out

def quat_norm_hess(q):
    qn = norm(q)
    jac = np.eye(4)/qn - np.outer(q,q)/qn**3
    tmp = -np.multiply.outer(jac,q/qn**2)
    # res = np.zeros((4,4,4))

    return tmp + np.transpose(tmp,(2,0,1)) + np.transpose(tmp,(1,2,0))

    # -(1/qn**3)*(unitvecs4[j]@q.T + q@unitvecs4[j].T + q[j]*np.eye(4)) + (3/qn**5)*q[j]*q@q.T
    #
    # return [-(1/qn**3)*(unitvecs4[j]@q.T + q@unitvecs4[j].T + (q[j,:]).item()*np.eye(4)) + (3/qn**5)*(q[j,:]).item()*q@q.T for j in range(4) ]


def state_norm_hess(xk):
    xk = xk
    l = xk.shape[0]
    q = xk[3:7]
    qn = norm(q)
    jac = np.eye(4)*(1/qn) - q@q.T*(1/qn**3)
    res = np.zeros((l,l,l))
    res[3:7,3:7,3:7] = quat_norm_hess(q)
    return res

    # return [np.zeros((l,l)) for j in range(3)]+[np.block([[np.zeros((3,l))],[np.zeros((4,3)),-(1/qn**3)*(unitvecs4[j]@q.T + q@unitvecs4[j].T + (q[j,:]).item()*np.eye(4)) + (3/qn**5)*(q[j,:]).item()*q@q.T,np.zeros((4,l-7))],[np.zeros((l-7,l))]]).reshape((l,l)) for j in range(4) ]+[np.zeros((l,l)) for j in range(l-7)]



def Wmat(q):
    """
    This function finds the Wmat matrix corresponding to a quaternion q, such that dq/dt = Wmat(q)*w
    Inputs:
        q -- (4,) size np array representing a quaternion
    Ouputs:
        Wmat -- 4 x 3 size np array representing Wmat(q)
    Duplicated from sim_helpers--estimator and components on board satellite must stand on their own.
    """
    # q = q.reshape((4,1))
    W = np.zeros([4, 3])
    qv = q[1:4]
    W[0,:] = -qv
    W[1:4,:] = q[0]*np.eye(3) + skewsym(qv)
    return W

def quat_left_mult_matrix(q):
    """
    This function finds the matrix corresponding to a quaternion left multiplication,L, such that L(q) * p = qp in quaternion multiplication
    Inputs:
        q -- (4,) size np array representing a quaternion
    Ouputs:
        Lmat -- 4 x 4 size np array representing L(q)
    from "Planning with Attitude" by Jackson,Tracy,Manchester
    """
    L = np.zeros([4, 4])
    qv = q[1:4]
    L[0,1:] = -qv
    L[:,0] = q
    L[1:,1:] = q[0]*np.eye(3) + skewsym(qv)
    return L

def quat_right_mult_matrix(q):
    """
    This function finds the matrix corresponding to a quaternion right multiplication, R, such that R(q) * p = pq in quaternion multiplication
    Inputs:
        q -- (4,) size np array representing a quaternion
    Ouputs:
        Rmat -- 4 x 4 size np array representing R(q)
    from "Planning with Attitude" by Jackson,Tracy,Manchester
    """
    R = np.zeros([4, 4])
    qv = q[1:4]
    R[0,1:] = -qv
    R[:,0] = q
    R[1:,1:] = q[0]*np.eye(3) - skewsym(qv)
    return R




def rot_mat_list(qlist,veclist,transpose = False):
    if qlist.shape[-1] != 4:
        raise ValueError("incorrectly shaped qlist")
    if veclist.shape[-1] != 3:
        raise ValueError("incorrectly shaped veclist")
    if len(qlist.shape) == 1:
        N = 1
    else:
        N = qlist.shape[0]
    if N != veclist.shape[0]:
        if N == 1:
            qlist = np.repeat(qlist.reshape((1,4)),veclist.shape[0],axis=0)
        elif veclist.shape[0] == 1:
            veclist = np.repeat(veclist.reshape((1,3)),N,axis=0)
        else:
            breakpoint()
            raise ValueError("lists of different lengths, and neither is length 1")
    if transpose:
        return np.vstack([rot_mat(qlist[j,:]).T@veclist[j,:] for j in range(N)])
    else:
        return np.vstack([rot_mat(qlist[j,:])@veclist[j,:] for j in range(N)])

def rot_mat(q):
    """
    This function finds the rotation matrix for a particular quaternion. Our quaternion formulation uses the Hamilton convention, and represents a
    rotation from the body frame to ECI frame (local-to-inertial rotation). So, the rotation matrix will rotate a body vector to ECI, and its transpose
    will rotate from ECI -> body.

    This means that if you want to rotate B_ECI to body frame, you MUST transpose the output of this function.
    Inputs:
        # q -- 4 x 1 numpy array, quaternion to find rot matrix of
    Returns:
        A -- 3 x 3 np matrix to rotate a vector between frames (should rotate body -> ECI based on our quaternion model)
    Duplicated from sim_helpers--estimator and components on board satellite must stand on their own.

    """
    # q = q.reshape((4,1))
    # q = normalize(q)
    q0 = q[0]
    qv = q[1:]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    # A = (q0**2-np.dot(qv,qv))*np.eye(3) + 2*qv@qv.T + 2*q0*skewsym(qv)
    A = np.array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                    [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
                    [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]])

    return A


def slerp(q0,q1,t):
    assert t<=1
    assert t>=0
    assert q0.size == 4
    # q0 = q0.reshape((4,1))
    assert q1.size == 4
    # q1 = q1.reshape((4,1))
    return quat_mult(q0,rot_exp(t*quat_log(quat_mult(quat_inv(q0),q1))))

def quat_power(q,power):
    assert q.size == 4
    # q = q.reshape((4,1))
    qv = q[1:]#.reshape((3,1))
    qvnorm = norm(qv)
    if qvnorm == 0:
        return zeroquat
    q0 = q[0]
    phi = 2*np.arctan2(qvnorm,q0)
    u = qv/qvnorm
    return np.concatenate([[math.cos(power*phi/2)],u*math.sin(power*phi/2)])

def quaternion_trig_combo(x,y,b,keep_qs_pos=False):
    assert x.size == 4
    assert y.size == 4
    # y = y.reshape((4,1))
    # x = x.reshape((4,1))
    q = x*math.cos(b) + y*math.sin(b)
    q0 = q[0]
    if keep_qs_pos and q0!=0:
        q *= np.sign(q0)
    return q

def rot_exp(v):
    #capitalized Exponential map in Sola
    assert v.size == 3
    # v = v.reshape((3,1))
    phi = norm(v)
    if phi == 0:
        return zeroquat
    u = v/phi
    return np.concatenate([[math.cos(phi/2)],u*math.sin(phi/2)])

def quat_log(q):
    #capitalized logarithmic map in Sola
    assert q.size == 4
    # q = q.reshape((4,1))
    qv = q[1:]#.reshape((3,1))
    qvnorm = norm(qv)
    if qvnorm == 0:
        return np.zeros(3)
    q0 = q[0]#).item()
    if qvnorm<1e-5:
        return 2*(qv/q0)*(1-((qvnorm/q0)**2.0)/3.0) #not tested
    phi = 2*np.arctan2(qvnorm,q0)
    u = qv/qvnorm
    return phi*u


def os_local_vecs(os,q):
    R = os.R
    V = os.V
    B = os.B
    S = os.S
    rho = os.rho

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
    vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":os}
    return vecs


#TODO maybe make enum with r, v, sun, B to pass in here? idk
# @staticmethod
def pointing_goal_vec_finder_times(mode, vec, os,quatmode = 0):
    """
    This function takes the control mode input and any associated information and
    provides a (normalized) ECI vector direction to point to.

    Parameters
    ------------
        mode: PointingGoalVectorMode
            pointing mode
        vec: np array (3 x 1)
            for PROVIDED_ECI_VECTOR, custom ECI vector; for PROVIDED_ECI_VECTOR_POSITION,
            custom ECI vector position; for PROVIDED_ECEF_GROUND_TARGET, custom ECEF ground
            target; not used in any other mode
        j2000: float
            Julian date rel. to J2000 epoch
        r_eci: numpy array (3 x 1)
            satellite orbital position in ECI at time j2000
        v_eci: numpy array (3 x 1)
            satellite orbital velocity in ECI at time j2000
        sun_eci: numpy array (3 x 1)
            sun position vector in ECI at time j2000
    Returns
    ----------
        eci_vec: np array (3 x 1)
            normalized 2D np array, direction to align satellite body vector to
    """
    if mode == PointingGoalVectorMode.NO_GOAL:
        eci_vec = np.array([0,0,0])
    elif mode == PointingGoalVectorMode.NADIR:
        eci_vec = normalize(-os.R)
    elif mode == PointingGoalVectorMode.ZENITH:
        eci_vec = normalize(os.R)
    elif mode == PointingGoalVectorMode.RAM:
        eci_vec = normalize(os.V)
    elif mode == PointingGoalVectorMode.ANTI_RAM:
        eci_vec = normalize(-os.V)
    elif mode == PointingGoalVectorMode.POSITIVE_ORBIT_NORMAL:
        eci_vec = normalize(cross(os.R,os.V))
    elif mode == PointingGoalVectorMode.NEGATIVE_ORBIT_NORMAL:
        eci_vec = -normalize(cross(os.R,os.V))
    elif mode == PointingGoalVectorMode.TOWARDS_SUN:
        eci_vec = normalize(os.S-os.R)
    elif mode == PointingGoalVectorMode.AWAY_FROM_SUN:
        eci_vec = -normalize(os.S-os.R)
    elif mode == PointingGoalVectorMode.PROVIDED_ECI_VECTOR:
        eci_vec = normalize(vec)
    elif mode == PointingGoalVectorMode.PROVIDED_ECI_VECTOR_POSITION:
        eci_vec = normalize(vec-os.R)
    elif mode == PointingGoalVectorMode.PROVIDED_ECEF_GROUND_TARGET:
        posvec = os.ecef_to_eci(vec) #Take target to ECI
        eci_vec = normalize(posvec-os.R)
    elif mode == PointingGoalVectorMode.PROVIDED_MRP:
        eci_vec = vec
    elif mode == PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI:
        framequat = vec3_to_quat(vec,quatmode)
        eciquat = two_vec_to_quat(os.R/norm(os.R),np.cross(os.R/norm(os.R),os.V/norm(os.V)),unitvecs[2],unitvecs[0])
        eci_vec = quat_to_vec3(quat_mult(eciquat,framequat),quatmode)
        # breakpoint()
    return eci_vec
