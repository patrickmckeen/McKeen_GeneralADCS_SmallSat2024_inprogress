from .disturbance import *
# from typeguard import typechecked

class GG_Disturbance(Disturbance):
    """
    This class describes a gravity gradient disturbance torque.

    Attributes
    ------------
        params -- None

    """
    def __init__(self,params=[],estimate = False,active=True):
        super().__init__(params,False,0,False,active)

    # @typechecked
    def set_params(self,params=[]):
        super().set_params(params)

    def torque(self,sat,vecs):
        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        nadir_vec = -r_body_hat
        const_term = 3.0*(mu_e)/(norm(R_B)**3.0)
        return const_term*np.cross(nadir_vec, nadir_vec@sat.J)*self.active

    def torque_qjac(self,sat,vecs): #torque jacobian over quaternion
        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        dr_body_hat__dq = normed_vec_jac(R_B,vecs["dr"])# - np.outer(np.dot(vecs["dr"],R_B),R_B)/norm(R_B)**3.0
        nadir_vec = -r_body_hat
        dnadir_vec__dq = -dr_body_hat__dq
        const_term = 3.0*(mu_e)/(norm(R_B)**3.0)
        # if np.isclose(2*norm(R_B),norm(vecs["dr"]@r_body_hat)):
        #     return const_term*(np.cross(dnadir_vec__dq, nadir_vec@sat.J) + np.cross(nadir_vec, dnadir_vec__dq@sat.J))*self.active
        # else:
        dc__dq = -9.0*(mu_e)*(vec_norm_jac(R_B,vecs["dr"]))/(norm(R_B)**4.0)
        dv__dq = np.cross(dnadir_vec__dq, nadir_vec@sat.J) + np.cross(nadir_vec, dnadir_vec__dq@sat.J)
        vec_term = np.cross(nadir_vec, nadir_vec@sat.J)
        return (np.outer(dc__dq,vec_term) + const_term*dv__dq)*self.active

    def torque_qqhess(self,sat,vecs): #hessian of torque elements over quaternion
        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        dr_body_hat__dq = normed_vec_jac(R_B,vecs["dr"])#vecs["dr"]/norm(R_B)# - np.outer(np.dot(vecs["dr"],R_B),R_B)/norm(R_B)**3.0
        ddr_body_hat__dqdq = normed_vec_hess(R_B,vecs["dr"],vecs["ddr"])#vecs["ddr"]/norm(R_B)# - np.outer(np.dot(vecs["dr"],R_B),R_B)/norm(R_B)**3.0
        nadir_vec = -r_body_hat
        dnadir_vec__dq = -dr_body_hat__dq
        ddnadir_vec__dqdq = -ddr_body_hat__dqdq
        # -z x -10z
        const_term = 3.0*(mu_e)/(norm(R_B)**3.0)
        tmp = np.cross(np.expand_dims(dnadir_vec__dq,1),np.expand_dims(dnadir_vec__dq@sat.J,0))
        # print(dr_body_hat__dq.shape,r_body_hat.shape)
        # print(2*norm(R_B),norm(vecs["dr"]@r_body_hat))
        # if np.isclose(2*norm(R_B),norm(vecs["dr"]@r_body_hat)):
        #     return const_term*(np.cross(ddnadir_vec__dqdq, nadir_vec@sat.J) + tmp + np.transpose(tmp,(1,0,2)) + np.cross(nadir_vec,ddnadir_vec__dqdq@sat.J))*self.active
        # else:
        dc__dq = -9.0*(mu_e)*(vec_norm_jac(R_B,vecs["dr"]))/(norm(R_B)**4.0)
        ddc__dqdq = -9.0*(mu_e)*(vec_norm_hess(R_B,vecs["dr"],vecs["ddr"])/(norm(R_B)**4.0) - 4.0*np.outer(vec_norm_jac(R_B,vecs["dr"]),vec_norm_jac(R_B,vecs["dr"]))/(norm(R_B)**5.0) )
        dv__dq = np.cross(dnadir_vec__dq, nadir_vec@sat.J) + np.cross(nadir_vec, dnadir_vec__dq@sat.J)
        ddv__dqdq = np.cross(ddnadir_vec__dqdq, nadir_vec@sat.J) + tmp + np.transpose(tmp,(1,0,2)) + np.cross(nadir_vec,ddnadir_vec__dqdq@sat.J)
        vec_term = np.cross(nadir_vec, nadir_vec@sat.J)
        tmp2 = np.multiply.outer(dc__dq,dv__dq)
        return (np.multiply.outer(ddc__dqdq,vec_term) + tmp2 + np.transpose(tmp2,(1,0,2)) + const_term*ddv__dqdq)*self.active
        # return (np.outer(dc__dq,np.cross(nadir_vec, nadir_vec@sat.J)) + const_term*(np.cross(dnadir_vec__dq, nadir_vec@sat.J) + np.cross(nadir_vec, dnadir_vec__dq@sat.J)))*self.active
