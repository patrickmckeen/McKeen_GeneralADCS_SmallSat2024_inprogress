from .disturbance import *
# from typeguard import typechecked

class SRP_Disturbance(Disturbance):
    """
    This class describes a SRP disturbance torque.

    Attributes
    ------------
        params -- list of faces, each composed of a list itself with form [index,area,centroid,normal,eta_s,eta_d,eta_a]
                    indices must go from 0 to N, sequentially, without repeats
                    area is the area of the face in square meters
                    centroid is the position of the centroid of the face in the satellite's body frame coordinates, expressed in meters
                    normal is a unit vector (np.array(3)) giving the direction of the faces outward-normal in the spacecraft body frame
                    eta_a is fractino of light that is absorbed
                    eta_d is frafction of light that has a diffuse reflection
                    eta_s is fraction of light htat has a specular reflection

    """
    #TODO: make solar constant change based on distance from Sun.
    def __init__(self,params,estimate = False,active=True):
        super().__init__(params,False,0,False,active)

    # @typechecked
    def set_params(self,params):
        super().set_params(params)
        face_inds = set([j[0] for j in params])
        if set(face_inds).difference(set([j for j in range(len(params))])):
            raise ValueError("indices must go from 0 to N-1 sequentially")
        self.numfaces = len(params)
        self.areas = np.array([j[1] for j in params])
        self.centroids = np.row_stack([j[2] for j in params])
        self.normals = np.row_stack([normalize(j[3]) for j in params])
        self.eta_a = np.array([j[4] for j in params])
        self.eta_d = np.array([j[5] for j in params])
        self.eta_s = np.array([j[6] for j in params])


    def torque(self,sat,vecs):
        S_B = vecs["s"]
        R_B = vecs["r"]
        os = vecs["os"]
        # r_body_hat = normalize(V_B)
        s_body = normalize(S_B-R_B)
        # magv_m = norm(V_B)*1000
        cos_gamma = np.maximum(0,np.dot(self.normals,s_body))
        proj_area = self.areas*cos_gamma
        cents = self.centroids-sat.COM
        m_s = proj_area*(self.eta_a+self.eta_d)
        t_s = m_s@np.cross(cents,s_body)
        m_n = proj_area*(2*self.eta_s*cos_gamma + (2/3)*self.eta_d)
        t_n = m_n@np.cross(cents,self.normals)
        # F_a = outer(self.eta_a,s_body)
        # F_d = self.eta_d*(2*self.normals/3 + s_body)
        # F_s = 2*self.eta_s*cos_gamma*self.normals
        # v2 = F_a+F_d+F_s
        # v1 = cents*self.areas*cos_gamma
        # t_s = np.np.cross(v1,F_s)
        # t_n = np.np.cross(v1,F_n)
        # t_s = np.np.cross(v1,F_s)
        return -(solar_constant/c)*(t_s+t_n)*self.active*( not os.in_eclipse())

    def torque_qjac(self,sat,vecs): #torque jacobian over quaternion
        S_B = vecs["s"]
        R_B = vecs["r"]
        os = vecs["os"]
        # r_body_hat = normalize(V_B)
        s_body = normalize(S_B-R_B)
        # magv_m = norm(V_B)*1000
        # dsb__dq = vecs["ds"]
        # ddsb__dqdq = vecs["dds"]
        # dsb__dq = vec_norm_jac(S_B,vecs["ds"])
        # drb__dq = vec_norm_jac(R_B,vecs["dr"])
        # ddrb__dqdq = vecs["dds"]
        ds_body__dq = normed_vec_jac(S_B-R_B,vecs["ds"]-vecs["dr"])#(dsb__dq-drb__dq)#/norm(S_B-R_B)
        # dds_body__dqdq = (ddsb__dqdq-ddrb__dqdq)/norm(S_B-R_B)
        cos_gamma = np.maximum(0,np.dot(self.normals,s_body))
        proj_area = self.areas*cos_gamma
        cents = self.centroids-sat.COM
        dcos_gamma__dq = (cos_gamma>0)*(ds_body__dq@self.normals.T)
        dproj_area__dq = self.areas*dcos_gamma__dq
        m_s = proj_area*(self.eta_a+self.eta_d)
        dm_s__dq = dproj_area__dq*(self.eta_a+self.eta_d)
        dt_s__dq = dm_s__dq@np.cross(cents,s_body) + np.cross(m_s@cents,ds_body__dq)
        dm_n__dq = dproj_area__dq*(2*self.eta_s*cos_gamma + (2/3)*self.eta_d) + proj_area*(2*self.eta_s*dcos_gamma__dq)
        dt_n__dq = dm_n__dq@np.cross(cents,self.normals)

        # cos_gamma = np.maximum(0,np.dot(self.normals,s_body))
        # cents = self.centroids-sat.COM
        # dcos_gamma__dq = (cos_gamma>0)*(ds_body__dq@self.normals.T)
        # m_s = self.areas*cos_gamma*(self.eta_a+self.eta_d)
        # dm_s__dq = self.areas*dcos_gamma__dq*(self.eta_a+self.eta_d)
        # t_s = np.cross(m_s@cents,s_body)
        # dt_s__dq = np.cross(dm_s__dq@cents,s_body) + np.cross(m_s@cents,ds_body__dq)
        # m_n = self.areas*(2*self.eta_s*cos_gamma + (2/3)*self.eta_d)
        # dm_n__dq = self.areas*dcos_gamma__dq*(4*self.eta_s)
        # # test = np.expand_dims(cos_gamma,1)*self.normals
        # t_n = np.dot(m_n,np.cross(cents,np.expand_dims(cos_gamma,1)*self.normals))
        # dt_n__dq = dm_n__dq@np.cross(cents,self.normals*np.expand_dims(cos_gamma,1)) + dcos_gamma__dq@np.cross(cents,self.normals*np.expand_dims(m_n,1))

        return -(solar_constant/c)*(dt_s__dq+dt_n__dq)*self.active*(not os.in_eclipse())

    def torque_qqhess(self,sat,vecs): #hessian of torque elements over quaternion
        S_B = vecs["s"]
        R_B = vecs["r"]
        os = vecs["os"]
        # r_body_hat = normalize(V_B)
        s_body = normalize(S_B-R_B)
        ds_body__dq = normed_vec_jac(S_B-R_B,vecs["ds"]-vecs["dr"])
        dds_body__dqdq = normed_vec_hess(S_B-R_B,vecs["ds"]-vecs["dr"],vecs["dds"]-vecs["ddr"])
        # magv_m = norm(V_B)*1000
        # dsb__dq = vecs["ds"]
        # ddsb__dqdq = vecs["dds"]
        # drb__dq = vecs["ds"]
        # ddrb__dqdq = vecs["dds"]
        # ds_body__dq = (dsb__dq-drb__dq)/norm(S_B-R_B)
        # dds_body__dqdq = (ddsb__dqdq-ddrb__dqdq)/norm(S_B-R_B)
        cos_gamma = np.maximum(0,np.dot(self.normals,s_body))
        proj_area = self.areas*cos_gamma
        cents = self.centroids-sat.COM
        dcos_gamma__dq = (cos_gamma>0)*(ds_body__dq@self.normals.T)
        ddcos_gamma__dqdq = (cos_gamma>0)*(dds_body__dqdq@self.normals.T)
        dproj_area__dq = self.areas*dcos_gamma__dq
        ddproj_area__dqdq = self.areas*ddcos_gamma__dqdq
        m_s = proj_area*(self.eta_a+self.eta_d)
        dm_s__dq = dproj_area__dq*(self.eta_a+self.eta_d)
        ddm_s__dqdq = ddproj_area__dqdq*(self.eta_a+self.eta_d)
        dt_s__dq = dm_s__dq@np.cross(cents,s_body) + np.cross(m_s@cents,ds_body__dq)
        tmp = np.cross(np.expand_dims(dm_s__dq@cents,0),np.expand_dims(ds_body__dq,1))
        ddt_s__dqdq = ddm_s__dqdq@np.cross(cents,s_body) + tmp + np.transpose(tmp,(1,0,2))+ np.cross(m_s@cents,dds_body__dqdq)
        dm_n__dq = dproj_area__dq*(2*self.eta_s*cos_gamma + (2/3)*self.eta_d) + proj_area*(2*self.eta_s*dcos_gamma__dq)
        tmp2 = np.expand_dims(dproj_area__dq,0)*np.expand_dims((2*self.eta_s*dcos_gamma__dq),1)
        ddm_n__dqdq = ddproj_area__dqdq*(2*self.eta_s*cos_gamma + (2/3)*self.eta_d) + tmp2 + np.transpose(tmp2,(1,0,2)) + proj_area*(2*self.eta_s*ddcos_gamma__dqdq)
        ddt_n__dqdq = ddm_n__dqdq@np.cross(cents,self.normals)

        return -(solar_constant/c)*(ddt_s__dqdq+ddt_n__dqdq)*self.active*(not os.in_eclipse())
