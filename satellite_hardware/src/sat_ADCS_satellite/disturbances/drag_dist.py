from .disturbance import *
# from typeguard import typechecked

class Drag_Disturbance(Disturbance):
    """
    This class describes a drag disturbance torque.

    Attributes
    ------------
        params -- list of faces, each composed of a list itself with form [index,area,centroid,normal,CD]
                    indices must go from 0 to N, sequentially, without repeats
                    area is the area of the face in square meters
                    centroid is the position of the centroid of the face in the satellite's body frame coordinates, expressed in meters
                    normal is a unit vector (np.array(3)) giving the direction of the faces outward-normal in the spacecraft body frame
                    CD is the drag coefficient of the face (usually 2.2)

    """

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
        self.CDs = np.array([j[4] for j in params])


    def torque(self,sat,vecs):
        V_B = vecs["v"]
        rho = vecs["rho"]
        # v_body_hat = normalize(V_B)
        # magv_m = norm(V_B)*1000
        cos_alpha = np.maximum(0,np.dot(self.normals,V_B))
        F = self.CDs*self.areas*cos_alpha
        cents = self.centroids-sat.COM
        ct = (0.5*rho)
        # print('drag',ct,norm(F@cents),norm(-ct*np.cross(F@cents,V_B)*self.active),norm(V_B),self.active)
        return -ct*np.cross(F@cents,V_B)*self.active

    def torque_qjac(self,sat,vecs): #torque jacobian over quaternion
        V_B = vecs["v"]
        rho = vecs["rho"]
        # v_body_hat = normalize(V_B)#.reshape((3,1))
        cos_alpha = np.maximum(0,np.dot(self.normals,V_B))
        dv_body__dq = vecs["dv"]
        F = self.CDs*self.areas*cos_alpha
        cents = self.centroids-sat.COM
        dcos_alpha__dq = (cos_alpha>0)*(dv_body__dq@self.normals.T)
        dF__dq = dcos_alpha__dq*self.CDs*self.areas
        ct = (0.5*rho)
        return -ct*(np.cross(dF__dq@cents,V_B) + np.cross(F@cents,dv_body__dq))*self.active

    def torque_qqhess(self,sat,vecs): #hessian of torque elements over quaternion
        V_B = vecs["v"]
        rho = vecs["rho"]
        dv_body__dq = vecs["dv"]
        ddv_body__dqdq = vecs["ddv"]
        # v_body_hat = normalize(V_B)#.reshape((3,1))
        cos_alpha = np.maximum(0,np.dot(self.normals,V_B))
        F = self.CDs*self.areas*cos_alpha
        cents = self.centroids-sat.COM
        dcos_alpha__dq = (cos_alpha>0)*(dv_body__dq@self.normals.T)
        ddcos_alpha__dqdq = (cos_alpha>0)*(ddv_body__dqdq@self.normals.T)
        dF__dq = dcos_alpha__dq*self.CDs*self.areas
        ddF__dqdq = ddcos_alpha__dqdq*self.CDs*self.areas
        ct = (0.5*rho)
        # breakpoint()
        tmp = np.cross(np.expand_dims(dF__dq@cents,0),np.expand_dims(dv_body__dq,1))
        # print('a',ct*tmp)
        # print('b',ct*np.transpose(tmp,(1,0,2)))
        # print('c',ct*np.cross(ddF__dqdq@cents,V_B))
        # print('d',ct*np.cross(F@cents,ddv_body__dqdq))
        # if np.any(np.dot(self.normals,V_B)<0):
        #     print(max([(90.0-180.0*np.arccos(j/norm(V_B))/np.pi) for j in np.dot(self.normals,V_B) if j<0]))
        # if np.any(cos_alpha>0):
        #     print(min([(90.0-180.0*np.arccos(j/norm(V_B))/np.pi) for j in cos_alpha if j>0]))
        # print('cd',self.CDs)
        # print('A',self.areas)
        # print('c',cents)
        # print('n',self.normals)
        # print('F',F)
        # print('cosal',cos_alpha/norm(V_B))
        # print('al',np.arccos(cos_alpha/norm(V_B))*180.0/math.pi)
        # print('v',V_B)
        # print(dcos_alpha__dq)
        # print(ddcos_alpha__dqdq)
        return -ct*(np.cross(ddF__dqdq@cents,V_B) + tmp + np.transpose(tmp,(1,0,2)) + np.cross(F@cents,ddv_body__dqdq))*self.active
