from .disturbance import *
# from typeguard import typechecked

class Dipole_Disturbance(Disturbance):
    """
    This class describes a magnetic dipole disturbance torque.

    Attributes
    ------------
        params -- inherent dipole in Am^2 in body frame

    """

    # @typechecked
    def set_params(self,params):
        #std is per second.
        super().set_params(params)
        self.main_param = params[0]
        if isinstance(params,np.ndarray):
            self.main_param = params
        self.last_update = np.nan
        if self.time_varying:
            self.mag_max = params[1]
        if np.size(self.std)==1:
            self.std = np.eye(3)*self.std

    def update(self,j2000):
        if np.isnan(self.last_update) or self.last_update == 0:
            self.last_update = j2000
            return self

        if j2000 > self.last_update:
            if self.time_varying:
                update_interference = np.random.multivariate_normal(self.main_param, (self.std*(j2000-self.last_update)*cent2sec)**2.0)
                self.main_param = update_interference/max(1.0,norm(update_interference)/self.mag_max)
            self.last_update = j2000
        return self

    def torque(self,sat,vecs):
        B_B = vecs["b"]
        return np.cross(self.main_param, B_B)*self.active

    def torque_qjac(self,sat,vecs): #torque jacobian over quaternion
        # B_B = vecs["b"]
        db_body__dq = vecs["db"]
        return np.cross(self.main_param, db_body__dq)*self.active

    def torque_qqhess(self,sat,vecs): #hessian of torque elements over quaternion
        # B_B = vecs["b"]
        ddb_body__dqdq = vecs["ddb"]
        return np.cross(self.main_param, ddb_body__dqdq)*self.active

    def torque_valjac(self,sat,vecs):
        B_B = vecs["b"]
        return np.cross(np.eye(3), B_B)*self.active

    def torque_qvalhess(self,sat,vecs):
        db_body__dq = vecs["db"]
        return np.cross(np.expand_dims(np.eye(3),0), np.expand_dims(db_body__dq,1))*self.active
