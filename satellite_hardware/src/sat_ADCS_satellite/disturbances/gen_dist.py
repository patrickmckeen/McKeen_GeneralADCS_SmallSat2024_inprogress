from .disturbance import *
# from typeguard import typechecked

class General_Disturbance(Disturbance):
    """
    This class describes a general disturbance torque.

    Attributes
    ------------
        params -- torque vector in vbody frame

    """

    # @typechecked
    def set_params(self,params):
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
        if j2000>self.last_update:
            if self.time_varying:
                update_torq = np.random.multivariate_normal(self.main_param, ((j2000-self.last_update)*cent2sec*self.std)**2.0)
                self.main_param = update_torq*min(1.0,self.mag_max/norm(update_torq))
            self.last_update = j2000
        return self

    def torque(self,sat,vecs):
        return self.main_param*self.active

    def torque_valjac(self,sat,vecs):
        return np.eye(3)*self.active
