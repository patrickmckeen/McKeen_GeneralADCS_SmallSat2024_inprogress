from .actuator import *



class MTQ(Actuator):
    """
    This class represents a magnetorquer for a satellite.
    but flexible.

    Parameters
    """
    def __init__(self, axis,std,max_moment,has_bias = False, bias = None,use_noise = True,bias_std_rate = None,estimate_bias=False):
        """
        Initialize the set of sensors.
        See class definition.
        """
        if bias is None:
            bias = np.zeros(1)
        if bias_std_rate is None:
            bias_std_rate = np.zeros(1)
        has_momentum = False
        momentum = np.nan
        momentum_sens_noise_std = np.nan

        self.std = std
        max_h=np.nan
        J=np.nan

        noise_model = lambda std: (np.random.normal(0,std),std)
        super().__init__(axis,max_moment,J,has_momentum,max_h, momentum, momentum_sens_noise_std,has_bias, bias,bias_std_rate,use_noise,noise_model,self.std,1,estimate_bias)

    def clean_torque(self, command,sat,state,vecs):
        """
        clean torque--no bias or noise
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (3), torque generated in body frame
        """''
        if abs(command)>self.max:
            warnings.warn("requested moment exceeds actuation limit")
        b_body = vecs["b"]
        return -np.cross(b_body,self.axis)*command

        # jacobian
        # hessian?
    def dtorq__du(self,command,sat,state,vecs):
        b_body = vecs["b"]
        return -np.cross(b_body,self.axis).reshape((1,3))

    def dtorq__dbasestate(self,command,sat,state,vecs):
        # b_body = vecs["b"]
        db_body__dq = vecs["db"]
        biased_command = command + self.has_bias*self.bias
        return np.vstack([np.zeros((3,3)),-np.cross(db_body__dq,self.axis)*biased_command])

    def ddtorq__dudbasestate(self,command,sat,state,vecs):
        db_body__dq = vecs["db"]
        return np.vstack([np.zeros((3,3)),-np.cross(db_body__dq,self.axis)]).reshape((1,7,3))

    def ddtorq__dbasestatedbasestate(self,command,sat,state,vecs):
        # db_body__dq = vecs["db"]
        ddb_body__dqdq = vecs["ddb"]
        biased_command = command + self.has_bias*self.bias
        out1 = np.zeros((7,7,3))
        out1[3:7,3:7,:] = -np.cross(ddb_body__dqdq,self.axis)*biased_command
        return out1
