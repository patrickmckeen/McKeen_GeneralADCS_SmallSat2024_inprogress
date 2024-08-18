from .actuator import *



class RW(Actuator):
    """
    This class represents a reaction wheel for a satellite.

    Parameters
    """

    def __init__(self, axis,std,max_torq,J,momentum,max_h,momentum_sens_noise_std,has_bias = False, bias = None,bias_std_rate = None,use_noise = True,estimate_bias=False):
        """
        Initialize the set of sensors.
        See class definition.
        """
        if bias is None:
            bias = np.zeros(1)
        if bias_std_rate is None:
            bias_std_rate = np.zeros(1)
        has_momentum = True
        self.std = std

        noise_model = lambda std: (np.random.normal(0,std),std)
        super().__init__(axis,max_torq,J,has_momentum, max_h,momentum,momentum_sens_noise_std,has_bias, bias,bias_std_rate,use_noise,noise_model,self.std,1,estimate_bias)

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
            warnings.warn("requested torque exceeds actuation limit")
        return self.axis*command

    def clean_storage_torque(self, command,sat,state,vecs):
        """
        clean torque on momentum storage--no bias or noise
        Parameters
        ----------
        command: numpy array, commanded actuation

        Returns
        ----------
        torque: numpy array (1), torque on own momentum storage
        """
        if abs(command)>self.max:
            warnings.warn("requested torque exceeds actuation limit")
        return -command
        # jacobian
        # hessian?

    def dtorq__du(self,command,sat,state,vecs):
        return self.axis.reshape((1,3))

    def dstor_torq__du(self,command,sat,state,vecs):
        return -np.ones((1,1))
