from .sensor import *

class Gyro(Sensor):
    """
    This class extends the Sensor class for a gyroscope with noise and bias that drifts over time.

    Parameters
    ----------
    """
    def __init__(self, axis, std, sample_time = 0.1, has_bias = True, bias = None, bias_std_rate = None, use_noise = True,estimate_bias = False,scale = 1):
        if bias_std_rate is None:
            bias_std_rate = np.zeros(1)
        if bias is None:
            bias = np.zeros(1)
        noise_model = lambda val,std,state,vecs: np.random.normal(val,std)
        noise_update_model = lambda val,std,state,vecs,dt: std
        self.std = std
        self.axis = normalize(axis)
        super().__init__(1, sample_time, has_bias, bias,bias_std_rate,use_noise,noise_model,noise_update_model,self.std,estimate_bias,scale)


    def clean_reading(self,state,vecs):
        return np.dot(state[0:3],self.axis)*self.scale

    def bias_jac(self,x,vecs):
        if self.has_bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))

    def basestate_jac(self,x,vecs):
        return np.vstack([self.axis.reshape((3,1)),np.zeros((4,1))])*self.scale
