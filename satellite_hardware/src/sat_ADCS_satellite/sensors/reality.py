from .sensor import *

class Reality(Sensor):
    """
    This class extends the Sensor class for a gyroscope with noise and bias that drifts over time.

    Parameters
    ----------
    """
    def __init__(self, std, sample_time = 0.1, has_bias = True, bias = None, bias_std_rate = None, use_noise = True,estimate_bias = False,scale = 1):
        if bias_std_rate is None:
            bias_std_rate = np.zeros(7)
        if bias is None:
            bias = np.zeros(7)
        noise_model =  lambda val,std,state,vecs: np.random.multivariate_normal(val,np.diagflat(std**2.0))
        noise_update_model = lambda val,std,state,vecs,dt: std
        self.std = std
        super().__init__(7, sample_time, has_bias, bias,bias_std_rate,use_noise,noise_model,noise_update_model,self.std,estimate_bias,scale)


    def clean_reading(self,state,vecs):
        st = state[0:7].copy()*self.scale
        return st

    def bias_jac(self,x,vecs):
        if self.has_bias:
            return np.eye(7)
        else:
            return np.zeros((0,7))

    def basestate_jac(self,x,vecs):
        return np.eye(7)*self.scale
