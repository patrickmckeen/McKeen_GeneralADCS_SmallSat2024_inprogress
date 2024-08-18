from .sensor import *
from scipy.linalg import block_diag

class GPS(Sensor):
    """
    This class extends the Sensor class to simulate GPS measurements. It simulates GPS propagation error with a
    constant bias offset, and also simulates GPS uncertainty/covariance. It simulates GPS orbital velocity values
    by differencing noisy GPS orbital position values (because this is assumed to be what would happen on orbit).
    """
    def __init__(self, std, sample_time = 0.1, has_bias = True, bias = None, bias_std_rate = None,use_noise = True,estimate_bias = False):
        """
        Initialize the GPS
        See class definition.
        """
        if bias_std_rate is None:
            bias_std_rate = np.zeros(6)
        if bias is None:
            bias = np.zeros(6)
        self.std = std
        noise_model = lambda val,std,state,vecs: np.random.normal(val,std)
        noise_update_model = lambda val,std,state,vecs,dt: std
        # self.prev_r_noisy = np.zeros(3)
        # self.prev_r = np.zeros(3)
        # self.prev_t = -1
        # self.t = 0
        # self.r = np.zeros(3)
        # self.r_noisy = np.zeros(3)
        super().__init__(6, sample_time, has_bias, bias,bias_std_rate,use_noise,noise_model,noise_update_model,self.std,estimate_bias,1)
        self.attitude_sensor = False

    def clean_reading(self,state,vecs):
        os = vecs['os']
        return np.concatenate([os.ECEF,os.eci_to_ecef(os.V)])

    # def reading(self,state,vecs):
    #     #Simulate GPS orbital pos measurement with some covariance, then add "propagation bias"
    #     os = vecs['os']
    #     self.prev_t = self.t
    #     self.t = os.J2000
    #     r = os.ECEF + self.bias[0:3]*self.has_bias
    #     r_noisy = self.add_noise(r,state,vecs)
    #
    #     if self.t == 0:
    #         self.prev_r_noisy = r_noisy
    #         self.prev_r = r
    #     else:
    #         self.prev_r_noisy = self.r_noisy
    #         self.prev_r = self.r
    #     self.r_noisy = r_noisy
    #     self.r = r
    #
    #     v = (self.r-self.prev_r)/((self.t-self.prev_t)*cent2sec)
    #     v_noisy = (self.r_noisy-self.prev_r_noisy)/((self.t-self.prev_t)*cent2sec) + self.bias[4:6]*self.has_bias
    #     return np.vstack([r_biased_noisy,v_noisy])

    def bias_jac(self,x,vecs):
        if self.has_bias:
            return np.eye(6)
        else:
            return np.zeros((0,6))

    def orbitRV_jac(self,x,vecs):
        os = vecs['os']
        mat = np.stack([os.eci_to_ecef(j) for j in unitvecs]).T
        return block_diag(mat,mat)

    # def gps_noise_model(self, truth_val, gps_noise_settings,state,vecs):
    #     #unpack noise settings
    #     r_noisy = np.random.normal(truth_val, self.std)
    #
    #     return r_noisy
