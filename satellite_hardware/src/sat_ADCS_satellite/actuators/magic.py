
# from flight_adcs.flight_utils.sat_helpers import *
# import sim_adcs.sim_helpers as sim_helpers#TODO: eliminate the need for this?
from .actuator import *



class Magic(Actuator):
  """
  This class represents a magic actuator (ideal thruster with limitless fuel) for a satellite.

  Parameters
  """

  def __init__(self, axis,std,max_torq,has_bias = False, bias = None,bias_std_rate = None,use_noise = True,estimate_bias=False):
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
      max_h = np.nan
      J = np.nan
      self.std = std

      noise_model = lambda std: (np.random.normal(0,std),std)
      super().__init__(axis,max_torq,J,has_momentum, momentum, max_h,momentum_sens_noise_std,has_bias, bias,bias_std_rate,use_noise,noise_model,self.std,1,estimate_bias)

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

  def dtorq__du(self,command,sat,state,vecs):
      return self.axis.reshape((1,3))
