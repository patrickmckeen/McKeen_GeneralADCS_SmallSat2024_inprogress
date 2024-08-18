import math
import numpy as np
from sat_ADCS_helpers import *
# from helpers import *

class Trajectory:
    def __init__(self,time=None,states=None,control=None,gains=None,ctg = None,torq = None):
        if time is None:
            self.times = np.array([])
        else:
            self.times = np.squeeze(np.array(time))
            if self.times.size != self.times.shape[0]:
                # print(self.times)
                raise ValueError("multi-dimensional time?")

        if states is None:
            self.states = np.array([])
        else:
            # print(state)
            # print(state.shape)
            # print(self.times.shape)
            self.states = np.array(states)
            if self.states.shape[1] != self.times.size:
                raise ValueError("states is shaped incorrectly")

        if control is None:
            self.control = np.array([])
        else:
            self.control = np.array(control)
            if self.control.shape[1] != self.times.size :
                raise ValueError("control is shaped incorrectly")

        if gains is None:
            self.gains = []
        else:
            self.gains = gains + [gains[-1]*np.nan]
            if len(self.gains) != self.times.size :
                breakpoint()
                raise ValueError("gains have wrong length in time")

        if ctg is None:
            self.ctg = []
        else:
            self.ctg = ctg
            if len(self.ctg) != self.times.size :
                breakpoint()
                raise ValueError("cost-to-go matrices have wrong length in time")

        if torq is None:
            self.torq = []
        else:
            self.torq = torq
            if self.torq.shape[1] != self.times.size :
                breakpoint()
                raise ValueError("torq matrices have wrong length in time")


    def is_empty(self):
        return self.times.size==0

    def max_time(self):
        if len(self.times) == 0:
            raise ValueError("time outside of span")
        return self.times.max()

    def min_time(self):
        if len(self.times) == 0:
            raise ValueError("time outside of span")
        return self.times.min()

    def time_in_span(self,t):
        if len(self.times) == 0:
            raise ValueError("time outside of span")
        return (t<=self.max_time() and t>=self.min_time())

    def state_nearest_to_time(self,t,return_extrema = False):
        """
        returns the state that occurs nearest to time t. t is in J2000.
        if return_extrema is true and t is greater/less than the time of the last/first state, this returns the last/first state.
        if return_extrema is false and t is greater/less than the time of the last/first state, this function errors.
        """
        if self.time_in_span(t):
            inds = np.argmin(np.abs(self.times-t))
            if np.isscalar(inds):
                ind = int(inds)
            else:
                if len(inds) > 1:
                    ind = int(inds[0])
                else:
                    ind = int(np.ndarray.item(inds))
            return self.states[:,inds]
        elif (t>self.max_time() and return_extrema) or self.time_in_span(t-time_eps/cent2sec):
            return self.states[:,-1]
        elif (t<self.min_time() and return_extrema) or self.time_in_span(t+time_eps/cent2sec):
            return self.states[:,0]
        else:
            breakpoint()
            raise ValueError("time outside of span")

    def gain_nearest_to_time(self,t,return_extrema = False):
        """
        returns the state that occurs nearest to time t. t is in J2000.
        if return_extrema is true and t is greater/less than the time of the last/first state, this returns the last/first state.
        if return_extrema is false and t is greater/less than the time of the last/first state, this function errors.
        """
        if self.time_in_span(t):
            inds = np.argmin(np.abs(self.times-t))
            if np.isscalar(inds):
                ind = int(inds)
            else:
                if len(inds) > 1:
                    ind = int(inds[0])
                else:
                    ind = int(np.ndarray.item(inds))
            return self.gains[ind]
        elif (t>self.max_time() and return_extrema) or self.time_in_span(t-time_eps/cent2sec):
            return self.gains[-1]
        elif (t<self.min_time() and return_extrema) or self.time_in_span(t+time_eps/cent2sec):
            return self.gains[0]
        else:
            raise ValueError("time outside of span")

    def ctg_nearest_to_time(self,t,return_extrema = False):
        """
        returns the cost-to-go matrix that occurs nearest to time t. t is in J2000.
        if return_extrema is true and t is greater/less than the time of the last/first state, this returns the last/first state.
        if return_extrema is false and t is greater/less than the time of the last/first state, this function errors.
        """
        if self.time_in_span(t):
            inds = np.argmin(np.abs(self.times-t))
            if np.isscalar(inds):
                ind = int(inds)
            else:
                if len(inds) > 1:
                    ind = int(inds[0])
                else:
                    ind = int(np.ndarray.item(inds))
            return self.ctg[ind]
        elif (t>self.max_time() and return_extrema) or self.time_in_span(t-time_eps/cent2sec):
            return self.ctg[-1]
        elif (t<self.min_time() and return_extrema) or self.time_in_span(t+time_eps/cent2sec):
            return self.ctg[0]
        else:
            raise ValueError("time outside of span")

    def control_nearest_to_time(self,t,return_extrema = False):
        """
        returns the control that occurs nearest to time t. t is in J2000.
        if return_extrema is true and t is greater/less than the time of the last/first control, this returns last/first control.
        if return_extrema is false and t is greater/less than the time of the last/first control, this function errors.
        """
        if self.time_in_span(t):
            inds = np.argmin(np.abs(self.times-t))
            if np.isscalar(inds):
                ind = int(inds)
            else:
                if len(inds) > 1:
                    ind = int(inds[0])
                else:
                    ind = int(np.ndarray.item(inds))
            return self.control[:,inds]
        elif (t>self.max_time() and return_extrema) or self.time_in_span(t-time_eps/cent2sec):
            return self.control[:,-1]
        elif (t<self.min_time() and return_extrema) or self.time_in_span(t+time_eps/cent2sec):
            return self.control[:,0]
        else:
            raise ValueError("time outside of span")

    def torque_nearest_to_time(self,t,return_extrema = False):
        """
        returns the disturbance torque that occurs nearest to time t. t is in J2000.
        if return_extrema is true and t is greater/less than the time of the last/first disturbance torque, this returns last/first disturbance torque.
        if return_extrema is false and t is greater/less than the time of the last/first disturbance torque, this function errors.
        """
        if self.time_in_span(t):
            inds = np.argmin(np.abs(self.times-t))
            if np.isscalar(inds):
                ind = int(inds)
            else:
                if len(inds) > 1:
                    ind = int(inds[0])
                else:
                    ind = int(np.ndarray.item(inds))
            return self.torque[:,inds]
        elif (t>self.max_time() and return_extrema) or self.time_in_span(t-time_eps/cent2sec):
            return self.torque[:,-1]
        elif (t<self.min_time() and return_extrema) or self.time_in_span(t+time_eps/cent2sec):
            return self.torque[:,0]
        else:
            raise ValueError("time outside of span")




    def info_nearest_to_time(self,t,return_extrema = False):
        """
        returns the planned state, control, gains, cost-to-go matrix, and disturbance torque that occurs nearest to time t. t is in J2000.
        if return_extrema is true and t is greater/less than the time of the last/first info, this returns the last/first info.
        if return_extrema is false and t is greater/less than the time of the last/first info, this function errors.
        """
        if self.time_in_span(t):
            inds = np.argmin(np.abs(self.times-t))
            if np.isscalar(inds):
                ind = int(inds)
            else:
                if len(inds) > 1:
                    ind = int(inds[0])
                else:
                    ind = int(np.ndarray.item(inds))
        elif (t>self.max_time() and return_extrema) or self.time_in_span(t-time_eps/cent2sec):
            ind = -1
        elif (t<self.min_time() and return_extrema) or self.time_in_span(t+time_eps/cent2sec):
            ind = 0
        else:
            raise ValueError("time outside of span")
        return [self.states[:,ind],self.control[:,ind],self.gains[ind],self.ctg[ind],self.torq[:,ind]]




    def last_state_before_time(self,t,return_last = True):
        """
        returns the last state that occurs before time t. t is in J2000.
        if return_last is true and t is greater than the time of the last state, this returns last.
        if return_last is false and t is greater than the time of the last state, this function errors.
        """
        if self.time_in_span(t):
            return self.states[:,np.where(self.times<=t)[0][-1]]
        elif t>self.max_time() and return_last:
            return self.states[:,-1]
        else:
            raise ValueError("time outside of span")

    def last_control_before_time(self,t,return_last = True):
        """
        returns the last control that occurs before time t. t is in J2000.
        if return_last is true and t is greater than the time of the last control, this returns last.
        if return_last is false and t is greater than the time of the last control, this function errors.
        """
        if self.time_in_span(t):
            return self.control[:,np.where(self.times<=t)[0][-1]]
        elif t>self.max_time() and return_last:
            return self.control[:,-1]
        else:
            raise ValueError("time outside of span")

    def first_state(self):
        return self.states[:,0]

    def last_state(self):
        return self.states[:,-1]

    def penultimate_state(self):
        return self.states[:,-2]

    def copy(self):
        out = Trajectory()
        out.times = np.copy(self.times)
        out.states = np.copy(self.states)
        out.control = np.copy(self.control)
        out.torq = np.copy(self.torq)
        out.ctg = [np.copy(j) for j in self.ctg]
        out.gains = [np.copy(j) for j in self.gains]
        return out
