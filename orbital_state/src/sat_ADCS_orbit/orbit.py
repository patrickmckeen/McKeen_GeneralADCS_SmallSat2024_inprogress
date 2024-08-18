
import numpy as np
from sat_ADCS_helpers import *
from skyfield import api, framelib, positionlib, units,functions,toposlib
from skyfield.functions import T as sffT
import pytest
from .orbital_state import *
import time

"""
to update from estimation folder

cd ../orbital_state && \
python3.10 -m build && \
pip3.10 install ./dist/sat_ADCS_orbit-0.0.1.tar.gz && \
cd ../estimation
"""

class Orbit:
    def __init__(self,orbit0,end_time = None,dt = None,use_J2 = True,each_calc_B = False,all_data = False): #dt is second, end_time is fractional centuries after 2000

        if isinstance(orbit0,Orbital_State):
            start_time = orbit0.J2000
            if end_time is None or dt is None or end_time == orbit0.J2000 or dt==0:
                #orbit comprised of just 1 state
                self.states = {orbit0.J2000 : orbit0.copy()}
                self.times = np.array([orbit0.J2000])
            else:
                #orbit propagated from that state
                duration = end_time - start_time
                l = np.floor(duration/(dt/cent2sec))
                times = [start_time] + [start_time + j*(dt/cent2sec) for j in range(1+int(l))] + [end_time]
                # print(times)
                times = list(set(times))#make unique
                times.sort()
                # states0 = [Orbital_State(j,unitvecs[0],unitvecs[1],S=np.nan*np.ones(3),B=np.nan*np.ones(3),rho=0,altrange = orbit0.altrange, rhovsalt = orbit0.rhovsalt) for j in times]
                states0 = [np.nan for j in times]
                states0[0] = orbit0.copy()
                times[0] = orbit0.J2000
                for j in list(range(1,len(times))):
                    states0[j] = states0[j-1].orbit_rk4((times[j]-times[j-1])*cent2sec, J2_on=use_J2, rk4_on=True,calc_B = each_calc_B,calc_all = all_data) #TODO: vectorize calculating environmental variables
                    # times[j] = states0[j].J2000
                    # print(j,len(times),times[j])
                self.states = {j.J2000 : j for j in states0}
                self.times = np.array(sorted([*self.states]))
                if not each_calc_B:
                    Bvecs = self.get_b_eci_orbit()
                    for j in range(len(self.times)):
                        self.states[self.times[j]].B = Bvecs[j,:]
        elif isinstance(orbit0,list) and all([isinstance(j,Orbital_State) for j in orbit0]):
            unique_times = set([j.J2000 for j in orbit0])
            self.states = {j.J2000 : j.copy() for j in orbit0 if j.J2000 in unique_times}
            self.times = np.array(sorted([*self.states]))
        else:
            raise ValueError("either an initial orbital state or a list of orbital states must be provided")

    def get_os(self,t):
        if t>self.max_time():
            # breakpoint()
            raise ValueError('this orbital state is not within this orbit (too far in future)')
        if t<self.min_time():
            raise ValueError('this orbital state is not within this orbit (too far in past)')
        close = np.isclose(self.times,t,rtol = 0.0,atol = 1e-2/cent2sec)
        if np.any(close):
            inds = np.flatnonzero(close)
            if len(inds)==1:
                #one match
                ind = inds[0]
                return self.states[self.times[ind]].copy()
            elif len(inds)>1:
                warnings.warn('more than one match---how???')
                close_times = self.times[inds]
                inds2 = np.argmin(np.abs(np.array(close_times) - t))
                if np.isscalar(inds2):
                    ind = int(inds2)
                else:
                    ind = int(inds2[0]) #just take the first one, even if its too long
                return self.states[close_times[ind]].copy()
        ##no perfect match
        i0 = np.flatnonzero(self.times<t)[-1]
        i1 = np.flatnonzero(self.times>t)[0]
        t0 = self.times[i0]
        t1 = self.times[i1]
        return self.states[t0].average(self.states[t1],ratio = (t-t0)/(t1-t0))


    def get_range(self,t_0,t_1,dt = None):
        """
        if dt unspecified, just returns all values betwen t_0 and t_1, including them if they are equal to the times of states in this object
        if dt is specified, this generates a new orbit object between t_0 and t_1 with time step dt, creating states from the states in this object if they don't match exactly
        """
        if t_1<t_0:
            raise ValueError('times are in wrong order')

        if t_1==t_0:
            if dt is not None:
                return self.get_os(t_0)
            elif t_0 in self.times:
                return self.get_os(t_0)
            raise ValueError('times are equal, no matching time exactly. Try again with a specified dt or a wider time bracket. (or use the get_os() method)')
        if t_0>self.max_time():
            raise ValueError('first orbital state is not within this orbit (too far in future)')
        if t_0<self.min_time():
            raise ValueError('first orbital state is not within this orbit (too far in past)')
        if t_1>self.max_time():
            raise ValueError('last orbital state is not within this orbit (too far in future)')
        if t_1<self.min_time():
            raise ValueError('last orbital state is not within this orbit (too far in past)')

        if dt is None:
            # newtimes = [j for j in self.times if (j<=t_1 and j>=t_0)]
            newstates = [self.states[j] for j in self.times if (j<=t_1 and j>=t_0)]
            if len(newstates)==0:
                raise ValueError('there are no pre-created states in this time span')
            orbit_out = Orbit(newstates)
            return orbit_out
        else:
            ts = np.concatenate([np.arange(t_0,t_1,dt/cent2sec),[t_1]])
            # ts = np.unique(ts)
            return self.new_orbit_from_times(ts)

    def new_orbit_from_times(self,time_list):
        if not np.all([self.time_in_span(j) for j in time_list]):
            print(self.max_time(), self.min_time(),min(time_list),max(time_list))
            raise ValueError('at least one time is not within this orbit span')
        # newtimes = sorted(list(set(time_list)))
        newstates = [self.get_os(j) for j in time_list]
        return Orbit(newstates)

    def next_state(self,input):
        if isinstance(input,Orbital_State):
            t = input.J2000
        elif isinstance(input,float):
            t = input
        else:
            raise ValueError("Must be j2000 time or orbital state")

        if t>self.max_time():
            raise ValueError('this orbital state is not within this orbit (too far in future)')
        if t<self.min_time():
            raise ValueError('this orbital state is not within this orbit (too far in past)')

        ind = np.flatnonzero(self.times>=t)[0]

        return self.states[self.times[ind]]

    def get_vecs(self):
        R = [self.states[j].R for j in self.times]
        V = [self.states[j].V for j in self.times]
        B = [self.states[j].B for j in self.times]
        S = [self.states[j].S for j in self.times]
        rho = [self.states[j].rho for j in self.times]
        return [R,V,B,S,rho]

    def min_time(self):
        return np.amin(self.times)

    def max_time(self):
        return np.amax(self.times)

    def time_in_span(self,t):
        return t<=self.max_time() and t>=self.min_time()

    def geocentric_to_ecef_orbit(self,vecs):
        ecef_mat = np.vstack([self.states[j].ECEF for j in self.times])
        n_ecef = matrix_row_normalize(ecef_mat)
        svec = matrix_row_normalize(np.cross(unitvecs[2],n_ecef))
        return vecs[:,0:1]*n_ecef + svec*vecs[:,2:] + matrix_row_normalize(np.cross(svec,n_ecef))*vecs[:,1:2]


    def ecef_to_eci_orbit(self,vecs):

        return np.stack([self.states[self.times[j]].ecef_to_eci(vecs[j,:]) for j in range(len(self.times))])

    def get_b_eci_orbit(self,scale_factor=1e-9):
        """
        This function gets the magnetic field at a position in ECI coordinates given
        a time in fractional centuries since the J2000 epoch and IGRF coefficients
        for the specific date loaded with get_igrf_coeffs(). Depends on (our version of)
        ppigrf. Verified.

        Vectorized. Can take N x 3 eci_pos and N x 1 j2000.

        Parameters
        -----------
            eci_pos: np array (3 x 1 or N x 3)
                ECI position, km
            j2000: float or N x 1 np array
                time since J2000 epoch in fractional centuries
            g_coeffs: Dataframe
                g coefficients for date
            h_coeffs: Dataframe
                h coefficients for date
            scale_factor (optional): float
                conversion to T, default is 1e-9 to go from nT -> T

        Returns
        --------
            b_eci: np array (3 x 1 or N x 3)
                magnetic field vector in ECI coordinates, in T
        """
        #Get lat/lon (in radians) + altitude above Earth
        # R = np.hstack([j.R for j in os])
        # print(R.shape)
        # j2000 = np.atleast_1d(np.squeeze(np.array([j.J2000 for j in os])))
        # lat, lon, alt = eci_to_lla(self.R, self.J2000)
        # ecef_vec = eci_to_ecef(self.R, self.J2000)
        #Use doctored ppigrf to get B-field in ENU
        # b_e,b_n,b_u = ppigrf.igrf(self.LLA[1]*180.0/np.pi,self.LLA[0]*180.0/np.pi, self.LLA[2], self.datetime)
        # print('b_enu',np.squeeze(np.array([b_e,b_n,b_u]))*1e-9)
        geos = np.vstack([self.states[j].geocentric for j in self.times])
        dts = [self.states[j].datetime for j in self.times]
        b_r, b_th, b_ph = ppigrf.igrf_gc(geos[:,0],geos[:,1]*180.0/np.pi,geos[:,2]*180.0/np.pi,dts)#ppigrf.igrf_preloaded(lon*180.0/np.pi, lat*180.0/np.pi, alt,self.g_coeffs,self.h_coeffs)
        b_r = np.diagonal(b_r)
        b_th = np.diagonal(b_th)
        b_ph = np.diagonal(b_ph)

        #assumes all orbital states have same IGRF coefficients
        #Convert to ECI + scale by correct factor
        b_ecef = self.geocentric_to_ecef_orbit(np.atleast_2d(np.squeeze(np.stack([b_r, b_th, b_ph])).T))
        # b_ecef = b_r.item()*ecef_vec/self.geocentric[0] + unitvecs[1]*b_ph.item() + np.cross(unitvecs[1],self.ECEF/self.geocentric[0])*b_th.item()
        # b_enu = np.array([b_r, b_th, b_ph])
        # b_ecef = enu_to_ecef(b_enu, lat, lon)
        b_eci = self.ecef_to_eci_orbit(b_ecef)
        return b_eci*scale_factor
