
import numpy as np
from skyfield import api, framelib, positionlib, units,functions,toposlib
from skyfield.functions import T as sffT
from sat_ADCS_helpers import *
import ppigrf
import pytz
import pytest
import warnings
from datetime import datetime, timezone
import time

SMAD_altrange = np.array([0, 100, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550,
    600, 650, 700, 750, 800, 850, 900, 950, 1000]) #km
SMAD_rhovsalt = np.array([1.2, 5.69e-7, 2.02e-9, 7.66e-10, 2.90e-10, 1.46e-10, 7.30e-11, 4.10e-11,
    2.30e-11, 1.38e-11, 8.33e-12, 5.24e-12, 3.29e-12, 1.39e-12, 6.15e-13, 2.84e-13, 1.37e-13, 6.87e-14, 3.63e-14,
    2.02e-14, 1.21e-14, 7.69e-15, 5.24e-15, 3.78e-15, 2.86e-15]) #kg/m^3

class Orbital_State:
    def __init__(self,t,R,V,S=None,B=None,rho=None,altrange = None, rhovsalt = None,timing=False,calc_all = True,calc_B= True,calc_S =True):#,gc = None, hc = None):
        self.J2000 = t
        self.R = R
        self.V = V
        if timing:
            print('TAI')
            t0 = time.process_time()
        self.TAI = self.j2000_to_tai()


        pos_time = ts.tai_jd(self.TAI)
        if timing:
            t1 = time.process_time()
            dt = t1-t0
            print(dt)
            print('sfpos')
            t0 = time.process_time()
        # posECIkm =
        pos = units.Distance(km=self.R.tolist())
        vel_sf = units.Velocity(km_per_s=self.V.tolist())
        #Create ICRF aka ECI position (Geocentric) from J2000 (ECI)
        # print(pos_time)
        self.sf_pos = positionlib.ICRF(pos.au.tolist(),velocity_au_per_d = vel_sf.au_per_d.tolist(),t = pos_time,center = 399)
        if timing:
            t1 = time.process_time()
            dt = t1-t0
            print(dt)
            print('datetime')
            t0 = time.process_time()
        self.datetime = self.sf_pos.t.astimezone(timezone.utc)
        self.datetime = self.datetime.replace(tzinfo = None)

        if timing:
            tall = time.process_time()
        if calc_all:
            if timing:
                t1 = time.process_time()
                dt = t1-t0
                print(dt)
                print('geopos')
                t0 = time.process_time()
            self.sf_geo_pos = api.wgs84.geographic_position_of(self.sf_pos)
            if timing:
                t1 = time.process_time()
                dt = t1-t0
                print(dt)
                print('lla')
                t0 = time.process_time()
            self.LLA = np.array([self.sf_geo_pos.latitude.radians,self.sf_geo_pos.longitude.radians,self.sf_geo_pos.elevation.km])
            if timing:
                t1 = time.process_time()
                dt = t1-t0
                print(dt)
                print('ecienumat')
                t0 = time.process_time()
            self.ECI2ENUmat = np.vstack([unitvecs[1],unitvecs[0],unitvecs[2]])@self.sf_geo_pos.rotation_at(self.sf_pos.t)#np.diagflat([1,-1,1])#np.vstack([xdiff,ydiff,zdiff])
            if timing:
                t1 = time.process_time()
                dt = t1-t0
                print(dt)
        # else:
        #     self.sf_geo_pos = []
        #     self.LLA = nan3.copy()
        #     self.ECI2ENUmat = np.nan*np.eye(3)
        if timing:
            dt = time.process_time() - tall
            print('t all')
            print(dt)


        if timing:
            print('ecef')
            t0 = time.process_time()
        self.ECEF = self.sf_pos.frame_xyz(framelib.itrs).km
        if timing:
            t1 = time.process_time()
            dt = t1-t0
            print(dt)
            print('geocentric')
            t0 = time.process_time()
        r = norm(self.ECEF)
        th = np.arccos(self.ECEF[2]/r)
        ph = np.arctan2(self.ECEF[1],self.ECEF[0])
        self.geocentric = np.array([r,th,ph])
        if timing:
            t1 = time.process_time()
            dt = t1-t0
            print(dt)
        if altrange is None or rhovsalt is None:
            self.altrange = SMAD_altrange
            self.rhovsalt = SMAD_rhovsalt
        else:
            self.altrange = altrange
            self.rhovsalt = rhovsalt

        if timing:
            if calc_S:
                if S is None:
                    print('S')
                    t0 = time.process_time()
                    self.S = self.get_sun_eci()
                    t1 = time.process_time()
                    dt = t1-t0
                    print(dt)
                else:
                    self.S = S
            else:
                self.S = nan3
            if calc_B:
                if B is None:
                    print('B')
                    t0 = time.process_time()
                    # self.B = 1e-4*np.array([[1,0,0]]).T
                    self.B = self.get_b_eci()
                    t1 = time.process_time()
                    dt = t1-t0
                    print(dt)
                else:
                    self.B = B
            else:
                self.B = nan3

            # self.B = np.array([[0,1,0]]).T*1e-4
            if rho is None:
                print('rho')
                t0 = time.process_time()
                self.rho = np.interp(norm(self.R) - R_e, self.altrange, self.rhovsalt)
                t1 = time.process_time()
                dt = t1-t0
                print(dt)
            else:
                self.rho = rho
        else:
            if calc_S:
                if S is None:
                    self.S = self.get_sun_eci()
                else:
                    self.S = S
            else:
                self.S = nan3

            if calc_B:
                if B is None:
                    self.B = self.get_b_eci()
                else:
                    self.B = B
            else:
                self.B = nan3

            if rho is None:
                self.rho = np.interp(norm(self.R) - R_e, self.altrange, self.rhovsalt)
            else:
                self.rho = rho

    def copy(self):
        return self.average(self,0)

    def average(self,os2,ratio = 0.5):
        j2000 = (1-ratio)*self.J2000 + ratio*os2.J2000
        R = (1-ratio)*self.R + ratio*os2.R
        V = (1-ratio)*self.V + ratio*os2.V
        S = (1-ratio)*self.S + ratio*os2.S
        B = (1-ratio)*self.B + ratio*os2.B
        rho = (1-ratio)*self.rho + ratio*os2.rho
        altrange = self.altrange
        rhovsalt = self.rhovsalt
        # gc = self.g_coeffs#.copy()
        # hc = self.h_coeffs#.copy()
        if not np.all(self.altrange == os2.altrange):
            warnings.warn('non-matching altitude range in atmospheric model between 2 orbital states')
            # print(self.altrange)
            # print(os2.altrange)
            altrange = SMAD_altrange#np.copy(SMAD_altrange)
            rho = None
            rhovsalt = SMAD_rhovsalt#np.copy(SMAD_rhovsalt)
        if not np.all(self.rhovsalt == os2.rhovsalt):
            warnings.warn('non-matching air density vs altitude in atmospheric model between 2 orbital states')
            altrange = SMAD_altrange#np.copy(SMAD_altrange)
            rho = None
            rhovsalt = SMAD_rhovsalt#np.copy(SMAD_rhovsalt)
        # if not (self.g_coeffs.equals(os2.g_coeffs) and self.h_coeffs.equals(os2.h_coeffs)):
        #     warnings.warn('non-matching coefficients in geomagnetic model between 2 orbital states')
        #     gc = None
        #     hc = None
        #     B = None
        return Orbital_State(j2000,R,V,S=S,B=B,rho=rho,altrange=altrange,rhovsalt=rhovsalt)#,gc=gc,hc=hc)

    def orbit_dynamics(self, J2_on = True,conc=False):
        #TODO: verify against STK.
        #Possible improvement: Add perturbations from additional bodies (moon + Sun)????
        #Possible improvement: Add J3 (should be super easy).
        r_ECIk = self.R
        v_ECIk = self.V
        rn = norm(r_ECIk)
        zk = r_ECIk[2]
        #J2 term comes from JGM-3 via Wikipedia
        v_dot = -mu_e*r_ECIk/rn**3.0
        rn_dot = np.dot(r_ECIk,v_ECIk)/rn
        # vdd = -mu_e*(v_ECIk + r_ECIk*(-3.0/rn)*rn_dot)/rn**3.0
        if J2_on:
            j2_mult = np.diagflat(np.array([1.0,1.0,3.0])*rn**2.0 - np.ones(3)*5.0*zk*zk)#r_ECIk@np.outer(-5*zk*np.outer(unitvecs[2],np.ones(3))+np.outer(r_ECIk,np.array([1,1,3])))
            coeff = mu_e*(1.0/rn**7.0)*(J2coeff*R_e**2) * (3.0/2.0)
            v_dot += -coeff * r_ECIk@j2_mult
            # dj2_mult__dr = 2.0*(-5*zk*np.outer(unitvecs[2],np.ones(3)) + np.outer(r_ECIk,np.array([1.0,1.0,3.0])) )#2.0*np.ones(3)@np.array([xk,yk,-4.0*zk]) + 4.0*np.vstack([np.zeros((2,3)),r_ECIk.T])
            # vdd += -coeff * ((-7.0/rn)*rn_dot*r_ECIk@j2_mult + v_ECIk@j2_mult + v_ECIk@dj2_mult__dr@np.diagflat(r_ECIk))
        # r_dot = v_ECIk
        # rdd = v_dot
        if conc:
            return np.concatenate([r_dot,v_dot]),v_ECIk,v_dot
        return v_ECIk, v_dot

    def orbit_dynamics_jacobians(self,J2_on = True,conc=False):
        #TODO: verify against STK.
        #Possible improvement: Add perturbations from additional bodies (moon + Sun)????
        #Possible improvement: Add J3 (should be super easy).
        r_ECIk = self.R
        v_ECIk = self.V
        rn = norm(r_ECIk)
        nr = r_ECIk/rn
        zk = r_ECIk[2]
        #J2 term comes from JGM-3 via Wikipedia
        v_dot = -mu_e*r_ECIk/rn**3.0
        rn_dot = np.dot(r_ECIk,v_ECIk)/rn
        # vdd = -mu_e*(v_ECIk + r_ECIk*(-3.0/rn)*rn_dot)/rn**3.0
        r_dot = v_ECIk
        drd__dr = np.zeros((3,3))
        drd__dv = np.eye(3)
        dvd__dr = -mu_e*(np.eye(3) - 3.0*np.outer(nr,nr))/rn**3.0
        dvd__dv = np.zeros((3,3))
        if J2_on:
            j2_mult = np.diagflat(np.array([1.0,1.0,3.0])*rn**2.0 - np.ones(3)*5.0*zk*zk)#r_ECIk@np.outer(-5*zk*np.outer(unitvecs[2],np.ones(3))+np.outer(r_ECIk,np.array([1,1,3])))
            coeff = mu_e*(1.0/rn**7.0)*(J2coeff*R_e**2) * (3.0/2.0)
            v_dot += -coeff * r_ECIk@j2_mult
            # dj2_mult__dr = 2.0*(-5*zk*np.outer(unitvecs[2],np.ones(3)) + np.outer(r_ECIk,np.array([1.0,1.0,3.0])) )#2.0*np.ones(3)@np.array([xk,yk,-4.0*zk]) + 4.0*np.vstack([np.zeros((2,3)),r_ECIk.T])
            # vdd += -coeff * ((-7.0/rn)*rn_dot*r_ECIk@j2_mult + v_ECIk@j2_mult + v_ECIk@dj2_mult__dr@np.diagflat(r_ECIk))
            dvd__dr += -coeff * (-7.0*(np.outer(r_ECIk,r_ECIk@j2_mult))/rn**2.0 + j2_mult + 2.0*(np.outer(-5*zk*unitvecs[2] + r_ECIk,r_ECIk) + 2.0*zk*np.outer(r_ECIk,np.array([0,0,1.0])))) #dj2_mult__dr@np.diagflat(r_ECIk))
        if conc:
            return np.block([[drd__dr,dvd__dr],[drd__dv,dvd__dv]])
        return drd__dr,drd__dv,dvd__dr,dvd__dv

    def orbit_rk4(self,dt, J2_on=True, rk4_on=True,calc_B = True,calc_S = True,calc_all = True):
        r_ECI = self.R
        v_ECI = self.V
        if rk4_on:
            k1a, k1b = self.orbit_dynamics(J2_on)
            k2a_in = r_ECI+k1a*0.5*dt
            k2b_in = v_ECI+k1b*0.5*dt
            os2_in = Orbital_State(self.J2000+(0.5*dt/cent2sec),k2a_in,k2b_in,S=nan3,B=nan3,rho=0,altrange = [], rhovsalt = [],calc_all = calc_all)

            k2a, k2b = os2_in.orbit_dynamics(J2_on)
            k3a_in = r_ECI+k2a*0.5*dt
            k3b_in = v_ECI+k2b*0.5*dt
            os3_in = Orbital_State(self.J2000+(0.5*dt/cent2sec),k3a_in,k3b_in,S=nan3,B=nan3,rho=0,altrange = [], rhovsalt = [],calc_all = calc_all)
            k3a, k3b = os3_in.orbit_dynamics(J2_on)
            k4a_in = r_ECI+k3a*dt
            k4b_in = v_ECI+k3b*dt
            os4_in = Orbital_State(self.J2000+(dt/cent2sec),k4a_in,k4b_in,S=nan3,B=nan3,rho=0,altrange = [], rhovsalt = [],calc_all = calc_all)
            k4a, k4b = os4_in.orbit_dynamics(J2_on)
            r_out = r_ECI + (dt/6.0)*(k1a+k2a*2.0+k3a*2.0+k4a)
            v_out = v_ECI + (dt/6.0)*(k1b+k2b*2.0+k3b*2.0+k4b)
        else:
            k1a, k1b = os.orbit_dynamics(J2_on)
            # k2a_in = r_ECI+k1a*0.5*dt
            # k2b_in = v_ECI+k1b*0.5*dt
            r_out = r_ECI+k1a*dt
            v_out = v_ECI+k1b*dt

        B = None
        rho = None
        S = None
        if not calc_B:
            B=nan3
        if not calc_S:
            S=nan3

        # if calc_env_vecs:
        #     return Orbital_State(self.J2000+(dt/cent2sec),r_out,v_out,altrange = self.altrange, rhovsalt = self.rhovsalt)
        # else:
        return Orbital_State(self.J2000+(dt/cent2sec),r_out,v_out,S=S,B=B,rho=rho,altrange = self.altrange, rhovsalt = self.rhovsalt,calc_all = calc_all)

    def orbit_rk4_jacobians(self,dt, J2_on=True, rk4_on=True):
        r0 = self.R
        v0 = self.V
        if rk4_on:
            rd0,vd0 = self.orbit_dynamics(J2_on)
            dsd0__ds0 = self.orbit_dynamics_jacobians(J2_on,conc=True)
            r1 = r0+rd0*0.5*dt
            v1 = v0+vd0*0.5*dt
            ds1__ds0 = np.eye(6)+0.5*dt*dsd0__ds0
            # dr1__drv0,dv1__drv0 = [dr1__dr0,dr1__dv0],[dv1__dr0,dv1__dv0]
            os1 = Orbital_State(self.J2000+(0.5*dt/cent2sec),r1,v1,S=nan3,B=nan3,rho=0,altrange = [], rhovsalt = [],calc_all=False)

            rd1, vd1 = os1.orbit_dynamics(J2_on)
            dsd1__ds1 = os1.orbit_dynamics_jacobians(J2_on,conc=True)
            dsd1__ds0 = ds1__ds0@dsd1__ds1
            r2 = r0+rd1*0.5*dt
            v2 = v0+vd1*0.5*dt
            ds2__ds0 = np.eye(6)+0.5*dt*dsd1__ds0
            os2 = Orbital_State(self.J2000+(0.5*dt/cent2sec),r2,v2,S=nan3,B=nan3,rho=0,altrange = [], rhovsalt = [],calc_all=False)

            rd2, vd2 = os2.orbit_dynamics(J2_on)
            dsd2__ds2 = os2.orbit_dynamics_jacobians(J2_on,conc=True)
            dsd2__ds0 = ds2__ds0@dsd2__ds2
            r3 = r0+rd2*dt
            v3 = v0+vd2*dt
            ds3__ds0 = np.eye(6)+dt*dsd2__ds0
            os3 = Orbital_State(self.J2000+(1.0*dt/cent2sec),r3,v3,S=nan3,B=nan3,rho=0,altrange = [], rhovsalt = [],calc_all=False)

            rd3, vd3 = os3.orbit_dynamics(J2_on)
            dsd3__ds3 = os3.orbit_dynamics_jacobians(J2_on,conc=True)
            dsd3__ds0 = ds3__ds0@dsd3__ds3
            r4 = r0+(dt/6.0)*(rd0+2.0*rd1+2.0*rd2+rd3)
            v4 = v0+(dt/6.0)*(vd0+2.0*vd1+2.0*vd2+vd3)
            ds4__ds0 = np.eye(6)+(dt/6.0)*(dsd0__ds0 + 2.0*dsd1__ds0 + 2.0*dsd2__ds0 + dsd3__ds0)
            os4 = Orbital_State(self.J2000+(1.0*dt/cent2sec),r4,v4,S=nan3,B=nan3,rho=0,altrange = [], rhovsalt = [],calc_all=False)
            # print(r4,v4)
            # print(os4.R,os4.V)
            return ds4__ds0[0:3,0:3],ds4__ds0[3:6,0:3],ds4__ds0[0:3,3:6],ds4__ds0[3:6,3:6]

        else:
            rd0, vd0 = os.orbit_dynamics(J2_on)
            r1 = r0+rd0*dt
            v1 = v0+vd0*dt
            dr1__dr0 = np.eye(3)+dt*drd0__dr0
            dr1__dv0 = dt*drd0__dv0
            dv1__dr0 = dt*dvd0__dr0
            dv1__dv0 = np.eye(3)+dt*dvd0__dv0
            return dr1__dr0,dr1__dv0,dv1__dr0,dv1__dv0

    def in_eclipse(self):
        #TODO test.
        return not self.sf_pos.is_sunlit(planets)

    def eci_to_ecef(self,vec):
        return framelib.itrs.rotation_at(self.sf_pos.t)@vec

    def ecef_to_eci(self,vec):
        return sffT(framelib.itrs.rotation_at(self.sf_pos.t))@vec

    def ecef_to_geocentric(self,vec):
        n_ecef = normalize(self.ECEF)
        svec = normalize(np.cross(unitvecs[2],n_ecef))
        R = np.vstack([n_ecef,normalize(np.cross(svec,n_ecef)),svec])
        return R.T@vec

    def geocentric_to_ecef(self,vec):
        n_ecef = normalize(self.ECEF) #"up"
        svec = normalize(np.cross(unitvecs[2],n_ecef))  #"east" on earch-centered sphere
        return vec[0]*n_ecef + svec*vec[2] + normalize(np.cross(svec,n_ecef))*vec[1]

    def eci_to_enu(self,vec):
        return vec@self.ECI2ENUmat.T

    def enu_to_eci(self,vec):
        return vec@self.ECI2ENUmat


    def get_b_eci(self,scale_factor=1e-9):
        """
        This function gets the magnetic field at a position in ECI coordinates given
        a time in fractional centuries since the J2000 epoch and IGRF coefficients
        for the specific date loaded with get_igrf_coeffs(). Depends on (our version of)
        ppigrf. Verified.

        Parameters
        -----------

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
        b_r, b_th, b_ph = ppigrf.igrf_gc(self.geocentric[0],self.geocentric[1]*180.0/np.pi, self.geocentric[2]*180.0/np.pi, self.datetime)#ppigrf.igrf_preloaded(lon*180.0/np.pi, lat*180.0/np.pi, alt,self.g_coeffs,self.h_coeffs)
        #these are radial component, "south" component (on an earth-centered sphere with the radius of self.geocentric[0]), "east"
        #assumes all orbital states have same IGRF coefficients
        #Convert to ECI + scale by correct factor
        b_ecef = self.geocentric_to_ecef(np.squeeze(np.array([b_r, b_th, b_ph])))
        # b_ecef = b_r.item()*ecef_vec/self.geocentric[0] + unitvecs[1]*b_ph.item() + np.cross(unitvecs[1],self.ECEF/self.geocentric[0])*b_th.item()
        # b_enu = np.array([b_r, b_th, b_ph])
        # b_ecef = enu_to_ecef(b_enu, lat, lon)
        b_eci = self.ecef_to_eci(b_ecef)
        return b_eci*scale_factor

    def j2000_to_tai(self):
        return self.J2000*36525.0+2451545.0

    def get_sun_eci(self):#, ephem_filepath='coeff_files/de421.bsp'):
        """
        This function gets the sun position in ECI coordinates given a time (or times)
        in fractional centuries since the J2000 epoch. Depends on skyfield, and uses
        JPL ephemeris. Verified.

        Parameters
        -----------
            j2000: np array (N x 1)
                time or times to find sun position at

        Returns
        --------
            sun_eci: np array (3 x 1)
                sun position at times, in ECI
        """
        #Load planets + load timescale
        # planets = api.load(ephem_filepath)
        # sun = planets['sun']
        # earth = planets['earth']
        # ts = api.load.timescale()

        #Create output array
        # j2000 = [j.J2000 for j in os]
        # j2000 = j2000.reshape(j2000.size, 1)
        # sun_eci = np.zeros((3, 1))

        #Loop over all j2000 times and get sun position rel. to Earth center
        # tai = j2000_to_tai(self.J2000)
        pos_time = ts.tai_jd(self.TAI)
        # print(pos_time)
        sun_eci = earth.at(pos_time).observe(sun).apparent().position.km
        # print( earth.at(pos_time).observe(sun).position.km)
        #
        # for i in range(j2000.size):
        #     ut1 = j2000_to_ut1(j2000[i])
        #     pos_time = ts.ut1_jd(ut1)
        #     sun_eci[i] = earth.at(pos_time).observe(sun).apparent().position.km.reshape((3,))

        return sun_eci
