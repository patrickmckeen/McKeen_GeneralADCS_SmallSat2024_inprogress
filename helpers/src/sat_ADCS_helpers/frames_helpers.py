from skyfield import api, framelib, positionlib, units,functions
from skyfield.functions import T as sffT
import importlib_resources as pkg_resources
import pytest
import numpy as np
import math
from .helpers import *
from . import data

pkg = pkg_resources.files("sat_ADCS_helpers")
pkg_data_file = pkg / "data" / "de421.bsp"

planets = api.load_file(pkg_data_file)
sun = planets['sun']
earth = planets['earth']
ts = api.load.timescale()
#
#
#
# def ecef_to_eci(ecef_vec, j2000_vec):
#     """
#     This function rotates a vector from the ECEF coordinate frame to the ECI frame.
#     Depends on skylib. Can take vectorized inputs (N x 3 and N x 1). Verified.
#
#     Parameters
#     -----------
#         ecef_vec: np array (3 x 1 or N x 3)
#             position vector in ECEF, km
#         j2000_vec: float or N x 1 np array
#             decimal centuries since start of J2000 epoch
#         num: N
#
#     Returns
#     ---------
#         eci_out: np array (3, or N x 3)
#             positions in ECI, km
#     """
#     #Reshape inputs if we are only converting one set of coordinates
#     # ecef_vec = np.array(ecef_vec)
#     # if ecef_vec.size == 3:
#     # ecef_vec = ecef_vec.reshape((3,1))
#     # j2000_vec = (np.array(j2000_vec)).item()
#     # elif ecef_vec.shape[0] != 3:
#     #     ecef_vec = ecef_vec.T
#
#     # eci_out = np.zeros(ecef_vec.shape)
#     #ts = api.load.timescale()
#     pos_time = ts.tai_jd(j2000_to_tai(j2000_vec))
#     rmats = framelib.itrs.rotation_at(pos_time)
#     # print(rmats)
#     # breakpoint()
#     if isinstance(j2000_vec,float) or j2000_vec.size==1:
#         eci_out = rmats.T@ecef_vec
#     else:
#         eci_out = rot_list([rmats[:,:,j] for j in range(int(rmats.size/9))],ecef_vec,transpose = True)
#
#     return eci_out
#
#
# def eci_to_ecef(eci_vec, j2000_vec):
#     """
#     This function rotates a vector from the ECI coordinate frame to the ECEF frame.
#     Depends on skylib. Can take vectorized inputs: N x 3 and N x 1 respectively.
#     Verified.
#
#     Parameters
#     -----------
#         eci_vec: np array (3 x 1 or N x 3)
#             position vector in ECI, km
#         j2000: float or N x 1 np array
#             decimal centuries since start of J2000 epoch
#
#     Returns
#     ---------
#         ecef_out:
#             3x0 or Nx3 np array of ECEF vectors, km
#     """
#     #Reshape inputs if we are only converting one set of coordinates
#     # if eci_vec.size == 3:
#     #     eci_vec = np.array(eci_vec).reshape((3,1))
#     #     j2000_vec = (np.array(j2000_vec)).item()
#     # elif eci_vec.shape[0] != 3:
#     #     eci_vec = np.array(eci_vec).T
#
#
#
#     jd = j2000_to_tai(j2000_vec)
#     t_skylib = ts.tai_jd(jd)
#     rot_mat = framelib.itrs.rotation_at(t_skylib)
#     # j2000 = time_utils.j2000_to_ut1(j2000_vec)
#     #Create time from UT1 (basically UTC) Julian Date
#     # pos_time = ts.ut1_jd(j2000)
#     #Get pos, set vel to 0 (doesn't matter)
#     # pos = units.Distance(km=[vec[0], vec[1], vec[2]])
#     #Create GCRS aka ECI position (Geocentric) from J2000 (ECI)
#     # sf_vec = positionlib.Geocentric(pos.au, t=pos_time)
#     #Convert to ITRS (ECEF)
#     ecef_out = np.squeeze(functions.mxv(rot_mat,eci_vec))
#
#     #
#     # ecef_out = np.zeros(eci_vec.shape)
#     # #ts = api.load.timescale()
#     #
#     # for i in range(eci_vec.shape[0]):
#     #     vec = eci_vec[i]
#     #     j2000 = time_utils.j2000_to_ut1(j2000_vec[i])
#     #     #Create time from UT1 (basically UTC) Julian Date
#     #     pos_time = ts.ut1_jd(j2000)
#     #     #Get pos, set vel to 0 (doesn't matter)
#     #     pos = units.Distance(km=[vec[0], vec[1], vec[2]])
#     #     #Create GCRS aka ECI position (Geocentric) from J2000 (ECI)
#     #     sf_vec = positionlib.Geocentric(pos.au, t=pos_time)
#     #     #Convert to ITRS (ECEF)
#     #     ecef_out[i] = np.array(sf_vec.frame_xyz(framelib.itrs).km).reshape(3,)
#
#     #Reshape output to (3,) if we are only converting one set of coordinates
#     # if eci_vec.size == 3:
#     #     ecef_out = ecef_out.reshape((3,1))
#
#     return ecef_out
#
# def eci_to_ecef_mat(j2000):
#     """
#     This function returns the ECI to ECEF rotation matrix at a given time. The
#     output of this function is rotmat, and you can expect v_ECEF = rotmat @ v_ECI.
#     This function has been verified. Depends on skylib.
#
#     Parameters
#     -----------
#         j2000: float
#             j2000 time
#
#     Returns
#     --------
#         rot_mat: np array (3 x 3)
#             rotation matrix from ECI to ECEF frame at time j2000
#     """
#     jd = j2000_to_tai(j2000)
#     #ts = api.load.timescale()
#     t_skylib = ts.tai_jd(jd)
#     rot_mat = framelib.itrs.rotation_at(t_skylib)
#     return rot_mat
#
# def j2000_to_tai(j2000, days_per_century=36525.0, j2000_tai=2451545.0):
#     """
#     This function converts j2000 time in fractional centuries since J2000 epoch
#     to TAI days.
#
#     Parameters
#     -----------
#         j2000: float
#             time in fractional centuries since J2000 epoch
#         days_per_century: float
#             days per century (default: 36525.0, assumes no leap seconds)
#         j2000_tai: float
#             TAI time at the start of J2000 epoch, default 2451544.5
#
#     Returns
#     ---------
#         tai: float
#             time in UT1 seconds
#     """
#     tai = j2000*days_per_century+j2000_tai
#     return tai
#
# def eci_to_lla(eci_vec, j2000_vec):
#     """
#     This function converts an ECI position into LLA coordinates. Uses skyfield.
#     Works on WGS84 ellipsoid only (so can't use on non-Earth ellipsoids). Verified.
#
#     Parameters
#     ------------
#         eci_pos: array-like (3x1)
#             position in ECI, in km
#         j2000: fractional centuries since J2000 epoch
#
#     Returns
#     ---------
#         lat: np array (1 x 1)
#             latitude in radians
#         lon: np array (1 x 1)
#             longitude in radians
#         alt: np array (1 x 1)
#             altitude above Earth in km
#     """
#     #Reshape inputs if we are only converting one set of coordinates
#     # if eci_vec.size == 3:
#     #     eci_vec = np.array(eci_vec).reshape((3,1))
#     #     j2000_vec = (np.array(j2000_vec)).item()#.reshape(1,1)
#     # elif eci_vec.shape[0] != 3:
#     #     eci_vec = np.array(eci_vec).T
#
#     # lla_out = np.zeros(eci_vec.shape)
#     #ts = api.load.timescale()
#
#     #Get time and create ECI Position
#     # eci_pos = eci_vec
#     jd = j2000_to_tai(j2000_vec)
#     pos_time = ts.tai_jd(jd)
#     pos = units.Distance(km=eci_vec)
#     ecef = positionlib.Geocentric(pos.km, t=pos_time)
#     #Convert to ECEF + get lat/lon/alt
#     latlon = api.wgs84.latlon_of(ecef)
#     # print(latlon)
#     # lla_out[0,:] = latlon[0].radians
#     # lla_out[1,:] = latlon[1].radians
#     # lla_out[2,:] = api.wgs84.height_of(eci).km
#
#     # for i in range(0, eci_vec.shape[0]):
#     #     #Get time and create ECI Position
#     #     eci_pos = eci_vec[i]
#     #     jd = time_utils.j2000_to_ut1(j2000_vec[i])
#     #     pos_time = ts.ut1_jd(jd)
#     #     pos = units.Distance(km=[eci_pos[0], eci_pos[1], eci_pos[2]])
#     #     eci = positionlib.Geocentric(pos.au, t=pos_time)
#     #     #Convert to ECEF + get lat/lon/alt
#     #     latlon = api.wgs84.latlon_of(eci)
#     #     lla_out[i, 0] = latlon[0].radians
#     #     lla_out[i, 1] = latlon[1].radians
#     #     lla_out[i, 2] = api.wgs84.height_of(eci).km
#     return latlon[0].radians,latlon[1].radians, api.wgs84.height_of(ecef).km
#
# def enu_to_ecef(enu_vec, lat_vec, lon_vec):
#     """
#     This function converts a *VECTOR* in East-North-Up coordinates to ECEF, given
#     the ENU vector and the lat/lon of the reference point on the WGS84 ellipsoid
#     for the ENU coordinates. Lat/lon must be in *RADIANS*. Verified (slightly half
#     assed though). Vectorized.
#
#     Source:
#     https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU
#
#     Parameters
#     -----------
#         enu_vec: np array (3 or N x 3)
#             vector in ENU coordinates (ex. B_ENU)
#         lat: float or N np array
#             latitude of reference pt, radians
#         lon: float or N np array
#             longitude of reference pt, radians
#
#     Returns
#     --------
#         ecef_vec: np array (3 x 1 or N x 3)
#             vector transformed into ECEF coordinate system
#     """
#     enu_vec = np.atleast_2d(enu_vec)
#     # ecef_out = np.zeros((3,int(enu_vec.size/3)))
#     # if enu_vec.size == 3:
#     #     lat_vec, lon_vec = np.atleast_1d(np.squeeze(np.array(lat_vec))), np.atleast_1d(np.squeeze(np.array(lon_vec)))
#     #     enu_vec = enu_vec.reshape((3,1))
#     if enu_vec.shape[1] != 3:
#         breakpoint()
#         enu_vec = enu_vec.T
#
#     #Construct rotation matrix + rotate each enu vec
#     Rz = rotz_q_p_pi2(np.expand_dims(lon_vec,axis=(1,2)))
#     Rx = rotx_pi2_m_q(np.expand_dims(lat_vec,axis=(1,2)))
#     ecef_out = np.squeeze(Rz@Rx@np.expand_dims(enu_vec,axis=2))
#     return ecef_out
#
#
# def latlon_from_eci(eci_pos, j2000):
#     """
#     This function gets latitude and longitude from a position in ECI. Verified.
#     Depends on skylib.
#
#     Parameters
#     -----------
#         eci_pos: np array (3 x 1)
#             position in ECI
#         j2000: float
#             j2000 time
#
#     Returns
#     ---------
#         lat: [float]
#             latitude in radians
#         lon: [float]
#             longitude in radians
#     """
#     lat, lon, _ = eci_to_lla(eci_pos, j2000)
#     return lat, lon
#
# def long_from_ecef(p_ecef):
#     """
#     Deprecated but verified. This function gets a longitude from an ECEF position.
#
#     Parameters
#     -----------
#         p_ecef: array-like
#             length-3 position in ECEF coordinates, km
#
#     Returns
#     ----------
#         long: list
#             length-1 list containing a float representing longitude, in radians
#     """
#     # p_ecef = dimcheck1D(p_ecef,0,3)
#     long = np.arctan2(p_ecef[1],p_ecef[0])
#     return long
#
# def rad_from_position(pos):
#     """
#     Gets elevation (or "radius" of circle with center at Earth center and point
#     at position) of a position in ECEF or ECI coordinates. Verified.
#
#     Parameters
#     -----------
#         pos: array-like
#             length-3 position in ECI or ECEF coordinates, km
#     Returns
#     --------
#         rad: float
#             distance to Earth center, km
#     """
#     # pos = dimcheck1D(pos,0,3)
#     # rad = np.sqrt(pos[0]**2 + pos[1]**2+pos[2]**2)
#     return norm(pos)
