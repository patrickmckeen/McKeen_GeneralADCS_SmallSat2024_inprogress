from .orbital_state import *
from .orbit import *
import ppigrf
from sat_ADCS_helpers import *
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation
import astropy.units as u
from skyfield import api, elementslib,framelib
import numdifftools as nd
import poliastro
import poliastro.bodies as bodies
import poliastro.twobody.orbit as paorb
import time
"""
tests needed


orbit
------
B for orbits
geocentric to ecef for orbits
ecef to eci for orbits


orbital state
------------
xxxxvalus of geocentric,sf_pos---??
frame conversion for other vectors
xxxxcopying
eclipse checking
better tests on geocentric, magnetic field, sun, etc.
"""


def test_orbital_state():
    os = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]))
    assert np.all(os.R == np.array([7e3,0,0]))
    assert np.all(os.V == np.array([0,8,0]))
    assert os.J2000 == 0.22

    os2 = Orbital_State(0.22,np.array([7000,0,0]),np.array([0,8,0]),np.array([1e10,0,0]),1e-9*np.array([0,5,0]),12,np.array([0,1,2,3]),np.array([0,2,12,15]))
    assert np.all(os2.R == np.array([7e3,0,0]))
    assert np.all(os2.V == np.array([0,8,0]))
    assert os2.J2000 == 0.22
    assert np.all(os2.S == np.array([1e10,0,0]))
    assert np.all(os2.B == np.array([0,5e-9,0]))
    assert os2.rho == 12
    assert np.all(os2.altrange == np.arange(0,4,1))
    assert np.all(os2.rhovsalt == np.array([0,2,12,15]))

    os3 = Orbital_State(0.22,np.array([200+R_e,0,0]),np.array([0,8,0]))
    assert np.isclose(os3.rho, 2.90e-10)

    os3 = Orbital_State(0.22,np.array([50+R_e,0,0]),np.array([0,8,0]))
    assert np.isclose(os3.rho, 0.5*(1.2+5.69e-7))

    os3 = Orbital_State(0.22,np.array([100+R_e,0,0]),np.array([0,8,0]),altrange=np.array([0,100,200,300]),rhovsalt=np.array([0,2,12,15]))
    assert np.isclose(os3.rho, 2)

    os3 = Orbital_State(0.22,np.array([50+R_e,0,0]),np.array([0,8,0]),altrange=np.array([0,100,200,300]),rhovsalt=np.array([0,2,12,15]))
    assert np.isclose(os3.rho, 1)

    dt = datetime(2023,4,29,0,0,0)
    t = Time(dt,format='datetime',scale='utc')
    os3 = Orbital_State((t.tai.jyear-2000)/100,np.array([250+R_e,0,0]),np.array([0,8,0]),altrange=np.array([0,100,200,300]),rhovsalt=np.array([0,2,12,15]))
    assert np.isclose(os3.TAI,t.tai.jd)
    print(os3.TAI-t.tai.jd)
    # tai = j2000_to_tai(os3.J2000)
    t2 = Time(os3.TAI,format='jd',scale = 'tai')
    # print(t.tai.jyear,t2.tai.jyear)
    v = get_sun(t)
    assert np.all(np.isclose(os3.S,v.cartesian.get_xyz().to_value('km')))

    pos = api.wgs84.latlon(40, -117, 4e5)
    ap_loc = EarthLocation(lat=40, lon=-117, height=4e5)
    eci_ap = ap_loc.get_gcrs(t)
    ecef_ap = ap_loc.get_itrs(t)

    B_enu_ref = np.array([3575.6,17579.0, -37186.0])
    os3 = Orbital_State((t.tai.jyear-2000)/100,eci_ap.cartesian.get_xyz().to_value('km'),np.array([0,8,0]),altrange=np.array([0,100,200,300]),rhovsalt=np.array([0,2,12,15]),calc_all = True)

    assert np.all(np.isclose(os3.R,eci_ap.cartesian.get_xyz().to_value('km')))
    assert dt == os3.datetime
    assert np.all(np.isclose(os3.ECEF,ecef_ap.cartesian.get_xyz().to_value('km')))
    assert np.all(np.isclose(np.mod(os3.LLA[1],np.pi*2),np.mod(-117*math.pi/180.0,np.pi*2)))
    assert np.all(np.isclose(os3.LLA[2],4e2))
    assert np.all(np.isclose(np.mod(os3.LLA[0],np.pi*2),np.mod(40*math.pi/180.0,np.pi*2)))
    assert np.all(np.isclose(B_enu_ref*1e-9,os3.eci_to_enu(os3.B)))

def test_orbit_dynamics_noJ2():
    n_n = random_n_unit_vec(3)
    ecc = np.random.uniform(0.1,0.4)
    rp = np.random.uniform(6800,9000)
    a = rp/(1-ecc)
    vp = math.sqrt(mu_e*(1+ecc)/(1-ecc)/a)
    h = rp*vp
    th = np.random.uniform(0,2*np.pi)
    r = h*h/(mu_e*(1+ecc*np.cos(th)))
    v = math.sqrt(mu_e*(2/r - 1/a))

    n_rp = normalize(np.cross(random_n_unit_vec(3),n_n))
    n_vp = normalize(np.cross(n_n,n_rp))

    rvec = r*(n_rp*np.cos(th)+n_vp*np.sin(th))
    # vvec = v*(n_rp*np.cos(th)+n_vp*np.sin(th)) + r*thd*(-n_rp*np.sin(th)+n_vp*np.cos(th))
    cphi = (1+ecc*np.cos(th))/math.sqrt(1+ecc*ecc+2*ecc*np.cos(th))
    phi = np.arccos(cphi)*np.sign(np.sin(th));
    psi = th + 0.5*math.pi - phi;
    vvec = v*(n_rp*np.cos(psi) + n_vp*np.sin(psi))
    os = Orbital_State(0.22,rvec,vvec)
    rd,vd = os.orbit_dynamics(J2_on=False)
    assert np.allclose(rd,vvec)
    assert np.isclose(norm(np.cross(rvec,vd)),0)
    assert np.isclose(mu_e/r/r, norm(vd))
    assert np.dot(vd,rvec) < 0
    assert np.isclose(np.dot(vd,n_n), 0)

@pytest.mark.slow
def test_orbit_rk4():
    n_n = random_n_unit_vec(3)
    ecc = np.random.uniform(0.1,0.7)
    rp = np.random.uniform(6800,30000)
    a = rp/(1-ecc)
    vp = math.sqrt(mu_e*(1+ecc)/(1-ecc)/a)
    h = rp*vp
    th = np.random.uniform(0,2*np.pi)
    r = h*h/(mu_e*(1+ecc*np.cos(th)))
    v = math.sqrt(mu_e*(2/r - 1/a))

    n_rp = normalize(np.cross(random_n_unit_vec(3),n_n))
    n_vp = normalize(np.cross(n_n,n_rp))

    rvec = r*(n_rp*np.cos(th)+n_vp*np.sin(th))
    # vvec = v*(n_rp*np.cos(th)+n_vp*np.sin(th)) + r*thd*(-n_rp*np.sin(th)+n_vp*np.cos(th))
    cphi = (1+ecc*np.cos(th))/math.sqrt(1+ecc*ecc+2*ecc*np.cos(th))
    phi = np.arccos(cphi)*np.sign(np.sin(th));
    psi = th + 0.5*math.pi - phi;
    vvec = v*(n_rp*np.cos(psi) + n_vp*np.sin(psi))
    os = Orbital_State(0.22,rvec,vvec)
    dt = 30.0
    osp1 = os.orbit_rk4(dt, J2_on=True, rk4_on=True,calc_B = False,calc_S = False)
    orbtest = paorb.Orbit.from_vectors(bodies.Earth,u.Quantity(rvec,unit=u.km),u.Quantity(vvec,unit=u.km/u.s), Time(os.TAI,format='jd',scale = 'tai'))
    orbtest2 = orbtest.propagate(u.Quantity(dt,unit=u.s))
    print(osp1.V)
    print(orbtest2.v.value)
    assert np.all(np.isclose(osp1.R,orbtest2.r.value))
    assert np.all(np.isclose(osp1.V,orbtest2.v.value,atol=1e-3))

def test_averaging_orbital_state():
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    B_ECI = random_n_unit_vec(3)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(69,80)#6900,7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,10)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    B_ECI0 = np.copy(B_ECI)
    V_ECI0 = np.copy(V_ECI)
    R_ECI0 = np.copy(R_ECI)
    S_ECI0 = np.copy(S_ECI)
    rho = np.random.uniform(1e-12,1e-6)
    rho0 = np.copy(rho)
    u = random_n_unit_vec(9)*np.random.uniform(0.3,2.1)
    os = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,rho = rho,S=S_ECI)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(69,80)#6900,7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6,10)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12,1e14)
    B_ECI = random_n_unit_vec(3)
    rho = np.random.uniform(1e-12,1e-6)
    assert np.all(os.B==B_ECI0)
    assert np.all(os.R==R_ECI0)
    assert np.all(os.S==S_ECI0)
    assert os.rho == rho0
    assert np.all(os.V==V_ECI0)
    os2 = Orbital_State(0.22,R_ECI,V_ECI,B=B_ECI,rho = rho,S=S_ECI)
    avg = os.average(os2,0.4)
    assert np.all(avg.R==(0.6*os.R + 0.4*os2.R))
    assert np.all(avg.S==(0.6*os.S + 0.4*os2.S))
    assert np.all(avg.B==(0.6*os.B + 0.4*os2.B))
    assert np.all(avg.V==(0.6*os.V + 0.4*os2.V))
    assert avg.rho==(0.6*os.rho + 0.4*os2.rho)
    assert np.all(os.B==B_ECI0)
    assert np.all(os.R==R_ECI0)
    assert np.all(os.S==S_ECI0)
    assert os.rho == rho0
    assert np.all(os.V==V_ECI0)

def test_orbit_dyn_Jacobians():
    for k in range(1):
        pos = 7000*random_n_unit_vec(3)
        vel = 8*normalize(np.cross(random_n_unit_vec(3),pos))
        os = Orbital_State(0.22,pos,vel)
        osp1 = os.orbit_dynamics(J2_on=True)

        rfun = lambda c: Orbital_State(0.22,np.array([c[0],c[1],c[2]]),np.array([c[3],c[4],c[5]])).orbit_dynamics(J2_on=True)[0]
        vfun = lambda c: Orbital_State(0.22,np.array([c[0],c[1],c[2]]),np.array([c[3],c[4],c[5]])).orbit_dynamics(J2_on=True)[1]

        Jrfun = nd.Jacobian(rfun)(pos.flatten().tolist()+vel.flatten().tolist())
        Jvfun = nd.Jacobian(vfun)(pos.flatten().tolist()+vel.flatten().tolist())
        wholething = os.orbit_dynamics_jacobians(J2_on=True, conc=True)

        Jrfuntest = np.array(Jrfun)#.reshape((6,3))
        Jvfuntest = np.array(Jvfun)#.reshape((6,3))
        np.set_printoptions(precision=3)
        assert np.allclose(Jrfuntest.T,wholething[:,0:3])
        assert np.allclose(Jvfuntest.T,wholething[:,3:6])

def test_orbit_rk4_Jacobians():
    for k in range(1):
        pos = 7000*random_n_unit_vec(3)
        vel = 8*normalize(np.cross(random_n_unit_vec(3),pos))
        os = Orbital_State(0.22,pos,vel)
        dt = 1.0

        rfun = lambda c: Orbital_State(0.22,np.array([c[0],c[1],c[2]]),np.array([c[3],c[4],c[5]])).orbit_rk4(dt, J2_on=True, rk4_on=True,calc_B = False,calc_S = False).R
        vfun = lambda c: Orbital_State(0.22,np.array([c[0],c[1],c[2]]),np.array([c[3],c[4],c[5]])).orbit_rk4(dt, J2_on=True, rk4_on=True,calc_B = False,calc_S = False).V

        Jrfun = nd.Jacobian(rfun)(pos.flatten().tolist()+vel.flatten().tolist())
        Jvfun = nd.Jacobian(vfun)(pos.flatten().tolist()+vel.flatten().tolist())
        osp1 = os.orbit_rk4(dt, J2_on=True, rk4_on=True,calc_B = False,calc_S = False)
        [drd__dr,drd__dv,dvd__dr,dvd__dv] = os.orbit_rk4_jacobians(dt, J2_on=True, rk4_on=True)
        # print(rfun(pos.flatten().tolist()+vel.flatten().tolist()),vfun(pos.flatten().tolist()+vel.flatten().tolist()))

        Jrfuntest = np.array(Jrfun)#.reshape((6,3))
        Jvfuntest = np.array(Jvfun)#.reshape((6,3))
        np.set_printoptions(precision=3)
        assert np.allclose(Jrfuntest.T,np.vstack([drd__dr,drd__dv]))
        assert np.allclose(Jvfuntest.T,np.vstack([dvd__dr,dvd__dv]))

def test_orbit_creation():
    n_n = random_n_unit_vec(3)
    ecc = np.random.uniform(0.1,0.4)
    rp = np.random.uniform(6800,9000)
    a = rp/(1-ecc)
    vp = math.sqrt(mu_e*(1+ecc)/(1-ecc)/a)
    h = rp*vp
    th = np.random.uniform(0,2*np.pi)
    r = h*h/(mu_e*(1+ecc*np.cos(th)))
    v = math.sqrt(mu_e*(2/r - 1/a))

    n_rp = normalize(np.cross(random_n_unit_vec(3),n_n))
    n_vp = normalize(np.cross(n_n,n_rp))

    rvec = r*(n_rp*np.cos(th)+n_vp*np.sin(th))
    # vvec = v*(n_rp*np.cos(th)+n_vp*np.sin(th)) + r*thd*(-n_rp*np.sin(th)+n_vp*np.cos(th))
    cphi = (1+ecc*np.cos(th))/math.sqrt(1+ecc*ecc+2*ecc*np.cos(th))
    phi = np.arccos(cphi)*np.sign(np.sin(th));
    psi = th + 0.5*math.pi - phi;
    vvec = v*(n_rp*np.cos(psi) + n_vp*np.sin(psi))
    zero_time = 0.22
    os = Orbital_State(zero_time,rvec,vvec)
    os0 = os.copy()
    orb0 = Orbit(os)
    dt = 3600
    N_dt = 24*5
    midpt = zero_time+sec2cent*dt*24*2
    print('with all')
    t0 = time.process_time()
    osw = Orbital_State(zero_time,rvec,vvec,timing=True)
    t1 = time.process_time()
    print('overall')
    print(t1-t0)

    print('\nwithout all')
    t0 = time.process_time()
    osw = Orbital_State(zero_time,rvec,vvec,calc_all = False,timing=True,)
    t1 = time.process_time()
    print('overall')
    print(t1-t0)

    print('\norb1')
    t0 = time.process_time()
    orb1 = Orbit(os,zero_time+sec2cent*dt*N_dt,dt,each_calc_B = True,all_data = True)
    t1 = time.process_time()
    print(t1-t0)
    orb2 = Orbit([*orb1.states.values()])
    print('\norb3')
    t0 = time.process_time()
    orb3 = Orbit(os,zero_time+sec2cent*dt*N_dt,dt,each_calc_B = False,all_data = False)
    t1 = time.process_time()
    print(t1-t0)
    # assert 1 == 0
    orbs = [orb1,orb2,orb3]

    #check times are right
    assert orb0.times == zero_time
    assert np.allclose(orb1.times , [zero_time+sec2cent*dt*j for j in range(N_dt + 1)])
    assert np.allclose(orb2.times , [zero_time+sec2cent*dt*j for j in range(N_dt+ 1)])
    assert np.allclose(orb3.times , [zero_time+sec2cent*dt*j for j in range(N_dt+ 1)])

    #check orbit states are right--J2000,R,V
    assert np.allclose(os0.R,orb0.states[zero_time].R)
    assert np.allclose(os0.V,orb0.states[zero_time].V)
    assert np.allclose(os0.J2000,orb0.states[zero_time].J2000)
    assert np.allclose(zero_time,orb0.states[zero_time].J2000)
    orb0statelist = [*orb0.states.values()]
    assert len(orb0statelist)==1
    assert np.allclose(os0.R,[j.R for j in orb0statelist][0])
    assert np.allclose(os0.V,[j.V for j in orb0statelist][0])
    assert np.allclose(os0.J2000,[j.J2000 for j in orb0statelist][0])

    ind = 0
    for t in orb1.times:
        assert orb1.states[t].J2000 == t
        assert t == zero_time+sec2cent*dt*ind
        if ind>0:
            os = orb1.states[t]
            test_os = prev_os.orbit_rk4(dt, J2_on=True, rk4_on=True,calc_B = False,calc_S = False)
            assert np.isclose( test_os.J2000 , os.J2000, rtol=1e-15, atol=1e-15)
            assert np.allclose(test_os.R , os.R)
            assert np.allclose(test_os.V , os.V)
        prev_os = orb1.states[t]
        ind += 1

    ind = 0
    for t in orb2.times:
        assert orb2.states[t].J2000 == t
        assert t == zero_time+sec2cent*dt*ind
        if ind>0:
            os = orb2.states[t]
            test_os = prev_os.orbit_rk4(dt, J2_on=True, rk4_on=True,calc_B = False,calc_S = False)
            assert np.isclose( test_os.J2000 , os.J2000, rtol=1e-15, atol=1e-15)
            assert np.allclose(test_os.R , os.R)
            assert np.allclose(test_os.V , os.V)
        prev_os = orb2.states[t]
        ind += 1

    ind = 0
    for t in orb3.times:
        assert orb3.states[t].J2000 == t
        assert t == zero_time+sec2cent*dt*ind
        if ind>0:
            os = orb3.states[t]
            test_os = prev_os.orbit_rk4(dt, J2_on=True, rk4_on=True,calc_B = False,calc_S = False)
            assert np.isclose( test_os.J2000 , os.J2000, rtol=1e-15, atol=1e-15)
            assert np.allclose(test_os.R , os.R)
            assert np.allclose(test_os.V , os.V)
        prev_os = orb3.states[t]
        ind += 1

    #check B,S,etc are correct as if generated from R,V

    assert np.allclose(os0.B,orb0statelist[0].B)
    assert np.allclose(os0.S,orb0statelist[0].S)
    assert np.allclose(os0.rho,orb0statelist[0].rho)
    assert np.allclose(os0.altrange,orb0statelist[0].altrange)
    assert np.allclose(os0.rhovsalt,orb0statelist[0].rhovsalt)

    for t in orb2.times:
        assert np.allclose(orb2.states[t].B,orb2.states[t].get_b_eci())
        assert np.allclose(orb2.states[t].S,orb2.states[t].get_sun_eci())
        osbackup = Orbital_State(t,orb2.states[t].R,orb2.states[t].V)
        assert np.allclose(orb2.states[t].rho,osbackup.rho)
        assert np.allclose(orb2.states[t].B,osbackup.get_b_eci())
        assert np.allclose(orb2.states[t].S,osbackup.get_sun_eci())
        assert np.allclose(orb2.states[t].TAI,osbackup.TAI)
        assert np.allclose(orb2.states[t].LLA,osbackup.LLA)
        assert np.allclose(orb2.states[t].ECEF,osbackup.ECEF)
        assert orb2.states[t].datetime == osbackup.datetime
        assert np.allclose(orb2.states[t].geocentric,osbackup.geocentric)
        assert np.allclose(orb2.states[t].ECI2ENUmat,osbackup.ECI2ENUmat)
        # assert np.allclose(orb2.states[t].sf_pos,osbackup.sf_pos)
        # assert np.allclose(orb2.states[t].sf_geo_pos,osbackup.sf_geo_pos)

    vecs2 = orb2.get_vecs()
    assert len(vecs2[0]) == 121
    assert len(vecs2[1]) == 121
    assert len(vecs2[2]) == 121
    assert len(vecs2[3]) == 121
    assert len(vecs2[4]) == 121
    assert np.all([np.allclose(vecs2[0][j],orb2.states[orb2.times[j]].R) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[1][j],orb2.states[orb2.times[j]].V) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[2][j],orb2.states[orb2.times[j]].B) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[3][j],orb2.states[orb2.times[j]].S) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[4][j],orb2.states[orb2.times[j]].rho) for j in range(len(vecs2[0]))])

    for t in orb1.times:
        assert np.allclose(orb1.states[t].B,orb1.states[t].get_b_eci())
        assert np.allclose(orb1.states[t].S,orb1.states[t].get_sun_eci())
        osbackup = Orbital_State(t,orb1.states[t].R,orb1.states[t].V)
        assert np.allclose(orb1.states[t].rho,osbackup.rho)
        assert np.allclose(orb1.states[t].B,osbackup.get_b_eci())
        assert np.allclose(orb1.states[t].S,osbackup.get_sun_eci())
        assert np.allclose(orb1.states[t].TAI,osbackup.TAI)
        assert np.allclose(orb1.states[t].LLA,osbackup.LLA)
        assert np.allclose(orb1.states[t].ECEF,osbackup.ECEF)
        assert orb1.states[t].datetime == osbackup.datetime
        assert np.allclose(orb1.states[t].geocentric,osbackup.geocentric)
        assert np.allclose(orb1.states[t].ECI2ENUmat,osbackup.ECI2ENUmat)
        # assert np.allclose(orb1.states[t].sf_pos,osbackup.sf_pos)
        # assert np.allclose(orb1.states[t].sf_geo_pos,osbackup.sf_geo_pos)

    vecs1 = orb1.get_vecs()
    assert len(vecs1[0]) == 121
    assert len(vecs1[1]) == 121
    assert len(vecs1[2]) == 121
    assert len(vecs1[3]) == 121
    assert len(vecs1[4]) == 121
    assert np.all([np.allclose(vecs1[0][j],orb1.states[orb1.times[j]].R) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[1][j],orb1.states[orb1.times[j]].V) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[2][j],orb1.states[orb1.times[j]].B) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[3][j],orb1.states[orb1.times[j]].S) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[4][j],orb1.states[orb1.times[j]].rho) for j in range(len(vecs1[0]))])


    for t in orb3.times:
        print(t)
        print(orb3.states[t].B)
        print(orb3.states[t].get_b_eci())
        assert np.allclose(orb3.states[t].B,orb3.states[t].get_b_eci())
        assert np.allclose(orb3.states[t].S,orb3.states[t].get_sun_eci())
        osbackup = Orbital_State(t,orb3.states[t].R,orb3.states[t].V)
        assert np.allclose(orb3.states[t].rho,osbackup.rho)
        assert np.allclose(orb3.states[t].B,osbackup.get_b_eci())
        assert np.allclose(orb3.states[t].S,osbackup.get_sun_eci())
        assert np.allclose(orb3.states[t].TAI,osbackup.TAI)
        # assert np.allclose(orb3.states[t].LLA,osbackup.LLA)
        assert np.allclose(orb3.states[t].ECEF,osbackup.ECEF)
        assert orb3.states[t].datetime == osbackup.datetime
        assert np.allclose(orb3.states[t].geocentric,osbackup.geocentric)
        # assert np.allclose(orb3.states[t].ECI2ENUmat,osbackup.ECI2ENUmat)
        # assert np.allclose(orb3.states[t].sf_pos,osbackup.sf_pos)
        # assert np.allclose(orb3.states[t].sf_geo_pos,osbackup.sf_geo_pos)

    vecs2 = orb3.get_vecs()
    assert len(vecs2[0]) == 121
    assert len(vecs2[1]) == 121
    assert len(vecs2[2]) == 121
    assert len(vecs2[3]) == 121
    assert len(vecs2[4]) == 121
    assert np.all([np.allclose(vecs2[0][j],orb3.states[orb3.times[j]].R) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[1][j],orb3.states[orb3.times[j]].V) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[2][j],orb3.states[orb3.times[j]].B) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[3][j],orb3.states[orb3.times[j]].S) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[4][j],orb3.states[orb3.times[j]].rho) for j in range(len(vecs2[0]))])

    #checking states
    #beginning match
    test0 = orb0.get_os(zero_time)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    test0 = orb0.next_state(zero_time)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    test0 = orb0.next_state(os0)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    test1 = orb1.get_os(zero_time)
    assert test1.J2000 == zero_time
    assert np.allclose(os0.R,test1.R)
    assert np.allclose(os0.V,test1.V)
    assert np.allclose(os0.B,test1.B)
    assert np.allclose(os0.S,test1.S)
    assert np.allclose(os0.rho,test1.rho)

    test1 = orb1.next_state(zero_time)
    assert test1.J2000 == zero_time
    assert np.allclose(os0.R,test1.R)
    assert np.allclose(os0.V,test1.V)
    assert np.allclose(os0.B,test1.B)
    assert np.allclose(os0.S,test1.S)
    assert np.allclose(os0.rho,test1.rho)

    test1 = orb1.next_state(os0)
    assert test1.J2000 == zero_time
    assert np.allclose(os0.R,test1.R)
    assert np.allclose(os0.V,test1.V)
    assert np.allclose(os0.B,test1.B)
    assert np.allclose(os0.S,test1.S)
    assert np.allclose(os0.rho,test1.rho)

    test2 = orb2.get_os(zero_time)
    assert test2.J2000 == zero_time
    assert np.allclose(os0.R,test2.R)
    assert np.allclose(os0.V,test2.V)
    assert np.allclose(os0.B,test2.B)
    assert np.allclose(os0.S,test2.S)
    assert np.allclose(os0.rho,test2.rho)

    test2 = orb2.next_state(zero_time)
    assert test2.J2000 == zero_time
    assert np.allclose(os0.R,test2.R)
    assert np.allclose(os0.V,test2.V)
    assert np.allclose(os0.B,test2.B)
    assert np.allclose(os0.S,test2.S)
    assert np.allclose(os0.rho,test2.rho)

    test2 = orb2.next_state(os0)
    assert test2.J2000 == zero_time
    assert np.allclose(os0.R,test2.R)
    assert np.allclose(os0.V,test2.V)
    assert np.allclose(os0.B,test2.B)
    assert np.allclose(os0.S,test2.S)
    assert np.allclose(os0.rho,test2.rho)



    test3 = orb3.get_os(zero_time)
    assert test3.J2000 == zero_time
    assert np.allclose(os0.R,test3.R)
    assert np.allclose(os0.V,test3.V)
    assert np.allclose(os0.B,test3.B)
    assert np.allclose(os0.S,test3.S)
    assert np.allclose(os0.rho,test3.rho)

    test3 = orb3.next_state(zero_time)
    assert test3.J2000 == zero_time
    assert np.allclose(os0.R,test3.R)
    assert np.allclose(os0.V,test3.V)
    assert np.allclose(os0.B,test3.B)
    assert np.allclose(os0.S,test3.S)
    assert np.allclose(os0.rho,test3.rho)

    test3 = orb3.next_state(os0)
    assert test3.J2000 == zero_time
    assert np.allclose(os0.R,test3.R)
    assert np.allclose(os0.V,test3.V)
    assert np.allclose(os0.B,test3.B)
    assert np.allclose(os0.S,test3.S)
    assert np.allclose(os0.rho,test3.rho)


    #last match
    test0 = orb0.get_os(zero_time)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    test1 = orb1.get_os(zero_time+sec2cent*dt*N_dt)
    test1a = orb1.states[orb1.times[-1]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(zero_time+sec2cent*dt*N_dt)
    test1a = orb1.states[orb1.times[-1]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(orb1.states[orb1.times[-1]])
    test1a = orb1.states[orb1.times[-1]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test2 = orb2.get_os(zero_time+sec2cent*dt*N_dt)
    test2a = orb2.states[orb2.times[-1]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(zero_time+sec2cent*dt*N_dt)
    test2a = orb2.states[orb2.times[-1]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(orb2.states[orb2.times[-1]])
    test2a = orb2.states[orb2.times[-1]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)


    test3 = orb3.get_os(zero_time+sec2cent*dt*N_dt)
    test3a = orb3.states[orb3.times[-1]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(zero_time+sec2cent*dt*N_dt)
    test3a = orb3.states[orb3.times[-1]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(orb3.states[orb3.times[-1]])
    test3a = orb3.states[orb3.times[-1]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)


    #middle one
    test1 = orb1.get_os(zero_time+sec2cent*dt*24*2)
    test1a = orb1.states[orb1.times[24*2]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(zero_time+sec2cent*dt*24*2)
    test1a = orb1.states[orb1.times[24*2]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(orb1.states[orb1.times[24*2]])
    test1a = orb1.states[orb1.times[24*2]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test2 = orb2.get_os(zero_time+sec2cent*dt*24*2)
    test2a = orb2.states[orb2.times[24*2]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(zero_time+sec2cent*dt*24*2)
    test2a = orb2.states[orb2.times[24*2]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(orb2.states[orb2.times[24*2]])
    test2a = orb2.states[orb2.times[24*2]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test3 = orb3.get_os(zero_time+sec2cent*dt*24*2)
    test3a = orb3.states[orb3.times[24*2]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(zero_time+sec2cent*dt*24*2)
    test3a = orb3.states[orb3.times[24*2]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(orb3.states[orb3.times[24*2]])
    test3a = orb3.states[orb3.times[24*2]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    for j in orbs:
        ##before first
        with pytest.raises(ValueError) as excinfo:
            j.get_os(0.21)
        assert "(too far in past)" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            j.next_state(0.21)
        assert "(too far in past)" in str(excinfo.value)


        ##after last
        with pytest.raises(ValueError) as excinfo:
            j.get_os(0.23)
        assert "(too far in future)" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            j.next_state(0.23)
        assert "(too far in future)" in str(excinfo.value)

    #between 2
    for k in range(5):
        ind = int(np.round(np.random.uniform(0,1)*(len(orb1.times)-2)))
        loc = np.random.uniform(0,1)
        t_search = dt*loc*sec2cent + orb1.times[ind]
        os_a = orb1.get_os(orb1.times[ind])
        os_b = orb1.get_os(orb1.times[ind+1])
        ostest = orb1.get_os(t_search)
        ostest0 = ostest.copy()

        assert np.isclose(ostest.J2000 , t_search, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_a.R*(1-loc) + os_b.R*loc)
        assert np.allclose(ostest.V,os_a.V*(1-loc) + os_b.V*loc)
        assert np.allclose(ostest.B,os_a.B*(1-loc) + os_b.B*loc)
        assert np.allclose(ostest.S,os_a.S*(1-loc) + os_b.S*loc)
        assert np.allclose(ostest.rho,os_a.rho*(1-loc) + os_b.rho*loc)

        ostest = orb1.next_state(t_search)
        print(k)
        print(ostest.J2000)
        print(os_b.J2000)
        assert np.isclose(ostest.J2000 , os_b.J2000, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_b.R)
        assert np.allclose(ostest.V,os_b.V)
        assert np.allclose(ostest.B,os_b.B)
        assert np.allclose(ostest.S,os_b.S)
        assert np.allclose(ostest.rho,os_b.rho)

        ostest = orb1.next_state(ostest0)
        assert np.isclose(ostest.J2000 , os_b.J2000, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_b.R)
        assert np.allclose(ostest.V,os_b.V)
        assert np.allclose(ostest.B,os_b.B)
        assert np.allclose(ostest.S,os_b.S)
        assert np.allclose(ostest.rho,os_b.rho)

        ind = int(np.round(np.random.uniform(0,1)*(len(orb2.times)-2)))
        loc = np.random.uniform(0,1)
        t_search = dt*loc*sec2cent + orb2.times[ind]
        os_a = orb2.get_os(orb2.times[ind])
        os_b = orb2.get_os(orb2.times[ind+1])
        ostest = orb2.get_os(t_search)
        ostest0 = ostest.copy()

        assert np.isclose(ostest.J2000 , t_search, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_a.R*(1-loc) + os_b.R*loc)
        assert np.allclose(ostest.V,os_a.V*(1-loc) + os_b.V*loc)
        assert np.allclose(ostest.B,os_a.B*(1-loc) + os_b.B*loc)
        assert np.allclose(ostest.S,os_a.S*(1-loc) + os_b.S*loc)
        assert np.allclose(ostest.rho,os_a.rho*(1-loc) + os_b.rho*loc)

        ostest = orb2.next_state(t_search)
        assert np.isclose(ostest.J2000 , os_b.J2000, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_b.R)
        assert np.allclose(ostest.V,os_b.V)
        assert np.allclose(ostest.B,os_b.B)
        assert np.allclose(ostest.S,os_b.S)
        assert np.allclose(ostest.rho,os_b.rho)

        ostest = orb2.next_state(ostest0)
        assert np.isclose(ostest.J2000 , os_b.J2000, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_b.R)
        assert np.allclose(ostest.V,os_b.V)
        assert np.allclose(ostest.B,os_b.B)
        assert np.allclose(ostest.S,os_b.S)
        assert np.allclose(ostest.rho,os_b.rho)

        ind = int(np.round(np.random.uniform(0,1)*(len(orb3.times)-2)))
        loc = np.random.uniform(0,1)
        t_search = dt*loc*sec2cent + orb3.times[ind]
        os_a = orb3.get_os(orb3.times[ind])
        os_b = orb3.get_os(orb3.times[ind+1])
        ostest = orb3.get_os(t_search)
        ostest0 = ostest.copy()

        assert np.isclose(ostest.J2000 , t_search, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_a.R*(1-loc) + os_b.R*loc)
        assert np.allclose(ostest.V,os_a.V*(1-loc) + os_b.V*loc)
        assert np.allclose(ostest.B,os_a.B*(1-loc) + os_b.B*loc)
        assert np.allclose(ostest.S,os_a.S*(1-loc) + os_b.S*loc)
        assert np.allclose(ostest.rho,os_a.rho*(1-loc) + os_b.rho*loc)

        ostest = orb3.next_state(t_search)
        assert np.isclose(ostest.J2000 , os_b.J2000, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_b.R)
        assert np.allclose(ostest.V,os_b.V)
        assert np.allclose(ostest.B,os_b.B)
        assert np.allclose(ostest.S,os_b.S)
        assert np.allclose(ostest.rho,os_b.rho)

        ostest = orb3.next_state(ostest0)
        assert np.isclose(ostest.J2000 , os_b.J2000, rtol=1e-15, atol=1e-15)
        assert np.allclose(ostest.R,os_b.R)
        assert np.allclose(ostest.V,os_b.V)
        assert np.allclose(ostest.B,os_b.B)
        assert np.allclose(ostest.S,os_b.S)
        assert np.allclose(ostest.rho,os_b.rho)

    #min/max times
    assert orb0.min_time() == zero_time
    assert orb0.max_time() == zero_time
    assert orb1.min_time() == zero_time
    assert orb1.max_time() == zero_time+N_dt*dt*sec2cent
    assert orb2.min_time() == zero_time
    assert orb2.max_time() == zero_time+N_dt*dt*sec2cent
    assert orb3.min_time() == zero_time
    assert orb3.max_time() == zero_time+N_dt*dt*sec2cent

    #time in time
    assert not orb0.time_in_span(0.221)
    assert not orb0.time_in_span(zero_time+1e-16)
    for j in orbs:
        assert not j.time_in_span(0.1)
        assert not j.time_in_span(0.219)
        assert not j.time_in_span(1)
        assert not j.time_in_span(0.225)
        assert j.time_in_span(zero_time)
        assert not j.time_in_span(zero_time+N_dt*dt*sec2cent+1e-16)
        assert not j.time_in_span(zero_time-1e-16)



    assert orb1.time_in_span(zero_time+N_dt*dt*sec2cent)
    assert orb2.time_in_span(zero_time+N_dt*dt*sec2cent)
    assert orb3.time_in_span(zero_time+N_dt*dt*sec2cent)
    assert orb1.time_in_span(zero_time+1*24*dt*sec2cent)
    assert orb2.time_in_span(zero_time+1*24*dt*sec2cent)
    assert orb3.time_in_span(zero_time+1*24*dt*sec2cent)
    for k in range(10):
        assert orb1.time_in_span(zero_time+np.random.uniform(0,N_dt*dt*sec2cent))
        assert orb2.time_in_span(zero_time+np.random.uniform(0,N_dt*dt*sec2cent))
        assert orb3.time_in_span(0.22+np.random.uniform(0,N_dt*dt*sec2cent))


    #new orbit from times
    indlist = [12,13,16,25]
    tlist = orb1.times[indlist]
    orbt = orb1.new_orbit_from_times(tlist)
    orb1states = [*orb1.states.values()]
    for j in range(len(indlist)):
        ind = indlist[j]
        tt = tlist[j]

        assert orbt.states[tt].J2000 == tt
        assert orb1.states[tt].J2000 == orbt.states[tt].J2000
        assert orb1states[ind].J2000 == orbt.states[tt].J2000
        assert np.allclose(orb1.states[tt].R , orbt.states[tt].R)
        assert np.allclose(orb1states[ind].R , orbt.states[tt].R)
        assert np.allclose(orb1.states[tt].rho , orbt.states[tt].rho)
        assert np.allclose(orb1states[ind].rho , orbt.states[tt].rho)
        assert np.allclose(orb1.states[tt].V , orbt.states[tt].V)
        assert np.allclose(orb1states[ind].V , orbt.states[tt].V)
        assert np.allclose(orb1.states[tt].B , orbt.states[tt].B)
        assert np.allclose(orb1states[ind].B , orbt.states[tt].B)
        assert np.allclose(orb1.states[tt].S , orbt.states[tt].S)
        assert np.allclose(orb1states[ind].S , orbt.states[tt].S)


    tlist = orb2.times[indlist]
    orbt = orb2.new_orbit_from_times(tlist)
    orb2states = [*orb2.states.values()]
    for j in range(len(indlist)):
        ind = indlist[j]
        tt = tlist[j]

        assert orbt.states[tt].J2000 == tt
        assert orb2.states[tt].J2000 == orbt.states[tt].J2000
        assert orb2states[ind].J2000 == orbt.states[tt].J2000
        assert np.allclose(orb2.states[tt].R , orbt.states[tt].R)
        assert np.allclose(orb2states[ind].R , orbt.states[tt].R)
        assert np.allclose(orb2.states[tt].rho , orbt.states[tt].rho)
        assert np.allclose(orb2states[ind].rho , orbt.states[tt].rho)
        assert np.allclose(orb2.states[tt].V , orbt.states[tt].V)
        assert np.allclose(orb2states[ind].V , orbt.states[tt].V)
        assert np.allclose(orb2.states[tt].B , orbt.states[tt].B)
        assert np.allclose(orb2states[ind].B , orbt.states[tt].B)
        assert np.allclose(orb2.states[tt].S , orbt.states[tt].S)
        assert np.allclose(orb2states[ind].S , orbt.states[tt].S)


    tlist = orb3.times[indlist]
    orbt = orb3.new_orbit_from_times(tlist)
    orb3states = [*orb3.states.values()]
    for j in range(len(indlist)):
        ind = indlist[j]
        tt = tlist[j]

        assert orbt.states[tt].J2000 == tt
        assert orb3.states[tt].J2000 == orbt.states[tt].J2000
        assert orb3states[ind].J2000 == orbt.states[tt].J2000
        assert np.allclose(orb3.states[tt].R , orbt.states[tt].R)
        assert np.allclose(orb3states[ind].R , orbt.states[tt].R)
        assert np.allclose(orb3.states[tt].rho , orbt.states[tt].rho)
        assert np.allclose(orb3states[ind].rho , orbt.states[tt].rho)
        assert np.allclose(orb3.states[tt].V , orbt.states[tt].V)
        assert np.allclose(orb3states[ind].V , orbt.states[tt].V)
        assert np.allclose(orb3.states[tt].B , orbt.states[tt].B)
        assert np.allclose(orb3states[ind].B , orbt.states[tt].B)
        assert np.allclose(orb3.states[tt].S , orbt.states[tt].S)
        assert np.allclose(orb3states[ind].S , orbt.states[tt].S)


    #get from time ranges

    indlist = [12,13,14,15]
    tlist = orb1.times[indlist]
    orbt = orb1.get_range(orb1.times[12],orb1.times[15])
    orb1states = [*orb1.states.values()]
    for j in range(len(indlist)):
        ind = indlist[j]
        tt = tlist[j]

        assert orbt.states[tt].J2000 == tt
        assert orb1.states[tt].J2000 == orbt.states[tt].J2000
        assert orb1states[ind].J2000 == orbt.states[tt].J2000
        assert np.allclose(orb1.states[tt].R , orbt.states[tt].R)
        assert np.allclose(orb1states[ind].R , orbt.states[tt].R)
        assert np.allclose(orb1.states[tt].rho , orbt.states[tt].rho)
        assert np.allclose(orb1states[ind].rho , orbt.states[tt].rho)
        assert np.allclose(orb1.states[tt].V , orbt.states[tt].V)
        assert np.allclose(orb1states[ind].V , orbt.states[tt].V)
        assert np.allclose(orb1.states[tt].B , orbt.states[tt].B)
        assert np.allclose(orb1states[ind].B , orbt.states[tt].B)
        assert np.allclose(orb1.states[tt].S , orbt.states[tt].S)
        assert np.allclose(orb1states[ind].S , orbt.states[tt].S)


    tlist = orb2.times[indlist]
    orbt = orb2.get_range(orb2.times[12],orb2.times[15])
    orb2states = [*orb2.states.values()]
    for j in range(len(indlist)):
        ind = indlist[j]
        tt = tlist[j]

        assert orbt.states[tt].J2000 == tt
        assert orb2.states[tt].J2000 == orbt.states[tt].J2000
        assert orb2states[ind].J2000 == orbt.states[tt].J2000
        assert np.allclose(orb2.states[tt].R , orbt.states[tt].R)
        assert np.allclose(orb2states[ind].R , orbt.states[tt].R)
        assert np.allclose(orb2.states[tt].rho , orbt.states[tt].rho)
        assert np.allclose(orb2states[ind].rho , orbt.states[tt].rho)
        assert np.allclose(orb2.states[tt].V , orbt.states[tt].V)
        assert np.allclose(orb2states[ind].V , orbt.states[tt].V)
        assert np.allclose(orb2.states[tt].B , orbt.states[tt].B)
        assert np.allclose(orb2states[ind].B , orbt.states[tt].B)
        assert np.allclose(orb2.states[tt].S , orbt.states[tt].S)
        assert np.allclose(orb2states[ind].S , orbt.states[tt].S)



    tlist = orb3.times[indlist]
    orbt = orb3.get_range(orb3.times[12],orb3.times[15])
    orb3states = [*orb3.states.values()]
    for j in range(len(indlist)):
        ind = indlist[j]
        tt = tlist[j]

        assert orbt.states[tt].J2000 == tt
        assert orb3.states[tt].J2000 == orbt.states[tt].J2000
        assert orb3states[ind].J2000 == orbt.states[tt].J2000
        assert np.allclose(orb3.states[tt].R , orbt.states[tt].R)
        assert np.allclose(orb3states[ind].R , orbt.states[tt].R)
        assert np.allclose(orb3.states[tt].rho , orbt.states[tt].rho)
        assert np.allclose(orb3states[ind].rho , orbt.states[tt].rho)
        assert np.allclose(orb3.states[tt].V , orbt.states[tt].V)
        assert np.allclose(orb3states[ind].V , orbt.states[tt].V)
        assert np.allclose(orb3.states[tt].B , orbt.states[tt].B)
        assert np.allclose(orb3states[ind].B , orbt.states[tt].B)
        assert np.allclose(orb3.states[tt].S , orbt.states[tt].S)
        assert np.allclose(orb3states[ind].S , orbt.states[tt].S)


    inda = int(np.round(np.random.uniform(0,1)*(len(orb1.times)-2)))
    indb = int(np.round(np.random.uniform(0,1)*(len(orb1.times)-2)))
    ind0 = min(inda,indb)
    ind1 = max(inda,indb)
    loc0 = np.random.uniform(0,1)
    loc1 = np.random.uniform(0,1)
    t_0 = dt*loc0*sec2cent + orb1.times[ind0]
    t_1 = dt*loc1*sec2cent + orb1.times[ind1]
    dt = 253
    tlist = np.concatenate([np.arange(t_0,t_1,dt*sec2cent),[t_1]])
    oslist = [orb1.get_os(j) for j in tlist]
    ostest = orb1.get_range(t_0,t_1,dt)
    assert np.allclose(ostest.min_time(),t_0, rtol=1e-15, atol=1e-15)#ostest.min_time() == t_0
    assert np.allclose(ostest.max_time(),t_1, rtol=1e-15, atol=1e-15)#ostest.max_time() == t_1
    assert len(ostest.times) == len(oslist)
    assert np.allclose(ostest.times,[j.J2000 for j in oslist], rtol=1e-15, atol=1e-15)
    assert np.allclose(ostest.get_vecs()[0],[j.R for j in oslist])
    assert np.allclose(ostest.get_vecs()[1],[j.V for j in oslist])
    assert np.allclose(ostest.get_vecs()[2],[j.B for j in oslist])
    assert np.allclose(ostest.get_vecs()[3],[j.S for j in oslist])
    assert np.allclose(ostest.get_vecs()[4],[j.rho for j in oslist])

    inda = int(np.round(np.random.uniform(0,1)*(len(orb2.times)-2)))
    indb = int(np.round(np.random.uniform(0,1)*(len(orb2.times)-2)))
    ind0 = min(inda,indb)
    ind1 = max(inda,indb)
    loc0 = np.random.uniform(0,1)
    loc1 = np.random.uniform(0,1)
    t_0 = dt*loc0*sec2cent + orb2.times[ind0]
    t_1 = dt*loc1*sec2cent + orb2.times[ind1]
    dt = 253
    tlist = np.concatenate([np.arange(t_0,t_1,dt*sec2cent),[t_1]])
    oslist = [orb2.get_os(j) for j in tlist]
    ostest = orb2.get_range(t_0,t_1,dt)
    assert np.isclose(ostest.min_time() , t_0, rtol=1e-15, atol=1e-15)
    assert np.isclose( ostest.max_time(), t_1, rtol=1e-15, atol=1e-15)
    assert len(ostest.times) == len(oslist)
    assert np.allclose(ostest.times,[j.J2000 for j in oslist], rtol=1e-15, atol=1e-15)
    assert np.allclose(ostest.get_vecs()[0],[j.R for j in oslist])
    assert np.allclose(ostest.get_vecs()[1],[j.V for j in oslist])
    assert np.allclose(ostest.get_vecs()[2],[j.B for j in oslist])
    assert np.allclose(ostest.get_vecs()[3],[j.S for j in oslist])
    assert np.allclose(ostest.get_vecs()[4],[j.rho for j in oslist])

    inda = int(np.round(np.random.uniform(0,1)*(len(orb3.times)-2)))
    indb = int(np.round(np.random.uniform(0,1)*(len(orb3.times)-2)))
    ind0 = min(inda,indb)
    ind1 = max(inda,indb)
    loc0 = np.random.uniform(0,1)
    loc1 = np.random.uniform(0,1)
    t_0 = dt*loc0*sec2cent + orb3.times[ind0]
    t_1 = dt*loc1*sec2cent + orb3.times[ind1]
    dt = 253
    tlist = np.concatenate([np.arange(t_0,t_1,dt*sec2cent),[t_1]])
    oslist = [orb3.get_os(j) for j in tlist]
    ostest = orb3.get_range(t_0,t_1,dt)
    assert np.isclose(ostest.min_time() , t_0, rtol=1e-15, atol=1e-15)
    assert np.isclose( ostest.max_time(), t_1, rtol=1e-15, atol=1e-15)
    assert len(ostest.times) == len(oslist)
    assert np.allclose(ostest.times,[j.J2000 for j in oslist], rtol=1e-15, atol=1e-15)
    assert np.allclose(ostest.get_vecs()[0],[j.R for j in oslist])
    assert np.allclose(ostest.get_vecs()[1],[j.V for j in oslist])
    assert np.allclose(ostest.get_vecs()[2],[j.B for j in oslist])
    assert np.allclose(ostest.get_vecs()[3],[j.S for j in oslist])
    assert np.allclose(ostest.get_vecs()[4],[j.rho for j in oslist])
