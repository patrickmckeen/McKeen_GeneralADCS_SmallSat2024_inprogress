from sat_ADCS_estimation import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
from common_sats import *
import numpy as np
import math
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import odeint, solve_ivp, RK45
from scipy.stats import chi2
from scipy.special import erfinv
import matplotlib.pyplot as plt
import time
import pickle


def test_ukf_super_basic_quat_not_vec():

    t0 = 0
    tf = 60*10
    tlim00 = 60
    tlim0 = 10*60
    tlim1 = 20*60
    tlim2 = 30*60
    dt = 1
    np.set_printoptions(precision=8)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 0.5
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8
    gyro_std = 0.0001# 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]


    sun_eff = 1.0
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in range(3)]

    dists = []
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = np.array([0.001,0,0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,np.array([0,10,0]),np.array([1,0,0]),B=np.array([0.01,0,0]),S=np.array([0,11,0]))
    dur = int((tf-t0)/dt)+10
    orbs = [os0]*(dur+10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j*dt*sec2cent
    orb = Orbit(orbs)
    #
    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]


    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in range(3)]

    dists_est = []
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(7)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(0.01)**2.0,np.eye(3)*10)
    # int_cov =  block_diag(np.eye(3)*(1e-4)**2.0,1e-4*np.eye(3))
    int_cov = np.zeros((6,6))

    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1.0
    est.kap = 0.0
    est.bet = 2.0#-1.0#2.0
    est.include_sens_noise_separately = False
    est.include_int_noise_separately = False
    est.use_cross_term = True
    est.scale_nonseparate_adds = False

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        _,extra = est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])


        #find control
        kw = 0.1
        ka = 0.01
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e10*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,1,0,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = state
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
        if True:

            errvec = est_state-state
            errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),5),errvec[7:]])
            print('err vec ',errvec )
            mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
            # print('cov err vec ', coverr)
            # print('cov err vec ',norm(coverr))
            print('mahalanobis_dist ',math.sqrt(mahalanobis_dist2))
            print('prob ',chi2.pdf(mahalanobis_dist2,6))
            cd = chi2.cdf(mahalanobis_dist2,6)
            print('cdf ',cd)
            print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))

            N = 5
            r_coord = np.linspace(0,2,3)
            sph_coord = [np.linspace(0,math.pi,N) for j in range(6-2)]+[np.linspace(0,2*math.pi,N)]
            se1,se2,se3,se4,se5 = np.meshgrid(*sph_coord)
            sph_mesh = np.stack([se1,se2,se3,se4,se5])
            cart_mesh = np.moveaxis(ct(sph_mesh),0,5)

            eiga0,eige0 = np.linalg.eigh(extra.cov0)
            eiga1,eige1 = np.linalg.eigh(extra.cov1)
            eiga2,eige2 = np.linalg.eigh(extra.cov2)
            r = math.sqrt(math.sqrt(mahalanobis_dist2))

            zpoint = extra.mean2
            md0 = est.states_diff(extra.mean0,zpoint)
            md1 = est.states_diff(extra.mean1,zpoint)
            md2 = est.states_diff(extra.mean2,zpoint)

            ell0 = cart_mesh@eige0@np.diagflat(np.sqrt(eiga0))*r*r + md0
            ell1 = cart_mesh@eige1.T@np.diagflat(np.sqrt(eiga1))*r*r + md1
            ell2 = cart_mesh@eige2.T@np.diagflat(np.sqrt(eiga2))*r*r + md2
            sigd0 = est.states_diff(extra.sig0,extra.mean0)*r + md0
            sigd1 = est.states_diff(extra.sig1,extra.mean1)*r + md1

            # breakpoint()

            sigd = np.stack([sigd0,sigd1])

            truth = est.states_diff(state,zpoint)

            sortinds = np.argsort(np.abs(eiga2))
            print(eiga2)
            indX = sortinds[-1]
            axX = eige2[:,indX]
            indY = sortinds[-2]
            axY = eige2[:,indY]#*np.sqrt(eiga2[indX]/eiga2[indY])
            print(eiga2[indX])
            print(eiga2[indY])
            if ind == 1:
                f = plt.figure()
                ax = plt.subplot()

            # ax.cla()
            # breakpoint()
            if ind > 1:
                ax.relim(visible_only=True)
                ax.autoscale(enable=True, axis='both', tight=False)
                pe0.remove()
                pe0, = ax.plot(np.ravel(ell0@axX),np.ravel(ell0@axY), marker='.', color='k', linestyle='none',markersize=0.1)
                pe1.remove()
                pe1, = ax.plot(np.ravel(ell1@axX),np.ravel(ell1@axY), marker='.', color='r', linestyle='none',markersize=0.1)
                pe2.remove()
                pe2, = ax.plot(np.ravel(ell2@axX),np.ravel(ell2@axY), marker='.', color='g', linestyle='none',markersize=0.1)
                pm0.remove()
                pm0, = ax.plot(np.ravel(md0@axX),np.ravel(md0@axY), marker='*', color='k', linestyle='none')
                pm1.remove()
                pm1, = ax.plot(np.ravel(md1@axX),np.ravel(md1@axY), marker='*', color='r', linestyle='none')
                pm2.remove()
                pm2, = ax.plot(np.ravel(md2@axX),np.ravel(md2@axY), marker='*', color='g', linestyle='none')
                pt.remove()
                pt, = ax.plot(np.ravel(truth@axX),np.ravel(truth@axY), marker='*', color='b', linestyle='none')
                [k.remove() for k in psd]
                psd = ax.plot(sigd@axX,sigd@axY, marker='none', color='y', linestyle=':')
                ps0.remove()
                ps0, = ax.plot(np.ravel(sigd0@axX),np.ravel(sigd0@axY), marker='^', color='k', linestyle='none')
                # ps1.remove()
                ps1, = ax.plot(np.ravel(sigd1@axX),np.ravel(sigd1@axY), marker='^', color='r', linestyle='none')
            else:
                pe0, = ax.plot(np.ravel(ell0@axX),np.ravel(ell0@axY), marker='.', color='k', linestyle='none',markersize=0.1)
                pe1, = ax.plot(np.ravel(ell1@axX),np.ravel(ell1@axY), marker='.', color='r', linestyle='none',markersize=0.1)
                pe2, = ax.plot(np.ravel(ell2@axX),np.ravel(ell2@axY), marker='.', color='g', linestyle='none',markersize=0.1)
                pm0, = ax.plot(np.ravel(md0@axX),np.ravel(md0@axY), marker='*', color='k', linestyle='none')
                pm1, = ax.plot(np.ravel(md1@axX),np.ravel(md1@axY), marker='*', color='r', linestyle='none')
                pm2, = ax.plot(np.ravel(md2@axX),np.ravel(md2@axY), marker='*', color='g', linestyle='none')
                pt, = ax.plot(np.ravel(truth@axX),np.ravel(truth@axY), marker='*', color='b', linestyle='none')
                psd = ax.plot(sigd@axX,sigd@axY, marker='none', color='y', linestyle=':')
                ps0, = ax.plot(np.ravel(sigd0@axX),np.ravel(sigd0@axY), marker='^', color='k', linestyle='none')
                ps1, = ax.plot(np.ravel(sigd1@axX),np.ravel(sigd1@axY), marker='^', color='r', linestyle='none')

            # plt.tight_layout()
            plt.draw()
            plt.pause(0.00001)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    breakpoint()


def test_ukf_basic_quat_not_vec_w_dist():
    t0 = 0
    tf = 60*5
    tlim00 = 10
    tlim0 = 0.5*60
    tlim1 = 2*60
    tlim2 = 4*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 1.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8
    gyro_std = 0.0001# 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]


    sun_eff = 1.0
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,np.array([0,1e5,0]),np.array([1,0,0]),B=np.array([0.01,0,0]),S=np.array([0,1e5+1,0]),rho = 1e-7)
    dur = int((tf-t0)/dt)+10
    orbs = [os0]*(dur+10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j*dt*sec2cent
    orb = Orbit(orbs)
    #
    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]


    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(7)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(0.01)**2.0,np.eye(3)*3)
    int_cov =  block_diag(np.eye(3)*(1e-4)**2.0,1e-4*np.eye(3))

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1e-3
    est.kap = 0
    est.bet = 2.0#-1.0#2.0

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e10*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 10
            ka = 1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 10
            ka = 1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 10
            ka = 1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = state
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")
    breakpoint()


def test_crassidis_ukf_crassidis_paper_case_1():
    t0 = 0
    tf = 60*30
    # tlim00 = 100
    # tlim0 = 5*60
    # tlim1 = 20*60
    # tlim2 = 40*60
    dt = 10
    np.set_printoptions(precision=3)

    #
    #real_sat
    acts =  []

    mtm_std = 50*1e-9
    gyro_std = 0.31623*1e-6#0.0001# 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
    gyro_bias0 = np.ones(3)*0.1*(math.pi/180)/3600
    gyro_bsr = 3.1623e-4 * 1e-6

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(gyro_bias0,j),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]


    # drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
    #                 [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
    #                 [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
    #                 [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
    #                 [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
    #                 [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    # drag = Drag_Disturbance(drag_faces)
    # gg = GG_Disturbance()
    dists = []#[drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))*500
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros, disturbances = dists)
    w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = zeroquat#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,np.array([0,7100,0]),np.array([8,0,0]))
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    #
    acts_est =  []

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False,sample_time = 10) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True,sample_time = 10) for j in unitvecs]


    # drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
    #                 [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
    #                 [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
    #                 [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
    #                 [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
    #                 [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    # drag_est = Drag_Disturbance(drag_faces)
    # gg_est = GG_Disturbance()
    dists_est = []
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))*500
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(10)
    estimate[3:7] = q0
    estimate[0:3] = np.nan
    estimate[7:10] = 0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(0.5*math.pi/180.0)**2.0,np.eye(3)*(0.2*math.pi/180.0/3600.0)**2.0)
    int_cov =  5*block_diag(np.nan*np.eye(3),np.eye(3)*(gyro_std**2.0-(1/6)*gyro_bsr**2.0*10**2.0),gyro_bsr**2.0*np.eye(3))

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 10)
    est.al = 1e-3
    est.kap = 0
    est.bet = 2.0#-1.0#2.0

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])
        print(autovar[6:9])

        # #find control
        # if t<tlim00:
        #     ud = np.zeros(3)
        # elif t<tlim0:
        #     #bdot
        #     Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
        #     ud = -1e10*(-np.cross(est.use_state.val[0:3],Bbody))
        # elif t<tlim1:
        #     #PID to zeroquat
        #     wdes = np.zeros(3)
        #     qdes = zeroquat
        #     q = est.use_state.val[3:7]
        #     w =  est.use_state.val[0:3]
        #     w_err =w-wdes
        #     q_err = quat_mult(quat_inv(qdes),q)
        #     kw = 10
        #     ka = 1
        #     nB2 = norm(orbt.B)
        #     Bbody = rot_mat(q).T@orbt.B
        #     ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        #
        # elif t<tlim2:
        #     #PID to [0,1,0,0]
        #     wdes = np.zeros(3)
        #     qdes = np.array([0,0,1,0])
        #     q = est.use_state.val[3:7]
        #     w =  est.use_state.val[0:3]
        #     w_err =w-wdes
        #     q_err = quat_mult(quat_inv(qdes),q)
        #     kw = 10
        #     ka = 1
        #     nB2 = norm(orbt.B)
        #     Bbody = rot_mat(q).T@orbt.B
        #     ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        # else:
        #     #PID to zeroquat
        #     wdes = np.zeros(3)
        #     qdes = zeroquat
        #     q = est.use_state.val[3:7]
        #     w =  est.use_state.val[0:3]
        #     w_err =w-wdes
        #     q_err = quat_mult(quat_inv(qdes),q)
        #     kw = 10
        #     ka = 1
        #     nB2 = norm(orbt.B)
        #     Bbody = rot_mat(q).T@orbt.B
        #     ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        #
        # offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = np.zeros(0)
        # print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()


def test_crassidis_ukf_crassidis_paper_case_2():
    # np.random.seed(1)
    t0 = 0
    tf = 60*60*3
    dt = 10
    np.set_printoptions(precision=3)

    #
    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True)

    w0 = np.array([0,2*math.pi/(60*90),0])#TRMM CONOPS
    # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
    os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7100,7.3*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
    q0 = two_vec_to_quat(-os0.R,os0.V,unitvecs[2],unitvecs[0]) #based on TRMM conops: q0: local +z is global -R. local +x is global +V
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    #
    est_sat = create_Crassidis_UKF_sat(real=False)

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[7:10] = 0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(0.2*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 5*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*10)**2.0),np.diagflat(gyro_bsr**2.0))

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 10)
    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    wdot_func = lambda w,tau: (tau-np.cross(np.array(w),np.array(w)@real_sat.J))@real_sat.invJ
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(np.sqrt(autovar[0:3]))
        print((180.0/math.pi)*4*np.arctan(np.sqrt(autovar[3:6])/4.0))
        print((180.0/math.pi)*np.sqrt(autovar[6:9]))

        control = np.zeros(0)
        # print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        # wp = state[0:3] + dt*(-(np.cross(state[0:3],state[0:3]@real_sat.J)@real_sat.invJ))
        # qp = quat_mult(state[3:7],rot_exp(dt*state[0:3]))
        # state = np.concatenate([wp,qp])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-10, atol=1e-15,max_step = 1.0)#,jac = ivp_jac)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_mrp(quatdiff[j,:])/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "av in deg/s")
    plot_the_thing(mrpdiff,title = "mrp diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(np.log10(angdiff),title = " log ang diff in deg")
    # plot_the_thing(autocov_hist,title = "cov_hist")
    # plot_the_thing(state_hist[:,3:7],title = "quat hist")
    # plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()


def test_crassidis_ukf_crassidis_paper_case_3():
    # np.random.seed(1)
    t0 = 0
    tf = 60*60*8
    # tlim00 = 100
    # tlim0 = 5*60
    # tlim1 = 20*60
    # tlim2 = 40*60
    dt = 10
    np.set_printoptions(precision=3)

    #
    #real_sat
    real_sat = create_Crassidis_UKF_sat(real=True)

    w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    # q0 = zeroquat#np.array([math.cos(0.5*35*math.pi/180),math.sin(0.5*35*math.pi/180),0,0])#zeroquat#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0: local +z is global -R. local +x is global +V
    # os0 = Orbital_State(0.22-1*sec2cent,np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])*7100,np.array([8,0,0]))
    os0 = Orbital_State(0.22-1*sec2cent,np.array([1,0,0])*7100,7.3*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)]))
    q0 = two_vec_to_quat(-os0.R,os0.V,unitvecs[2],unitvecs[0]) #based on TRMM conops
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    #
    est_sat = create_Crassidis_UKF_sat(real=False)

    estimate = np.zeros(10)
    estimate[3:7] = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    estimate[0:3] = np.nan
    estimate[7:10] = 0
    estimate[8] = 20*(math.pi/180.0)/3600.0
    cov_estimate = block_diag(np.eye(3)*np.nan,np.eye(3)*(50*math.pi/180.0)**2.0,np.eye(3)*(20.0*(math.pi/180.0)/3600.0)**2.0)
    gyro_bsr = np.array([j.bias_std_rate for j in est_sat.sensors if isinstance(j,Gyro)])
    gyro_std = np.array([j.std for j in est_sat.sensors if isinstance(j,Gyro)])
    int_cov = 5*block_diag(np.nan*np.eye(3),np.diagflat(gyro_std**2.0-(1/6)*(gyro_bsr*10)**2.0),np.diagflat(gyro_bsr**2.0))

    est = Crassidis_UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 10)
    est.lam = 0
    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(0)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    wdot_func = lambda w,tau: (tau-np.cross(np.array(w),np.array(w)@real_sat.J))@real_sat.invJ
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(np.sqrt(autovar[0:3]))
        print((180.0/math.pi)*4*np.arctan(np.sqrt(autovar[3:6])/4.0))
        print((180.0/math.pi)*np.sqrt(autovar[6:9]))

        control = np.zeros(0)
        # print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)

        # wp = state[0:3] + dt*(-(np.cross(state[0:3],state[0:3]@real_sat.J)@real_sat.invJ))
        # qp = quat_mult(state[3:7],rot_exp(dt*state[0:3]))
        # state = np.concatenate([wp,qp])

        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt),state, method="RK45", args=(control, prev_os,orbt), rtol=1e-10, atol=1e-15,max_step = 1)#,jac = ivp_jac)
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        # vin = state.copy()
        # N = 10
        # ti0 = 0
        # ti1 = dt/N
        # for j in range(N):
        #     out = solve_ivp(real_sat.dynamics_for_solver, (ti0, ti1),vin, method="RK45", args=(control, prev_os,orbt), rtol=1e-10, atol=1e-15)#,jac = ivp_jac)
        #     # # print('step done')
        #     vin = out.y[:,-1]
        #     vin[3:7] = normalize(vin[3:7])
        #     ti0 = ti1
        #     ti1 += dt/N
        # state = vin.copy()

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    quatdiff = np.stack([quat_mult(quat_inv(est_state_hist[j,3:7]),state_hist[j,3:7]) for j in range(state_hist.shape[0])])
    mrpdiff = (180/np.pi)*4*np.arctan(np.stack([quat_to_mrp(quatdiff[j,:])/2 for j in range(quatdiff.shape[0])]))
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(state_hist[:,0:3]*180.0/math.pi,title = "av in deg/s")
    plot_the_thing(mrpdiff,title = "mrp diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(np.log10(angdiff),title = " log ang diff in deg")
    # plot_the_thing(autocov_hist,title = "cov_hist")
    # plot_the_thing(state_hist[:,3:7],title = "quat hist")
    # plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")
    breakpoint()


def test_crassidis_ukf_basic_quat_not_vec_w_dist():
    t0 = 0
    tf = 60*5
    tlim00 = 10
    tlim0 = 0.5*60
    tlim1 = 2*60
    tlim2 = 4*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 1.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8
    gyro_std = 0.0001# 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]


    sun_eff = 1.0
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,np.array([0,1e5,0]),np.array([1,0,0]),B=np.array([0.01,0,0]),S=np.array([0,1e5+1,0]),rho = 1e-7)
    dur = int((tf-t0)/dt)+10
    orbs = [os0]*(dur+10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j*dt*sec2cent
    orb = Orbit(orbs)
    #
    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]


    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(7)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(0.01)**2.0,np.eye(3)*3)
    int_cov =  block_diag(np.eye(3)*(1e-4)**2.0,1e-4*np.eye(3))

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1e-3
    est.kap = 0
    est.bet = 2.0#-1.0#2.0

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e10*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 10
            ka = 1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 10
            ka = 1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 10
            ka = 1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = state
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")
    breakpoint()


def test_srukf_basic_quat_not_vec_w_dist_w_real_orbit():
    # np.random.seed(1)
    t0 = 0
    tf = 60*10
    tlim00 = 5
    tlim0 = 0.5*60
    tlim1 = 2*60
    tlim2 = 4*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 10.0
    mtq_std = 0.1#0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0,scale = 5e2) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0,scale = 1) for j in unitvecs]


    sun_eff = 1.0
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0,scale = 1e-1) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 =  random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180#np.zeros(3)#np.array([0,2*math.pi/(60*90),0])#
    q0 =  random_n_unit_vec(4)# np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#zeroquat#
    try:
        with open("myorb", "rb") as fp:   #unPickling
            orb = pickle.load(fp)
    except:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False,scale = 5e2) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False,scale = 1) for j in unitvecs]
    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False,scale = 1e-1) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(7)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180)**2.0,np.eye(3)*10)
    int_cov =  block_diag(np.eye(3)*(1e-5)**2.0,np.eye(3)*1e-20)
    # int_cov = np.zeros((6,6))

    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1.0
    est.kap = 0
    est.bet = 2#5#2.0#2.0#-1.0#2.0
    est.include_sens_noise_separately = False
    est.include_int_noise_separately = False
    est.use_cross_term = True
    est.scale_nonseparate_adds = False

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)


    kw = 10
    ka = 0.1

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        exact_sens = real_sat.noiseless_sensor_values(state,real_vecs)
        _,extra = est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])


        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print('ctrl ',control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = state
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
        if True:

            errvec = est_state-state
            errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),5),errvec[7:]])
            # print('cov err vec ', coverr)
            # print('cov err vec ',norm(coverr))
            mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
            print('mahalanobis_dist ',math.sqrt(mahalanobis_dist2))
            print('prob ',chi2.pdf(mahalanobis_dist2,6))
            cd = chi2.cdf(mahalanobis_dist2,6)
            print('cdf ',cd)
            print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))

            msize = 0.5
            show0 = False
            show1 = True
            show2 = True
            r_coord = np.linspace(0,2,3)

            eiga0,eige0 = np.linalg.eigh(extra.cov0)
            eiga1,eige1 = np.linalg.eigh(extra.cov1)
            eiga2,eige2 = np.linalg.eigh(extra.cov2)
            ellr = 7#math.sqrt(math.sqrt(mahalanobis_dist2))
            sigr = 1
            if ind == 1:
                f, (ax, ax2) = plt.subplots(1, 2)

                N = 8
                state_sph_coord = [np.linspace(0,math.pi,N) for j in range(6-2)]+[np.linspace(0,2*math.pi,N)]
                se1,se2,se3,se4,se5 = np.meshgrid(*state_sph_coord)
                state_sph_mesh = np.stack([se1,se2,se3,se4,se5])
                state_cart_mesh = np.moveaxis(ct(state_sph_mesh),0,5)

            zpoint = extra.mean2
            md0 = est.states_diff(extra.mean0,zpoint)
            md1 = est.states_diff(extra.mean1,zpoint)
            md2 = est.states_diff(extra.mean2,zpoint)
            s2 = est.states_diff(extra.sens_state,zpoint)

            ell0 = state_cart_mesh@np.diagflat(np.sqrt(eiga0))@eige0.T*ellr + md0
            ell1 = state_cart_mesh@np.diagflat(np.sqrt(eiga1))@eige1.T*ellr + md1
            ell2 = state_cart_mesh@np.diagflat(np.sqrt(eiga2))@eige2.T*ellr + md2
            sigd0 =  est.states_diff(extra.sig0,extra.mean0)*sigr + md0
            sigd1 = est.states_diff(extra.sig1,extra.mean1)*sigr + md1

            # breakpoint()

            sigd = np.stack([sigd0,sigd1])

            truth = 0.01*est.states_diff(state,zpoint)

            sortinds = np.argsort(np.abs(errvec.T@eige2@np.diagflat(1/np.sqrt(eiga2))))
            print('eigs ',eiga2)
            indX = sortinds[-1]
            axX = eige2[:,indX]
            # axX = eige2[:,indX]*0
            # axX[4] = 1
            wtaxX = eiga2[indX]
            indY = sortinds[-2]
            axY = eige2[:,indY]#*(eiga2[indX]/eiga2[indY])
            # axY = np.zeros(6)#eige2[:,indY]#*(eiga2[indX]/eiga2[indY])
            # axY[1] = 1
            wtaxY = eiga2[indY]
            print('wt of axX ',wtaxX)
            print('axis of X ',axX)
            print('frac of X ',np.round(axX**2.0/np.amax(axX**2.0),2))
            # print(np.sum(np.diagflat(np.sqrt(eiga2))@eige2.T,axis=0))
            # matt = np.diagflat(np.sqrt(eiga2))@eige2.T
            # print(sum([matt[j:j+1,:] for j in range(matt.shape[0])]))
            print('vec frac that is X ',np.round(((axX*math.sqrt(wtaxX))**2.0)/(np.sum((np.diagflat(np.sqrt(eiga2))@eige2.T)**2.0,axis=0)),3))
            print('wt of axY ',wtaxY)
            print('axis of Y ',axY)
            print('frac of Y ',np.round(axY**2.0/np.amax(axY**2.0),2))
            print('vec frac that is Y ',np.round(((axY*math.sqrt(wtaxY))**2.0)/(np.sum((np.diagflat(np.sqrt(eiga2))@eige2.T)**2.0,axis=0)),3))
            print('err vec ',errvec )
            print('err vec X,Y ',errvec@axX,errvec@axY )
            # ax.cla()
            # breakpoint()
            if ind > 1:
                ax.relim(visible_only=True)
                ax.autoscale(enable=True, axis='both', tight=False)
                if show0:
                    pe0.remove()
                    pe0, = ax.plot(np.ravel(ell0@axX),np.ravel(ell0@axY), marker='.', color='k', linestyle='none',markersize=msize)
                    ps0.remove()
                    ps0, = ax.plot(np.ravel(sigd0@axX),np.ravel(sigd0@axY), marker='^', color='k', linestyle='none')
                    pm0.remove()
                    pm0, = ax.plot(np.ravel(md0@axX),np.ravel(md0@axY), marker='s', color='k', linestyle='none',markersize=4,markeredgecolor = 'k')
                if show1:
                    pe1.remove()
                    pe1, = ax.plot(np.ravel(ell1@axX),np.ravel(ell1@axY), marker='.', color='r', linestyle='none',markersize=msize)
                    ps1.remove()
                    ps1, = ax.plot(np.ravel(sigd1@axX),np.ravel(sigd1@axY), marker='^', color='r', linestyle='none')
                    pm1.remove()
                    pm1, = ax.plot(np.ravel(md1@axX),np.ravel(md1@axY), marker='s', color='r', linestyle='none',markersize=4,markeredgecolor = 'k')
                    if show0:
                        [k.remove() for k in psd]
                        psd = ax.plot(sigd@axX,sigd@axY, marker='none', color='y', linestyle=':')
                if show2:
                    pe2.remove()
                    pe2, = ax.plot(np.ravel(ell2@axX),np.ravel(ell2@axY), marker='.', color='g', linestyle='none',markersize=msize)
                    pm2.remove()
                    pm2, = ax.plot(np.ravel(md2@axX),np.ravel(md2@axY), marker='s', color='g', linestyle='none',markersize=4,markeredgecolor = 'k')
                    pt.remove()
                    pt, = ax.plot(np.ravel(truth@axX),np.ravel(truth@axY), marker='*', color='b', linestyle='none')
                    ps2.remove()
                    ps2, = ax.plot(np.ravel(s2@axX),np.ravel(s2@axY), marker='v', color='c', linestyle='none')
            else:
                if show0:
                    pe0, = ax.plot(np.ravel(ell0@axX),np.ravel(ell0@axY), marker='.', color='k', linestyle='none',markersize=msize)
                    pm0, = ax.plot(np.ravel(md0@axX),np.ravel(md0@axY), marker='s', color='k', linestyle='none',markersize=4,markeredgecolor = 'k')
                    ps0, = ax.plot(np.ravel(sigd0@axX),np.ravel(sigd0@axY), marker='^', color='k', linestyle='none')
                if show1:
                    pe1, = ax.plot(np.ravel(ell1@axX),np.ravel(ell1@axY), marker='.', color='r', linestyle='none',markersize=msize)
                    pm1, = ax.plot(np.ravel(md1@axX),np.ravel(md1@axY), marker='s', color='r', linestyle='none',markersize=4,markeredgecolor = 'k')
                    ps1, = ax.plot(np.ravel(sigd1@axX),np.ravel(sigd1@axY), marker='^', color='r', linestyle='none')
                    if show0:
                        psd = ax.plot(sigd@axX,sigd@axY, marker='none', color='y', linestyle=':')
                if show2:
                    pe2, = ax.plot(np.ravel(ell2@axX),np.ravel(ell2@axY), marker='.', color='g', linestyle='none',markersize=msize)
                    pt, = ax.plot(np.ravel(truth@axX),np.ravel(truth@axY), marker='*', color='b', linestyle='none')
                    pm2, = ax.plot(np.ravel(md2@axX),np.ravel(md2@axY), marker='s', color='g', linestyle='none',markersize=4,markeredgecolor = 'k')
                    ps2, = ax.plot(np.ravel(s2@axX),np.ravel(s2@axY), marker='v', color='c', linestyle='none')

            # plt.tight_layout()
            plt.draw()
            plt.pause(0.00001)
            # breakpoint()

            eiga,eige = np.linalg.eigh(extra.senscov)
            eigab,eigeb = np.linalg.eigh(real_sat.sensor_cov())
            ellr = 9#math.sqrt(math.sqrt(mahalanobis_dist2))
            sigr = 1
            if ind == 1:
                # f2 = plt.figure()
                # ax2 = plt.subplot()

                N = 5
                sens_sph_coord = [np.linspace(0,math.pi,N) for j in range(9-2)]+[np.linspace(0,2*math.pi,N)]
                se1,se2,se3,se4,se5,se6,se7,se8 = np.meshgrid(*sens_sph_coord)
                sens_sph_mesh = np.stack([se1,se2,se3,se4,se5,se6,se7,se8])
                sens_cart_mesh = np.moveaxis(ct(sens_sph_mesh),0,8)


            zpoint = exact_sens
            s1 = extra.sens1-zpoint
            s1a = extra.sens_of_state1-zpoint
            s2 = extra.sens_of_state2-zpoint
            sd = np.stack([s1a,s2])

            sig = extra.sens_sig-zpoint
            r = sens - zpoint
            r0 = exact_sens-zpoint

            ell = sens_cart_mesh@np.diagflat(np.sqrt(eiga))@eige.T*ellr + s1
            ellb = sens_cart_mesh@np.diagflat(np.sqrt(eigab))@eigeb.T*ellr + r0


            sortinds = np.argsort(np.abs((extra.sens_of_state2-exact_sens).T@eige@np.diagflat(1/np.sqrt(eiga))))
            indX = sortinds[-1]
            axX = eige[:,indX]#*0
            # axX[4] = 1
            wtaxX = eiga[indX]
            indY = sortinds[-2]
            # axY = eige[:,indY]#*(eiga2[indX]/eiga2[indY])
            axY = eige[:,indY]#*(eiga2[indX]/eiga2[indY])#*0
            # axY[1] = 1
            wtaxY = eiga[indY]
            print('wt of axX2 ',wtaxX)
            print('axis of X2 ',axX)
            print('frac of X2 ',np.round(axX**2.0/np.amax(axX**2.0),2))
            # print(np.sum(np.diagflat(np.sqrt(eiga2))@eige2.T,axis=0))
            # matt = np.diagflat(np.sqrt(eiga2))@eige2.T
            # print(sum([matt[j:j+1,:] for j in range(matt.shape[0])]))
            print('vec frac that is X2 ',np.round(((axX*math.sqrt(wtaxX))**2.0)/(np.sum((np.diagflat(np.sqrt(eiga))@eige.T)**2.0,axis=0)),3))
            print('wt of axY2 ',wtaxY)
            print('axis of Y2 ',axY)
            print('frac of Y2 ',np.round(axY**2.0/np.amax(axY**2.0),2))
            print('vec frac that is Y2 ',np.round(((axY*math.sqrt(wtaxY))**2.0)/(np.sum((np.diagflat(np.sqrt(eiga))@eige.T)**2.0,axis=0)),3))
            print('sens err vec ',(extra.sens_of_state2-exact_sens) )
            print('sens err vec X2,Y2 ',(extra.sens_of_state2-exact_sens)@axX,(extra.sens_of_state2-exact_sens)@axY )

            mahalanobis_dist2_sens = (sens-exact_sens).T@np.linalg.inv(real_sat.sensor_cov())@(sens-exact_sens)
            print('mahalanobis_dist sens ',math.sqrt(mahalanobis_dist2_sens))
            print('prob sens ',chi2.pdf(mahalanobis_dist2_sens,9))
            cd = chi2.cdf(mahalanobis_dist2_sens,9)
            print('cdf sens ',cd)
            print('std dev eq sens ',math.sqrt(2)*erfinv(2*cd-1))


            mahalanobis_dist2_sens1 = (extra.sens1-sens).T@np.linalg.inv(real_sat.sensor_cov())@(extra.sens1-sens)
            mahalanobis_dist2_sens2 = (extra.sens_of_state2-sens).T@np.linalg.inv(real_sat.sensor_cov())@(extra.sens_of_state2-sens)
            print('mahalanobis_dist sensd ',math.sqrt(mahalanobis_dist2_sens1),math.sqrt(mahalanobis_dist2_sens2))
            print('prob sensd ',chi2.pdf(mahalanobis_dist2_sens1,9),chi2.pdf(mahalanobis_dist2_sens2,9))
            cd1 = chi2.cdf(mahalanobis_dist2_sens1,9)
            cd2 = chi2.cdf(mahalanobis_dist2_sens2,9)
            print('cdf sensd ',cd1,cd2)
            print('std dev eq sensd ',math.sqrt(2)*erfinv(2*cd1-1),math.sqrt(2)*erfinv(2*cd2-1))

            # ax.cla()
            # breakpoint()
            if ind > 1:
                ax2.relim(visible_only=True)
                ax2.autoscale(enable=True, axis='both', tight=False)
                p2e.remove()
                p2e, = ax2.plot(np.ravel(ell@axX),np.ravel(ell@axY), marker='.', color='r', linestyle='none',markersize=msize)
                p2eb.remove()
                p2eb, = ax2.plot(np.ravel(ellb@axX),np.ravel(ellb@axY), marker='.', color='b', linestyle='none',markersize=msize)
                p2s1.remove()
                p2s1, = ax2.plot(np.ravel(s1@axX),np.ravel(s1@axY), marker='v', color='r', linestyle='none')
                p2s1a.remove()
                p2s1a, = ax2.plot(np.ravel(s1a@axX),np.ravel(s1a@axY), marker='o', color='r', linestyle='none')
                p2s2.remove()
                p2s2, = ax2.plot(np.ravel(s2@axX),np.ravel(s2@axY), marker='v', color='g', linestyle='none')
                [k.remove() for k in p2sd]
                p2sd = ax2.plot(sd@axX,sd@axY, marker='none', color='y', linestyle=':')
                p2sig.remove()
                p2sig, = ax2.plot(np.ravel(sig@axX),np.ravel(sig@axY), marker='^', color='r', linestyle='none')
                p2r.remove()
                p2r, = ax2.plot(np.ravel(r@axX),np.ravel(r@axY), marker='s', color='b', linestyle='none')
                p2r0.remove()
                p2r0, = ax2.plot(np.ravel(r0@axX),np.ravel(r0@axY), marker='*', color='b', linestyle='none')
            else:
                p2e, = ax2.plot(np.ravel(ell@axX),np.ravel(ell@axY), marker='.', color='r', linestyle='none',markersize=msize)
                p2s1, = ax2.plot(np.ravel(s1@axX),np.ravel(s1@axY), marker='v', color='r', linestyle='none')
                p2s1a, = ax2.plot(np.ravel(s1a@axX),np.ravel(s1a@axY), marker='o', color='r', linestyle='none')
                p2s2, = ax2.plot(np.ravel(s2@axX),np.ravel(s2@axY), marker='v', color='g', linestyle='none')
                p2sd = ax2.plot(sd@axX,sd@axY, marker='none', color='y', linestyle=':')
                p2sig, = ax2.plot(np.ravel(sig@axX),np.ravel(sig@axY), marker='^', color='r', linestyle='none')
                p2r, = ax2.plot(np.ravel(r@axX),np.ravel(r@axY), marker='s', color='b', linestyle='none')
                p2r0, = ax2.plot(np.ravel(r0@axX),np.ravel(r0@axY), marker='*', color='b', linestyle='none')
                p2eb, = ax2.plot(np.ravel(ellb@axX),np.ravel(ellb@axY), marker='.', color='b', linestyle='none',markersize=msize)

            # plt.tight_layout()
            plt.draw()
            plt.pause(0.00001)
            breakpoint()

    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()


def test_ukf_basic_quat_not_vec_w_dist_w_real_orbit():
    t0 = 0
    tf = 60*10
    tlim00 = 5
    tlim0 = 0.5*60
    tlim1 = 2*60
    tlim2 = 4*60
    dt = 1
    np.set_printoptions(precision=8)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 50.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]


    sun_eff = 1.0
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    try:
        with open("myorb", "rb") as fp:   #unPickling
            orb = pickle.load(fp)
    except:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    acts_est =  [MTQ(j,mtq_std*0,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(7)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180)**2.0,np.eye(3)*10)
    # int_cov =  block_diag(np.eye(3)*(1e-4)**2.0,1e-4*np.eye(3))
    int_cov = np.zeros((6,6))

    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 0.5
    est.kap = 0
    est.bet = 2.0#-1.0#2.0
    est.include_sens_noise_separately = False
    est.include_int_noise_separately = False
    est.use_cross_term = True
    est.scale_nonseparate_adds = False

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)


    kw = 10
    ka = 0.1

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        _,extra = est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])


        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = state
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
        if True:

            errvec = est_state-state
            errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),5),errvec[7:]])
            print('err vec ',errvec )
            mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
            # print('cov err vec ', coverr)
            # print('cov err vec ',norm(coverr))
            print('mahalanobis_dist ',math.sqrt(mahalanobis_dist2))
            print('prob ',chi2.pdf(mahalanobis_dist2,6))
            cd = chi2.cdf(mahalanobis_dist2,6)
            print('cdf ',cd)
            print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))

            N = 8
            msize = 0.5
            r_coord = np.linspace(0,2,3)
            sph_coord = [np.linspace(0,math.pi,N) for j in range(6-2)]+[np.linspace(0,2*math.pi,N)]
            se1,se2,se3,se4,se5 = np.meshgrid(*sph_coord)
            sph_mesh = np.stack([se1,se2,se3,se4,se5])
            cart_mesh = np.moveaxis(ct(sph_mesh),0,5)

            eiga0,eige0 = np.linalg.eigh(extra.cov0)
            eiga1,eige1 = np.linalg.eigh(extra.cov1)
            eiga2,eige2 = np.linalg.eigh(extra.cov2)
            r = math.sqrt(math.sqrt(mahalanobis_dist2))

            zpoint = extra.mean0
            md0 = est.states_diff(extra.mean0,zpoint)
            md1 = est.states_diff(extra.mean1,zpoint)
            md2 = est.states_diff(extra.mean2,zpoint)

            ell0 = cart_mesh@eige0@np.diagflat(np.sqrt(eiga0))*r + md0
            ell1 = cart_mesh@eige1.T@np.diagflat(np.sqrt(eiga1))*r + md1
            ell2 = cart_mesh@eige2.T@np.diagflat(np.sqrt(eiga2))*r + md2
            sigd0 = est.states_diff(est.add_to_state(extra.mean0,est.states_diff(extra.sig0,extra.mean0)*r),zpoint)
            sigd1 = est.states_diff(extra.sig1,extra.mean1)*r + md1

            # breakpoint()

            sigd = np.stack([sigd0,sigd1])

            truth = normalize(est.states_diff(state,zpoint))*1.1*norm(md2)

            sortinds = np.argsort(np.abs(eiga2))
            print(eiga2)
            indX = sortinds[-1]
            axX = eige2[:,indX]
            indY = sortinds[-2]
            axY = eige2[:,indY]#*(eiga2[indX]/eiga2[indY])
            print(eiga2[indX])
            print(eiga2[indY])
            if ind == 1:
                f = plt.figure()
                ax = plt.subplot()

            # ax.cla()
            # breakpoint()
            if ind > 1:
                ax.relim(visible_only=True)
                ax.autoscale(enable=True, axis='both', tight=False)
                pe0.remove()
                pe0, = ax.plot(np.ravel(ell0@axX),np.ravel(ell0@axY), marker='.', color='k', linestyle='none',markersize=msize)
                pe1.remove()
                pe1, = ax.plot(np.ravel(ell1@axX),np.ravel(ell1@axY), marker='.', color='r', linestyle='none',markersize=msize)
                pe2.remove()
                pe2, = ax.plot(np.ravel(ell2@axX),np.ravel(ell2@axY), marker='.', color='g', linestyle='none',markersize=msize)
                pt.remove()
                pt, = ax.plot(np.ravel(truth@axX),np.ravel(truth@axY), marker='*', color='b', linestyle='none')
                [k.remove() for k in psd]
                psd = ax.plot(sigd@axX,sigd@axY, marker='none', color='y', linestyle=':')
                ps0.remove()
                ps0, = ax.plot(np.ravel(sigd0@axX),np.ravel(sigd0@axY), marker='^', color='k', linestyle='none')
                ps1.remove()
                ps1, = ax.plot(np.ravel(sigd1@axX),np.ravel(sigd1@axY), marker='^', color='r', linestyle='none')
                pm0.remove()
                pm0, = ax.plot(np.ravel(md0@axX),np.ravel(md0@axY), marker='s', color='k', linestyle='none',markersize=4,markeredgecolor = 'k')
                pm1.remove()
                pm1, = ax.plot(np.ravel(md1@axX),np.ravel(md1@axY), marker='s', color='r', linestyle='none',markersize=4,markeredgecolor = 'k')
                pm2.remove()
                pm2, = ax.plot(np.ravel(md2@axX),np.ravel(md2@axY), marker='s', color='g', linestyle='none',markersize=4,markeredgecolor = 'k')
            else:
                pe0, = ax.plot(np.ravel(ell0@axX),np.ravel(ell0@axY), marker='.', color='k', linestyle='none',markersize=msize)
                pe1, = ax.plot(np.ravel(ell1@axX),np.ravel(ell1@axY), marker='.', color='r', linestyle='none',markersize=msize)
                pe2, = ax.plot(np.ravel(ell2@axX),np.ravel(ell2@axY), marker='.', color='g', linestyle='none',markersize=msize)
                pt, = ax.plot(np.ravel(truth@axX),np.ravel(truth@axY), marker='*', color='b', linestyle='none')
                psd = ax.plot(sigd@axX,sigd@axY, marker='none', color='y', linestyle=':')
                ps0, = ax.plot(np.ravel(sigd0@axX),np.ravel(sigd0@axY), marker='^', color='k', linestyle='none')
                ps1, = ax.plot(np.ravel(sigd1@axX),np.ravel(sigd1@axY), marker='^', color='r', linestyle='none')
                pm0, = ax.plot(np.ravel(md0@axX),np.ravel(md0@axY), marker='s', color='k', linestyle='none',markersize=4,markeredgecolor = 'k')
                pm1, = ax.plot(np.ravel(md1@axX),np.ravel(md1@axY), marker='s', color='r', linestyle='none',markersize=4,markeredgecolor = 'k')
                pm2, = ax.plot(np.ravel(md2@axX),np.ravel(md2@axY), marker='s', color='g', linestyle='none',markersize=4,markeredgecolor = 'k')

            # plt.tight_layout()
            plt.draw()
            plt.pause(0.00001)
            # breakpoint()


    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()

def test_ukf_quat_not_vec_w_dist_w_gbias():
    t0 = 0
    tf = 60*20
    tlim00 = 60
    tlim0 = 5*60
    tlim1 = 10*60
    tlim2 = 15*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 1.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8

    gyro_bias0_std = np.random.uniform(0.01,0.2)*math.pi/180.0
    real_gyro_bias0 = gyro_bias0_std*random_n_unit_vec(3)
    gyro_bsr = 0.0004*math.pi/180.0#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]


    sun_eff = 1.0
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]

    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(10)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(0.01)**2.0,np.eye(3)*3,np.eye(3)*(0.01)**2.0)*10
    int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,1e-8*np.eye(3),np.eye(3)*gyro_bsr**2.0)

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1e-3
    est.kap = 0
    est.bet = 2.0#-1.0#2.0

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)


    kw = 1500
    ka = 10

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',np.concatenate([j.bias for j in real_sat.sensors if j.has_bias]))
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        # autovar = np.diagonal(est.full_state.cov)
        # print(autovar[0:3])
        # print(autovar[3:6])
        # print(autovar[6:9])

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()


def test_ukf_quat_not_vec_w_dist_w_gbias_starting_converged():
    t0 = 0
    tf = 60*10
    tlim00 = 10
    tlim0 = 1*60
    tlim1 = 5*60
    tlim2 = 8*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 1.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8

    gyro_bias0_std = np.random.uniform(0.01,0.2)*math.pi/180.0
    real_gyro_bias0 = gyro_bias0_std*random_n_unit_vec(3)
    gyro_bsr = 0.0004*math.pi/180.0#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0,scale = 1e4) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]


    sun_eff = 1.0
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False,scale = 1e4) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]

    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.concatenate([w0,q0,real_gyro_bias0])
    estimate = np.array([j*np.random.uniform(0.99,1.01) for j in estimate])
    estimate[3:7] = normalize(estimate[3:7])
    int_cov =  block_diag(np.eye(3)*(1e-8)**2.0,1e-8*np.eye(3),np.eye(3)*gyro_bsr**2.0)
    cov_estimate = 100*int_cov.copy()

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1e-3
    est.kap = 0
    est.bet = 2.0#-1.0#2.0

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)


    kw = 1000
    ka = 100

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',np.concatenate([j.bias for j in real_sat.sensors if j.has_bias]))
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])
        print(autovar[6:9])

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])*180.0/math.pi
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()


def test_ukf_quat_not_vec_w_dist_w_sgbias():
    t0 = 0
    tf = 60*20
    tlim00 = 60
    tlim0 = 5*60
    tlim1 = 20*60
    tlim2 = 30*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 1.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 1e-8

    gyro_bias0_std = np.random.uniform(0.01,0.2)*math.pi/180.0
    real_gyro_bias0 = gyro_bias0_std*random_n_unit_vec(3)
    gyro_bsr = 0.0004*math.pi/180.0#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]


    sun_eff = 0.3
    noise_sun = 0.0001*sun_eff #0.01% of range
    sun_bias0_std = 0.1
    real_sun_bias0 = [np.random.uniform(-sun_bias0_std,sun_bias0_std)*sun_eff for j in range(6)]
    sun_bsr = 0.00001*sun_eff #0.001% of range /sec

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j],use_noise = True,bias_std_rate = sun_bsr) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
        [5.88304e-05, 0.03409127827, -0.00012334756],
        [-0.00671361357, -0.00012334756, 0.01004091997]])
    J = 0.5*(J+J.T)
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    try:
        with open("myorb", "rb") as fp:   #unPickling
            orb = pickle.load(fp)
    except:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]

    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = J.copy()
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(13)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(0.1)**2.0,np.eye(3)*10,np.eye(3)*(0.01)**2.0,np.eye(3)*(0.1)**2.0)
    int_cov =  block_diag(np.eye(3)*(1e-6)**2.0,1e-6*np.eye(3),np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1.0
    est.kap = 0#3-sum([est.sat.control_cov().shape[0],(1+est.include_int_noise_separately)*cov_estimate.shape[0],est.sat.sensor_cov().shape[0]*est.include_sens_noise_separately])
    est.bet = 2.0#-1.0#2.0
    vec_mode = 6

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)


    kw = 1000
    ka = 100

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',np.concatenate([j.bias for j in real_sat.sensors if j.has_bias]))
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        # autovar = np.diagonal(est.full_state.cov)
        # print(autovar[0:3])
        # print(autovar[3:6])
        # print(autovar[6:9])

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-10)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])*180.0/math.pi
    sbdiff = (est_state_hist[:,10:13]-state_hist[:,10:13])
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(np.log10(angdiff),title = "ang diff in lgo10deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(sbdiff,norm=True,title = "sbias diff")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()


def test_ukf_quat_not_vec_w_dist_w_smgbias_wbound(cov_est_mult=1.0,mrp_ic = 1e-8,av_ic = 1e-16,al = 1.0,kap = 0,bet = 2,sep_int=False,sep_sens = False,xtermmult = 0,ang_werr_mult = 0,invjpow = 0):
    np.random.seed(1)
    t0 = 0
    tf = 60*60
    tlim00 = 10
    tlim0 = 5*60
    tlim1 = 10*60
    tlim2 = 20*60
    tlim3 = 35*60
    tlim4 = 50*60
    tlim5 = 65*60
    tlim6 = 80*60
    # tlim6 = 80*60
    dt = 1
    np.set_printoptions(precision=6)
    mtqnoise = True
    mtmscale = 1e0
    #real_sat
    real_sat = create_BC_sat(real=True,use_dipole = False,include_mtqbias = False,rand=True,mtm_scale = mtmscale,include_mtq_noise = True)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    # w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180

    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    try:
        with open("myorb", "rb") as fp:   #unPickling
            orb = pickle.load(fp)
    except:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,use_dipole = False,include_mtqbias = False,estimate_dipole = False,mtm_scale = mtmscale,include_mtq_noise = mtqnoise)
    info_sat = create_BC_sat(real=False,use_dipole = False,include_mtqbias = False,estimate_dipole = False,mtm_scale = mtmscale,include_mtq_noise = True)
    estimate = np.zeros(16)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*10,np.eye(3)*(1e-8*mtmscale)**2.0,np.eye(3)*(0.1*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0)*cov_est_mult

    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*block_diag(dt*np.block([[np.eye(3)*(1/dt)*av_ic,0.5*np.eye(3)*av_ic],[0.5*np.eye(3)*av_ic,(1/3)*np.eye(3)*dt*av_ic + np.eye(3)*mrp_ic]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.attitude_sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    werr = (np.linalg.matrix_power(est_sat.invJ.copy(),int(invjpow)))*av_ic
    int_cov = dt*block_diag(dt*np.block([[werr/dt,xtermmult*werr],[xtermmult*werr,ang_werr_mult*dt*werr + np.eye(3)*mrp_ic]]),np.diagflat([j.bias_std_rate**2.0*j.scale*j.scale for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    # int_cov =  block_diag(np.eye(3)*(1e-4)**2.0,1e-12*np.eye(3),np.diagflat([j.bias_std_rate**2.0*j.scale*j.scale for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    attempted_info_mat = np.linalg.inv(cov_estimate.copy())#np.linalg.cholesky(cov_estimate.copy())
    # int_cov =  block_diag(np.eye(3)*0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    mtq_max = [j.max for j in est_sat.actuators]
    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.include_sens_noise_separately = sep_sens
    est.include_int_noise_separately = sep_int
    est.included_int_noise_where = 2
    est.use_cross_term = True
    est.al = al#1.0#1#1.0#0.001#0.01#0.01#0.99#0.9#1e-1#0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = kap#3-21#3-15#3-12#0#3-18#3-18#0#3-18#-10#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.bet = bet
    est.scale_nonseparate_adds = False
    est.vec_mode = 6


    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    info_mat_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    vecs_hist = []
    sens_hist = np.nan*np.zeros((int((tf-t0)/dt),9))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0-dt)*sec2cent)

    sens = np.zeros(len(real_sat.attitude_sensors))
    QQ0 = int_cov.copy()
    # Qinv = np.linalg.inv(QQ)



    kw = 1000
    ka = 1
    prev_full_state = np.concatenate([state.copy()]+[j.bias for j in real_sat.attitude_sensors if j.has_bias])

    # breakpoint()
    #covestimate at time0, state at time 0, 0 control, estimate at time0,os at time 0
    real_vecs = os_local_vecs(orbt,state[3:7])
    real_sbias = np.concatenate([j.bias for j in real_sat.attitude_sensors if j.has_bias])
    real_full_state = np.concatenate([state.copy(),real_sbias.copy()])
    sens = real_sat.sensor_values(state,real_vecs)
    # est.initialize_estimate(sens,[np.array([0,1,2]),np.array([6,7,8])],[orbt.B*mtmscale,est_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)],np.array([3,4,5]),orbt)
    # print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est.full_state.val[3:7],state[3:7]),-1,1)**2.0 ))
    # print('av ',(norm(est.full_state.val[0:3]-state[0:3])*180.0/math.pi))
    # print('gb ',norm(est.full_state.val[10:13]-real_sbias[3:6])*180.0/math.pi)
    # print('mb (x1e8/scale)',norm(est.full_state.val[7:10]-real_sbias[0:3])*1e8/mtmscale)
    # print('sb (x1e4)',norm(est.full_state.val[13:16]-real_sbias[6:9])*1e4)

    # beci = orbt.B*mtmscale
    # seci = real_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)
    # print('pred/act B field')
    # print(np.stack([rot_mat(est.full_state.val[3:7])@(sens[0:3]-est.full_state.val[7:10]),beci]))
    # berr = rot_mat(est.full_state.val[3:7])@(sens[0:3]-est.full_state.val[7:10])-beci
    # print(berr)
    # print(norm(berr),np.dot(rot_mat(est.full_state.val[3:7])@(sens[0:3]-est.full_state.val[7:10]),beci)/(norm(rot_mat(est.full_state.val[3:7])@(sens[0:3]-est.full_state.val[7:10]))*norm(orbt.B*mtmscale)))
    # print('pred/halfpred/act S')
    # print(np.stack([rot_mat(est.full_state.val[3:7])@(sens[6:9]-est.full_state.val[13:16]),seci]))
    # serr = rot_mat(est.full_state.val[3:7])@(sens[6:9]-est.full_state.val[13:16])-real_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)
    # print(serr)
    # print(norm(serr),np.dot(rot_mat(est.full_state.val[3:7])@(sens[6:9]-est.full_state.val[13:16]),real_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R))/(norm(rot_mat(est.full_state.val[3:7])@(sens[6:9]-est.full_state.val[13:16]))*norm(real_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R))))
    #
    # msens = sens[0:3]
    # mbcov = est.full_state.cov[6,6]
    # mncov = est.sat.sensor_cov()[0,0]
    # ssens = sens[6:9]
    # sbcov = est.full_state.cov[12,12]
    # sncov = est.sat.sensor_cov()[6,6]
    # wcov = est.full_state.cov[0,0]
    # gsens = sens[3:6]
    # gbcov = est.full_state.cov[9,9]
    # gncov = est.sat.sensor_cov()[3,3]
    #
    # print('std dev distance for estimated values ')
    # print('mb ',(est.full_state.val[7:10])/np.diag(est.full_state.cov)[6:9])
    # print('mnoise ',(msens - rot_mat(est.full_state.val[3:7]).T@beci - est.full_state.val[7:10])/np.diag(est.sat.sensor_cov())[0:3])
    # print('sb ',(est.full_state.val[13:16])/np.diag(est.full_state.cov)[12:15])
    # print('snoise ',(ssens - rot_mat(est.full_state.val[3:7]).T@seci - est.full_state.val[13:16])/np.diag(est.sat.sensor_cov())[6:9])
    # print('ang mdist ',np.sum((ssens - rot_mat(est.full_state.val[3:7]).T@seci - est.full_state.val[13:16])**2.0/np.diag(est.sat.sensor_cov())[6:9]) + np.sum((est.full_state.val[13:16])**2.0/np.diag(est.full_state.cov)[12:15]) + np.sum((msens - rot_mat(est.full_state.val[3:7]).T@beci - est.full_state.val[7:10])**2.0/np.diag(est.sat.sensor_cov())[0:3]) + np.sum((est.full_state.val[7:10])**2.0/np.diag(est.full_state.cov)[6:9]))
    # # print('av ',(est.full_state.val[0:3])/np.diag(est.full_state.cov)[0:3])
    # # print('gb ',(est.full_state.val[10:13])/np.diag(est.full_state.cov)[9:12])
    # # print('gnoise ',(sens[3:6] - est.full_state.val[0:3] - est.full_state.val[10:13])/np.diag(est.sat.sensor_cov())[3:6])
    # # print('av mdist ',np.sum((est.full_state.val[0:3])**2.0/wcov) + np.sum((gsens - est.full_state.val[10:13] - est.full_state.val[0:3])**2.0/gncov) + np.sum((est.full_state.val[10:13] )**2.0/gbcov))
    #
    # probfunc = lambda x : (np.sum((np.array(x[4:7]))**2.0/mbcov) + np.sum((msens - np.array(x[4:7]) - rot_mat(np.array(x[0:4])).T@beci)**2.0/mncov) + np.sum((np.array(x[7:10]))**2.0/sbcov) + np.sum((ssens - np.array(x[7:10]) - rot_mat(np.array(x[0:4])).T@seci)**2.0/sncov) )
    # constfunc = lambda x : np.dot(np.array(x[0:4]),np.array(x[0:4])) - 1
    # nlc = NonlinearConstraint(constfunc,0.0,0.0)
    #
    # N = 100
    # qlist = np.nan*np.ones((4,N+1))
    # flist = np.nan*np.ones(N+1)
    # x0 = np.zeros(10)
    # x0[0] = 1
    # res = minimize(probfunc,x0,constraints = nlc,tol = 1e-10)
    # # print('ang opt ',res.x,res.fun)
    # # print(normalize(res.x[0:4]))
    # qlist[:,0] = normalize(res.x[0:4])
    # flist[0] = res.fun
    # for j in range(N):
    #     x0 = np.zeros(10)
    #     x0[0:4] = random_n_unit_vec(4)
    #     st = est.initialize_estimate(sens,[np.array([0,1,2]),np.array([6,7,8])],[orbt.B*mtmscale,est_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)],np.array([3,4,5]),orbt,q= x0[0:4])
    #     x0[4:7] = st[7:10]
    #     x0[7:10] = st[10:13]
    #     res = minimize(probfunc,x0,constraints = nlc,tol = 1e-10)
    #     # print('ang opt ',res.x,res.fun)
    #     # print(normalize(res.x[0:4]))
    #     qlist[:,j+1] = normalize(res.x[0:4])
    #     flist[j+1] = res.fun
    # optq = qlist[:,np.argmin(flist)]
    # x0 = np.zeros(10)
    # x0[0:4] = optq
    #
    # res = minimize(probfunc,x0,constraints = nlc,tol = 1e-10)
    # optq = normalize(res.x[0:4])
    # print(optq)
    #
    # # probfunc2 = lambda x : np.sum((np.array(x[0:3]))**2.0/wcov) + np.sum((gsens - np.array(x[3:6]) - np.array(x[0:3]))**2.0/gncov) + np.sum((np.array(x[3:6]))**2.0/gbcov)
    # # x02 = np.zeros(6)
    # # res2 = minimize(probfunc2,x02)
    # # print('av opt ',res2.x,res2.fun)
    #
    #
    # print('std dev distance for optimized values ')
    # print('mb ',(res.x[4:7])/np.diag(est.full_state.cov)[6:9])
    # print('mnoise ',(msens - rot_mat(optq).T@beci - res.x[4:7])/np.diag(est.sat.sensor_cov())[0:3])
    # print('sb ',(res.x[7:10])/np.diag(est.full_state.cov)[12:15])
    # print('snoise ',(ssens - rot_mat(optq).T@seci - res.x[7:10])/np.diag(est.sat.sensor_cov())[6:9])
    # print('ang mdist ',np.sum((ssens - rot_mat(optq).T@seci - res.x[7:10])**2.0/np.diag(est.sat.sensor_cov())[6:9]) + np.sum((res.x[7:10])**2.0/np.diag(est.full_state.cov)[12:15]) + np.sum((msens - rot_mat(optq).T@beci - res.x[4:7])**2.0/np.diag(est.sat.sensor_cov())[0:3]) + np.sum((res.x[4:7])**2.0/np.diag(est.full_state.cov)[6:9]))
    # # print('w ',(res2.x[0:3])/np.diag(est.full_state.cov)[0:3])
    # # print('gb ',(res2.x[3:6])/np.diag(est.full_state.cov)[9:12])
    # # print('gnoise ',(gsens - res2.x[3:6] - res2.x[0:3])/np.diag(est.sat.sensor_cov())[3:6])
    # # print('av mdist ',np.sum((np.array(res2.x[0:3]))**2.0/wcov) + np.sum((gsens - np.array(res2.x[3:6]) - np.array(res2.x[0:3]))**2.0/gncov) + np.sum((np.array(res2.x[3:6]))**2.0/gbcov))
    # print('comparisons for optimzied values')
    # print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(normalize(optq),state[3:7]),-1,1)**2.0 ))
    # # print('av ',(norm(res2.x[0:3]-state[0:3])*180.0/math.pi))
    # # print('gb ',norm(res2.x[3:6]-real_sbias[3:6])*180.0/math.pi)
    # print('mb (x1e8/scale)',norm(res.x[4:7]-real_sbias[0:3])*1e8/mtmscale)
    # print('sb (x1e4)',norm(res.x[7:10]-real_sbias[6:9])*1e4)
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(qlist[1,:],qlist[2,:],qlist[3,:])
    # plt.draw()
    # plt.pause(1)


    # breakpoint()
    info_est = estimated_nparray(real_full_state,np.linalg.inv(attempted_info_mat),int_cov)
    info_sat.match_estimate(info_est,dt)
    prev_os = orb.get_os(0.22+(t-t0-dt)*sec2cent)

    while t<tf:
        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.attitude_sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.attitude_sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]
        info_mat_hist += [attempted_info_mat.copy()]
        vecs_hist += [copy.deepcopy( real_vecs)]
        sens_hist[ind,:] = sens


        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        pre_sens = sens.copy()
        prev_vecs = copy.deepcopy( real_vecs)
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        prev_full_state = real_full_state.copy()
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state.copy(), method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-10)#,jac = ivp_jac)

        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])

        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)

        # breakpoint()


        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        real_sbias = np.concatenate([j.bias for j in real_sat.attitude_sensors if j.has_bias])
        real_full_state = np.concatenate([state.copy(),real_sbias.copy()])
        sens = real_sat.sensor_values(state,real_vecs)


        # for j in range(len(est.use_state.val)):
        #     if j==0:
        #         est.use_state.val[j] = prev_full_state[j].copy()
        # est.use_state.val[3:7] = normalize(est.use_state.val[3:7])

        est.update(control,sens,orbt)
        est_state = est.use_state.val

        F0nm1,B0nm1,dab0,dsb0,ddmp0 = info_sat.rk4Jacobians(prev_full_state[:info_sat.state_len].copy(),control,dt,prev_os,orbt)
        # T1 =  block_diag(np.eye(3),vec3_to_quat_deriv(quat_to_vec3(prev_full_state[3:7],est.vec_mode),est.vec_mode),np.eye(info_sat.state_len - 7))
        T1 =  0.5*block_diag(np.eye(3),Wmat(prev_full_state[3:7]).T,np.eye(info_sat.state_len - 7))

        # T2 = block_diag(np.eye(3),quat_to_vec3_deriv(state[3:7],est.vec_mode),np.eye(info_sat.state_len - 7))
        T2 = 2.0*block_diag(np.eye(3),Wmat(state[3:7]),np.eye(info_sat.state_len - 7))
        B = B0nm1@T2
        # breakpoint()
        combo = np.vstack([T1@F0nm1,dab0,dsb0,ddmp0])@T2
        Fnm1 = np.eye(len(est_state) - 1 + est.quat_as_vec)
        Bnm1 = np.zeros((est.sat.control_len,len(est_state) - 1 + est.quat_as_vec))
        Fnm1[:,:info_sat.state_len - 1 + est.quat_as_vec] = combo
        Bnm1[:,:info_sat.state_len - 1 + est.quat_as_vec] = B
        info_est = estimated_nparray(real_full_state,np.linalg.inv(attempted_info_mat),int_cov)
        info_sat.match_estimate(info_est,dt)
        # Finv = np.linalg.inv(Fnm1)
        QQ = QQ0 + Bnm1.T@info_sat.control_cov()@Bnm1
        Qinv = np.linalg.inv(QQ)
        # Fnm1 = Fnm1.T.copy()


        D12 = -Qinv@Fnm1.T
        D11 = Fnm1@Qinv@Fnm1.T
        state_jac,bias_jac = info_sat.sensor_state_jacobian(state,real_vecs)
        # breakpoint()
        Hn = np.vstack([T2.T@state_jac,np.zeros((info_sat.act_bias_len,len(sens))),bias_jac,np.zeros((info_sat.dist_param_len-(len(est.use)-sum(est.use)),len(sens)))])

        D22 = Qinv + Hn@np.diag(1/np.diag(info_sat.sensor_cov()))@Hn.T
        attempted_info_mat = D22 - D12@np.linalg.inv(attempted_info_mat+D11)@D12.T
        # attempted_info_mat = QQ@np.linalg.inv(QQ + QQ@Hn@np.diag(1/np.diag(info_sat.sensor_cov()))@Hn.T@QQ - QQ@np.linalg.inv(QQ@Finv.T@attempted_info_mat@Finv@QQ + QQ)@QQ)@QQ
        # if not np.all(np.linalg.eigvals(attempted_info_mat) > 0):
        #     breakpoint()
        # info_est = estimated_nparray(real_full_state,attempted_info_mat,int_cov)
        # info_sat.match_estimate(info_est)


        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',real_sbias)
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        # var_bound = np.sqrt(np.diagonal(np.linalg.inv(attempted_info_mat)))
        var_bound = 1/np.sqrt(np.diagonal(attempted_info_mat))
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        # print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        # print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        # print(autovar[0:3]*180.0/math.pi)
        # print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))


        # print('real state ',state)
        # print('est  state ',est_state[0:7])
        # print('real sbias ',real_sbias)
        # print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        # autovar = np.sqrt(np.diagonal(est.use_state.cov))
        # est_vecs = os_local_vecs(orbt,est_state[3:7])
        unbiased_sens = sens-est_state[7:]
        print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)
        print(var_bound[0:3]*180.0/math.pi)
        print('simple av ',(norm(unbiased_sens[3:6]-state[0:3])*180.0/math.pi))
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi)-np.pi))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*((4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)+np.pi)%(2*np.pi) -np.pi))
        sbvec_est = unbiased_sens[6:9]
        bbvec_est = unbiased_sens[0:3]
        srvec_real = real_sat.sensors[6].efficiency[0]*normalize(orbt.S-orbt.R)
        brvec_real = mtmscale*orbt.B
        print('ang to S/B est ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(two_vec_to_quat(srvec_real,brvec_real,sbvec_est,bbvec_est),state[3:7]),-1,1)**2.0 ))
        # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi,4*norm(np.arctan(autovar[3:6]/2.0))*180.0/math.pi)
        print(4*np.arctan(var_bound[3:6]/2.0)*180.0/math.pi)
        print('gb ',norm(est_state[10:13]-real_sbias[3:6])*180.0/math.pi)
        print((est_state[10:13]-real_sbias[3:6])*180.0/math.pi)
        print(autovar[9:12]*180.0/math.pi)
        print(var_bound[9:12]*180.0/math.pi)

        print('mb (x1e8/scale)',norm(est_state[7:10]-real_sbias[0:3])*1e8/mtmscale)
        print((est_state[7:10]-real_sbias[0:3])*1e8/mtmscale)
        print(autovar[6:9]*1e8/mtmscale)
        print(var_bound[6:9]*1e8/mtmscale)

        print('sb (x1e4)',norm(est_state[13:16]-real_sbias[6:9])*1e4)
        print((est_state[13:16]-real_sbias[6:9])*1e4)
        print(autovar[12:15]*1e4)
        print(var_bound[12:15]*1e4)

        # print('dip ',norm(est_state[16:19]-real_dist))
        # print(est_state[16:19]-real_dist)
        # print(autovar[15:18])
        # print(var_bound[15:18])
        errvec = est_state-real_full_state
        errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(real_full_state[3:7]),est_state[3:7]),5),errvec[7:]])
        print('err vec ',errvec )
        # coverr = np.sqrt(errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec)
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        # print('cov err ',coverr)
        mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        print('mahalanobis_dist ',np.sqrt(mahalanobis_dist2))
        print('prob ',chi2.pdf(mahalanobis_dist2,15))
        cd = chi2.cdf(mahalanobis_dist2,15)
        print('cdf ',cd)
        print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))
        # print(est.wts_m)
        # print(est.wts_c)

        # print('pred/halfpred/act B field')
        # print(np.stack([rot_mat(est_state[3:7])@bbvec_est,rot_mat(state[3:7])@bbvec_est,brvec_real]))
        # print('pred/halfpred/act S')
        # print(np.stack([rot_mat(est_state[3:7])@sbvec_est,rot_mat(state[3:7])@sbvec_est,srvec_real]))


        # print('mrp ',(180/np.pi)*norm(4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0)))
        # # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        # print((180/np.pi)*4.0*np.arctan(quat_to_vec3(quat_mult(quat_inv(state[3:7]),est_state[3:7]),0)/2.0))
        # # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        # print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi)
        # print('gb ',norm(est_state[10:13]-real_sbias[3:6])*180.0/math.pi)
        # print((est_state[10:13]-real_sbias[3:6])*180.0/math.pi)
        # print(autovar[9:12]*180.0/math.pi)
        #
        # print('mb ',norm(est_state[7:10]-real_sbias[0:3]))
        # print((est_state[7:10]-real_sbias[0:3])*1.0)
        # print(autovar[6:9]*1.0)
        #
        # print('sb ',norm(est_state[13:16]-real_sbias[6:9]))
        # print((est_state[13:16]-real_sbias[6:9])*1.0)
        # print(autovar[12:15]*1.0)
        # errvec = est_state-real_full_state
        # errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(real_full_state[3:7]),est_state[3:7]),0),errvec[7:]])
        # print('err vec ',errvec )


        # coverr = np.linalg.inv(est.use_state.cov)@errvec
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        # print('cov err vec ',errvec@coverr)

        # invcov = np.linalg.inv(est.use_state.cov)
        # coverr = np.sqrt(errvec.T@invcov@errvec)
        # print('cov err ',coverr,np.log10(coverr))
        # try:
        #     ea,ee = np.linalg.eig(est.use_state.cov)
        #     sqrt_invcov = ee@np.diagflat(1/np.sqrt(ea))#*np.linalg.cholesky(invcov)
        #     # sqrt_invcov2 = np.linalg.cholesky(np.linalg.inv(est.use_state.cov))
        #     # sqrt_invcov3 = np.linalg.inv(np.linalg.cholesky(est.use_state.cov))
        #     # sqrt_cov = np.linalg.cholesky(est.use_state.cov)
        #     # sqrt_cov = np.linalg.cholesky(est.use_state.cov)
        #     # print(sum(np.abs(ee@np.diagflat(ea)@ee.T - invcov)))
        #     # breakpoint()
        #     # print(sqrt_invcov@sqrt_invcov.T - invcov)
        #     # print('scaled state err', sqrt_invcov@errvec)
        #     sc_err = sqrt_invcov.T@errvec
        #     biggest = np.argmax(np.abs(sc_err))
        #     print('cov-eig state err',sc_err)
        #     tt = ee@((np.sign(sc_err)*sc_err[biggest]*(sc_err/sc_err[biggest])**2.0)*ea)
        #     print('rescaled state err',np.sign(tt)*np.sqrt(np.abs(tt)))
        #     # print('rescaled state err', ee@sqrt_invcov.T@errvec)
        #     print('biggest dir state err', np.dot(ee[:,biggest],errvec)*ee[:,biggest],sc_err[biggest])
        #     # print('rescaled state err', ee@np.diagflat(np.sqrt(ea))@sqrt_invcov.T@errvec)
        #     print('overall err', norm(sc_err))
        #     # print('scaled state err3', sqrt_cov@errvec)
        #
        #
        #     # print('scaled state err4', sqrt_invcov3@errvec)
        #     # print('rescaled state err4', (1/np.mean(matrix_row_norm(sqrt_invcov3)))*matrix_row_normalize(sqrt_invcov3).T@sqrt_invcov3@errvec)
        #     # print('overall err', math.sqrt(np.dot(sqrt_invcov3@errvec,sqrt_invcov3@errvec)))
        #
        #
        #
        # except:
        #     pass
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))

        # print(orbt.B,real_vecs['b'],est_vecs['b'])
        # print(orbt.S-orbt.R,real_vecs['s']-real_vecs['r'],est_vecs['s']-est_vecs['r'])
        # print(sens)
        # print(est.sat.sensor_values(est_state,est_vecs))
        # print( np.array([j.clean_reading(est_state,est_vecs) for j in est_sat.attitude_sensors]))
        # # breakpoint()
        # print(autovar[6:9])

        #find control
        q = est.use_state.val[3:7]
        w =  est.use_state.val[0:3]
        wdes = np.zeros(3)
        nB2 = norm(orbt.B)
        Bbody = rot_mat(q).T@orbt.B
        w_err = w-wdes
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e10*(sens[0:3]-pre_sens[0:3])
            # ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))\
        else:
            if t<tlim1:
                #PD to zeroquat
                qdes = zeroquat

            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])

            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            elif t<tlim4:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,0,1])
            elif t<tlim5:
                #PID to [0,1,0,0]
                qdes = np.array([1,0,0,0])
            elif t<tlim6:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            print('qdes', qdes)

            q_err = quat_mult(quat_inv(qdes),q)
            print('control err',(180/np.pi)*math.acos(-1 + 2*np.clip(q_err[0],-1,1)**2.0 ))
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        # print('ud   ', ud)
        print('ctrl ',control)

        # #

    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    mrpdiff = np.vstack([quat_to_vec3(quat_mult(quat_inv(state_hist[j,3:7]),est_state_hist[j,3:7]),0)/2.0 for j in range(state_hist.shape[0])])
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    mbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])/np.array([j.scale for j in real_sat.sensors[0:3]])
    gbdiff = (est_state_hist[:,10:13]-state_hist[:,10:13])*180.0/math.pi
    sbdiff = (est_state_hist[:,13:16]-state_hist[:,13:16])
    sd = [est.states_diff(est_state_hist[j,:],state_hist[j,:]) for j in range(state_hist.shape[0])]
    # dipdiff =  (est_state_hist[:,16:19]-state_hist[:,16:19])/1e4
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    # range = (tf-t0)/dt
    log_angdiff = np.log(angdiff[int(0.1*ind):int(0.9*ind)])
    tc = np.polyfit(log_angdiff,range(int(0.9*ind)-int(0.1*ind)),1)[1]
    res = [np.mean(matrix_row_norm(avdiff.T)), np.amax(matrix_row_norm(avdiff.T)), np.mean(angdiff), np.amax(angdiff), np.where(angdiff<1)[0],tc]
    # breakpoint()
    qhist = state_hist[:,3:7]
    rhist = np.vstack([j.R for j in orb_hist])
    shist = np.vstack([j.S for j in orb_hist])
    bhist = np.vstack([j.B for j in orb_hist])

    rvechist = np.vstack([j["r"] for j in vecs_hist])
    svechist = np.vstack([j["s"] for j in vecs_hist])
    bvechist = np.vstack([j["b"] for j in vecs_hist])
    srvechist = svechist-rvechist
    srhist = shist-rhist
    # return res
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(mrpdiff,title = "mrp diff")
    plot_the_thing(matrix_row_normalize(mrpdiff),title = "mrp diff normalized")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(np.log10(matrix_row_norm(gbdiff)),norm=False,title = "log10 gbias diff in deg/s")
    plot_the_thing(sbdiff,norm=True,title = "sbias diff")
    plot_the_thing(np.log10(matrix_row_norm(sbdiff)),norm=False,title = "log10 sbias diff")
    plot_the_thing(mbdiff,norm=True,title = "mbias diff in nT")
    plot_the_thing(np.log10(matrix_row_norm(mbdiff)),norm=False,title = "log10 mbias diff in nT")
    # plot_the_thing(dipdiff,norm=True,title = "dipole est diff in Am2")
    plot_the_thing(np.hstack([est_state_hist[:,7:10]/np.array([j.scale for j in real_sat.sensors[0:3]]),state_hist[:,7:10]/np.array([j.scale for j in real_sat.sensors[0:3]])]),norm=False,title = "mbias in nT")
    plot_the_thing(np.hstack([est_state_hist[:,10:13],state_hist[:,10:13]])*180.0/math.pi,norm=False,title = "gbias in nT")
    plot_the_thing(np.hstack([est_state_hist[:,13:16],state_hist[:,13:16]]),norm=False,title = "sbias")
    plot_the_thing(np.hstack([est_state_hist[:,0:3],state_hist[:,0:3]])*180.0/math.pi,norm=False,title = "av in deg/s")
    plot_the_thing(np.hstack([est_state_hist[:,3:7],state_hist[:,3:7]]),norm=False,title = "quat")
    mahal_hist = np.sqrt(np.array([sd[j].T@np.linalg.inv(cov_hist[j])@sd[j] for j in range(len(cov_hist))]))
    plot_the_thing(np.log10(mahal_hist),norm=False,title = "log10 mahalanobis distance")
    plot_the_thing(mahal_hist,norm=False,title = "mahalanobis distance")

    # plot_the_thing(autocov_hist,title = "cov_hist")
    # plot_the_thing(state_hist[:,3:7],title = "quat hist")
    # plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")
    plot_the_thing(np.log10(angdiff),title = "log angdiff (log10 deg) hist")
    plot_the_thing(np.log10(matrix_row_norm(avdiff)),title = "log avdiff norm (log10 deg/s) hist")
    # plot_the_thing(rhist,norm=True,title = "R hist")
    # plot_the_thing(shist,norm=True,title = "S hist")
    # plot_the_thing(bhist,norm=True,title = "B hist")
    # plot_the_thing(srhist,norm=True,title = "S-R hist")
    # plot_the_thing(rvechist,norm=True,title = "R vec hist")
    # plot_the_thing(svechist,norm=True,title = "S vec hist")
    # plot_the_thing(bvechist,norm=True,title = "B vec hist")
    # plot_the_thing(srvechist,norm=True,title = "S-R vec hist")
    # plot_the_thing(rot_mat_list(qhist,rvechist),norm=True,title = "R vec ECI hist")
    # plot_the_thing(rot_mat_list(qhist,svechist),norm=True,title = "S vec ECI hist")
    # plot_the_thing(rot_mat_list(qhist,bvechist),norm=True,title = "B vec ECI hist")
    # plot_the_thing(rot_mat_list(qhist,srvechist),norm=True,title = "S-R vec ECI hist")
    # plot_the_thing(rot_mat_list(qhist,rvechist) - rhist,norm=True,title = "R vec ECI hist err")
    # plot_the_thing(rot_mat_list(qhist,svechist) - shist,norm=True,title = "S vec ECI hist err")
    # plot_the_thing(rot_mat_list(qhist,bvechist) - bhist,norm=True,title = "B vec ECI hist err")
    # plot_the_thing(rot_mat_list(qhist,srvechist) - srhist,norm=True,title = "S-R vec ECI hist err")

    # plot_the_thing(sens_hist[:,0:3]/mtmscale - state_hist[:,7:10]/mtmscale,norm=True, title = "B sens minus real bias")
    # plot_the_thing(sens_hist[:,0:3]/mtmscale - est_state_hist[:,7:10]/mtmscale,norm=True, title = "B sens minus est bias")
    # plot_the_thing(sens_hist[:,0:3]/mtmscale - state_hist[:,7:10]/mtmscale - bvechist,norm=True, title = "B sens minus real bias minus real B body vec")
    # plot_the_thing(sens_hist[:,0:3]/mtmscale - est_state_hist[:,7:10]/mtmscale - bvechist,norm=True, title = "B sens minus est bias minus real B body vec")
    plot_the_thing(rot_mat_list(qhist,(sens_hist[:,0:3] - est_state_hist[:,7:10])/mtmscale,transpose=False) - bhist,norm=True, title = "B ECI est minus real B vec")

    # plot_the_thing(sens_hist[:,6:9] - state_hist[:,13:16],norm=True, title = "S-R sens minus real bias")
    # plot_the_thing(sens_hist[:,6:9] - est_state_hist[:,13:16],norm=True, title = "S-R sens minus est bias")
    # plot_the_thing(sens_hist[:,6:9] - state_hist[:,13:16] - 0.3*matrix_row_normalize(srvechist),norm=True, title = "S-R sens minus real bias minus real S-R body vec")
    # plot_the_thing(sens_hist[:,6:9] - est_state_hist[:,13:16] - 0.3*matrix_row_normalize(srvechist),norm=True, title = "S-R sens minus est bias minus real S-R body vec")
    plot_the_thing(rot_mat_list(qhist,(sens_hist[:,6:9] - est_state_hist[:,13:16]),transpose=False) - 0.3*matrix_row_normalize(srhist),norm=True, title = "S-R ECI est minus real S-R vec")
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(angdiff,avdiff,mahal_hist)
    ax.plot3D(angdiff,matrix_row_norm(avdiff),mahal_hist)
    ax.set_zlabel('mahal')
    ax.set_xlabel('ang')
    ax.set_ylabel('av')
    plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot3D(np.log10(angdiff),np.log10(np.abs(avdiff)),np.log10(mahal_hist))
    ax2.plot3D(np.log10(angdiff),np.log10(matrix_row_norm(avdiff)),np.log10(mahal_hist))
    ax2.set_zlabel('mahal')
    ax2.set_xlabel('ang')
    ax2.set_ylabel('av')
    plt.draw()
    plt.pause(1)
    breakpoint()

def test_ukf_quat_not_vec_w_dist_w_smgbias(cov_est_mult=1,mrp_ic = 1e-16,av_ic = 1e-12,al = 1e-3,kap = 0.0,bet = 2,sep_int=False,sep_sens = False,xtermmult = 0,ang_werr_mult = 0,invjpow = 0):
    np.random.seed(1)
    t0 = 0
    tf = 60*60
    tlim00 = 10
    tlim0 = 5*60
    tlim1 = 1e12#10*60
    tlim2 = 1e12#20*60
    tlim3 = 1e12#35*60
    tlim4 = 1e12#50*60
    tlim5 = 1e12#65*60
    tlim6 = 1e12#80*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    real_sat = create_BC_sat(real=True,include_mtqbias = False,rand=True,use_dipole = False)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    # w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180

    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    try:
        with open("myorb", "rb") as fp:   #unPickling
            orb = pickle.load(fp)
    except:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,include_mtqbias = False,estimate_dipole = False,use_dipole = False)
    estimate = np.zeros(16)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*10,np.eye(3)*(1e-4)**2.0,np.eye(3)*(0.1*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0)*cov_est_mult

    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*block_diag(dt*np.block([[np.eye(3)*(1/dt)*av_ic,0.5*np.eye(3)*av_ic],[0.5*np.eye(3)*av_ic,(1/3)*np.eye(3)*dt*av_ic + np.eye(3)*mrp_ic]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.attitude_sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    werr = (np.linalg.matrix_power(est_sat.invJ.copy(),int(invjpow)))*av_ic
    int_cov = dt*block_diag(dt*np.block([[werr/dt,xtermmult*werr],[xtermmult*werr,ang_werr_mult*dt*werr + np.eye(3)*mrp_ic]]),np.diagflat([j.bias_std_rate**2.0*j.scale*j.scale for j in est_sat.sensors if j.has_bias and j.estimated_bias]))
    # int_cov =  block_diag(np.eye(3)*(1e-4)**2.0,1e-12*np.eye(3),np.diagflat([j.bias_std_rate**2.0*j.scale*j.scale for j in est_sat.sensors if j.has_bias and j.estimated_bias]))

    # int_cov =  block_diag(np.eye(3)*0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    mtq_max = [j.max for j in est_sat.actuators]
    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.include_sens_noise_separately = sep_sens
    est.include_int_noise_separately = sep_int
    est.included_int_noise_where = 2
    est.use_cross_term = True
    est.al = al#1.0#1#1.0#0.001#0.01#0.01#0.99#0.9#1e-1#0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = kap#3-21#3-15#3-12#0#3-18#3-18#0#3-18#-10#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.bet = bet
    est.scale_nonseparate_adds = False

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)



    kw = 50
    ka = 1
    prev_full_state = np.concatenate([state.copy()]+[j.bias for j in real_sat.sensors if j.has_bias])

    # breakpoint()

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])
        real_full_state = np.concatenate([state.copy(),real_sbias.copy()])
        sens = real_sat.sensor_values(state,real_vecs)
        # for j in range(len(est.use_state.val)):
        #     if j==0:
        #         est.use_state.val[j] = prev_full_state[j].copy()
        # est.use_state.val[3:7] = normalize(est.use_state.val[3:7])
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',real_sbias)
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm(4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi)
        print('gb ',norm(est_state[10:13]-real_sbias[3:6]))
        print((est_state[10:13]-real_sbias[3:6]))
        print(autovar[9:12]*180.0/math.pi)

        print('mb ',norm(est_state[7:10]-real_sbias[0:3]))
        print((est_state[7:10]-real_sbias[0:3]))
        print(autovar[6:9])

        print('sb ',norm(est_state[13:16]-real_sbias[6:9]))
        print((est_state[13:16]-real_sbias[6:9]))
        print(autovar[12:15])

        errvec = real_full_state-est_state
        errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(est_state[3:7]),real_full_state[3:7]),5),errvec[7:]])
        print('err vec ',errvec )
        coverr = np.sqrt(errvec.T@np.linalg.inv(est.use_state.cov)@errvec)
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        print('cov err ',coverr)
        mahalanobis_dist2 = errvec.T@np.linalg.inv(est.use_state.cov.copy())@errvec
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        print('mahalanobis_dist ',math.sqrt(mahalanobis_dist2))
        print('prob ',chi2.pdf(mahalanobis_dist2,15))
        cd = chi2.cdf(mahalanobis_dist2,15)
        print('cdf ',cd)
        print('std dev eq ',math.sqrt(2)*erfinv(2*cd-1))
        # print(orbt.B,real_vecs['b'],est_vecs['b'])
        # print(orbt.S-orbt.R,real_vecs['s']-real_vecs['r'],est_vecs['s']-est_vecs['r'])
        # print(sens)
        # print(est.sat.sensor_values(est_state,est_vecs))
        # print( np.array([j.clean_reading(est_state,est_vecs) for j in est_sat.sensors]))
        # # breakpoint()
        # print(autovar[6:9])

        #find control
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        nB2 = norm(orbt.B)
        Bbody = rot_mat(q).T@orbt.B
        w_err = w-wdes
        if t<tlim00:
            control = np.zeros(3)
        elif t<tlim0:
            #bdot
            # Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e8*(sens[0:3]-pre_sens[0:3])
            # ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
            control = limit(ud,mtq_max)
        else:
            if t<tlim1:
                #PD to zeroquat
                qdes = zeroquat
            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            elif t<tlim4:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,0,1])
            elif t<tlim5:
                #PID to [0,1,0,0]
                qdes = np.array([1,0,0,0])
            elif t<tlim6:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

            offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
            control = limit(ud-offset_vec,mtq_max)
        # print('ud   ', ud)
        print('ctrl ',control)
        # print(np.concatenate([j.bias_std_rate for j in est_sat.sensors]).T)
        # print([j.bias_std_rate for j in real_sat.sensors])
        # print(np.sqrt(np.diag(est.full_state.int_cov)[-9:]))

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]


        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        pre_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        prev_full_state = real_full_state.copy()
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state.copy(), method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-10)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
        # breakpoint()
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    mbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])/np.array([j.scale for j in real_sat.sensors[0:3]])
    gbdiff = (est_state_hist[:,10:13]-state_hist[:,10:13])*180.0/math.pi
    sbdiff = (est_state_hist[:,13:16]-state_hist[:,13:16])/1e4
    # dipdiff =  (est_state_hist[:,16:19]-state_hist[:,16:19])/1e4
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    # range = (tf-t0)/dt
    log_angdiff = np.log(angdiff[int(0.1*ind):int(0.9*ind)])
    tc = np.polyfit(log_angdiff,range(int(0.9*ind)-int(0.1*ind)),1)[1]
    res = [np.mean(matrix_row_norm(avdiff.T)), np.amax(matrix_row_norm(avdiff.T)), np.mean(angdiff), np.amax(angdiff), np.where(angdiff<1)[0],tc]
    # breakpoint()
    # return res
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(np.log10(matrix_row_norm(gbdiff)),norm=False,title = "log10 gbias diff in deg/s")
    plot_the_thing(sbdiff,norm=True,title = "sbias diff")
    plot_the_thing(np.log10(matrix_row_norm(sbdiff)),norm=False,title = "log10 sbias diff")
    plot_the_thing(mbdiff,norm=True,title = "mbias diff in nT")
    plot_the_thing(np.log10(matrix_row_norm(mbdiff)),norm=False,title = "log10 mbias diff in nT")
    # plot_the_thing(dipdiff,norm=True,title = "dipole est diff in Am2")
    plot_the_thing(np.stack([est_state_hist[:,7:10]/np.array([j.scale for j in real_sat.sensors[0:3]]),state_hist[:,7:10]/np.array([j.scale for j in real_sat.sensors[0:3]])]),norm=False,title = "mbias in nT")
    plot_the_thing(np.stack([est_state_hist[:,10:13],state_hist[:,10:13]])*180.0/math.pi,norm=False,title = "gbias in nT")
    plot_the_thing(np.stack([est_state_hist[:,13:16],state_hist[:,13:16]]),norm=False,title = "sbias")
    plot_the_thing(np.stack([est_state_hist[:,0:3],state_hist[:,0:3]])*180.0/math.pi,norm=False,title = "av in deg/s")
    plot_the_thing(np.stack([est_state_hist[:,3:7],state_hist[:,3:7]]),norm=False,title = "quat")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")
    plot_the_thing(np.log10(angdiff),title = "log angdiff (log10 deg) hist")

    breakpoint()

def test_ukf_quat_not_vec_w_dist_w_smgbias_wdipole(cov_est_mult=1,mrp_ic = 0*1e-20,av_ic = 0*1e-12,al = 1,kap = 1.0,bet = 2,sep_int=False,sep_sens = False):
    np.random.seed(1)
    t0 = 0
    tf = 60*45
    tlim00 = 10
    tlim0 = 5*60
    tlim1 = 10*60
    tlim2 = 20*60
    tlim3 = 35*60
    tlim4 = 50*60
    tlim5 = 65*60
    tlim6 = 80*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    real_sat = create_BC_sat(real=True,include_mtqbias = False,rand=True)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    # w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180

    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    try:
        with open("myorb", "rb") as fp:   #unPickling
            orb = pickle.load(fp)
    except:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_BC_sat(real=False,include_mtqbias = False,estimate_dipole = True)
    estimate = np.zeros(19)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*10,np.eye(3)*(1e-3)**2.0,np.eye(3)*(0.2*math.pi/180)**2.0,np.eye(3)*(3e-2)**2.0,np.eye(3)*0.25**2.0)*cov_est_mult

    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    # int_cov =  dt*block_diag(dt*np.block([[np.eye(3)*(1/dt)*av_ic,0.5*np.eye(3)*av_ic],[0.5*np.eye(3)*av_ic,(1/3)*np.eye(3)*dt*av_ic + np.eye(3)*mrp_ic]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.attitude_sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    int_cov =  block_diag(np.block([[np.eye(3)*av_ic,0.5*np.eye(3)*av_ic*dt],[0.5*np.eye(3)*av_ic*dt,(1/3)*np.eye(3)*dt*dt*av_ic + np.eye(3)*mrp_ic*dt]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)
    int_cov =  block_diag(np.eye(3)*(1e-4)**2.0,1e-12*np.eye(3),np.diagflat([j.bias_std_rate**2.0*j.scale*j.scale for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)

    # int_cov =  block_diag(np.eye(3)*0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    mtq_max = [j.max for j in est_sat.actuators]
    est = SRUKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.include_sens_noise_separately = sep_sens
    est.include_int_noise_separately = sep_int
    est.included_int_noise_where = 2
    est.use_cross_term = True
    est.al = 0.001#1.0#1#1.0#0.001#0.01#0.01#0.99#0.9#1e-1#0.99#1#1e-3#e-1#e-3#1#1e-1#al# 1e-1
    est.kap = 0#3-21#3-15#3-12#0#3-18#3-18#0#3-18#-10#0#-15#3-18#0#3-21##0#3-3-21*2-9#0#3-24#0#1-21#3-21#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    est.bet = 2
    est.scale_nonseparate_adds = False

    est.bet = bet#2#0#1.0#2.0#2.0#-1.0#2.0

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)



    kw = 150
    ka = 10
    prev_full_state = np.concatenate([state.copy()]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])

    # breakpoint()

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])
        real_dist = np.concatenate([j.main_param for j in real_sat.disturbances if j.time_varying])
        real_full_state = np.concatenate([state.copy(),real_sbias.copy(),real_dist.copy()])
        sens = real_sat.sensor_values(state,real_vecs)
        # for j in range(len(est.use_state.val)):
        #     if j==0:
        #         est.use_state.val[j] = prev_full_state[j].copy()
        # est.use_state.val[3:7] = normalize(est.use_state.val[3:7])
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',real_sbias)
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm(4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi)
        print('gb ',norm(est_state[10:13]-real_sbias[3:6]))
        print((est_state[10:13]-real_sbias[3:6]))
        print(autovar[9:12]*180.0/math.pi)

        print('mb ',norm(est_state[7:10]-real_sbias[0:3]))
        print((est_state[7:10]-real_sbias[0:3]))
        print(autovar[6:9])

        print('sb ',norm(est_state[13:16]-real_sbias[6:9]))
        print((est_state[13:16]-real_sbias[6:9]))
        print(autovar[12:15])

        print('dip ',norm(est_state[16:19]-real_dist))
        print(est_state[16:19]-real_dist)
        print(autovar[15:18])
        errvec = real_full_state-est_state
        errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(est_state[3:7]),real_full_state[3:7]),5),errvec[7:]])
        print('err vec ',errvec )
        coverr = np.sqrt(errvec.T@np.linalg.inv(est.use_state.cov)@errvec)
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        print('cov err ',coverr)
        # print(orbt.B,real_vecs['b'],est_vecs['b'])
        # print(orbt.S-orbt.R,real_vecs['s']-real_vecs['r'],est_vecs['s']-est_vecs['r'])
        # print(sens)
        # print(est.sat.sensor_values(est_state,est_vecs))
        # print( np.array([j.clean_reading(est_state,est_vecs) for j in est_sat.sensors]))
        # # breakpoint()
        # print(autovar[6:9])

        #find control
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        nB2 = norm(orbt.B)
        Bbody = rot_mat(q).T@orbt.B
        w_err = w-wdes
        if t<tlim00:
            control = np.zeros(3)
        elif t<tlim0:
            #bdot
            # Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e8*(sens[0:3]-pre_sens[0:3])
            # ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
            control = limit(ud,mtq_max)
        else:
            if t<tlim1:
                #PD to zeroquat
                qdes = zeroquat
            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            elif t<tlim4:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,0,1])
            elif t<tlim5:
                #PID to [0,1,0,0]
                qdes = np.array([1,0,0,0])
            elif t<tlim6:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

            offset_vec = est_state[16:19]#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
            control = limit(ud-offset_vec,mtq_max)
        # print('ud   ', ud)
        print('ctrl ',control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]


        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        pre_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        prev_full_state = real_full_state.copy()
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state.copy(), method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-10)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
        # breakpoint()
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])*180.0/math.pi
    sbdiff = (est_state_hist[:,10:13]-state_hist[:,10:13])
    mbdiff = (est_state_hist[:,13:16]-state_hist[:,13:16])/1e4
    dipdiff =  (est_state_hist[:,16:19]-state_hist[:,16:19])/1e4
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    # range = (tf-t0)/dt
    log_angdiff = np.log(angdiff[int(0.1*ind):int(0.9*ind)])
    tc = np.polyfit(log_angdiff,range(int(0.9*ind)-int(0.1*ind)),1)[1]
    res = [np.mean(matrix_row_norm(avdiff.T)), np.amax(matrix_row_norm(avdiff.T)), np.mean(angdiff), np.amax(angdiff), np.where(angdiff<1)[0],tc]
    # breakpoint()
    # return res
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(sbdiff,norm=True,title = "sbias diff")
    plot_the_thing(mbdiff,norm=True,title = "mbias diff in nT")
    plot_the_thing(dipdiff,norm=True,title = "dipole est diff in Am2")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")
    plot_the_thing(np.log10(angdiff),title = "log angdiff (log10 deg) hist")

    breakpoint()


def test_ukf_quat_not_vec_w_dist_w_smgbias_wdipole_fancysat(cov_est_mult=1,mrp_ic = 1e-8,av_ic = 1e-12,al = 1,kap = 1.0,bet = 2,sep_int=False,sep_sens = False):
    np.random.seed(1)
    t0 = 0
    tf = 60*45
    tlim00 = 10
    tlim0 = 5*60
    tlim1 = 10*60
    tlim2 = 20*60
    tlim3 = 35*60
    tlim4 = 50*60
    tlim5 = 65*60
    tlim6 = 80*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    real_sat = create_fancy_BC_sat(real=True,include_mtqbias = False,rand=True)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    # w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180

    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    # q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    try:
        with open("myorb", "rb") as fp:   #unPickling
            orb = pickle.load(fp)
    except:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    est_sat = create_fancy_BC_sat(real=False,include_mtqbias = False,estimate_dipole = True)
    estimate = np.zeros(19)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*10,np.eye(3)*(1e-3)**2.0,np.eye(3)*(0.1*math.pi/180)**2.0,np.eye(3)*(6e-2)**2.0,np.eye(3)*0.25**2.0)*cov_est_mult
    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    int_cov =  block_diag(np.eye(3)*av_ic,mrp_ic*np.eye(3),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.sensors if j.has_bias and j.estimated_bias]),np.eye(3)*0.0001**2.0)

    # int_cov =  block_diag(np.eye(3)*0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    mtq_max = [j.max for j in est_sat.actuators]
    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.use_cross_term = True
    est.include_sens_noise_separately = sep_sens
    est.include_int_noise_separately = sep_int
    est.included_int_noise_where = 2

    est.al = 1e-3#e-3#1#1e-1#al# 1e-1
    # est.kap = 3-sum([est.sat.control_cov().shape[0],cov_estimate.shape[0]])#
    est.kap =  18#1#kap#0.1#kap#3-sum([cov_estimate.shape[0]])#
    # est.kap = 3-sum([est.sat.control_cov().shape[0],2*cov_estimate.shape[0],est.sat.sensor_cov().shape[0]])#
    # est.kap = 3-sum([est.sat.control_cov().shape[0],2*cov_estimate.shape[0],est.sat.sensor_cov().shape[0]])#
    # est.kap = 3-sum([est.sat.control_cov().shape[0],2*cov_estimate.shape[0]])
    # est.kap = 3-sum([cov_estimate.shape[0],est.sat.sensor_cov().shape[0]])
    est.bet = 2#1.5#bet#2#0#1.0#2.0#2.0#-1.0#2.0
    est.bet0 = 2#1.5
    est.betr = 0#-0.001

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)



    kw = 1500
    ka = 10
    prev_full_state = np.concatenate([state.copy()]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])

    # breakpoint()

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])
        real_dist = np.concatenate([j.main_param for j in real_sat.disturbances if j.time_varying])
        real_full_state = np.concatenate([state.copy(),real_sbias.copy(),real_dist.copy()])
        sens = real_sat.sensor_values(state,real_vecs)
        # for j in range(len(est.use_state.val)):
        #     if j==0:
        #         est.use_state.val[j] = prev_full_state[j].copy()
        # est.use_state.val[3:7] = normalize(est.use_state.val[3:7])
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',real_sbias)
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm(4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi)
        print('gb ',norm(est_state[10:13]-real_sbias[3:6]))
        print((est_state[10:13]-real_sbias[3:6]))
        print(autovar[9:12]*180.0/math.pi)

        print('mb ',norm(est_state[7:10]-real_sbias[0:3]))
        print((est_state[7:10]-real_sbias[0:3]))
        print(autovar[6:9])

        print('sb ',norm(est_state[13:16]-real_sbias[6:9]))
        print((est_state[13:16]-real_sbias[6:9]))
        print(autovar[12:15])

        print('dip ',norm(est_state[16:19]-real_dist))
        print(est_state[16:19]-real_dist)
        print(autovar[15:18])
        # errvec = est_state-real_full_state
        # errvec = np.concatenate([errvec[0:3],quat_to_vec3(quat_mult(quat_inv(real_full_state[3:7]),est_state[3:7]),5),errvec[7:]])
        # print('err vec ',errvec )
        # invcov = np.linalg.inv(est.use_state.cov)
        # coverr = np.sqrt(errvec.T@invcov@errvec)
        # print('cov err ',coverr,np.log10(coverr))
        # try:
        #     ea,ee= np.linalg.eig(invcov)
        #     sqrt_invcov = ee@np.diagflat(np.sqrt(ea))#*np.linalg.cholesky(invcov)
        #     print(sum(sqrt_invcov@sqrt_invcov.T))
        #     print('scaled state err', sqrt_invcov@errvec)
        #     print('scaled state err', sqrt_invcov.T@errvec)
        # except:
        #     pass
        # print('cov err vec ', coverr)
        # print('cov err vec ',norm(coverr))
        # print(orbt.B,real_vecs['b'],est_vecs['b'])
        # print(orbt.S-orbt.R,real_vecs['s']-real_vecs['r'],est_vecs['s']-est_vecs['r'])
        # print(sens)
        # print(est.sat.sensor_values(est_state,est_vecs))
        # print( np.array([j.clean_reading(est_state,est_vecs) for j in est_sat.sensors]))
        # # breakpoint()
        # print(autovar[6:9])

        #find control
        q = est.use_state.val[3:7]
        w = est.use_state.val[0:3]
        wdes = np.zeros(3)
        nB2 = norm(orbt.B)
        Bbody = rot_mat(q).T@orbt.B
        w_err = w-wdes
        if t<tlim00:
            control = np.zeros(3)
        elif t<tlim0:
            #bdot
            # Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e8*(sens[0:3]-pre_sens[0:3])
            # ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
            control = limit(ud,mtq_max)
        else:
            if t<tlim1:
                #PD to zeroquat
                qdes = zeroquat
            elif t<tlim2:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            elif t<tlim3:
                #PID to [0,1,0,0]
                qdes = np.array([0,1,0,0])
            elif t<tlim4:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,0,1])
            elif t<tlim5:
                #PID to [0,1,0,0]
                qdes = np.array([1,0,0,0])
            elif t<tlim6:
                #PID to [0,1,0,0]
                qdes = np.array([0,0,1,0])
            else:
                #PID to zeroquat
                qdes = zeroquat
            q_err = quat_mult(quat_inv(qdes),q)
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

            offset_vec = est_state[16:19]#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
            control = limit(ud-offset_vec,mtq_max)
        # print('ud   ', ud)
        print('ctrl ',control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias]+[j.main_param for j in real_sat.disturbances if j.time_varying])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]


        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        pre_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        prev_full_state = real_full_state.copy()
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state.copy(), method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-10)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])*180.0/math.pi
    sbdiff = (est_state_hist[:,10:13]-state_hist[:,10:13])
    mbdiff = (est_state_hist[:,13:16]-state_hist[:,13:16])/1e4
    dipdiff =  (est_state_hist[:,16:19]-state_hist[:,16:19])/1e4
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    # range = (tf-t0)/dt
    log_angdiff = np.log(angdiff[int(0.1*ind):int(0.9*ind)])
    tc = np.polyfit(log_angdiff,range(int(0.9*ind)-int(0.1*ind)),1)[1]
    res = [np.mean(matrix_row_norm(avdiff.T)), np.amax(matrix_row_norm(avdiff.T)), np.mean(angdiff), np.amax(angdiff), np.where(angdiff<1)[0],tc]
    # breakpoint()
    # return res
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(np.log10(angdiff),title = "log ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(sbdiff,norm=True,title = "sbias diff")
    plot_the_thing(mbdiff,norm=True,title = "mbias diff in nT")
    plot_the_thing(dipdiff,norm=True,title = "dipole est diff in Am2")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")
    breakpoint()


def test_ukf_quat_not_vec_w_dist_w_smgbias_specified_norand(orb = None,cov_est_mult=1,mrp_ic = 0*1e-12,av_ic = 0*1e-6,al = 1,kap = 3-15,bet = 2,sep_int=False,sep_sens = False):
    np.random.seed(1)
    t0 = 0
    tf = 60*20
    tlim00 = 10
    tlim0 = 1*60
    tlim1 = 5*60
    tlim2 = 10*60
    tlim3 = 20*60
    tlim4 = 30*60
    tlim5 = 40*60
    tlim6 = 50*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 1.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]


    gyro_bias0_std = np.random.uniform(0.01,0.2)*math.pi/180.0
    gyro_bias0_std = 0.05*math.pi/180.0
    real_gyro_bias0 = gyro_bias0_std*random_n_unit_vec(3)
    real_gyro_bias0 = gyro_bias0_std*normalize(np.array([1,-2,3]))
    gyro_bsr = 0.0004*math.pi/180.0#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s


    mtm_bias0_std = np.random.uniform(1e-5,1e-3)
    mtm_bias0_std = 1e-4# np.random.uniform(1e-5,1e-3)
    real_mtm_bias0 = mtm_bias0_std*random_n_unit_vec(3)
    real_mtm_bias0 = mtm_bias0_std*normalize(np.array([-1,1,1]))
    mtm_bsr = 1e-5 #1nT/sec
    mtm_std = 3*1e-3

    mtms = [MTM(j,mtm_std,has_bias = True,bias = np.dot(real_mtm_bias0,j).item(),use_noise = True,bias_std_rate = mtm_bsr,scale = 1e4) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]


    sun_eff = 0.3
    noise_sun = 0.001*sun_eff #0.1% of range
    sun_bias0_std = 0.1
    real_sun_bias0 = [0.01,0,-0.0005]#[np.random.uniform(-sun_bias0_std,sun_bias0_std)*sun_eff for j in range(6)]
    sun_bsr = 0.00001*sun_eff #0.001% of range /sec

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j],use_noise = True,bias_std_rate = sun_bsr) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [drag,gg]
    J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
        [5.88304e-05, 0.03409127827, -0.00012334756],
        [-0.00671361357, -0.00012334756, 0.01004091997]])
    J = 0.5*(J+J.T)
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180

    q0 = random_n_unit_vec(4)#np.array([math.sqrt(2)/2,math.sqrt(2)/2,0,0])#random_n_unit_vec(4)
    q0 = quat_mult(q0,mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4)))
    if orb is None:
        try:
            with open("myorb", "rb") as fp:   #unPickling
                orb = pickle.load(fp)
        except:
            os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
            orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = mtm_bsr,estimate_bias = True,scale = 1e4) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]

    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [drag_est,gg_est]
    J_EST = J.copy()
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(16)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*100,np.eye(3)*(2*1e-3)**2.0,np.eye(3)*(2*0.2*math.pi/180)**2.0,np.eye(3)*(2*3e-2)**2.0)*cov_est_mult
    int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    int_cov =  dt*block_diag(dt*np.block([[np.eye(3)*(1/dt)*av_ic,0.5*np.eye(3)*av_ic],[0.5*np.eye(3)*av_ic,(1/3)*np.eye(3)*dt*av_ic + np.eye(3)*mrp_ic]]),np.diagflat([j.bias_std_rate**2.0 for j in est_sat.attitude_sensors if j.has_bias and j.estimated_bias]))
    # Qinv0 = np.linalg.inv(int_cov)
    # int_cov =  block_diag(np.eye(3)*0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    #fuck with, in order: int_cov, initial cov_est multiplier,bet, kap. Maybe with al but this seems ok. also can do int noise separately on/off

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.include_sens_noise_separately = sep_sens
    est.include_int_noise_separately = sep_int

    est.al = al# 1e-1
    # est.kap = 3-sum([est.sat.control_cov().shape[0],cov_estimate.shape[0]])#
    est.kap =  kap#3-sum([cov_estimate.shape[0]])#
    # est.kap = 3-sum([est.sat.control_cov().shape[0],2*cov_estimate.shape[0],est.sat.sensor_cov().shape[0]])#
    # est.kap = 3-sum([est.sat.control_cov().shape[0],2*cov_estimate.shape[0],est.sat.sensor_cov().shape[0]])#
    # est.kap = 3-sum([est.sat.control_cov().shape[0],2*cov_estimate.shape[0]])
    # est.kap = 3-sum([cov_estimate.shape[0],est.sat.sensor_cov().shape[0]])
    est.bet = bet#2#0#1.0#2.0#2.0#-1.0#2.0
    est.included_int_noise_where = 2
    est.scale_nonseparate_adds = False

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)



    kw = 2000
    ka = 20
    prev_full_state = np.concatenate([state.copy()]+[j.bias for j in real_sat.sensors if j.has_bias])

    while t<tf:
        print(t,ind)
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])
        real_full_state = np.concatenate([state.copy(),real_sbias.copy()])
        sens = real_sat.sensor_values(state,real_vecs)
        # for j in range(len(est.use_state.val)):
        #     if j==0:
        #         est.use_state.val[j] = prev_full_state[j].copy()
        # est.use_state.val[3:7] = normalize(est.use_state.val[3:7])
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print('real state ',state)
        print('est  state ',est_state[0:7])
        print('real sbias ',real_sbias)
        print(' est sbias ',est_state[7:])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        autovar = np.sqrt(np.diagonal(est.use_state.cov))
        est_vecs = os_local_vecs(orbt,est_state[3:7])
        print('av ',(norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        print(((est_state[0:3]-state[0:3])*180.0/math.pi))
        print(autovar[0:3]*180.0/math.pi)
        print('ang ',(180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print('mrp ',(180/np.pi)*norm(4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        # print((180/np.pi)*norm(4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0)))
        print((180/np.pi)*4.0*np.arctan(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        # print((180/np.pi)*4.0*(quat_to_mrp(quat_mult(quat_inv(state[3:7]),est_state[3:7]))/2.0))
        print(4*np.arctan(autovar[3:6]/2.0)*180.0/math.pi)
        print('gb ',norm(est_state[10:13]-real_sbias[3:6]))
        print((est_state[10:13]-real_sbias[3:6]))
        print(autovar[9:12]*180.0/math.pi)

        print('mb ',norm(est_state[7:10]-real_sbias[0:3]))
        print((est_state[7:10]-real_sbias[0:3]))
        print(autovar[6:9])

        print('sb ',norm(est_state[13:16]-real_sbias[6:9]))
        print((est_state[13:16]-real_sbias[6:9]))
        print(autovar[12:15])
        # print(orbt.B,real_vecs['b'],est_vecs['b'])
        # print(orbt.S-orbt.R,real_vecs['s']-real_vecs['r'],est_vecs['s']-est_vecs['r'])
        # print(sens)
        # print(est.sat.sensor_values(est_state,est_vecs))
        # print( np.array([j.clean_reading(est_state,est_vecs) for j in est_sat.sensors]))
        # # breakpoint()
        # print(autovar[6:9])

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = 1e12*(sens[0:3]-pre_sens[0:3])
            # ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim3:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,1,0,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        elif t<tlim4:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,0,1])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        elif t<tlim5:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([1,0,0,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        elif t<tlim6:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,0,1,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]


        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        pre_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # prev_full_state = real_full_state.copy()
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state.copy(), method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-10)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    gbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])*180.0/math.pi
    sbdiff = (est_state_hist[:,10:13]-state_hist[:,10:13])
    mbdiff = (est_state_hist[:,13:16]-state_hist[:,13:16])/1e4
    autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    # range = (tf-t0)/dt
    # breakpoint()
    log_angdiff = np.log(angdiff[int(0.1*ind):int(0.9*ind)])
    tc = np.polyfit(log_angdiff,range(int(0.9*ind)-int(0.1*ind)),1)[0]
    converged_times = np.where(angdiff<1)[0]
    if len(converged_times)>0:
        time_to_conv = (converged_times[0],np.where(angdiff>=1)[0][-1]+1)
        av_on_conv = matrix_row_norm(avdiff[time_to_conv[0]:,:].T)
        ang_on_conv = angdiff[time_to_conv[0]:,]

        res = [np.mean(av_on_conv), np.amax(av_on_conv), np.mean(ang_on_conv), np.amax(ang_on_conv),time_to_conv,tc]
    else:
        time_to_conv = (np.nan,np.nan)
        res = [np.nan,np.nan,np.nan,np.nan,time_to_conv,tc]

    # breakpoint()
    # return res
    print(np.amax(angdiff))
    print(np.amax(avdiff,0))
    plot_the_thing(angdiff,title = "ang diff in deg")
    plot_the_thing(avdiff,norm=True,title = "av diff in deg/s")
    plot_the_thing(gbdiff,norm=True,title = "gbias diff in deg/s")
    plot_the_thing(sbdiff,norm=True,title = "sbias diff")
    plot_the_thing(mbdiff,norm=True,title = "mbias diff in nT")
    plot_the_thing(autocov_hist,title = "cov_hist")
    plot_the_thing(state_hist[:,3:7],title = "quat hist")
    plot_the_thing(state_hist[:,0:3],norm=True,title = "av hist")

    breakpoint()


def test_ukf_quat_not_vec_w_dist_w_smgbias_for_gridtest(real_sat=None,est_sat=None,orb = None,cov_est_mult=1,mrp_ic = 1e-10,av_ic = 1e-4,al = 1e-2,kap = 0,bet = 2,sep_int=True,sep_sens = True):
    np.random.seed(1)
    t0 = 0
    tf = 60*10
    tlim00 = 10
    tlim0 = 1*60
    tlim1 = 5*60
    tlim2 = 10*60
    tlim3 = 20*60
    dt = 1
    # np.set_printoptions(precision=3)

    #
    #real_sat
    if real_sat is None:
        mtq_bias0_std = np.random.uniform(0.01,0.1)
        mtq_std = 0.0001
        mtq_max = 1.0
        acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]


        gyro_bias0_std = np.random.uniform(0.01,0.2)*math.pi/180.0
        gyro_bias0_std = 0.05*math.pi/180.0
        real_gyro_bias0 = gyro_bias0_std*random_n_unit_vec(3)
        real_gyro_bias0 = gyro_bias0_std*normalize(np.array([1,-2,3]))
        gyro_bsr = 0.0004*math.pi/180.0#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
        gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s


        mtm_bias0_std = np.random.uniform(1e-5,1e-3)
        mtm_bias0_std = 1e-4# np.random.uniform(1e-5,1e-3)
        real_mtm_bias0 = mtm_bias0_std*random_n_unit_vec(3)
        real_mtm_bias0 = mtm_bias0_std*normalize(np.array([-1,1,1]))
        mtm_bsr = 1e-5 #1nT/sec
        mtm_std = 3*1e-3

        mtms = [MTM(j,mtm_std,has_bias = True,bias = np.dot(real_mtm_bias0,j).item(),use_noise = True,bias_std_rate = mtm_bsr,scale = 1e4) for j in unitvecs]
        gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]


        sun_eff = 0.3
        noise_sun = 0.001*sun_eff #0.1% of range
        sun_bias0_std = 0.1
        real_sun_bias0 = [0.01,0,-0.0005]#[np.random.uniform(-sun_bias0_std,sun_bias0_std)*sun_eff for j in range(6)]
        sun_bsr = 0.00001*sun_eff #0.001% of range /sec

        suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j],use_noise = True,bias_std_rate = sun_bsr) for j in range(3)]

        drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                        [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                        [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                        [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                        [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                        [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
        drag = Drag_Disturbance(drag_faces)
        gg = GG_Disturbance()
        dists = []#[drag,gg]
        J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
            [5.88304e-05, 0.03409127827, -0.00012334756],
            [-0.00671361357, -0.00012334756, 0.01004091997]])
        J = 0.5*(J+J.T)
        real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    else:
        mtq_max = real_sat.actuators[0].max
        mtm_bsr = real_sat.sensors[[j for j in range(len(real_sat.sensors)) if isinstance(real_sat.sensors[j],MTM)][0]].bias_std_rate
        gyro_bsr = real_sat.sensors[[j for j in range(len(real_sat.sensors)) if isinstance(real_sat.sensors[j],Gyro)][0]].bias_std_rate
        sun_bsr = real_sat.sensors[[j for j in range(len(real_sat.sensors)) if isinstance(real_sat.sensors[j],SunSensorPair)][0]].bias_std_rate

    w0 = np.array([0,2*math.pi/(60*90),0])#random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = mrp_to_quat(2*np.tan(np.array([-50,50,160])*(math.pi/180.0)/4))
    if orb is None:
        os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
        orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    if est_sat is None:

        acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

        mtms_est = [MTM(j,mtm_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = mtm_bsr,estimate_bias = True,scale = 1e4) for j in unitvecs]
        gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]

        suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(3)]

        drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                        [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                        [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                        [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                        [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                        [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
        drag_est = Drag_Disturbance(drag_faces)
        gg_est = GG_Disturbance()
        dists_est = []#[drag_est,gg_est]
        J_EST =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
            [5.88304e-05, 0.03409127827, -0.00012334756],
            [-0.00671361357, -0.00012334756, 0.01004091997]])
        J_EST = 0.5*(J_EST+J_EST.T)
        est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(16)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(math.pi/180.0)**2.0,np.eye(3)*100,np.eye(3)*(2*1e-3)**2.0,np.eye(3)*(2*0.2*math.pi/180)**2.0,np.eye(3)*(2*3e-2)**2.0)*cov_est_mult
    # int_cov =  block_diag(np.eye(3)*(1e-7)**2.0,(1e-8)**2.0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)
    int_cov =  block_diag(np.eye(3)*av_ic,mrp_ic*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    # int_cov =  block_diag(np.eye(3)*0,0*np.eye(3),np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*sun_bsr**2.0)

    #fuck with, in order: int_cov, initial cov_est multiplier,bet, kap. Maybe with al but this seems ok. also can do int noise separately on/off

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.include_sens_noise_separately = sep_sens
    est.include_int_noise_separately = sep_int

    est.al = al# 1e-1
    # est.kap = 3-sum([est.sat.control_cov().shape[0],cov_estimate.shape[0]])#
    est.kap =  kap#3-sum([cov_estimate.shape[0]])#
    est.bet = bet#2#0#1.0#2.0#2.0#-1.0#2.0

    t = t0
    # t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    # orb_hist = []
    # control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    # cov_hist = []
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)

    kw = 2000
    ka = 20
    # prev_full_state = np.concatenate([state.copy()]+[j.bias for j in real_sat.sensors if j.has_bias])

    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])
        real_full_state = np.concatenate([state.copy(),real_sbias.copy()])
        sens = real_sat.sensor_values(state,real_vecs)
        # for j in range(len(est.use_state.val)):
        #     if j==0:
        #         est.use_state.val[j] = prev_full_state[j].copy()
        # est.use_state.val[3:7] = normalize(est.use_state.val[3:7])
        est.update(control,sens,orbt)
        # est_state = est.use_state.val

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = 1e12*(sens[0:3]-pre_sens[0:3])
            # ud = -1e12*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2


        control = limit(ud,mtq_max)

        #save info
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        # orb_hist += [orbt]
        # control_hist[ind,:] = control
        # cov_hist += [est.use_state.cov]


        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        pre_sens = sens.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        # prev_full_state = real_full_state.copy()
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state.copy(), method="RK45", args=(control, prev_os,orbt), rtol=1e-6, atol=1e-10)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
    # breakpoint()
    angdiff = (180/np.pi)*np.arccos(-1 + 2*np.clip(np.sum(est_state_hist[:,3:7]*state_hist[:,3:7],axis = 1),-1,1)**2.0 )
    avdiff = (est_state_hist[:,0:3]-state_hist[:,0:3])*180.0/math.pi
    # gbdiff = (est_state_hist[:,7:10]-state_hist[:,7:10])*180.0/math.pi
    # sbdiff = (est_state_hist[:,10:13]-state_hist[:,10:13])
    # mbdiff = (est_state_hist[:,13:16]-state_hist[:,13:16])/1e4
    # autocov_hist = np.stack([np.diagonal(j) for j in cov_hist])
    # range = (tf-t0)/dt
    # breakpoint()
    log_angdiff = np.log(angdiff[int(0.1*ind):int(0.9*ind)])
    tc = np.polyfit(log_angdiff,range(int(0.9*ind)-int(0.1*ind)),1)[0]
    converged_times = np.where(angdiff<1)[0]
    if len(converged_times)>0:
        time_to_conv = (converged_times[0],np.where(angdiff>=1)[0][-1]+1)
        av_on_conv = matrix_row_norm(avdiff[time_to_conv[0]:,:].T)
        ang_on_conv = angdiff[time_to_conv[0]:,]

        res = [np.mean(av_on_conv), np.amax(av_on_conv), np.mean(ang_on_conv), np.amax(ang_on_conv),time_to_conv,tc]
    else:
        time_to_conv = (np.nan,np.nan)
        res = [np.nan,np.nan,np.nan,np.nan,time_to_conv,tc]

    # breakpoint()
    return res

def test_grid_ukf_quat_not_vec_w_all():

    sep_sens_list = [True,False]#[True,False]
    sep_int_list = [True,False]#[True,False]
    cov_est_mult_list = [1,10]#[1e-2,1,1e2]#[1e-1,1,1e1]#[1e-2,1,1e2]# [1e-2,1,100,1e4]
    mrp_ic_list = [(10**(j))**2.0 for j in [-10,-8,-6]]#[(10**(j))**2.0 for j in [-6,-4,-2]] #smaller seems to be better
    av_ic_list = [(10**(j))**2.0 for j in [-2,-1,0]]
    al_list = [1e-4,1e-3,1e-2]#[1e-4,1e-2,1]#[1e-4,1e-3,1e-2,1e-1,1] #smaller is bettter?
    kap_list = [-10,0,1,3]#[-3,-1,0,1,3]
    bet_list = [1,2]#[1,2,3] # 1 may be better than 2
    lol = [sep_sens_list,sep_int_list,cov_est_mult_list,mrp_ic_list,av_ic_list,al_list,kap_list, bet_list]
    results = []
    ind = 0
    total = np.prod([len(j) for j in lol])
    tf = 10*60
    t0 = 0
    dt = 1
    os0 = Orbital_State(0.22-1*sec2cent,7e3*np.array([0,math.sqrt(2)/2,math.sqrt(2)/2]),np.array([8,0,0]))
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)

    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 1.0
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]


    gyro_bias0_std = np.random.uniform(0.01,0.2)*math.pi/180.0
    gyro_bias0_std = 0.05*math.pi/180.0
    real_gyro_bias0 = gyro_bias0_std*random_n_unit_vec(3)
    real_gyro_bias0 = gyro_bias0_std*normalize(np.array([1,-2,3]))
    gyro_bsr = 0.0004*math.pi/180.0#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s


    mtm_bias0_std = np.random.uniform(1e-5,1e-3)
    mtm_bias0_std = 1e-4# np.random.uniform(1e-5,1e-3)
    real_mtm_bias0 = mtm_bias0_std*random_n_unit_vec(3)
    real_mtm_bias0 = mtm_bias0_std*normalize(np.array([-1,1,1]))
    mtm_bsr = 1e-5 #1nT/sec
    mtm_std = 3*1e-3

    mtms = [MTM(j,mtm_std,has_bias = True,bias = np.dot(real_mtm_bias0,j).item(),use_noise = True,bias_std_rate = mtm_bsr,scale = 1e4) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]


    sun_eff = 0.3
    noise_sun = 0.001*sun_eff #0.1% of range
    sun_bias0_std = 0.1
    real_sun_bias0 = [0.01,0,-0.0005]#[np.random.uniform(-sun_bias0_std,sun_bias0_std)*sun_eff for j in range(6)]
    sun_bsr = 0.00001*sun_eff #0.001% of range /sec

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j],use_noise = True,bias_std_rate = sun_bsr) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = []#[drag,gg]
    J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
        [5.88304e-05, 0.03409127827, -0.00012334756],
        [-0.00671361357, -0.00012334756, 0.01004091997]])
    J = 0.5*(J+J.T)
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)

    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = mtm_bsr,estimate_bias = True,scale = 1e4) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]

    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(3)]

    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = []#[drag_est,gg_est]
    J_EST =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
        [5.88304e-05, 0.03409127827, -0.00012334756],
        [-0.00671361357, -0.00012334756, 0.01004091997]])
    J_EST = 0.5*(J_EST+J_EST.T)
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)




    t0 = time.process_time()
    lastt = t0
    for v1 in sep_sens_list:
        for v2 in sep_int_list:
            for v3 in cov_est_mult_list:
                for v4 in mrp_ic_list:
                    for v5 in av_ic_list:
                        for v6 in al_list:
                            for v7 in kap_list:
                                for v8 in bet_list:
                                    try:
                                        vals = test_ukf_quat_not_vec_w_dist_w_smgbias_for_gridtest(real_sat=real_sat,est_sat = est_sat,orb=orb,cov_est_mult=v3,mrp_ic = v4,av_ic = v5,al = v6,kap = v7,bet = v8,sep_int=v2,sep_sens = v1)
                                    except:
                                        vals =  [np.nan,np.nan,np.nan,np.nan,(np.nan,np.nan),np.nan]
                                        print("FAILED!")
                                    tj = time.process_time()
                                    results += [([v1,v2,v3,v4,v5,v6,v7,v8],tj-lastt,vals)]
                                    ind += 1
                                    print(str(ind)," of ",str(total)," in ",str(tj-lastt),". Time so far: ",str(tj-t0),". Expected time remaining: ",str((total-ind)*(tj-t0)/(ind)))
                                    lastt = tj
    params = ["tuple containing list of lists, followed by list of results. each result is a tuple containing, in order: a list of the values used from each list of lists,the time that case took to run, then the result values in a list.",(lol)]
    try:
        with open("grid_test__"+time.strftime("%Y%m%d-%H%M%S"), "wb") as fp:   #Pickling
            pickle.dump(params+results, fp)
    except:
        print("DIDNT SAVE")
    breakpoint()


def test_ukf_basic_quat_not_vec():

    t0 = 0
    tf = 60*10
    tlim00 = 60
    tlim0 = 10*60
    tlim1 = 20*60
    tlim2 = 30*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    mtq_bias0_std = np.random.uniform(0.01,0.1)
    mtq_max = 0.5
    mtq_std = 0.0001
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    mtm_std = 300*1e-9
    gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]


    sun_eff = 0.3
    noise_sun = 0.0001*sun_eff #0.01% of range

    suns = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in range(3)]

    dists = []
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,np.array([0,7000/math.sqrt(2),7000/math.sqrt(2)]),np.array([8,0,0]),timing = True)
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    #
    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]


    suns_est = [SunSensorPair(unitvecs[j],noise_sun,sun_eff,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in range(3)]

    dists_est = []
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(7)
    estimate[3:7] = 0.5
    estimate[0:3] = 0.00001
    cov_estimate = block_diag(np.eye(3)*(0.5*math.pi/180.0)**2.0,np.eye(3)*3)
    int_cov =  block_diag(np.eye(3)*(0.001*math.pi/180.0)**2.0,0.0001*np.eye(3))

    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = False)
    est.al = 1e-3
    est.kap = 0
    est.bet = 2.0#-1.0#2.0

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
        autovar = np.diagonal(est.full_state.cov)
        print(autovar[0:3])
        print(autovar[3:6])

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e10*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 3
            ka = 0.1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,1,0,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 3
            ka = 0.1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw = 3
            ka = 0.1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = np.zeros(3)#est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = state
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)


def test_ukf_quat_as_vec():
    t0 = 0
    tf = 60*10
    tlim00 = 2*60
    tlim0 = 10*60
    tlim1 = 20*60
    tlim2 = 40*60
    dt = 1
    np.set_printoptions(precision=3)

    #
    #real_sat
    real_bias_mtq0 = np.random.uniform(0.01,0.2)*random_n_unit_vec(3)
    mtq_bsr = 0.005
    mtq_std = 0.0001*0.001
    mtq_max = 0.5
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = True, bias = np.dot(real_bias_mtq0,j).item(),use_noise=True,bias_std_rate=mtq_bsr) for j in unitvecs]
    acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

    real_mtm_bias0 = np.random.uniform(1e-9,1e-4)*random_n_unit_vec(3)
    mtm_bsr = 1e-12
    mtm_std = 1e-10


    real_gyro_bias0 = np.random.uniform(0.01,0.2)*random_n_unit_vec(3)*math.pi/180.0
    gyro_bsr = 0.01*math.pi/180.0
    gyro_std = 0.01*math.pi/180.0

    mtms = [MTM(j,mtm_std,has_bias = True,bias = np.dot(real_mtm_bias0,j).item(),use_noise = True,bias_std_rate = mtm_bsr) for j in unitvecs]
    mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
    gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]
    # gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]


    sun_eff = 0.3
    noise_sun = 0.005*sun_eff
    real_sun_bias0 = [np.random.uniform(-0.1,0.1)*sun_eff for j in range(4)]
    sun_bsr = 0.001*sun_eff

    suns1 = [SunSensor(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j],use_noise = True,bias_std_rate = sun_bsr) for j in range(2)]
    suns2 = [SunSensor(-unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j+2],use_noise = True,bias_std_rate = sun_bsr) for j in range(2)]


    mp_dipole = random_n_unit_vec(3)*np.random.uniform(0,0.1)
    dipole_std = 0.001
    dipole = Dipole_Disturbance([mp_dipole,0.5],estimate = False,time_varying = True,std = dipole_std)
    dipole = Dipole_Disturbance([np.zeros(3),0.5],estimate = False,time_varying = False,std = dipole_std)
    drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag = Drag_Disturbance(drag_faces)
    gg = GG_Disturbance()
    dists = [dipole,drag,gg]
    dists = [dipole,drag,gg]
    J = np.diagflat(np.array([3.4,2.9,1.3]))
    real_sat = Satellite(mass = 4,J = J,actuators = acts, sensors= mtms+gyros+suns1+suns2, disturbances = dists)
    w0 = random_n_unit_vec(3)*np.random.uniform(0,1.0)*math.pi/180
    q0 = random_n_unit_vec(4)
    os0 = Orbital_State(0.22-1*sec2cent,np.array([0,7000/math.sqrt(2),7000/math.sqrt(2)]),np.array([8,0,0]),timing = True)
    orb = Orbit(os0,end_time = 0.22+(tf-t0)*sec2cent,dt = dt,use_J2 = True)
    #
    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = True, bias = 0,use_noise=False,bias_std_rate=mtq_bsr,estimate_bias = True) for j in unitvecs]
    acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

    mtms_est = [MTM(j,mtm_std,has_bias = True,bias =0,use_noise = False,bias_std_rate = mtm_bsr,estimate_bias = True) for j in unitvecs]
    mtms_est = [MTM(j,mtm_std,has_bias = False,bias =0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
    gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]
    # gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]


    suns1_est = [SunSensor(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(2)]
    suns2_est = [SunSensor(-unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(2)]


    dipole_est = Dipole_Disturbance([np.zeros(3)],estimate=False,time_varying = False,std = dipole_std )
    drag_faces = [  [0,0.1*0.33,unitvecs[0]*0.05,unitvecs[0],2.2], \
                    [1,0.1*0.26,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                    [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                    [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                    [4,0.1*0.14,unitvecs[2]*0.15,unitvecs[2],2.2], \
                    [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
    drag_est = Drag_Disturbance(drag_faces)
    gg_est = GG_Disturbance()
    dists_est = [dipole_est,drag_est,gg_est]
    dists_est = [dipole_est,drag_est,gg_est]
    J_EST = np.diagflat(np.array([3.4,2.9,1.3]))
    est_sat = Satellite(mass = 4,J = J_EST,actuators = acts_est, sensors= mtms_est+gyros_est+suns1_est+suns2_est, disturbances = dists_est,estimated = True)

    estimate = np.zeros(7+3+3+3+3)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(1.0*math.pi/180.0)*2.0,np.eye(4),np.eye(3)*0.2**2.0,np.eye(3)*1e-8,np.eye(3)*0.2*0.2,np.eye(3)*0.1*0.1)
    int_cov =  block_diag(np.eye(3)*(0.002*math.pi/180.0)**2.0,(0.01*math.pi/180)**2.0*np.eye(4),block_diag(np.eye(3)*mtq_bsr**2.0,np.eye(3)*mtm_bsr**2.0,np.eye(3)*gyro_bsr**2.0,np.eye(3)*dipole_std**2.0))

    estimate = np.zeros(7+3+4)
    estimate[3] = 1
    cov_estimate = block_diag(np.eye(3)*(2.0*math.pi/180.0)**2.0,np.eye(4)*25,np.eye(3)*4,np.eye(4)*1)
    int_cov =  block_diag(np.eye(3)*(0.001*math.pi/180.0)**2.0,(0.01*math.pi/180)**2.0*np.eye(4),np.eye(3)*gyro_bsr**2.0,np.eye(4)*sun_bsr**2.0)


    est = UKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = True)

    t = t0
    t_hist = np.nan*np.zeros(int((tf-t0)/dt))
    ind = 0
    state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    est_state_hist =  np.nan*np.zeros((int((tf-t0)/dt),est.state_len))
    orb_hist = []
    control_hist = np.nan*np.zeros((int((tf-t0)/dt),est_sat.control_len))
    cov_hist = []
    control = np.zeros(3)
    state = np.concatenate([w0,q0])
    orbt = orb.get_os(0.22+(t-t0)*sec2cent)
    while t<tf:
        #
        #update estimator
        real_vecs = os_local_vecs(orbt,state[3:7])
        sens = real_sat.sensor_values(state,real_vecs)
        est.update(control,sens,orbt)
        est_state = est.use_state.val
        print(t,ind)
        print('real state ',state)
        print('est  state ',est_state[0:7])

        # print(np.concatenate([j.bias for j in real_sat.actuators]))
        # print(est_state[7:10])
        print('real sbias ',np.concatenate([j.bias for j in real_sat.sensors if j.has_bias]))
        print(' est sbias ',est_state[7:14])
        print('real dipole ',real_sat.disturbances[0].main_param)
        # print(est_state[7:10])
        print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
        print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))

        #find control
        if t<tlim00:
            ud = np.zeros(3)
        elif t<tlim0:
            #bdot
            Bbody = rot_mat(est.use_state.val[3:7]).T@orbt.B
            ud = -1e10*(-np.cross(est.use_state.val[0:3],Bbody))
        elif t<tlim1:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw =10
            ka = 0.1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        elif t<tlim2:
            #PID to [0,1,0,0]
            wdes = np.zeros(3)
            qdes = np.array([0,1,0,0])
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw =10
            ka = 0.1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2
        else:
            #PID to zeroquat
            wdes = np.zeros(3)
            qdes = zeroquat
            q = est.use_state.val[3:7]
            w =  est.use_state.val[0:3]
            w_err =w-wdes
            q_err = quat_mult(quat_inv(qdes),q)
            kw =10
            ka = 0.1
            nB2 = norm(orbt.B)
            Bbody = rot_mat(q).T@orbt.B
            ud = -kw*np.cross(Bbody, est_sat.J@w_err)/nB2-ka*np.cross(Bbody,q_err[1:]*np.sign(q_err[0]))/nB2

        offset_vec = est_sat.disturbances[0].main_param# + np.concatenate([j.bias for j in est_sat.actuators])
        control = limit(ud-offset_vec,mtq_max)
        print(control)

        #save info
        # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
        state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.sensors if j.has_bias])
        est_state_hist[ind,:] = est.use_state.val
        orb_hist += [orbt]
        control_hist[ind,:] = control
        cov_hist += [est.use_state.cov]

        #propagate
        ind += 1
        t += dt
        prev_os = orbt.copy()
        orbt = orb.get_os(0.22+(t-t0)*sec2cent)
        out = solve_ivp(real_sat.dynamics_for_solver, (0, dt), state, method="RK45", args=(control, prev_os,orbt), rtol=1e-7, atol=1e-7)#,jac = ivp_jac)
        # print('step done')
        state = out.y[:,-1]
        state[3:7] = normalize(state[3:7])
        real_sat.update_actuator_noise()
        real_sat.update_actuator_biases(orbt.J2000)
        real_sat.update_sensor_biases(orbt.J2000)
        real_sat.update_disturbances(orbt.J2000)
