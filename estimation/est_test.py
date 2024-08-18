from sat_ADCS_estimation import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
import numpy as np
import math
from scipy.integrate import odeint, solve_ivp, RK45

t0 = 0
tf = 60*15
tlim00 = 2*60
tlim0 = 10*60
tlim1 = 20*60
tlim2 = 30*60
dt = 1
np.set_printoptions(precision=3)

#
#real_sat
mtq_bias0_std = np.random.uniform(0.01,0.1)
real_bias_mtq0 = mtq_bias0_std*random_n_unit_vec(3)
mtq_bsr = 0.0001
mtq_max = 0.5
mtq_std = 0.001
acts =  [MTQ(j,mtq_std,mtq_max,has_bias = True, bias = np.dot(real_bias_mtq0,j).item(),use_noise=True,bias_std_rate=mtq_bsr) for j in unitvecs]
# acts =  [MTQ(j,mtq_std,mtq_max,has_bias = False,bias = 0,use_noise=True,bias_std_rate=0) for j in unitvecs]

mtm_bias0_std = np.random.uniform(1e-9,1e-6)
real_mtm_bias0 = mtm_bias0_std*random_n_unit_vec(3)
mtm_bsr = 1e-9 #1nT/sec
mtm_std = 300*1e-9


gyro_bias0_std = np.random.uniform(0.01,0.2)*math.pi/180.0
real_gyro_bias0 = gyro_bias0_std*random_n_unit_vec(3)
gyro_bsr = 0.0004*math.pi/180.0#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
gyro_std = 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s

mtms = [MTM(j,mtm_std,has_bias = True,bias = np.dot(real_mtm_bias0,j).item(),use_noise = True,bias_std_rate = mtm_bsr) for j in unitvecs]
# mtms = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]
gyros = [Gyro(j,gyro_std,has_bias = True,bias = np.dot(real_gyro_bias0,j).item(),use_noise = True,bias_std_rate = gyro_bsr) for j in unitvecs]
# gyros = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = True,bias_std_rate = 0) for j in unitvecs]


sun_eff = 0.3
noise_sun = 0.001*sun_eff #0.1% of range
sun_bias0_std = 0.1
real_sun_bias0 = [np.random.uniform(-sun_bias0_std,sun_bias0_std)*sun_eff for j in range(6)]
sun_bsr = 0.00001*sun_eff #0.001% of range /sec

suns1 = [SunSensor(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j],use_noise = True,bias_std_rate = sun_bsr) for j in range(3)]
suns2 = [SunSensor(-unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = real_sun_bias0[j+2],use_noise = True,bias_std_rate = sun_bsr) for j in range(3)]


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
# acts_est =  [MTQ(j,mtq_std,mtq_max,has_bias = False, bias = 0,use_noise=False,bias_std_rate=0,estimate_bias = False) for j in unitvecs]

mtms_est = [MTM(j,mtm_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = mtm_bsr,estimate_bias = True) for j in unitvecs]
# mtms_est = [MTM(j,mtm_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]
gyros_est = [Gyro(j,gyro_std,has_bias = True,bias = 0,use_noise = False,bias_std_rate = gyro_bsr,estimate_bias = True) for j in unitvecs]
# gyros_est = [Gyro(j,gyro_std,has_bias = False,bias = 0,use_noise = False,bias_std_rate = 0,estimate_bias = False) for j in unitvecs]


suns1_est = [SunSensor(unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(3)]
suns2_est = [SunSensor(-unitvecs[j],noise_sun,sun_eff,has_bias = True,bias = 0,use_noise = False,bias_std_rate = sun_bsr,estimate_bias = True) for j in range(3)]


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

estimate = np.zeros(7+3+3+3+6)
estimate[3:7] = 0.5
estimate[0:3] = 0.0001
estimate[7:10] = 0.0001
estimate[10:13] = 0
estimate[13:16] = 0.0001
estimate[16:22] = 0.0001
# estimate[10:14] = 0.0001
# estimate[3] = 1
cov_estimate = block_diag(np.eye(3)*(1.0*math.pi/180.0)**2.0,np.eye(4)*10,np.eye(3)*mtq_bias0_std**2.0,4*np.eye(3)*mtm_bias0_std**2.0,4*np.eye(3)*gyro_bias0_std**2.0,4*np.eye(6)*sun_bias0_std**2.0)
int_cov =  block_diag(np.eye(3)*(0.0001*math.pi/180.0)**2.0,(0.005*math.pi/180)**2.0*np.eye(4),2*np.eye(3)*mtq_bsr**2.0,2*np.eye(3)*mtm_bsr**2.0,2*np.eye(3)*gyro_bsr**2.0,2*np.eye(6)*sun_bsr**2.0)#,np.eye(3)*dipole_std**2.0))

# estimate = np.zeros(7+3+4)
# estimate[3:7] = 0.5
# estimate[0:3] = 0.0001
# estimate[7:10] = 0.0001
# estimate[10:14] = 0.0001
# cov_estimate = block_diag(np.eye(3)*(1.0*math.pi/180.0)**2.0,np.eye(4)*10,10*np.eye(3)*gyro_bias0_std**2.0,10*np.eye(4)*sun_bias0_std**2.0)
# int_cov =  block_diag(np.eye(3)*(0.0005*math.pi/180.0)**2.0,(0.01*math.pi/180)**2.0*np.eye(4),4*np.eye(3)*gyro_bsr**2.0,4*np.eye(4)*sun_bsr**2.0)


est = EKF(est_sat,estimate,cov_estimate,int_cov,sample_time = 1,quat_as_vec = True)

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
    real_sbias = np.concatenate([j.bias for j in real_sat.sensors if j.has_bias])

    print('real abias ',np.concatenate([j.bias for j in real_sat.actuators]))
    print(' est abias ',est_state[7:10])
    print('real sbias mtm ',real_sbias[0:3])
    print(' est sbias mtm ',est_state[10:13])
    print('real sbias gyro ',real_sbias[3:6])
    print(' est sbias gyro ',est_state[13:16])
    print('real sbias sun ',real_sbias[6:12])
    print(' est sbias sun ',est_state[16:22])
    print('real dipole ',real_sat.disturbances[0].main_param)
    # print(est_state[7:10])
    print((180/np.pi)*math.acos(-1 + 2*np.clip(np.dot(est_state[3:7],state[3:7]),-1,1)**2.0 ))
    print((norm(est_state[0:3]-state[0:3])*180.0/math.pi))
    autovar = np.diagonal(est.full_state.cov)
    print(autovar[0:3])
    print(autovar[3:7])
    print(autovar[7:10]*1000)
    print(autovar[10:13])
    print(autovar[13:16])
    print(autovar[16:22])

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
        ka = 1
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

    offset_vec = est_sat.disturbances[0].main_param + np.concatenate([j.bias for j in est_sat.actuators])
    control = limit(ud-offset_vec,mtq_max)
    print(control)

    #save info
    # state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors]+[real_sat.disturbances[0].main_param])
    state_hist[ind,:] = np.concatenate([state]+[j.bias for j in real_sat.actuators]+[j.bias for j in real_sat.sensors if j.has_bias])
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
