from sat_ADCS_estimation import *
from sat_ADCS_helpers import *
from sat_ADCS_orbit import *
from sat_ADCS_satellite import *
import numpy as np
import math
from scipy.integrate import odeint, solve_ivp, RK45
import matplotlib.pyplot as plt
import time
import pickle


def create_BC_sat(  real = True, rand=False,extra_big_est_randomness = False,
                    mass = None, J = None, COM = None, use_J_diag = False,
                    include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = True, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = True,
                    use_drag = True, drag_dist = None,
                    use_dipole = True, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True: #just to make code folding work better in atom.

        if mass is None:
            mass = 4
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
            if use_J_diag:
                wa,va = np.linalg.eigh(J)
                # idx =
                J = np.diagflat(wa[[1,2,0]])
        else:
            if use_J_diag:
                warnings.warn('the options "use_J_diag" is not available when a J matrix is provided')
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.05*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.15)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 1.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
            if extra_big_est_randomness and not real:
                mtq_std = 2*0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
            if extra_big_est_randomness and not real:
                mtq_bsr = 3*0.000001*np.ones(3)

        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(math.pi/180.0)
                else:
                    gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 0.0004*math.pi/180.0*np.ones(3)#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
            if extra_big_est_randomness and not real:
                gyro_bsr = 3*0.0004*math.pi/180.0*np.ones(3)
        if gyro_std is None:
            gyro_std = 0.03*math.pi/180.0*np.ones(3)#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
            if extra_big_est_randomness and not real:
                gyro_std = 2*0.03*math.pi/180.0*np.ones(3)
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1e4
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
                else:
                    mtm_bias0 = 1e-8*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = 1e-9*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 3*1e-7*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.2)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.25)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 0.5
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],0.5,0.2,0.3], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],0.5,0.2,0.3], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],0.5,0.2,0.3], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],0.5,0.3,0.2],  #this one is a little different from the first 3 on purpose for a camera aperature, etc.
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],0.3,0.5,0.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],0.25,0.6,0.15]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-8
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-8,2e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-7
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-6)
                    else:
                        gen_torq0 = 1e-6*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat


def create_GPS_BC_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None, use_J_diag = False,
                    include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = True, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    include_gpsbias = False, include_gps_noise = True, gps_std = None, gps_scale = None, gps_bsr = None, gps_bias0 = None, estimate_gps_bias = None,
                    use_gg = True,
                    use_drag = True, drag_dist = None,
                    use_dipole = True, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True: #just to make code folding work better in atom.

        if mass is None:
            mass = 4
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
            if use_J_diag:
                wa,va = np.linalg.eigh(J)
                # idx =
                J = np.diagflat(wa[[1,2,0]])
        else:
            if use_J_diag:
                warnings.warn('the options "use_J_diag" is not available when a J matrix is provided')
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.05*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.15)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 1.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)

        if gps_scale is None:
            gps_scale = 1
        if gps_bias0 is None:
            if real:
                if rand:
                    gps_bias0 = np.stack([random_n_unit_vec(3)*np.random.uniform(1,10),random_n_unit_vec(3)*np.random.uniform(0.01,0.5)])*gps_scale
                else:
                    gps_bias0 = np.array([1,2,3,0.1,0.01,0.2])*gps_scale
            else:
                gps_bias0 = np.zeros(3)
        if gps_bsr is None:
            gps_bsr = np.array([0.01,0.01,0.01,0.0001,0.0001,0.0001])
        if gps_std is None:
            gps_std = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        if estimate_gps_bias is None and not real:
            estimate_gps_bias = include_gpsbias and np.all(gps_bsr>1e-20)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(math.pi/180.0)
                else:
                    gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 0.0004*math.pi/180.0*np.ones(3)#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
        if gyro_std is None:
            gyro_std = 0.03*math.pi/180.0*np.ones(3)#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1e4
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
                else:
                    mtm_bias0 = 1e-8*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = 1e-9*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 3*1e-7*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.2)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.25)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 0.5
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],0.5,0.2,0.3], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],0.5,0.2,0.3], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],0.5,0.2,0.3], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],0.5,0.3,0.2],  #this one is a little different from the first 3 on purpose for a camera aperature, etc.
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],0.3,0.5,0.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],0.25,0.6,0.15]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-8
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-8,2e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-7
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-6)
                    else:
                        gen_torq0 = 1e-6*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            gps = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns+gps, disturbances = dists)
            return real_sat
        else:
            acts_est = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            gps_est = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est+gps_est, disturbances = dists,estimated = True)
            return est_sat



def create_GPS_6U_sat(  real = True, rand=False, extra_big_est_randomness = False,
                    mass = None, J = None, COM = None, use_J_diag = False,
                    use_mtq = False, include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    use_thrusters = False, include_thrbias = True, include_thr_noise = True, thr_bias0 = None,thr_std = None, thr_bsr = None, thr_max = None,estimate_thr_bias = None,
                    use_RW = True, include_rwbias = False, include_rw_noise = True, rw_bias0 = None,rw_std = None, rw_bsr = None, rw_max = None,rw_J = None, rw_mom = None,rw_maxh = None,rw_mom_sens_noise_std = None,estimate_rw_bias = None,
                    include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = True, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    include_gpsbias = False, include_gps_noise = True, gps_std = None, gps_scale = None, gps_bsr = None, gps_bias0 = None, estimate_gps_bias = None,
                    use_gg = True,
                    use_drag = True, drag_dist = None,
                    use_dipole = True, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True, estimate_dipole = None,
                    use_SRP = True, SRP_dist = None,
                    use_prop = True, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True, estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True, estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True: #just to make code folding work better in atom.

        if mass is None:
            mass = 10.165 #(based on Asteria from https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=4173&context=smallsat)
        if COM is None:
            COM = np.zeros(3)#np.array([-1.93,-5.1,-1.36])/1000 #(based on Asteria from https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=4173&context=smallsat)
        if J is None:
            J =  np.diagflat([0.0969,0.1235,0.1918]) #(based on Asteria from https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=4173&context=smallsat)
            if use_J_diag:
                wa,va = np.linalg.eigh(J)
                # idx =
                J = np.diagflat(wa[[1,2,0]])
        else:
            if use_J_diag:
                warnings.warn('the options "use_J_diag" is not available when a J matrix is provided')
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.1*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.15)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 5.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.001*np.ones(3)*include_mtq_noise
        mtq_std = mtq_std*include_mtq_noise
        if mtq_bsr is None:
            mtq_bsr = 0.00001*np.ones(3)
            if extra_big_est_randomness and not real:
                mtq_bsr *= 3
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)

        if rw_bias0 is None: #https://storage.googleapis.com/blue-canyon-tech-news/1/2024/03/ACS-1_2024.pdf XACT-15
            if real:
                if not rand:
                    rw_bias0 = 0.0001*normalize(np.array([1,1,4]))
                else:
                    rw_bias0 = random_n_unit_vec(3)*np.random.uniform(0.00001,0.0002)#normalize(np.array([1,1,4]))
            else:
                rw_bias0 = np.zeros(3)
        if rw_max is None:
            rw_max = 0.005*np.ones(3)
        if rw_std is None:
            rw_std = 0.00001*np.ones(3)*include_rw_noise
            if extra_big_est_randomness and not real:
                rw_std *= 1.5
        rw_std = rw_std*include_rw_noise
        if rw_bsr is None:
            rw_bsr = 0.0000001*np.ones(3)
            if extra_big_est_randomness and not real:
                rw_bsr *= 3
        if rw_J is None: #XACT15 max momenutm storage combined with 6500 RPM max from this https://nanoavionics.com/cubesat-components/cubesat-reaction-wheels-control-system-satbus-4rw/
            rw_J = 0.0014*np.ones(3)
        if rw_mom is None:
            if real:
                if not rand:
                    rw_mom = (0.5/1000)*normalize(np.array([1,-2,2]))
                else:
                    rw_mom = random_n_unit_vec(3)*np.random.uniform(0.1/1000,1/1000)#normalize(np.array([1,1,4]))
            else:
                rw_mom = np.zeros(3)
        if rw_maxh is None: #XACT
            rw_maxh = (15/1000)*np.ones(3)
        if rw_mom_sens_noise_std is None: #made up
            rw_mom_sens_noise_std =  1e-12*np.ones(3)
            if extra_big_est_randomness and not real:
                rw_mom_sens_noise_std *= 1.5
        if estimate_rw_bias is None and not real:
            estimate_rw_bias = include_rwbias and np.all(rw_bsr>1e-15)

        if thr_bias0 is None: #look at https://cubesat-propulsion.com/jpl-marco-micro-propulsion-system/. also this https://issfd.org/ISSFD_2019/ISSFD_2019_AIAC18_Young-Brian.pdf. assuming 20 mN thrust perpendicular at at a distance of 10 cm
            if real:
                if not rand:
                    thr_bias0 = 0.00001*normalize(np.array([1,1,4]))
                else:
                    thr_bias0 = random_n_unit_vec(3)*np.random.uniform(0.00001,0.0005)#normalize(np.array([1,1,4]))
            else:
                thr_bias0 = np.zeros(3)
        if thr_max is None:
            thr_max = 0.002*np.ones(3) #assuming 20 mN thrust perpendicular at at a distance of 10 cm
        if thr_std is None:
            thr_std = 0.00001*np.ones(3)*include_thr_noise
            if extra_big_est_randomness and not real:
                thr_std *= 1.5
        thr_std = thr_std*include_thr_noise
        if thr_bsr is None:
            thr_bsr = 0.000001*np.ones(3)
            if extra_big_est_randomness and not real:
                thr_bsr *= 3
        if estimate_thr_bias is None and not real:
            estimate_thr_bias = include_thrbias and np.all(thr_bsr>1e-15)

        if gps_scale is None:
            gps_scale = 1
        if gps_bias0 is None:
            if real:
                if rand:
                    gps_bias0 = np.stack([random_n_unit_vec(3)*np.random.uniform(1,10),random_n_unit_vec(3)*np.random.uniform(0.01,0.5)])*gps_scale
                else:
                    gps_bias0 = np.array([1,2,3,0.1,0.01,0.2])*gps_scale
            else:
                gps_bias0 = np.zeros(3)
        if gps_bsr is None:
            gps_bsr = np.array([0.01,0.01,0.01,0.0001,0.0001,0.0001])
        if gps_std is None:
            gps_std = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        if estimate_gps_bias is None and not real:
            estimate_gps_bias = include_gpsbias and np.all(gps_bsr>1e-20)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(math.pi/180.0)
                else:
                    gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 0.0004*math.pi/180.0*np.ones(3)#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill -- see Table 11
            if extra_big_est_randomness and not real:
                gyro_bsr *= 1.5
        if gyro_std is None:
            gyro_std = 0.03*math.pi/180.0*np.ones(3)*include_gyro_noise#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
            if extra_big_est_randomness and not real:
                gyro_std *= 1.5
        gyro_std = gyro_std*include_gyro_noise
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1e4
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-9,3e-6) #---see Tuthill
                else:
                    mtm_bias0 = 1e-6*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = 1e-9*np.ones(3)#*mtm_scale #1nT/sec
            if extra_big_est_randomness and not real:
                mtm_bsr *= 1.5
        if mtm_std is None:
            mtm_std = 3*1e-7*np.ones(3)*include_mtm_noise #---see Tuthill
            if extra_big_est_randomness and not real:
                mtm_std *= 1.5
        mtm_std = mtm_std*include_mtm_noise
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff*include_sun_noise #0.1% of range
            if extra_big_est_randomness and not real:
                sun_std *= 1.5
        thr_std = sun_std*include_sun_noise
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.2)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
            if extra_big_est_randomness and not real:
                sun_bsr *= 3
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.2,unitvecs[0]*0.15,unitvecs[0],2.2], \
                                [1,0.1*0.2,-unitvecs[0]*0.15,-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.1,unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.1,-unitvecs[1],2.2], \
                                [4,0.3*0.2,unitvecs[2]*0.05,unitvecs[2],2.2], \
                                [5,0.3*0.2*3,-unitvecs[2]*0.05,-unitvecs[2],2.2],\
                                [6,0.3*0.2,-unitvecs[2]*0.05 + 0.2*unitvecs[1],unitvecs[2],2.2],\
                                [7,0.3*0.2,-unitvecs[2]*0.05 - 0.2*unitvecs[1],unitvecs[2],2.2],\
                                ]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
                if extra_big_est_randomness and not real:
                    dipole_std *= 2
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.25)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 0.5
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                #https://onlinelibrary.wiley.com/doi/10.1155/2015/928206
                al_a = 0.12 #aluminum
                al_d = 0.08
                al_s = 0.8

                spf_a = 0.92 #solar panel front
                spf_d = 0.007
                spf_s = 0.073

                wp_a = 0.24 #white paintz
                wp_d = 0.38 #white paint
                wp_s = 0.38 #white paint

                bp_a = 0.97#black paintz
                bp_d = 0.015#black paint
                bp_s = 0.015#black paint
                #roughly modeled on an imagined variant of asteria with a camera at +y and a thruster at -z
                SRP_faces = [  [0,0.1*0.2,unitvecs[0]*0.15,unitvecs[0],wp_a,wp_d,wp_s], \
                                [1,0.1*0.2,-unitvecs[0]*0.15,-unitvecs[0],0.5*bp_a+0.5*al_a,0.5*bp_d+0.5*al_d,0.5*bp_s+0.5*al_s], \
                                [2,0.1*0.3,unitvecs[1]*0.1,unitvecs[1],0.5*bp_a+0.5*al_a,0.5*bp_d+0.5*al_d,0.5*bp_s+0.5*al_s], \
                                [3,0.1*0.3,-unitvecs[1]*0.1,-unitvecs[1],wp_a,wp_d,wp_s],\
                                [4,0.3*0.2,unitvecs[2]*0.05,unitvecs[2],wp_a,wp_d,wp_s], \
                                [5,0.3*0.2*3,-unitvecs[2]*0.05,-unitvecs[2],spf_a*0.9+0.1*al_a,spf_d*0.9+0.1*al_d,spf_s*0.9+0.1*al_s],\
                                [6,0.3*0.2,-unitvecs[2]*0.05 + 0.2*unitvecs[1],-unitvecs[2],al_a,al_d,al_s],\
                                [7,0.3*0.2,-unitvecs[2]*0.05 - 0.2*unitvecs[1],-unitvecs[2],al_a,al_d,al_s],\
                                ]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-7
                if extra_big_est_randomness and not real:
                    prop_torq_std *= 2
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-8,2e-6)
                    else:
                        prop_torq0 = 5e-5*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-4
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-7
                if extra_big_est_randomness and not real:
                    gen_torq_std *= 3
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-6)
                    else:
                        gen_torq0 = 1e-6*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            mtqs = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            rws = [RW(unitvecs[j],rw_std[j],rw_max[j],rw_J[j],rw_mom[j],rw_maxh[j],rw_mom_sens_noise_std[j],has_bias = include_rwbias,bias = rw_bias0[j],use_noise=include_rw_noise,bias_std_rate=rw_bsr[j]) for j in range(3)]
            thrs = [Magic(unitvecs[j],thr_std[j],thr_max[j],has_bias = include_thrbias,bias = thr_bias0[j],use_noise=include_thr_noise,bias_std_rate=thr_bsr[j]) for j in range(3)]
            acts = mtqs*use_mtq + rws*use_RW + thrs*use_thrusters
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            gps = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns+gps, disturbances = dists)
            return real_sat
        else:
            mtqs_est = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            rws_est = [RW(unitvecs[j],rw_std[j],rw_max[j],rw_J[j],rw_mom[j],rw_maxh[j],rw_mom_sens_noise_std[j],has_bias = include_rwbias,bias = rw_bias0[j],use_noise=include_rw_noise,bias_std_rate=rw_bsr[j],estimate_bias = estimate_rw_bias) for j in range(3)]
            thrs_est = [Magic(unitvecs[j],thr_std[j],thr_max[j],has_bias = include_thrbias,bias = thr_bias0[j],use_noise = include_thr_noise,bias_std_rate=thr_bsr[j],estimate_bias = estimate_thr_bias) for j in range(3)]
            acts_est = mtqs_est*use_mtq + rws_est*use_RW + thrs_est*use_thrusters
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            gps_est = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est+gps_est, disturbances = dists,estimated = True)
            return est_sat





def create_GPS_BC_sat_plus_1magic(  real = True, rand=False,
                    mass = None, J = None, COM = None, use_J_diag = False,
                    include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magicbias = True, include_magic_noise = True, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,magic_axis = unitvecs[0],
                    include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = True, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    include_gpsbias = False, include_gps_noise = True, gps_std = None, gps_scale = None, gps_bsr = None, gps_bias0 = None, estimate_gps_bias = None,
                    use_gg = True,
                    use_drag = True, drag_dist = None,
                    use_dipole = True, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True: #just to make code folding work better in atom.

        if mass is None:
            mass = 4
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
            if use_J_diag:
                wa,va = np.linalg.eigh(J)
                # idx =
                J = np.diagflat(wa[[1,2,0]])
        else:
            if use_J_diag:
                warnings.warn('the options "use_J_diag" is not available when a J matrix is provided')
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.05*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.15)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 1.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)

        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = 0.02
                else:
                    magic_bias0 = np.random.uniform(0.0001,0.05)#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = 0
        if magic_max is None:
            magic_max = 0.1
        if magic_std is None:
            magic_std = 0.00001
        if magic_bsr is None:
            magic_bsr = 0.0000001
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)

        if gps_scale is None:
            gps_scale = 1
        if gps_bias0 is None:
            if real:
                if rand:
                    gps_bias0 = np.stack([random_n_unit_vec(3)*np.random.uniform(1,10),random_n_unit_vec(3)*np.random.uniform(0.01,0.5)])*gps_scale
                else:
                    gps_bias0 = np.array([1,2,3,0.1,0.01,0.2])*gps_scale
            else:
                gps_bias0 = np.zeros(3)
        if gps_bsr is None:
            gps_bsr = np.array([0.01,0.01,0.01,0.0001,0.0001,0.0001])
        if gps_std is None:
            gps_std = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        if estimate_gps_bias is None and not real:
            estimate_gps_bias = include_gpsbias and np.all(gps_bsr>1e-20)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(math.pi/180.0)
                else:
                    gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 0.0004*math.pi/180.0*np.ones(3)#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
        if gyro_std is None:
            gyro_std = 0.03*math.pi/180.0*np.ones(3)#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1e4
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
                else:
                    mtm_bias0 = 1e-8*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = 1e-9*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 3*1e-7*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.2)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.25)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 0.5
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],0.5,0.2,0.3], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],0.5,0.2,0.3], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],0.5,0.2,0.3], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],0.5,0.3,0.2],  #this one is a little different from the first 3 on purpose for a camera aperature, etc.
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],0.3,0.5,0.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],0.25,0.6,0.15]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-8
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-8,2e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-7
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-6)
                    else:
                        gen_torq0 = 1e-6*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts =     [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]+[Magic(normalize(magic_axis),magic_std,magic_max,has_bias = include_magicbias,bias = magic_bias0,use_noise = include_magic_noise,bias_std_rate=magic_bsr)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            gps = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns+gps, disturbances = dists)
            return real_sat
        else:
            acts_est = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]+[Magic(normalize(magic_axis),magic_std,magic_max,has_bias = include_magicbias,bias = magic_bias0,use_noise = include_magic_noise,bias_std_rate=magic_bsr,estimate_bias = estimate_magic_bias)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            gps_est = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est+gps_est, disturbances = dists,estimated = True)
            return est_sat

def create_BC_sat_more_drag(  real = True, rand=False,
                    mass = None, J = None, COM = None, use_J_diag = False,
                    include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = True, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = True,
                    use_drag = True, drag_dist = None,drag_zshift = 0.01,
                    use_dipole = True, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True: #just to make code folding work better in atom.

        if mass is None:
            mass = 4
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
            if use_J_diag:
                wa,va = np.linalg.eigh(J)
                # idx =
                J = np.diagflat(wa[[1,2,0]])
        else:
            if use_J_diag:
                warnings.warn('the options "use_J_diag" is not available when a J matrix is provided')
        J = 0.5*(J+J.T)


        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.05*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.15)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 1.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(math.pi/180.0)
                else:
                    gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 0.0004*math.pi/180.0*np.ones(3)#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
        if gyro_std is None:
            gyro_std = 0.03*math.pi/180.0*np.ones(3)#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1e4
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
                else:
                    mtm_bias0 = 1e-8*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = 1e-9*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 3*1e-7*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.2)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05+drag_zshift*unitvecs[2],unitvecs[0],2.2], \
                                [1,0.1*0.3,-unitvecs[0]*0.05+drag_zshift*unitvecs[2],-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.05+drag_zshift*unitvecs[2],unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.05+drag_zshift*unitvecs[2],-unitvecs[1],2.2], \
                                [4,0.1*0.1,unitvecs[2]*(0.15+drag_zshift),unitvecs[2],2.2], \
                                [5,0.1*0.1,-unitvecs[2]*(0.15-drag_zshift),-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.25)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 0.5
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],0.5,0.2,0.3], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],0.5,0.2,0.3], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],0.5,0.2,0.3], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],0.5,0.3,0.2],  #this one is a little different from the first 3 on purpose for a camera aperature, etc.
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],0.3,0.5,0.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],0.25,0.6,0.15]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-8
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-8,2e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-7
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-6)
                    else:
                        gen_torq0 = 1e-6*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat

def create_fancy_BC_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None,
                    include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = True, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = True,
                    use_drag = True, drag_dist = None,
                    use_dipole = True, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True:

        if mass is None:
            mass = 4
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
        J = 0.5*(J+J.T)


        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.05*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.15)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 1.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 1e-5*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 1e-8*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.1)*(math.pi/180.0)
                else:
                    gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 1e-10*np.ones(3)#(3/3600/60)*math.pi/180.0 #20 deg/s/hour #Trawny formulas, data from Tuthill
        if gyro_std is None:
            gyro_std = 1e-7*np.ones(3)#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1e4
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
                else:
                    mtm_bias0 = 1e-8*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = 1e-9*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 1e-8*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.00001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.2)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.0000001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.25)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 0.5
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],0.5,0.2,0.3], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],0.5,0.2,0.3], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],0.5,0.2,0.3], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],0.5,0.3,0.2],  #this one is a little different from the first 3 on purpose for a camera aperature, etc.
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],0.3,0.5,0.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],0.25,0.6,0.15]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-8
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-8,2e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-7
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-6)
                    else:
                        gen_torq0 = 1e-6*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat


def create_BC_plus_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None,
                    include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = True, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = True,
                    use_drag = True, drag_dist = None,
                    use_dipole = True, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = True, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True:

        if mass is None:
            mass = 4
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
        J = 0.5*(J+J.T)


        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.05*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.15)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 1.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(math.pi/180.0)
                else:
                    gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 3.1623e-4 * 1e-6*np.ones(3) #from Crassidis
        if gyro_std is None:
            gyro_std = 0.31623*1e-6*np.ones(3)#0.0001# 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1e4
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
                else:
                    mtm_bias0 = 1e-8*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = 1e-9*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 3*1e-7*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.2)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0.01,0.25)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 0.5
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],0.5,0.2,0.3], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],0.5,0.2,0.3], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],0.5,0.2,0.3], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],0.5,0.3,0.2],  #this one is a little different from the first 3 on purpose for a camera aperature, etc.
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],0.3,0.5,0.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],0.25,0.6,0.15]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-8
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-8,2e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-7
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-6)
                    else:
                        gen_torq0 = 1e-6*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat


def create_Crassidis_UKF_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None,jmult = 1,
                    include_mtq = False, include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = False, include_magicbias = True, include_magic_noise = True, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = False, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = False,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = False
                    ):
    if True:
        if mass is None:
            mass = 3e3
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([1,3,3]))*500
            J = jmult*J
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 5*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 100.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.5)#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 1.0*np.ones(3)
        if magic_std is None:
            magic_std = 0.00001*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.0000001*np.ones(3)
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*(math.pi/180.0)/3600
                else:
                    gyro_bias0 = np.ones(3)*0.1*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 3.1623e-4 * 1e-6*np.ones(3)
        if gyro_std is None:
            gyro_std = 0.31623*1e-6*np.ones(3)#0.0001# 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-9)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 50*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        lmult = jmult**(1/3)
        amult = jmult**(2/3)
        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,1.3*amult,unitvecs[0]*2.2*lmult,unitvecs[0],1.9], \
                                [1,1.3*amult,-unitvecs[0]*0.8*lmult,-unitvecs[0],2.2], \
                                [2,3.25*amult,unitvecs[1]*5*lmult,unitvecs[1],2.2], \
                                [3,3.25*amult,-unitvecs[1]*5*lmult,-unitvecs[1],2.2], \
                                [4,2.0*amult,unitvecs[2]*0.5*lmult,unitvecs[2],2.2], \
                                [5,2.0*amult,-unitvecs[2]*1.5*lmult,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])*10*math.sqrt(jmult)
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 2.0*math.sqrt(jmult)
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,1.3*amult,unitvecs[0]*2.2*lmult,unitvecs[0],0.3,0.4,0.3], \
                                [1,1.3*amult,-unitvecs[0]*0.8*lmult,-unitvecs[0],0.3,0.4,0.3], \
                                [2,3.25*amult,unitvecs[1]*5*lmult,unitvecs[1],0.5,0.2,0.3], \
                                [3,3.25*amult,-unitvecs[1]*5*lmult,-unitvecs[1],0.5,0.2,0.3], \
                                [4,2.0*amult,unitvecs[2]*0.5*lmult,unitvecs[2],0.2,0.5,0.3], \
                                [5,2.0*amult,-unitvecs[2]*1.5*lmult,-unitvecs[2],0.3,0.4,0.3]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-6
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        prop_torq0 = 1e-5*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-4*math.sqrt(jmult)
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-6
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-4)
                    else:
                        gen_torq0 = 1e-4*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-3*math.sqrt(jmult)
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat



def create_Crassidis_UKF_cubesat(  real = True, rand=False,
                    mass = None, J = None, COM = None,
                    include_mtq = False, include_mtqbias = True, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = False, include_magicbias = True, include_magic_noise = True, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = False, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = False,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    ):
    if True:
        if mass is None:
            mass = 4
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([1,3,3]))*0.1
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.05*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 10.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.00001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.5)#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 0.1*np.ones(3)
        if magic_std is None:
            magic_std = 0.001*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.00001*np.ones(3)
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*(math.pi/180.0)/3600
                else:
                    gyro_bias0 = np.ones(3)*0.1*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = 3.1623e-4 * 1e-1*np.ones(3)
        if gyro_std is None:
            gyro_std = 3.1623e-2*1e-1*np.ones(3)#0.0001# 0.03*math.pi/180.0#(0.5*math.pi/180/math.sqrt(3600))#0.05*math.pi/180.0 #0.1 deg/s
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-8)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 300*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)


        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],2.2], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],2.2], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],2.2], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],2.2], \
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],2.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.0001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 2.0
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.1*0.3,unitvecs[0]*0.05,unitvecs[0],0.5,0.2,0.3], \
                                [1,0.1*0.3,-unitvecs[0]*0.05,-unitvecs[0],0.5,0.2,0.3], \
                                [2,0.1*0.3,unitvecs[1]*0.05,unitvecs[1],0.5,0.2,0.3], \
                                [3,0.1*0.3,-unitvecs[1]*0.05,-unitvecs[1],0.5,0.3,0.2],  #this one is a little different from the first 3 on purpose for a camera aperature, etc.
                                [4,0.1*0.1,unitvecs[2]*0.15,unitvecs[2],0.3,0.5,0.2], \
                                [5,0.1*0.1,-unitvecs[2]*0.15,-unitvecs[2],0.25,0.6,0.15]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-8
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        prop_torq0 = 1e-8*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-6
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-9
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-4)
                    else:
                        gen_torq0 = 1e-7*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat



def create_Wie_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None,jmult = 1,
                    include_mtq = False, include_mtqbias = False, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = True, include_magicbias = False, include_magic_noise = False, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = True, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = False,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = True
                    ):
    if True:
        if mass is None:
            mass = 7500
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([10,9,12]))*1000
            J = jmult*J
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 10*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 500.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0005*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000005*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))*2.0*include_magicbias
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.5,2.5)*include_magicbias#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 20.0*np.ones(3)
        if magic_std is None:
            magic_std = 0.01*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.001*np.ones(3)*include_magicbias
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*((math.pi/180.0)/3600)*include_gbias
                else:
                    gyro_bias0 = include_gbias*np.ones(3)*0.1*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = include_gbias*np.ones(3)*(0.1*math.pi/180.0)/(3600.0) #0.0000005 #5e-7
        if gyro_std is None:
            gyro_std = np.ones(3)*(0.0025*math.pi/180.0) #0.00005 # 5e-5
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)



        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-9)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 50*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        lmult = jmult**(1/3)
        amult = jmult**(2/3)
        dists = []
        com_to_center = unitvecs[0]*0.4*lmult + 0.8*lmult*unitvecs[2]
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,2.7*amult,unitvecs[0]*0.75*lmult-com_to_center,unitvecs[0],2.2], \
                                [1,2.7*amult,-unitvecs[0]*0.75*lmult-com_to_center,-unitvecs[0],2.2], \
                                [2,4.05*amult,unitvecs[1]*0.5*lmult-com_to_center,unitvecs[1],2.2], \
                                [3,4.05*amult,-unitvecs[1]*0.5*lmult-com_to_center,-unitvecs[1],2.2], \
                                [4,1.5*amult,unitvecs[2]*1.35*lmult-com_to_center,unitvecs[2],2.2], \
                                [5,1.5*amult,-unitvecs[2]*1.35*lmult-com_to_center,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.1
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)*5.0
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])*200*math.sqrt(jmult)
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 5.0*math.sqrt(jmult)
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,2.7*amult,unitvecs[0]*0.75*lmult-com_to_center,unitvecs[0],0.3,0.4,0.3], \
                                [1,2.7*amult,-unitvecs[0]*0.75*lmult-com_to_center,-unitvecs[0],0.3,0.4,0.3], \
                                [2,4.05*amult,unitvecs[1]*0.5*lmult-com_to_center,unitvecs[1],0.7,0.1,0.2], \
                                [3,4.05*amult,-unitvecs[1]*0.5*lmult-com_to_center,-unitvecs[1],0.3,0.4,0.3], \
                                [4,1.5*amult,unitvecs[2]*1.35*lmult-com_to_center,unitvecs[2],0.2,0.6,0.2], \
                                [5,1.5*amult,-unitvecs[2]*1.35*lmult-com_to_center,-unitvecs[2],0.3,0.4,0.3]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-2
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-3,1e-1)
                    else:
                        prop_torq0 = 1e-1*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-1*math.sqrt(jmult)
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-4
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-2)
                    else:
                        gen_torq0 = 1e-3*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-1*math.sqrt(jmult)
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat


def create_Wie_sat_w_GPS(  real = True, rand=False,
                    mass = None, J = None, COM = None,jmult = 1,
                    include_mtq = False, include_mtqbias = False, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = True, include_magicbias = False, include_magic_noise = False, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = True, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    include_gpsbias = False, include_gps_noise = True, gps_std = None, gps_scale = None, gps_bsr = None, gps_bias0 = None, estimate_gps_bias = None,
                    use_gg = False,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = True
                    ):
    if True:
        if mass is None:
            mass = 7500
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([10,9,12]))*1000
            J = jmult*J
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 10*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 500.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0005*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000005*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))*2.0*include_magicbias
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.5,2.5)*include_magicbias#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 20.0*np.ones(3)
        if magic_std is None:
            magic_std = 0.01*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.001*np.ones(3)*include_magicbias
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gps_scale is None:
            gps_scale = 1
        if gps_bias0 is None:
            if real:
                if rand:
                    gps_bias0 = np.stack([random_n_unit_vec(3)*np.random.uniform(1,10),random_n_unit_vec(3)*np.random.uniform(0.01,0.5)])*gps_scale
                else:
                    gps_bias0 = np.array([1,2,3,0.1,0.01,0.2])*gps_scale
            else:
                gps_bias0 = np.zeros(3)
        if gps_bsr is None:
            gps_bsr = np.array([0.01,0.01,0.01,0.0001,0.0001,0.0001]) #1nT/sec
        if gps_std is None:
            gps_std = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        if estimate_gps_bias is None and not real:
            estimate_gps_bias = include_gpsbias and np.all(gps_bsr>1e-20)

        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*((math.pi/180.0)/3600)*include_gbias
                else:
                    gyro_bias0 = include_gbias*np.ones(3)*0.1*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = include_gbias*np.ones(3)*(0.1*math.pi/180.0)/(3600.0) #0.0000005 #5e-7
        if gyro_std is None:
            gyro_std = np.ones(3)*(0.0025*math.pi/180.0) #0.00005 # 5e-5
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)



        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-9)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 50*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        lmult = jmult**(1/3)
        amult = jmult**(2/3)
        dists = []
        com_to_center = unitvecs[0]*0.4*lmult + 0.8*lmult*unitvecs[2]
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,2.7*amult,unitvecs[0]*0.75*lmult-com_to_center,unitvecs[0],2.2], \
                                [1,2.7*amult,-unitvecs[0]*0.75*lmult-com_to_center,-unitvecs[0],2.2], \
                                [2,4.05*amult,unitvecs[1]*0.5*lmult-com_to_center,unitvecs[1],2.2], \
                                [3,4.05*amult,-unitvecs[1]*0.5*lmult-com_to_center,-unitvecs[1],2.2], \
                                [4,1.5*amult,unitvecs[2]*1.35*lmult-com_to_center,unitvecs[2],2.2], \
                                [5,1.5*amult,-unitvecs[2]*1.35*lmult-com_to_center,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.1
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)*5.0
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])*200*math.sqrt(jmult)
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 5.0*math.sqrt(jmult)
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,2.7*amult,unitvecs[0]*0.75*lmult-com_to_center,unitvecs[0],0.3,0.4,0.3], \
                                [1,2.7*amult,-unitvecs[0]*0.75*lmult-com_to_center,-unitvecs[0],0.3,0.4,0.3], \
                                [2,4.05*amult,unitvecs[1]*0.5*lmult-com_to_center,unitvecs[1],0.7,0.1,0.2], \
                                [3,4.05*amult,-unitvecs[1]*0.5*lmult-com_to_center,-unitvecs[1],0.3,0.4,0.3], \
                                [4,1.5*amult,unitvecs[2]*1.35*lmult-com_to_center,unitvecs[2],0.2,0.6,0.2], \
                                [5,1.5*amult,-unitvecs[2]*1.35*lmult-com_to_center,-unitvecs[2],0.3,0.4,0.3]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-2
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-3,1e-1)
                    else:
                        prop_torq0 = 1e-1*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-1*math.sqrt(jmult)
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-4
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-2)
                    else:
                        gen_torq0 = 1e-3*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-1*math.sqrt(jmult)
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            gps = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns+gps, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            gps_est = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]
            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est+gps_est, disturbances = dists,estimated = True)
            return est_sat

def create_Wisniewski_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None,jmult = 1,
                    include_mtq = True, include_mtqbias = False, include_mtq_noise = False, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = False, include_magicbias = False, include_magic_noise = True, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = True, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = True,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = True
                    ):
    if True:
        if mass is None:
            mass = 61.8
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([181.78,181.25,1.28])) #boom deployed
            J = jmult*J
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.5*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 20.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.1*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.01*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))*0.5*include_magicbias
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.5,1.0)*include_magicbias#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 5.0*np.ones(3)
        if magic_std is None:
            magic_std = 0.00001*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.0000001*np.ones(3)*include_magicbias
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*((math.pi/180.0)/3600)*include_gbias
                else:
                    gyro_bias0 = include_gbias*np.ones(3)*1.0*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = include_gbias*np.ones(3)*(1.0*math.pi/180.0)/(3600.0)
        if gyro_std is None:
            gyro_std = np.ones(3)*(0.025*math.pi/180.0)
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-9)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 50*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        lmult = jmult**(1/3)
        amult = jmult**(2/3)
        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.65*amult,lmult*(unitvecs[0]*0.17 + unitvecs[2]*3),unitvecs[0],2.0], \
                                [1,0.65*amult,lmult*(-unitvecs[0]*0.17 + unitvecs[2]*3),-unitvecs[0],2.0], \
                                [2,0.6*amult,lmult*(unitvecs[1]*0.225 + unitvecs[2]*3),unitvecs[1],2.0], \
                                [3,0.6*amult,lmult*(-unitvecs[1]*0.225 + unitvecs[2]*3),-unitvecs[1],2.0], \
                                [4,0.12*amult,unitvecs[2]*8.34*lmult,unitvecs[2],1.7], \
                                [5,0.153*amult,-unitvecs[2]*0.34*lmult,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)*5.0
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])*20*math.sqrt(jmult)
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 5.0*math.sqrt(jmult)
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces =  [  [0,0.65*amult,lmult*(unitvecs[0]*0.17 + unitvecs[2]*3),unitvecs[0],0.3,0.3,0.4], \
                                [1,0.65*amult,lmult*(-unitvecs[0]*0.17 + unitvecs[2]*3),-unitvecs[0],0.3,0.3,0.4], \
                                [2,0.6*amult,lmult*(unitvecs[1]*0.225 + unitvecs[2]*3),unitvecs[1],0.3,0.3,0.4], \
                                [3,0.6*amult,lmult*(-unitvecs[1]*0.225 + unitvecs[2]*3),-unitvecs[1],0.3,0.3,0.4], \
                                [4,0.12*amult,unitvecs[2]*8.34*lmult,unitvecs[2],0.2,0.4,0.4], \
                                [5,0.153*amult,-unitvecs[2]*0.34*lmult,-unitvecs[2],0.2,0.6,0.3]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-6
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-6*math.sqrt(jmult)
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-6
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        gen_torq0 = 1e-7*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5*math.sqrt(jmult)
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat


def create_Wisniewski_stowed_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None,jmult = 1,
                    include_mtq = True, include_mtqbias = False, include_mtq_noise = False, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = False, include_magicbias = False, include_magic_noise = True, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = True, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = True,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = True
                    ):
    if True:
        if mass is None:
            mass = 61.8
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([3.428,2.904,1.275]))
            J = jmult*J
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.5*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 20.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.01*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))*0.00005*include_magicbias
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.5,1.0)*include_magicbias#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 5.0*np.ones(3)
        if magic_std is None:
            magic_std = 0.00001*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.0000001*np.ones(3)*include_magicbias
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*((math.pi/180.0)/3600)*include_gbias
                else:
                    gyro_bias0 = include_gbias*np.ones(3)*1.0*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = include_gbias*np.ones(3)*(1.0*math.pi/180.0)/(3600.0)
        if gyro_std is None:
            gyro_std = np.ones(3)*(0.025*math.pi/180.0)
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-9)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 50*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        lmult = jmult**(1/3)
        amult = jmult**(2/3)
        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.306*amult,lmult*(unitvecs[0]*0.17 ),unitvecs[0],2.2], \
                                [1,0.306*amult,lmult*(-unitvecs[0]*0.17),-unitvecs[0],2.2], \
                                [2,0.231*amult,lmult*(unitvecs[1]*0.25 ),unitvecs[1],2.2], \
                                [3,0.231*amult,lmult*(-unitvecs[1]*0.2),-unitvecs[1],2.2], \
                                [4,0.153*amult,unitvecs[2]*0.28*lmult,unitvecs[2],1.2], \
                                [5,0.153*amult,-unitvecs[2]*0.4*lmult,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)*5.0
                    else:
                        dipole0 = np.array([0.05,0.0001,0.2])*20*math.sqrt(jmult)
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 5.0*math.sqrt(jmult)
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces =   [  [0,0.306*amult,lmult*(unitvecs[0]*0.17 ),unitvecs[0],0.3,0.3,0.4], \
                                [1,0.306*amult,lmult*(-unitvecs[0]*0.17),-unitvecs[0],0.3,0.3,0.4], \
                                [2,0.231*amult,lmult*(unitvecs[1]*0.25 ),unitvecs[1],0.3,0.3,0.4], \
                                [3,0.231*amult,lmult*(-unitvecs[1]*0.2),-unitvecs[1],0.3,0.3,0.4], \
                                [4,0.153*amult,unitvecs[2]*0.28*lmult,unitvecs[2],0.2,0.4,0.4], \
                                [5,0.153*amult,-unitvecs[2]*0.4*lmult,-unitvecs[2],0.2,0.6,0.3]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-6
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        prop_torq0 = 1e-6*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-6*math.sqrt(jmult)
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-6
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        gen_torq0 = 1e-7*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5*math.sqrt(jmult)
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat

def create_Lovera_sat(  real = True, rand=False,
                    mass = None, J = None, COM = None,jmult = 1,
                    include_mtq = True, include_mtqbias = False, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = False, include_magicbias = True, include_magic_noise = True, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = True, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    use_gg = False,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = True
                    ):
    if True:
        if mass is None:
            mass = 70
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([27,17,25]))
            J = jmult*J
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.5*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 10.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))*0.5*include_magicbias
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)*include_magicbias#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 2.0*np.ones(3)
        if magic_std is None:
            magic_std = 0.00001*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.0000001*np.ones(3)*include_magicbias
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*((math.pi/180.0)/3600)*include_gbias
                else:
                    gyro_bias0 = include_gbias*np.ones(3)*1.0*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = include_gbias*np.ones(3)*(0.1*math.pi/180.0)/(3600.0)
        if gyro_std is None:
            gyro_std = np.ones(3)*(0.0025*math.pi/180.0)
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-9)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 50*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        lmult = jmult**(1/3)
        amult = jmult**(2/3)
        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.375*amult,unitvecs[0]*0.3*lmult,unitvecs[0],2.2], \
                                [1,0.375*amult,-unitvecs[0]*0.3*lmult,-unitvecs[0],2.2], \
                                [2,0.75*amult,unitvecs[1]*0.15*lmult,unitvecs[1],2.2], \
                                [3,0.75*amult,-unitvecs[1]*0.15*lmult,-unitvecs[1],2.2], \
                                [4,0.18*amult,unitvecs[2]*0.25*lmult,unitvecs[2],2.2], \
                                [5,0.18*amult,-unitvecs[2]*1.25*lmult,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)
                    else:
                        dipole0 = 0.5*np.ones(3)#np.array([0.1,0.0001,0.5])*math.sqrt(jmult)
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 2*math.sqrt(jmult)
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.375*amult,unitvecs[0]*0.3*lmult,unitvecs[0],0.3,0.4,0.3], \
                                [1,0.375*amult,-unitvecs[0]*0.3*lmult,-unitvecs[0],0.3,0.4,0.3], \
                                [2,0.75*amult,unitvecs[1]*0.15*lmult,unitvecs[1],0.3,0.4,0.3], \
                                [3,0.75*amult,-unitvecs[1]*0.15*lmult,-unitvecs[1],0.3,0.4,0.3], \
                                [4,0.18*amult,unitvecs[2]*0.25*lmult,unitvecs[2],0.2,0.7,0.1], \
                                [5,0.18*amult,-unitvecs[2]*1.25*lmult,-unitvecs[2],0.3,0.4,0.3]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-6
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        prop_torq0 = 1e-7*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5*math.sqrt(jmult)
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-8
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        gen_torq0 = 1e-7*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5*math.sqrt(jmult)
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est, disturbances = dists,estimated = True)
            return est_sat


def create_Lovera_sat_GPS(  real = True, rand=False,
                    mass = None, J = None, COM = None,jmult = 1,
                    include_mtq = True, include_mtqbias = False, include_mtq_noise = True, mtq_bias0 = None,mtq_std = None, mtq_bsr = None, mtq_max = None,estimate_mtq_bias = None,
                    include_magic = False, include_magicbias = True, include_magic_noise = True, magic_bias0 = None,magic_std = None, magic_bsr = None, magic_max = None,estimate_magic_bias = None,
                    include_sun = True, include_sbias = True, include_sun_noise = True, sun_std = None, sun_bias0 = None, sun_bsr = None, sun_eff = None,estimate_sun_bias = None,
                    include_gbias = True, include_gyro_noise = True, gyro_bias0 = None, gyro_bsr = None, gyro_std = None, estimate_gyro_bias = None,
                    include_mtmbias = False, include_mtm_noise = True, mtm_std = None, mtm_scale = None, mtm_bsr = None, mtm_bias0 = None, estimate_mtm_bias = None,
                    include_gpsbias = False, include_gps_noise = True, gps_std = None, gps_scale = None, gps_bsr = None, gps_bias0 = None, estimate_gps_bias = None,
                    use_gg = False,
                    use_drag = False, drag_dist = None,
                    use_dipole = False, dipole0 = None, dipole_std = None, dipole_mag_max = None, varying_dipole = True,estimate_dipole = None,
                    use_SRP = False, SRP_dist = None,
                    use_prop = False, prop_torq0 = None, prop_torq_std = None, prop_mag_max = None, varying_prop = True,estimate_prop_torq = None,
                    use_gen = False, gen_torq0 = None, gen_torq_std = None, gen_mag_max = None, varying_gen = True,estimate_gen_torq = None,
                    care_about_eclipse = True
                    ):
    if True:
        if mass is None:
            mass = 70
        if COM is None:
            COM = np.zeros(3)
        if J is None:
            J = np.diagflat(np.array([27,17,25]))
            J = jmult*J
        J = 0.5*(J+J.T)

        if mtq_bias0 is None:
            if real:
                if not rand:
                    mtq_bias0 = 0.5*normalize(np.array([1,1,4]))
                else:
                    mtq_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)#normalize(np.array([1,1,4]))
            else:
                mtq_bias0 = np.zeros(3)
        if mtq_max is None:
            mtq_max = 10.0*np.ones(3)
        if mtq_std is None:
            mtq_std = 0.0001*np.ones(3)
        if mtq_bsr is None:
            mtq_bsr = 0.000001*np.ones(3)
        if estimate_mtq_bias is None and not real:
            estimate_mtq_bias = include_mtqbias and np.all(mtq_bsr>1e-15)


        if magic_bias0 is None:
            if real:
                if not rand:
                    magic_bias0 = normalize(np.array([1,1,4]))*0.5*include_magicbias
                else:
                    magic_bias0 = random_n_unit_vec(3)*np.random.uniform(0.1,1.0)*include_magicbias#normalize(np.array([1,1,4]))
            else:
                magic_bias0 = np.zeros(3)
        if magic_max is None:
            magic_max = 2.0*np.ones(3)
        if magic_std is None:
            magic_std = 0.00001*np.ones(3)
        if magic_bsr is None:
            magic_bsr = 0.0000001*np.ones(3)*include_magicbias
        if estimate_magic_bias is None and not real:
            estimate_magic_bias = include_magicbias and np.all(magic_bsr>1e-15)


        if gyro_bias0 is None:
            if real:
                if rand:
                    gyro_bias0 = random_n_unit_vec(3)*np.random.uniform(0,5)*((math.pi/180.0)/3600)*include_gbias
                else:
                    gyro_bias0 = include_gbias*np.ones(3)*1.0*(math.pi/180)/3600
            else:
                gyro_bias0 = np.zeros(3)
        if gyro_bsr is None:
            gyro_bsr = include_gbias*np.ones(3)*(0.1*math.pi/180.0)/(3600.0)
        if gyro_std is None:
            gyro_std = np.ones(3)*(0.0025*math.pi/180.0)
        if estimate_gyro_bias is None and not real:
            estimate_gyro_bias = include_gbias and np.all(gyro_bsr>1e-15)

        if mtm_scale is None:
            mtm_scale = 1
        if mtm_bias0 is None:
            if real:
                if rand:
                    mtm_bias0 = random_n_unit_vec(3)*np.random.uniform(1e-10,5e-8)
                else:
                    mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))
            else:
                mtm_bias0 = np.zeros(3)
        if mtm_bsr is None:
            mtm_bsr = (1e-9)*np.ones(3) #1nT/sec
        if mtm_std is None:
            mtm_std = 50*1e-9*np.ones(3)
        if estimate_mtm_bias is None and not real:
            estimate_mtm_bias = include_mtmbias and np.all(mtm_bsr>1e-20)


        if gps_scale is None:
            gps_scale = 1
        if gps_bias0 is None:
            if real:
                if rand:
                    gps_bias0 = np.stack([random_n_unit_vec(3)*np.random.uniform(1,10),random_n_unit_vec(3)*np.random.uniform(0.01,0.5)])*gps_scale
                else:
                    gps_bias0 = np.array([1,2,3,0.1,0.01,0.2])*gps_scale
            else:
                gps_bias0 = np.zeros(3)
        if gps_bsr is None:
            gps_bsr = np.array([0.01,0.01,0.01,0.0001,0.0001,0.0001]) #1nT/sec
        if gps_std is None:
            gps_std = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        if estimate_gps_bias is None and not real:
            estimate_gps_bias = include_gpsbias and np.all(gps_bsr>1e-20)

        if sun_eff is None:
            sun_eff = 0.3*np.ones(3)
        if sun_std is None:
            sun_std = 0.001*sun_eff #0.1% of range
        if sun_bias0 is None:
            if real:
                if rand:
                    sun_bias0 = random_n_unit_vec(3)*np.random.uniform(0,0.05)*sun_eff
                else:
                    sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff
            else:
                sun_bias0 = np.zeros(3)
        if sun_bsr is None:
            sun_bsr = 0.00001*sun_eff #0.001% of range /sec
        if estimate_sun_bias is None and not real:
            estimate_sun_bias = include_sbias and np.all(sun_bsr>1e-15)

        lmult = jmult**(1/3)
        amult = jmult**(2/3)
        dists = []
        if use_drag:
            if drag_dist is None:
                drag_faces = [  [0,0.375*amult,unitvecs[0]*0.3*lmult,unitvecs[0],2.2], \
                                [1,0.375*amult,-unitvecs[0]*0.3*lmult,-unitvecs[0],2.2], \
                                [2,0.75*amult,unitvecs[1]*0.15*lmult,unitvecs[1],2.2], \
                                [3,0.75*amult,-unitvecs[1]*0.15*lmult,-unitvecs[1],2.2], \
                                [4,0.18*amult,unitvecs[2]*0.25*lmult,unitvecs[2],2.2], \
                                [5,0.18*amult,-unitvecs[2]*1.25*lmult,-unitvecs[2],2.2]]
                drag_dist = Drag_Disturbance(drag_faces)
            dists += [drag_dist]
        if use_gg:
            gg = GG_Disturbance()
            dists += [gg]
        if use_dipole:
            if dipole_std is None:
                dipole_std = 0.001
            if dipole0 is None:
                if real:
                    if rand:
                        dipole0 = random_n_unit_vec(3)*np.random.uniform(0,1)
                    else:
                        dipole0 = 0.5*np.ones(3)#np.array([0.1,0.0001,0.5])*math.sqrt(jmult)
                    estimate_dipole = False
                else:
                    dipole0 = np.zeros(3)
                    if estimate_dipole is None:
                        estimate_dipole = np.all(dipole_std>1e-15)
            if dipole_mag_max is None:
                dipole_mag_max = 2*math.sqrt(jmult)
            dip_dist = Dipole_Disturbance([dipole0,dipole_mag_max],time_varying=varying_dipole,std = dipole_std,estimate=estimate_dipole)
            dists += [dip_dist]
        if use_SRP:
            if SRP_dist is None:
                SRP_faces = [  [0,0.375*amult,unitvecs[0]*0.3*lmult,unitvecs[0],0.3,0.4,0.3], \
                                [1,0.375*amult,-unitvecs[0]*0.3*lmult,-unitvecs[0],0.3,0.4,0.3], \
                                [2,0.75*amult,unitvecs[1]*0.15*lmult,unitvecs[1],0.3,0.4,0.3], \
                                [3,0.75*amult,-unitvecs[1]*0.15*lmult,-unitvecs[1],0.3,0.4,0.3], \
                                [4,0.18*amult,unitvecs[2]*0.25*lmult,unitvecs[2],0.2,0.7,0.1], \
                                [5,0.18*amult,-unitvecs[2]*1.25*lmult,-unitvecs[2],0.3,0.4,0.3]]
                srp_dist = SRP_Disturbance(SRP_faces)
            dists += [srp_dist]
        if use_prop:
            if prop_torq_std is None:
                prop_torq_std = 1e-6
            if prop_torq0 is None:
                if real:
                    if rand:
                        prop_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        prop_torq0 = 1e-7*normalize(np.array([-3,-8,1]))
                    estimate_prop_torq = False
                else:
                    prop_torq0 = np.zeros(3)
                    if estimate_prop_torq is None:
                        estimate_prop_torq = np.all(prop_torq_std>1e-15)
            if prop_mag_max is None:
                prop_mag_max = 1e-5*math.sqrt(jmult)
            prop_dist = Prop_Disturbance([prop_torq0,prop_mag_max],time_varying=varying_prop,std = prop_torq_std,estimate = estimate_prop_torq)
            dists += [prop_dist]
        if use_gen:
            if gen_torq_std is None:
                gen_torq_std = 1e-8
            if gen_torq0 is None:
                if real:
                    if rand:
                        gen_torq0 = random_n_unit_vec(3)*np.random.uniform(1e-12,1e-6)
                    else:
                        gen_torq0 = 1e-7*normalize(np.array([1,-1,1]))
                    estimate_gen_torq = False
                else:
                    gen_torq0 = np.zeros(3)
                    if estimate_gen_torq is None:
                        estimate_gen_torq = np.all(gen_torq_std>1e-15)
            if gen_mag_max is None:
                gen_mag_max = 1e-5*math.sqrt(jmult)
            gen_dist = General_Disturbance([gen_torq0,gen_mag_max],time_varying=varying_gen,std = gen_torq_std,estimate = estimate_gen_torq)
            dists += [gen_dist]


        if real:
            acts = []
            if include_mtq:
                acts +=  [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise=include_mtq_noise,bias_std_rate=mtq_bsr[j]) for j in range(3)]
            if include_magic:
                acts +=  [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise=include_magic_noise,bias_std_rate=magic_bsr[j]) for j in range(3)]
            mtms = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale) for j in range(3)]
            gyros = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j]) for j in range(3)]
            gps = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]
            suns = []
            if include_sun:
                j = 1
                suns = [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],respect_eclipse = care_about_eclipse) for j in range(3)]
            real_sat = Satellite(mass = mass,J = J,actuators = acts, sensors = mtms+gyros+suns+gps, disturbances = dists)
            return real_sat
        else:
            acts_est = []
            if include_mtq:
                acts_est += [MTQ(unitvecs[j],mtq_std[j],mtq_max[j],has_bias = include_mtqbias,bias = mtq_bias0[j],use_noise = include_mtq_noise,bias_std_rate=mtq_bsr[j],estimate_bias = estimate_mtq_bias) for j in range(3)]
            if include_magic:
                acts_est += [Magic(unitvecs[j],magic_std[j],magic_max[j],has_bias = include_magicbias,bias = magic_bias0[j],use_noise = include_magic_noise,bias_std_rate=magic_bsr[j],estimate_bias = estimate_magic_bias) for j in range(3)]
            mtms_est = [MTM(unitvecs[j],mtm_std[j],has_bias = include_mtmbias,bias = mtm_bias0[j],use_noise = include_mtm_noise,bias_std_rate = mtm_bsr[j],scale = mtm_scale,estimate_bias = estimate_mtm_bias) for j in range(3)]
            gyros_est = [Gyro(unitvecs[j],gyro_std[j],has_bias = include_gbias,bias = gyro_bias0[j],use_noise = include_gyro_noise,bias_std_rate = gyro_bsr[j],estimate_bias = estimate_gyro_bias) for j in range(3)]
            gps_est = [GPS(gps_std,has_bias = include_gpsbias,bias = gps_bias0,use_noise = include_gps_noise,bias_std_rate = gps_bsr)]

            suns_est = []
            if include_sun:
                j = 1
                suns_est =  [SunSensorPair(unitvecs[j],sun_std[j],sun_eff[j],has_bias = include_sbias,bias = sun_bias0[j],use_noise = include_sun_noise,bias_std_rate = sun_bsr[j],estimate_bias = estimate_sun_bias,respect_eclipse = care_about_eclipse) for j in range(3)]
            est_sat = Satellite(mass = mass,J = J,actuators = acts_est, sensors = mtms_est+gyros_est+suns_est+gps_est, disturbances = dists,estimated = True)
            return est_sat
