from .control_mode import *
from scipy.optimize import minimize,Bounds,LinearConstraint
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class TrajectoryMPC(ControlMode):
    def __init__(self,gain_info,sat,maintain_RW = True,include_disturbances=True,calc_av_from_quat = False,include_rotational_motion = True):
        ModeName = GovernorMode.PLAN_AND_TRACK_MPC

        params = Params()
        # self.tracking_LQR_formulation = tracking_LQR_formulation
        if len(gain_info) == 0:
            gain_info = [1,10] + [0]*6 + [1,0,1e-10,0]
        elif len(gain_info) == 1:
            gain_info = gain_info + [10] + [0]*6 + [1,0,1e-10,0]
        elif len(gain_info) < 8:
            gain_info = gain_info + [0]*(8-len(gain_info))  + [1,0,1e-10,0]
        elif len(gain_info) == 8:
            gain_info = gain_info  + [1,0,1e-10,0]
        elif len(gain_info) == 9:
            gain_info = gain_info  + [0,1e-10,0]
        elif len(gain_info) == 10:
            gain_info = gain_info  + [1e-10,0]
        elif len(gain_info) == 11:
            gain_info = gain_info  + [0]
        params.dt = gain_info[0]
        params.addl_ang_err_wt_boundary = gain_info[1] #in degrees. if ang error more than this, uses a higher weight to angular error.
        params.addl_ang_err_wt_low = gain_info[2]
        params.addl_ang_err_wt_high = gain_info[3]
        params.addl_av_err_wt = gain_info[4]
        params.addl_avang_err_wt = gain_info[5]
        params.addl_extra_err_wt = gain_info[6]
        params.addl_ctrl_diff_from_plan_wt = gain_info[7]
        params.addl_ctrl_diff_from_prev_wt = gain_info[8]
        params.mpc_lqrwt_mult = gain_info[9]
        params.extra_tests = gain_info[10]
        params.tol = gain_info[11]
        params.plot_resultsN = gain_info[12]
        self.prev_ctrl = None
        self.prev_exists = False
        self.prev_guess_state = np.zeros(7)
        self.quatvecmode = 0

        if params.plot_resultsN>0:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')


        # self.gain = gain
        super().__init__(ModeName,sat,params,False,include_disturbances,False,False)


    def scoring_func(self,u,rk4func,weight,u_weight_from_plan, u_weight_from_prev,next_plan,plan_control,print_data=False):
        xkp1 = rk4func(u)
        # print('test',u.T)
        xerr = self.reduced_state_err(xkp1,next_plan,quatvecmode=self.quatvecmode)
        # breakpoint()
        cost = xerr@weight@xerr
        cost += (u-plan_control)@u_weight_from_plan@(u-plan_control)
        if self.prev_exists:
            cost += (u-self.prev_ctrl)@u_weight_from_prev@(u-self.prev_ctrl)
        if print_data:
            err = self.state_err(xkp1,next_plan)
            print("quat diff: ",err[3:])
            print("av diff: ",err[:3])
            print("ctrl diff: ",u-plan_control)
            print("diff mags: ",quat_ang_diff_deg(xkp1[3:],next_plan.state[3:]),norm(err[:3])*180.0/math.pi,norm(u-plan_control))
        return cost

    def scoring_func_du(self,u,rk4func,rk4jac_func,weight,u_weight_from_plan, u_weight_from_prev,next_plan,plan_control):
        xkp1 = rk4func(u)
        xerr = self.reduced_state_err(xkp1,next_plan,quatvecmode=self.quatvecmode)
        # breakpoint()
        __,dxkp1_du = rk4jac_func(u)
        # print(dxkp1_du)
        dcost_du = 2*xerr@weight@self.reduced_state_err_jac(xkp1,next_plan,quatvecmode=self.quatvecmode).T@dxkp1_du.T
        dcost_du += 2*(u-plan_control)@u_weight_from_plan
        if self.prev_exists:
            dcost_du += 2*(u-self.prev_ctrl)@u_weight_from_prev
        return dcost_du


    def scoring_func_dudu(self,u,rk4func,rk4jac_func,rk4hess_func,weight,u_weight_from_plan, u_weight_from_prev,next_plan,plan_control):
        xkp1 = rk4func(u)
        xerr = self.reduced_state_err(xkp1,next_plan,quatvecmode=self.quatvecmode)
        # breakpoint()
        __,dxkp1_du = rk4jac_func(u)
        __,dxkp1_dudu,__,__,__  = rk4hess_func(u)
        dxerr_dx = self.reduced_state_err_jac(xkp1,next_plan,quatvecmode=self.quatvecmode)
        dexerr_du = dxerr_dx@dxkp1_du.T
        dexerr_dxdx = self.reduced_state_err_hess(xkp1,next_plan,quatvecmode=self.quatvecmode)

        dxerr_dudu = np.tensordot(dexerr_dxdx,dxkp1_du) + np.tensordot(dxerr_dx@dxkp1_dudu)
        dcost_dudu = 2*dxerr_du.T@weight@dxerr_du + 2*np.tensordot(xerr@weight,dxerr_dudu)
        dcost_dudu += 2*u_weight_from_plan
        if self.prev_exists:
            dcost_dudu += 2*u_weight_from_prev
        return dcost_dudu




    def find_actuation(self, state, os, osp1, goal_state, prev_goal,next_goal, sens,planner_params,is_fake):
        """
        This function finds the commanded control input.

        """
        q = state[3:7]
        plan_state = planner_params[0][-1]
        plan_control = planner_params[0][1]
        plan_gain = planner_params[0][2]
        planned_torq = planner_params[0][4]

        next_plan_state = planner_params[1][-1]
        next_plan_control = planner_params[1][1]
        next_plan_gain = planner_params[1][2]
        next_plan_ctg = planner_params[1][3]
        next_planned_torq = planner_params[1][4]

        err = self.reduced_state_err(state,plan_state,quatvecmode=self.quatvecmode)

        print('mag w err',norm(err[0:3]))
        print('mag ang err',norm(err[3:6]))

        u = plan_control - err@plan_gain.T
        # print(plan_gain)

        # breakpoint()
        ang_weight = self.params.addl_ang_err_wt_low
        angerr = quat_ang_diff_deg(q,plan_state.state[3:7])#(180.0/math.pi)*math.acos(np.clip(2.0*err[3]**2.0-1.0,-1,1))
        # print('angerr ',angerr)
        if angerr > self.params.addl_ang_err_wt_boundary:
            ang_weight = self.params.addl_ang_err_wt_high
        wt = next_plan_gain.T@next_plan_gain*self.params.mpc_lqrwt_mult + scipy.linalg.block_diag(np.block([[np.eye(3)*self.params.addl_av_err_wt,np.eye(3)*self.params.addl_avang_err_wt],[np.eye(3)*self.params.addl_avang_err_wt,np.eye(3)*ang_weight]]), np.eye(self.sat.state_len-7)*self.params.addl_extra_err_wt)

        print(wt)
        print(next_plan_gain.T@next_plan_gain)
        ctrlwt1 = np.eye(self.sat.control_len)*self.params.addl_ctrl_diff_from_plan_wt
        ctrlwt2 = np.eye(self.sat.control_len)*self.params.addl_ctrl_diff_from_prev_wt
        # breakpoint()
        next_state_func = lambda u, self=self,os = os,osp1=osp1: self.sat.noiseless_rk4(state[:self.sat.state_len],u,self.params.dt,os,osp1,verbose=False,quat_as_vec = True,save_info = False)
        func = lambda u, self=self,next_state_func=next_state_func,wt=wt, next_plan_state = next_plan_state: self.scoring_func(u,next_state_func,wt,ctrlwt1,ctrlwt2,next_plan_state,plan_control)
        printfunc = lambda u, self=self,next_state_func=next_state_func,wt=wt, next_plan_state = next_plan_state: self.scoring_func(u,next_state_func,wt,ctrlwt1,ctrlwt2,next_plan_state,plan_control,print_data=True)
        next_state_jac_func = lambda u, self=self,os = os,osp1=osp1: self.sat.rk4Jacobians(state[:self.sat.state_len],u,self.params.dt,os,osp1,quat_as_vec = True)
        func_jac = lambda u, self=self,next_state_func=next_state_func,wt=wt, next_plan_state = next_plan_state: self.scoring_func_du(u,next_state_func,next_state_jac_func,wt,ctrlwt1,ctrlwt2,next_plan_state,plan_control).reshape(self.sat.control_len,)
        # next_state_hess_func = lambda u, self=self,os = os,osp1=osp1: self.sat.rk4_u_Hessians(state[:self.sat.state_len],u,self.params.dt,os,osp1)
        # func_hess = lambda u, self=self,next_state_func=next_state_func,wt=wt, next_plan_state = next_plan_state: self.scoring_func_dudu(u,next_state_func,next_state_jac_func,next_state_hess_func,wt,ctrlwt1,ctrlwt2,next_plan_state,plan_control)
        # func = lambda u, self=self,next_state=next_state,next_plan_gain = next_plan_gain, next_plan_state = next_plan_state: self.scoring_func(u,next_state,next_plan_ctg,next_plan_state)
        # print('current: ',state[:self.sat.state_len].T)
        # print('pred   : ',self.prev_guess_state.T)
        # print('plan   : ',plan_state.state.T)
        #
        # print('err pred v real   : ',self.reduced_state_err(state[:self.sat.state_len],self.prev_guess_state,quatvecmode=self.quatvecmode).T)
        # print('err pred v goal   : ',self.reduced_state_err(self.prev_guess_state,plan_state.state,quatvecmode=self.quatvecmode).T)
        # print("orig ",u.T)
        lb,ub = self.sat.control_bounds()
        if self.sat.number_RW>0:
            pass #add constraints on h
        bounds = Bounds(lb=lb,ub=ub)
        # res = minimize(func,u,bounds=bounds,tol = self.params.tol,jac = func_jac)#,constraints=y)
        res = minimize(func,u,bounds=bounds,tol = self.params.tol,method='Nelder-Mead')#,jac = func_jac)#,constraints=y)

        bestres = res.fun
        bestu = res.x


        combomat = np.stack([lb,ub])
        reses = [[u,res]]
        scores = [res.fun]
        if self.params.extra_tests > 0:
            for k in range(2**lb.size):
                bitstr = np.binary_repr(k,width = lb.size)
                testu = np.array([combomat[int(bitstr[j]),j] for j in range(lb.size)])
                # print(testu.T)
                newres =  minimize(func,testu,bounds=bounds,tol = self.params.tol)#,jac = func_jac)
                reses = reses + [[testu,newres]]
                scores = scores + [newres.fun]
                if newres.fun < bestres:
                    print("defeated!",bestres,newres.fun)
                    bestres = newres.fun
                    bestu = newres.x
                    res = newres
            for k in range(self.params.extra_tests):
                uin = 2*np.random.rand(lb.size) - 1
                newres =  minimize(func,uin,bounds=bounds,tol = self.params.tol)#,jac = func_jac)
                reses = reses + [[uin,newres]]
                scores = scores + [newres.fun]
                if newres.fun < bestres:
                    print("defeated!",bestres,newres.fun)
                    bestres = newres.fun
                    bestu = newres.x
                    res = newres

        if self.params.plot_resultsN > 0:
            sc_factor = reses[0][1].fun
            self.ax.clear()
            #generate N^N_ctrl points
            #TODO: make this work for N_ctrl > 3
            x = np.linspace(lb[0], ub[0], self.params.plot_resultsN)
            y = np.linspace(lb[1], ub[1], self.params.plot_resultsN)
            z = np.linspace(lb[2], ub[2], self.params.plot_resultsN)
            mesh = list(np.meshgrid(*[x,y,z]))
            mesh = np.stack([j.flatten() for j in mesh])
            #test each
            pts = [np.concatenate([mesh[:,i],[func(mesh[:,i])/sc_factor]]) for i in range(mesh.shape[1])]
            #plot with projection onto B body #TODO change for other plotting cases (not just MTQs)

            vecs = os_local_vecs(os,q)
            ax1 = normalize(np.cross(unitvecs[0],vecs["b"]))
            ax2 = normalize(np.cross(ax1,vecs["b"]))

            reduced = np.stack([[np.dot(ax1,pts[k][0:3]),np.dot(ax2,pts[k][0:3]),pts[k][3]] for k in range(len(pts))])
            self.ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2])
            scores = scores + [k[3]*sc_factor for k in pts]


            for r in reses:
                x1 = np.dot(ax1,r[0])
                y1 = np.dot(ax2,r[0])
                z1 = func(r[0])/sc_factor
                x2 = np.dot(ax1,r[1].x)
                y2 = np.dot(ax2,r[1].x)
                z2 = r[1].fun/sc_factor
                self.ax.quiver(x1,y1,z1,x2-x1,y2-y1,z2-z1,color='r')
                self.ax.scatter(x1,y1,z1,c='g',marker = '^')
                self.ax.scatter(x2,y2,z2,c='g',marker = '*')


                # arw = Arrow3D([x1,x2],[y1,y2],[z1,z2], arrowstyle="->", color="purple", lw = 3, mutation_scale=25)
                # self.ax.add_artist(arw)
            plt.draw()
            plt.pause(1)
            # breakpoint()



        print("optd ",res.x.T)
        # print("optres ",res.fun)
        # print(printfunc(u),printfunc(res.x),res.fun,min(scores))
        print(printfunc(u),res.fun)
        # print(next_state_func(u).T)

        # print(next_plan_state.state.T)
        # print(next_plan_gain.T@next_plan_gain)

        # print(res.success)
        print(res.message)
        # breakpoint()

        if not is_fake:
            self.prev_ctrl = res.x
            self.prev_guess_state = next_state_func(res.x)
            if not self.prev_exists:
                self.prev_exists = True
        return res.x
