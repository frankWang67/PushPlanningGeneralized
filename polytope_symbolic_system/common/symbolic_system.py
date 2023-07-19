import casadi as cs
import pydrake.symbolic as sym
import numpy as np
from shapely.geometry import MultiPolygon
from shapely import intersects
from pypolycontain.lib.zonotope import *
# from pypolycontain.lib.zonotope import *
from pypolycontain.lib.AH_polytope import AH_polytope
from pypolycontain.lib.objects import H_polytope
from pypolycontain.lib.operations import *
from polytope_symbolic_system.common.intfunc import rect_cs
from polytope_symbolic_system.common.utils import *


def extract_variable_value_from_env(symbolic_var, env):
    # symbolic_var is a vector
    var_value = np.zeros(symbolic_var.shape[0])
    for i in range(symbolic_var.shape[0]):
        var_value[i] = env[symbolic_var[i]]
    return var_value

class Dynamics:
    def __init__(self):
        self.type='undefined'
        pass

    def construct_linearized_system_at(self, env):
        raise NotImplementedError


class ContinuousLinearDynamics(Dynamics):
    def __init__(self, A, B, c):
        Dynamics.__init__(self)
        self.A = A
        self.B = B
        self.c = c
        self.type='continuous'

    def construct_linearized_system_at(self, env):
        print('Warning: system is already linear!')
        return self

    def evaluate_xdot(self, x, u):
        return np.dot(self.A,x)+np.dot(self.B,u)+self.c


class ContinuousDynamics(Dynamics):
    '''
    System described by xdot(t) = f(x(t), u(t))
    '''
    def __init__(self, f, x, u):
        Dynamics.__init__(self)
        self.f = f
        self.x = x
        self.u = u
        self.type='continuous'
        self._linearlize_system()

    def _linearlize_system(self):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def construct_linearized_system_at(self, env):
        return ContinuousLinearDynamics(sym.Evaluate(self.A, env), sym.Evaluate(self.B, env), sym.Evaluate(self.c, env))

    def evaluate_xdot(self, env, linearize):
        if linearize:
            linsys = self.construct_linearized_system_at(env)
            x_env = extract_variable_value_from_env(self.x, env)
            u_env = extract_variable_value_from_env(self.u, env)
            return linsys.evaluate_xdot(x_env, u_env)
        else:
            return sym.Evaluate(self.f, env)

class DiscreteLinearDynamics(Dynamics):
    def __init__(self, A, B, c, E=None, Xi=None, x_bar=None, u_bar=None):
        Dynamics.__init__(self)
        self.A = A
        self.B = B
        self.c = c
        self.E = E
        self.Xi = Xi
        self.x_bar = x_bar
        self.u_bar = u_bar
        self.type='discrete'

    def construct_linearized_system_at(self, env):
        print('Warning: system is already linear!')
        return self

    def evaluate_x_next(self, x, u):
        return np.dot(self.A,x)+np.dot(self.B,u)+self.c

class DiscreteDynamics(Dynamics):
    '''
    System described by x[t+1] = f(x[t], u[t])
    '''
    def __init__(self, f, x, u):
        Dynamics.__init__(self)
        self.f = f
        self.x = x
        self.u = u
        self.type='discrete'
        self._linearlize_system()

    def _linearlize_system(self):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def construct_linearized_system_at(self, env):
        return DiscreteLinearDynamics(sym.Evaluate(self.A, env), sym.Evaluate(self.B, env), sym.Evaluate(self.c, env))

    def evaluate_x_next(self, env, linearize=False):
        if linearize:
            linsys = self.construct_linearized_system_at(env)
            x_env = extract_variable_value_from_env(self.x, env)
            u_env = extract_variable_value_from_env(self.u, env)
            return linsys.evaluate_x_next(x_env, u_env)
        return sym.Evaluate(self.f, env)

class DTContinuousSystem:
    def __init__(self, f, x, u, initial_env=None, input_limits = None):
        '''
        Continuous dynamical system x_dot = f(x,u)
        :param f: A symbolic expression of the system dynamics.
        :param x: A list of symbolic variables. States.
        :param u: A list of symbolic variable. Inputs.
        :param initial_env: A dictionary "environment" specifying the initial state of the system
        :param input_limits: Input limits of the system
        '''
        self.dynamics = ContinuousDynamics(f,x,u)
        if input_limits is None:
            self.input_limits = np.vstack([np.full(u.shape[0], -1e9),np.full(u.shape[0], 1e9)]) #stack add rows
        else:
            self.input_limits = input_limits
        self.u_bar = np.average(self.input_limits, axis=0)
        self.u_diff = np.diff(self.input_limits, axis=0)/2.
        if initial_env is None:
            self.env = {}
            for x_i in self.dynamics.x:
                self.env[x_i] = 0.
            for i,u_i in enumerate(self.dynamics.u):
                self.env[u_i]=self.u_bar[i]
        else:
            self.env = initial_env


    def forward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False, starting_state=None):
        #propagate state by one time step
        if starting_state is not None:
            new_env = self._state_to_env(starting_state, u)  #just fill in the particular state and control input in discretized system
        elif not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                #bound the control input and add to the system environment info
                new_env[self.dynamics.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
        else:
            for i in range(self.dynamics.u.shape[0]):
                new_env[self.dynamics.u[i]] = self.u_bar[i]
        delta_x = self.dynamics.evaluate_xdot(new_env, linearlize)*step_size
        #assign new xs
        for i in range(delta_x.shape[0]):
            new_env[self.dynamics.x[i]] += delta_x[i]
        if return_as_env:
            return new_env
        else:
            return extract_variable_value_from_env(self.dynamics.x, new_env)

    def get_reachable_polytopes(self, state, step_size = 1e-2, use_convex_hull=False):
        current_linsys = self.get_linearization(state, self.u_bar)
        #x_{k+1} = (A \Delta t+ I)x_k + B\Delta t u_bar + c \Delta
        x = np.ndarray.flatten(np.dot(current_linsys.A*step_size+np.eye(current_linsys.A.shape[0]),state))+\
            np.dot(current_linsys.B*step_size, self.u_bar)+np.ndarray.flatten(current_linsys.c*step_size)
        # print(f"symbolic system: x before :{x}, x shape: {x.shape}")
        x = np.atleast_2d(x).reshape(-1,1) #convert to column vector if a row vector
        # x = np.atleast_2d(x)
        # print(f"symbolic system: x after:{x}, state:{state} ")
        assert(len(x)==len(state))
        G_before = np.dot(current_linsys.B*step_size, np.diag(self.u_diff))
        # print(f"G before:{G_before}")
        # G = np.atleast_2d(np.dot(current_linsys.B*step_size, np.diag(self.u_diff)))
        G = np.atleast_2d(np.dot(current_linsys.B*step_size, np.diag(self.u_diff))).reshape(-1,1)
        if use_convex_hull:
            # print("in convex hull")
            # print(f"x:{x}")
            # print(f"G:{G}")
            return convex_hull_of_point_and_polytope(state.reshape(x.shape),zonotope(x,G))
        return to_AH_polytope(zonotope(x,G)) #this is a bad approximation


    def get_linearization(self, state=None, u_bar = None, mode=None):
        if state is None:
            return self.dynamics.construct_linearized_system_at(self.env)
        else:
            env = self._state_to_env(state, u_bar)
            return self.dynamics.construct_linearized_system_at(env)

    def _state_to_env(self, state, u=None):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.dynamics.x[i]] = s_i  #i^th component of state
        if u is None:
            for i, u_i in enumerate(self.dynamics.u):
                env[u_i] = self.u_bar[i]
        else:
            for i, u_i in enumerate(u):
                env[self.dynamics.u[i]] = u[i] # i^th control input
        return env

    def get_current_state(self):
        return extract_variable_value_from_env(self.dynamics.x, self.env)

def in_mode(c_i, env):
    for c_ij in c_i:
        if c_ij.Evaluate(env) is False:
            return False
    return True

class DTHybridSystem:
    def __init__(self, f_list, f_type_list, x, u, c_list, initial_env=None, input_limits=None):
        '''
        Hybrid system with multiple dynamics modes
        :param f_list: numpy array of system dynamics modes
        :param x: pydrake symbolic variables representing the states
        :param u: pydrake symbolic variables representing the inputs
        :param c_list: numpy array of Pydrake symbolic expressions c_i(x,u) describing when the system belong in that mode.
                        c_i(x,u) is a vector of functions describing the mode. All of c_i(x,u) >= 0 when system is in mode i.
                        Modes should be complete and mutually exclusive.
        :param initial_env: A dictionary "environment" specifying the initial state of the system
        :param input_limits: Input limits of the system
        '''
        assert f_list.shape[0] == c_list.shape[0]
        self.mode_count = f_list.shape[0]
        self.x = x
        self.u = u
        dynamics_list = []
        for i, f in enumerate(f_list):
            if f_type_list[i] == 'continuous':
                dynamics_list.append(ContinuousDynamics(f,self.x,self.u))
            elif f_type_list[i] == 'discrete':
                dynamics_list.append(DiscreteDynamics(f, self.x, self.u))
            else:
                raise ValueError
        self.dynamics_list = np.asarray(dynamics_list)

        if input_limits is None:
            self.input_limits = np.vstack([np.full(u.shape[0], -1e9),np.full(u.shape[0], 1e9)])
        else:
            self.input_limits = input_limits
        self.u_bar = np.atleast_2d((self.input_limits[1,:]+self.input_limits[0,:])/2.)
        self.u_diff = np.atleast_2d((self.input_limits[1,:]-self.input_limits[0,:])/2.)
        if initial_env is None:
            self.env = {}
            for x_i in self.x:
                self.env[x_i] = 0.
            for i,u_i in enumerate(self.u):
                self.env[u_i]=self.u_bar[i]
        else:
            self.env = initial_env
        self.c_list = c_list
        # Check the mode the system is in
        self.current_mode = -1
        #TODO

    def do_internal_updates(self):
        pass

    def forward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False,
                     return_mode = False, starting_state=None):
        if starting_state is not None:
            new_env = self._state_to_env(starting_state, u)
        elif not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                new_env[self.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
        else:
            for i in range(self.u.shape[0]):
                #ensure u is scalar
                new_env[self.u[i]] = np.ndarray.flatten(np.atleast_1d(self.u_bar))[i]
        # Check for which mode the system is in
        delta_x = None
        x_new = None
        mode = -1
        for i, c_i in enumerate(self.c_list):
            is_in_mode = in_mode(c_i,new_env)
            if not is_in_mode:
                continue
            if self.dynamics_list[i].type == 'continuous':
                delta_x = self.dynamics_list[i].evaluate_xdot(new_env, linearlize)*step_size
            elif self.dynamics_list[i].type == 'discrete':
                x_new = self.dynamics_list[i].evaluate_x_next(new_env, linearlize)
            else:
                raise ValueError
            mode = i
            break
        assert(mode != -1) # The system should always be in one mode
        # print('mode', mode)
        # print(self.env)
        #FIXME: check if system is in 2 modes (illegal)

        #assign new xs
        if self.dynamics_list[mode].type=='continuous':
            for i in range(delta_x.shape[0]):
                new_env[self.x[i]] += delta_x[i]
        elif self.dynamics_list[mode].type=='discrete':
            for i in range(x_new.shape[0]):
                new_env[self.x[i]] = x_new[i]
        else:
            raise ValueError

        self.do_internal_updates()

        #return options
        if return_as_env and not return_mode:
            return new_env
        elif return_as_env and return_mode:
            return new_env, mode
        elif not return_as_env and not return_mode:
            return extract_variable_value_from_env(self.x, new_env)
        else:
            return extract_variable_value_from_env(self.x, new_env), mode

    def get_reachable_polytopes(self, state, step_size=1e-2, use_convex_hull=False):
        polytopes_list = []
        for mode, c_i in enumerate(self.c_list):
            # FIXME: better way to check if the mode is possible
            # Very naive check: if all-min and all-max input lead to not being in mode, assume state is not in mode
            lower_bound_env = self.env.copy()
            upper_bound_env = self.env.copy()
            unactuated_env = self.env.copy()
            for i, u_i in enumerate(self.u):
                lower_bound_env[u_i] = self.input_limits[0,i]
                upper_bound_env[u_i] = self.input_limits[1, i]
                unactuated_env[u_i] = 0
            for i, x_i in enumerate(self.x):
                lower_bound_env[x_i] = state[i]
                upper_bound_env[x_i] = state[i]
                unactuated_env[x_i] = state[i]

            if (not in_mode(c_i, lower_bound_env)) and (not in_mode(c_i, upper_bound_env)) and not in_mode(c_i, unactuated_env):
                # print('dropping mode %i' %mode)
                continue

            current_linsys = self.get_linearization(state, mode=mode)
            if current_linsys is None:
                # this should not happen?
                raise Exception
            u_bar = (self.input_limits[1, :] + self.input_limits[0, :]) / 2.
            u_diff = (self.input_limits[1, :] - self.input_limits[0, :]) / 2.
            # print(mode)
            #     print('A', current_linsys.A)
            # print('B', current_linsys.B)
            #     print('c', current_linsys.c)

            if self.dynamics_list[mode].type == 'continuous':
                x = np.ndarray.flatten(
                    np.dot(current_linsys.A * step_size + np.eye(current_linsys.A.shape[0]), state)) + \
                    np.dot(current_linsys.B * step_size, u_bar) + np.ndarray.flatten(current_linsys.c * step_size)
                x = np.atleast_2d(x).reshape(-1, 1)
                assert (len(x) == len(state))
                G = np.atleast_2d(np.dot(current_linsys.B * step_size, np.diag(u_diff)))

            elif self.dynamics_list[mode].type == 'discrete':
                x = np.ndarray.flatten(
                    np.dot(current_linsys.A, state)) + \
                    np.dot(current_linsys.B, u_bar) + np.ndarray.flatten(current_linsys.c)
                x = np.atleast_2d(x).reshape(-1, 1)
                assert (len(x) == len(state))
                G = np.atleast_2d(np.dot(current_linsys.B, np.diag(u_diff)))
                # print('x', x)
                # print('G', G)
            else:
                raise ValueError
            # if mode==1:
            #     print(G, x)
            if use_convex_hull:
                polytopes_list.append(convex_hull_of_point_and_polytope(state.reshape(x.shape), zonotope(x,G)))
            else:
                polytopes_list.append(to_AH_polytope(zonotope(x, G)))
        return np.asarray(polytopes_list)

    def get_linearization(self, state=None, u_bar = None, mode=None):
        if state is None:
            return self.dynamics_list[self.current_mode].construct_linearized_system_at(self.env)
        else:
            env = self._state_to_env(state, u_bar)
            if mode is not None:
                # FIXME: construct but don't ask questions?
                # assert in_mode(self.c_list[mode], env)
                return self.dynamics_list[mode].construct_linearized_system_at(env)
            for mode, c_i in enumerate(self.c_list):
                if in_mode(c_i, env):
                    return self.dynamics_list[mode].construct_linearized_system_at(env)
            print('Warning: state is not in any mode')
            return None

    def _state_to_env(self, state, u=None):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.x[i]] = s_i
        if u is None:
            for i, u_i in enumerate(self.u):
                env[u_i] = self.u_bar[i]
        else:
            for i, u_i in enumerate(u):
                env[self.u[i]] = u[i]
        return env

    def get_current_state(self):
        return extract_variable_value_from_env(self.x, self.env)



class PushDTHybridSystem:
    def __init__(self, f_lim=0.3, dpsic_lim=3.0, unilateral_sliding_region=0.005, slider_geom=[0.07, 0.12, 0.01],
                       miu_slider_pusher=0.3, miu_slider_ground=0.2, quad_cost_input=[0., 0., 0.],
                       reachable_set_time_step=0.05, nldynamics_time_step=0.01) -> None:
        """
        The PushDTHybridSystem, to do forward simulation, compute reachable polytopes of the pusher-slider system
        :param f_lim: contact force limit
        :param dpsic_lim: pusher velocity limit
        :param unilateral_sliding_region: width of the region where only unilateral sliding mode is allowed
        :param slider_geom: [xl, yl, rl] of the pusher-slider system
        :param miu_slider_pusher: friction coeff between slider and pusher
        :param miu_slider_ground: friction coeff between slider and ground
        :param reachable_set_time_step: time step when computing reachable set
        :param nldynamics_time_step: time step when doing forward simulation
        """
        self.f_lim = f_lim  # default: 0.3
        self.dpsic_lim = dpsic_lim  # default: 3.0
        self.unilateral_sliding_region = unilateral_sliding_region  # default: 0.005
        self.slider_geom = slider_geom  # default: [0.07, 0.12, 0.01]

        self.miu_slider_pusher = miu_slider_pusher  # default: 0.3
        self.miu_slider_ground = miu_slider_ground  # default: 0.2

        self.quad_cost_input = quad_cost_input  # default: [0., 0., 0.]

        self.reachable_set_time_step = reachable_set_time_step  # default: 0.05
        self.nldynamics_time_step = nldynamics_time_step  # default: 0.01

        self.contact_face_list = ['front', 'back', 'left', 'right']
        # self.contact_mode_list = ['sticking', 'sliding_left', 'sliding_right']  # not necessary to explicitly use sliding
        self.contact_mode_list = ['sticking']
        self.psic_each_face_center = {'front': 0.,
                                      'back': np.pi,
                                      'left': 0.5*np.pi,
                                      'right': -0.5*np.pi}
        self.u_domain_polytope_keypoint = {'sticking': np.array([[0., 0., 0.],
                                                                 [self.f_lim, self.miu_slider_pusher*self.f_lim, 0.],
                                                                 [self.f_lim, -self.miu_slider_pusher*self.f_lim, 0.]]),
                                           'sliding_left': np.array([[0., 0., 0.],
                                                                     [self.f_lim, self.miu_slider_pusher*self.f_lim, 0.],
                                                                     [0., 0., -self.dpsic_lim],
                                                                     [self.f_lim, self.miu_slider_pusher*self.f_lim, -self.dpsic_lim]]),
                                           'sliding_right': np.array([[0., 0., 0.],
                                                                      [self.f_lim, -self.miu_slider_pusher*self.f_lim, 0.],
                                                                      [0., 0., self.dpsic_lim],
                                                                      [self.f_lim, -self.miu_slider_pusher*self.f_lim, self.dpsic_lim]])}
        self.num_contact_face = len(self.contact_face_list)
        self.num_contact_mode = len(self.contact_mode_list)

        self.dynamics_mode_list = []
        for contact_face in self.contact_face_list:
            for contact_mode in self.contact_mode_list:
                self.dynamics_mode_list.append((contact_face, contact_mode))

        #  -------------------------------------------------------------------

        __x = cs.SX.sym('x')
        __y = cs.SX.sym('y')
        __theta = cs.SX.sym('theta')
        __psic = cs.SX.sym('psic')
        self.x = cs.veccat(__x, __y, __theta, __psic)
        self.dim_x = 4

        __x_bar = cs.SX.sym('x_bar')
        __y_bar = cs.SX.sym('y_bar')
        __theta_bar = cs.SX.sym('theta_bar')
        __psic_bar =cs.SX.sym('psic_bar')
        self.x_bar = cs.veccat(__x_bar, __y_bar, __theta_bar, __psic_bar)

        __fn = cs.SX.sym('fn')
        __ft = cs.SX.sym('ft')
        __dpsic = cs.SX.sym('dpsic')
        __f_ctact = cs.veccat(__fn, __ft)
        self.u = cs.veccat(__fn, __ft, __dpsic)
        self.dim_u = 3

        __fn_bar = cs.SX.sym('fn_bar')
        __ft_bar = cs.SX.sym('ft_bar')
        __dpsic_bar = cs.SX.sym('dpsic_bar')
        self.u_bar = cs.veccat(__fn_bar, __ft_bar, __dpsic_bar)

        __xl = cs.SX.sym('xl')
        __yl = cs.SX.sym('yl')
        __rl = cs.SX.sym('rl')
        self.beta = cs.veccat(__xl, __yl, __rl)
        
        __Area = __xl*__yl
        __int_Area = rect_cs(__xl, __yl)
        __c = __int_Area/__Area
        __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
        __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2)

        #  -------------------------------------------------------------------

        __norm = {'front': cs.DM([-1., 0.]),
                  'back':  cs.DM([1., 0.]),
                  'left':  cs.DM([0., 1.]),
                  'right': cs.DM([0., -1.])}
        __tangent = {'front': cs.DM([0., -1.]),
                     'back': cs.DM([0., 1.]),
                     'left': cs.DM([-1., 0.]),
                     'right': cs.DM([1., 0.])}

        __xc = cs.SX.sym('xc')
        __yc = cs.SX.sym('yc')
        __Jc = cs.SX(2,3)
        __Jc[0,0] = 1; __Jc[1,1] = 1; __Jc[0,2] = -__yc; __Jc[1,2] = __xc
        Jc = cs.Function('contact_jacobian', [__xc, __yc], [__Jc])

        __x_ctact = {'front':  0.5*__xl,
                     'back':  -0.5*__xl,
                     'left':   0.5*__yl/cs.tan(__psic),
                     'right': -0.5*__yl/cs.tan(__psic)}
        __y_ctact = {'front':  0.5*__xl*cs.tan(__psic),
                     'back':  -0.5*__xl*cs.tan(__psic),
                     'left':   0.5*__yl,
                     'right': -0.5*__yl}
        __Jc_ctact = {'front': Jc(__x_ctact['front'], __y_ctact['front']),
                      'back':  Jc(__x_ctact['back'], __y_ctact['back']),
                      'left':  Jc(__x_ctact['left'], __y_ctact['left']),
                      'right': Jc(__x_ctact['right'], __y_ctact['right'])}

        __ctheta = cs.cos(__theta)
        __stheta = cs.sin(__theta)
        __R = cs.SX(3, 3)
        __R[0,0] = __ctheta; __R[0,1] = -__stheta; __R[1,0] = __stheta; __R[1,1] = __ctheta; __R[2,2] = 1.0

        #  -------------------------------------------------------------------
        # auxiliary
        self.lsurf_A = cs.Function('A', [self.beta], [__A])
        self.RxA = cs.Function('RxA', [self.x, self.beta], [cs.mtimes(__R, __A)])
        # rechable set dynamics (dt = reachable_set_time_step)
        self.f_rcbset = {'front': self.x + self.reachable_set_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['front'].T, 
                            cs.vertcat(cs.mtimes(__norm['front'].T, __f_ctact), cs.mtimes(__tangent['front'].T, __f_ctact))))),
                            __dpsic),
                         'back':  self.x + self.reachable_set_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['back'].T,
                            cs.vertcat(cs.mtimes(__norm['back'].T, __f_ctact), cs.mtimes(__tangent['back'].T, __f_ctact))))),
                            __dpsic),
                         'left':  self.x + self.reachable_set_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['left'].T,
                            cs.vertcat(cs.mtimes(__norm['left'].T, __f_ctact), cs.mtimes(__tangent['left'].T, __f_ctact))))),
                            __dpsic),
                         'right': self.x + self.reachable_set_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['right'].T,
                            cs.vertcat(cs.mtimes(__norm['right'].T, __f_ctact), cs.mtimes(__tangent['right'].T, __f_ctact))))),
                            __dpsic)
        }

        ## for debug
        # self.jacobian_u_test = {'front': self.reachable_set_time_step * 
        #                     cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['front'].T, 
        #                     cs.vertcat(__norm['front'].T, __tangent['front'].T)))),
        #                  'back':  self.reachable_set_time_step * 
        #                     cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['back'].T,
        #                     cs.vertcat(__norm['back'].T, __tangent['back'].T)))),
        #                  'left':  self.reachable_set_time_step * 
        #                     cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['left'].T,
        #                     cs.vertcat(__norm['left'].T, __tangent['left'].T)))),
        #                  'right': self.reachable_set_time_step * 
        #                     cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['right'].T,
        #                     cs.vertcat(__norm['right'].T, __tangent['right'].T))))
        # }

        # R_func = cs.Function('R', [self.x], [__R])
        # A_func  =cs.Function('A', [self.beta], [__A])
        # Jc_ctact_func = cs.Function('Jc', [self.x], [cs.mtimes(__Jc_ctact['front'].T, cs.vertcat(__norm['front'].T, __tangent['front'].T))])
        # A_Jc_ctact_func = cs.Function('Jc', [self.x, self.beta], [cs.mtimes(__A, cs.mtimes(__Jc_ctact['front'].T, cs.vertcat(__norm['front'].T, __tangent['front'].T)))])

        # self.jacobian_u_test_func = {
        #                  'front': cs.Function('f_jacobi_u_front', [self.x, self.u, self.beta], [self.jacobian_u_test['front']]),
        #                  'back':  cs.Function('f_jacobi_u_back', [self.x, self.u, self.beta], [self.jacobian_u_test['back']]),
        #                  'left':  cs.Function('f_jacobi_u_left', [self.x, self.u, self.beta], [self.jacobian_u_test['left']]),
        #                  'right': cs.Function('f_jacobi_u_right', [self.x, self.u, self.beta], [self.jacobian_u_test['right']])
        # }
        # import pdb; pdb.set_trace()

        # differentiable function
        self.f_rcbset_func = {
                        'front': cs.Function('f_rcbset_front', [self.x, self.u, self.beta], [self.f_rcbset['front']]),
                        'back':  cs.Function('f_rcbset_back', [self.x, self.u, self.beta], [self.f_rcbset['back']]),
                        'left':  cs.Function('f_rcbset_left', [self.x, self.u, self.beta], [self.f_rcbset['left']]),
                        'right': cs.Function('f_rcbset_right', [self.x, self.u, self.beta], [self.f_rcbset['right']])
        }

        # non-linear dynamics (dt = nldynamics_time_step)
        self.f_nldyn = {'front': self.x + self.nldynamics_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['front'].T, 
                            cs.vertcat(cs.mtimes(__norm['front'].T, __f_ctact), cs.mtimes(__tangent['front'].T, __f_ctact))))),
                            __dpsic),
                         'back':  self.x + self.nldynamics_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['back'].T,
                            cs.vertcat(cs.mtimes(__norm['back'].T, __f_ctact), cs.mtimes(__tangent['back'].T, __f_ctact))))),
                            __dpsic),
                         'left':  self.x + self.nldynamics_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['left'].T,
                            cs.vertcat(cs.mtimes(__norm['left'].T, __f_ctact), cs.mtimes(__tangent['left'].T, __f_ctact))))),
                            __dpsic),
                         'right': self.x + self.nldynamics_time_step * cs.vertcat(
                            cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['right'].T,
                            cs.vertcat(cs.mtimes(__norm['right'].T, __f_ctact), cs.mtimes(__tangent['right'].T, __f_ctact))))),
                            __dpsic)
        }

        # differentiable function
        self.f_nldyn_func = {
                        'front': cs.Function('f_nldyn_front', [self.x, self.u, self.beta], [self.f_nldyn['front']]),
                        'back':  cs.Function('f_nldyn_back', [self.x, self.u, self.beta], [self.f_nldyn['back']]),
                        'left':  cs.Function('f_nldyn_left', [self.x, self.u, self.beta], [self.f_nldyn['left']]),
                        'right': cs.Function('f_nldyn_right', [self.x, self.u, self.beta], [self.f_nldyn['right']])
        }

        # symbolic for LQR
        self.xdot = {'front': cs.vertcat(cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['front'].T, 
                       cs.vertcat(cs.mtimes(__norm['front'].T, __f_ctact), cs.mtimes(__tangent['front'].T, __f_ctact))))),
                       __dpsic),
                     'back':  cs.vertcat(cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['back'].T,
                       cs.vertcat(cs.mtimes(__norm['back'].T, __f_ctact), cs.mtimes(__tangent['back'].T, __f_ctact))))),
                       __dpsic),
                     'left':  cs.vertcat(cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['left'].T,
                       cs.vertcat(cs.mtimes(__norm['left'].T, __f_ctact), cs.mtimes(__tangent['left'].T, __f_ctact))))),
                       __dpsic),
                     'right': cs.vertcat(cs.mtimes(__R, cs.mtimes(__A, cs.mtimes(__Jc_ctact['right'].T,
                       cs.vertcat(cs.mtimes(__norm['right'].T, __f_ctact), cs.mtimes(__tangent['right'].T, __f_ctact))))),
                       __dpsic)
        }

        # differentiable function
        self.xdot_func = {
                        'front': cs.Function('xdot_front', [self.x, self.u, self.beta], [self.xdot['front']]),
                        'back':  cs.Function('xdot_back', [self.x, self.u, self.beta], [self.xdot['back']]),
                        'left':  cs.Function('xdot_left', [self.x, self.u, self.beta], [self.xdot['left']]),
                        'right': cs.Function('xdot_right', [self.x, self.u, self.beta], [self.xdot['right']])
        }

        #  -------------------------------------------------------------------
        # matrices for AH-polytope construction
        self.A_mat_func = {}
        self.B_mat_func = {}
        self.c_mat_func = {}
        for contact_face in self.contact_face_list:
            A_mat_temp = cs.jacobian(self.f_rcbset[contact_face], self.x)
            B_mat_temp = cs.jacobian(self.f_rcbset[contact_face], self.u)
            A_mat_temp_func = cs.Function('A_mat_temp_func', [self.x, self.u, self.beta], [A_mat_temp])
            B_mat_temp_func = cs.Function('B_mat_temp_func', [self.x, self.u, self.beta], [B_mat_temp])

            c_mat_temp = self.f_rcbset_func[contact_face](self.x_bar, self.u_bar, self.beta) \
                            - cs.mtimes(A_mat_temp_func(self.x_bar, self.u_bar, self.beta), self.x_bar) \
                            - cs.mtimes(B_mat_temp_func(self.x_bar, self.u_bar, self.beta), self.u_bar)
            
            c_mat_temp_func = cs.Function('c_mat_temp_func', [self.x_bar, self.u_bar, self.beta], [c_mat_temp])

            self.A_mat_func[contact_face] = cs.Function('A_mat_func', [self.x_bar, self.u_bar, self.beta],
                                                        [A_mat_temp_func(self.x_bar, self.u_bar, self.beta)])
            self.B_mat_func[contact_face] = cs.Function('B_mat_func', [self.x_bar, self.u_bar, self.beta],
                                                        [B_mat_temp_func(self.x_bar, self.u_bar, self.beta)])
            self.c_mat_func[contact_face] = cs.Function('c_mat_func', [self.x_bar, self.u_bar, self.beta],
                                                        [c_mat_temp_func(self.x_bar, self.u_bar, self.beta)])

        self.E_mat = {'sticking': cs.DM([[-self.miu_slider_pusher, 1., 0.],
                                         [-self.miu_slider_pusher, -1., 0.],
                                         [1., 0., 0.],
                                         [-1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., -1., 0.],
                                         [0., 0., 1.],
                                         [0., 0., -1.]]),
                      'sliding_left': cs.DM([[-self.miu_slider_pusher, 1., 0.],
                                             [self.miu_slider_pusher, -1., 0.],
                                             [1., 0., 0.],
                                             [-1., 0., 0.],
                                             [0., 1., 0.],
                                             [0., -1., 0.],
                                             [0., 0., 1.],
                                             [0., 0., -1.]]),
                      'sliding_right': cs.DM([[self.miu_slider_pusher, 1., 0.],
                                              [-self.miu_slider_pusher, -1., 0.],
                                              [1., 0., 0.],
                                              [-1., 0., 0.],
                                              [0., 1., 0.],
                                              [0., -1., 0.],
                                              [0., 0., 1.],
                                              [0., 0., -1.]])
        }

        self.Xi_mat = {'sticking': cs.DM([[0.],
                                          [0.],
                                          [self.f_lim],
                                          [0.],
                                          [self.f_lim],
                                          [self.f_lim],
                                          [0.],
                                          [0.]]),
                       'sliding_left': cs.DM([[0.],
                                              [0.],
                                              [self.f_lim],
                                              [0.],
                                              [self.f_lim],
                                              [0.],
                                              [0.],
                                              [self.dpsic_lim]]),
                       'sliding_right': cs.DM([[0.],
                                               [0.],
                                               [self.f_lim],
                                               [0.],
                                               [0.],
                                               [self.f_lim],
                                               [self.dpsic_lim],
                                               [0.]])
        }

        #  -------------------------------------------------------------------
        # matrices for LQR
        self.A_mat_LQR_func = {}
        self.B_mat_LQR_func = {}
        for contact_face in self.contact_face_list:
            A_mat_LQR_temp = cs.jacobian(self.xdot[contact_face], self.x)
            B_mat_LQR_temp = cs.jacobian(self.xdot[contact_face], self.u)
            self.A_mat_LQR_func[contact_face] = cs.Function('A_mat_LQR_func', [self.x, self.u, self.beta], [A_mat_LQR_temp])
            self.B_mat_LQR_func[contact_face] = cs.Function('B_mat_LQR_func', [self.x, self.u, self.beta], [B_mat_LQR_temp])

    def forward_step(self, u=None, linearize=False, modify_system=None, step_size=1e-2, return_as_env=None,
                        return_mode=None, starting_state=None, mode_string=None):
        """
        Doing forward simulation of the pusher-slider system
        :param u: input (fn, ft, dpsic)
        :param linearize: if true, use linearized dynamics (not implemented); if false, use nonlinear dynamics
        :param step_size: time step of the dynamics simulation, equals to nldynamics_time_step
        :param starting state: current(initial) state x0, with psic
        :param mode_string: (contact_face, contact_mode)
        :return: next_state, the state after 1 temp step
        """
        try:
            assert (starting_state is not None) and (mode_string is not None)
        except:
            raise AssertionError('PushDTHybridSystem: The starting_state and mode_string are not provided!')

        try:
            assert step_size == self.nldynamics_time_step
        except:
            raise AssertionError('PushDTHybridSystem: The step_size:{0} is not equal to nldynamics_time_step:{1}!'.format(step_size, self.nldynamics_time_step))

        if u is not None:
            nominal_input = u.copy()
        else:
            nominal_input = np.zeros(self.dim_u)

        if linearize:
            raise NotImplementedError('PushDTHybridSystem: The linearized forward dynamics is not implemented!')

        contact_face, contact_mode = mode_string
        # try:
        #     E_mat = self.E_mat[contact_mode].toarray()
        #     Xi_mat = self.Xi_mat[contact_mode].toarray()
        #     assert (np.all(np.matmul(E_mat, nominal_input).reshape(-1,) <= Xi_mat.reshape(-1,)))
        # except:
        #     raise AssertionError('PushDTHybridSystem: The input:{0} is not compatible with the contact_mode:{1}!'.format(nominal_input, contact_mode))

        current_contact_face = self._get_contact_face_from_state(starting_state)

        # try:
        #     assert current_contact_face == contact_face
        # except:
        #     raise AssertionError('PushDTHybridSystem: The contact_face provided:{0} is not compatible with state:{1}, current_contact_face:{2}!'.format(contact_face, starting_state, current_contact_face))
        if current_contact_face != contact_face:
            current_psic = self.psic_each_face_center[contact_face]
        else:
            # get psic in [-pi, pi]
            current_psic = restrict_angle_in_unit_circle(starting_state[-1])

        # update the x_bar
        nominal_state = starting_state.copy()
        nominal_state[-1] = current_psic

        dynamics = self.f_nldyn_func[contact_face]
        next_state = dynamics(nominal_state, nominal_input, self.slider_geom).toarray()

        return next_state

    def get_reachable_polytopes(self, state, u=None, step_size=5e-2, use_convex_hull=False):
        """
        Get reachable polytopes of the current state
        :param state: x_bar, linearize point, with psic, psic could be any value in (-inf, inf)
        :param u: u_bar, linearize point
        :param step_size: step size of the reachable set, equals to reachable_set_time_step
        :param use_convex_hull: if true, returns conv(state, reachable_polytope)
                                if false, returns reachable_polytope only
        :return: polytopes_list, in ndarray form, list of reachable polytopes
        """
        try:
            assert step_size == self.reachable_set_time_step
        except:
            raise AssertionError('PushDTHybridSystem: The step_size:{0} is not equal to reachable_set_time_step:{1}!'.format(step_size, self.reachable_set_time_step))
        polytopes_list = []

        # get psic in [-pi, pi]
        current_psic = restrict_angle_in_unit_circle(state[-1])

        # get the contact face
        current_contact_face = self._get_contact_face_from_state(state)

        # get the unilateral sliding flag (True/False), the prohibited sliding mode
        unilateral_sliding_flag, prohibited_sliding_mode = self._get_unilateral_sliding_flag_from_state(state, current_contact_face)
        
        # get the AH_polytope expression of all legal reachable sets
        for mode_index, mode_string in enumerate(self.dynamics_mode_list):
            contact_face, contact_mode = mode_string
            
            # skip the prohibited sliding mode
            if unilateral_sliding_flag and (contact_mode == prohibited_sliding_mode):
                continue

            # 1. psic should be continuous when contact face remains unchanged
            # 2. set psic to the center after the contact face changes
            # 3. update the u_bar
            if contact_face == current_contact_face:
                start_psic = current_psic
                # nominal_input = u.copy() if u is not None else np.zeros(self.dim_u)
                nominal_input = np.zeros(self.dim_u)
            else:
                start_psic = self.psic_each_face_center[contact_face]
                nominal_input = np.zeros(self.dim_u)

            # update the x_bar
            nominal_state = state.copy()
            nominal_state[-1] = start_psic

            # get the matrices (exclude the last row of psic, which is unnecessary when computing reachable set)
            A_mat = self.A_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom)[:-1, :]
            B_mat = self.B_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom)[:-1, :]
            c_mat = self.c_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom)[:-1, :]

            E_mat = self.E_mat[contact_mode]
            Xi_mat = self.Xi_mat[contact_mode]

            # construct the input H-polytope
            H_polytope_input = H_polytope(H=E_mat.toarray(), h=Xi_mat.toarray())
            # construct the reachable set as AH_polytopes
            AH_polytope_state = AH_polytope(T=B_mat,
                                            t=(cs.mtimes(A_mat, nominal_state) + c_mat).toarray(),
                                            P=H_polytope_input,
                                            mode_string=(contact_face, contact_mode),
                                            applied_u=nominal_input)

            # print('A mat: ', A_mat)
            print('B_mat: ', B_mat)
            # print('jacobi_u: ', self.jacobian_u_test_func[contact_face](nominal_state, nominal_input, self.slider_geom))
            # print('C_mat: ', c_mat)
            # print('F(x_bar, u_bar): ', self.f_rcbset_func[contact_face](nominal_state, nominal_input, self.slider_geom))
            # print('B_mat * u_bar: ', self.B_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom) @ nominal_input)
            # print('F(x_bar, u_bar) - B_mat * u_bar: ', self.f_rcbset_func[contact_face](nominal_state, nominal_input, self.slider_geom) - self.B_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom) @ nominal_input)
            # print('linearize point: x={0}, u={1}'.format(nominal_state, nominal_input))
            print('AH.t={0}'.format((cs.mtimes(A_mat, nominal_state) + c_mat).toarray()))

            if use_convex_hull:
                polytopes_list.append(convex_hull_of_point_and_polytope(x=state.reshape(-1, 1)[:-1, :], Q=AH_polytope_state))
                # polytopes_list.append(AH_polytope_state)
            else:
                polytopes_list.append(AH_polytope_state)

        # FIXME: no need to align the polytopes in memory, with a dtype=AH_polytope argument?
        return np.asarray(polytopes_list)

    def get_reachable_polytopes_with_variable_psic(self, state, u=None, step_size=5e-2, use_convex_hull=False):
        """
        Get reachable polytopes of the current state
        Return the convex hull of a bunch of AH_polytopes, with variable psic in [psic_min, psic_max]
        :param state: x_bar, linearize point, with psic, psic could be any value in (-inf, inf)
        :param u: u_bar, linearize point
        :param step_size: step size of the reachable set, equals to reachable_set_time_step
        :param use_convex_hull: if true, returns conv(state, reachable_polytope)
                                if false, returns reachable_polytope only
        :return: polytopes_list, in ndarray form, list of reachable polytopes
        """
        try:
            assert step_size == self.reachable_set_time_step
        except:
            raise AssertionError('PushDTHybridSystem: The step_size:{0} is not equal to reachable_set_time_step:{1}!'.format(step_size, self.reachable_set_time_step))
        polytopes_list = []

        # get psic in [-pi, pi]
        current_psic = restrict_angle_in_unit_circle(state[-1])

        # get the contact face
        current_contact_face = self._get_contact_face_from_state(state)

        # get the unilateral sliding flag (True/False), the prohibited sliding mode
        unilateral_sliding_flag, prohibited_sliding_mode = self._get_unilateral_sliding_flag_from_state(state, current_contact_face)
        
        # get the AH_polytope expression of all legal reachable sets
        for mode_index, mode_string in enumerate(self.dynamics_mode_list):
            contact_face, contact_mode = mode_string
            
            # skip the prohibited sliding mode
            if unilateral_sliding_flag and (contact_mode == prohibited_sliding_mode):
                continue

            # 1. psic should be continuous when contact face remains unchanged
            # 2. set psic to the center after the contact face changes
            # 3. update the u_bar
            if contact_face == current_contact_face:
                start_psic = current_psic
            else:
                start_psic = self.psic_each_face_center[contact_face]
            nominal_input = np.zeros(self.dim_u)

            # get the range of psic
            psic_min, psic_max = self._get_psic_range(start_psic, contact_face)

            E_mat_psic = self.E_mat[contact_mode]
            Xi_mat_psic = self.Xi_mat[contact_mode]

            # construct the input H-polytope
            H_polytope_input = H_polytope(H=E_mat_psic.toarray(), h=Xi_mat_psic.toarray())
            polytope_keypoints = self.u_domain_polytope_keypoint[contact_mode]

            # PSIC_MIN
            # update the x_bar
            nominal_state_psic_min = state.copy()
            nominal_state_psic_min[-1] = psic_min

            # get the matrices (exclude the last row of psic, which is unnecessary when computing reachable set)
            A_mat_psic_min = self.A_mat_func[contact_face](nominal_state_psic_min, nominal_input, self.slider_geom)[:-1, :]
            B_mat_psic_min = self.B_mat_func[contact_face](nominal_state_psic_min, nominal_input, self.slider_geom)[:-1, :]
            c_mat_psic_min = self.c_mat_func[contact_face](nominal_state_psic_min, nominal_input, self.slider_geom)[:-1, :]

            # construct the reachable set as AH_polytopes
            AH_polytope_state_psic_min = AH_polytope(T=B_mat_psic_min,
                                                     t=(cs.mtimes(A_mat_psic_min, nominal_state_psic_min) + c_mat_psic_min).toarray(),
                                                     P=H_polytope_input,
                                                     mode_string=(contact_face, contact_mode),
                                                     mode_consistent=(contact_face==current_contact_face),
                                                     applied_u=nominal_input,
                                                     psic_range=(psic_min, psic_max))
            ah_polytope_keypoints = AH_polytope_state_psic_min.t.reshape(-1) + np.matmul(AH_polytope_state_psic_min.T, polytope_keypoints.T)
            AH_polytope_state_psic_min.key_vertex.update(tuple(map(tuple, ah_polytope_keypoints.toarray().T)))

            # PSIC_MAX
            # update the x_bar
            nominal_state_psic_max = state.copy()
            nominal_state_psic_max[-1] = psic_max

            # get the matrices (exclude the last row of psic, which is unnecessary when computing reachable set)
            A_mat_psic_max = self.A_mat_func[contact_face](nominal_state_psic_max, nominal_input, self.slider_geom)[:-1, :]
            B_mat_psic_max = self.B_mat_func[contact_face](nominal_state_psic_max, nominal_input, self.slider_geom)[:-1, :]
            c_mat_psic_max = self.c_mat_func[contact_face](nominal_state_psic_max, nominal_input, self.slider_geom)[:-1, :]

            # construct the reachable set as AH_polytopes
            AH_polytope_state_psic_max = AH_polytope(T=B_mat_psic_max,
                                                     t=(cs.mtimes(A_mat_psic_max, nominal_state_psic_max) + c_mat_psic_max).toarray(),
                                                     P=H_polytope_input,
                                                     mode_string=(contact_face, contact_mode),
                                                     mode_consistent=(contact_face==current_contact_face),
                                                     applied_u=nominal_input,
                                                     psic_range=(psic_min, psic_max))

            ah_polytope_keypoints = AH_polytope_state_psic_max.t.reshape(-1) + np.matmul(AH_polytope_state_psic_max.T, polytope_keypoints.T)
            AH_polytope_state_psic_max.key_vertex.update(tuple(map(tuple, ah_polytope_keypoints.toarray().T)))

            AH_polytope_state = convex_hull_of_ah_polytopes(P1=AH_polytope_state_psic_min, P2=AH_polytope_state_psic_max)
            
            if use_convex_hull:
                polytopes_list.append(convex_hull_of_point_and_polytope(x=state.reshape(-1, 1)[:-1, :], Q=AH_polytope_state))
                # polytopes_list.append(AH_polytope_state)
            else:
                polytopes_list.append(AH_polytope_state)

        # FIXME: no need to align the polytopes in memory, with a dtype=AH_polytope argument?
        return np.asarray(polytopes_list)

    def get_linearization(self, state=None, u_bar=None, mode=None):
        """
        Return the DiscreteLinearDynamics of current state
        :param state: x_bar, with psic, current state, linearize point
        :param u_bar: u_bar, linearize point
        :param mode: (contact_face, contact_mode)
        :return: DiscreteLinearDynamics
        """
        try:
            assert (state is not None) and (mode is not None)
        except:
            raise AssertionError('PushDTHybridSystem: The state and mode are not provided!')

        # update the u_bar
        if u_bar is not None:
            # nominal_input = u_bar.copy()
            nominal_input = np.zeros(self.dim_u)
        else:
            nominal_input = np.zeros(self.dim_u)

        # get psic in [-pi, pi]
        current_psic = restrict_angle_in_unit_circle(state[-1])

        # get the contact face
        contact_face, contact_mode = mode
        current_contact_face = self._get_contact_face_from_state(state)

        # try:
        #     assert current_contact_face == contact_face
        # except:
        #     raise AssertionError('PushDTHybridSystem: The contact_face provided:{0} is not compatible with state:{1}, current_contact_face:{2}!'.format(contact_face, state, current_contact_face))
        if current_contact_face != contact_face:
            current_psic = self.psic_each_face_center[contact_face]
        
        nominal_state = state.copy()
        nominal_state[-1] = current_psic

        # exclude the last row of psic, which is unnecessary when computing reachable set
        return DiscreteLinearDynamics(A=self.A_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom)[:-1, :].toarray(),
                                      B=self.B_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom)[:-1, :].toarray(),
                                      c=self.c_mat_func[contact_face](nominal_state, nominal_input, self.slider_geom)[:-1, :].toarray(),
                                      E=self.E_mat[contact_mode].toarray(),
                                      Xi=self.Xi_mat[contact_mode].toarray(),
                                      x_bar=nominal_state,
                                      u_bar=nominal_input)

    def calculate_input(self, goal_state, nominal_x, nominal_u, mode_string):
        """
        Calculate approximated input
        :param goal_state: the goal state, without psic
        :param nominal_x: the nominal state, with psic, linearize point
        :param nominal_t: the nominal input, with psic, linearize point
        :param mode_string: (contact_face, contact_mode)
        :return: approximate_input
        """
        # calculate desired dpsic
        goal_psic = self._calculate_desired_psic_from_state_transition(from_state=nominal_x[:-1],
                                                                       to_state=goal_state,
                                                                       contact_face=mode_string[0])
        current_psic = nominal_x[-1]
        dpsic = angle_diff(current_psic, goal_psic) / self.reachable_set_time_step

        # calculate desired force
        linear_system = self.get_linearization(state=np.append(nominal_x[:-1], (current_psic+goal_psic)/2.),
                                               u_bar=nominal_u,
                                               mode=mode_string)

        A_mat = linear_system.A.copy()
        B_mat_pinv = np.linalg.pinv(linear_system.B[:, :-1])
        c_mat = linear_system.c.copy()
        nominal_state = linear_system.x_bar.copy()
        # nominal_state = nominal_x.copy().flatten()
        approximate_input = np.zeros(self.dim_u)
        approximate_input[:-1] = np.matmul(B_mat_pinv, goal_state.flatten() - np.matmul(A_mat, nominal_state) - c_mat.flatten())
        approximate_input[-1] = dpsic
        approximate_input = approximate_input.flatten()[:self.dim_u]

        return approximate_input

    def solve_optimal_control(self, start_state, end_state, contact_face, optimal_time=None):
        """
        # FIXME: it sometimes works, but not very good; the model causes A=zero matrix
        Solve the fixed final state, fixed final time optimal control problem
        (reference: https://ieeexplore.ieee.org/document/6631299)
        :param start_state: x[0], with psic
        :param end_state: x[tau], with psic
        :param contact_face: the contact face
        :param optimal_time: tau
        :return: u, function(input: 0<=t<=tau, output: input u)
        """
        # import pdb; pdb.set_trace()
        # time
        tau = self.reachable_set_time_step if optimal_time is not None else optimal_time

        # auxiliary matrix
        nominal_state = start_state.copy()
        nominal_input = np.zeros(self.dim_u)
        # nominal_input = self.calculate_input(goal_state=end_state[:-1], nominal_x=nominal_state, nominal_u=np.zeros(self.dim_u), mode_string=('left', 'sticking'))
        A = self.A_mat_LQR_func[contact_face](nominal_state, nominal_input, self.slider_geom).toarray()
        B = self.B_mat_LQR_func[contact_face](nominal_state, nominal_input, self.slider_geom).toarray()
        R = np.diag(self.quad_cost_input)
        R_inv = np.linalg.pinv(R)
        M = matrix_mult([B,R_inv,B.T])

        # controlibility Gramian
        # G = (M) * (tau) + \
        #     (matrix_mult(A,M) + matrix_mult(M,A.T)) * ((1/2) * tau ** 2) + \
        #     ((1/2)*(matrix_mult(A,A,M)+matrix_mult(M,A.T,A.T)) + matrix_mult(A,M,A.T)) * ((1/3) * tau ** 3) + \
        #     ((1/6)*(matrix_mult(A,A,A,M)+matrix_mult(M,A.T,A.T,A.T)) + (1/2)*(matrix_mult(A,A,M,A.T)+matrix_mult(A,M,A.T,A.T))) * ((1/4) * tau ** 4) + \
        #     ((1/6)*(matrix_mult(A,M,A.T,A.T,A.T)+matrix_mult(A,A,A,M,A.T)) + (1/4)*matrix_mult(A,A,M,A.T,A.T)) * ((1/5) * tau ** 5) + \
        #     ((1/12)*(matrix_mult(A,A,M,A.T,A.T,A.T)+matrix_mult(A,A,A,M,A.T,A.T))) * ((1/6) * tau ** 6) + \
        #     ((1/36)*matrix_mult(A,A,A,M,A.T,A.T,A.T)) * ((1/7) * tau ** 7)

        G = (M) * (tau) + \
            (matrix_mult([A,M]) + matrix_mult([M,A.T])) * ((1/2) * tau ** 2) + \
            ((1/2)*(matrix_mult([A,A,M])+matrix_mult([M,A.T,A.T])) + matrix_mult([A,M,A.T])) * ((1/3) * tau ** 3) + \
            ((1/2)*(matrix_mult([A,A,M,A.T])+matrix_mult([A,M,A.T,A.T]))) * ((1/4) * tau ** 4) + \
            ((1/4)*(matrix_mult([A,A,M,A.T,A.T]))) * ((1/5) * tau ** 5)

        # xbar
        # xbar = np.matmul(np.eye(self.dim_x) + A*tau + (1/2)*matrix_mult([A,A])*tau**2, start_state)
        xbar = start_state

        # input
        def get_input(t):
            return matrix_mult([R_inv,
                                B.T,
                                (np.eye(self.dim_x) + A.T*(tau-t) + (1/2)*matrix_mult([A.T,A.T])*(tau-t)**2),
                                np.linalg.pinv(G),
                                (end_state-xbar)])

        return get_input

    def solve_discrete_lqr(self, actual_state_x, desired_state_xf, dt, contact_face, Q=None, R=None, A=None, B=None):
        """
        Solve the discrete-time LQR for a nonlinear system.
        - Compute the optimal control inputs given a nonlinear system, cost matrices, 
        current state, and a final state.
        - Compute the control variables that minimize the cumulative cost.
        - Solve for P using the dynamic programming method.
        (reference: https://automaticaddison.com/linear-quadratic-regulator-lqr-with-python-code-example/)
        :param actual_state_x: The current state of the system, with psic
            3x1 NumPy Array given the state is [x,y,yaw angle] --->
            [meters, meters, radians]
        :param desired_state_xf: The desired state of the system, with psic
            3x1 NumPy Array given the state is [x,y,yaw angle] --->
            [meters, meters, radians]   
        :param Q: The state cost matrix
            3x3 NumPy Array
        :param R: The input cost matrix
            2x2 NumPy Array
        :param dt: The size of the timestep in seconds -> float
    
        :return: u_star: Optimal action u for the current state 
            2x1 NumPy Array given the control input vector is
            [linear velocity of the car, angular velocity of the car]
            [meters per second, radians per second]
        """
        if Q is None:
            Q = np.append(np.ones(self.dim_x-1), 0.)
        if R is None:
            R = np.diag(self.quad_cost_input)
        if A is None:
            A = np.eye(self.dim_x)
        if B is None:
            B = dt*self.B_mat_LQR_func[contact_face](actual_state_x, np.zeros(self.dim_u), self.slider_geom).toarray()

        # We want the system to stabilize at desired_state_xf.
        x_error = actual_state_x - desired_state_xf
    
        # Solutions to discrete LQR problems are obtained using the dynamic programming method.
        # The optimal solution is obtained recursively, starting at the last timestep and working backwards.
        # You can play with this number
        N = 50
        # Create a list of N + 1 elements
        P = [None] * (N + 1)
        Qf = Q
        # LQR via Dynamic Programming
        P[N] = Qf
    
        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati equation to calculate the optimal state cost matrix
            P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
    
        # Create a list of N elements
        K = [None] * N
        u = [None] * N
    
        # For i = 0, ..., N - 1
        for i in range(N):
            # Calculate the optimal feedback gain K
            K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
            u[i] = K[i] @ x_error
    
        # Optimal control input is u_star
        u_star = u[N-1]

        return u_star

    def collision_free_forward_simulation(self, starting_state, goal_state, u, max_steps, mode_string, Z_obs_list:MultiPolygon):
        """
        Forward simulation with collision check
        :param starting_state: the starting state, with psic
        :param goal_state: the goal state, without psic
        :param u: the input
        :param max_steps: the max simulation steps
        :param mode_string: (contact_face, contact_mode)
        :param Z_obs_list: the MultiPolygon object of all obstacles
        :return: flag (collision or not), final state, state list
        """
        state = starting_state.copy()
        state_list = [state]
        
        for step in range(max_steps):
            # simulate one step
            state = self.forward_step(u=u,
                                      linearize=False,
                                      step_size=self.nldynamics_time_step,
                                      starting_state=state,
                                      mode_string=mode_string)
            # collision check
            if Z_obs_list is not None:
                collision_flag = intersects(Z_obs_list, gen_polygon(state[:-1], self.slider_geom[:2]))
                if collision_flag:
                    # FIXME: closest point is unreachable, should return parent_state, state or closest_point?
                    return False, None, None

            state_list.append(state.reshape(-1,))
        
        state_array_without_psic = np.array(state_list)[:, :-1]
        # nearest_state_index = np.argmin(np.linalg.norm(state_array_without_psic - goal_state.reshape(-1,), axis=1))
        nearest_state_index = len(state_list)-1

        return True, state_list[nearest_state_index], state_list[:nearest_state_index+1]

    def get_pusher_location(self, state, contact_face):
        """
        Get the pusher centroid
        :param state: the state, including psic
        :param contact_face: the contact face
        """
        psic = state[3]
        xl, yl, rl = self.slider_geom
        # compute contact location and bias
        if contact_face == 'front':
            x = xl/2
            y = x*np.tan(psic)
            bias = np.array([rl, 0.])
        elif contact_face == 'back':
            x = -xl/2
            y = x*np.tan(psic)
            bias = np.array([-rl, 0.])
        elif contact_face == 'left':
            y = yl/2
            x = y/np.tan(psic)
            bias = np.array([0., rl])
        elif contact_face == 'right':
            y = -yl/2
            x = y/np.tan(psic)
            bias = np.array([0., -rl])
        else:
            raise NotImplementedError('PushDTHybridSystem: the contact face {0} is not supported!'.format(contact_face))
        
        # convert to world frame
        pusher_centroid = state[:2] + np.matmul(rotation_matrix(state[2]), np.array([x, y])+bias)
        return pusher_centroid
    
    def get_contact_location(self, state, contact_face):
        """
        Get the pusher contact point (on the slider's boundary)
        :param state: the state, including psic
        :param contact_face: the contact face
        """
        psic = state[3]
        xl, yl, rl = self.slider_geom
        # compute contact location and bias
        if contact_face == 'front':
            x = xl/2
            y = x*np.tan(psic)
        elif contact_face == 'back':
            x = -xl/2
            y = x*np.tan(psic)
        elif contact_face == 'left':
            y = yl/2
            x = y/np.tan(psic)
        elif contact_face == 'right':
            y = -yl/2
            x = y/np.tan(psic)
        else:
            raise NotImplementedError('PushDTHybridSystem: the contact face {0} is not supported!'.format(contact_face))
        
        # convert to world frame
        pusher_centroid = state[:2] + np.matmul(rotation_matrix(state[2]), np.array([x, y]))
        return pusher_centroid

    def _get_contact_face_from_state(self, state):
        """
        Get the contact face from state
        :param state: x_bar, the state, including psic
        :return: string of the contact face
        """
        # restrict angle in range [-pi, pi]
        psic = restrict_angle_in_unit_circle(state[-1])
        xl, yl = self.slider_geom[0], self.slider_geom[1]

        # pre-compute psic of four corners
        psic_front_left_corner = np.arctan2(0.5*yl, 0.5*xl)
        psic_front_right_corner = np.arctan2(-0.5*yl, 0.5*xl)
        psic_back_left_corner = np.arctan2(0.5*yl, -0.5*xl)
        psic_back_right_corner = np.arctan2(-0.5*yl, -0.5*xl)

        # detect the contact face
        if psic_front_right_corner <= psic <= psic_front_left_corner:
            contact_face = 'front'
        elif psic_front_left_corner <= psic <= psic_back_left_corner:
            contact_face = 'left'
        elif psic_back_right_corner <= psic <= psic_front_right_corner:
            contact_face = 'right'
        else:
            contact_face = 'back'

        return contact_face

    def _get_unilateral_sliding_flag_from_state(self, state, contact_face):
        """
        If the pusher is too near the corner, sliding towards the corner is prohibited
        :param state: x_bar, including psic, the contact location
        :param contact_face: the contact face
        :return: unilateral_sliding_flag, true if sliding towards the corner is prohibited, false otherwise
        :return: prohibited_sliding_mode, string of the prohibited contact mode
        """
        # restrict angle in range [-pi, pi]
        psic = restrict_angle_in_unit_circle(state[-1])
        xl, yl = self.slider_geom[0], self.slider_geom[1]
        x_unilateral_sliding_region_border = 0.5*xl - self.unilateral_sliding_region
        y_unilateral_sliding_region_border = 0.5*yl - self.unilateral_sliding_region

        try:
            assert (x_unilateral_sliding_region_border > 0) \
                    and (y_unilateral_sliding_region_border > 0)
        except:
            raise AssertionError('PushDTHybridSystem: The bilateral sliding region width {0} and {1} is not enough!'.format(x_unilateral_sliding_region_border,
                                                                                                        y_unilateral_sliding_region_border))

        # detect if angle falls into unilateral sliding region
        unilateral_sliding_flag = False
        prohibited_sliding_mode = None
        if contact_face == 'front':
            y = 0.5*xl*np.tan(psic)
            if y > y_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_right'
            elif y < -y_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_left'
        elif contact_face == 'back':
            y = -0.5*xl*np.tan(psic)
            if y > y_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_left'
            elif y < -y_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_right'
        elif contact_face == 'left':
            x = 0.5*yl/np.tan(psic)
            if x > x_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_left'
            elif x < -x_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_right'
        elif contact_face == 'right':
            x = -0.5*yl/np.tan(psic)
            if x > x_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_right'
            elif x < -x_unilateral_sliding_region_border:
                unilateral_sliding_flag = True
                prohibited_sliding_mode = 'sliding_left'
        else:
            raise ValueError('PushDTHybridSystem: The contact face {0} provided is illegal!'.format(contact_face))

        return unilateral_sliding_flag, prohibited_sliding_mode

    def _state_to_env(self, state, u=None):
        raise NotImplementedError('PushDTHybridSystem: env is not used, thus _state_to_env is not implemented!')

    def _extract_variable_value_from_env(self, symbolic_var, env):
        raise NotImplementedError('PushDTHybridSystem: env is not used, thus _extract_variable_value_from_env is not implemented!')

    def _get_psic_range(self, psic0, contact_face):
        """
        Get the range of psic in the next time step, with possible dpsic in [-dpsic_lim, dpsic_lim]
        :param psic0: initial psic
        :param contact_face: the contact face
        :return: psic_min, the bound when sliding left
        :return: psic_max, the bound when sliding right, it's unecessary that psic_min < psic_max
        """
        xl, yl = self.slider_geom[:2]
        psic_max = restrict_angle_in_unit_circle(psic0 + self.dpsic_lim * self.reachable_set_time_step)
        psic_min = restrict_angle_in_unit_circle(psic0 - self.dpsic_lim * self.reachable_set_time_step)
        if contact_face == 'front':
            x = 0.5 * xl
            y_left = max(-0.5*yl+self.unilateral_sliding_region, x*np.tan(psic_min))
            y_right = min(0.5*yl-self.unilateral_sliding_region, x*np.tan(psic_max))
        elif contact_face == 'back':
            x = -0.5 * xl
            y_left = min(0.5*yl-self.unilateral_sliding_region, x*np.tan(psic_min))
            y_right = max(-0.5*yl+self.unilateral_sliding_region, x*np.tan(psic_max))
        elif contact_face == 'left':
            y = 0.5 * yl
            x_left = min(0.5*xl-self.unilateral_sliding_region, y/np.tan(psic_min))
            x_right = max(-0.5*xl+self.unilateral_sliding_region, y/np.tan(psic_max))
        elif contact_face == 'right':
            y = -0.5 * yl
            x_left = max(-0.5*xl+self.unilateral_sliding_region, y/np.tan(psic_min))
            x_right = min(0.5*xl-self.unilateral_sliding_region, y/np.tan(psic_max))
        else:
            raise ValueError('PushDTHybridSystem: The contact face {0} provided is illegal!'.format(contact_face))

        if contact_face == 'front' or contact_face == 'back':
            psic_min = np.arctan2(y_left, x)
            psic_max = np.arctan2(y_right, x)
        elif contact_face == 'left' or contact_face == 'right':
            psic_min = np.arctan2(y, x_left)
            psic_max = np.arctan2(y, x_right)
        else:
            raise ValueError('PushDTHybridSystem: The contact face {0} provided is illegal!'.format(contact_face))

        return psic_min, psic_max

    def _calculate_desired_psic_from_state_transition(self, from_state, to_state, contact_face):
        """
        Calculate the psic needed to transit from from_state to to_state, for the given contact face
        :param from_state: from state, without psic
        :param to_state: to state, without psic
        :param contact_face: contact face
        :return: psic in [-pi, pi]
        """
        xl, yl = self.slider_geom[:2]
        delta_state = to_state - from_state
        RxA = self.RxA(np.append(from_state, 0.), self.slider_geom).toarray()  # psic does not matters, set to zero
        pseudo_f_ctact = np.matmul(np.linalg.inv(RxA[:2, :2]), delta_state[:2])
        fn, ft = pseudo_f_ctact[0], pseudo_f_ctact[1]
        if contact_face == 'front':
            xc = 0.5*xl
            yc = ((delta_state[2]/RxA[2,2])-ft*xc)/(-fn)
        elif contact_face == 'back':
            xc = -0.5*xl
            yc = ((delta_state[2]/RxA[2,2])-ft*xc)/(-fn)
        elif contact_face == 'left':
            yc = 0.5*yl
            xc = ((delta_state[2]/RxA[2,2])+fn*yc)/(ft)
        elif contact_face == 'right':
            yc = -0.5*yl
            xc = ((delta_state[2]/RxA[2,2])+fn*yc)/(ft)
        psic = restrict_angle_in_unit_circle(np.arctan2(yc, xc))
        return psic

    def get_current_state(self):
        raise NotImplementedError('PushDTHybridSystem: env is not used, thus get_current_state is not implemented!')


if __name__ == '__main__':
    ## import libraries for running test
    ## ----------------------------------------------------
    import time
    from matplotlib import pyplot as plt
    from r3t.symbolic_system.symbolic_system_r3t import PolytopeReachableSet
    from pypolycontain.visualization.visualize_2D import visualize_3D_AH_polytope_push_planning
    ## ----------------------------------------------------


    dyn = PushDTHybridSystem(quad_cost_input=[0.001, 0.001, 5e-6])  # test with default value
    
    distance_scaling_array = np.array([1.0, 1.0, 0.0695])
    # polytope_parent_state = np.array([0.345, 0.165, -0.99*np.pi, 0.9*np.pi])
    polytope_parent_state = np.array([0.0, 0.0, 0.0, 0.0])
    # polytope_parent_state2 = np.array([0.28, 0.24, -0.6*np.pi, -0.9*np.pi])
    polytope_parent_state2 = np.array([0.32, 0.20, -2.5, -0.9*np.pi])


    ## TEST-1 (PushDTHybridSystem._get_contact_face_from_state)
    ## ----------------------------------------------------
    # start_time = time.time()
    # print(dyn._get_contact_face_from_state(np.array([2.3419, 0.8627, 0., 0.25*np.pi])))  # front
    # print(dyn._get_contact_face_from_state(np.array([1.3419, -0.8627, 0.25*np.pi, 0.59*np.pi])))  # left
    # print(dyn._get_contact_face_from_state(np.array([0.3419, -1.8627, 0.5*np.pi, -0.25*np.pi])))  # front
    # print(dyn._get_contact_face_from_state(np.array([-0.3419, -2.8627, 0.75*np.pi, -0.59*np.pi])))  # right
    # print(dyn._get_contact_face_from_state(np.array([-1.3419, -3.8627, 1.00*np.pi, 0.98*np.pi])))  # back
    # print(dyn._get_contact_face_from_state(np.array([-2.3419, -4.8627, 1.25*np.pi, -0.98*np.pi])))  # back
    # print(dyn._get_contact_face_from_state(np.array([-2.3419, -4.8627, 1.5*np.pi, 1.22*np.pi])))  # back
    # print(dyn._get_contact_face_from_state(np.array([-2.3419, -4.8627, 1.75*np.pi, -1.22*np.pi])))  # back
    # print(dyn._get_contact_face_from_state(np.array([-2.3419, -4.8627, 2.00*np.pi, 2.59*np.pi])))  # left
    # print(dyn._get_contact_face_from_state(np.array([-2.3419, -4.8627, 2.25*np.pi, -2.59*np.pi])))  # right
    # print('test get_contact_face_from_state, time_cost={0}'.format(time.time() - start_time))
    ## ----------------------------------------------------


    ## TEST-2 (PushDTHybridSystem._get_unilateral_sliding_flag_from_state)
    ## ----------------------------------------------------
    # start_time = time.time()
    # print(dyn._get_unilateral_sliding_flag_from_state(np.array([2.3419, 0.8627, 0., 1.0201]), contact_face='front'))  # (True, 'sliding_right')
    # print(dyn._get_unilateral_sliding_flag_from_state(np.array([2.3419, 0.8627, 0., 0.9307]), contact_face='front'))  # (False, None)
    # print(dyn._get_unilateral_sliding_flag_from_state(np.array([2.3419, 0.8627, 0., 2.1215+2*np.pi]), contact_face='back'))  # (True, 'sliding_left')
    # print(dyn._get_unilateral_sliding_flag_from_state(np.array([2.3419, 0.8627, 0., -1.0808]), contact_face='right'))  # (True, 'sliding_right')
    # print(dyn._get_unilateral_sliding_flag_from_state(np.array([2.3419, 0.8627, 0., -2.0607-2*np.pi]), contact_face='right'))  # (True, 'sliding_left')
    # print('test get_unilateral_sliding_flag_from_state, time_cost={0}'.format(time.time() - start_time))
    ## ----------------------------------------------------


    ## TEST-3 (PushDTHybridSystem._get_unilateral_sliding_flag_from_state)
    ## ----------------------------------------------------
    # start_time = time.time()
    # polytope_list0 = dyn.get_reachable_polytopes(state=np.array([0.345, 0.165, 0.0*np.pi, 0.25*np.pi]), u=np.array([0.0, 0.0, 0.0]))
    polytope_list1 = dyn.get_reachable_polytopes(state=np.array([0.0, 0.0, 0.0*np.pi, 0.0*np.pi]), u=np.array([0.0, 0.0, 0.0]))
    # polytope_list2 = dyn.get_reachable_polytopes(state=polytope_parent_state, u=np.array([0.0, 0.0, 0.0]))
    # polytope_list3 = dyn.get_reachable_polytopes(state=np.array([0.345, 0.165, -0.99*np.pi, 1.0*np.pi]), u=np.array([0.0, 0.0, 0.0]))
    # polytope_list4 = dyn.get_reachable_polytopes(state=np.array([0.345, 0.165, -0.99*np.pi, -0.9*np.pi]), u=np.array([0.0, 0.0, 0.0]))
    # polytope_list5 = dyn.get_reachable_polytopes(state=np.array([0.345, 0.165, -0.99*np.pi, -0.8*np.pi]), u=np.array([0.0, 0.0, 0.0]))

    # print('length of polytope list: {0}'.format(len(polytope_list)))
    # print('test get_unilateral_sliding_flag_from_state, time_cost={0}'.format(time.time() - start_time))
    # print('test get_reachable_polytopes, time_cost={0}'.format(time.time() - start_time))
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    # fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list0, dyn, color='red', alpha=0.2, distance_scaling_array=distance_scaling_array)
    # fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list1, dyn, color='red', alpha=0.2, distance_scaling_array=distance_scaling_array)
    # fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list2, dyn, fig, ax, color='orange', alpha=0.5, distance_scaling_array=distance_scaling_array)
    # fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list3, dyn, fig, ax, color='yellow', alpha=0.5, distance_scaling_array=distance_scaling_array)
    # fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list4, dyn, fig, ax, color='green', alpha=0.5, distance_scaling_array=distance_scaling_array)
    # fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list5, dyn, fig, ax, color='cyan', alpha=0.5, distance_scaling_array=distance_scaling_array)
    
    # fig1, ax1 = visualize_ND_AH_polytope(list_of_AH_polytopes=polytope_list, dim1=0, dim2=1, N=360, epsilon=3e-4)
    # fig2, ax2 = visualize_ND_AH_polytope(list_of_AH_polytopes=polytope_list, dim1=1, dim2=2, N=360, epsilon=3e-4)
    # fig3, ax3 = visualize_ND_AH_polytope(list_of_AH_polytopes=polytope_list, dim1=0, dim2=2, N=360, epsilon=3e-4)
    # print('test get_reachable_polytopes & visualize_ND_AH_polytope, time_cost={0}'.format(time.time() - start_time))
    # plt.show()
    from pypolycontain.visualization.visualize_2D import visualize_ND_AH_polytope
    # E_mat = dyn.E_mat['sliding_right']
    # Xi_mat = dyn.Xi_mat['sliding_right']
    # h_polytope = H_polytope(H=E_mat, h=Xi_mat)
    # ah_polytope = to_AH_polytope(h_polytope)
    # fig1, ax1 = visualize_ND_AH_polytope([ah_polytope], dim1=0, dim2=1)
    # fig2, ax2 = visualize_ND_AH_polytope([ah_polytope], dim1=1, dim2=2)
    # fig3, ax3 = visualize_ND_AH_polytope([ah_polytope], dim1=0, dim2=2)
    # plt.show()
    ## ----------------------------------------------------


    ## TEST-4 (pypolycontain.lib.operations.distance_point_polytope)
    ## ----------------------------------------------------
    # start_time = time.time()
    # from pypolycontain.lib.operations import distance_point_polytope
    # distance, nearest_point = distance_point_polytope(polytope_list[0], np.array([0.24, 0.925, 0.5*np.pi]), ball='infinity', distance_scaling_array=distance_scaling_array)
    # print('distance:{0}, nearest_point:{1}'.format(distance, nearest_point))
    # print('test distance_point_polytope, time_cost={0}'.format(time.time() - start_time))
    ## ----------------------------------------------------


    ## TEST-5 (pypolycontain.lib.common.utils.duplicate_state_with_multiple_azimuth_angle)
    ## ----------------------------------------------------
    # start_time = time.time()
    # print(duplicate_state_with_multiple_azimuth_angle(origin_state=np.array([0.3418, 0.1792, 0.2314])))
    # print(duplicate_state_with_multiple_azimuth_angle(origin_state=np.array([0.3418, 0.1792, 1.5687])))
    # print(duplicate_state_with_multiple_azimuth_angle(origin_state=np.array([0.3418, 0.1792, 3.1398])))
    # print('test duplicate_state_with_multiple_azimuth_angle, time_cost={0}'.format(time.time() - start_time))
    ## ----------------------------------------------------


    ## TEST-6 (pypolycontain.lib.operation.distance_point_polytope_with_multiple_azimuth)
    ## ----------------------------------------------------
    # start_time = time.time()
    # query_point = np.array([0.3418, 0.1792, 3.1398])
    # distance, nearest_point, modified_query_point = distance_point_polytope_with_multiple_azimuth(polytope_list1[6], query_point, ball='l2', distance_scaling_array=distance_scaling_array, return_modifed_query_point=True)
    # ax.scatter(nearest_point[0], nearest_point[1], nearest_point[2], c='deepskyblue', marker='o', s=20, label='x_near')
    # ax.scatter(query_point[0], query_point[1], query_point[2], c='forestgreen', marker='o', s=20, label='x_rand')
    # ax.scatter(modified_query_point[0], modified_query_point[1], modified_query_point[2], c='orange', marker='o', s=20, label='x_rand2')
    # ax.plot([nearest_point[0], modified_query_point[0]], [nearest_point[1], modified_query_point[1]], [nearest_point[2], modified_query_point[2]], linestyle='--')
    # print('nearest distance: {0}'.format(distance))
    # print('nearest point: {0}'.format(nearest_point))
    # print('test distance_point_polytope_with_multiple_azimuth, time_cost={0}'.format(time.time() - start_time))
    # plt.legend()
    # plt.show()
    ## ----------------------------------------------------


    ## TEST-7 (PushDTHybridSystem.collision_free_forward_simulation)
    ## ----------------------------------------------------
    # from diversipy.polytope import sample as poly_sample
    # start_time = time.time()
    # num_input_samples = 20
    # A1_mat = np.array([[-dyn.miu_slider_pusher, 1., 0.],
    #                    [-dyn.miu_slider_pusher, -1., 0.]])
    # b1_mat = np.array([0., 0.])
    # A2_mat = np.array([[0, 0, 1]])
    # b2_mat = np.array([0.])
    
    # sampled_inputs = poly_sample(n_points=num_input_samples,
    #                              lower=np.array([0, -dyn.f_lim, -dyn.dpsic_lim]),
    #                              upper=np.array([dyn.f_lim, dyn.f_lim, dyn.dpsic_lim]),
    #                              A1=A1_mat,
    #                              b1=b1_mat,
    #                              A2=A2_mat,
    #                              b2=b2_mat)
    
    # for i in range(num_input_samples):
    #     ui = sampled_inputs[i]
    #     planning_sucess_flag, final_state, state_list = dyn.collision_free_forward_simulation(starting_state=np.array([0.345, 0.165, -0.99*np.pi, 0.5*np.pi]),
    #                                                         goal_state=nearest_point,
    #                                                         u=ui,
    #                                                         max_steps=round(dyn.reachable_set_time_step/dyn.nldynamics_time_step),
    #                                                         mode_string=('left', 'sticking'),
    #                                                         Z_obs_list=None)
    #     state_array = np.array(state_list)
    #     # print('final_state: ', final_state.tolist())
    #     # print('state_list: ', state_list)
    #     if i == 0:
    #         ax.scatter(final_state[0], final_state[1], final_state[2], c='deeppink', marker='o', s=20, label='x_final')
    #     else:
    #         ax.scatter(final_state[0], final_state[1], final_state[2], c='deeppink', marker='o', s=20)
    #     ax.plot(state_array[:, 0], state_array[:, 1], state_array[:, 2], marker='x', markersize=5, linestyle=':')

    # plt.legend()
    # plt.show()
    # print('test collision_free_forward_simulation, time_cost={0}'.format(time.time() - start_time))
    ## ----------------------------------------------------


    ## TEST-8 (PolytopeReachableSet.find_closest_state_with_hybrid_dynamics)
    ## ----------------------------------------------------
    # start_time = time.time()
    # reachable_set = PolytopeReachableSet(parent_state=polytope_parent_state,
    #                                      polytope_list=polytope_list2,
    #                                      sys=dyn,
    #                                      distance_scaling_array=distance_scaling_array)
    # new_state, discard, true_dynamics_path, nearest_polytope = reachable_set.find_closest_state_with_hybrid_dynamics(query_point=query_point,
    #                                                                                                                  Z_obs_list=None,
    #                                                                                                                  duplicate_search_azimuth=True)
    # modified_new_state = reachable_set.get_state_in_set_with_correct_azimuth(query_point, nearest_polytope)
    # ax.scatter(new_state[0], new_state[1], new_state[2], c='deepskyblue', marker='o', s=20, label='x_near')
    # ax.scatter(modified_new_state[0], modified_new_state[1], modified_new_state[2], c='orange', marker='o', s=20, label='x_rand2')
    # print('nearest polytope: ', nearest_polytope.mode_string)
    # print('test find_closest_state_with_hybrid_dynamics, time_cost={0}'.format(time.time() - start_time))
    
    # plan_success_flag, cost_to_go, state_list, reached_state, approximate_input = \
    #     reachable_set.plan_path_in_set_with_hybrid_dynamics(goal_state=new_state,
    #                                                         closest_polytope=nearest_polytope,
    #                                                         Z_obs_list=None)

    # state_array = np.array(state_list)
    # ax.scatter(reached_state[0], reached_state[1], reached_state[2], c='deeppink', marker='o', s=20, label='x_reach')
    # ax.plot(state_array[:, 0], state_array[:, 1], state_array[:, 2], marker='x', markersize=5, linestyle=':')
    # print('applied input: {0}'.format(approximate_input))
    # print('goal_state: {0}, reached_state: {1}'.format(new_state, reached_state))
    # print('test find_closest_state_with_hybrid_dynamics & plan_path_in_set_with_hybrid_dynamics, time_cost={0}'.format(time.time() - start_time))
    
    # plt.title('reachable polytopes (xyz weighted)')
    # plt.legend()
    # plt.show()
    ## ----------------------------------------------------


    ## TEST-9 (PushDTHybridSystem.get_reachable_polytopes_with_variable_psic)
    ## ----------------------------------------------------
    start_time = time.time()
    polytope_list = dyn.get_reachable_polytopes_with_variable_psic(state=polytope_parent_state, u=np.array([0.0, 0.0, 0.0]))
    fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list, dyn, color='red', alpha=0.2, distance_scaling_array=distance_scaling_array)
    # query_point = np.array([0.347, 0.170, 3.1])
    # reachable_set = PolytopeReachableSet(parent_state=polytope_parent_state,
    #                                      polytope_list=polytope_list,
    #                                      sys=dyn,
    #                                      distance_scaling_array=distance_scaling_array)

    # new_state, discard, true_dynamics_path, nearest_polytope = reachable_set.find_closest_state_with_hybrid_dynamics(query_point=query_point,
    #                                                                                                                  Z_obs_list=None,
    #                                                                                                                  duplicate_search_azimuth=True)
    # modified_query_point = reachable_set.get_state_in_set_with_correct_azimuth(query_point, nearest_polytope)
    # plan_success_flag, cost_to_go, state_list, reached_state, approximate_input = \
    #     reachable_set.plan_path_in_set_with_hybrid_dynamics(goal_state=new_state,
    #                                                         closest_polytope=nearest_polytope,
    #                                                         Z_obs_list=None)
    # state_array = np.array(state_list)
    # ax.scatter(new_state[0], new_state[1], new_state[2], c='deepskyblue', marker='o', s=20, label='x_near')
    # # ax.scatter(query_point[0], query_point[1], query_point[2], c='forestgreen', marker='o', s=20, label='x_rand')
    # ax.scatter(modified_query_point[0], modified_query_point[1], modified_query_point[2], c='orange', marker='o', s=20, label='x_rand')
    # ax.scatter(reached_state[0], reached_state[1], reached_state[2], c='deeppink', marker='o', s=20, label='x_reach0')
    # ax.plot(state_array[:, 0], state_array[:, 1], state_array[:, 2], marker='x', markersize=5, linestyle=':')

    # print('query point: {0}\n nearest point: {1}\n modified query point: {2}\n reached point: {3}' \
    #             .format(query_point, new_state, modified_query_point, reached_state))
    # print('state list: ', state_list)
    # # print('nearest polytope: ', nearest_polytope.mode_string)
    # print('test get_reachable_polytopes_with_variable_psic, time_cost={0}'.format(time.time() - start_time))
    plt.savefig('./reachable_set.pdf')
    plt.legend()
    plt.show()
    ## ----------------------------------------------------


    ## TEST-10 (PolytopeReachableSetTree.nearest_k_neighbor_ids)
    ## ----------------------------------------------------
    # from r3t.symbolic_system.symbolic_system_r3t import PolytopeReachableSetTree, SymbolicSystem_Hybrid_R3T
    # start_time = time.time()
    # poly_set_tree = PolytopeReachableSetTree(distance_scaling_array=distance_scaling_array)
    # state_id_to_state = {}

    # state1 = polytope_parent_state
    # state2 = polytope_parent_state2
    # state1_id = hash(str(state1[:-1]))
    # state2_id = hash(str(state2[:-1]))
    # state_id_to_state[state1_id] = state1
    # state_id_to_state[state2_id] = state2

    # # query state without psic
    # query_state = np.array([0.29, 0.232, -2.0])
    # # query_state = np.array([0.34228213, 0.18011015, -3.19628698])

    # sys_r3t = SymbolicSystem_Hybrid_R3T(init_state=polytope_parent_state,
    #                                     sys=dyn,
    #                                     sampler=None,
    #                                     goal_sampling_bias=0.1,
    #                                     mode_consistent_sampling_bias=0.2,
    #                                     step_size=5e-2,
    #                                     distance_scaling_array=distance_scaling_array,
    #                                     nonlinear_dynamic_step_size=1e-2)
    # compute_reachable_set = sys_r3t.compute_reachable_set_func

    # reachable_set1 = compute_reachable_set(state=state1, u=np.array([0., 0., 0.]))
    # reachable_set2 = compute_reachable_set(state=state2, u=np.array([0., 0., 0.]))
    # poly_set_tree.insert(state1_id, reachable_set1)
    # poly_set_tree.insert(state2_id, reachable_set2)
    # nearest_id_list = poly_set_tree.nearest_k_neighbor_ids(query_state=query_state,
    #                                                        k=1,
    #                                                        duplicate_search_azimuth=True)
    # total_polytope_list = []
    # total_polytope_list.extend(reachable_set1.polytope_list)
    # total_polytope_list.extend(reachable_set2.polytope_list)
    # closest_parent_state = state_id_to_state[nearest_id_list[0]]
    
    # fig, ax = visualize_3D_AH_polytope_push_planning(total_polytope_list, dyn, color='red', alpha=0.2, distance_scaling_array=distance_scaling_array)
    # ax.scatter(closest_parent_state[0], closest_parent_state[1], closest_parent_state[2], c='darkgreen', marker='o', s=50, label='x_parent')

    # # nearest reachable set
    # reachable_set = poly_set_tree.id_to_reachable_sets[nearest_id_list[0]]
    # new_state, discard, true_dynamics_path, nearest_polytope = reachable_set.find_closest_state_with_hybrid_dynamics(query_point=query_state,
    #                                                                                                                  Z_obs_list=None,
    #                                                                                                                  duplicate_search_azimuth=True)
    # ax.scatter(new_state[0], new_state[1], new_state[2], c='deepskyblue', marker='o', s=20, label='x_near')
    # ax.plot([query_state[0], new_state[0]],
    #         [query_state[1], new_state[1]],
    #         [query_state[2], new_state[2]], linestyle=':')

    # # modify the azimuth angle until in set
    # modified_query_point = reachable_set.get_state_in_set_with_correct_azimuth(query_state, nearest_polytope)
    # ax.scatter(modified_query_point[0], modified_query_point[1], modified_query_point[2], c='darkorange', marker='o', s=20, label='x_rand')
    
    # ## approximate connect
    # plan_success_flag, cost_to_go, state_list, reached_state, approximate_input = \
    #     reachable_set.plan_path_in_set_with_hybrid_dynamics(goal_state=new_state,
    #                                                         closest_polytope=nearest_polytope,
    #                                                         Z_obs_list=None)
    # state_array = np.array(state_list)
    # ax.scatter(reached_state[0], reached_state[1], reached_state[2], c='deeppink', marker='o', s=20, label='x_reach0')
    # ax.plot(state_array[:, 0], state_array[:, 1], state_array[:, 2], color='orange', marker='x', markersize=5, linestyle=':')
    # print('input array: ', approximate_input)

    # ## exact connect
    # plan_success_flag, other_info = \
    #     reachable_set.plan_exact_path_in_set_with_hybrid_dynamics(goal_state=new_state,
    #                                                               closest_polytope=nearest_polytope,
    #                                                               Z_obs_list=None)
    # cost_to_go, state_list, reached_state, input_list = other_info

    # state_array = np.array(state_list)
    # ax.scatter(reached_state[0], reached_state[1], reached_state[2], c='deeppink', marker='o', s=20, label='x_reach1')
    # ax.plot(state_array[:, 0], state_array[:, 1], state_array[:, 2], color='cyan', marker='x', markersize=5, linestyle=':')
    # print('input array: ', input_list)
    # print('test nearest_k_neighbor_ids, time_cost={0}'.format(time.time() - start_time))
    # import pdb; pdb.set_trace()

    # plt.legend()
    # plt.show()
    ## ----------------------------------------------------


    ## TEST-11 (PolytopeReachableSet.contains, PolytopeReachableSet.contains_goal)
    ## ----------------------------------------------------
    # root_state = np.array([0.25, 0.05, 0.5*np.pi, 1.0*np.pi])
    # goal_state = np.array([0.25, 0.45, 0.5*np.pi])
    # empty_input = np.zeros(3,)
    # plan_step_size = 0.05
    # sim_step_size = 0.01

    # polytope_list = dyn.get_reachable_polytopes_with_variable_psic(root_state, empty_input, plan_step_size, use_convex_hull=True)
    # reachable_set = PolytopeReachableSet(parent_state=None, polytope_list=polytope_list, sys=dyn, epsilon=0.001, contains_goal_function=None,
    #                                      cost_to_go_function=None, mode_consistent_sampling_bias=0., distance_scaling_array=distance_scaling_array,
    #                                      deterministic_next_state=None, use_true_reachable_set=False, reachable_set_step_size=plan_step_size,
    #                                      nonlinear_dynamic_step_size=sim_step_size)
    # fig, ax = visualize_3D_AH_polytope_push_planning(polytope_list, dyn, color='red', alpha=0.01, distance_scaling_array=distance_scaling_array)
    # ax.scatter(root_state[0], root_state[1], root_state[2], color='blue', s=20)
    # ax.scatter(goal_state[0], goal_state[1], goal_state[2], color='green', s=20)

    # import pdb; pdb.set_trace()
    # # contains_goal, closest_polytope, exact_path_info = reachable_set.contains_goal(goal_state)

    # num_samples = 800
    # num_contains = 0
    # for i in range(num_samples):
    #     if i % 100 == 0:
    #         print('sampling: {0}/{1}'.format(i,num_samples))
    #     new_sample = np.random.uniform([0.235, 0.035, 1.3], [0.265, 0.065, 1.8])
    #     contain_flag, proj_state = reachable_set.contains(new_sample, return_closest_state=True)
    #     if contain_flag:
    #         num_contains += 1
    #     ax.scatter(proj_state[0], proj_state[1], proj_state[2], color='purple', s=5)
    # print('{0}/{1} contained in polytopes'.format(num_contains,num_samples))

    # plt.show()
    ## ----------------------------------------------------
