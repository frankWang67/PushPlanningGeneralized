import pydrake
from r3t.common.r3t import *
from r3t.common.r3t_contact import *
from polytope_symbolic_system.common.symbolic_system import *
from polytope_symbolic_system.common.utils import *
from pypolycontain.lib.operations import distance_point_polytope, distance_point_polytope_with_multiple_azimuth
from collections import deque
from rtree import index
from closest_polytope_algorithms.bounding_box.polytope_tree import PolytopeTree
from closest_polytope_algorithms.bounding_box.box import AH_polytope_to_box, \
    point_to_box_dmax, point_to_box_distance
from closest_polytope_algorithms.bounding_box.box import point_in_box
from pypolycontain.lib.operations import to_AH_polytope

from shapely.geometry import MultiPolygon
from shapely import intersects

from diversipy.polytope import sample as poly_sample

class PolytopeReachableSet(ReachableSet):
    def __init__(self, parent_state, polytope_list, sys:PushDTHybridSystem=None, epsilon=1e-3, contains_goal_function = None, cost_to_go_function=None, \
                 mode_consistent_sampling_bias=0., distance_scaling_array = None, \
                 deterministic_next_state = None, use_true_reachable_set=False, reachable_set_step_size=5e-2, nonlinear_dynamic_step_size=1e-2, num_input_samples=10):
        ReachableSet.__init__(self, parent_state=parent_state, path_class=PolytopePath)
        """
        The PolytopeReachableSet, to do nearest point search, path planning, and goal checking
        :param parent_state: parent state of the reachable polytopes
        :param polytope_list: ndarray of reachable polytopes
        :param sys: dynamical system, to do forward simulation, dynamics linearization
        :param epsilon: the distance threshold to tell two points are near enough
        :param contains_goal_function: the function that tells whether a goal is contained in the polytopes
        :param cost_to_go_function: the function that computes one step state transition cost
        :param deterministic_next_state: list of next deterministic states in the reachable polytopes
        :param use_true_reachable_set: if true, forward simulation is supported
        :param reachable_set_step_size: step size of the reachable polytopes
        :param nonlinear_dynamic_step_size: step size of the forward simulation
        :param num_input_samples: the number of input samples when connecting to point
        """
        self.polytope_list = polytope_list
        # TODO: distance_scaling_array is not designed
        # precompute all polytopes to AABB
        try:
            self.aabb_list = [AH_polytope_to_box(p, return_AABB=True) for p in self.polytope_list]
        except TypeError:
            self.aabb_list = None
        # parameters
        self.epsilon = epsilon
        self.mode_consistent_sampling_bias = mode_consistent_sampling_bias
        self.distance_scaling_array = distance_scaling_array
        self.deterministic_next_state = deterministic_next_state
        self.sys=sys
        self.use_true_reachable_set = use_true_reachable_set
        self.reachable_set_step_size = reachable_set_step_size
        self.nonlinear_dynamic_step_size = nonlinear_dynamic_step_size
        # try:
        #     self.parent_distance = min([distance_point(p, self.parent_state)[0] for p in self.polytope_list])
        # except TypeError:
        #     self.parent_distance = distance_point(self.polytope_list, self.parent_state)[0]

        self.contains_goal_function = contains_goal_function
        if cost_to_go_function is None:
            def cost_to_go(from_state, to_state, u):
                return 0.
            self.cost_to_go_function = cost_to_go
        else:
            self.cost_to_go_function = cost_to_go_function
        # assert(self.parent_distance<self.epsilon)
        self.num_input_samples = num_input_samples

    def contains(self, goal_state, return_closest_state=True, return_closest_polytope=False, duplicate_search_azimuth=False):
        """
        Check is the goal_state in contained in any of the polytopes
        The distance threshold of containment is controlled by epsilon 
        :param goal_state: the query state, without psic
        :param return_closest_state: if true, return the closest state in polytopes
        :param return_closest_polytope: if true, return the closest polytope
        :param duplicate_search_azimuth: if true, query multiple points with theta, theta-2pi, theta+2pi
        :return: flag (True/False), contains or not
        :return: (optional) the closest state in the polytopes
        :return: (optional) the closest polytope
        """
        # print(distance_point(self.polytope, goal_state)[0])
        # print(self.polytope)
        contains_flag = False
        distance = np.inf
        closest_state = None
        closest_polytope = None
        try:
            # multiple polytopes
            for i, polytope in enumerate(self.polytope_list):
                # if point_to_box_distance(goal_state, self.aabb_list[i])>0:
                #     continue
                # FIXME: the distance_scaling_array for planar coordinates (x, y, theta)
                if duplicate_search_azimuth:
                    current_distance = np.inf
                    current_closest_state = None
                    duplicate_states = duplicate_state_with_multiple_azimuth_angle(goal_state)
                    for i in range(len(duplicate_states)):
                        candidate_distance, candidate_closest_state = \
                            distance_point_polytope(polytope, duplicate_states[i],
                                                    ball='l2', distance_scaling_array=self.distance_scaling_array)
                        if candidate_distance < current_distance:
                            current_distance = candidate_distance
                            current_closest_state = candidate_closest_state
                else:
                    current_distance, current_closest_state = \
                        distance_point_polytope(polytope, goal_state,
                                                ball='l2', distance_scaling_array=self.distance_scaling_array)
                if current_distance < self.epsilon:
                    contains_flag = True
                    closest_state = goal_state
                    closest_polytope = polytope
                    break
                else:
                    if current_distance < distance:
                        distance = current_distance
                        closest_state = current_closest_state
                        closest_polytope = polytope
        except TypeError:
            # single polytope
            if duplicate_search_azimuth:
                current_distance = np.inf
                current_closest_state = None
                duplicate_states = duplicate_state_with_multiple_azimuth_angle(goal_state)
                for i in range(len(duplicate_states)):
                    candidate_distance, candidate_closest_state = \
                        distance_point_polytope(polytope, duplicate_states[i],
                                                ball='l2', distance_scaling_array=self.distance_scaling_array)
                    if candidate_distance < current_distance:
                        current_distance = candidate_distance
                        current_closest_state = candidate_closest_state
            else:
                distance, closest_state = \
                    distance_point_polytope(self.polytope_list, goal_state,
                                            ball='l2', distance_scaling_array=self.distance_scaling_array)
            if distance < self.epsilon:
                contains_flag = True
                closest_state = goal_state
                closest_polytope = self.polytope_list
        if return_closest_state and return_closest_polytope:
            return contains_flag, closest_state, closest_polytope
        elif return_closest_state and not return_closest_polytope:
            return contains_flag, closest_state
        elif not return_closest_state and return_closest_polytope:
            return contains_flag, closest_polytope
        else:
            return contains_flag

    def contains_goal(self, goal_state):
        """
        Check if the reachable polytopes contain goal
        Call the underlying contains_goal_function
        :param goal_state: the query goal_state, without psic
        :return: Tuple, (flag (contains goal or not), true_dynamics_path, closest_polytope, exact_path_info (cost_to_go, path, new_state_with_psic, applied_u))
        """
        # check if goal is epsilon away from the reachable sets
        if self.contains_goal_function is not None:
            return self.contains_goal_function(self, goal_state)
        else:
            contains_goal_flag, closest_polytope = self.contains(goal_state,
                                                                 return_closest_state=False,
                                                                 return_closest_polytope=True,
                                                                 duplicate_search_azimuth=True)
            # CHECK CONTAINMENT
            if not contains_goal_flag:
                return False, None, None
            # PLAN PATH
            # FIXME: implement plan_exact_path_in_set_with_hybrid_dynamics
            plan_success_flag, exact_path_info = self.plan_exact_path_in_set_with_hybrid_dynamics(goal_state, closest_polytope, Z_obs_list=None)
            if not plan_success_flag:
                return True, None, None
            return True, closest_polytope, exact_path_info

    def find_closest_state(self, query_point, save_true_dynamics_path=False, Z_obs_list=None):
        '''
        Find the closest state from the query point to a given polytope
        :param query_point:
        :return: Tuple (closest_point, discard flag, state)
        '''
        ## Upper Bound method
        distance = np.inf
        closest_point = None
        # the polytope of the nearest point FIXME: how to chooce the best one if multiple polytopes overlap
        p_used = None
        try:
            # multiple polytopes
            #use AABB to upper bound distance
            min_dmax = np.inf
            for i, aabb in enumerate(self.aabb_list):
                # the max distance between query_point and aabb
                dmax = point_to_box_dmax(query_point, aabb)
                if dmax < min_dmax:
                    min_dmax = dmax
            for i, p in enumerate(self.polytope_list):
                #ignore polytopes that are impossible
                if point_to_box_distance(query_point, self.aabb_list[i]) > min_dmax:
                    continue
                # compute and update the nearest point and nearest distance
                d, proj = distance_point_polytope(p, query_point, ball='l2', distance_scaling_array=self.distance_scaling_array)
                if d<distance:
                    distance = d
                    closest_point = proj
                    p_used = p
            assert closest_point is not None
        except TypeError:
            # single polytope
            closest_point = distance_point_polytope(self.polytope_list, query_point, ball='l2', distance_scaling_array=self.distance_scaling_array)[1]
            p_used = self.polytope_list
        # if the query_point is near the parent state (root of the reachable polytopes)
        if np.linalg.norm(closest_point-self.parent_state)<self.epsilon:
            if save_true_dynamics_path:
                return np.ndarray.flatten(closest_point), True, np.asarray([])
            return np.ndarray.flatten(closest_point), True, np.asarray([self.parent_state, np.ndarray.flatten(closest_point)])
        if self.use_true_reachable_set and self.reachable_set_step_size:
            #solve for the control input that leads to this state
            current_linsys = self.sys.get_linearization(self.parent_state) #FIXME: does mode need to be specified?
            u = np.dot(np.linalg.pinv(current_linsys.B*self.reachable_set_step_size), \
                       (np.ndarray.flatten(closest_point)-np.ndarray.flatten(self.parent_state)-\
                        self.reachable_set_step_size*(np.dot(current_linsys.A, self.parent_state)+\
                                                                                      np.ndarray.flatten(current_linsys.c))))
            u = np.ndarray.flatten(u)[0:self.sys.u.shape[0]]
            #simulate nonlinear forward dynamics
            state = self.parent_state
            state_list = [self.parent_state]
            for step in range(int(self.reachable_set_step_size/self.nonlinear_dynamic_step_size)):
                try:
                    state = self.sys.forward_step(u=np.atleast_1d(u), linearlize=False, modify_system=False, step_size = self.nonlinear_dynamic_step_size, return_as_env = False,
                         starting_state= state)

                    # TODO: check for collision
                    # r3t/dubin_car/dubin_car_rg_rrt_star.py
                    # point_collides_with_obstacle
                    if Z_obs_list is not None:
                        for obs in Z_obs_list:
                            obs_p = to_AH_polytope(obs)
                            obs_box = AH_polytope_to_box(obs_p, return_AABB = True)
                            if point_in_box(state, obs_box):
                                # collided, Just discard, same as plan_collision_free_path_in_set()
                                return np.ndarray.flatten(closest_point), True, np.asarray([])   
                        
                    state_list.append(state)

                except Exception as e:
                    print('Caught %s' %e)
                    return np.ndarray.flatten(closest_point), True, np.asarray([])
                # print step,state
            # print(state, closest_point)
            if save_true_dynamics_path:
                if len(state_list)<=3:
                    print('Warning: short true dynamics path')
                return np.ndarray.flatten(state), False, np.asarray(state_list)
            else:
                return np.ndarray.flatten(state), False, np.asarray([self.parent_state, np.ndarray.flatten(state)])
        else:
            return np.ndarray.flatten(closest_point), False, np.asarray([self.parent_state, np.ndarray.flatten(closest_point)])

    def find_closest_state_with_hybrid_dynamics(self, query_point, save_true_dynamics_path=False, Z_obs_list:MultiPolygon=None,
                                                    duplicate_search_azimuth=False, slider_in_contact=False):
        """
        Find the closest state in the polytopes with hybrid dynamics
        :param query_point: the queried point, without psic
        :param save_true_dynamic_path: if true, save all states traversed in forward simulation
                                       if false, save only the start point (parent_state) and end point (nearest_point) 
        :param Z_obs_list: shapely.geometry.MultiPolygon object, including all obstacles
        :param duplicate_search_azimuth: if true, query multiple points with theta, theta-2pi, theta+2pi
        :param slider_in_contact: if true, return contact_mode_consistent polytope with probability=0.5, and
                                           return nearest polytope with probability=0.5
        :return: Tuple (nearest_point, flag (whether the nearest_point is close enough to parent_state), states traversed from start point to end point (with psic), closest_polytope)
        """
        # initialize the closest point to query_point, the corresponding minimum distance and the polytope of it
        min_distance = np.inf
        closest_point = None
        closest_polytope = None

        if (not slider_in_contact and np.random.rand() <= self.mode_consistent_sampling_bias) or \
            (slider_in_contact and np.random.rand() <= 0.5):
            # find the mode consistent polytope (to prevent contact face switch)
            for polytope in self.polytope_list:
                if polytope.mode_consistent:
                    min_distance, closest_point = distance_point_polytope_with_multiple_azimuth(polytope,
                                                                                                query_point,
                                                                                                ball='l2',
                                                                                                distance_scaling_array=self.distance_scaling_array)
                    closest_polytope = polytope

        else:
            if self.aabb_list is None:
                # single polytope
                if duplicate_search_azimuth:
                    _, closest_point = distance_point_polytope_with_multiple_azimuth(self.polytope_list,
                                                                                     query_point,
                                                                                     ball='l2',
                                                                                     distance_scaling_array=self.distance_scaling_array)
                else:
                    _, closest_point = distance_point_polytope(self.polytope_list,
                                                               query_point,
                                                               ball='l2',
                                                               distance_scaling_array=self.distance_scaling_array)
                closest_polytope = self.polytope_list
            else:
                # minimize the maximum distance between the query point and each AABB
                minimax_distance = np.inf
                for i, aabb in enumerate(self.aabb_list):
                    # the max distance between query_point and aabb
                    max_distance = point_to_box_dmax(query_point, aabb)
                    if max_distance < minimax_distance:
                        minimax_distance = max_distance
                for i, polytope in enumerate(self.polytope_list):
                    #ignore polytopes that are impossible
                    if point_to_box_distance(query_point, self.aabb_list[i]) > minimax_distance:
                        continue
                    # compute and update the nearest point and nearest distance
                    if duplicate_search_azimuth:
                        distance, projected_point = distance_point_polytope_with_multiple_azimuth(polytope,
                                                                                                  query_point,
                                                                                                  ball='l2',
                                                                                                  distance_scaling_array=self.distance_scaling_array)
                    else:
                        distance, projected_point = distance_point_polytope(polytope,
                                                                            query_point,
                                                                            ball='l2',
                                                                            distance_scaling_array=self.distance_scaling_array)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = projected_point
                        closest_polytope = polytope
        # assert we find the closest point
        assert closest_point is not None

        # if the query_point is near (reachable in one step) the parent state (root of the reachable polytopes)
        if np.linalg.norm(closest_point - self.parent_state[:-1]) < self.epsilon:
            return closest_point.flatten(()), True, np.asarray([]), closest_polytope

        # if the query_point is reachable in multiple steps
        if self.use_true_reachable_set and self.reachable_set_step_size:
            # parameter check
            try:
                assert (self.reachable_set_step_size == self.sys.reachable_set_time_step)
            except:
                raise AssertionError('PolytopeReachableSet: reachable_set_step_size:{0} does not equals sys.reachable_set_time_step:{1}!'.format(self.reachable_set_step_size, self.sys.reachable_set_time_step))

            try:
                assert (self.nonlinear_dynamic_step_size == self.sys.nldynamics_time_step)
            except:
                raise AssertionError('PolytopeReachableSet: nonlinear_dynamic_step_size:{0} does not equals sys.nldynamics_time_step:{1}!'.format(self.nonlinear_dynamic_step_size, self.sys.nldynamics_time_step))

            # linearize the system at current state, current mode, and last applied input (if the contact face remains unchanged)
            linear_system = self.sys.get_linearization(state=self.parent_state,
                                                       u_bar=closest_polytope.applied_u,
                                                       mode=closest_polytope.mode_string)
            
            # compute the approximated input
            A_mat = linear_system.A.copy()
            B_mat_pinv = np.linalg.pinv(linear_system.B)
            c_mat = linear_system.c.copy()
            # E_mat = linear_system.E.copy()
            # Xi_mat = linear_system.Xi.copy()
            nominal_state = self.parent_state.copy().flatten()  # exclude psic

            # FIXME: make sure the formula is correct, should the '- nominal_state[:-1]' be added ?
            # approximate_input = np.matmul(B_mat_pinv, closest_point.flatten() - nominal_state[:-1] - np.matmul(A_mat, nominal_state) - c_mat)
            approximate_input = np.matmul(B_mat_pinv, closest_point.flatten() - np.matmul(A_mat, nominal_state) - c_mat)
            approximate_input = approximate_input.flatten()[:self.sys.dim_u]
            # FIXME: project approximated_input to Eu <= Xi?
            # H_polytope_input = H_polytope(H=E_mat, h=Xi_mat)
            # constraint_violation, projected_input = distance_point_polytope(H_polytope_input, approximate_input, ball='l2')

            # simulate nonlinear forward dynamics
            state = nominal_state.copy()
            state_list = [state]
            for step in range(round(self.reachable_set_step_size/self.nonlinear_dynamic_step_size)):
                # simulate one step
                state = self.sys.forward_step(u=approximate_input,
                                              linearize=False,
                                              step_size=self.nonlinear_dynamic_step_size,
                                              starting_state=state,
                                              mode_string=closest_polytope.mode_string)
                # collision check
                if Z_obs_list is not None:
                    collision_flag = intersects(Z_obs_list, gen_polygon(state[:-1], self.sys.slider_geom[:2]))
                    if collision_flag:
                        # FIXME: closest point is unreachable, should return parent_state, state or closest_point?
                        return closest_point.flatten(), True, np.asarray([]), closest_polytope

                state_list.append(state)

            # FIXME: should we choose the nearest state to closest_point during forward simulation?
            nearest_state_idx = len(state_list)-1
            # nearest_state_idx = np.argmin(np.linalg.norm(state_list - closest_point, ord=2, axis=1))

            # return the final state reached in simulation
            if save_true_dynamics_path:
                if len(state_list[:nearest_state_idx+1]) <= 3:
                    print('Warning: short true dynamics path')
                return state_list[nearest_state_idx].flatten(), False, np.asarray(state_list[:nearest_state_idx+1]), closest_polytope
            else:
                return state_list[nearest_state_idx].flatten(), False, np.asarray([self.parent_state, state_list[:nearest_state_idx+1].flatten()]), closest_polytope
        else:
            return closest_point.flatten(), False, np.asarray([self.parent_state, closest_point.flatten()]), closest_polytope

    def get_state_in_set_with_correct_azimuth(self, goal_state, closest_polytope):
        """
        Get the state in polytope with correct azimuth
        :param goal_state: the goal state, without psic
        :param closest_polytope: the closest polytope
        :return modifed goal_state
        """
        duplicated_states = duplicate_state_with_multiple_azimuth_angle(goal_state)
        min_distance = np.pi
        modified_goal_state = None
        for i in range(len(duplicated_states)):
            distance, _ = distance_point_polytope(closest_polytope, duplicated_states[i], ball='l2', distance_scaling_array=self.distance_scaling_array)
            if distance < min_distance:
                min_distance = distance
                modified_goal_state = duplicated_states[i]

        return modified_goal_state

    def plan_path_in_set_with_hybrid_dynamics(self, goal_state, closest_polytope, Z_obs_list=None):
        """
        Plan approximate path for goal state in set
        :param goal_state: goal state, without psic
        :param closest_polytope: the closest polytope
        :param Z_obs_list: the MultiPolygon object for obstacles
        """
        # get goal state with correct azimuth
        goal_state = self.get_state_in_set_with_correct_azimuth(goal_state, closest_polytope)

        # get parent state with correct psic
        current_contact_face = self.sys._get_contact_face_from_state(self.parent_state)
        if current_contact_face != closest_polytope.mode_string[0]:
            current_psic = self.sys.psic_each_face_center[closest_polytope.mode_string[0]]
        else:
            current_psic = self.parent_state[-1]
        actual_parent_state = np.append(self.parent_state[:-1], current_psic)

        # calculate input
        approximate_input = self.sys.calculate_input(goal_state=goal_state,
                                                     nominal_x=actual_parent_state,
                                                     nominal_u=closest_polytope.applied_u,
                                                     mode_string=closest_polytope.mode_string)

        # calculate path
        collision_free_flag, reached_state, state_list = self.sys.collision_free_forward_simulation(starting_state=actual_parent_state,
                                                                                                    goal_state=goal_state,
                                                                                                    u=approximate_input,
                                                                                                    max_steps=round(self.reachable_set_step_size/self.nonlinear_dynamic_step_size),
                                                                                                    mode_string=closest_polytope.mode_string,
                                                                                                    Z_obs_list=Z_obs_list)
        
        # collision
        if not collision_free_flag:
            return False, None, None, None, None

        cost_to_go = self.cost_to_go_function(self.parent_state[:-1], goal_state, approximate_input)
        return True, cost_to_go, state_list, reached_state, approximate_input

    def plan_exact_path_in_set_with_hybrid_dynamics(self, goal_state, closest_polytope, Z_obs_list=None):
        """
        Plan approximate path for goal state in set
        :param goal_state: goal state, without psic
        :param closest_polytope: the closest polytope
        :param Z_obs_list: the MultiPolygon object for obstacles
        """
        # get goal state with correct azimuth
        goal_state = self.get_state_in_set_with_correct_azimuth(goal_state, closest_polytope)

        # get parent state with correct psic
        current_contact_face = self.sys._get_contact_face_from_state(self.parent_state)
        if current_contact_face != closest_polytope.mode_string[0]:
            current_psic = self.sys.psic_each_face_center[closest_polytope.mode_string[0]]
        else:
            current_psic = self.parent_state[-1]
        actual_parent_state = np.append(self.parent_state[:-1], current_psic)

        # get goal state with correct psic
        goal_psic = self.sys._calculate_desired_psic_from_state_transition(from_state=self.parent_state[:-1],
                                                                           to_state=goal_state,
                                                                           contact_face=closest_polytope.mode_string[0])
        actual_goal_state = np.append(goal_state, angle_diff(current_psic, goal_psic)+current_psic)
        
        # solve optimal control
        # control_input = self.sys.solve_optimal_control(start_state=actual_parent_state,
        #                                                end_state=actual_goal_state,
        #                                                contact_face=closest_polytope.mode_string[0],
        #                                                optimal_time=self.reachable_set_step_size)

        # forward dynamics
        state = actual_parent_state.copy()
        state_list = [state]
        input_list = []
        # t = 0.
        max_steps = round(self.reachable_set_step_size/self.nonlinear_dynamic_step_size)
        input_t = closest_polytope.applied_u
        for i in range(max_steps):
            # input_t = control_input(t)

            # input_t = self.sys.solve_discrete_lqr(actual_state_x=state,
            #                                       desired_state_xf=actual_goal_state,
            #                                       dt=self.nonlinear_dynamic_step_size,
            #                                       contact_face=closest_polytope.mode_string[0],
            #                                       Q=np.diag(10*np.append(self.distance_scaling_array, 0.0001)))

            input_t = self.sys.calculate_input(goal_state=goal_state,
                                               nominal_x=state,
                                               nominal_u=input_t,
                                               mode_string=closest_polytope.mode_string)
            input_t[:-1] = input_t[:-1]*(max_steps/(max_steps-i))
            input_t[-1] = (goal_psic - current_psic) / self.reachable_set_step_size

            state = self.sys.forward_step(u=input_t,
                                          step_size=self.nonlinear_dynamic_step_size,
                                          starting_state=state,
                                          mode_string=closest_polytope.mode_string).reshape(-1)

            if Z_obs_list is not None:
                collision_flag = intersects(Z_obs_list, gen_polygon(state[:-1], self.sys.slider_geom[:2]))
                if collision_flag:
                    return (False, None, None, None, None)
            state_list.append(state)
            input_list.append(input_t)
            # t += self.nonlinear_dynamic_step_size

        cost_to_go = self.cost_to_go_function(self.parent_state[:-1], goal_state, input_list)
        return True, (cost_to_go, state_list, state_list[-1], input_list)

    def plan_collision_free_path_in_set(self, goal_state, return_deterministic_next_state = False):
        """
        Plan collision-free path for goal_state in the reachable set
        Call the underlying plan_collision_free_path_in_set_function, seems not implemented
        :param goal_state: the query goal_state
        :param return_deterministic_next_state: if true, return list of next deterministic states
        :return: distance between parent_state and goal_state
        :return: deque of parent_state and goal_state
        :return: (optional) deterministic_next_state
        """
        try:
            if self.plan_collision_free_path_in_set_function:
                return self.plan_collision_free_path_in_set_function(goal_state, return_deterministic_next_state)
        except AttributeError:
            pass
        #fixme: support collision checking
        #
        # is_contain, closest_state = self.contains(goal_state)
        # if not is_contain:
        #     print('Warning: this should never happen')
        #     return np.linalg.norm(self.parent_state-closest_state), deque([self.parent_state, closest_state]) #FIXME: distance function

        # Simulate forward dynamics if there can only be one state in the next timestep
        if not return_deterministic_next_state:
            return np.linalg.norm(self.parent_state-goal_state), deque([self.parent_state, goal_state])
        return np.linalg.norm(self.parent_state-goal_state), deque([self.parent_state, goal_state]), self.deterministic_next_state

class PolytopePath:
    def __init__(self):
        self.path = deque()
    def __repr__(self):
        return str(self.path)
    def append(self, path):
        self.path+=path #TODO

class PolytopeReachableSetTree(ReachableSetTree):
    '''
    Polytopic reachable set with PolytopeTree
    '''
    def __init__(self, key_vertex_count=0, distance_scaling_array=None):
        """
        The PolytopeReachableSetTree, to do nearest k neighbors search
        :param key_vertex_count: number of sampled key points
        :param distance_scaling_array: the distance metric
        """
        ReachableSetTree.__init__(self)

        # the PolytopeTree object, to do nearest k polytope search
        self.polytope_tree = None

        # stores the aabb of all polytopes
        p = index.Property(dimension=len(distance_scaling_array))
        self.aabb_idx = index.Index(properties=p)

        # convert polytope id --> (single) polytope
        self.polytope_id_to_polytope = {}

        # convert: state id --> reachable set (PolytopeReachableSet object)
        self.id_to_reachable_sets = {}

        # convert: polytope (AH_polytope object) --> state id
        self.polytope_to_id = {}

        self.key_vertex_count = key_vertex_count

        # for d_neighbor_ids
        # self.state_id_to_state = {}
        # self.state_idx = None
        # self.state_tree_p = index.Property()
        self.distance_scaling_array = distance_scaling_array
        self.repeated_distance_scaling_array = np.tile(self.distance_scaling_array, 2)

    def insert(self, state_id, reachable_set):
        """
        Insert reachable polytopes of a new state
        :param state_id: state id
        :param reachable_set: list of PolytopeReachableSet object
        """
        try:
            iter(reachable_set.polytope_list)
            if self.polytope_tree is None:
                self.polytope_tree = PolytopeTree(np.array(reachable_set.polytope_list),
                                                  key_vertex_count=self.key_vertex_count,
                                                  distance_scaling_array=self.distance_scaling_array)
                # for d_neighbor_ids
                # self.state_tree_p.dimension = to_AH_polytope(reachable_set.polytope[0]).t.shape[0]
            else:
                self.polytope_tree.insert(np.array(reachable_set.polytope_list))
            self.id_to_reachable_sets[state_id] = reachable_set
            for p in reachable_set.polytope_list:
                # insert the polytope
                self.polytope_to_id[p] = state_id
                # insert the aabb
                lu = AH_polytope_to_box(p)
                scaled_lu = np.multiply(self.repeated_distance_scaling_array, lu)
                self.aabb_idx.insert(hash(str(state_id)+str(p.mode_string)), scaled_lu)
                # polytope id
                self.polytope_id_to_polytope[hash(str(state_id)+str(p.mode_string))] = p
        except TypeError:
            if self.polytope_tree is None:
                self.polytope_tree = PolytopeTree(np.atleast_1d([reachable_set.polytope_list]).flatten(),
                                                  key_vertex_count=self.key_vertex_count,
                                                  distance_scaling_array=self.distance_scaling_array)
                # for d_neighbor_ids
                # self.state_tree_p.dimension = to_AH_polytope(reachable_set.polytope[0]).t.shape[0]
            else:
                self.polytope_tree.insert(np.array([reachable_set.polytope_list]))
            self.id_to_reachable_sets[state_id] = reachable_set
            # insert the polytope
            self.polytope_to_id[reachable_set.polytope_list] = state_id
            # insert the aabb
            lu = AH_polytope_to_box(reachable_set.polytope_list)
            scaled_lu = np.multiply(self.repeated_distance_scaling_array, lu)
            self.aabb_idx.insert(hash(str(state_id)+str(reachable_set.polytope_list.mode_string)), scaled_lu)
            # polytope id
            self.polytope_id_to_polytope[hash(str(state_id)+str(reachable_set.polytope_list.mode_string))] = reachable_set.polytope_list
        # for d_neighbor_ids
        # state_id = hash(str(reachable_set.parent_state))
        # self.state_idx.insert(state_id, np.repeat(reachable_set.parent_state, 2))
        # self.state_id_to_state[state_id] = reachable_set.parent_state

    def delete(self, state_id, reachable_set):
        """
        Delete reachable polytopes of a new state
        :param state_id: state id
        :param reachable_set: list of PolytopeReachableSet object
        """
        num_aabb_idx = self.aabb_idx.get_size()
        try:
            iter(reachable_set.polytope_list)
            self.polytope_tree.delete(np.array(reachable_set.polytope_list))
            polytope_list = self.id_to_reachable_sets[state_id].polytope_list
            self.id_to_reachable_sets.pop(state_id)
            for p in polytope_list:
                # delete the polytope
                self.polytope_to_id.pop(p)
                # delete the aabb
                lu = AH_polytope_to_box(p)
                scaled_lu = np.multiply(self.repeated_distance_scaling_array, lu)
                self.aabb_idx.delete(hash(str(state_id)+str(p.mode_string)), scaled_lu)
                # delete polytope state
                self.polytope_id_to_polytope.pop(hash(str(state_id)+str(p.mode_string)))
            if self.aabb_idx.get_size() > (num_aabb_idx-len(polytope_list)):
                print('PolytopeReachableSetTree: warning, aabb of state {0} was not deleted successfully'.format(state_id))

        except TypeError:
            self.polytope_tree.delete(np.array([reachable_set.polytope_list]))
            self.id_to_reachable_sets.pop(state_id)
            # delete the polytope
            self.polytope_to_id.pop(reachable_set.polytope_list)
            # delete the aabb
            lu = AH_polytope_to_box(reachable_set.polytope_list)
            scaled_lu = np.multiply(self.repeated_distance_scaling_array, lu)
            self.aabb_idx.delete(hash(str(state_id)+str(reachable_set.polytope_list.mode_string)), scaled_lu)
            # delete polytope state
            self.polytope_id_to_polytope.pop(hash(str(state_id)+str(reachable_set.polytope_list.mode_string)))
            if self.aabb_idx.get_size() > (num_aabb_idx-1):
                print('PolytopeReachableSetTree: warning, aabb of state {0} was not deleted successfully'.format(state_id))

    def nearest_k_neighbor_ids(self, query_state, k=1, return_state_projection=False, duplicate_search_azimuth=False):
        """
        Return the ids of k nearest polytopes
        :param query_state: the query state, without psic
        :param k: number of nearest polytopes
        :param return_state_projection: if true, return the state projections
        :param duplicate_search_azimuth: if true, query multiple points with theta, theta-2pi, theta+2pi
        """
        if k is None:
            if self.polytope_tree is None:
                return None
            # assert(len(self.polytope_tree.find_closest_polytopes(query_state))==1)
            try:
                assert duplicate_search_azimuth == False
            except:
                raise AssertionError('PolytopeReachableSetTree: duplicate azimuth search is not implemented for returning multiple nearest polytopes!')
            best_polytopes, best_distance, state_projections = self.polytope_tree.find_closest_polytopes(query_state, return_state_projection=True, may_return_multiple=True)
            if not return_state_projection:
                return [self.polytope_to_id[bp] for bp in best_polytopes]
            return [self.polytope_to_id[bp] for bp in best_polytopes], best_polytopes, best_distance, state_projections

        else:
            if self.polytope_tree is None:
                return None
            # assert(len(self.polytope_tree.find_closest_polytopes(query_state))==1)
            if duplicate_search_azimuth:
                duplicate_states = duplicate_state_with_multiple_azimuth_angle(query_state)
                best_polytope, state_projection = None, None
                best_distance = np.inf
                for i in range(len(duplicate_states)):
                    nearest_polytopes, nearest_distance, nearest_state_projection = self.polytope_tree.find_closest_polytopes(duplicate_states[i], return_state_projection=True)
                    if nearest_distance < best_distance:
                        best_distance = nearest_distance
                        best_polytope = nearest_polytopes
                        state_projection = nearest_state_projection
            else:
                best_polytope, best_distance, state_projection = self.polytope_tree.find_closest_polytopes(query_state, return_state_projection=True)

            if not return_state_projection:
                return [self.polytope_to_id[best_polytope[0]]]
            return [self.polytope_to_id[best_polytope[0]]], best_polytope, [best_distance], [state_projection]

    def get_polytopes_contains_state(self, state, duplicate_search_azimuth=False):
        """
        Return the polytopes that contains state
        :param state: the state, without psic
        :param duplicate_search_azimuth: if true, query multiple points with theta, theta-2pi, theta+2pi
        :return: list of polytopes
        """
        scaled_state = np.multiply(self.repeated_distance_scaling_array, np.tile(state, 2))
        polytope_ids = list(self.aabb_idx.intersection(scaled_state))
        polytope_list = [self.polytope_id_to_polytope[id] for id in polytope_ids]
        if duplicate_search_azimuth:
            duplicate_states = duplicate_state_with_multiple_azimuth_angle(state)
            # theta-2pi
            state_minus_2pi = duplicate_states[1, :]
            scaled_state = np.multiply(self.repeated_distance_scaling_array, np.tile(state_minus_2pi, 2))
            polytope_ids = list(self.aabb_idx.intersection(scaled_state))
            polytope_list.extend([self.polytope_id_to_polytope[id] for id in polytope_ids])
            # theta+2pi
            state_plus_2pi = duplicate_states[2, :]
            scaled_state = np.multiply(self.repeated_distance_scaling_array, np.tile(state_plus_2pi, 2))
            polytope_ids = list(self.aabb_idx.intersection(scaled_state))
            polytope_list.extend([self.polytope_id_to_polytope[id] for id in polytope_ids])
        return polytope_list

    def d_neighbor_ids(self, query_state, d = np.inf):
        '''
        :param query_state:
        :param d:
        :return:
        '''
        # return self.state_idx.intersection(, objects=False)
        raise NotImplementedError

class SymbolicSystem_StateTree(StateTree):
    def __init__(self, distance_scaling_array=None):
        """
        The SymbolicSystem_StateTree, to do state ids in reachable set query
        :param distance_scaling_array: the distance metric
        """
        StateTree.__init__(self)
        self.state_id_to_state = {}
        self.state_tree_p = index.Property()
        self.state_idx = None
        self.state_idx_minus_2pi = None
        self.state_idx_plus_2pi = None
        self.distance_scaling_array = distance_scaling_array
    # delayed initialization to consider dimensions
    def initialize(self, dim):
        """
        Initialize the tree structure
        :param dim: tree dimension
        """
        self.state_tree_p.dimension=dim
        if self.distance_scaling_array is None:
            self.distance_scaling_array = np.ones(dim, dtype='float')
        self.repeated_distance_scaling_array = np.tile(self.distance_scaling_array, 2)
        print('Symbolic System State Tree dimension is %d-D' % self.state_tree_p.dimension)
        self.state_idx = index.Index(properties=self.state_tree_p)
        self.state_idx_minus_2pi = index.Index(properties=self.state_tree_p)
        self.state_idx_plus_2pi = index.Index(properties=self.state_tree_p)

    def insert(self, state_id, state):
        """
        Insert new state to the tree structure
        :param state_id: the state id
        :param state: the unscaled state
        """
        if not self.state_idx:
            self.initialize(state.shape[0])
        scaled_state = np.multiply(self.distance_scaling_array,
                                   state)
        # insert the state as a point
        self.state_idx.insert(state_id, np.tile(scaled_state, 2))
        # insert the state with modified azimuth
        state_plus_2pi = state.copy()
        state_plus_2pi[-1] += 2 * np.pi
        scaled_state_plus_2pi = np.multiply(self.distance_scaling_array, state_plus_2pi)
        state_minus_2pi = state.copy()
        state_minus_2pi[-1] -= 2 * np.pi
        scaled_state_minus_2pi = np.multiply(self.distance_scaling_array, state_minus_2pi)
        self.state_idx_plus_2pi.insert(state_id, np.tile(scaled_state_plus_2pi, 2))
        self.state_idx_minus_2pi.insert(state_id, np.tile(scaled_state_minus_2pi, 2))
        # insert index for querying state
        self.state_id_to_state[state_id] = state

    def delete(self, state_id, state):
        """
        Delete state from the tree structure
        :param state_id: the state id
        :param state: the unscaled state
        """
        index_num = self.state_idx.get_size()
        index_num_plus_2pi = self.state_idx_plus_2pi.get_size()
        index_num_minus_2pi = self.state_idx_minus_2pi.get_size()

        scaled_state = np.multiply(self.distance_scaling_array,
                                   state)
        # insert the state as a point
        self.state_idx.delete(state_id, np.tile(scaled_state, 2))
        # insert the state with modified azimuth
        state_plus_2pi = state.copy()
        state_plus_2pi[-1] += 2 * np.pi
        scaled_state_plus_2pi = np.multiply(self.distance_scaling_array, state_plus_2pi)
        state_minus_2pi = state.copy()
        state_minus_2pi[-1] -= 2 * np.pi
        scaled_state_minus_2pi = np.multiply(self.distance_scaling_array, state_minus_2pi)
        self.state_idx_plus_2pi.delete(state_id, np.tile(scaled_state_plus_2pi, 2))
        self.state_idx_minus_2pi.delete(state_id, np.tile(scaled_state_minus_2pi, 2))

        if (self.state_idx.get_size() > index_num-1) or \
            (self.state_idx_plus_2pi.get_size() > index_num_plus_2pi-1) or \
            (self.state_idx_minus_2pi.get_size() > index_num_minus_2pi-1):
            print('SymbolicSystem_StateTree: warning, state {0} was not deleted successfully'.format(state))
        
        self.state_id_to_state.pop(state_id)

    def state_ids_in_reachable_set(self, query_reachable_set, duplicate_search_azimuth=False):
        """
        Return state ids (not exactly!!!) contained in the query reachable set
        The returned states are only guaranteed to contain in the reachable set's AABBs
        :param query_reachable_set: the query reachable set
        :param duplicate_search_azimuth: if true, query multiple points with theta, theta-2pi, theta+2pi
        :return: list of state ids
        """
        assert(self.state_idx is not None)
        try:
            state_ids_list = []
            for p in query_reachable_set.polytope_list:
                lu = AH_polytope_to_box(p)
                scaled_lu = np.multiply(self.repeated_distance_scaling_array, lu)
                state_ids_list.extend(list(self.state_idx.intersection(scaled_lu)))
                if duplicate_search_azimuth:
                    state_ids_list.extend(list(self.state_idx_plus_2pi.intersection(scaled_lu)))
                    state_ids_list.extend(list(self.state_idx_minus_2pi.intersection(scaled_lu)))
            return state_ids_list
        except TypeError:
            state_ids_list = []
            lu = AH_polytope_to_box(query_reachable_set.polytope_list)
            scaled_lu = np.multiply(self.repeated_distance_scaling_array, lu)
            state_ids_list.extend(list(self.state_idx.intersection(scaled_lu)))
            if duplicate_search_azimuth:
                state_ids_list.extend(list(self.state_idx_plus_2pi.intersection(scaled_lu)))
                state_ids_list.extend(list(self.state_idx_minus_2pi.intersection(scaled_lu)))
            return state_ids_list

class SymbolicSystem_R3T(R3T):
    def __init__(self, sys, sampler, step_size, contains_goal_function = None, compute_reachable_set=None, use_true_reachable_set=False, \
                 nonlinear_dynamic_step_size=1e-2, use_convex_hull=True, goal_tolerance = 1e-2):
        self.sys = sys
        self.step_size = step_size
        self.contains_goal_function = contains_goal_function
        self.goal_tolerance = goal_tolerance
        if compute_reachable_set is None:
            def compute_reachable_set(state):
                '''
                Compute polytopic reachable set using the system
                :param h:
                :return:
                '''
                deterministic_next_state = None
                reachable_set_polytope = self.sys.get_reachable_polytopes(state, step_size=self.step_size, use_convex_hull=use_convex_hull)
                return PolytopeReachableSet(state,reachable_set_polytope, sys=self.sys, contains_goal_function=self.contains_goal_function, \
                                            deterministic_next_state=None, reachable_set_step_size=self.step_size, use_true_reachable_set=use_true_reachable_set,\
                                            nonlinear_dynamic_step_size=nonlinear_dynamic_step_size)
        R3T.__init__(self, self.sys.get_current_state(), compute_reachable_set, sampler, PolytopeReachableSetTree, SymbolicSystem_StateTree, PolytopePath)


class SymbolicSystem_Hybrid_R3T(R3T_Hybrid):
    def __init__(self, init_state, sys:PushDTHybridSystem, \
                 sampler, goal_sampling_bias, mode_consistent_sampling_bias, \
                 step_size, \
                 contains_goal_function = None, cost_to_go_function=None, \
                 distance_scaling_array = None, \
                 compute_reachable_set=None, use_true_reachable_set=False, \
                 nonlinear_dynamic_step_size=1e-2, use_convex_hull=True, goal_tolerance = 1e-3):
        self.init_state = init_state
        self.sys = sys
        self.step_size = step_size
        self.contains_goal_function = contains_goal_function
        self.distance_scaling_array = distance_scaling_array
        self.cost_to_go_function = cost_to_go_function
        self.goal_tolerance = goal_tolerance
        if compute_reachable_set is None:
            def compute_reachable_set(state, u):
                '''
                Compute polytopic reachable set using the system
                :param state: nominal state, with psic, linearization point
                :param u: nominal input, linearization point
                :return: PolytopeReachableSet object
                '''
                deterministic_next_state = None
                # GET REACHABLE POLYTOPES
                # reachable_set_polytope = self.sys.get_reachable_polytopes(state, u, step_size=self.step_size, use_convex_hull=use_convex_hull)
                reachable_set_polytope = self.sys.get_reachable_polytopes_with_variable_psic(state, u, step_size=self.step_size, use_convex_hull=use_convex_hull)
                # QUASI-STATIC ASSUMPTION
                if np.all(u == 0):
                    if use_true_reachable_set:
                        deterministic_next_state=[state]
                        for step in range(round(self.step_size / nonlinear_dynamic_step_size)):
                            deterministic_next_state.append(state)
                    else:
                        deterministic_next_state = [state, state]
                return PolytopeReachableSet(parent_state=state,
                                            polytope_list=reachable_set_polytope,
                                            sys=self.sys,
                                            epsilon=goal_tolerance,
                                            contains_goal_function=self.contains_goal_function,
                                            cost_to_go_function=self.cost_to_go_function,
                                            mode_consistent_sampling_bias=mode_consistent_sampling_bias,
                                            distance_scaling_array=self.distance_scaling_array,
                                            deterministic_next_state=deterministic_next_state,
                                            use_true_reachable_set=use_true_reachable_set,
                                            reachable_set_step_size=self.step_size,
                                            nonlinear_dynamic_step_size=nonlinear_dynamic_step_size)
            self.compute_reachable_set_func = compute_reachable_set

        R3T_Hybrid.__init__(self, root_state=init_state,
                                  compute_reachable_set=compute_reachable_set,
                                  sampler=sampler,
                                  goal_sampling_bias=goal_sampling_bias,
                                  distance_scaling_array=distance_scaling_array,
                                  reachable_set_tree_class=PolytopeReachableSetTree,
                                  state_tree_class=SymbolicSystem_StateTree,
                                  path_class=PolytopePath,
                                  dim_u=self.sys.dim_u)

class SymbolicSystem_Hybrid_R3T_Contact(R3T_Hybrid_Contact):
    def __init__(self, init_state, sys:PushDTHybridSystem, \
                 sampler, goal_sampling_bias, mode_consistent_sampling_bias, \
                 step_size, \
                 planning_scene_pkl = None, \
                 contains_goal_function = None, cost_to_go_function=None, \
                 distance_scaling_array = None, \
                 compute_reachable_set=None, use_true_reachable_set=False, \
                 nonlinear_dynamic_step_size=1e-2, use_convex_hull=True, goal_tolerance = 1e-3):
        self.init_state = init_state
        self.sys = sys
        self.planning_scene_pkl = planning_scene_pkl
        self.step_size = step_size
        self.contains_goal_function = contains_goal_function
        self.distance_scaling_array = distance_scaling_array
        self.cost_to_go_function = cost_to_go_function
        self.goal_tolerance = goal_tolerance
        if compute_reachable_set is None:
            def compute_reachable_set(state, u):
                '''
                Compute polytopic reachable set using the system
                :param state: nominal state, with psic, linearization point
                :param u: nominal input, linearization point
                :return: PolytopeReachableSet object
                '''
                deterministic_next_state = None
                # GET REACHABLE POLYTOPES
                # reachable_set_polytope = self.sys.get_reachable_polytopes(state, u, step_size=self.step_size, use_convex_hull=use_convex_hull)
                reachable_set_polytope = self.sys.get_reachable_polytopes_with_variable_psic(state, u, step_size=self.step_size, use_convex_hull=use_convex_hull)
                # QUASI-STATIC ASSUMPTION
                if np.all(u == 0):
                    if use_true_reachable_set:
                        deterministic_next_state=[state]
                        for step in range(round(self.step_size / nonlinear_dynamic_step_size)):
                            deterministic_next_state.append(state)
                    else:
                        deterministic_next_state = [state, state]
                return PolytopeReachableSet(parent_state=state,
                                            polytope_list=reachable_set_polytope,
                                            sys=self.sys,
                                            epsilon=goal_tolerance,
                                            contains_goal_function=self.contains_goal_function,
                                            cost_to_go_function=self.cost_to_go_function,
                                            mode_consistent_sampling_bias=mode_consistent_sampling_bias,
                                            distance_scaling_array=self.distance_scaling_array,
                                            deterministic_next_state=deterministic_next_state,
                                            use_true_reachable_set=use_true_reachable_set,
                                            reachable_set_step_size=self.step_size,
                                            nonlinear_dynamic_step_size=nonlinear_dynamic_step_size)
            self.compute_reachable_set_func = compute_reachable_set

        try:
            assert self.planning_scene_pkl is not None
        except:
            print('SymbolicSystem_Hybrid_R3T: planning scene file is not provided!')

        R3T_Hybrid_Contact.__init__(self, root_state=init_state,
                                    planning_scene_pkl=self.planning_scene_pkl,
                                    compute_reachable_set=compute_reachable_set,
                                    sampler=sampler,
                                    goal_sampling_bias=goal_sampling_bias,
                                    distance_scaling_array=distance_scaling_array,
                                    reachable_set_tree_class=PolytopeReachableSetTree,
                                    state_tree_class=SymbolicSystem_StateTree,
                                    path_class=PolytopePath,
                                    dim_u=self.sys.dim_u)

    def get_plan_anim_raw_data(self):
        X_slider = []
        U_slider = []
        X_pusher = []
        X_obstacles = []

        import pdb; pdb.set_trace()
        node = self.goal_node
        while True:
            assert (node.planning_scene is not None)
            if node == self.root_node:
                slider_state = node.path_from_parent.reshape(-1)
                X_slider.append(slider_state[:3].tolist())
                X_pusher.append(self.sys.get_pusher_location(slider_state, contact_face=node.mode_from_parent[0]).reshape(-1).tolist())
            else:
                for i in range(len(node.path_from_parent)-1,0,-1):
                    slider_state = node.path_from_parent[i].reshape(-1)
                    X_slider.append(slider_state[:3].tolist())
                    X_pusher.append(self.sys.get_pusher_location(slider_state, contact_face=node.mode_from_parent[0]).reshape(-1).tolist())
                    if isinstance(node.input_from_parent, list):
                        U_slider.append(node.input_from_parent[i-1].reshape(-1).tolist())
                    else:
                        U_slider.append(node.input_from_parent.reshape(-1).tolist())

            obstacle_states = []
            for i in range(len(node.planning_scene.states)):
                obstacle_states.append(np.array(node.planning_scene.states)[i].reshape(-1).tolist())
            X_obstacles.append([obstacle_states]*(len(node.path_from_parent)-1))

            node = node.parent
            if node is None:
                break

        # reverse all
        X_slider.reverse()
        U_slider.reverse()
        X_pusher.reverse()
        X_obstacles.reverse()

        data_collection = {
                            'X_slider': X_slider,
                            'U_slider': U_slider,
                            'X_pusher': X_pusher,
                            'X_obstacles': X_obstacles
                          }

        data_root = '/home/yongpeng/research/R3T_shared/data/debug'
        timestamp = self.debugger.timestamp
        os.mkdir(os.path.join(data_root, timestamp))

        with open(os.path.join(data_root, timestamp, 'output.pkl'), 'wb') as f:
            pickle.dump(data_collection, f)
