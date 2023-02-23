# --------------------------------------------------
# Test Pushing With R3T
# --------------------------------------------------

# --------------------------------------------------
# Ignore warnings
# --------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from polytope_symbolic_system.common.symbolic_system import *
from r3t.symbolic_system.symbolic_system_r3t import *


# scene configuration
# --------------------------------------------------
planning_scene_pkl = '/home/yongpeng/research/R3T_shared/data/test_scene_0.pkl'
# --------------------------------------------------

xmin, xmax, ymin, ymax, thetamin, thetamax = 0.0, 0.5, 0.0, 0.5, -np.pi, np.pi

search_space_dimensions = np.array([(xmin, xmax), (ymin, ymax), (thetamin, thetamax)])
# state_space_obstacles = MultiPolygon()  # empty obstacles
state_space_obstacles = None

# dynamics configuration
force_limit = 0.15  # old: 0.3
pusher_vel_limit = 1.0  # dpsic, old: 3.0
unilateral_sliding_region_width = 0.01  # old: 0.005
# slider_geometry = [0.07, 0.12, 0.01]

# test on robot
slider_geometry = [0.07, 0.12, 0.01]

fric_coeff_slider_pusher = 0.2  # old: 0.3
fric_coeff_slider_ground = 0.2
reachable_set_time_step = 0.05
nonlinear_dynamics_time_step = 0.01

# planner_configuration
max_planning_time = 100.0
max_nodes_in_tree = 1000
goal_tolerance = 0.001
goal_sampling_bias = 0.1  # take sample from goal
mode_consistent_sampling_bias = 0.2  # keey dynamic mode consistent (invariant contact face)
distance_scaling_array = np.array([1.0, 1.0, 0.0695])
quad_cost_state = np.diag([1.0, 1.0, 0.0695])
# quad_cost_input = np.diag([0.01, 0.01, 0.])
quad_cost_input = np.diag([0.001, 0.001, 5e-6])

# x_init = [0.15, 0.05, 0.]
# x_goal = [0.40, 0.30, 0.25*np.pi]

# test planning
x_init = [0.25, 0.05, 0.5*np.pi]
x_goal = [0.25, 0.45, 0.5*np.pi]

# test on robot
# x_init = [0.3836, 0.0014, 0.0011945]
# x_goal = [0.7836, 0.0014, 0.0011945]

psic_init = np.pi

# underlying functions
def sampler():
    return np.random.uniform(search_space_dimensions[:, 0], search_space_dimensions[:, 1])

def cost_to_go(start_state, goal_state, applied_input):
    # convert to array of shape (3,) or (N,3)
    applied_input = np.atleast_2d(np.array(applied_input)).T

    state_error = goal_state - start_state
    state_error[-1] = angle_diff(start_state[-1], goal_state[-1])
    # in case of list of input, compute the average cost
    state_cost = np.matmul(state_error.T, np.matmul(quad_cost_state, state_error))
    input_cost = np.mean(np.diagonal(np.matmul(applied_input.T, np.matmul(quad_cost_input, applied_input))))
    return state_cost + input_cost

planning_dyn = PushDTHybridSystem(f_lim=force_limit,
                                  dpsic_lim=pusher_vel_limit,
                                  unilateral_sliding_region=unilateral_sliding_region_width,
                                  slider_geom=slider_geometry,
                                  miu_slider_pusher=fric_coeff_slider_pusher,
                                  miu_slider_ground=fric_coeff_slider_ground,
                                  quad_cost_input=quad_cost_input,
                                  reachable_set_time_step=reachable_set_time_step,
                                  nldynamics_time_step=nonlinear_dynamics_time_step)

# --------------------------------------------------
# General Hybrid R3T Algorithm
# --------------------------------------------------

# planner = SymbolicSystem_Hybrid_R3T(init_state=np.append(x_init, psic_init),
#                                     sys=planning_dyn,
#                                     sampler=sampler,
#                                     goal_sampling_bias=goal_sampling_bias,
#                                     mode_consistent_sampling_bias=mode_consistent_sampling_bias,
#                                     step_size=reachable_set_time_step,
#                                     contains_goal_function=None,
#                                     cost_to_go_function=cost_to_go,
#                                     distance_scaling_array=distance_scaling_array,
#                                     compute_reachable_set=None,
#                                     use_true_reachable_set=False,
#                                     nonlinear_dynamic_step_size=nonlinear_dynamics_time_step,
#                                     use_convex_hull=True,
#                                     goal_tolerance=goal_tolerance)

# --------------------------------------------------
# Contact-Aware Hybrid R3T Algorithm
# --------------------------------------------------
planner = SymbolicSystem_Hybrid_R3T_Contact(init_state=np.append(x_init, psic_init),
                                            sys=planning_dyn,
                                            sampler=sampler,
                                            goal_sampling_bias=goal_sampling_bias,
                                            mode_consistent_sampling_bias=mode_consistent_sampling_bias,
                                            step_size=reachable_set_time_step,
                                            planning_scene_pkl=planning_scene_pkl,
                                            contains_goal_function=None,
                                            cost_to_go_function=cost_to_go,
                                            distance_scaling_array=distance_scaling_array,
                                            compute_reachable_set=None,
                                            use_true_reachable_set=False,
                                            nonlinear_dynamic_step_size=nonlinear_dynamics_time_step,
                                            use_convex_hull=True,
                                            goal_tolerance=goal_tolerance)

success, goal_node = planner.build_tree_to_goal_state(goal_state=np.array(x_goal),
                                                      allocated_time=max_planning_time,
                                                      stop_on_first_reach=False,
                                                      rewire=False,
                                                      explore_deterministic_next_state=False,
                                                      max_nodes_to_add=max_nodes_in_tree,
                                                      Z_obs_list=state_space_obstacles)

## functions for debug
## ----------------------------------------------------
def test_nearest_polytope_search(planner:R3T_Hybrid, query_state):
    """
    Test the AABB-based heuristic nearest neighbor searching
    :param planner: the R3T_Hybrid
    :param query_state: the query state
    :return: candidate polytope parent states, 
    """
    candidate_polytopes, centroid_polytope = \
        planner.reachable_set_tree.polytope_tree.find_closest_polytopes(query_state, return_candiate_polytopes_only=True)
    candidate_states = [poly.t for poly in candidate_polytopes]
    centroid_state = centroid_polytope.t
    return candidate_states, centroid_state, candidate_polytopes, [centroid_polytope]

## extract all states
# state_dict = planner.state_tree.state_id_to_state
# state_list = []
# for state_id, state in state_dict.items():
#     state_list.append(state.tolist())
# state_array = np.array(state_list)

# plot states
## ----------------------------------------------------
# control flag
plot_3d_flag = False
plot_2d_flag = True
## ----------------------------------------------------
from matplotlib import pyplot as plt
from pypolycontain.visualization.visualize_2D import visualize_3D_AH_polytope_push_planning
fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax_2d = fig.add_subplot(132)
ax_time = fig.add_subplot(133)

# ax.scatter(state_array[:, 0], state_array[:, 1], state_array[:, 2], c='orange', s=4)

## extract tree structure
r3tree = planner.get_r3t_structure()
for node_id, node_state in r3tree['v'].items():
    if plot_3d_flag:
        ax.scatter(node_state[0], node_state[1], node_state[2], color='orange', s=4)
    if plot_2d_flag:
        ax_2d.scatter(node_state[0], node_state[1], color='orange', s=4)
for (node1_id, node2_id) in r3tree['e']:
    node1_state = r3tree['v'][node1_id]
    node2_state = r3tree['v'][node2_id]
    if plot_3d_flag:
        ax.plot([node1_state[0], node2_state[0]], [node1_state[1], node2_state[1]], [node1_state[2], node2_state[2]], color='grey')
    if plot_2d_flag:
        ax_2d.plot([node1_state[0], node2_state[0]], [node1_state[1], node2_state[1]], color='grey')

if plot_3d_flag:
    ax.scatter(x_init[0], x_init[1], x_init[2], color='red', marker='o', s=10)
    ax.scatter(x_goal[0], x_goal[1], x_goal[2], color='green', marker='o', s=10)
if plot_2d_flag:
    ax_2d.scatter(x_init[0], x_init[1], color='red', marker='o', s=10)
    ax_2d.scatter(x_goal[0], x_goal[1], color='green', marker='o', s=10)

query_state = np.array([0.2, 0.1, 2.5])
candidate_states, centroid_state, candidate_polytopes, centroid_polytope = test_nearest_polytope_search(planner, query_state=query_state)
candidate_states = np.atleast_2d(np.array(candidate_states).squeeze())
centroid_state = np.atleast_2d(centroid_state.squeeze())
if plot_3d_flag:
    ax.scatter(query_state[0], query_state[1], query_state[2], color='forestgreen', marker='o', s=20, label='x_query')
    try:
        ax.scatter(candidate_states[:, 0], candidate_states[:, 1], candidate_states[:, 2], color='deepskyblue', marker='o', s=20, label='x_candidate')
        visualize_3D_AH_polytope_push_planning(candidate_polytopes, planning_dyn, fig, ax,
                                            color='orange', alpha=0.2,
                                            distance_scaling_array=distance_scaling_array)
    except:
        pass
    try:
        ax.scatter(centroid_state[:, 0], centroid_state[:, 1], centroid_state[:, 2], color='coral', marker='o', s=20, label='x_centroid')
        visualize_3D_AH_polytope_push_planning(centroid_polytope, planning_dyn, fig, ax,
                                            color='red', alpha=0.2,
                                            distance_scaling_array=distance_scaling_array)
    except:
        pass

ax_time.plot(planner.time_cost['nn_search'], label='t_nn_search')
ax_time.plot(planner.time_cost['extend'], label='t_extend')
ax_time.plot(planner.time_cost['store_and_rewire'], label='t_rewire')
ax_time.plot(planner.time_cost['rewire_parent'], label='t_rewire1')
ax_time.plot(planner.time_cost['rewire_child'], label='t_rewire2')
ax_time.plot(planner.time_cost['state_tree_insert'], label='t_state_i')
ax_time.plot(planner.time_cost['set_tree_insert'], label='t_set_i')
ax_time.grid('on')

fig_data = plt.figure()
ax_state = fig_data.add_subplot(131)
ax_input = fig_data.add_subplot(132)
ax_mode = fig_data.add_subplot(133)

if success:
    final_path = planner.get_root_to_node_path(planner.goal_node)
    states = np.array(final_path[0])
    ax_state.plot(states[:,0], label='state:x')
    ax_state.plot(states[:,1], label='state:y')
    ax_state.plot(states[:,2], label='state:theta')
    inputs = np.array(final_path[1])
    ax_input.plot(inputs[:,0], label='input:fn')
    ax_input.plot(inputs[:,1], label='input:ft')
    ax_input.plot(inputs[:,2], label='input:dpsic')
    mode_ids = []
    for mode_string in final_path[2]:
        if mode_string is None:
            mode_ids.append(-1)
        else:
            mode_ids.append(planning_dyn.dynamics_mode_list.index(mode_string))
    ax_mode.plot(mode_ids, label='mode')

print('Report: mode consistency rate {0}!'.format(np.sum(planner.polytope_data['consistent'])/len(planner.polytope_data['consistent'])))

import pdb; pdb.set_trace()

timestamp = time.strftime('%Y_%m_%d_%H_%M',time.localtime(int(round(time.time()*1000))/1000))
report_path = '/home/yongpeng/research/R3T_shared/data/debug' + '/' + str(timestamp)

try:
    os.mkdir(report_path)
except:
    pass

# planner.debugger.save()
# planner.get_scene_of_planned_path(save_dir='/home/yongpeng/research/R3T_shared/data/debug/planned_path')
planner.get_plan_anim_raw_data(data_root=report_path)
planner.get_control_nom_data(data_root=report_path)
fig.legend()
fig_data.legend()
plt.show()
