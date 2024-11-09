from collections import deque
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import transforms, animation
import numpy as np
import pypolycontain.visualization.visualize_2D as vis2D
from polytope_symbolic_system.common.utils import *

## utility functions copied from (planar-)rrt-algorithms
## ----------------------------------------
def angle_limit(angle):
    # soft limit yaw angle to [-pi, pi]
    return np.arctan2(np.sin(angle), np.cos(angle))

def interpolate_pose(pose0, pose1, geom, p):
    """
    Get the linear interpolated pose between start and end
    :param start: start pose
    :param end: end pose
    :param p: coefficient (result=start*(1-p)+end*p)
    """
    p_start, p_end = np.zeros((2, 2)), np.zeros((2, 2))
    p_start[0, :], p_end[0, :] = pose0[:2], pose1[:2]
    p_start[1, :] = pose0[:2] + np.array([np.cos(pose0[2]), np.sin(pose0[2])]) * geom[0] * 0.5
    p_end[1, :] = pose1[:2] + np.array([np.cos(pose1[2]), np.sin(pose1[2])]) * geom[0] * 0.5
    A = 2 * (p_end - p_start)
    b = np.sum(np.power(p_end, 2) - np.power(p_start, 2), axis=1)

    pose0, pose1 = np.array(pose0), np.array(pose1)

    if np.linalg.matrix_rank(A) < 2:
        pose_interp = pose0
        pose_interp[:2] = pose0[:2] * (1-p) + pose1[:2] * p
        return pose_interp
    else:
        center = np.linalg.inv(A) @ b
        # theta = angle_limit(end[2] - start[2])
        theta = angle_limit(angle_diff(pose0[2], pose1[2]))
        pose_interp = pose0
        rot_matrix = rotation_matrix(theta*p)
        pose_interp[:2] = center + rot_matrix.dot(pose0[:2]-center)
        pose_interp[2] = angle_limit(pose0[2]+theta*p)
        return pose_interp

def distance_between_pose(pose0, pose1, geom):
    """
    Convert the change between pose to revolution w.r.t. fixed point
    :param start: current pose
    :param end: goal pose
    :return: Revolute (center: (x, y), angle: theta)
    """
    p_start, p_end = np.zeros((2, 2)), np.zeros((2, 2))
    p_start[0, :], p_end[0, :] = pose0[:2], pose1[:2]
    p_start[1, :] = pose0[:2] + np.array([np.cos(pose0[2]), np.sin(pose0[2])]) * geom[0] * 0.5
    p_end[1, :] = pose1[:2] + np.array([np.cos(pose1[2]), np.sin(pose1[2])]) * geom[0] * 0.5
    A = 2 * (p_end - p_start)
    b = np.sum(np.power(p_end, 2) - np.power(p_start, 2), axis=1)

    if np.linalg.matrix_rank(A) < 2:
        return np.linalg.norm(pose0[:2]-pose1[:2])
    else:
        center = np.linalg.inv(A) @ b
        # theta = angle_limit(end[2] - start[2])
        theta = angle_diff(pose0[2], pose1[2])
        radius = np.linalg.norm(pose0[0:2] - center[0:2])
        return radius*theta
    
def get_pusher_abs_pos(x_slider, rel_pos):
    Ri = rotation_matrix(x_slider[2])
    return x_slider[:2] + Ri.dot(rel_pos.flatten())
    
class Revolute(object):
    def __init__(self, finite, x, y, theta, radius) -> None:
        self.finite = finite  # False=trans(lation) only or True=revol(ution)
        self.x = x
        self.y = y
        self.theta = theta
        self.radius = radius
    
def pose2steer(start, end, geom):
    """
    Convert the change between pose to revolution w.r.t. fixed point
    :param start: current pose
    :param end: goal pose
    :return: Revolute (center: (x, y), angle: theta)
    """
    p_start, p_end = np.zeros((2, 2)), np.zeros((2, 2))
    p_start[0, :], p_end[0, :] = start[:2], end[:2]
    p_start[1, :] = start[:2] + np.array([np.cos(start[2]), np.sin(start[2])]) * geom[0] * 0.5
    p_end[1, :] = end[:2] + np.array([np.cos(end[2]), np.sin(end[2])]) * geom[0] * 0.5
    # p_center = np.concatenate((np.expand_dims(p_start[0, :], 0), np.expand_dims(p_end[0, :], 0)), axis=0)
    # p_bound = np.concatenate((np.expand_dims(p_start[1, :], 0), np.expand_dims(p_end[1, :], 0)), axis=0)
    A = 2 * (p_end - p_start)
    b = np.sum(np.power(p_end, 2) - np.power(p_start, 2), axis=1)

    if np.linalg.matrix_rank(A) < 2:
        revol = Revolute(finite=False, x=0, y=0, theta=0, radius=0)
    else:
        center = np.linalg.inv(A) @ b
        # theta = angle_limit(end[2] - start[2])
        theta = angle_diff(start[2], end[2])
        radius = np.linalg.norm(start[0:2] - center[0:2])
        revol = Revolute(finite=True, x=center[0], y=center[1], theta=theta, radius=radius)

    return revol
    
def get_pusher_rel_pos(start, end, geom, ab_ratio=1/726.136, miu=0.3, rl=0.01):
    """
    Check if a Revolute satisfies differential flatness constraints
    If true, compute the contact point and force direction
    :param start: starting pose
    :param end: ending pose
    :return: True if Revolute is feasible, and False if Revolute is unfeasible
    """
    revol = pose2steer(start, end, geom)
    Xrev = np.array([revol.x, revol.y])
    Xrev = np.linalg.inv(rotation_matrix(start[2])) @ (Xrev - start[:2])
    Xc, Yc = Xrev[0] + np.sign(Xrev[0])*1e-10, Xrev[1] + np.sign(Xrev[1])*1e-10  # ROC coordinates
    Kc = Yc / Xc
    
    # forwarding direction
    forward_dir = rotation_matrix(start[2]).T @ (np.array(end[:2]) - np.array(start[:2]))
    
    # check feasible contact point on all 4 faces
    x_lim, y_lim = 0.5 * geom[0], 0.5 * geom[1]
    force_dirs, contact_pts = [], []
    
    # +X face
    y0 = (-ab_ratio - x_lim * Xc) / Yc
    if (-y_lim <= y0 <= y_lim) and (Kc <= -1 / miu or Kc >= 1 / miu):
        contact_pts.append([x_lim+rl, y0])
        if Yc >= 0:
            force_dirs.append([-Yc, Xc])
        else:
            force_dirs.append([Yc, -Xc])
    # -X face
    y0 = (-ab_ratio - (-x_lim) * Xc) / Yc
    if (-y_lim <= y0 <= y_lim) and (Kc <= -1 / miu or Kc >= 1 / miu):
        contact_pts.append([-x_lim-rl, y0])
        if Yc >= 0:
            force_dirs.append([Yc, -Xc])
        else:
            force_dirs.append([-Yc, Xc])
    # +Y face
    x0 = (-ab_ratio - y_lim * Yc) / Xc
    if (-x_lim <= x0 <= x_lim) and (-miu <= Kc <= miu):
        contact_pts.append([x0, y_lim+rl])
        if Xc >= 0:
            force_dirs.append([Yc, -Xc])
        else:
            force_dirs.append([-Yc, Xc])
    # -Y face
    x0 = (-ab_ratio - (-y_lim) * Yc) / Xc
    if (-x_lim <= x0 <= x_lim) and (-miu <= Kc <= miu):
        contact_pts.append([x0, -y_lim-rl])
        if Xc >= 0:
            force_dirs.append([-Yc, Xc])
        else:
            force_dirs.append([Yc, -Xc])
    
    # the pusher's contact force and the slider's forwarding direction keeps acute angle
    idx = np.where((np.array(force_dirs).reshape(-1, 2) @ forward_dir) > 0)[0]
    contact_pts = np.array(contact_pts).T
    return contact_pts[:, idx]
## ----------------------------------------

def visualize_node_tree_2D(rrt, fig=None, ax=None, s=1, linewidths = 0.25, show_path_to_goal=False, goal_override=None, dims=[0,1]):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    node_queue = deque([rrt.root_node])
    lines = []
    i = 0
    while node_queue:
        i+=1
        node = node_queue.popleft()
        #handle indexing
        if dims:
            state = np.ndarray.flatten(node.state)[dims]
        else:
            state = np.ndarray.flatten(node.state)
        #handle goal
        if goal_override is not None and node==rrt.goal_node:
            lines.append([state, goal_override])
        elif node == rrt.root_node or node==rrt.goal_node:
            pass
        else:
            for i in range(len(node.true_dynamics_path)-1):
                # handle indexing
                if dims:
                    lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[dims],
                                   np.ndarray.flatten(node.true_dynamics_path[i + 1])[dims]])
                else:
                    lines.append([np.ndarray.flatten(node.true_dynamics_path[i]),
                                  np.ndarray.flatten(node.true_dynamics_path[i + 1])])

        if node.children is not None:
            # print(len(node.children))
            node_queue.extend(list(node.children))
        ax.scatter(*state, c='gray', s=s)
    if show_path_to_goal:
        goal_lines = []
        node = rrt.goal_node
        if goal_override is not None:
            #FIXME: make this cleaner
            goal_lines.append([goal_override[dims], np.ndarray.flatten(node.parent.state)[dims]])
            node = node.parent
        else:
            goal_lines.append([rrt.goal_node.state[dims], np.ndarray.flatten(node.parent.state)[dims]])
            node = node.parent
        while node.parent is not None:
            for i in range(len(node.true_dynamics_path)-1):
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[dims], np.ndarray.flatten(node.true_dynamics_path[i+1])[dims]])
            assert(node in node.parent.children)
            # hack for 1D hopper visualization
            if node.parent==rrt.root_node:
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[-1])[dims], np.ndarray.flatten(node.parent.state)[dims]])
            node = node.parent
        line_colors = np.full(len(lines), 'gray')
        line_widths = np.full(len(lines), linewidths)
        goal_line_colors = np.full(len(goal_lines), 'cyan')
        goal_line_widths= np.full(len(goal_lines), linewidths*4)
        lines.extend(goal_lines)
        all_colors = np.hstack([line_colors, goal_line_colors])
        all_widths = np.hstack([line_widths, goal_line_widths])
        lc = mc.LineCollection(lines, linewidths=all_widths, colors=all_colors)
        ax.add_collection(lc)
    else:
        lc = mc.LineCollection(lines, linewidths=linewidths, colors='gray')
        ax.add_collection(lc)
    return fig, ax

def visualize_node_tree_2D_old(rrt, fig=None, ax=None, s=1, linewidths = 0.25, show_path_to_goal=False, goal_override=None, dims=None):
    """
    Deprecated function for visualizing RRT trees with no true dynamics path (prior to 9/9/19)
    :param rrt:
    :param fig:
    :param ax:
    :param s:
    :param linewidths:
    :param show_path_to_goal:
    :param goal_override:
    :param dims:
    :return:
    """
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    node_queue = deque([rrt.root_node])
    lines = []
    i = 0
    while node_queue:
        i+=1
        node = node_queue.popleft()
        if dims:
            state = np.ndarray.flatten(node.state)[dims]
        else:
            state = np.ndarray.flatten(node.state)
        if node.children is not None:
            # print(len(node.children))
            node_queue.extend(list(node.children))
            for child in node.children:
                if dims:
                    child_state = np.ndarray.flatten(child.state)[dims]
                else:
                    child_state = np.ndarray.flatten(child.state)
                # don't plot the goal if goal override is on
                if goal_override is not None and child==rrt.goal_node:
                    lines.append([state, goal_override])
                else:
                    lines.append([state, child_state])
        ax.scatter(*state, c='gray', s=s)
    if show_path_to_goal:
        goal_lines = []
        node = rrt.goal_node
        if goal_override is not None:
            if dims:
                goal_lines.append([goal_override[dims], np.ndarray.flatten(node.parent.state)[dims]])
            else:
                goal_lines.append([goal_override, np.ndarray.flatten(node.parent.state)])
            node = node.parent
        while node.parent is not None:
            for i in range(len(node.true_dynamics_path)-1):
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[dims], np.ndarray.flatten(node.true_dynamics_path[i+1])[dims]])
            assert(node in node.parent.children)
            # hack for 1D hopper visualization
            if node.parent==rrt.root_node:
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[-1])[dims], np.ndarray.flatten(node.parent.state)[dims]])

            node = node.parent
        line_colors = np.full(len(lines), 'gray')
        line_widths = np.full(len(lines), linewidths)
        goal_line_colors = np.full(len(goal_lines), 'cyan')
        goal_line_widths= np.full(len(goal_lines), linewidths*4)
        lines.extend(goal_lines)
        all_colors = np.hstack([line_colors, goal_line_colors])
        all_widths = np.hstack([line_widths, goal_line_widths])
        lc = mc.LineCollection(lines, linewidths=all_widths, colors=all_colors)
        ax.add_collection(lc)
    else:
        lc = mc.LineCollection(lines, linewidths=linewidths, colors='gray')
        ax.add_collection(lc)
    return fig, ax

def visualize_node_tree_hopper_2D(rrt, fig=None, ax=None, s=1, linewidths = 0.25, show_path_to_goal=False, goal_override=None,\
                                  dims=[0,1], show_body_attitude='goal', scaling_factor=1, draw_goal =False, ground_height_function = None, downsample=3):
    """

    :param rrt:
    :param fig:
    :param ax:
    :param s:
    :param linewidths:
    :param show_path_to_goal:
    :param goal_override:
    :param dims:
    :param show_body_attitude: 'goal', 'all', or nothing
    :param scaling_factor:
    :param draw_goal:
    :param ground_height_function:
    :return:
    """
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    node_queue = deque([rrt.root_node])
    lines = []
    i = 0
    nodes_to_visualize = [rrt.root_node]

    while node_queue:
        i+=1
        node = node_queue.popleft()
        #handle indexing
        if dims:
            state = np.ndarray.flatten(node.state)[dims]
        else:
            state = np.ndarray.flatten(node.state)
        #handle goal
        if goal_override is not None and node==rrt.goal_node:
            lines.append([state, goal_override])
        elif node==rrt.goal_node:
            pass
        else:
            for i in range(len(node.true_dynamics_path)-1):
                # handle indexing
                if dims:
                    lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[dims],
                                   np.ndarray.flatten(node.true_dynamics_path[i + 1])[dims]])
                else:
                    lines.append([np.ndarray.flatten(node.true_dynamics_path[i]),
                                  np.ndarray.flatten(node.true_dynamics_path[i + 1])])
        if node.parent == rrt.root_node:
            lines.append([np.ndarray.flatten(node.parent.state)[dims],
                          np.ndarray.flatten(node.true_dynamics_path[0])[dims]])
        if node.children is not None:
            # print(len(node.children))
            nodes_to_visualize.extend(list(node.children))
            node_queue.extend(list(node.children))
        ax.scatter(*state, c='gray', s=s)

    #
    # while node_queue:
    #     i+=1
    #     node = node_queue.popleft()
    #     #handle indexing
    #     if dims:
    #         state = np.ndarray.flatten(node.state)[dims]
    #     else:
    #         state = np.ndarray.flatten(node.state)
    #     for i in range(len(node.true_dynamics_path) - 1):
    #         # handle indexing
    #         if dims:
    #             lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[dims],
    #                           np.ndarray.flatten(node.true_dynamics_path[i + 1])[dims]])
    #         else:
    #             lines.append([np.ndarray.flatten(node.true_dynamics_path[i]),
    #                           np.ndarray.flatten(node.true_dynamics_path[i + 1])])
    #
    #     if node.children is not None:
    #         # print(len(node.children))
    #         node_queue.extend(list(node.children))
    #         nodes_to_visualize.extend(list(node.children))
    #         for child in node.children:
    #             if dims:
    #                 child_state = np.ndarray.flatten(child.state)[dims]
    #             else:
    #                 child_state = np.ndarray.flatten(child.state)
    #             # # don't plot the goal if goal override is on
    #             # if goal_override is not None and child==rrt.goal_node:
    #             #     lines.append([state, goal_override])
    #             # else:
    #             #     lines.append([state, child_state])
    #     ax.scatter(*state, c='gray', s=s)

    if show_path_to_goal:
        goal_lines = []
        node = rrt.goal_node
        if not draw_goal:
            node = node.parent
        if goal_override is not None:
            goal_lines.append([goal_override[dims], np.ndarray.flatten(node.parent.state)[dims]])
        while node.parent is not None:
            for i in range(len(node.true_dynamics_path)-1):
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[dims], np.ndarray.flatten(node.true_dynamics_path[i+1])[dims]])
            assert(node in node.parent.children)
            # hack for 1D hopper visualization
            if node.parent==rrt.root_node:
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[-1])[dims], np.ndarray.flatten(node.parent.state)[dims]])
            node = node.parent
        line_colors = np.full(len(lines), 'gray')
        line_widths = np.full(len(lines), linewidths)
        goal_line_colors = np.full(len(goal_lines), 'cyan')
        goal_line_widths= np.full(len(goal_lines), linewidths*4)
        lines.extend(goal_lines)
        all_colors = np.hstack([line_colors, goal_line_colors])
        all_widths = np.hstack([line_widths, goal_line_widths])
        lc = mc.LineCollection(lines, linewidths=all_widths, colors=all_colors)
        ax.add_collection(lc)
    else:
        lc = mc.LineCollection(lines, linewidths=linewidths, colors='gray')
        ax.add_collection(lc)
    if show_body_attitude =='goal':
        node = rrt.goal_node
        skip = downsample
        if not draw_goal and node is not None:
            node = node.parent
        while node is not None:
            if node.parent is None:
                #reached root
                #plot root
                fig, ax = hopper_plot(node.state, fig, ax, alpha=0.2, scaling_factor=scaling_factor)
                node = node.parent
                break
            if skip<downsample:
                #skipping
                node = node.parent
                skip+=1
            else:
                #plot
                fig, ax = hopper_plot(node.state, fig, ax, alpha=0.25, scaling_factor=scaling_factor)
                skip = 0
    elif show_body_attitude=='all':
        for i, n in enumerate(nodes_to_visualize):
            # fig, ax = hopper_plot(n.state, fig, ax, alpha=0.5/len(nodes_to_visualize)*i+0.1)
            fig, ax = hopper_plot(n.state, fig, ax, alpha=0.15, scaling_factor=scaling_factor)
    return fig, ax

def visualize_projected_ND_polytopic_tree(rrt, dim1, dim2, fig=None, ax=None, s=10, linewidths =1., show_path_to_goal=False, goal_override=None, polytope_alpha=0.06):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    node_queue = deque([rrt.root_node])
    lines = []
    i = 0
    polytopes_list = []
    while node_queue:
        i+=1
        node = node_queue.popleft()
        # give a separate color for all polytopes in one node
        random_color = np.random.rand(3)
        for p in node.reachable_set.polytope_list:
            p.color = random_color
            polytopes_list.append(p)
        #handle indexing
        state = np.ndarray.flatten(node.state)[np.asarray([dim1, dim2])]
        # plot the polytope
        if goal_override is not None and node==rrt.goal_node:
            lines.append([state, goal_override])
        elif node == rrt.root_node or node==rrt.goal_node:
            pass
        else:
            try:
                for i in range(len(node.true_dynamics_path) - 1):
                    # handle indexing
                    lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[np.asarray([dim1, dim2])],
                                  np.ndarray.flatten(node.true_dynamics_path[i + 1])[np.asarray([dim1, dim2])]])
            except:
                lines.append([np.ndarray.flatten(node.path_from_parent.state_from)[np.asarray([dim1, dim2])],
                              np.ndarray.flatten(node.path_from_parent.state_to)[np.asarray([dim1, dim2])]])
        if node.children is not None:
            # print(len(node.children))
            node_queue.extend(list(node.children))
        ax.scatter(*state, c='black', s=s, alpha=1)
    if show_path_to_goal and rrt.goal_node:
        goal_lines = []
        node = rrt.goal_node
        if goal_override is not None:
            #FIXME: make this cleaner
            goal_lines.append([goal_override[np.asarray([dim1, dim2])], np.ndarray.flatten(node.parent.state)[np.asarray([dim1, dim2])]])
            node = node.parent
        else:
            goal_lines.append([rrt.goal_node.state[np.asarray([dim1, dim2])], np.ndarray.flatten(node.parent.state)[np.asarray([dim1, dim2])]])
            node = node.parent
        while node.parent is not None:
            for i in range(len(node.true_dynamics_path)-1):
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[i])[np.asarray([dim1, dim2])], np.ndarray.flatten(node.true_dynamics_path[i+1])[np.asarray([dim1, dim2])]])
            assert(node in node.parent.children)
            # hack for 1D hopper visualization
            if node.parent==rrt.root_node:
                goal_lines.append([np.ndarray.flatten(node.true_dynamics_path[-1])[np.asarray([dim1, dim2])], np.ndarray.flatten(node.parent.state)[np.asarray([dim1, dim2])]])
            node = node.parent
        line_colors = np.full(len(lines), 'gray', alpha=1)
        line_widths = np.full(len(lines), linewidths)
        goal_line_colors = np.full(len(goal_lines), 'cyan')
        goal_line_widths= np.full(len(goal_lines), linewidths*4)
        lines.extend(goal_lines)
        all_colors = np.hstack([line_colors, goal_line_colors])
        all_widths = np.hstack([line_widths, goal_line_widths])
        lc = mc.LineCollection(lines, linewidths=all_widths, colors=all_colors)
        ax.add_collection(lc)
    else:
        lc = mc.LineCollection(lines, linewidths=linewidths, colors='black', alpha=1)
        ax.add_collection(lc)

    vis2D.visualize_ND_AH_polytope(polytopes_list, dim1, dim2, fig=fig, ax=ax, alpha=polytope_alpha)

    return fig, ax

class PushPlanningVisualizer:
    def __init__(self, basic_info, visual_data, X0_obstacles, pose_interp) -> None:
        self.contact_basic = basic_info
        self.X_slider = visual_data['X_slider']
        if 'X_pusher' in visual_data.keys():
            self.X_pusher = visual_data['X_pusher']
        else:
            self.X_pusher = []

        # interpolate poses
        interpolation_distance = 0.005
        if pose_interp:
            X_slider_interpolated = []
            for i in range(len(self.X_slider)-1):
                pusher_rel_pos = get_pusher_rel_pos(self.X_slider[i], self.X_slider[i+1], \
                                                    geom=self.contact_basic.geom_target)
                X_slider_interpolated.append(self.X_slider[i])
                self.X_pusher.append(get_pusher_abs_pos(np.array(self.X_slider[i]), pusher_rel_pos).tolist())

                path_seg_length = distance_between_pose(self.X_slider[i], self.X_slider[i+1], \
                                                        geom=self.contact_basic.geom_target)
                num_interp_steps = round(abs(path_seg_length)/interpolation_distance)
                for p in np.linspace(0.0, 1.0, max(5, num_interp_steps))[1:-1]:
                    new_pose_interp = interpolate_pose(pose0=self.X_slider[i], \
                                                       pose1=self.X_slider[i+1], \
                                                       geom=self.contact_basic.geom_target, \
                                                       p=p)
                    X_slider_interpolated.append(list(new_pose_interp))
                    self.X_pusher.append(get_pusher_abs_pos(np.array(new_pose_interp), pusher_rel_pos).tolist())
            X_slider_interpolated.append(self.X_slider[-1])
            self.X_pusher.append(get_pusher_abs_pos(np.array(self.X_slider[-1]), pusher_rel_pos).tolist())

            self.X_slider = X_slider_interpolated.copy()

        self.num_frames = len(self.X_slider)

        if 'U_slider' in visual_data.keys():
            self.U_slider = visual_data['U_slider']
        else:
            self.U_slider = None
        if 'X_obstacles' in visual_data.keys():
            self.X_obstacles = visual_data['X_obstacles']
        else:
            self.X_obstacles = [X0_obstacles] * self.num_frames

    def set_patches(self, ax):
        """
        Set patches of slider, pusher and obstacles
        :param ax: the Axes object
        """
        # plot slider
        Xl, Yl, Rl = self.contact_basic.geom_target
        x_s0 = self.X_slider[0]
        R_s0 = rotation_matrix(0.0)
        bias_s0 = R_s0.dot([-Xl/2., -Yl/2.])
        self.slider = patches.Rectangle(
            x_s0[:2]+bias_s0[:2], Xl, Yl, angle=0.0, facecolor='#1f77b4', edgecolor='black'
        )
        ax.add_patch(self.slider)

        # plot pusher
        if self.X_pusher is not None:
            x_p0 = self.X_pusher[0]
            self.pusher = patches.Circle(
                x_p0, radius=Rl, facecolor='#7f7f7f', edgecolor='black'
            )
            ax.add_patch(self.pusher)

        # plot obstacles
        if self.X_obstacles is not None:
            self.obstacles = []
            self.num_obstacles = len(self.X_obstacles[0])
            cmap = plt.cm.Pastel2
            for k in range(self.num_obstacles):
                if k==1:
                    polygon_facecolor = "grey"
                else:
                    polygon_facecolor = cmap(k)
                if self.contact_basic.is_rect_flag[k]:
                    x_o0 = self.X_obstacles[0][k]
                    Xl, Yl = self.contact_basic.geom_list[k]
                    R_o0 = rotation_matrix(0.0)
                    bias_o0 = R_o0.dot([-Xl/2., -Yl/2.])
                    new_obstacle = patches.Rectangle(
                        x_o0[:2]+bias_o0[:2], Xl, Yl, angle=0.0, facecolor=polygon_facecolor, edgecolor='black'
                    )
                    self.obstacles.append(new_obstacle)
                else:
                    coords_before_transform = self.contact_basic.geom_list[k]
                    R_o0 = rotation_matrix(0.0)
                    coords_after_transform = np.array([0.0, 0.0]) + R_o0.dot(np.array(coords_before_transform).T).T
                    new_obstacle = patches.Polygon(
                        xy=coords_after_transform, facecolor=polygon_facecolor, edgecolor='black'
                    )
                    self.obstacles.append(new_obstacle)
            for k in range(self.num_obstacles):
                ax.add_patch(self.obstacles[k])

    def animate(self, i, ax):
        """
        Render the ith frame
        :param i: frame index
        :param ax: the Axes object
        """
        trans_ax = ax.transData

        # render slider
        Xl, Yl, Rl = self.contact_basic.geom_target
        x_si = self.X_slider[i]
        R_si = rotation_matrix(x_si[2])
        bias_si = R_si.dot([-Xl/2., -Yl/2.])

        bottom_left_coords = x_si[:2] + bias_si[:2]
        bottom_left_coords = trans_ax.transform(bottom_left_coords)
        trans_si = transforms.Affine2D().rotate_around(
            bottom_left_coords[0], bottom_left_coords[1], x_si[2]
        )
        self.slider.set_transform(trans_ax+trans_si)
        self.slider.set_xy([x_si[0]+bias_si[0], x_si[1]+bias_si[1]])

        # render pusher
        if self.X_pusher is not None:
            x_pi = self.X_pusher[i]
            self.pusher.set_center(x_pi)

        # render obstacles
        if self.X_obstacles is not None:
            for k in range(self.num_obstacles):
                if self.contact_basic.is_rect_flag[k]:
                    x_oi = self.X_obstacles[i][k]
                    Xl, Yl = self.contact_basic.geom_list[k]
                    R_oi = rotation_matrix(x_oi[2])
                    bias_oi = R_oi.dot([-Xl/2., -Yl/2.])

                    bottom_left_coords = x_oi[:2] + bias_oi[:2]
                    bottom_left_coords = trans_ax.transform(bottom_left_coords)
                    trans_oi = transforms.Affine2D().rotate_around(
                        bottom_left_coords[0], bottom_left_coords[1], x_oi[2]
                    )
                    self.obstacles[k].set_transform(trans_ax+trans_oi)
                    self.obstacles[k].set_xy([x_oi[0]+bias_oi[0], x_oi[1]+bias_oi[1]])
                else:
                    x_oi = self.X_obstacles[i][k]
                    coords_before_transform = self.contact_basic.geom_list[k]
                    R_oi = rotation_matrix(x_oi[2])
                    coords_after_transform = x_oi[:2] + R_oi.dot(np.array(coords_before_transform).T).T
                    self.obstacles[k].set_xy(coords_after_transform)
        
        return []

    def plot_input(self, axes):
        """
        Plot the control input
        :param axes: the Axes objects
        """
        t_u = np.linspace(0., self.contact_basic.contact_time*(self.num_frames-2), self.num_frames-1)
        U_array = np.array(self.U_slider)

        # plot normal force
        axes[0].plot(t_u, U_array[:, 0], label='f_n')
        # plot force limit
        axes[0].plot(t_u, 0.3*np.ones(self.num_frames-1), color='red', linestyle='--', label='f_max')
        axes[0].plot(t_u, -0.3*np.ones(self.num_frames-1), color='green', linestyle='--', label='f_min')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels)
        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('normal force (N)')
        axes[0].grid('on')

        # plot tangent force
        axes[1].plot(t_u, U_array[:, 1], label='f_t')
        # plot force limit
        axes[1].plot(t_u, 0.3*np.ones(self.num_frames-1), color='red', linestyle='--', label='f_max')
        axes[1].plot(t_u, -0.3*np.ones(self.num_frames-1), color='green', linestyle='--', label='f_min')
        axes[1].plot(t_u, self.contact_basic.miu_pusher_slider*U_array[:, 0], color='lightcoral', linestyle='--', label='f_t_max')
        axes[1].plot(t_u, -self.contact_basic.miu_pusher_slider*U_array[:, 0], color='lightgreen', linestyle='--', label='f_t_min')
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels)
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('tangent force (N)')
        axes[1].grid('on')

        # plot dpsic
        axes[2].plot(t_u, U_array[:, 2], label='dpsic')
        # plot dpsic limit
        axes[2].plot(t_u, 3.0*np.ones(self.num_frames-1), color='red', linestyle='--', label='dpsic_max')
        axes[2].plot(t_u, -3.0*np.ones(self.num_frames-1), color='red', linestyle='--', label='dpsic_min')
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles, labels)
        axes[2].set_xlabel('time (s)')
        axes[2].set_ylabel('pusher velocity (rad/s)')
        axes[2].grid('on')

    def plot_state(self, axes):
        """
        Plot the slider states
        :param axes: the Axes objects
        """
        t_x = np.linspace(0., self.contact_basic.contact_time*(self.num_frames-1), self.num_frames)
        X_array = np.array(self.X_slider)
        X_pusher = np.array(self.X_pusher)
        X_pusher_rel = []
        for i in range(self.num_frames):
            R_i = rotation_matrix(X_array[i, 2])
            X_pusher_rel.append(R_i.T.dot(X_pusher[i]-X_array[i,:2]))
        X_pusher_rel = np.array(X_pusher_rel)

        # plot x and y
        axes[0].plot(t_x, X_array[:, 0], label='x')
        axes[0].plot(t_x, X_array[:, 1], label='y')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels)
        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('position (m)')
        axes[0].grid('on')

        # plot azimuth angle
        axes[1].plot(t_x, restrict_angle_in_unit_circle(X_array[:, 2]), label='theta')
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels)
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('yaw (rad)')
        axes[1].grid('on')

        # plot psic
        axes[2].plot(t_x, X_pusher_rel[:, 0], label='px')
        axes[2].plot(t_x, X_pusher_rel[:, 1], label='py')
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles, labels)
        axes[2].set_xlabel('time (s)')
        axes[2].set_ylabel('psic (rad)')
        axes[2].grid('on')

def test_plot_push_planning(visualizer:PushPlanningVisualizer, vel_scale=1.0, enlarge_canvas=False):
    """
    Plot animation of push planning
    :param visualizer: the PushPlanningVisualizer object
    :param vel_scale: the video speed scale factor
    """
    # get figure
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Planning Scene')

    # set limit
    if enlarge_canvas:
        ax.set_xlim([-0.05, 0.55])
        ax.set_ylim([-0.05, 0.55])
    else:
        ax.set_xlim([0.0, 0.5])
        ax.set_ylim([0.0, 0.5])

    x_slider_arr = np.array(visualizer.X_slider)
    ax.plot(x_slider_arr[:, 0], x_slider_arr[:, 1], linewidth=2, linestyle='--', color='lightcoral')

    # for robot experiment
    # ax.set_xlim([0.3, 0.9])
    # ax.set_ylim([-0.3, 0.3])

    # other settings
    ax.set_autoscale_on(False)
    ax.grid('on')
    ax.set_aspect('equal', 'box')
    ax.set_title('Push Planning Result Animation')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m))')

    # set patch
    visualizer.set_patches(ax)

    # animation
    anim = animation.FuncAnimation(
        fig,
        visualizer.animate,
        frames=visualizer.num_frames,
        fargs=(ax,),
        interval=0.05*1000*vel_scale,
        blit=True,
        repeat=False
    )

    anim.save('./video/R3T_contact_planning.mp4', fps=25, extra_args=['-vcodec', 'mpeg4'])

    # plot control inputs
    # fig2, axes2 = plt.subplots(1, 3, sharex=True)
    # fig2.set_size_inches(9, 3, forward=True)
    # visualizer.plot_input(axes2)

    # plot slider states
    # fig3, axes3 = plt.subplots(1, 3, sharex=True)
    # fig3.set_size_inches(9, 3, forward=True)
    # visualizer.plot_state(axes3)

    plt.show()

if __name__ == '__main__':
    from r3t.polygon.scene import *
    # WARNING: partially initialized
    # test_scene_0
    # basic_info = ContactBasic(miu_list=[0.3 for i in range(3)],
    #                           geom_list=[[0.07, 0.12] for i in range(3)],
    #                           geom_target=[0.07, 0.12, 0.01],
    #                           contact_time=0.05
    #                          )

    ## ???
    # test_scene_1 (deprecated)
    # basic_info = ContactBasic(miu_list=[0.3 for i in range(5)],
    #                           geom_list=[[0.06, 0.12], [0.12, 0.12], [0.06, 0.12], [0.06, 0.12], [0.06, 0.12]],
    #                           geom_target=[0.07, 0.12, 0.01],
    #                           contact_time=0.05
    #                          )

    # test_scene_1
    basic_info = ContactBasic(miu_list=[0.3 for i in range(5)],
                              geom_list=[[0.06, 0.12], \
                                         [[0.03, 0.03*np.sqrt(3)], [0.06, 0], [0.03, -0.03*np.sqrt(3)], [-0.03, -0.03*np.sqrt(3)], [-0.06, 0], [-0.03, 0.03*np.sqrt(3)]], \
                                         [0.06, 0.12], \
                                         [0.06, 0.12], \
                                         [0.06, 0.12]],
                              geom_target=[0.07, 0.12, 0.01],
                              contact_time=0.05,
                              is_rect_flag=[True, False, True, True, True]
                             )
    X0_obstacles = np.array([[0.12, 0.20, 0.75*np.pi],
                             [0.25, 0.30, 0.0*np.pi],
                             [0.38, 0.20, 0.25*np.pi],
                             [0.12, 0.40, -0.75*np.pi],
                             [0.38, 0.40, -0.25*np.pi]])

    # robot experiment
    # basic_info = ContactBasic(miu_list=[0.3 for i in range(3)],
    #                           geom_list=[[0.07, 0.122], [0.1, 0.102], [0.1, 0.102]],
    #                           geom_target=[0.08, 0.15, 0.01],
    #                           contact_time=0.05
    #                          )

    # timestamp = 'segmented_path/2023_02_05_11_37'
    # data = pickle.load(open('/home/yongpeng/research/R3T_shared/data/debug/{0}/output.pkl'.format(timestamp), 'rb'))
    # data = pickle.load(open('/home/yongpeng/research/R3T_shared/data/exp/2023_07_07_01_13/0.pkl', 'rb'))

    # data = pickle.load(open('/home/yongpeng/research/R3T_shared/data/exp/new_model/scene1/2023_07_07_15_27_(2)/3.pkl', 'rb'))
    data = pickle.load(open('/home/yongpeng/research/rrt-algorithms/data/exp/search_space_0.0_0.5_0.0_0.5/scene_1_10_round_#1/4.pkl', 'rb'))
    
    visualizer = PushPlanningVisualizer(basic_info=basic_info,
                                        visual_data=data,
                                        X0_obstacles=X0_obstacles,
                                        pose_interp=True)

    # TEST ANIMATION
    test_plot_push_planning(visualizer, vel_scale=1.0, enlarge_canvas=True)
    
