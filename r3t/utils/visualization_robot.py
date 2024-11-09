from collections import deque
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import transforms, animation
import numpy as np
import os
import pypolycontain.visualization.visualize_2D as vis2D
from polytope_symbolic_system.common.utils import *

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
    def __init__(self, basic_info, visual_data) -> None:
        self.contact_basic = basic_info
        self.X_slider = visual_data['X_slider']
        self.X_pusher = visual_data['X_pusher']
        self.U_slider = visual_data['U_slider']
        self.X_obstacles = visual_data['X_obstacles']

        self.num_frames = len(self.X_slider)

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

        # plot pusher
        x_p0 = self.X_pusher[0]
        self.pusher = patches.Circle(
            x_p0, radius=Rl, facecolor='#7f7f7f', edgecolor='black'
        )

        # plot obstacles
        self.obstacles = []
        self.num_obstacles = len(self.X_obstacles[0])
        cmap = plt.cm.Pastel2
        for k in range(self.num_obstacles):
            x_o0 = self.X_obstacles[0][k]
            Xl, Yl = self.contact_basic.geom_list[k]
            R_o0 = rotation_matrix(0.0)
            bias_o0 = R_o0.dot([-Xl/2., -Yl/2.])
            new_obstacle = patches.Rectangle(
                x_o0[:2]+bias_o0[:2], Xl, Yl, angle=0.0, facecolor=cmap(k), edgecolor='black'
            )
            self.obstacles.append(new_obstacle)
        
        # add patches
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
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
        x_pi = self.X_pusher[i]
        self.pusher.set_center(x_pi)

        # render obstacles
        for k in range(self.num_obstacles):
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

def test_plot_push_planning(visualizer:PushPlanningVisualizer, vel_scale=1.0, xlim=None, ylim=None):
    """
    Plot animation of push planning
    :param visualizer: the PushPlanningVisualizer object
    :param vel_scale: the video speed scale factor
    """
    # get figure
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Planning Scene')

    # set limit
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

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
    fig2, axes2 = plt.subplots(1, 3, sharex=True)
    fig2.set_size_inches(9, 3, forward=True)
    visualizer.plot_input(axes2)

    # plot slider states
    fig3, axes3 = plt.subplots(1, 3, sharex=True)
    fig3.set_size_inches(9, 3, forward=True)
    visualizer.plot_state(axes3)

    plt.show()

if __name__ == '__main__':
    from r3t.polygon.scene import *
    # WARNING: partially initialized

    planned_path_name = '/home/yongpeng/research/R3T_shared/data/debug/real_experiment/circular_pushaway_saved_path/saved_path0'
    planned_file_name = 'planned_path.pkl'
    planned_data = pickle.load(open(os.path.join(planned_path_name, planned_file_name), 'rb'))

    # robot experiment
    # basic_info = ContactBasic(miu_list=[0.3 for i in range(3)],
    #                           geom_list=[[0.07, 0.122], [0.1, 0.102], [0.1, 0.102]],
    #                           geom_target=[0.08, 0.15, 0.01],
    #                           contact_time=0.05
    #                          )

    # robot experiment - special scenes
    scene_path_name = '/home/yongpeng/research/R3T_shared/data/debug/real_experiment'
    scene_file_name = 'pushaway_circle_obstacle_scene.pkl'
    scene_data = pickle.load(open(os.path.join(scene_path_name, scene_file_name), 'rb'))
    basic_info = ContactBasic(miu_list=scene_data['obstacle']['miu'],
                              geom_list=scene_data['obstacle']['geom'],
                              geom_target=[scene_data['target']['geom'][0], scene_data['target']['geom'][1], 0.0075],
                              contact_time=scene_data['contact']['dt']
                             )

    data = planned_data
    visualizer = PushPlanningVisualizer(basic_info=basic_info,
                                        visual_data=data)

    # TEST ANIMATION
    # test_plot_push_planning(visualizer, vel_scale=1.0, xlim=[0.26, 0.80], ylim=[-0.09, 0.45])
    test_plot_push_planning(visualizer, vel_scale=1.0, xlim=[0.23, 0.77], ylim=[-0.20, 0.48])
    
