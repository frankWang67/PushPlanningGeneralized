import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches, transforms
from matplotlib.lines import Line2D

from r3t.polygon.scene import load_planning_scene_from_file

PUSHER_R = 0.0075

r3t_root_dir = os.environ.get("R3T_HOME")
cmap = plt.cm.Pastel2

planning_scene_path = os.path.join(r3t_root_dir, "data", "wshf", "2025_05_02_22_19_backup")
planning_scene_pkl  = os.path.join(planning_scene_path, "scene.pkl")
planned_file_name = 'planned_path.pkl'
planned_data = pickle.load(open(os.path.join(planning_scene_path, planned_file_name), 'rb'))

X_slider = planned_data['X_slider']
X_pusher = planned_data['X_pusher']
X_obstacles = planned_data['X_obstacles']
N = len(X_slider)
idxs = np.linspace(0, N-1, 10, dtype=int)

scene, contact_basic = load_planning_scene_from_file(planning_scene_pkl)

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Planned Path')

ax.set_autoscale_on(False)
ax.grid('on')
ax.set_aspect('equal', 'box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
xlim = [-0.15, 0.35]
ylim = [-0.30, 0.30]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

for i in idxs:
    alpha = 1.0 if i == 0 else 0.5
    linestyle = '-' if i == 0 else '--'
    
    x_slider = X_slider[i]
    x_pusher = X_pusher[i]
    x_obstacle = X_obstacles[i]
    types = scene.types
    x_goal = scene.goal_state

    # plot slider
    x_s0 = x_slider
    contour_path = Path(contact_basic.curve_target.pt_samples)
    trans_ax = ax.transData
    transf_s0 = transforms.Affine2D().translate(x_s0[0], x_s0[1]).rotate_around(x_s0[0], x_s0[1], x_s0[2])
    slider = patches.PathPatch(
        contour_path, transform=transf_s0+trans_ax, facecolor=cmap(2), edgecolor='black', linestyle=linestyle
    )
    slider.set_alpha(alpha)

    # plot pusher
    x_p0 = X_pusher[i]
    pusher = patches.Circle(
        x_p0, radius=PUSHER_R, facecolor=cmap(4), edgecolor='black', linestyle=linestyle
    )
    pusher.set_alpha(alpha)

    # plot goal
    x_g0 = x_goal
    transf_g0 = transforms.Affine2D().translate(x_g0[0], x_g0[1]).rotate_around(x_g0[0], x_g0[1], x_g0[2])
    goal = patches.PathPatch(
        contour_path, transform=transf_g0+trans_ax, facecolor=cmap(3), edgecolor='black', linestyle='--'
    )
    goal.set_alpha(0.5)

    # plot obstacles
    num_obstacles = len(x_obstacle)
    for k in range(num_obstacles):
        if (not types[k]) and i > 0:
            continue 
        x_o0 = x_obstacle[k]
        contour_path = Path(contact_basic.curve_list[k].pt_samples)
        transf_o0 = transforms.Affine2D().translate(x_o0[0], x_o0[1]).rotate_around(x_o0[0], x_o0[1], x_o0[2])
        new_obstacle = patches.PathPatch(
            contour_path, transform=transf_o0+trans_ax, facecolor=cmap(1-types[k]), edgecolor='black', linestyle=linestyle
        )
        new_obstacle.set_alpha(alpha)
        ax.add_patch(new_obstacle)

    # add patches
    ax.add_patch(slider)
    ax.add_patch(pusher)
    ax.add_patch(goal)

X_slider = np.array(X_slider)
X_obstacles = np.array(X_obstacles)
ax.plot(X_slider[:, 0], X_slider[:, 1], '--', color='red', alpha=1.0)
for i in range(num_obstacles):
    if not types[i]:
        continue
    ax.plot(X_obstacles[:, i, 0], X_obstacles[:, i, 1], '--', color='blue', alpha=1.0)

# 定义图例条目
legend_elements = [
    Line2D([0], [0], color=cmap(2), lw=4, label='Slider'),
    Line2D([0], [0], color=cmap(4), lw=4, label='Pusher'),
    Line2D([0], [0], color=cmap(0), lw=4, label='Movable Obstacle'),
    Line2D([0], [0], color=cmap(1), lw=4, label='Fixed Obstacle'),
    Line2D([0], [0], color=cmap(3), lw=4, label='Goal'),
    Line2D([0], [0], color="red",   lw=1, label='Planned Path',  linestyle='--'),
    Line2D([0], [0], color="blue",  lw=1, label='Obstacle Path', linestyle='--'),
]

# 添加图例
ax.legend(handles=legend_elements, loc='upper right')

# 显示图形
plt.show()
