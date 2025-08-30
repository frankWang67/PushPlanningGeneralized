import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches, transforms
from matplotlib.lines import Line2D

from r3t.polygon.scene import load_planning_scene_from_file

r3t_root_dir = os.environ.get("R3T_HOME")
cmap = plt.cm.Pastel2

planning_scene_path = os.path.join(r3t_root_dir, "data", "wshf", "2025_05_02_22_19")
planning_scene_pkl  = os.path.join(planning_scene_path, "scene.pkl")
# scene = pickle.load(open(planning_scene_pkl, 'rb'))

scene, contact_basic = load_planning_scene_from_file(planning_scene_pkl)

x_slider = scene.target_state
x_obstacle = scene.states
types = scene.types
x_goal = scene.goal_state

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Planning Scene')

ax.set_autoscale_on(False)
# ax.grid('on')
ax.set_aspect('equal', 'box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
xlim = [-0.15, 0.35]
ylim = [-0.30, 0.30]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# plot slider
x_s0 = x_slider
contour_path = Path(contact_basic.curve_target.pt_samples)
trans_ax = ax.transData
transf_s0 = transforms.Affine2D().translate(x_s0[0], x_s0[1]).rotate_around(x_s0[0], x_s0[1], x_s0[2])
# slider = patches.PathPatch(
#     contour_path, transform=transf_s0+trans_ax, facecolor='#1f77b4', edgecolor='black'
# )
slider = patches.PathPatch(
    contour_path, transform=transf_s0+trans_ax, facecolor=cmap(2), edgecolor='black'
)

x_g0 = x_goal
transf_g0 = transforms.Affine2D().translate(x_g0[0], x_g0[1]).rotate_around(x_g0[0], x_g0[1], x_g0[2])
goal = patches.PathPatch(
    contour_path, transform=transf_g0+trans_ax, facecolor=cmap(3), edgecolor='black', linestyle='--'
)

# plot obstacles
obstacles = []
num_obstacles = len(x_obstacle)
for k in range(num_obstacles):
    x_o0 = x_obstacle[k]
    contour_path = Path(contact_basic.curve_list[k].pt_samples)
    transf_o0 = transforms.Affine2D().translate(x_o0[0], x_o0[1]).rotate_around(x_o0[0], x_o0[1], x_o0[2])
    new_obstacle = patches.PathPatch(
        contour_path, transform=transf_o0+trans_ax, facecolor=cmap(1-types[k]), edgecolor='black'
    )
    obstacles.append(new_obstacle)

# add patches
ax.add_patch(slider)
ax.add_patch(goal)
for k in range(num_obstacles):
    ax.add_patch(obstacles[k])

# 定义图例条目
legend_elements = [
    Line2D([0], [0], color=cmap(2), lw=4, label='Slider'),
    Line2D([0], [0], color=cmap(0), lw=4, label='Movable Obstacle'),
    Line2D([0], [0], color=cmap(1), lw=4, label='Fixed Obstacle'),
    Line2D([0], [0], color=cmap(3), lw=4, label='Goal')
]

# 添加图例
ax.legend(handles=legend_elements, loc='upper right')

# 显示图形
plt.show()
