import os
r3t_root_dir = os.environ.get("R3T_HOME")
from r3t.polygon.scene import *

## obstacle avoidance scene
#  -------------------------------------------------
# scene_path_name = os.path.join(r3t_root_dir, "data", "debug/real_experiment")
# scene_file_name = 'obstacle_avoidance_scene.pkl'
# scene_pkl = os.path.join(scene_path_name, scene_file_name)
# scene_data = pickle.load(open(scene_pkl, 'rb'))

# import pdb; pdb.set_trace()

# scene, basic = load_planning_scene_from_file(scene_pkl)
# fig, ax = visualize_scene(scene, alpha=0.25, xlim=[0.26, 0.80], ylim=[-0.09, 0.45], movability=scene.types)
# plt.show()
#  -------------------------------------------------

## obstacle pushaway scene
#  -------------------------------------------------
# scene_path_name = os.path.join(r3t_root_dir, "data", "debug/real_experiment")
# scene_file_name = 'obstacle_pushaway_scene.pkl'
# scene_pkl = os.path.join(scene_path_name, scene_file_name)
# scene_data = pickle.load(open(scene_pkl, 'rb'))

# import pdb; pdb.set_trace()

# scene, basic = load_planning_scene_from_file(scene_pkl)
# fig, ax = visualize_scene(scene, alpha=0.25, xlim=[0.26, 0.80], ylim=[-0.06, 0.48], movability=scene.types)
# plt.show()
#  -------------------------------------------------

## pushaway circle obstacle scene
#  -------------------------------------------------
scene_path_name = os.path.join(r3t_root_dir, "data", "debug/real_experiment")
scene_file_name = 'pushaway_circle_obstacle_scene.pkl'
scene_pkl = os.path.join(scene_path_name, scene_file_name)
scene_data = pickle.load(open(scene_pkl, 'rb'))

import pdb; pdb.set_trace()

scene, basic = load_planning_scene_from_file(scene_pkl)
fig, ax = visualize_scene(scene, alpha=0.25, xlim=[0.23, 0.77], ylim=[-0.06, 0.48], movability=scene.types)
plt.show()
#  -------------------------------------------------
