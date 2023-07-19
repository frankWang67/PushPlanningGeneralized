# plot the planned trajectory X_slider

import os

import pickle
from matplotlib import pyplot as plt
import numpy as np

# data_dir = "/home/yongpeng/research/R3T_shared/data/exp/new_model/all_in_one"
data_dir = "/home/yongpeng/research/R3T_shared/data/exp"

scene_no = 4

for i in [1, 2, 3]:
    for round in range(10):
        data = pickle.load(open(os.path.join(data_dir, "scene_{0}_10_round_#{1}/{2}.pkl".format(scene_no, i, round)), "rb"))
        if isinstance(data, list) and len(data) == 0:
            continue
        x_slider = np.array(data["X_slider"])
        plt.plot(x_slider[:, 0], x_slider[:, 1], alpha=0.5, linewidth=5.0)

plt.xlim([0.0, 0.5])
plt.ylim([0.0, 0.5])
plt.show()
