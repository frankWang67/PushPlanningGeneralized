from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

from polytope_symbolic_system.common.utils import gen_polygon

from matplotlib import pyplot as plt

def plot_polygons_consecutive_timesteps(states1, states2, orig_state_list, geoms):
    fig, ax = plt.subplots()
    
    polygons1 = []
    for i in range(states1.shape[0]):
        if len(geoms[i]) <= 2:
            polygons1.append(gen_polygon(states1[i], geoms[i], 'box'))
        else:
            polygons1.append(gen_polygon(states1[i], geoms[i], 'polygon'))

    for polygon in polygons1:
        plot_polygon(polygon, ax, facecolor="blue", alpha=0.5, edgecolor="deepskyblue")
    
    polygons2 = []
    for i in range(states2.shape[0]):
        if len(geoms[i]) <= 2:
            polygons2.append(gen_polygon(states2[i], geoms[i], 'box'))
        else:
            polygons2.append(gen_polygon(states2[i], geoms[i], 'polygon'))

    for polygon in polygons2:
        plot_polygon(polygon, ax, facecolor=(1.0, 1.0, 1.0, 0.0), edgecolor="deepskyblue", linestyle=":")

    orig_slider_polygon = gen_polygon(orig_state_list[-1][:3], geoms[0], 'box')
    plot_polygon(orig_slider_polygon, ax, facecolor=(1.0, 1.0, 1.0, 0.0), edgecolor="coral")

    plt.gca().set_aspect("equal")
    plt.show()
