import numpy as np

from r3t.polygon.collision import *
from r3t.polygon.utils import gen_polygon
from shapely.geometry import Polygon

def convert_polygon_shapely_to_headbutt(polygon:Polygon, state):
    vertices = np.array(polygon.exterior.xy).T
    vertices_headbutt = []
    for i in range(vertices.shape[0]-1):
        vtx = glm_Vec2Base()
        vtx.x, vtx.y = vertices[i, 0], vertices[i, 1]
        vertices_headbutt.append(vtx)
    polygon_headbutt = headbutt_twod_shapes_Polygon(vertices_headbutt)

    t_, r_, s_ = glm_Vec2Base(), 0, glm_Vec2Base()
    t_.x, t_.y = state[0], state[1]
    r_ = state[2]
    s_.x, s_.y = 1, 1

    polygon_headbutt.setTransform(t_, r_, s_)

    return polygon_headbutt

def collision_reaction(states, geoms, fixed_idx=0):
    n_p = states.shape[0]
    assert len(geoms) == n_p

    headbutt_polygons = []  # no transform
    for i in range(n_p):
        if len(geoms[i]) <= 2:
            polygon_i = gen_polygon([0.0, 0.0, 0.0], geoms[i], type="box")
        else:
            polygon_i = gen_polygon([0.0, 0.0, 0.0], geoms[i], type="polygon")
        headbutt_polygon_i = convert_polygon_shapely_to_headbutt(polygon_i, states[i])
        headbutt_polygons.append(headbutt_polygon_i)

    states_updated = states.copy()

    for i in range(n_p):
        if i == fixed_idx:
            continue

        result = headbutt_twod_Headbutt.intersect(headbutt_twod_Headbutt.test(headbutt_polygons[fixed_idx], \
                                                                              headbutt_polygons[i]))
        
        if not result:
            continue

        # import pdb; pdb.set_trace()

        penetration_vector = np.array([result.intersection.x, result.intersection.y])
        states_updated[i, :2] += penetration_vector

        # from r3t.polygon.plot_utils import plot_polygons_consecutive_timesteps
        # plot_polygons_consecutive_timesteps(states1=states, states2=states, geoms=geoms)
        # plot_polygons_consecutive_timesteps(states1=states, states2=states_updated, geoms=geoms)

    return states_updated
