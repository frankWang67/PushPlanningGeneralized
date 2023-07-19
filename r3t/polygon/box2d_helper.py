import numpy as np

from Box2D import b2PolygonShape, b2TimeOfImpact, b2Sweep

from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.plotting import plot_polygon

from r3t.polygon.utils import *

def cvt_xy_to_array(xy):
    return np.array(xy).T

def cvt_array_to_b2fmt(arr):
    return [tuple(ai) for ai in arr]

def cvt_pt_global_to_local(point, frame):
    _R = rotation_matrix(frame[2])
    return np.matmul(_R.T, point-frame[0:2])

class _Sweep:
    def __init__(self) -> None:
        self.c0 = (0.0, 0.0)
        self.a0 = 0.0
        self.c = (0.0, 0.0)
        self.a = 0.0
        self.localCenter = (0.0, 0.0)

class _ContactInfo:
    def __init__(self) -> None:
        # contact points (relative)
        self.c1 = np.array([0.0, 0.0])
        self.c2 = np.array([0.0, 0.0])
        # contact normal (relative)
        self.a1 = np.array([0.0, 0.0])
        self.a2 = np.array([0.0, 0.0])
        # contact tangential (relative)
        self.b1 = np.array([0.0, 0.0])
        self.b2 = np.array([0.0, 0.0])
        # 1 - normal&tangent on obj1, 2 - normal&tangent on obj2
        self.valid_index = 1
        # index of contact objects
        self.id1 = 0
        self.id2 = 0

def _TimeOfImpact(proxyA:Polygon, proxyB:Polygon, sweepA:_Sweep, sweepB:_Sweep, tMax=1.0):
    """
        Get the time of impact.
    """
    proxyA = b2PolygonShape(vertices=cvt_array_to_b2fmt(cvt_xy_to_array(proxyA.exterior.xy)-cvt_xy_to_array(proxyA.exterior.xy)[:-1, :].mean(axis=0)))
    proxyB = b2PolygonShape(vertices=cvt_array_to_b2fmt(cvt_xy_to_array(proxyB.exterior.xy)-cvt_xy_to_array(proxyB.exterior.xy)[:-1, :].mean(axis=0)))
    sweepA = b2Sweep(c0=sweepA.c0, a0=sweepA.a0, c=sweepA.c, a=sweepA.a, localCenter=sweepA.localCenter)
    sweepB = b2Sweep(c0=sweepB.c0, a0=sweepB.a0, c=sweepB.c, a=sweepB.a, localCenter=sweepB.localCenter)
    tMax = tMax
    _type, _time_of_impact = b2TimeOfImpact(shapeA=proxyA, \
                                            shapeB=proxyB, \
                                            sweepA=sweepA, \
                                            sweepB=sweepB, \
                                            tMax=tMax)
    
    return _time_of_impact

def _ContactConfig(moved_states, moved_polygon, fixed_state, fixed_polygon, scale=1000.0):
    """
        Get the contact information (contact point, normal, tangential).
        :param scale: the coordinates should be scaled to guarantee precision
    """
    moved_sweep = _Sweep()
    moved_sweep.c0 = tuple(moved_states[0, 0:2] * scale)
    moved_sweep.c = tuple(moved_states[-1, 0:2] * scale)
    moved_sweep.a = moved_states[-1, 2] - moved_states[0, 2]
    fixed_sweep = _Sweep()
    fixed_sweep.c0 = tuple(np.array(fixed_state[0:2]) * scale)
    fixed_sweep.c = tuple(np.array(fixed_state[0:2]) * scale)

    moved_polygon_scaled = affinity.scale(affinity.translate(moved_polygon, \
                                          xoff=-moved_states[0, 0], \
                                          yoff=-moved_states[0, 1]), xfact=scale, yfact=scale)
    fixed_polygon_scaled = affinity.scale(affinity.translate(fixed_polygon, \
                                              xoff=-fixed_state[0], \
                                              yoff=-fixed_state[1]), xfact=scale, yfact=scale)

    time_of_impact = _TimeOfImpact(proxyA=moved_polygon_scaled, \
                                   proxyB=fixed_polygon_scaled, \
                                   sweepA=moved_sweep, \
                                   sweepB=fixed_sweep)
    
    _trans_to_impact = (np.array(moved_sweep.c) - np.array(moved_sweep.c0)) / scale * time_of_impact
    _rot_to_impact = moved_sweep.a * time_of_impact

    moved_polygon_impact = affinity.rotate(moved_polygon, angle=_rot_to_impact, use_radians=True)
    moved_polygon_impact = affinity.translate(moved_polygon_impact, xoff=_trans_to_impact[0], yoff=_trans_to_impact[1])
    
    # moved_polygon_terminal = affinity.rotate(moved_polygon, angle=moved_sweep.a, use_radians=True)
    # moved_polygon_terminal = affinity.translate(moved_polygon_terminal, xoff=(np.array(moved_sweep.c) - np.array(moved_sweep.c0))[0] / scale, \
    #                                             yoff=(np.array(moved_sweep.c) - np.array(moved_sweep.c0))[1] / scale)

    contact_pt1, contact_pt2 = nearest_points(moved_polygon_impact, fixed_polygon)
    contact_pt1, contact_pt2 = cvt_xy_to_array(contact_pt1.xy).squeeze(), cvt_xy_to_array(contact_pt2.xy).squeeze()

    try:
        norm1, tangent1 = get_normal_and_tangent_on_polygon(contact_pt1, moved_polygon_impact)
        norm2, tangent2 = get_normal_and_tangent_on_polygon(contact_pt2, fixed_polygon)
    except:
        import pdb; pdb.set_trace()

    # convert contact points from {global frame} to {local frame}
    contact_info = _ContactInfo()
    moved_state_impact = np.append(np.array(moved_sweep.c0) / scale + _trans_to_impact, moved_states[0, 2] + _rot_to_impact)
    contact_info.c1 = cvt_pt_global_to_local(point=contact_pt1, frame=moved_state_impact)
    contact_info.c2 = cvt_pt_global_to_local(point=contact_pt2, frame=fixed_state)

    # moved_theta_impact = moved_state_impact[2]
    # contact_info.a1 = np.matmul(rotation_matrix(moved_theta_impact).T, norm1)
    # contact_info.a2 = np.matmul(rotation_matrix(moved_theta_impact).T, norm2)
    # contact_info.b1 = np.matmul(rotation_matrix(moved_theta_impact).T, tangent1)
    # contact_info.b2 = np.matmul(rotation_matrix(moved_theta_impact).T, tangent2)
    contact_info.a1 = norm1
    contact_info.a2 = norm2
    contact_info.b1 = tangent1
    contact_info.b2 = tangent2

    if is_line_vertex(contact_pt1, moved_polygon_impact.boundary):
        contact_info.valid_index = 2

    ## debug
    ## --------------------
    # print("sweepA: c0=", moved_sweep.c0, " c=", moved_sweep.c, " a=", moved_sweep.a)
    # print("sweepB: c0=", fixed_sweep.c0, " c=", fixed_sweep.c)
    # print("nearest points, on A: ", contact_pt1, " on B: ", contact_pt2)
    # print("valid index: ", contact_info.valid_index)
    # print("time of impact: ", time_of_impact)

    # fig, ax = plt.subplots()
    # plot_polygon(moved_polygon, ax, color="blue", alpha=0.3)
    # # plot_polygon(moved_polygon_terminal, ax, color="green", alpha=0.3)
    # plot_polygon(fixed_polygon, ax, color="red", alpha=0.3)
    # plot_polygon(moved_polygon_impact, ax, color="red", alpha=0.3)
    # ax.scatter([contact_pt1[0], contact_pt2[0]], [contact_pt1[1], contact_pt2[1]], c="black", s=80)
    # ax.arrow(contact_pt1[0], contact_pt1[1], norm1[0]*0.02, norm1[1]*0.02, color="red")
    # ax.arrow(contact_pt2[0], contact_pt2[1], norm2[0]*0.02, norm2[1]*0.02, color="red")
    # ax.arrow(contact_pt1[0], contact_pt1[1], tangent1[0]*0.02, tangent1[1]*0.02, color="blue")
    # ax.arrow(contact_pt2[0], contact_pt2[1], tangent2[0]*0.02, tangent2[1]*0.02, color="blue")
    # plt.gca().set_aspect("equal")
    # plt.show()
    ## --------------------
    
    return contact_info

if __name__ == "__main__":
    length = 100
    polyA = Polygon([[length, 0], [0, length], [-length, 0], [0, -length]])
    polyB = Polygon([[length, 0], [0, length], [-length, 0], [0, -length]])
    sweepA = _Sweep()
    sweepB = _Sweep()
    sweepB.c0 = (202.5, 0.0)
    sweepB.c = (197.5, 0.0)
    toi = _TimeOfImpact(polyA, polyB, sweepA, sweepB, tMax=1.0)
    print("Time of impact: ", toi)
