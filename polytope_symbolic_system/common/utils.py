import numpy as np
from shapely.geometry import Polygon
from shapely import affinity


def restrict_angle_in_unit_circle(angle):
    """
    Restrict angle in range [-pi, pi]
    :param: angle in (-inf, inf)
    :return: angle in [-pi, pi]
    """
    return -np.pi + (angle - (-np.pi)) % (2*np.pi)

def gen_polygon(coord, geom):
    """
    Return the shapely.Polygon object of (x, y, theta)
    :param coord: (x, y, theta) coordinates
    :param beta: (xl, yl) geometry
    :return: Polygon
    """
    x, y, theta = coord
    xl, yl = geom
    poly = Polygon([(0.5*xl, 0.5*yl), (-0.5*xl, 0.5*yl), (-0.5*xl, -0.5*yl), (0.5*xl, -0.5*yl), (0.5*xl, 0.5*yl)])
    poly = affinity.rotate(poly, theta, origin='center', use_radians=True)
    poly = affinity.translate(poly, x, y)
    
    return poly

def duplicate_state_with_multiple_azimuth_angle(origin_state):
    """
    Duplicate the state with multiple azimuth angle theta, theta-2pi, theta+2pi
    :param origin_state: original state, with theta, without psic
    """
    duplicate_states = np.repeat(np.atleast_2d(origin_state), 3, axis=0)
    duplicate_states[1, 2] -= 2 * np.pi
    duplicate_states[2, 2] += 2 * np.pi

    return duplicate_states

def angle_diff(angle1, angle2):
    # calculate the difference (angle2-angle1) ∈ [-pi, pi]
    diff1 = angle2 - angle1
    diff2 = 2 * np.pi - np.abs(diff1)
    if diff1 > 0:
        diff2 = -diff2
    if np.abs(diff1) < np.abs(diff2):
        return diff1
    else:
        return diff2

def matrix_mult(matrix_list):
    """
    Calculate the matrix multiplication in order
    :param matrix_list: [M1, M2, ..., Mk]
    :return: M1 * M2 * ... * Mk
    """
    result = np.eye(matrix_list[0].shape[0])
    for matrix in matrix_list:
        result = np.matmul(result, matrix)
    return result
