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

def gen_polygon(coord, geom, type='box'):
    """
    Return the shapely.Polygon object of (x, y, theta)
    :param coord: (x, y, theta) coordinates
    :param beta: (xl, yl) geometry
    :param type: 'box', 'polygon'
    :return: Polygon
    """
    x, y, theta = coord

    if type == 'box':
        xl, yl = geom
        poly = Polygon([(0.5*xl, 0.5*yl), (-0.5*xl, 0.5*yl), (-0.5*xl, -0.5*yl), (0.5*xl, -0.5*yl), (0.5*xl, 0.5*yl)])
    elif type == 'polygon':
        # geom => 2d coordinates of vertex, in order, does not matter if the line ring is not closed
        poly = Polygon(geom)
    
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

def rotation_matrix(theta):
    """
    Return the rotation matrix CCW
    :param theta: rotation angle
    :return: 2X2 rotation matrix
    """
    mat = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return mat

def normalize(vector):
    """
    Normalize a vector
    :param vector: the vector
    :return: the normalized vector
    """
    return vector/np.linalg.norm(vector,ord=2)

def vector_included_angle(vec1, vec2):
    """
    Get the included angle of vectors
    :param vec1: the vector 1
    :param vec2: the vector 2
    :return: angle [0,pi]
    """
    return np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1,ord=2)*np.linalg.norm(vec2,ord=2)))

def get_vector_azimuth_angle(vec):
    """
    Get the azimuth angle of vector
    :param vec: the vector
    :return: the azimuth angle
    """
    return np.arctan2(vec[1],vec[0])
