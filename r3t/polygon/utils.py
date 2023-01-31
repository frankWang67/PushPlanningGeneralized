import numpy as np
from shapely.geometry import LinearRing, LineString, Point

from polytope_symbolic_system.common.utils import *

TOL = 1e-8  # zero threshold

def is_line_vertex(point, line):
    """
    Check if point is any vertex on line
    :param point: the point
    :param line: the line
    :return: is vertex or not
    """
    line_coords = np.array(line.coords)
    return np.linalg.norm(line_coords-point, ord=2, axis=1).min() < TOL

def get_rel_coords(point, base):
    """
    Get the relative coordinates in local frame
    :param point: the point
    :param base: the local frame {base}, (x,y,theta)
    :return: relative 2d coordinate
    """
    origin = base[:2]
    rot_world2base = rotation_matrix(-base[2])
    return np.matmul(rot_world2base, point-origin)

def get_normal_and_tangent_on_polygon(point, poly):
    """
    Get the normal and tangent vector in world frame
    :param point: the point
    :param poly: the polygon
    :return: norm, tangent
    """
    assert poly.boundary.is_ring
    line_ring = LinearRing(poly.boundary)
    # the linestring should be CW
    if line_ring.is_ccw:
        line_ring = line_ring.reverse()
    line_coords = np.array(line_ring.coords)
    point_diff_line_coords = point-line_coords

    normal, tangent = None, None

    for i in range(len(point_diff_line_coords)-1):
        if np.abs(np.cross(point_diff_line_coords[i, :], point_diff_line_coords[i+1, :])) < TOL:
            tangent = normalize(line_coords[i+1, :]-line_coords[i,:])
            normal = np.matmul(rotation_matrix(-0.5*np.pi), tangent)
            break
    
    if normal is None or tangent is None:
        raise RuntimeError('get_normal_and_tangent_on_polygon: the point is not on the given polygon\'s boundary!')

    return normal, tangent

def get_contact_jacobian(point):
    """
    Get the contact jacobian
    :param point: contact point (x, y)
    :return: 2X3 contact jacobian
    """
    px, py = point
    jacobian = np.array([[1, 0, -py],
                         [0, 1,  px]])
    return jacobian

def affine_transform_polygon(old_polygon, old_state=None, new_state=None, affine_matrix=None):
    """
    Apply affine transformation on polygon, according to old state and new state
    :param old_polygon: the old polygon
    :param old_state: the old state (x, y, theta)
    :param new_state: the new state (x, y, theta)
    :param affine_matrix: the matrix [a, b, d, e, xoff, yoff]
    :return: new polygon
    """
    if affine_matrix is None:
        dtheta = new_state[2] - old_state[2]
        a, b, d, e = rotation_matrix(dtheta).flatten()
        xoff, yoff = new_state[:2] - old_state[:2]
        affine_matrix = [a, b, d, e, xoff, yoff]
    new_polygon = affinity.affine_transform(old_polygon, affine_matrix)

    return new_polygon

def point_difference(point1, point2):
    """
    Calculate the displacement point1-point2:
    :param point1: the first point
    :param point2: the second point
    :return: ndarray(2,)
    """
    return (np.array(point1.xy)-np.array(point2.xy)).reshape(-1)

def apply_translation_and_rotation(poly, delta):
    """
    Apply translation and rotation on the polygon
    :param poly: the polygon
    :param delta: the (dx, dy, dtheta) transformation
    :return: the transformed polygon
    """
    dx, dy, dtheta = delta
    return affinity.translate(affinity.rotate(poly, dtheta, origin='center', use_radians=True), dx, dy)

def get_polygon_vertex_connection(poly1, poly2):
    """
    Get the directed LineString connections between corresponding vertex from polygon1 to polygon2
    :param poly1: the polygon1
    :param poly2: the polygon2
    :return: list of LineString
    """
    list_of_line = []
    for coord1, coord2 in zip(np.array(poly1.boundary.coords.xy).T, np.array(poly2.boundary.coords.xy).T):
        list_of_line.append(LineString([coord1, coord2]))
    return list_of_line

def get_adjacent_edge_partition(point, polygon):
    """
    Get the edge partition on polygon
    The point lies on one of the edges, and divides the edge into two beams
    :param point: the point
    :param polygon: the polygon
    :return: beam1, beam2
    """
    if isinstance(point, Point):
        point = np.array(point.coords)
    point = point.reshape(-1)
    poly_vertex = np.array(polygon.boundary.coords)
    for i in range(len(poly_vertex)-1):
        if np.abs(np.cross(point-poly_vertex[i], point-poly_vertex[i+1])) < TOL:
            return [LineString([point, poly_vertex[i]]), \
                    LineString([point, poly_vertex[i+1]])]
    raise RuntimeError('The point is not on the edges of polygon!')

def get_adjacent_edge(point, polygon):
    """
    Get the edge on polygon
    The point is one of the vertex, return the adjacent edges
    :param point: the point
    :param polygon: the polygon
    :return: beam1, beam2
    """
    if isinstance(point, Point):
        point = np.array(point.coords)
    point = point.reshape(-1)
    poly_vertex = np.array(polygon.boundary.coords)[:-1]
    num_vertex = len(poly_vertex)
    for i in range(len(poly_vertex)):
        if np.linalg.norm(point-poly_vertex[i]) < TOL:
            return [LineString([point, poly_vertex[(i+1)%num_vertex]]), \
                    LineString([point, poly_vertex[(i-1)%num_vertex]])]
    raise RuntimeError('The point is not vertex of polygon!')

def get_rotation_before_penetration(m_beam, s_beam, edge_on_which):
    """
    Get the rotation angle before penetration
    :param m_beam: the beam on m_poly (target)
    :param s_beam: the beam on s_poly (obstacle)
    :param edge_on_which: which polygon provide the edge contact
    :return: angle1, angle2
    """
    # convert beam to vectors (x1,y1)&(x2,y2) --> (dx,dy)
    m_beam[0] = np.diff(np.array(m_beam[0].coords), axis=0).reshape(-1)
    m_beam[1] = np.diff(np.array(m_beam[1].coords), axis=0).reshape(-1)
    s_beam[0] = np.diff(np.array(s_beam[0].coords), axis=0).reshape(-1)
    s_beam[1] = np.diff(np.array(s_beam[1].coords), axis=0).reshape(-1)
    # get pairs of beam
    if edge_on_which == 'target':
        if vector_included_angle(m_beam[0], s_beam[0]) <= vector_included_angle(m_beam[0], s_beam[1]):
            beam_pair1 = (m_beam[0], s_beam[0])
            beam_pair2 = (m_beam[1], s_beam[1])
        else:
            beam_pair1 = (m_beam[0], s_beam[1])
            beam_pair2 = (m_beam[1], s_beam[0])
    else:
        if vector_included_angle(s_beam[0], m_beam[0]) <= vector_included_angle(s_beam[0], m_beam[1]):
            beam_pair1 = (m_beam[0], s_beam[0])
            beam_pair2 = (m_beam[1], s_beam[1])
        else:
            beam_pair1 = (m_beam[0], s_beam[1])
            beam_pair2 = (m_beam[1], s_beam[0])

    # get rotation angle
    return angle_diff(get_vector_azimuth_angle(beam_pair1[1]),get_vector_azimuth_angle(beam_pair1[0])), \
           angle_diff(get_vector_azimuth_angle(beam_pair2[1]),get_vector_azimuth_angle(beam_pair2[0]))
