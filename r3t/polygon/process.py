import casadi as cs
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
from shapely.geometry import MultiPolygon, Point
from shapely.ops import nearest_points, unary_union
from shapely import affinity, intersects, intersection, distance

from polytope_symbolic_system.common.utils import *
from r3t.polygon.utils import *
from r3t.polygon.solver import LCP


def get_polygon_in_collision(target_state, target_poly, state_of_poly, list_of_poly):
    """
    Get polygons that will collide with the specific polygon
    :param target_state: the motions of target_poly, including current state,
                         each line represents one configuration
    :param target_poly: the specific polygon
    :param state_of_poly: state of candidate polygons
    :param list_of_poly: list of candiate polygons
    :return: flag - collision or not
    :return: coll_poly - list of polygons in collision
    :return: coll_state - states of polygons in coll_poly
    """
    # make multipolygons of future motion
    if not isinstance(target_state, np.ndarray):
        target_state = np.array(target_state)
    target_poly_future = []
    delta_target_state = target_state - target_state[0, :]
    # skip the first(current) state
    for i in range(1,len(delta_target_state)):
        dx_t, dy_t, dtheta_t = delta_target_state[i, :]
        target_poly_future.append(affinity.translate(affinity.rotate(target_poly, dtheta_t, 'center', use_radians=True), dx_t, dy_t))
    target_poly_future = unary_union(MultiPolygon(target_poly_future))

    # check collision
    num_coll_poly = 0
    coll_poly, coll_state = [], []
    for poly, state in zip(list_of_poly, state_of_poly):
        if intersects(poly, target_poly_future):
            coll_poly.append(poly)
            coll_state.append(state)
            num_coll_poly += 1
        else:
            coll_poly.append(None)
            coll_state.append(None)

    # return
    if num_coll_poly == 0:
        return False, None, None
    else:
        return True, coll_poly, coll_state

def get_polygon_in_collision(target_state, target_poly, state_of_poly, list_of_poly):
    """
    Get polygons that will collide with the specific polygon
    :param target_state: the motions of target_poly, including current state,
                         each line represents one configuration
    :param target_poly: the specific polygon
    :param state_of_poly: state of candidate polygons
    :param list_of_poly: list of candiate polygons
    :return: flag - collision or not
    :return: coll_poly - list of polygons in collision
    :return: coll_state - states of polygons in coll_poly
    """
    # make multipolygons of future motion
    if not isinstance(target_state, np.ndarray):
        target_state = np.array(target_state)
    target_poly_future = []
    delta_target_state = target_state - target_state[0, :]
    # skip the first(current) state
    for i in range(1,len(delta_target_state)):
        dx_t, dy_t, dtheta_t = delta_target_state[i, :]
        target_poly_future.append(affinity.translate(affinity.rotate(target_poly, dtheta_t, 'center', use_radians=True), dx_t, dy_t))
    target_poly_future = unary_union(MultiPolygon(target_poly_future))

    # check collision
    num_coll_poly = 0
    coll_poly, coll_state = [], []
    for poly, state in zip(list_of_poly, state_of_poly):
        if intersects(poly, target_poly_future):
            coll_poly.append(poly)
            coll_state.append(state)
            num_coll_poly += 1
        else:
            coll_poly.append(None)
            coll_state.append(None)

    # return
    if num_coll_poly == 0:
        return False, None, None
    else:
        return True, coll_poly, coll_state

def contact_location_detector(m_states, m_poly, s_states, s_poly):
    """
    Detect the contact location between two polygons
    The input polygons should be guaranteed to collide
    :param state_list: the list of states of m_poly
    :param m_poly: the movable polygon in {world}
    :param s_poly: the stationary polygon in {world}
    :return: m_coll_point, s_coll_point, the collision point in {world}, on m_poly and s_poly
    """
    if not isinstance(m_states, np.ndarray):
        m_states = np.array(m_states)
    m_states = m_states.reshape(-1,3)

    num_steps = len(m_states)

    if not isinstance(s_states, np.ndarray):
        s_states = np.array(s_states)
    s_states = s_states.reshape(-1,3)

    # assume s_poly is stationary, the states of m_poly is m_states in world frame
    # assume m_poly is stationary, compute the states of s_poly in world frame
    for i in range(1,len(m_states)):
        m_rel2_s = m_states[i,:2] - s_states[0,:2]  # in {world}
        s_rel2_m = -m_rel2_s
        s_poly_cent = m_states[0,:2] + rotation_matrix(m_states[0,2]) @ \
                                       rotation_matrix(m_states[i,2]).T @ s_rel2_m
        s_poly_theta = s_states[0,2] - (m_states[i,2]-m_states[0,2])
        s_states = np.append(s_states, np.append(s_poly_cent, s_poly_theta).reshape(1,3), axis=0)

    # the earliest collision distance and point
    early_coll_dist = np.inf
    early_coll_point = None

    m_poly_now = m_poly
    s_poly_now = s_poly
    for i in range(1, num_steps):
        m_poly_next = apply_translation_and_rotation(m_poly, m_states[i]-m_states[0])
        list_of_line = get_polygon_vertex_connection(m_poly_now, m_poly_next)
        for line in list_of_line:
            point = intersection(line, s_poly)
            if point.is_empty:
                continue
            if isinstance(point, LineString):
                point = Point(np.array(point.xy).T[0])
            coll_dist = line.line_locate_point(point)/line.length
            if coll_dist < early_coll_dist:
                early_coll_dist = coll_dist
                early_coll_point = (get_rel_coords(np.array(line.xy).T[0], m_states[i-1]), \
                                    get_rel_coords(np.array(point.xy).reshape(-1), s_states[0]))
        m_poly_now = m_poly_next

        s_poly_next = apply_translation_and_rotation(s_poly, s_states[i]-s_states[0])
        list_of_line = get_polygon_vertex_connection(s_poly_now, s_poly_next)
        for line in list_of_line:
            point = intersection(line, m_poly)
            if point.is_empty:
                continue
            if isinstance(point, LineString):
                point = Point(np.array(point.xy).T[0])
            coll_dist = line.line_locate_point(point)/line.length
            if coll_dist < early_coll_dist:
                early_coll_dist = coll_dist
                early_coll_point = (get_rel_coords(np.array(point.xy).reshape(-1), m_states[0]), \
                                    get_rel_coords(np.array(line.xy).T[0], s_states[i-1]))
        s_poly_now = s_poly_next

        if early_coll_point is not None:
            return early_coll_point
    
    raise RuntimeError('contact_location_detector: No collision detected between input polygons!')

def get_polygon_contact_configuration(target_state, target_poly, state_of_poly, list_of_poly):
    """
    Get contact location, normal, tangent between polygons
    :param target_state: the motions of target_poly, including current state,
                         each line represents one configuration
    :param target_poly: the specific polygon
    :param state_of_poly: state of candidate polygons
    :param list_of_poly: list of candiate polygons
    :return: contact configuration
    """
    if not isinstance(target_state, np.ndarray):
        target_state = np.array(target_state)
    contact_config = []
    target_state = target_state.reshape(-1, 3)
    
    for poly, state in zip(list_of_poly, state_of_poly):
        # not in collision
        if poly is None or state is None:
            contact_config.append(None)
            continue

        # in collision
        config = {'basic': {},
                  'target': {},
                  'obstacle': {},
                  'force': {}}
        # get states
        config['target']['x'] = target_state[0, :3]
        config['obstacle']['x'] = state[:3]

        # get contact location and rotation matrix
        # point_on_target, point_on_obstacle = nearest_points(target_poly.boundary, poly.boundary)
        # point_on_target, point_on_obstacle = np.array(point_on_target.coords.xy).reshape(-1), \
        #                                      np.array(point_on_obstacle.coords.xy).reshape(-1)
        # config['target']['rel_p'] = get_rel_coords(point_on_target, target_state[0, :3])
        # config['obstacle']['rel_p'] = get_rel_coords(point_on_obstacle, state[:3])

        # relative contact coordinate
        point_on_target, point_on_obstacle = contact_location_detector(target_state, target_poly, state, poly)
        config['target']['rel_p'] = point_on_target
        config['obstacle']['rel_p'] = point_on_obstacle

        # absolute contact coordinate (first state)
        start_point_on_target = target_state[0,:2] + rotation_matrix(target_state[0,2]) @ config['target']['rel_p']
        # absolute contact coordinate (final state)
        final_point_on_target = target_state[-1,:2] + rotation_matrix(target_state[-1,2]) @ config['target']['rel_p']
        point_on_obstacle = state[:2] + rotation_matrix(state[2]) @ config['obstacle']['rel_p']
        
        # the contact point (final state)
        config['target']['abs_p'] = final_point_on_target
        config['obstacle']['abs_p'] = point_on_obstacle

        # the polygons (final state)
        config['target']['polygon'] = apply_translation_and_rotation(target_poly, target_state[-1]-target_state[0])
        config['obstacle']['polygon'] = poly

        # the rotation matrix (first state)
        config['target']['rmat'] = rotation_matrix(target_state[0, 2])
        config['obstacle']['rmat'] = rotation_matrix(state[2])

        # get norm and tangent vector (first state)
        try:
            if is_line_vertex(start_point_on_target, target_poly.boundary):
                normal, tangent = get_normal_and_tangent_on_polygon(point_on_obstacle, poly)
                config['basic']['edge_on_which'] = 'obstacle'  # the obstacle contacts with edge
            elif is_line_vertex(point_on_obstacle, poly.boundary):
                normal, tangent = get_normal_and_tangent_on_polygon(start_point_on_target, target_poly)
                config['basic']['edge_on_which'] = 'target'  # the target contacts with edge
            config['force']['normal'] = normal
            config['force']['tangent'] = tangent
            config['force']['rmat'] = np.c_[normal, tangent]
            contact_config.append(config)
        except:
            import pdb; pdb.set_trace()
    
    return contact_config

def update_contact_configuration(target_state, contact_config):
    """
    Update the contact configuration between polygons
    :param target_state: the motions of target_poly, including current state,
                         each line represents one configuration
    :param contact_config: the old contact configurations
    :return new_contact_config: the new contact configurations
    """
    new_contact_config = []
    for config in contact_config:
        # import pdb; pdb.set_trace()
        # empty config --> not in contact
        if config is None:
            new_contact_config.append(None)
            continue
        # contact basic info
        dt = config['basic']['dt']
        miu = config['basic']['miu']
        v_t = config['target']['vel']  # (vx, vy, omega), in {world}
        edge_on_which = config['basic']['edge_on_which']

        # rotation and limit surface
        A_o = config['obstacle']['A']
        R_t = np.block([[config['target']['rmat'], np.zeros((2,1))], [np.zeros((1,2)), 1]])  # {target} => {world}
        R_o = np.block([[config['obstacle']['rmat'], np.zeros((2,1))], [np.zeros((1,2)), 1]])  # {obstacle} => {world}

        # contact jacobian
        p_t, p_o = config['target']['rel_p'], config['obstacle']['rel_p']
        jac_t = get_contact_jacobian(p_t)  # (vx, vy, omega) in {target} => (vp_x, vp_y) in {target}
        jac_o = get_contact_jacobian(p_o)  # (vx, vy, omega) in {obstacle} => (vp_x, vp_y) in {obstacle}
        vp_t = np.matmul(R_t[:2,:2], np.matmul(jac_t, np.matmul(R_t.T, v_t)))  # (2,), (vp_x, vp_y) in {world}

        # norm & tangent
        normal, tangent = config['force']['normal'], config['force']['tangent']
        N, T = normal.reshape(-1, 1), np.c_[tangent, -tangent]

        # get matrix
        K_o = matrix_mult([A_o, jac_o.T, R_o[:2,:2].T])  # (3,2), map (fn, ft) in {world} => (vx, vy, omega) in {obstacle}
        Kp_o = np.matmul(R_o[:2,:2], np.matmul(jac_o, K_o))  # (2,2), map (fn, ft) in {world} => (vp_x, vp_y) in {world}
        Kpn_o = np.matmul(Kp_o, N)  # (2,1), map (fn) => (vp_x, vp_y) in {world}
        Kpt_o = np.matmul(Kp_o, T)  # (2,2), map (ft+, ft-) => (vp_x, vp_y) in {world}

        M = np.block([[np.matmul(N.T, Kpn_o), np.matmul(N.T, Kpt_o), 0.],
                      [np.matmul(T.T, Kpn_o), np.matmul(T.T, Kpt_o), np.ones((2,1))],
                      [miu, -np.ones((1,2)), 0.]])
        
        q = -np.r_[np.matmul(N.T, vp_t), np.matmul(T.T, vp_t), 0.]

        if edge_on_which == 'target':
            M[:3, :3] = -(-M[:3, :3])  # two minus, one for reaction force, one for relative velocity
            q[:3] = -q[:3]
        elif edge_on_which == 'obstacle':
            pass

        ## build the QP (qpsolvers, not work)
        # QP = {'P': csc_matrix(2*M),
        #       'q': q,
        #       'G': csc_matrix(-np.block([[M], [np.eye(M.shape[1])]])),
        #       'h': np.r_[q, np.zeros(M.shape[1])],
        #       'x': None}
        # QP['x'] = solve_qp(P=QP['P'], q=QP['q'], G=QP['G'], h=QP['h'], solver='osqp')

        ## build the QP (casadi with qpoases, works)
        qp = {}
        # qp['h'] = cs.DM(2*M).sparsity()
        # qp['a'] = cs.DM(M).sparsity()
        # QP = cs.conic('LCP2QP', 'qpoases', qp)
        try:
            # import pdb; pdb.set_trace()
            # ans = solveLCP(M=M, q=q)
            # res = QP(h=2*M, g=q, a=M, lba=-q, lbx=0.)
            res = {'x': LCP(M, q)}
        except:
            import pdb; pdb.set_trace()
        qp['x'] = res['x']

        # solve contact force and velocity
        f = qp['x'][:3]  # (fn, ft+, ft-)
        print('complementary variable Mx+q: ', M@qp['x']+q)

        # (fn, ft) in {world}
        if edge_on_which == 'target':
            f = np.matmul(np.c_[normal, tangent, -tangent], -f)  # one minus, for reaction force
        elif edge_on_which == 'obstacle':
            f = np.matmul(np.c_[normal, tangent, -tangent], f)
        
        vp_o = np.matmul(Kp_o, f)  # (2,)
        dtheta_o = dt * np.matmul(K_o, f)[2]
        config['obstacle']['dtheta'] = dtheta_o

        # modity the contact location
        if edge_on_which == 'target':
            vp_rel = np.dot(tangent, vp_o - vp_t) * tangent  # (2,), (vp_rel_x, vp_rel_y) in {world}
            config['target']['rel_p'] += dt * np.matmul(R_t[:2,:2].T, vp_rel)
        elif edge_on_which == 'obstacle':
            vp_rel = np.dot(tangent, vp_t - vp_o) * tangent  # (2,), (vp_rel_x, vp_rel_y) in {world}
            config['obstacle']['rel_p'] += dt * np.matmul(R_o[:2,:2].T, vp_rel)

        # calculate new state
        config['target']['x'] = target_state[-1, :]

        # handle penetration
        penetration_flag, dtheta_lim = handle_possible_penetration(config)

        ## FOR DEBUG
        import copy
        old_config = copy.deepcopy(config)

        # update obstacle
        old_obstacle_x = config['obstacle']['x']
        new_obstacle_theta = old_obstacle_x[2]+dtheta_lim
        # update rotation matrix
        R_t = rotation_matrix(config['target']['x'][2])
        R_o = rotation_matrix(new_obstacle_theta)
        config['obstacle']['x'] = np.append(config['target']['x'][:2] + np.matmul(R_t[:2,:2], config['target']['rel_p']) \
                                                                      - np.matmul(R_o[:2,:2], config['obstacle']['rel_p']),
                                            new_obstacle_theta)
        dx_o = config['obstacle']['x'] - old_obstacle_x

        if penetration_flag:
            # clear dx_o
            dx_o = None

            target_poly_in_collision = gen_polygon(config['target']['x'], config['target']['geom'], 'box')
            obstacle_poly_in_collision = gen_polygon(config['obstacle']['x'], config['obstacle']['geom'], 'box')
            candidate_contact_point = []
            # enumerate vertex
            for vertex in np.array(target_poly_in_collision.boundary.coords):
                if distance(Point(vertex), obstacle_poly_in_collision) <= TOL:
                    candidate_contact_point.append(vertex)
            for vertex in np.array(obstacle_poly_in_collision.boundary.coords):
                if distance(Point(vertex), target_poly_in_collision) <= TOL:
                    candidate_contact_point.append(vertex)
            # select new contact point
            if len(candidate_contact_point) == 0:
                raise RuntimeError('No proper contect point to be selected!')
            for point in candidate_contact_point:
                new_obstacle_poly = affinity.rotate(obstacle_poly_in_collision, np.sign(dtheta_o)*np.abs(dtheta_o-dtheta_lim), Point(point), use_radians=True)
                if intersection(target_poly_in_collision, 
                                new_obstacle_poly).area <= TOL:
                    config['obstacle']['x'] = np.append(np.array(new_obstacle_poly.centroid.coords).reshape(-1), old_obstacle_x[2]+dtheta_o)
                    dx_o = config['obstacle']['x'] - old_obstacle_x

            if dx_o is None:
                import pdb; pdb.set_trace()
        
        ## FOR DEBUG
        if intersection(gen_polygon(config['target']['x'], config['target']['geom'], 'box'), \
                        gen_polygon(config['obstacle']['x'], config['obstacle']['geom'], 'box')).area > TOL:
            import pdb; pdb.set_trace()

        # get transformation for polygon
        config['obstacle']['dx'] = dx_o
        
        new_contact_config.append(config)

    return new_contact_config

def handle_possible_penetration(contact_config):
    """
    Handle possible penetration
    :param contact_config: the contact configuration

    """
    # get edge beam
    if contact_config['basic']['edge_on_which'] == 'target':
        edge_part_beam = get_adjacent_edge_partition(point=contact_config['target']['abs_p'],
                                                     polygon=contact_config['target']['polygon'])
        edge_beam = get_adjacent_edge(point=contact_config['obstacle']['abs_p'],
                                      polygon=contact_config['obstacle']['polygon'])
        # get rotation angle
        angle1, angle2 = get_rotation_before_penetration(m_beam=edge_part_beam,
                                                         s_beam=edge_beam,
                                                         edge_on_which='target')
    else:
        edge_part_beam = get_adjacent_edge_partition(point=contact_config['obstacle']['abs_p'],
                                                     polygon=contact_config['obstacle']['polygon'])
        edge_beam = get_adjacent_edge(point=contact_config['target']['abs_p'],
                                      polygon=contact_config['target']['polygon'])
        # get rotation angle
        angle1, angle2 = get_rotation_before_penetration(m_beam=edge_beam,
                                                         s_beam=edge_part_beam,
                                                         edge_on_which='obstacle')

    # detect penetration
    dtheta = contact_config['obstacle']['dtheta']
    if min(angle1, angle2) <= dtheta <= max(angle1, angle2):
        return False, dtheta
    else:
        if dtheta >= 0:
            return True, max(angle1, angle2)
        else:
            return True, min(angle1, angle2)

## ---------------------------------------------------
## DEPRECATED FUNCTIONS
## ---------------------------------------------------

def contact_location_detector_v2(m_states, m_poly, s_states, s_poly):
    """
    ## DEPRECATED
    Detect the contact location between two polygons
    The input polygons should be guaranteed to collide
    :param state_list: the list of states of m_poly
    :param m_poly: the movable polygon in {world}
    :param s_poly: the stationary polygon in {world}
    :return: m_coll_point, s_coll_point, the collision point in {world}, on m_poly and s_poly
    """
    if not isinstance(m_states, np.ndarray):
        m_states = np.array(m_states)
    m_states = m_states.reshape(-1,3)

    if not isinstance(s_states, np.ndarray):
        s_states = np.array(s_states)
    s_states = s_states.reshape(-1,3)

    # calculate the relative position in all time steps, in {world}
    m_rel_to_s = m_states[:,:2] - s_states[:,:2]

    delta_state = np.diff(m_states, axis=0)

    # the earliest collision distance and point
    early_coll_dist = np.inf
    early_coll_point = None

    m_poly_now = m_poly
    s_poly_now = s_poly
    s_poly_theta_now = s_states[0,2]
    for i in range(len(delta_state)):
        m_poly_next = apply_translation_and_rotation(m_poly_now, delta_state[i])
        list_of_line = get_polygon_vertex_connection(m_poly_now, m_poly_next)
        for line in list_of_line:
            point = intersection(line, s_poly)
            if point.is_empty:
                continue
            if isinstance(point, LineString):
                point = Point(np.array(point.xy).T[0])
            coll_dist = line.line_locate_point(point)/line.length
            if coll_dist < early_coll_dist:
                early_coll_dist = coll_dist
                early_coll_point = (get_rel_coords(np.array(line.xy).T[0], m_states[i]), \
                                    get_rel_coords(np.array(point.xy).reshape(-1), s_states[0]))
        m_poly_now = m_poly_next

        s_poly_centroid_next = rotation_matrix(m_states[0,2]) @ \
                               rotation_matrix(m_states[i+1,2]).T @ \
                               (-m_rel_to_s[i+1]) + m_states[0,:2]
        s_poly_centroid_now = np.array(s_poly_now.centroid.coords).reshape(-1)
        s_poly_dtheta = -delta_state[i,2]
        s_poly_next = apply_translation_and_rotation(s_poly_now, 
                        np.append(s_poly_centroid_next-s_poly_centroid_now, s_poly_dtheta))
        list_of_line = get_polygon_vertex_connection(s_poly_now, s_poly_next)
        for line in list_of_line:
            point = intersection(line, m_poly)
            if point.is_empty:
                continue
            if isinstance(point, LineString):
                point = Point(np.array(point.xy).T[0])
            coll_dist = line.line_locate_point(point)/line.length
            if coll_dist < early_coll_dist:
                early_coll_dist = coll_dist
                early_coll_point = (get_rel_coords(np.array(point.xy).reshape(-1), m_states[0]), \
                                    get_rel_coords(np.array(line.xy).T[0], np.append(s_poly_centroid_now, s_poly_theta_now)))
        s_poly_now = s_poly_next
        s_poly_theta_now += s_poly_dtheta
        if early_coll_point is not None:
            return early_coll_point
    
    raise RuntimeError('contact_location_detector: No collision detected between input polygons!')

def recover_from_penetration(target_poly, new_target_poly, obstacle_poly, new_obstacle_poly):
    """
    ## DEPRECATED
    Due to discrete simulation, obstacle might penetrate with the target
    This function detects the situation and get rid of penetration
    :param target_poly: the target polygon
    :param new_target_poly: the new target polygon
    :param obstacle_poly: the obstacle polygon
    :param new_obstacle_poly: the new obstacle polygon
    :return: flag, if penetration detected
    :return: new_obstacle_poly, the obstacle without penetration
    """
    # detect penetration
    if intersection(new_target_poly, new_obstacle_poly).area <= 1e-5:
        return False, None

    # get polygon coods
    target_coord = np.array(target_poly.exterior.coords.xy).reshape(-1,2)[:-1]
    target_coord_new = np.array(new_target_poly.exterior.coords.xy).reshape(-1,2)[:-1]
    obstacle_coord = np.array(obstacle_poly.exterior.coords.xy).reshape(-1,2)[:-1]
    obstacle_coord_new = np.array(new_obstacle_poly.exterior.coords.xy).reshape(-1,2)[:-1]

    # enumerate vertex of target polygon
    for i in range(len(target_coord)):
        intersection_point = intersection(LineString([target_coord[i], target_coord_new[i]]), new_obstacle_poly.boundary)
        if not intersection_point.is_empty:
            xoff, yoff = np.array(intersection_point.xy) - target_coord_new[i]
            update_obstacle_poly = affinity.translate(new_obstacle_poly, -xoff, -yoff)
            if intersection(new_target_poly, update_obstacle_poly).area <= 1e-5:
                return True, update_obstacle_poly

    # enumerate vertex of obstacle polygon
    for i in range(len(obstacle_coord)):
        intersection_point = intersection(LineString([obstacle_coord[i], obstacle_coord_new[i]]), new_target_poly.boundary)
        if not intersection_point.is_empty:
            xoff, yoff = np.array(intersection_point.xy) - obstacle_coord_new[i]
            update_obstacle_poly = affinity.translate(new_obstacle_poly, xoff, yoff)
            if intersection(new_target_poly, update_obstacle_poly).area <= 1e-5:
                return True, update_obstacle_poly
