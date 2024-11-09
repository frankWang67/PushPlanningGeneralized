import casadi as cs
import copy
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.spatial.transform import Rotation as R
import pickle

from polytope_symbolic_system.common.intfunc import *
from polytope_symbolic_system.common.utils import *
from r3t.polygon.utils import *
from r3t.polygon.process import *
from r3t.polygon.plot_utils import *
from r3t.polygon.collision_interface import collision_reaction


class ContactBasic:
    """
    Underlying class of R3T, including basic information about object-object contact,
    including friction coefficient, geometry ...
    """
    def __init__(self, miu_list=None, geom_list=None, A_list=None, geom_target=None, A_target=None, \
                    miu_pusher_slider=0.3, contact_time=0., is_rect_flag=None) -> None:
        self.miu_list = miu_list  # friction coeff between all objects and target
        self.geom_list = geom_list  # geometry of all objects
        self.A_list = A_list  # limit surface of all obstacles
        self.geom_target = geom_target  # geometry of the target object
        self.A_target = A_target  # limit surface of the target object
        self.miu_pusher_slider = miu_pusher_slider  # the friction coefficient between pusher and slider
        self.contact_time = contact_time
        if is_rect_flag is None:
            self.is_rect_flag = [True for i in range(len(self.geom_list))]  # same length as geom_list
        else:
            self.is_rect_flag = is_rect_flag
        # FIXME: more to be added

class PlanningScene:
    """
    Underlying class of NodeHybrid, including contact_flag, all polygons, and obstacle
    states
    """
    def __init__(self, in_contact=False, target_polygon=None, polygons=None, states=None, types=None) -> None:
        self.in_contact = in_contact
        self.target_polygon = target_polygon
        self.polygons = polygons
        self.states = states
        self.types = types
        self.goal_poly = None

def load_planning_scene_from_file(scene_pkl):
    """
    Load planning scene from pickle file
    :param scene_pkl: pickle file path
    :return: PlanningScene object
    :return: ContactBasic object
    """
    raw = pickle.load(open(scene_pkl, 'rb'))
    target_polygon = gen_polygon(coord=raw['target']['x'], geom=raw['target']['geom'], type='box')
    obstacle_states, obstacle_polygons = [], []
    
    for i in range(raw['obstacle']['num']):
        obstacle_states.append(raw['obstacle']['x'][i])
        
        ## for box shape only
        if not 'shape' in raw['obstacle']:
            obstacle_polygons.append(gen_polygon(coord=raw['obstacle']['x'][i],
                                                 geom=raw['obstacle']['geom'][i],
                                                 type='box'))

        ## for general polygon shapes
        else:
            obstacle_polygons.append(gen_polygon(coord=raw['obstacle']['x'][i],
                                                 geom=raw['obstacle']['geom'][i],
                                                 type=raw['obstacle']['shape'][i]))

    if 'type' in raw['obstacle']:
        types = raw['obstacle']['type']
    else:
        types = [1] * raw['obstacle']['num']

    # prepare planning scene and basic information
    scene = PlanningScene(in_contact=False,
                          target_polygon=target_polygon,
                          polygons=obstacle_polygons,
                          states=obstacle_states,
                          types=types)

    info = ContactBasic(miu_list=raw['obstacle']['miu'],
                        geom_list=raw['obstacle']['geom'],
                        A_list=raw['obstacle']['A_list'],
                        contact_time=raw['contact']['dt'],
                        geom_target=raw['target']['geom'])

    if 'goal' in raw:
        scene.goal_poly=raw['goal']

    return scene, info

def collision_check_and_contact_reconfig(basic:ContactBasic, scene:PlanningScene, state_list):
    """
    Collision check and contact reconfiguration
    :param basic: the basic information in contact analysis
    :param scene: the current planning scene
    :param state_list: the list of target polygon states in the following steps
    :return: flag - in contact or not in the following steps
    :return: flag - can extend or not
                    if obstacles in contact collide with other obstacles, extension failed
    :return: new scene - new PlanningScene object
    """
    # copy new planning scene
    new_scene = copy.deepcopy(scene)
    # create the target object
    if not isinstance(state_list, np.ndarray):
        target_state = np.array(state_list)
    else:
        target_state = state_list
    target_state = np.atleast_2d(target_state)[:, :3]

    target_polygon_next = gen_polygon(target_state[-1, :], basic.geom_target, 'box')
    new_scene.target_polygon = target_polygon_next
    
    # first step velocity
    target_velocity = (target_state[1, :] - target_state[0, :]) / (basic.contact_time / (target_state.shape[0]-1))
    
    # contact check
    in_contact_flag, polygons_in_contact, states_in_contact = \
        get_polygon_in_collision(target_state=target_state,
                                 target_poly=scene.target_polygon,
                                 state_of_poly=scene.states,
                                 list_of_poly=scene.polygons)

    # check if collision with static obstacles
    if polygons_in_contact is not None:
        for idx, polygon in enumerate(polygons_in_contact):
            if polygon is not None and new_scene.types[idx] == 0:
                new_scene.in_contact = True
                return True, False, new_scene
    
    if not in_contact_flag:
        # no contact, can extend
        new_scene.in_contact = False
        return False, True, new_scene
    else:
        new_scene.in_contact = True
    
    # contact configuration
    import pdb; pdb.set_trace()
    contact_config = get_polygon_contact_configuration(target_state=target_state,
                                                       target_poly=scene.target_polygon,
                                                       state_of_poly=states_in_contact,
                                                       list_of_poly=polygons_in_contact)
    
    for idx in range(len(contact_config)):
        if contact_config[idx] is None:
            continue
        contact_config[idx]['basic']['dt'] = basic.contact_time
        contact_config[idx]['basic']['miu'] = basic.miu_list[idx]
        contact_config[idx]['obstacle']['A'] = basic.A_list[idx]
        contact_config[idx]['target']['vel'] = target_velocity
        contact_config[idx]['target']['geom'] = basic.geom_target
        contact_config[idx]['obstacle']['geom'] = basic.geom_list[idx]

    new_contact_config = update_contact_configuration(target_state=target_state,
                                                      contact_config=contact_config)

    for idx, config in enumerate(new_contact_config):
        if config is None:
            continue
        dx_o, dy_o, dtheta_o = config['obstacle']['dx']
        obstacle_poly = affinity.translate(affinity.rotate(new_scene.polygons[idx], dtheta_o, 'center', use_radians=True), dx_o, dy_o)
        # recover from possible pentration
        # import pdb; pdb.set_trace()

        # penetration_flag, new_obstacle_poly = recover_from_penetration(target_poly=target_polyon_old,
        #                                                                new_target_poly=target_polygon_next,
        #                                                                obstacle_poly=obstacle_poly_old,
        #                                                                new_obstacle_poly=obstacle_poly)
        # if penetration_flag:
        #     new_scene.polygons[idx] = new_obstacle_poly
        #     new_scene.states[idx] = np.append(np.array(new_obstacle_poly.centroid.xy), config['obstacle']['x'][2])
        # else:

        new_scene.polygons[idx] = obstacle_poly
        new_scene.states[idx] = config['obstacle']['x']

    # further collision check
    for idx, polygon in enumerate(new_scene.polygons):
        other_polygons = copy.deepcopy(new_scene.polygons)
        other_polygons.pop(idx)
        # collide with other polygons
        if not MultiPolygon(other_polygons).is_valid:
            # in contact, cannot extend
            return True, False, new_scene
        # collide with other polygons
        if intersects(polygon, MultiPolygon(other_polygons)):
            # in contact, cannot extend
            return True, False, new_scene
    
    # in contact, can extend
    return True, True, new_scene

def collision_check_and_contact_reconfig_v2(basic:ContactBasic, scene:PlanningScene, state_list, 
                                            p_pusher, v_pusher):
    """
    Collision check and contact reconfiguration
    :param basic: the basic information in contact analysis
    :param scene: the current planning scene
    :param state_list: the list of target polygon states in the following steps
    :return: flag - in contact or not in the following steps
    :return: flag - can extend or not
                    if obstacles in contact collide with other obstacles, extension failed
    :return: new scene - new PlanningScene object
    """
    # copy new planning scene
    new_scene = copy.deepcopy(scene)
    # create the target object
    if not isinstance(state_list, np.ndarray):
        target_state = np.array(state_list)
    else:
        target_state = state_list
    target_state = np.atleast_2d(target_state)[:, :3]

    target_polygon_next = gen_polygon(target_state[-1, :], basic.geom_target, 'box')
    new_scene.target_polygon = target_polygon_next
    
    # first step velocity
    target_velocity = (target_state[1, :] - target_state[0, :]) / (basic.contact_time / (target_state.shape[0]-1))
    
    # contact check
    in_contact_flag, polygons_in_contact, states_in_contact = \
        get_polygon_in_collision(target_state=target_state,
                                 target_poly=scene.target_polygon,
                                 state_of_poly=scene.states,
                                 list_of_poly=scene.polygons)

    # check if collision with static obstacles
    if polygons_in_contact is not None:
        for idx, polygon in enumerate(polygons_in_contact):
            if polygon is not None and new_scene.types[idx] == 0:
                new_scene.in_contact = True
                return True, False, None, None, new_scene
    
    if not in_contact_flag:
        # no contact, can extend
        new_scene.in_contact = False
        return False, True, target_state[-1][:3], state_list, new_scene
    else:
        new_scene.in_contact = True
    
    # contact configuration
    # import pdb; pdb.set_trace()
    M, N, P = get_polygon_contact_configuration_v2(target_state=target_state,
                                                   target_poly=scene.target_polygon,
                                                   state_of_poly=states_in_contact,
                                                   list_of_poly=polygons_in_contact,
                                                   p_pusher=p_pusher, 
                                                   v_pusher=v_pusher,
                                                   fric_coeff_pusher=basic.miu_pusher_slider,
                                                   fric_coeff=basic.miu_list,
                                                   limit_surface=[basic.A_target]+basic.A_list)
    nc = int(M.shape[0] / 4)
    contact_force = solve_contact_force(M=M, q=N.squeeze()).flatten()[:3*nc]
    block_v = np.matmul(P, contact_force)
    block_current_state = np.concatenate((target_state[0].reshape(-1, 3), np.array(scene.states).reshape(-1, 3)), axis=0)
    block_next_state = move_blocks_one_step(block_current_state, block_v, dt=basic.contact_time)

    state_list_updated = update_state_list(state_list=np.array(state_list), velocity=block_v[0], dt=basic.contact_time)

    # import pdb; pdb.set_trace()
    block_geoms = [basic.geom_target]+basic.geom_list
    block_next_state_updated = collision_reaction(states=block_next_state, 
                                                  geoms=block_geoms, 
                                                  fixed_idx=0)
    # plot_polygons_consecutive_timesteps(states1=block_current_state, states2=block_next_state_updated, 
    #                                     orig_state_list=state_list, 
    #                                     geoms=block_geoms)

    for idx in range(block_next_state_updated.shape[0]):
        if idx == 0:
             continue
        
        if len(block_geoms[idx]) <= 2:
            obstacle_poly = gen_polygon(block_next_state_updated[idx], block_geoms[idx], type="box")
        else:
            obstacle_poly = gen_polygon(block_next_state_updated[idx], block_geoms[idx], type="polygon")

        new_scene.polygons[idx-1] = obstacle_poly
        new_scene.states[idx-1] = block_next_state_updated[idx]

    target_polygon_updated = gen_polygon(block_next_state_updated[0], block_geoms[0], type="box")
    new_scene.target_polygon = target_polygon_updated

    # further collision check
    for idx, polygon in enumerate(new_scene.polygons):
        other_polygons = copy.deepcopy(new_scene.polygons)
        other_polygons.pop(idx)
        # collide with other polygons
        if not MultiPolygon(other_polygons).is_valid:
            # in contact, cannot extend
            return True, False, None, None, new_scene
        # collide with other polygons
        if intersects(polygon, MultiPolygon(other_polygons)):
            # in contact, cannot extend
            return True, False, None, None, new_scene
    
    # in contact, can extend
    return True, True, block_next_state_updated[0], state_list_updated, new_scene

def visualize_scene(scene:PlanningScene, fig=None, ax=None, alpha=1.0, \
                    xlim=[0.0, 0.5], ylim=[0.0, 0.5], movability=None, background=None, scene_index=0):
    """
    Visualize the planning scene
    :param scene: the planning scene
    :return: fig & ax, if not provided
    """
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if background is not None:
        img = plt.imread(background)
        ax.imshow(img, extent=xlim+ylim)

    cmap = plt.cm.Pastel2
    for idx, polygon in enumerate(scene.polygons):
        if bool(movability[idx]) is False:
            obs_patch = patches.Polygon(np.array(polygon.exterior.coords.xy).T, facecolor=cmap(idx), alpha=alpha, edgecolor='black', linewidth=3, hatch='x')
        else:
            obs_patch = patches.Polygon(np.array(polygon.exterior.coords.xy).T, facecolor=cmap(idx), alpha=alpha, edgecolor='black', linewidth=3)
        ax.add_artist(obs_patch)

    target_patch = patches.Polygon(np.array(scene.target_polygon.exterior.coords.xy).T, facecolor='#1f77b4', alpha=alpha, edgecolor='black', linewidth=3)
    ax.add_patch(target_patch)

    # visualize goal
    if scene.goal_poly is not None:
        target_patch = patches.Polygon(np.array(scene.goal_poly.exterior.coords.xy).T, facecolor='#1f77b4', alpha=alpha, edgecolor='black', linewidth=3, linestyle='--')
        ax.add_patch(target_patch)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xticks([])
    plt.yticks([])

    plt.title('scene{0}'.format(scene_index))

    # plt.xlim([0.3, 0.9])
    # plt.ylim([-0.3, 0.3])

    plt.gca().set_aspect('equal')
    # plt.grid('on')

    return fig, ax    

if __name__ == '__main__':
    ## -------------------------------------------------
    ## SETTINGS
    ## -------------------------------------------------
    # Planning Scene Unit Test
    # --------------------------------------------------
    # # x_init = [0.25, 0.05, 0.5*np.pi]
    # x_init = [0.18, 0.16, 0.1*np.pi+0.01]
    # x_init = [0.32, 0.13, 0.1*np.pi+0.01]

    # R3T-Contact Test
    # --------------------------------------------------
    x_init = [0.25, 0.05, 0.5*np.pi]

    # real robot experiment
    # quat_init = [-0.0087054, -0.0025106, 0.00057546, 0.99996]
    # theta_init = R.from_quat(quat_init).as_euler('xyz')[2]
    # print('theta0: ', theta_init)
    # x_init = [0.3836, 0.0014, theta_init]

    # # obstacle settings
    # # --------------------------------------------------
    # num_obs = 3

    # # other settings
    # # --------------------------------------------------
    miu_default = 0.3
    geom_default = [0.07, 0.12]  # box

    ## scene0
    pkl_file = '/home/yongpeng/research/R3T_shared/data/test_scene_0.pkl'
    num_obs = 3
    x_obs = [[0.1, 0.25, 0.6*np.pi],
             [0.25, 0.25, 0.4*np.pi],
             [0.4, 0.25, 0.6*np.pi]]
    movability = [1, 1, 1]

    __geom = cs.SX.sym('geom', 2)
    __c = rect_cs(__geom[0], __geom[1])/(__geom[0]*__geom[1])
    __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
    __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2)
    A = cs.Function('A', [__geom], [__A])

    lim_surf_A_obs = A(geom_default).toarray()

    ## scene1
    # pkl_file = '/home/yongpeng/research/R3T_shared/data/test_scene_1.pkl'
    # num_obs = 5
    # x_obs = [[0.12, 0.20, 0.75*np.pi],
    #          [0.25, 0.30, 0.0*np.pi],
    #          [0.38, 0.20, 0.25*np.pi],
    #          [0.12, 0.40, -0.75*np.pi],
    #          [0.38, 0.40, -0.25*np.pi]]
    # geom_obs = [[0.06, 0.12], \
    #             [[0.03, 0.03*np.sqrt(3)], [0.06, 0], [0.03, -0.03*np.sqrt(3)], [-0.03, -0.03*np.sqrt(3)], [-0.06, 0], [-0.03, 0.03*np.sqrt(3)]], \
    #             [0.06, 0.12], \
    #             [0.06, 0.12], \
    #             [0.06, 0.12]]
    # shape_obs = ['box', 'polygon', 'box', 'box', 'box']
    # type_obs = [1, 0, 1, 1, 1]
    # movability = type_obs
    # A_list = []
    # for i in range(num_obs):
    #     if shape_obs[i] == 'box':
    #         A_list.append(A(geom_obs[i]).toarray())
    #     else:
    #         A_list.append(None)

    ## scene2
    # pkl_file = '/home/yongpeng/research/R3T_shared/data/test_scene_2.pkl'
    # num_obs = 5
    # x_obs = [[0.1, 0.05, 0.0*np.pi],
    #          [0.4, 0.05, 0.0*np.pi],
    #          [0.25, 0.20, 0.0*np.pi],
    #          [0.1, 0.35, 0.0*np.pi],
    #          [0.4, 0.35, 0.0*np.pi]]
    # geom_obs = [[0.06, 0.06], \
    #             [0.06, 0.06], \
    #             [0.06, 0.06], \
    #             [0.06, 0.06], \
    #             [0.06, 0.06]]
    # shape_obs = ['box', 'box', 'box', 'box', 'box']
    # type_obs = [0, 0, 1, 1, 1]
    # movability = type_obs
    # A_list = []
    # for i in range(num_obs):
    #     if shape_obs[i] == 'box':
    #         A_list.append(A(geom_obs[i]).toarray())
    #     else:
    #         A_list.append(None)

    ## scene3
    # pkl_file = '/home/yongpeng/research/R3T_shared/data/test_scene_3.pkl'
    # num_obs = 6
    # x_obs = [[0.10, 0.25, 0.3*np.pi],
    #          [0.24, 0.25, -0.2*np.pi],
    #          [0.08, 0.08, 0.0*np.pi],
    #          [0.10, 0.42, 0.0*np.pi],
    #          [0.33, 0.33, 0.32*np.pi],
    #          [0.40, 0.11, -0.13*np.pi]]
    # geom_obs = [[[0.03, 0.03*np.sqrt(3)], [0.06, 0], [0.03, -0.03*np.sqrt(3)], [-0.03, -0.03*np.sqrt(3)], [-0.06, 0], [-0.03, 0.03*np.sqrt(3)]], \
    #             [[0.03, 0.03*np.sqrt(3)], [0.06, 0], [0.03, -0.03*np.sqrt(3)], [-0.03, -0.03*np.sqrt(3)], [-0.06, 0], [-0.03, 0.03*np.sqrt(3)]], \
    #             [0.06, 0.06], \
    #             [0.06, 0.06], \
    #             [0.06, 0.06], \
    #             [0.06, 0.06]]
    # shape_obs = ['polygon', 'polygon', 'box', 'box', 'box', 'box']
    # type_obs = [0, 0, 0, 0, 1, 1]
    # movability = type_obs
    # A_list = []
    # for i in range(num_obs):
    #     if shape_obs[i] == 'box':
    #         A_list.append(A(geom_obs[i]).toarray())
    #     else:
    #         A_list.append(None)

    ## scene4
    # pkl_file = '/home/yongpeng/research/R3T_shared/data/test_scene_4.pkl'
    # num_obs = 3
    # x_obs = [[0.08, 0.32, 0.4*np.pi],
    #          [0.38, 0.32, 0.7*np.pi],
    #          [0.28, 0.18, 0.2*np.pi]]
    # geom_obs = [[[0, 0.08], [0.04*np.sqrt(3), -0.04], [-0.04*np.sqrt(3), -0.04]], \
    #             [[0.03, 0.03*np.sqrt(3)], [0.06, 0], [0.03, -0.03*np.sqrt(3)], [-0.03, -0.03*np.sqrt(3)], [-0.06, 0], [-0.03, 0.03*np.sqrt(3)]], \
    #             [[0.04, 0.05], [-0.04, 0.05], [-0.06, -0.05], [0.06, -0.05]]]
    # shape_obs = ['polygon', 'polygon', 'polygon']
    # type_obs = [0, 0, 0]
    # A_list = [None for i in range(num_obs)]
    # movability = type_obs

    # real robot experiment
    # quat_obs = [[-0.01064, -0.014352, -0.0031915, 0.99984],
    #             [0.0013715, -0.0074795, 0.38732, 0.92192],
    #             [-0.012429, -0.00096422, 0.42742, 0.90397]]
    # x_obs = [[0.5815, -0.0181],
    #          [0.6836, 0.1534],
    #          [0.4941, -0.1999]]
    # theta_obs = R.from_quat(quat_obs).as_euler('xyz')
    # x_obs = np.concatenate((x_obs, theta_obs[:, 2].reshape(-1, 1)), axis=1).tolist()
    # A_list = [None for i in range(num_obs)]

    # create planning scene and save to file
    # --------------------------------------------------
    # pkl_file = '/home/yongpeng/research/R3T_shared/data/debug/real_obstacle_avoidance_robot/planning_scene_robot.pkl'

    dt_contact = 0.05
    x_goal = [0.25, 0.45, 0.5*np.pi]
    goal_poly = gen_polygon(x_goal, geom_default,'box')

    scene_pkl = {'target': {'x': x_init, 'geom': geom_default},
                 'obstacle': {'num': num_obs,
                              'miu': [miu_default for i in range(num_obs)],
                              'geom': [geom_default for i in range(num_obs)],
                              'A_list': [lim_surf_A_obs for i in range(num_obs)],
                              'x': x_obs},
                 'goal': goal_poly,
                 'contact': {'dt': dt_contact}}

    # scene_pkl = {'target': {'x': x_init, 'geom': [0.08, 0.15]},
    #              'obstacle': {'num': num_obs,
    #                           'miu': [miu_default for i in range(num_obs)],
    #                           'geom': [[0.07, 0.122], [0.1, 0.102], [0.1, 0.102]],
    #                           'A_list': [A([0.07, 0.122]).toarray(), A([0.1, 0.102]).toarray(), A([0.1, 0.102]).toarray()],
    #                           'type': [1, 0, 0],
    #                           'x': x_obs},
    #              'contact': {'dt': dt_contact}}

    # scene_pkl = {'target': {'x': x_init, 'geom': geom_default},
    #              'obstacle': {'num': num_obs,
    #                           'miu': [miu_default for i in range(num_obs)],
    #                           'geom': geom_obs,
    #                           'A_list': A_list,
    #                           'type': type_obs,
    #                           'shape': shape_obs,
    #                           'x': x_obs},
    #              'goal': goal_poly,
    #              'contact': {'dt': dt_contact}}

    pickle.dump(scene_pkl, open(pkl_file, 'wb'))

    scene, basic = load_planning_scene_from_file(pkl_file)
    fig, ax = visualize_scene(scene, alpha=0.75, movability=movability, background='/home/yongpeng/下载/wood.jpg',
                              scene_index=0)
    plt.savefig('./scene0_visual.pdf')
    plt.show()
    
    exit(0)

    ## -------------------------------------------------
    ## TEST UPDATE CONTACT CONFIGURATION
    ## -------------------------------------------------

    contact_config = pickle.load(open('/home/yongpeng/research/R3T_shared/data/debug/contact_config.pkl', 'rb'))
    target_state = np.load('/home/yongpeng/research/R3T_shared/data/debug/target_state.npy')
    contact_config[2]['obstacle']['x'] = [0.4, 0.25, 0.6*np.pi]
    contact_config[2]['obstacle']['polygon'] = gen_polygon(contact_config[2]['obstacle']['x'], [0.07, 0.12], 'box')
    contact_config[2]['obstacle']['abs_p']
    import pdb; pdb.set_trace()
    new_contact_config = update_contact_configuration(target_state, contact_config)

    exit(0)

    ## -------------------------------------------------
    ## TEST FORWARD SIMULATION
    ## -------------------------------------------------

    # move forward
    # --------------------------------------------------
    displacement = np.append(0.3*0.05*np.array([np.cos(x_init[2]+0.5*np.pi), np.sin(x_init[2]+0.5*np.pi)]), 0.) / 5
    
    ## ROUND 1
    state_list = []
    for i in range(5):
        state_list.append(x_init+i*displacement)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    ## ROUND 2
    state_list = []
    for i in range(4, 9):
        state_list.append(x_init+i*displacement)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    ## ROUND 3
    state_list = []
    for i in range(8, 13):
        state_list.append(x_init+i*displacement)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    ## ROUND 4
    state_list = []
    for i in range(12, 17):
        state_list.append(x_init+i*displacement)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    ## ROUND 5
    state_list = []
    for i in range(16, 21):
        state_list.append(x_init+i*displacement)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # ## ROUND 6
    # state_list = []
    # for i in range(20, 25):
    #     state_list.append(x_init+i*displacement)
    # state_list = np.array(state_list)
    # in_contact_flag, can_extend_flag, new_scene = \
    #     collision_check_and_contact_reconfig(basic, new_scene, state_list)
    # print('in contact: ', in_contact_flag)
    # print('can extend: ', can_extend_flag)
    # fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # state_list = []
    # for i in range(24, 29):
    #     state_list.append(x_init+i*displacement)
    # state_list = np.array(state_list)
    # in_contact_flag, can_extend_flag, new_scene = \
    #     collision_check_and_contact_reconfig(basic, new_scene, state_list)
    # print('in contact: ', in_contact_flag)
    # print('can extend: ', can_extend_flag)
    # fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # state_list = []
    # for i in range(28, 33):
    #     state_list.append(x_init+i*displacement)
    # state_list = np.array(state_list)
    # in_contact_flag, can_extend_flag, new_scene = \
    #     collision_check_and_contact_reconfig(basic, new_scene, state_list)
    # print('in contact: ', in_contact_flag)
    # print('can extend: ', can_extend_flag)
    # fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # state_list = []
    # for i in range(32, 37):
    #     state_list.append(x_init+i*displacement)
    # state_list = np.array(state_list)
    # in_contact_flag, can_extend_flag, new_scene = \
    #     collision_check_and_contact_reconfig(basic, new_scene, state_list)
    # print('in contact: ', in_contact_flag)
    # print('can extend: ', can_extend_flag)
    # fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # state_list = []
    # for i in range(36, 41):
    #     state_list.append(x_init+i*displacement)
    # state_list = np.array(state_list)
    # in_contact_flag, can_extend_flag, new_scene = \
    #     collision_check_and_contact_reconfig(basic, new_scene, state_list)
    # print('in contact: ', in_contact_flag)
    # print('can extend: ', can_extend_flag)
    # fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # move right
    # --------------------------------------------------
    displacement2 = np.append(0.3*0.05*np.array([np.cos(x_init[2]), np.sin(x_init[2])]), 0.) / 5

    x_state_now = state_list[-1, :]
    state_list = []
    for i in range(0, 5):
        state_list.append(x_state_now+i*displacement2)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    state_list = []
    for i in range(4, 9):
        state_list.append(x_state_now+i*displacement2)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    state_list = []
    for i in range(8, 13):
        state_list.append(x_state_now+i*displacement2)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    state_list = []
    for i in range(12, 17):
        state_list.append(x_state_now+i*displacement2)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    state_list = []
    for i in range(16, 21):
        state_list.append(x_state_now+i*displacement2)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    state_list = []
    for i in range(20, 25):
        state_list.append(x_state_now+i*displacement2)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # state_list = []
    # for i in range(24, 29):
    #     state_list.append(x_state_now+i*displacement2)
    # state_list = np.array(state_list)
    # in_contact_flag, can_extend_flag, new_scene = \
    #     collision_check_and_contact_reconfig(basic, new_scene, state_list)
    # print('in contact: ', in_contact_flag)
    # print('can extend: ', can_extend_flag)

    # move forward again
    # --------------------------------------------------

    # ## ROUND 7
    x_state_now = state_list[-1, :]
    state_list = []
    for i in range(0, 5):
        state_list.append(x_state_now+i*displacement)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # ## ROUND 8
    state_list = []
    for i in range(4, 9):
        state_list.append(x_state_now+i*displacement)
    state_list = np.array(state_list)
    in_contact_flag, can_extend_flag, new_scene = \
        collision_check_and_contact_reconfig(basic, new_scene, state_list)
    print('in contact: ', in_contact_flag)
    print('can extend: ', can_extend_flag)
    # fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.25)

    # ## ROUND 9
    # state_list = []
    # for i in range(32, 37):
    #     state_list.append(x_init+i*displacement)
    # state_list = np.array(state_list)
    # in_contact_flag, can_extend_flag, new_scene = \
    #     collision_check_and_contact_reconfig(basic, new_scene, state_list)
    # print('in contact: ', in_contact_flag)
    # print('can extend: ', can_extend_flag)
    fig, ax = visualize_scene(new_scene, fig, ax, alpha=0.5)

    plt.show()

    ## --------------------------------------------------------
    ## TEST CONTACT LOCATION DETECTOR
    ## --------------------------------------------------------

    # sp = Polygon([[1,0],[0,1],[-1,0],[0,-1],[1,0]])

    # mp0 = Polygon([[0.0,1],[1.0,1],[1.0,2],[0.0,2],[0.0,1]])
    # mp1 = Polygon([[0.0,-1],[1.0,-1],[1.0,0],[0.0,0],[0.0,-1]])

    # mp0 = Polygon([[1.5,-0.5],[2.5,-0.5],[2.5,0.5],[1.5,0.5],[1.5,-0.5]])
    # mp1 = Polygon([[0.5,-0.5],[1.5,-0.5],[1.5,0.5],[0.5,0.5],[0.5,-0.5]])

    # mp0 = Polygon([[1.0,0.5],[1.5,1.0],[1.0,1.5],[0.5,1.0],[1.0,0.5]])
    # mp1 = Polygon([[1.5,0.0],[1.0,0.5],[0.5,0.0],[1.0,-0.5],[1.5,0.0]])

    # mp0 = Polygon([[1.0,0.5],[1.5,1.0],[1.0,1.5],[0.5,1.0],[1.0,0.5]])
    # mp1 = Polygon([[1.0,-0.5],[0.5,-1.0],[1.0,-1.5],[1.5,-1.0],[1.0,-0.5]])

    # sp = gen_polygon(np.array([0.25, 0.25, 1.2566370614359172]), [0.07, 0.12], 'box')
    # mp0 = gen_polygon(np.array([0.24647304, 0.17610421, 1.27488887]), [0.07, 0.12], 'box')
    # mp1 = gen_polygon(np.array([0.24961313, 0.19094834, 1.20993865]), [0.07, 0.12], 'box')

    # scene0 = PlanningScene(target_polygon=sp, polygons=[mp0])
    # scene1 = PlanningScene(target_polygon=sp, polygons=[mp1])
    # # m_state_list = np.linspace(np.append(np.array(mp0.centroid.xy), 0.), np.append(np.array(mp1.centroid.xy), 0.5*np.pi), 100)
    
    # m_state_list = np.array([[0.24647304, 0.17610421, 1.27488887],
    #                          [0.24703759, 0.17908619, 1.26651328],
    #                          [0.2476271 , 0.18206334, 1.25583671],
    #                          [0.24824836, 0.18503403, 1.242855  ],
    #                          [0.24890813, 0.1879964 , 1.22755984],
    #                          [0.24961313, 0.19094834, 1.20993865]])

    # s_state = np.array([0.25, 0.25, 1.2566370614359172])
    
    # # s_state = np.array([0, 0, 0])
    # m_coll_point, s_coll_point = contact_location_detector(m_state_list, mp0, s_state, sp)
    # import pdb; pdb.set_trace()

    # fig, ax = visualize_scene(scene0, alpha=0.25)
    # visualize_scene(scene1, fig, ax, alpha=0.5)

    # m_coll_point = np.array(m_coll_point).reshape(-1)
    # s_coll_point = np.array(s_coll_point).reshape(-1)

    # m_coll_point = m_state_list[0,:2] + rotation_matrix(m_state_list[0,2])@m_coll_point
    # s_coll_point = s_state[:2] + rotation_matrix(s_state[2])@s_coll_point

    # ax.scatter(m_coll_point[0], m_coll_point[1])
    # ax.scatter(s_coll_point[0], s_coll_point[1])

    # plt.show()
