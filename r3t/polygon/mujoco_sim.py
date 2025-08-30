import mujoco as mj
import numpy as np
import scipy.spatial.transform.rotation as R
import copy
import cv2

from r3t.polygon.scene import ContactBasic, PlanningScene
from polytope_symbolic_system.common.symbolic_system import PushDTHybridSystem
from polytope_symbolic_system.common.utils import gen_polygon

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

WIDTH = 640
HEIGHT = 480

OBS_HEIGHT = 0.03
PUSHER_HEIGHT = 0.15
DAMPING_RATIO = 1.0

class MujocoSimulator:
    """
    MuJoCo collision check reference: https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
    """
    def __init__(self, info: ContactBasic, scene: PlanningScene, sys: PushDTHybridSystem, contact_face, psic, time_step=0.01, time_scale=10, vis=False):
        try:
            assert isinstance(info, ContactBasic)
            assert isinstance(scene, PlanningScene)
        except AssertionError:
            raise TypeError(f"Arguments `target_state`, `info` and `scene` must be `np.ndarray`, `ContactBasic` and `PlanningScene` instances respectively, now they are {type(info)} and {type(scene)}.")

        self.info = info
        self.scene = scene
        self.sys = sys
        self.time_step = time_step
        self.time_scale = time_scale
        self.vis = vis

        mesh_str = "<asset>"
        vertex_str, face_str = self.get_vertices_and_faces(info.curve_target.pt_samples)
        mesh_str += f"<mesh name='slider' vertex='{vertex_str}' face='{face_str}' />"
        for i in range(len(info.curve_list)):
            vertex_str, face_str = self.get_vertices_and_faces(info.curve_list[i].pt_samples)
            mesh_str += f"<mesh name='obs{i+1}' vertex='{vertex_str}' face='{face_str}' />"
        mesh_str += "</asset>"

        xml = f"""
            <mujoco model="pushing">
                <option timestep="{time_step}"/>
                {mesh_str}

                <worldbody>
                    <light name="top" pos="0 0 1"/>

                    <body name="ground" pos="0.0 0.0 0.0">
                        <geom type="plane" size="2.0 2.0 0.01" rgba="0.8 0.8 0.8 1"/>
                    </body>

                    <body name="pusher" mocap="true" pos="0.0 0.0 {PUSHER_HEIGHT + 0.01}">
                        <geom type="cylinder" size="{sys.pusher_r} {PUSHER_HEIGHT}" rgba="1 0 0 1" solimp="0.9 0.95 0.001 0.5 2" solref="{time_step * 2} {DAMPING_RATIO}"/>
                    </body>
                </worldbody>
            </mujoco>
        """
        self.spec = mj.MjSpec()
        self.spec.from_string(xml)

        # Add obstacles
        self.fixed_obs_list = []
        self.fixed_obs_pos_list = []
        for i in range(len(scene.states)):
            obs_name = f'obs{i+1}'
            self.add_object(obs_name, scene.states[i], True)
            if not scene.types[i]:
                self.fixed_obs_list.append(obs_name)
                self.fixed_obs_pos_list.append(np.array(scene.states[i][:2]))

        # Add slider
        self.add_object('slider', scene.target_state, False)

        # Compile model
        self.model = self.spec.compile()
        self.data = mj.MjData(self.model)
        if self.vis:
            self.renderer = mj.Renderer(self.model, height=HEIGHT, width=WIDTH)
        mj.mj_forward(self.model, self.data)

        # Set pusher
        pusher_pos = self.sys.get_pusher_location(np.append(scene.target_state, psic), contact_face)
        self.mocap_id = int(self.model.body("pusher").mocapid)
        self.data.mocap_pos[self.mocap_id][0] = pusher_pos[0]
        self.data.mocap_pos[self.mocap_id][1] = pusher_pos[1]
        mj.mj_step(self.model, self.data)

    def add_object(self, name, state, obs):
        rgba = [0, 1, 0, 1] if obs else [0, 0, 1, 1]
        quat = self.get_quat_from_theta(state[2])
        body = self.spec.worldbody.add_body(
            name=name,
            pos=[state[0], state[1], 0.0],
            quat=quat,
        )
        geom = body.add_geom(
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=name,
            rgba=rgba,
        )
        # geom.solimp[3], geom.solimp[4] = 0.1, 6
        geom.solref[0] = self.time_step * 2
        geom.solref[1] = DAMPING_RATIO
        joint = body.add_joint(
            name=name+'_joint',
            type=mj.mjtJoint.mjJNT_FREE,
        )

    def set_object(self, name, state):
        quat = self.get_quat_from_theta(state[2])
        joint = self.data.joint(name+'_joint')
        joint.qpos[0] = state[0]
        joint.qpos[1] = state[1]
        joint.qpos[2] = 0.0
        joint.qpos[3:] = quat
        joint.qvel[:] = 0.0

    def reset_scene(self, scene: PlanningScene, contact_face, psic):
        # Set obstacles
        for i in range(len(scene.states)):
            self.set_object(f'obs{i+1}', scene.states[i])

        # Set slider
        self.set_object('slider', scene.target_state)

        # Set pusher
        pusher_pos = self.sys.get_pusher_location(np.append(scene.target_state, psic), contact_face)
        self.data.mocap_pos[self.mocap_id][0] = pusher_pos[0]
        self.data.mocap_pos[self.mocap_id][1] = pusher_pos[1]

        mj.mj_step(self.model, self.data)

    def simulate(self, path, contact_face, v_pusher, sim_time):
        """
        Simulate to update planning scene
        :param v_pusher: the velocity of the pusher
        :param sim_time: the simulation time
        :return: flag - in contact or not in the following steps
        :return: flag - can extend or not
                        if obstacles in contact collide with other obstacles, extension failed
        :return: new state updated - the updated state of the slider
        :return: state list updated - the updated path of the slider from the last node to the new one
        :return: new scene - new PlanningScene object
        """
        state_list_updated = self.get_target_state_with_psic().reshape(-1, 4)
        sim_time *= self.time_scale
        v_pusher /= self.time_scale
        steps = round(sim_time / self.time_step)
        if steps == 0:
            raise ValueError(f"MuJoCo simulation time step too small. Now it's {self.time_step}, while the simulation time is {sim_time}.")
        # if np.abs(path[-1][-1] - path[0][-1]) > np.pi:
        #     if path[-1][-1] < path[0][-1]:
        #         path[-1][-1] += 2*np.pi
        #     else:
        #         path[0][-1] += 2*np.pi
        # path_interp = np.linspace(path[0], path[-1], steps+1)
        for i in range(steps):
            self.data.mocap_pos[self.mocap_id][0] += v_pusher[0] * self.time_step
            self.data.mocap_pos[self.mocap_id][1] += v_pusher[1] * self.time_step
            # pusher_loc = self.sys.get_pusher_location(path_interp[i], contact_face)
            # self.data.mocap_pos[self.mocap_id][0] = pusher_loc[0]
            # self.data.mocap_pos[self.mocap_id][1] = pusher_loc[1]
            mj.mj_step(self.model, self.data)
            if i % self.time_scale == 0 or i == steps - 1:
                if self.vis:
                    self.renderer.update_scene(self.data)
                    img = self.renderer.render()
                    cv2.imshow("Pushing Simulation", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        exit(0)

                # pusher_slider_in_contact = self.get_slider_pusher_contact_flag()
                state_with_psic = self.get_target_state_with_psic()
                state_list_updated = np.concatenate((state_list_updated, state_with_psic.reshape(-1, 4)), axis=0)

        if self.vis:
            cv2.destroyAllWindows()

        new_state_updated = state_with_psic

        new_planning_scene = copy.deepcopy(self.scene)
        for i in range(len(new_planning_scene.states)):
            obs_name = f"obs{i+1}"
            obs_pos = self.get_body_xy_array(obs_name).tolist()
            obs_pos.append(self.get_body_theta(obs_name))
            new_planning_scene.states[i] = obs_pos
            new_planning_scene.polygons[i] = gen_polygon(coord=obs_pos, bbox=self.info.bbox_list[i])
        new_planning_scene.target_state = new_state_updated[:3]
        new_planning_scene.target_polygon = gen_polygon(coord=new_state_updated[:3], bbox=self.info.bbox_target)
        
        in_contact_flag = self.get_slider_contact_flag()
        new_planning_scene.in_contact = in_contact_flag
        for i in range(len(self.fixed_obs_list)):
            obs_name = self.fixed_obs_list[i]
            obs_state = self.data.body(obs_name).xpos[:2]
            if np.linalg.norm(obs_state - self.fixed_obs_pos_list[i]) > 1e-2:
                # print(f"{obs_name=}")
                # print(f"{obs_state=}")
                # print(f"{self.fixed_obs_pos_list[i]=}")
                return in_contact_flag, False, None, None, new_planning_scene

        return in_contact_flag, True, new_state_updated, state_list_updated, new_planning_scene
        
    def get_quat_from_theta(self, theta):
        quat_xyzw = R.Rotation.from_euler('xyz', [0.0, 0.0, theta]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return quat_wxyz
    
    def get_slider_pusher_contact_flag(self):
        pusher_id = self.data.body("pusher").id
        slider_id = self.data.body("slider").id
        contact_geom1_slider_idx = np.where(self.data.contact.geom1 == slider_id)[0]
        contact_geom1_slider_idx = contact_geom1_slider_idx[self.data.contact.geom2[contact_geom1_slider_idx] == pusher_id]
        contact_geom2_slider_idx = np.where(self.data.contact.geom2 == slider_id)[0]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom1[contact_geom2_slider_idx] == pusher_id]
        contact_slider_idx = np.concatenate((contact_geom1_slider_idx, contact_geom2_slider_idx))
        in_contact_flag = (contact_slider_idx.size > 0)

        return in_contact_flag
        
    def get_slider_contact_flag(self):
        world_id  = self.data.body("world").id
        ground_id = self.data.body("ground").id
        pusher_id = self.data.body("pusher").id
        slider_id = self.data.body("slider").id
        contact_geom1_slider_idx = np.where(self.data.contact.geom1 == slider_id)[0]
        contact_geom1_slider_idx = contact_geom1_slider_idx[self.data.contact.geom2[contact_geom1_slider_idx] != world_id]
        contact_geom1_slider_idx = contact_geom1_slider_idx[self.data.contact.geom2[contact_geom1_slider_idx] != ground_id]
        contact_geom1_slider_idx = contact_geom1_slider_idx[self.data.contact.geom2[contact_geom1_slider_idx] != pusher_id]
        contact_geom2_slider_idx = np.where(self.data.contact.geom2 == slider_id)[0]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom1[contact_geom2_slider_idx] != world_id]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom1[contact_geom2_slider_idx] != ground_id]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom1[contact_geom2_slider_idx] != pusher_id]
        contact_slider_idx = np.concatenate((contact_geom1_slider_idx, contact_geom2_slider_idx))
        in_contact_flag = (contact_slider_idx.size > 0)

        return in_contact_flag
    
    def get_obstacle_contact_flag(self, obs_name):
        world_id  = self.data.body("world").id
        ground_id = self.data.body("ground").id
        obs_id = self.data.body(obs_name).id
        contact_geom1_obs_idx = np.where(self.data.contact.geom1 == obs_id)[0]
        contact_geom1_obs_idx = contact_geom1_obs_idx[self.data.contact.geom2[contact_geom1_obs_idx] != world_id]
        contact_geom1_obs_idx = contact_geom1_obs_idx[self.data.contact.geom2[contact_geom1_obs_idx] != ground_id]
        contact_geom2_obs_idx = np.where(self.data.contact.geom2 == obs_id)[0]
        contact_geom2_obs_idx = contact_geom2_obs_idx[self.data.contact.geom1[contact_geom2_obs_idx] != world_id]
        contact_geom2_obs_idx = contact_geom2_obs_idx[self.data.contact.geom1[contact_geom2_obs_idx] != ground_id]
        contact_obs_idx = np.concatenate((contact_geom1_obs_idx, contact_geom2_obs_idx))
        in_contact_flag = (contact_obs_idx.size > 0)

        return in_contact_flag
    
    def get_body_xy_array(self, body_name):
        xy = self.data.body(body_name).xpos[:2]

        return xy
    
    def get_body_theta(self, body_name):
        quat_wxyz = self.data.body(body_name).xquat
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        theta = R.Rotation.from_quat(quat_xyzw).as_euler("xyz")[2]

        return theta
    
    def get_pusher_slider_xy(self, state, pusher_xy_world):
        x_world, y_world = pusher_xy_world[0], pusher_xy_world[1]
        slider_x, slider_y, slider_theta = state[0], state[1], state[2]
        T_mat = np.array([[np.cos(slider_theta), -np.sin(slider_theta), slider_x], 
                          [np.sin(slider_theta),  np.cos(slider_theta), slider_y], 
                          [                 0.0,                   0.0,      1.0]])
        T_inv = np.linalg.inv(T_mat)
        xy_world_aug = np.array([x_world, y_world, 1.0])
        xy_slider_aug = T_inv @ xy_world_aug
        x = xy_slider_aug[0]
        y = xy_slider_aug[1]

        return x, y
        
    def get_psic(self, state, pusher_xy_world):
        x, y = self.get_pusher_slider_xy(state, pusher_xy_world)
        psic = np.arctan2(y, x)

        return psic
    
    def get_target_state_with_psic(self):
        state = np.append(self.get_body_xy_array("slider"), self.get_body_theta("slider"))
        pusher_xy = self.get_body_xy_array("pusher")
        state_with_psic = np.append(state, self.get_psic(state, pusher_xy))

        return state_with_psic
    
    def get_vertices_and_faces(self, sample_pts):
        vertices = []
        vertices.append([0, 0, 0])
        vertices.append([0, 0, OBS_HEIGHT])
        for x, y in sample_pts:
            vertices.append([x, y, 0])
            vertices.append([x, y, OBS_HEIGHT])

        n = len(sample_pts)
        faces = []

        for i in range(n):
            b_current = 2 + 2 * i
            b_next = 2 + 2 * ((i + 1) % n)
            t_current = b_current + 1
            t_next = b_next + 1
            
            # 侧面三角形
            faces.append([b_current, b_next, t_current])
            faces.append([t_current, b_next, t_next])
            
            # 底面三角形（中心0 → 当前 → 下一个）
            faces.append([0, b_next, b_current])
            
            # 顶面三角形（中心1 → 当前 → 下一个）
            faces.append([1, t_current, t_next])

        vertex_str = ' '.join(map(str, np.array(vertices).flatten()))
        face_str = ' '.join([' '.join(map(str, f)) for f in faces])

        return vertex_str, face_str
