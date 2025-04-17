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

class MujocoSimulator:
    def __init__(self, target_state: np.ndarray, info: ContactBasic, scene: PlanningScene, sys: PushDTHybridSystem, contact_face, time_step=0.001, vis=False):
        if isinstance(target_state, list):
            target_state = np.array(target_state)
        try:
            assert isinstance(target_state, np.ndarray)
            assert isinstance(info, ContactBasic)
            assert isinstance(scene, PlanningScene)
        except AssertionError:
            raise TypeError(f"Arguments `target_state`, `info` and `scene` must be `np.ndarray`, `ContactBasic` and `PlanningScene` instances respectively, now they are {type(info)} and {type(scene)}.")

        self.target_state = target_state
        self.info = info
        self.scene = scene
        self.sys = sys
        self.time_step = time_step
        self.vis = vis

        xml = f"""
            <mujoco model="pushing">
                <option timestep="{time_step}"/>

                <worldbody>
                    <light name="top" pos="0 0 1"/>

                    <body name="ground" pos="0.0 0.0 0.0">
                        <geom type="plane" size="10.0 10.0 0.01" rgba="0.8 0.8 0.8 1"/>
                    </body>

                    <body name="pusher" mocap="true" pos="0.0 0.0 1.0">
                        <geom type="cylinder" size="{sys.slider_geom[2]} 0.15" rgba="1 0 0 1"/>
                    </body>
                </worldbody>
            </mujoco>
        """
        self.spec = mj.MjSpec()
        self.spec.from_string(xml)

        # Add obstacles
        for i in range(len(scene.states)):
            quat_xyzw = R.Rotation.from_euler('xyz', [0.0, 0.0, scene.states[i][2]]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            obs_body = self.spec.worldbody.add_body(
                name=f'obs{i+1}',
                pos=[scene.states[i][0], scene.states[i][1], 0.025],
                quat=quat_wxyz,
            )
            obs_geom = obs_body.add_geom(
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[info.geom_list[i][0] / 2.0, info.geom_list[i][1] / 2.0, 0.025],
                rgba=[0, 1, 0, 1],
            )
            obs_geom.solimp[3], obs_geom.solimp[4] = 0.1, 6
            obs_geom.solref[0] = 0.001
            obs_joint = obs_body.add_joint(
                name=f'obs{i+1}_joint',
                type=mj.mjtJoint.mjJNT_FREE,
            )

        # Add slider
        quat_xyzw = R.Rotation.from_euler('xyz', [0.0, 0.0, target_state[2]]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        slider_body = self.spec.worldbody.add_body(
            name='slider',
            pos=[target_state[0], target_state[1], 0.025],
            quat=quat_wxyz,
        )
        slider_geom = slider_body.add_geom(
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[info.geom_target[0] / 2.0, info.geom_target[1] / 2.0, 0.025],
            rgba=[0, 0, 1, 1],
        )
        slider_geom.solimp[3], slider_geom.solimp[4] = 0.1, 6
        slider_geom.solref[0] = 0.001
        slider_joint = slider_body.add_joint(
            name='slider_joint',
            type=mj.mjtJoint.mjJNT_FREE,
        )

        # Add pusher
        pusher_pos = self.sys.get_pusher_location(target_state, contact_face)
        pusher_body = self.spec.find_body('pusher')
        pusher_body.pos = np.concatenate([pusher_pos, [0.1]])

        self.model = self.spec.compile()
        self.data = mj.MjData(self.model)
        if self.vis:
            self.renderer = mj.Renderer(self.model, height=HEIGHT, width=WIDTH)
        mj.mj_forward(self.model, self.data)

        # print(f"{self.data.body('world').id=}")
        # print(f"{self.data.body('ground').id=}")
        # print(f"{self.data.body('pusher').id=}")
        # for i in range(len(scene.states)):
        #     print(f"{self.data.body(f'obs{i+1}').id=}")
        # print(f"{self.data.body('slider').id=}")
        # print(f"{self.data.contact.geom1=}")
        # print(f"{self.data.contact.geom2=}")
        # print(f"{self.data.contact.mu=}")
        # exit(0)

    def simulate(self, v_pusher, sim_time):
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
        try:
            mocap_id = int(self.model.body("pusher").mocapid)

            state_list_updated = self.target_state.reshape(-1, 4)
            steps = int(sim_time / self.time_step)
            if steps == 0:
                raise ValueError(f"MuJoCo simulation time step too small. Now it's {self.time_step}, while the simulation time is {sim_time}.")
            for i in range(steps):
                self.data.mocap_pos[mocap_id][0] += v_pusher[0] * self.time_step
                self.data.mocap_pos[mocap_id][1] += v_pusher[1] * self.time_step
                mj.mj_step(self.model, self.data)
                if self.vis:
                    self.renderer.update_scene(self.data)
                    img = self.renderer.render()
                    cv2.imshow("Pushing Simulation", img)
                    if cv2.waitKey(100) & 0xFF == ord('q'): 
                        exit(0)

                state = np.append(self.get_body_xy_array("slider"), self.get_body_theta("slider"))
                pusher_xy = self.get_body_xy_array("pusher")
                state_with_psic = np.append(state, self.get_psic(state, pusher_xy))
                state_list_updated = np.concatenate((state_list_updated, state_with_psic.reshape(-1, 4)), axis=0)

            # cv2.destroyAllWindows()

            new_state_updated = state

            new_planning_scene = copy.deepcopy(self.scene)
            for i in range(len(new_planning_scene.states)):
                obs_name = f"obs{i+1}"
                obs_pos = self.get_body_xy_array(obs_name).tolist()
                obs_pos.append(self.get_body_theta(obs_name))
                new_planning_scene.states[i] = obs_pos
                new_planning_scene.polygons[i] = gen_polygon(coord=obs_pos, geom=self.info.geom_list[i])
            new_planning_scene.target_polygon = gen_polygon(coord=new_state_updated, geom=self.info.geom_target)
            
            in_contact_flag = self.get_slider_contact_flag()
            new_planning_scene.in_contact = in_contact_flag

            return in_contact_flag, True, new_state_updated, state_list_updated, new_planning_scene
        except:
            return False, False, None, None, None
        
    def get_slider_contact_flag(self):
        world_id  = self.data.body("world").id
        pusher_id = self.data.body("pusher").id
        slider_id = self.data.body("slider").id
        contact_geom1_slider_idx = np.where(self.data.contact.geom1 == slider_id)[0]
        contact_geom1_slider_idx = contact_geom1_slider_idx[self.data.contact.geom2[contact_geom1_slider_idx] != world_id]
        contact_geom1_slider_idx = contact_geom1_slider_idx[self.data.contact.geom2[contact_geom1_slider_idx] != pusher_id]
        contact_geom2_slider_idx = np.where(self.data.contact.geom2 == slider_id)[0]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom1[contact_geom2_slider_idx] != world_id]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom1[contact_geom2_slider_idx] != pusher_id]
        contact_slider_idx = np.concatenate((contact_geom1_slider_idx, contact_geom2_slider_idx))
        in_contact_flag = (contact_slider_idx.size > 0)

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
