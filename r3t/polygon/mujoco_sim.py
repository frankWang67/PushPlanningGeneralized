import mujoco
import numpy as np
import scipy.spatial.transform.rotation as R
import copy

from r3t.polygon.scene import ContactBasic, PlanningScene
from polytope_symbolic_system.common.symbolic_system import PushDTHybridSystem
from polytope_symbolic_system.common.utils import gen_polygon

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

class MujocoSimulator:
    def __init__(self, target_state: np.ndarray, info: ContactBasic, scene: PlanningScene, sys: PushDTHybridSystem, contact_face, time_step=0.01):
        if isinstance(target_state, list):
            target_state = np.array(target_state)
        try:
            assert isinstance(target_state, np.ndarray)
            assert isinstance(info, ContactBasic)
            assert isinstance(scene, PlanningScene)
        except AssertionError:
            raise TypeError(f"Arguments `target_state`, `info` and `scene` must be `np.ndarray`, `ContactBasic` and `PlanningScene` instances respectively, now they are {type(info)} and {type(scene)}.")

        self.target_state = target_state[:3]
        self.info = info
        self.scene = scene
        self.sys = sys
        self.time_step = time_step
        xml = f"""
            <mujoco model="pushing">
                <option timestep="{time_step}"/>

                <worldbody>
                    <light name="top" pos="0 0 1"/>

                    <geom type="plane" pos="0.0 0.0 -0.01" size="1.0 1.0 0.01" rgba="0.8 0.8 0.8 1"/>

                    <body name="pusher" mocap="true" pos="0.0 0.0 0.1">
                        <geom type="cylinder" size="{sys.slider_geom[2]} 0.15" rgba="1 0 0 1"/>
                    </body>
                </worldbody>
            </mujoco>
        """
        self.spec = mujoco.MjSpec()
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
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[info.geom_list[i][0] / 2.0, info.geom_list[i][1] / 2.0, 0.025],
                rgba=[0, 1, 0, 1],
            )
            obs_joint = obs_body.add_joint(
                name=f'obs{i+1}_joint',
                type=mujoco.mjtJoint.mjJNT_FREE,
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
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[info.geom_target[0], info.geom_target[1], 0.025],
            rgba=[0, 0, 1, 1],
        )
        slider_joint = slider_body.add_joint(
            name='slider_joint',
            type=mujoco.mjtJoint.mjJNT_FREE,
        )

        # Add pusher
        pusher_pos = self.sys.get_pusher_location(target_state, contact_face)
        pusher_body = self.spec.find_body('pusher')
        pusher_body.pos = np.concatenate([pusher_pos, [0.1]])

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

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

            state_list_updated = self.target_state.reshape(-1, 3)
            steps = int(sim_time / self.time_step)
            if steps == 0:
                raise ValueError(f"MuJoCo simulation time step too small. Now it's {self.time_step}, while the simulation time is {sim_time}.")
            for i in range(steps):
                self.data.mocap_pos[mocap_id][0] += v_pusher[0] * self.time_step
                self.data.mocap_pos[mocap_id][1] += v_pusher[1] * self.time_step
                mujoco.mj_step(self.model, self.data)

                state = np.append(self.data.body("slider").xpos[:2], self.get_body_theta("slider"))
                state_list_updated = np.concatenate((state_list_updated, state.reshape(-1, 3)), axis=0)

            new_state_updated = state

            new_planning_scene = copy.deepcopy(self.scene)
            for i in range(len(new_planning_scene.states)):
                obs_name = f"obs{i+1}"
                obs_pos = self.data.body(obs_name).xpos[:2].tolist()
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
        contact_geom1_slider_idx = contact_geom1_slider_idx[self.data.contact.geom2[contact_geom1_slider_idx] == pusher_id]
        contact_geom2_slider_idx = np.where(self.data.contact.geom2 == slider_id)[0]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom2[contact_geom2_slider_idx] != world_id]
        contact_geom2_slider_idx = contact_geom2_slider_idx[self.data.contact.geom2[contact_geom2_slider_idx] == pusher_id]
        contact_slider_idx = np.concatenate((contact_geom1_slider_idx, contact_geom2_slider_idx))
        in_contact_flag = (contact_slider_idx.size > 0)

        return in_contact_flag
    
    def get_body_theta(self, body_name):
        quat_wxyz = self.data.body(body_name).xquat
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        theta = R.Rotation.from_quat(quat_xyzw).as_euler("xyz")[2]

        return theta