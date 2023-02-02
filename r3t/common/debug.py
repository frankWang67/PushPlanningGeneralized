import copy
from collections import namedtuple
import json
from matplotlib import pyplot as plt
import os
import time
from treelib import Tree

from r3t.polygon.scene import visualize_scene

Scene = namedtuple('Scene', ['url'])

class JSONDebugger:
    def __init__(self, scene_root='/home/yongpeng/research/R3T_shared/data/scene', \
                    json_root='/home/yongpeng/research/R3T_shared/data/json') -> None:
        self.data = None
        self.path = None
        self.scene_data = Tree()
        self.scene_root = scene_root
        self.json_root = json_root
        self.timestamp = time.strftime('%Y_%m_%d_%H_%M',time.localtime(int(round(time.time()*1000))/1000))

        # make directory
        os.mkdir(os.path.join(self.scene_root, self.timestamp))
        os.mkdir(os.path.join(self.json_root, self.timestamp))

    def add_node_data(self, new_node):
        try:
            # compute node id
            node_id = hash(str(new_node.state[:3]))
            if new_node.parent is not None:
                parent_id = hash(str(new_node.parent.state[:3]))
            else:
                parent_id = 'null'

            # compute state and input list from parent
            if new_node.parent is not None:
                state_list = []
                for i in range(len(new_node.path_from_parent)):
                    state_list.append(str(new_node.path_from_parent[i]))
                input_list = str([new_node.input_from_parent])
                mode_string = new_node.mode_from_parent
            else:
                state_list = []
                input_list = []
                mode_string = ('', '')

            plt.clf()
            visualize_scene(new_node.planning_scene, alpha=0.5)
            scene_url = os.path.join(self.scene_root, self.timestamp, '{0}_scene.png'.format(node_id))
            scene_url_via_http = os.path.join('http://127.0.0.1:8777', self.timestamp, '{0}_scene.png'.format(node_id))
            plt.savefig(scene_url)
            plt.close()

            node_data = {
                        'id': node_id,
                        'state': str(new_node.state),
                        'contact': {
                                    'face': mode_string[0],
                                    'mode': mode_string[1]
                                },
                        'parent': {
                                    'id': parent_id,
                                    'path': state_list,
                                    'input': input_list
                                },
                        'cost': {
                                'from_parent': new_node.cost_from_parent,
                                'from_root': new_node.cost_from_root
                                },
                        'scene': scene_url_via_http,
                        'child': {}
                        }

            # add to tree
            if parent_id is 'null':
                self.scene_data.create_node(node_id, node_id, parent=None, data=Scene(url=scene_url_via_http))
            else:
                self.scene_data.create_node(node_id, node_id, parent=parent_id, data=Scene(url=scene_url_via_http))

            # add the root node
            if self.data is None:
                self.data = copy.deepcopy(node_data)
                self.path = {}
                self.path[node_id] = []
            else:
                parent_path = copy.deepcopy(self.path[parent_id])
                node_ptr = self.data
                for id in parent_path:
                    node_ptr = node_ptr['child'][id]

                node_ptr['child'][node_id] = copy.deepcopy(node_data)

                parent_path.append(node_id)
                node_path = parent_path
                self.path[node_id] = node_path
        except Exception as e:
            print('JSONDebugger: caught exception:', e)
            import pdb; pdb.set_trace()

    def save(self):
        # save full info
        full_json_file = os.path.join(self.json_root, self.timestamp, '{0}_full.json'.format(self.timestamp))
        with open(full_json_file, 'w') as f:
            f.write(json.dumps(self.data, ensure_ascii=False, indent=4, separators=(',', ':')))
        
        print('JSONDebugger: planning log saved ({0})'.format(full_json_file))

        # save scene info
        scene_json_file = os.path.join(self.json_root, self.timestamp, '{0}_scene.json'.format(self.timestamp))
        with open(scene_json_file, 'w') as f:
            f.write(self.scene_data.to_json(with_data=True))
        print('JSONDebugger: scene index file saved ({0})'.format(scene_json_file))

        # save tree stucture
        scene_structure_file = os.path.join(self.json_root, self.timestamp, '{0}_tree.txt'.format(self.timestamp))
        self.scene_data.save2file(scene_structure_file)
        print('JSONDebugger: scene tree structure saved ({0})'.format(scene_structure_file))
