import copy
import json
from matplotlib import pyplot as plt
import os
import time

from r3t.polygon.scene import visualize_scene

class JSONDebugger:
    def __init__(self, scene_root='/home/yongpeng/research/R3T_shared/data/scene', \
                    json_root='/home/yongpeng/research/R3T_shared/data/json') -> None:
        self.data = None
        self.path = None
        self.scene_root = scene_root
        self.json_root = json_root
        self.timestamp = time.strftime('%Y_%m_%d_%H_%M',time.localtime(int(round(time.time()*1000))/1000))

        # make directory
        os.mkdir(os.path.join(self.scene_root, self.timestamp))

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
                state_list = [str(new_node.path_from_parent[0])]
                input_list = []
                for i in range(len(new_node.input_from_parent)):
                    input_list.append(str(new_node.input_from_parent[i]))
                    state_list.append(str(new_node.path_from_parent[i+1]))
                mode_string = new_node.mode_from_parent
            else:
                state_list = []
                input_list = []
                mode_string = ('', '')

            plt.clf()
            visualize_scene(new_node.planning_scene, alpha=0.5)
            scene_url = os.path.join(self.scene_root, self.timestamp, '{0}_scene.png'.format(node_id))
            plt.savefig(scene_url)
            plt.close()

            node_data = {
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
                        'scene': scene_url,
                        'child': {}
                        }

            # add the root node
            if self.data is None:
                self.data = copy.deepcopy(node_data)
                self.path = {}
                self.path[node_id] = []
            else:
                parent_path = self.path[parent_id]
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
        json_file = os.path.join(self.json_root, '{0}.json'.format(self.timestamp))
        with open(json_file, 'w') as f:
            f.write(json.dumps(self.data, ensure_ascii=False, indent=4, separators=(',', ':')))
        
        print('JSONDebugger: planning log saved ({0})'.format(json_file))
