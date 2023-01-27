'''
RG-RRT*

'''

import numpy as np
import random
from timeit import default_timer
from polytope_symbolic_system.common.utils import *


class Path:
    #TODO
    def __init__(self):
        pass

    def __repr__(self):
        raise ('NotImplementedError')

    def append(self, path):
        raise ('NotImplementedError')

class ReachableSet:
    '''
    Base class of ReachableSet
    '''
    def __init__(self,parent_state=None,path_class=None):#, state, planner, in_set, collision_test):
        '''

        :param state: The state this reachable set belongs to
        :param planner: Local planner for planning paths between self.state and any point in the reachable set.
        :param in_set: A fast checker for checking whether a state is in the set
        Planner takes in (state, goal) and returns (cost_to_go, path)
        '''
        self.path_class = path_class
        self.parent_state = parent_state
        pass
        # self.state = state
        # self.state_dim = state.shape[0]
        # self.planner = planner
        # self.in_set = in_set
        # self.collision_test = collision_test

    def contains(self, goal_state):
        '''
        Check if the state given is in this set
        :param state: query state
        :return: Boolean of whether the state is reachable
        '''
        raise('NotImplementedError')

    def contains_goal(self, goal_state):
        '''
        Containment check for goal state. Supports fuzzy checks.
        :param goal_state:
        :return: contains_goal, path_if_goal_is_contained
        '''
        # if goal is epsilon-away from reachable set, do fine goal check
        # enumerate control inputs to see if goal can be reached
        raise NotImplementedError

    def plan_collision_free_path_in_set(self, goal_state, return_deterministic_next_state=False):
        '''
        Plans a path between self.state and goal_state. Goals state must be in this reachable set
        :param state:
        :return: Tuple (cost_to_go, path). path is a Path class object
        '''
        # if not self.contains(goal_state):
        #     return (np.inf, None)
        # return self.planner(self.state, goal_state)
        raise('NotImplementedError')


    def find_closest_state(self, query_point, save_true_dynamics_path=False):
        '''
        Finds the closest point in the reachable set to the query point
        :param query_point:
        :return: Tuple (closest_point, closest_point_is_self.state)
        '''
        raise('NotImplementedError')

class Node:
    def __init__(self, state, reachable_set, true_dynamics_path, parent = None, path_from_parent = None,
                 children = None, cost_from_parent = np.inf):
        '''
        A node in the RRT tree
        :param state: the state associated with the node
        :param reachable_set: the reachable points from this node
        :param parent: the parent of the node
        :param path_from_parent: the path connecting the parent to state
        :param children: the children of the node
        :param cost_from_parent: cost to go from the parent to the node
        '''
        self.state = state
        self.reachable_set = reachable_set
        self.parent = parent  # Node object
        self.path_from_parent = path_from_parent
        self.cost_from_parent = cost_from_parent
        self.true_dynamics_path = true_dynamics_path
        if children is not None:
            self.children = children  # Node object
        else:
            self.children = set()
        if self.parent:
            self.cost_from_root = self.parent.cost_from_root + self.cost_from_parent
        else:
            self.cost_from_root = cost_from_parent

    def __repr__(self):
        if self.parent:
            return '\nRG-RRT* Node: '+'\n'+ \
                    '   state: ' + str(self.state) +'\n'+\
                    '   parent state: ' + str(self.parent.state) +'\n'+ \
                    '   path from parent: ' + self.path_from_parent.__repr__()+'\n'+ \
                    '   cost from parent: ' + str(self.cost_from_parent) + '\n' + \
                    '   cost from root: ' + str(self.cost_from_root) + '\n' #+ \
                    # '   children: ' + self.children.__repr__() +'\n'
        else:
            return '\nRG-RRT* Node: '+'\n'+ \
                    '   state: ' + str(self.state) +'\n'+\
                    '   parent state: ' + str(None) +'\n'+ \
                    '   cost from parent: ' + str(self.cost_from_parent) + '\n' + \
                    '   cost from root: ' + str(self.cost_from_root) + '\n' #+ \
                    # '   children: ' + self.children.__repr__() +'\n'

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.__hash__()==other.__hash__()

    def add_children(self, new_children_and_paths):
        #TODO: deprecated this function
        '''
        adds new children, represented as a set, to the node
        :param new_children:
        :return:
        '''
        self.children.update(new_children_and_paths)

    def update_parent(self, new_parent=None, cost_self_from_parent=None, path_self_from_parent=None):
        '''
        updates the parent of the node
        :param new_parent:
        :return:
        '''
        if self.parent is not None and new_parent is not None:  #assigned a new parent
            self.parent.children.remove(self)
            self.parent = new_parent
            self.parent.children.add(self)
        #calculate new cost from root #FIXME: redundancy
        assert(self.parent.reachable_set.contains(self.state))
        cost_self_from_parent, path_self_from_parent = self.parent.reachable_set.plan_collision_free_path_in_set(self.state)
        cost_root_to_parent = self.parent.cost_from_root
        self.cost_from_parent = cost_self_from_parent
        self.cost_from_root = cost_root_to_parent+self.cost_from_parent
        self.path_from_parent = path_self_from_parent
        #calculate new cost for children
        for child in self.children:
            child.update_parent()
        # print(self.parent.state, 'path', self.path_from_parent)
        # assert(np.all(self.parent.state==self.path_from_parent[0]))
        # assert(np.all(self.state==self.path_from_parent[1]))

class ReachableSetTree:
    '''
    Wrapper for a fast data structure that can help querying
    '''
    def __init__(self):
        pass

    def insert(self, id, reachable_set):
        raise('NotImplementedError')

    def nearest_k_neighbor_ids(self, query_state, k=1):
        raise('NotImplementedError')

    def d_neighbor_ids(self, query_state, d = np.inf):
        '''
        For rewiring only
        :param query_state:
        :param d:
        :return:
        '''
        raise('NotImplementedError')

class StateTree:
    '''
    Wrapper for a fast data structure that can help querying
    '''
    def __init__(self):
        pass

    def insert(self, id, state):
        raise('NotImplementedError')

    def state_ids_in_reachable_set(self, query_reachable_set):
        '''
        For rewiring only
        :param query_reachable_set:
        :return:
        '''
        raise('NotImplementedError')

class R3T:
    def __init__(self, root_state, compute_reachable_set, sampler, reachable_set_tree, state_tree, path_class, rewire_radius = None):
        '''
        Base RG-RRT*
        :param root_state: The root state
        :param compute_reachable_set: A function that, given a state, returns its reachable set
        :param sampler: A function that randomly samples the state space
        :param reachable_set_tree: A StateTree object for fast querying
        :param path_class: A class handel that is used to represent path
        '''
        self.root_node = Node(root_state, compute_reachable_set(root_state), np.asarray([root_state,root_state]),cost_from_parent=0)
        self.root_id = hash(str(root_state))
        self.state_dim = root_state[0]
        ## compute_reachable_set
        # given a state, return the PolytopeReachableSetTree object
        self.compute_reachable_set = compute_reachable_set
        self.sampler = sampler
        self.goal_state = None
        self.goal_node = None
        self.path_class = path_class
        ## state_tree
        # only store state of the Node
        self.state_tree = state_tree()
        self.state_tree.insert(self.root_id,self.root_node.state)
        ## reachable_set_tree
        # only store reachable set of the Node
        self.reachable_set_tree = reachable_set_tree() #tree for fast node querying
        self.reachable_set_tree.insert(self.root_id, self.root_node.reachable_set)
        ## state_to_node_map
        # given a state, return the Node object
        self.state_to_node_map = dict()
        self.state_to_node_map[self.root_id] = self.root_node
        ## node_tally
        # total number of nodes in the tree
        self.node_tally = 0
        self.rewire_radius=rewire_radius

    def create_child_node(self, parent_node, child_state, true_dynamics_path, cost_from_parent = None, path_from_parent = None):
        '''
        Given a child state reachable from a parent node, create a node with that child state
        :param parent_node: parent
        :param child_state: state inside parent node's reachable set
        :param cost_from_parent: FIXME: currently unused
        :param path_from_parent: FIXME: currently unused
        :return:
        '''
        # Update the nodes
        # compute the cost to go and path to reach from parent
        # if cost_from_parent is None or path_from_parent is None:
        # assert (parent_node.reachable_set.contains(child_state))
        cost_from_parent, path_from_parent = parent_node.reachable_set.plan_collision_free_path_in_set(child_state)
        # construct a new node
        new_node = Node(child_state, self.compute_reachable_set(child_state), true_dynamics_path,
                        parent=parent_node, path_from_parent=path_from_parent, cost_from_parent=cost_from_parent)
        parent_node.children.add(new_node)
        return new_node

    def extend(self, new_state, nearest_node, true_dynamics_path, explore_deterministic_next_state=True):
        """

        :param new_state:
        :param nearest_node:
        :param true_dynamics_path:
        :param explore_deterministic_next_state:
        :return: is_extended, new_node, deterministic_next_state
        """
        # check for obstacles
        if explore_deterministic_next_state:
            cost_to_go, path, deterministic_next_state = nearest_node.reachable_set.plan_collision_free_path_in_set(new_state,
                                                                                          return_deterministic_next_state=explore_deterministic_next_state)
        else:
            cost_to_go, path = nearest_node.reachable_set.plan_collision_free_path_in_set(new_state, return_deterministic_next_state=explore_deterministic_next_state)
        #FIXME: Support for partial extensions
        if path is None:
            if explore_deterministic_next_state:
                return False, None, None
            else:
                return False, None
        new_node = self.create_child_node(nearest_node, new_state, cost_from_parent=cost_to_go, path_from_parent=path, true_dynamics_path=true_dynamics_path)
        if explore_deterministic_next_state:
            return True, new_node, deterministic_next_state
        else:
            return True, new_node

    def build_tree_to_goal_state(self, goal_state, allocated_time = 20, stop_on_first_reach = False, rewire=False, explore_deterministic_next_state = True, max_nodes_to_add = int(1e3),\
                                 save_true_dynamics_path = False, Z_obs_list=None):
        '''
        Builds a RG-RRT* Tree to solve for the path to a goal.
        :param goal_state:  The goal for the planner.
        :param allocated_time: Time allowed (in seconds) for the planner to run. If time runs out before the planner finds a path, the code will be terminated.
        :param stop_on_first_reach: Whether the planner should continue improving on the solution if it finds a path to goal before time runs out.
        :param rewire: Whether to do RRT*
        :param explore_deterministic_next_state: perform depth-first exploration (no sampling, just build node tree) when the reachable set is a point
        :return: The goal node as a Node object. If no path is found, None is returned. self.goal_node is set to the return value after running.
        '''
        #TODO: Timeout and other termination functionalities
        start = default_timer()
        self.goal_state = goal_state
        # For cases where root node can lead directly to goal
        contains_goal, true_dynamics_path = self.root_node.reachable_set.contains_goal(self.goal_state)  # returns flag(T/F) and states list from root to goal
        if contains_goal:
            # check for obstacles
            # cost_to_go, path = new_node.reachable_set.plan_collision_free_path_in_set(goal_state)
            # allow for fuzzy goal check
            # if cost_to_go == np.inf:
            #     continue
            # if cost_to_go != np.inf:  #allow for fuzzy goal check
                # add the goal node to the tree
            goal_node = self.create_child_node(self.root_node, goal_state, true_dynamics_path)
            if rewire:
                self.rewire(goal_node)
            self.goal_node=goal_node

        while True:
            print("running search")
            print("current time: {}".format(default_timer()-start))
            if stop_on_first_reach:
                if self.goal_node is not None:
                    print('Found path to goal with cost %f in %f seconds after exploring %d nodes' % (self.goal_node.cost_from_root,
                    default_timer() - start, self.node_tally))
                    return self.goal_node
            if default_timer()-start>allocated_time:
                if self.goal_node is None:
                    print('Unable to find path within %f seconds!' % (default_timer() - start))
                    return None
                else:
                    print('Found path to goal with cost %f in %f seconds after exploring %d nodes' % (self.goal_node.cost_from_root,
                    default_timer() - start, self.node_tally))
                    return self.goal_node
            # print("running search 1")
            #sample the state space
            sample_is_valid = False
            sample_count = 0
            while not sample_is_valid:
                random_sample = self.sampler()
                sample_count+=1
                # map the states to nodes
                try:
                    # print("running sampler 1")
                    nearest_state_id_list = list(self.reachable_set_tree.nearest_k_neighbor_ids(random_sample, k=1))  # FIXME: necessary to cast to list?
                    discard = True
                    # print("running sampler 1_1")
                    nearest_node = self.state_to_node_map[nearest_state_id_list[0]]
                    # print("running sampler 1_2")
                    # find the closest state in the reachable set and use it to extend the tree
                    new_state, discard, true_dynamics_path = nearest_node.reachable_set.find_closest_state(random_sample, save_true_dynamics_path=save_true_dynamics_path, Z_obs_list=Z_obs_list)
                    # print("running sampler 1_3")
                    new_state_id = hash(str(new_state))
                    # print("running sampler 2")
                    # add the new node to the set tree if the new node is not already in the tree
                    # if new_state_id in self.state_to_node_map or discard:
                    if new_state_id in self.state_to_node_map or discard:
                        # FIXME: how to prevent repeated state exploration?
                        # print('Warning: state already explored')
                        # print("running sampler 3")
                        continue  # #sanity check to prevent numerical errors

                    if not explore_deterministic_next_state:
                        # print("running sampler 4")
                        is_extended, new_node = self.extend(new_state, nearest_node, true_dynamics_path, explore_deterministic_next_state=False)
                    else:
                        # print("running sampler 5")
                        is_extended, new_node, deterministic_next_state = self.extend(new_state, nearest_node, true_dynamics_path, explore_deterministic_next_state=True)
                except Exception as e:
                    print('Caught %s' % e)
                    # print("running sampler 6")
                    is_extended = False
                if not is_extended:
                    # print('Extension failed')
                    # print("running sampler 7")
                    continue
                else:
                    sample_is_valid = True
                #FIXME: potential infinite loop
            # print("running search 2")
            if sample_count>100:
                print('Warning: sample count %d' %sample_count)  # just warning that cannot get to a new sample even after so long
            if not explore_deterministic_next_state:
                # print("running search 3")
                self.reachable_set_tree.insert(new_state_id, new_node.reachable_set)
                self.state_tree.insert(new_state_id, new_node.state)
                try:
                    assert(new_state_id not in self.state_to_node_map)
                except:
                    print('State id hash collision!')
                    print('Original state is ', self.state_to_node_map[new_state_id].state)
                    print('Attempting to insert', new_node.state)
                    raise AssertionError
                self.state_to_node_map[new_state_id] = new_node
                self.node_tally = len(self.state_to_node_map)
                #rewire the tree
                if rewire:
                    self.rewire(new_node)
                #In "find path" mode, if the goal is in the reachable set, we are done
                contains_goal, true_dynamics_path = new_node.reachable_set.contains_goal(self.goal_state)
                if contains_goal:
                    # check for obstacles
                    # add the goal node to the tree
                    # cost_to_go, path = new_node.reachable_set.plan_collision_free_path_in_set(goal_state)
                    # allow for fuzzy goal check
                    # if cost_to_go == np.inf:
                    #     continue
                    goal_node = self.create_child_node(new_node, self.goal_state, true_dynamics_path)

                    #Hardik: commented two lines below
                    # diff = np.subtract(self.goal_state, true_dynamics_path)
                    # diff_norm = np.linalg.norm(diff, axis=1)
                    if rewire:
                        self.rewire(goal_node)
                    self.goal_node=goal_node
            else:
                # print("running search 4")
                nodes_to_add = [new_node]
                for iteration_count in range(int(max_nodes_to_add)):
                    # No longer deterministic
                    if new_node.reachable_set.deterministic_next_state is None:
                        break
                    # Already added
                    if hash(str(new_node.reachable_set.deterministic_next_state[-1])) in self.state_to_node_map:
                        break
                    try:
                        if save_true_dynamics_path:
                            is_extended, new_node,deterministic_next_state = self.extend(new_node.reachable_set.deterministic_next_state[-1], \
                                                                                         new_node, true_dynamics_path=new_node.reachable_set.deterministic_next_state,explore_deterministic_next_state=True)
                        else:
                            is_extended, new_node, deterministic_next_state = self.extend(
                                new_node.reachable_set.deterministic_next_state[-1],
                                new_node, true_dynamics_path=[new_node.state,new_node.reachable_set.deterministic_next_state[-1]],
                                explore_deterministic_next_state=True)
                    except Exception as e:
                        # print('Caught %s' %e)
                        is_extended=False
                    if not is_extended:  # extension failed
                        break
                    nodes_to_add.append(new_node)
                # print("running search 5")
                if iteration_count == max_nodes_to_add-1:
                    print('Warning: hit max_nodes_to_add')
                for new_node in nodes_to_add:
                    new_state_id = hash(str(new_node.state))
                    try:
                        assert(new_state_id not in self.state_to_node_map)
                    except:
                        print('State id hash collision!')
                        print('Original state is ', self.state_to_node_map[new_state_id].state)
                        print('Attempting to insert', new_node.state)
                        raise AssertionError
                    self.reachable_set_tree.insert(new_state_id, new_node.reachable_set)
                    self.state_tree.insert(new_state_id, new_node.state)
                    self.state_to_node_map[new_state_id] = new_node
                    self.node_tally = len(self.state_to_node_map)
                    #rewire the tree
                    if rewire:
                        self.rewire(new_node)
                    #In "find path" mode, if the goal is in the reachable set, we are done
                    contains_goal, true_dynamics_path = new_node.reachable_set.contains_goal(self.goal_state)
                    if contains_goal:
                    # check for obstacles
                        cost_to_go, path = new_node.reachable_set.plan_collision_free_path_in_set(goal_state)
                        #allow for fuzzy goal check
                        # if cost_to_go == np.inf:
                        #     continue
                        # add the goal node to the tree
                        goal_node = self.create_child_node(new_node, self.goal_state, true_dynamics_path)
                        if rewire:
                            self.rewire(goal_node)
                        self.goal_node=goal_node
        print("finished search")

    def rewire(self, new_node):
        # rewire_parent_candidate_states = list(self.reachable_set_tree.d_neighbor_ids(new_node.state))
        # #rewire the parent of the new node
        # best_new_parent = None
        # best_cost_to_new_node = new_node.cost_from_root
        # for cand_state in rewire_parent_candidate_states:
        #     parent_candidate_node = self.state_to_node_map[cand_state]
        #     if parent_candidate_node==new_node.parent or parent_candidate_node==new_node:
        #         continue
        #     #check if it's better to connect to new node through the candidate
        #     if not parent_candidate_node.reachable_set.contains(new_node.state):
        #         continue
        #     cost_to_go, path = parent_candidate_node.reachable_set.plan_collision_free_path_in_set(new_node.state)
        #     if parent_candidate_node.cost_from_root+cost_to_go < best_cost_to_new_node:
        #         best_new_parent = parent_candidate_node
        #         best_cost_to_new_node = parent_candidate_node.cost_from_root+cost_to_go
        #
        # #update if the best parent is changed
        # if best_new_parent is not None:
        #     new_node.update_parent(best_new_parent)

        cand_ids = self.state_tree.state_ids_in_reachable_set(new_node.reachable_set)
        #try to make the new node the candidate's parent
        for cand_id in cand_ids:
            candidate_node = self.state_to_node_map[cand_id]
            if candidate_node == new_node.parent or candidate_node == self.root_node:
                continue
            if not new_node.reachable_set.contains(candidate_node.state):
                continue
            cost_to_go, path = new_node.reachable_set.plan_collision_free_path_in_set(candidate_node.state)
            if candidate_node.cost_from_root > cost_to_go+new_node.cost_from_root:
                # print('rewired!')
                candidate_node.update_parent(new_node, cost_to_go, path)
        return True

    def get_root_to_node_path(self, node):
        states = []
        n = node
        while True:
            states.append(n.state)
            n = n.parent
            if n is None:
                break
        return states.reverse()

class Node_Hybrid:
    def __init__(self, state, reachable_set, parent = None, path_from_parent = None,
                 input_from_parent=None, mode_string_from_parent=None,
                 children = None, cost_from_parent = np.inf):
        '''
        A node in the RRT tree
        :param state: the state associated with the node
        :param reachable_set: the reachable points from this node
        :param parent: the parent of the node, also Node_Hybrid object
        :param path_from_parent: the path connecting the parent to state
        :param input_from_parent: the input from parent, as the time step is nearly zero, the input is viewed as constant
        :param mode_string_from_parent: the contact mode from parent
        :param children: the children of the node, a set
        :param cost_from_parent: cost to go from the parent to the node
        '''
        self.state = state
        self.reachable_set = reachable_set
        self.parent = parent  # Node_Hybrid object
        self.path_from_parent = path_from_parent
        self.cost_from_parent = cost_from_parent
        self.input_from_parent = input_from_parent
        self.mode_from_parent = mode_string_from_parent
        if children is not None:
            self.children = children  # Node_Hybrid object
        else:
            self.children = set()
        if self.parent:
            self.cost_from_root = self.parent.cost_from_root + self.cost_from_parent
        else:
            self.cost_from_root = cost_from_parent

    def __repr__(self):
        if self.parent:
            return '\nRG-RRT* Node: '+'\n'+ \
                    '   state: ' + str(self.state) +'\n'+\
                    '   parent state: ' + str(self.parent.state) +'\n'+ \
                    '   path from parent: ' + self.path_from_parent.__repr__()+'\n'+ \
                    '   cost from parent: ' + str(self.cost_from_parent) + '\n' + \
                    '   input from parent: ' + str(self.input_from_parent) + '\n' + \
                    '   mode from parent: ' + str(self.mode_from_parent) + '\n' + \
                    '   cost from root: ' + str(self.cost_from_root) + '\n' #+ \
                    # '   children: ' + self.children.__repr__() +'\n'
        else:
            return '\nRG-RRT* Node: '+'\n'+ \
                    '   state: ' + str(self.state) +'\n'+\
                    '   parent state: ' + str(None) +'\n'+ \
                    '   cost from parent: ' + str(self.cost_from_parent) + '\n' + \
                    '   input from parent: ' + str(self.input_from_parent) + '\n' + \
                    '   mode from parent: ' + str(self.mode_from_parent) + '\n' + \
                    '   cost from root: ' + str(self.cost_from_root) + '\n' #+ \
                    # '   children: ' + self.children.__repr__() +'\n'

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.__hash__()==other.__hash__()

    def add_children(self, new_children_and_paths):
        #TODO: deprecated this function
        '''
        adds new children, represented as a set, to the node
        :param new_children:
        :return:
        '''
        self.children.update(new_children_and_paths)

    def update_parent(self, new_parent=None, cost_self_from_parent=None, path_self_from_parent=None,
                            input_from_parent=None, mode_string_from_parent=None):
        '''
        Currently deprecated!
        updates the parent of the node
        :param new_parent:
        :return:
        '''
        if self.parent is not None and new_parent is not None:  #assigned a new parent
            self.parent.children.remove(self)
            self.parent = new_parent
            self.parent.children.add(self)
            self.input_from_parent = input_from_parent
            self.mode_from_parent = mode_string_from_parent
        #calculate new cost from root #FIXME: redundancy
        assert(self.parent.reachable_set.contains(self.state))
        # exact path planning #FIXME: cost should contain input quadratic cost
        cost_self_from_parent, path_self_from_parent = self.parent.reachable_set.plan_collision_free_path_in_set(self.state)
        cost_root_to_parent = self.parent.cost_from_root
        self.cost_from_parent = cost_self_from_parent
        self.cost_from_root = cost_root_to_parent+self.cost_from_parent
        self.path_from_parent = path_self_from_parent
        #calculate new cost for children
        for child in self.children:
            child.update_parent()
        # print(self.parent.state, 'path', self.path_from_parent)
        # assert(np.all(self.parent.state==self.path_from_parent[0]))
        # assert(np.all(self.state==self.path_from_parent[1]))

    def update_state_and_reachable_set(self, state, u, compute_reachable_set):
        """
        Update state and reachable set
        :param state: new state
        :param u: the input to reach current state
        :param compute_reachable_set: the function to compute reachable set
        """
        self.state = state
        self.reachable_set = compute_reachable_set(state, u)

class R3T_Hybrid:
    def __init__(self, root_state, compute_reachable_set, sampler, \
                 goal_sampling_bias, mode_consistent_sampling_bias, \
                 distance_scaling_array, \
                 reachable_set_tree_class, state_tree_class, path_class, dim_u, rewire_radius = None):
        '''
        Base RG-RRT*
        :param root_state: The root state, with psic
        :param compute_reachable_set: A function that, given a state, returns its reachable set
        :param sampler: A function that randomly samples the state space
        :param reachable_set_tree: A StateTree object for fast querying
        :param dim_u: the input dimension
        :param path_class: A class handel that is used to represent path
        '''
        self.dim_u = dim_u
        self.u_empty = np.zeros(dim_u)
        self.u_bar = np.zeros(dim_u)
        self.root_node = Node_Hybrid(state=root_state,
                                     reachable_set=compute_reachable_set(root_state, self.u_empty),
                                     input_from_parent=self.u_empty,
                                     cost_from_parent=0)
        self.root_id = hash(str(root_state[:-1]))  # exclude psic
        self.state_dim = root_state.shape[0]

        ## compute_reachable_set
        # (state with psic, input) --> the PolytopeReachableSetTree object
        self.compute_reachable_set = compute_reachable_set
        self.sampler = sampler
        self.goal_sampling_bias = goal_sampling_bias
        self.goal_state = None
        self.goal_node = None
        self.path_class = path_class
        
        ## state_tree
        # only store state of the Node
        self.state_tree = state_tree_class(distance_scaling_array=self.distance_scaling_array)
        self.state_tree.insert(self.root_id, self.root_node.state[:-1])  # exclude psic

        ## distance scaling array
        self.distance_scaling_array = distance_scaling_array
        
        ## reachable_set_tree
        # only store reachable set of the Node
        self.reachable_set_tree = \
            reachable_set_tree_class(distance_scaling_array=self.distance_scaling_array)
            # tree for fast node querying
        self.reachable_set_tree.insert(self.root_id, self.root_node.reachable_set)
        
        ## state_to_node_map
        # given a state id, return the Node object
        self.state_to_node_map = dict()
        self.state_to_node_map[self.root_id] = self.root_node

        ## node_tally
        # total number of nodes in the tree
        self.node_tally = 0
        self.rewire_radius=rewire_radius  # NOT USED

        ## time cost
        self.time_cost = {'nn_search': [],
                          'extend': [],
                          'state_tree_insert': [],
                          'set_tree_insert': [],
                          'store_and_rewire': [],
                          'rewire_parent': [],
                          'rewire_child': []}

        ## polytope data
        self.polytope_data = {'consistent': []}

    def create_child_node(self, parent_node, child_state, input_from_parent, 
                                mode_string_from_parent, cost_from_parent = None, path_from_parent = None):
        '''
        Given a child state reachable from a parent node, create a node with that child state
        :param parent_node: parent node, Node_Hybrid object
        :param child_state: state inside parent node's reachable set, with psic
        :param input_from_parent: input from parent
        :param mode_string_from_parent: (contact face, contact mode) from parent
        :param cost_from_parent: cost from parent
        :param path_from_parent: path from parent
        :return: new node, Node_Hybrid object
        '''
        # normalize input
        child_state[2] = restrict_angle_in_unit_circle(child_state[2])
        child_state[3] = restrict_angle_in_unit_circle(child_state[3])
        # construct a new node
        new_node = Node_Hybrid(state=child_state,
                               reachable_set=self.compute_reachable_set(child_state, input_from_parent),
                               parent=parent_node,
                               path_from_parent=path_from_parent,
                               input_from_parent=input_from_parent,
                               mode_string_from_parent=mode_string_from_parent,
                               cost_from_parent=cost_from_parent)
        parent_node.children.add(new_node)
        return new_node

    def extend(self, new_state, nearest_node, closest_polytope, Z_obs_list=None):
        """

        :param new_state: the new state, excluding psic
        :param nearest_node: the nearest node, including parent state
        :param closest_polytope: the closest polytope
        :param Z_obs_list: the MultiPolygon object for obstacles
        :return: is_extended, new_node
        """
        # check for obstacles
        plan_success_flag, cost_to_go, path, new_state_with_psic, applied_u = \
            nearest_node.reachable_set.plan_path_in_set_with_hybrid_dynamics(new_state, closest_polytope, Z_obs_list)
        # FIXME: Support for partial extensions
        
        # cannot connect
        if not plan_success_flag:
            return False, None

        # can connect
        new_node = self.create_child_node(parent_node=nearest_node,
                                          child_state=new_state_with_psic,
                                          input_from_parent=applied_u,
                                          mode_string_from_parent=closest_polytope.mode_string,
                                          cost_from_parent=cost_to_go,
                                          path_from_parent=path)
        return True, new_node

    def build_tree_to_goal_state(self, goal_state, allocated_time = 20, stop_on_first_reach = False, rewire=False, explore_deterministic_next_state = False, max_nodes_to_add = int(1e3),\
                                 Z_obs_list=None):
        '''
        Builds a RG-RRT* Tree to solve for the path to a goal.
        :param goal_state: The goal for the planner, without psic
        :param allocated_time: Time allowed (in seconds) for the planner to run. If time runs out before the planner finds a path, the code will be terminated.
        :param stop_on_first_reach: Whether the planner should continue improving on the solution if it finds a path to goal before time runs out.
        :param rewire: Whether to do RRT*
        :param explore_deterministic_next_state: perform depth-first exploration (no sampling, just build node tree) when the reachable set is a point
        :return: is goal reached
                 The goal node as a Node object. If no path is found, None is returned. self.goal_node is set to the return value after running.
        '''
        #TODO: Timeout and other termination functionalities
        start = default_timer()
        # state: (x, y, theta), without psic
        self.goal_state = goal_state
        # CHECK IF ROOT NODE COULD CONNECT TO GOAL FIXME: modify contains_goal
        contains_goal, closest_polytope, exact_path_info = self.root_node.reachable_set.contains_goal(self.goal_state)  # returns flag(T/F) and states list from root to goal
        if contains_goal:
            # check for obstacles
            # cost_to_go, path = new_node.reachable_set.plan_collision_free_path_in_set(goal_state)
            # allow for fuzzy goal check
            # if cost_to_go == np.inf:
            #     continue
            # if cost_to_go != np.inf:  #allow for fuzzy goal check
                # add the goal node to the tree

            # EXTEND TO GOAL
            cost_to_go, path, new_goal_with_psic, applied_u = exact_path_info
            goal_node = self.create_child_node(parent_node=self.root_node, 
                                               child_state=new_goal_with_psic,
                                               input_from_parent=applied_u,
                                               mode_string_from_parent=closest_polytope.mode_string,
                                               cost_from_parent=cost_to_go,
                                               path_from_parent=path)

            self.goal_node=goal_node
            # print("finished search")
            return True, self.goal_node

        while True:
            # print("R3T_Hybrid: running search")
            print("R3T_Hybrid: current time: {0}, total nodes: {1}".format(default_timer()-start, self.node_tally))
            if stop_on_first_reach:
                if self.goal_node is not None:
                    print('R3T_Hybrid: Found path to goal with cost %f in %f seconds after exploring %d nodes' % (self.goal_node.cost_from_root,
                    default_timer() - start, self.node_tally))
                    return True, self.goal_node
            if default_timer()-start>allocated_time:
                if self.goal_node is None:
                    print('R3T_Hybrid: Unable to find path within %f seconds!' % (default_timer() - start))
                    return False, None
                else:
                    print('R3T_Hybrid: Found path to goal with cost %f in %f seconds after exploring %d nodes' % (self.goal_node.cost_from_root,
                    default_timer() - start, self.node_tally))
                    return True, self.goal_node
            # print("running search 1")
            #sample the state space
            sample_is_valid = False
            sample_count = 0
            while not sample_is_valid:
                # state: (x, y, theta), without psic
                if np.random.rand() <= self.goal_sampling_bias:
                    random_sample = goal_state
                    print('R3T_Hybrid: sampling from goal!')
                else:
                    random_sample = self.sampler()
                sample_count+=1
                # map the states to nodes
                try:
                    T_NN_SEARCH_START = default_timer()
                    # FIND THE NEAREST REACHABLE SET
                    nearest_state_id_list = \
                        self.reachable_set_tree.nearest_k_neighbor_ids(random_sample, k=1, duplicate_search_azimuth=True)
                    discard = True
                    nearest_node = self.state_to_node_map[nearest_state_id_list[0]]

                    # FIND THE NEAREST NODE
                    new_state, discard, _, nearest_polytope = \
                        nearest_node.reachable_set.find_closest_state_with_hybrid_dynamics(random_sample,
                                                                                           Z_obs_list=Z_obs_list,
                                                                                           duplicate_search_azimuth=True)
                    if discard:
                        print('R3T_Hybrid: extension from {0} to {1} failed!'.format(nearest_node.parent.state[:-1], new_state))
                        continue
                    self.time_cost['nn_search'].append(default_timer()-T_NN_SEARCH_START)

                    self.polytope_data['consistent'].append(nearest_polytope.mode_consistent)

                    T_EXTEND_START = default_timer()
                    # EXTENSION
                    is_extended, new_node = self.extend(new_state, nearest_node, nearest_polytope, Z_obs_list)
                    if not is_extended:
                        print('R3T_Hybrid: extension from {0} to {1} failed!'.format(nearest_node.parent.state[:-1], new_state))
                        continue

                    # DISCARD THE STATE IF HASH VALUE COLLIDES, OR TOO CLOSE TO PARENT STATE, OR COLLISION IN CONNECTION
                    # FIXME: may two different states have the same hash value, and then discarded unexpectedly?
                    # FIXME: how to prevent repeated state exploration?
                    # FIXME: sanity check to prevent numerical errors
                    new_state_id = hash(str(new_node.state[:-1]))  # without psic
                    if new_state_id in self.state_to_node_map:
                        print('R3T_Hybrid: extension from {0} to {1} failed!'.format(nearest_node.parent.state[:-1], new_state))
                        continue
                    self.time_cost['extend'].append(default_timer()-T_EXTEND_START)

                except Exception as e:
                    print('R3T_Hybrid: caught exeption %s' % e)
                    is_extended = False

                if not is_extended:
                    print('R3T_Hybrid: Extension failed')
                    continue
                else:
                    sample_is_valid = True
                #FIXME: potential infinite loop

            # TWO MUCH FAILED SAMPLES WARNING
            # print('R3T_Hybrid: sample amount {0}!'.format(sample_count))
            if sample_count>100:
                print('R3T_Hybrid: Warning: sample count %d' % sample_count)  # just warning that cannot get to a new sample even after so long
            
            T_REWIRE_START = default_timer()
            # ADD NEW SAMPLE TO REACHABLE_SET_TREE AND STATE_TREE
            self.state_tree.insert(new_state_id, new_node.state[:-1])  # exclude psic
            self.time_cost['state_tree_insert'].append(default_timer()-T_REWIRE_START)
            T_REWIRE_START = default_timer()
            self.reachable_set_tree.insert(new_state_id, new_node.reachable_set)
            self.time_cost['set_tree_insert'].append(default_timer()-T_REWIRE_START)

            # HASH COLLISION CHECK
            try:
                assert(new_state_id not in self.state_to_node_map)
            except:
                print('R3T_Hybrid: State id hash collision!')
                print('R3T_Hybrid: Original state is ', self.state_to_node_map[new_state_id].state[:-1])  # exclude psic
                print('R3T_Hybrid: Attempting to insert', new_node.state[:-1])  # exclude psic
                raise AssertionError

            # ADD NEW SAMPLE TO STATE_TO_NODE_MAP
            self.state_to_node_map[new_state_id] = new_node
            self.node_tally = len(self.state_to_node_map)

            # REWIRE THE TREE
            if rewire:
                self.rewire(new_node)
            self.time_cost['store_and_rewire'].append(default_timer()-T_REWIRE_START)

            # CHECK THE CONNECTION TO GOAL
            contains_goal, closest_polytope, exact_path_info = new_node.reachable_set.contains_goal(self.goal_state)
            if contains_goal:
                # check for obstacles
                # add the goal node to the tree
                # cost_to_go, path = new_node.reachable_set.plan_collision_free_path_in_set(goal_state)
                # allow for fuzzy goal check
                # if cost_to_go == np.inf:
                #     continue

                # EXTEND TO GOAL NODE
                cost_to_go, path, new_goal_with_psic, applied_u = exact_path_info
                goal_node = self.create_child_node(parent_node=new_node, 
                                                   child_state=new_goal_with_psic,
                                                   input_from_parent=applied_u,
                                                   mode_string_from_parent=closest_polytope.mode_string,
                                                   cost_from_parent=cost_to_go,
                                                   path_from_parent=path)

                self.goal_node=goal_node
                print("finished search")
                return True, self.goal_node

    def rewire(self, new_node):
        """
        Rewire I: rewire the nodes that can reach new node (not implemented)
        Rewire II: rewire the nodes that can be reached from new node
        :param new_node: the new node
        :return: if rewire successful
        """
        ## ----------------------------------------------------
        # FIND NODES THAT CAN REACH NEW NODE
        # print('R3T_Hybrid: attempting...rewire nodes that can reach new node!')
        T_REWIRE_PARENT = default_timer()
        nearest_polytopes = self.reachable_set_tree.get_polytopes_contains_state(new_node.state[:-1])
        self.time_cost['rewire_parent'].append(default_timer()-T_REWIRE_PARENT)

        # print('R3T_Hybrid: {0} nearest polytopes when rewiring!'.format(len(nearest_polytopes)))
        best_new_parent = {'node':None,
                           'cost':np.inf,
                           'cost_to_go':None,
                           'new_state':None,
                           'mode_string':None,
                           'path':None,
                           'u':None}

        for candidate_polytope in nearest_polytopes:
            state_id = self.reachable_set_tree.polytope_to_id[candidate_polytope]
            candidate_node = self.state_to_node_map[state_id]

            # CHECK EDGE CYCLE (CANDIDATE NODE IS NOT NEW NODE'S PARENT)
            # if self.reachable_set_tree.polytope_id_to_polytope[hash(str(hash(str(new_node.parent.state[:-1])))+str(new_node.mode_from_parent))] not in nearest_polytopes:
            #     print('R3T_Hybrid: warning, parent state can not reach new_node!')
            if (candidate_node == new_node.parent) or (candidate_node == new_node):
                # print('R3T_Hybrid: refuse rewiring, detect edge cycle!')
                continue
            
            plan_success_flag, exact_path_info = \
                candidate_node.reachable_set.plan_exact_path_in_set_with_hybrid_dynamics(new_node.state[:-1], candidate_polytope, Z_obs_list=None)
            if not plan_success_flag:
                # print('R3T_Hybrid: refuse rewiring, collision free path not found!')
                continue

            cost_to_go, path, new_state_with_psic, applied_u = exact_path_info
            new_cost_from_root = cost_to_go + candidate_node.cost_from_root
            if new_node.cost_from_root > new_cost_from_root:
                if new_cost_from_root < best_new_parent['cost']:
                    best_new_parent['node'] = candidate_node
                    best_new_parent['cost'] = new_cost_from_root
                    best_new_parent['cost_to_go'] = cost_to_go
                    best_new_parent['new_state'] = new_state_with_psic
                    best_new_parent['mode_string'] = candidate_polytope.mode_string
                    best_new_parent['path'] = path
                    best_new_parent['u'] = applied_u

        if best_new_parent['node'] is not None:
            print('R3T_Hybrid: 1 parent rewired!')
            self.delete_node(new_node)
            new_node = self.create_child_node(parent_node=best_new_parent['node'], 
                                              child_state=best_new_parent['new_state'],
                                              input_from_parent=best_new_parent['u'],
                                              mode_string_from_parent=best_new_parent['mode_string'],
                                              cost_from_parent=best_new_parent['cost_to_go'],
                                              path_from_parent=best_new_parent['path'])
            # add new node
            new_node_id = hash(str(new_node.state[:-1]))
            self.state_tree.insert(new_node_id, new_node.state[:-1])
            self.reachable_set_tree.insert(new_node_id, new_node.reachable_set)
            self.state_to_node_map[new_node_id] = new_node
        else:
            # print('R3T_Hybrid: refuse rewiring, best parent not found!')
            pass

        ## ----------------------------------------------------
        # FIND NODES THAT CAN BE REACHED FROM NEW_NODE
        # print('R3T_Hybrid: attempting...rewire nodes that new node can reach!')
        T_REWIRE_CHILD = default_timer()
        candidate_state_ids = self.state_tree.state_ids_in_reachable_set(new_node.reachable_set, duplicate_search_azimuth=True)
        self.time_cost['rewire_child'] = default_timer()-T_REWIRE_CHILD

        # print('R3T_Hybrid: {0} candidate states when rewiring!'.format(len(candidate_state_ids)))
        for state_id in candidate_state_ids:
            try:
                candidate_node = self.state_to_node_map[state_id]
            except:
                import pdb; pdb.set_trace()
            
            # CHECK EDGE CYCLE (CANDIDATE NODE IS NOT NEW NODE'S PARENT)
            if (candidate_node == new_node.parent) or (candidate_node == new_node) or (candidate_node == self.root_node):
                # print('R3T_Hybrid: refuse rewiring, detect edge cycle!')
                continue

            # CHECK CHILD (CANNOT PLAN EXACT PATH)
            if len(candidate_node.children) > 0:
                # print('R3T_Hybrid: refuse rewiring, not leaf node!')
                continue

            # PRECISE CHECK (CANDIDATE IS ONLY GUARANTEED TO LIE IN REACHABLE SET'S AABBs)
            reachable_flag, closest_polytope = \
                new_node.reachable_set.contains(candidate_node.state[:-1],
                                                return_closest_state=False,
                                                return_closest_polytope=True,
                                                duplicate_search_azimuth=True)
            if not reachable_flag:
                # print('R3T_Hybrid: refuse rewiring, node unreachable!')
                continue

            # REWIRE AND UPDATE THE CANDIDATE
            plan_success_flag, exact_path_info = \
                new_node.reachable_set.plan_exact_path_in_set_with_hybrid_dynamics(candidate_node.state[:-1], closest_polytope, Z_obs_list=None)
            if not plan_success_flag:
                # print('R3T_Hybrid: refuse rewiring, collision free path not found!')
                continue

            cost_to_go, path, new_state_with_psic, applied_u = exact_path_info
            if candidate_node.cost_from_root > cost_to_go + new_node.cost_from_root:
                print('R3T_Hybrid: 1 child rewired!')
                # delete new state
                self.delete_node(candidate_node)
                candidate_node = self.create_child_node(parent_node=new_node, 
                                                        child_state=new_state_with_psic,
                                                        input_from_parent=applied_u,
                                                        mode_string_from_parent=closest_polytope.mode_string,
                                                        cost_from_parent=cost_to_go,
                                                        path_from_parent=path)
                # add new node
                candidate_node_id = hash(str(new_state_with_psic[:-1]))
                self.state_tree.insert(candidate_node_id, new_state_with_psic[:-1])
                self.reachable_set_tree.insert(candidate_node_id, candidate_node.reachable_set)
                self.state_to_node_map[candidate_node_id] = candidate_node
            else:
                # print('R3T_Hybrid: refuse rewiring, best child not found!')
                pass
                
        return True

    def delete_node(self, node):
        """
        Delete a node
        :param node: the node to be deleted
        """
        node_state = node.state[:-1]
        state_id = hash(str(node_state))
        self.state_tree.delete(state_id, node_state)
        self.reachable_set_tree.delete(state_id, node.reachable_set)
        self.state_to_node_map.pop(state_id)

    def get_r3t_structure(self):
        """
        Get R3T structure
        """
        tree = {'v': {}, 'e': []}
        for node_id, node in self.state_to_node_map.items():
            tree['v'][node_id] = tuple(node.state)
            if node.parent:
                tree['e'].append((hash(str(node.parent.state[:-1])), hash(str(node.state[:-1]))))
        return tree

    def get_root_to_node_path(self, node):
        states = []
        inputs = []
        modes = []
        n = node
        while True:
            states.append(n.state)
            inputs.append(np.mean(np.array(n.input_from_parent).reshape(-1, self.dim_u), axis=0))
            modes.append(n.mode_from_parent)
            n = n.parent
            if n is None:
                break
        states.reverse(), inputs.reverse(), modes.reverse()
        return states, inputs, modes
