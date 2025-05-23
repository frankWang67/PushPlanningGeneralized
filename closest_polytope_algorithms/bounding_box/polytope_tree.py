# -*- coding: utf-8 -*-
'''

@author: wualbert
'''
from closest_polytope_algorithms.bounding_box.box_tree import *
from closest_polytope_algorithms.bounding_box.box import *
from pypolycontain.lib.operations import distance_point_polytope, to_AH_polytope
from closest_polytope_algorithms.utils.utils import build_key_point_kd_tree, update_key_point_kd_tree
from rtree import index

class PolytopeTree:
    def __init__(self, polytopes, key_vertex_count=0, distance_scaling_array=None):
        """
        The PolytopeTree, base class, to do insert, find_closest_polytopes
        :param polytopes: ndarray or list of polytopes
        :param key_vertex_count: related, but not equal to key points (1+2^key_vertex_count) sampled on a polytope
                                 for AH_polytope, key_vertex_count=0 (only 1 key point)
        :param distance_scaling_array: the distance metric
        """
        '''
        Updated implementation using rtree
        :param polytopes:
        '''
        self.polytopes = polytopes
        self.key_vertex_count = key_vertex_count
        # Create box data structure from zonotopes
        # self.type = self.polytopes[0].type
        # Initialize rtree structure (for AABB and nearest point query)
        # indexed by hash(polytope object)
        self.rtree_p = index.Property()
        self.rtree_p.dimension = to_AH_polytope(self.polytopes[0]).t.shape[0]
        print(('PolytopeTree dimension is %d-D' % self.rtree_p.dimension))
        self.idx = index.Index(properties=self.rtree_p)
        self.index_to_polytope_map = {}
        if distance_scaling_array is not None:
            assert(self.rtree_p.dimension==distance_scaling_array.shape[0])
        else:
            distance_scaling_array = np.ones(self.rtree_p.dimension)
        self.distance_scaling_array = distance_scaling_array
        self.repeated_scaling_matrix = np.tile(self.distance_scaling_array, 2)
        for i, z in enumerate(self.polytopes):
            lu = np.multiply(self.repeated_scaling_matrix, AH_polytope_to_box(to_AH_polytope(z)))
            # assert(hash(z) not in self.index_to_polytope_map)
            #FIXME
            if hash(z) not in self.index_to_polytope_map:
                self.idx.insert(hash(z), lu)
                self.index_to_polytope_map[hash(z)] = z

        # build key point tree for query box size guess
        self.scaled_key_point_tree, self.key_point_to_zonotope_map = build_key_point_kd_tree(self.polytopes, self.key_vertex_count, self.distance_scaling_array)

    def insert(self, new_polytopes):
        """
        Inserts a new polytope to the tree structure
        :param new_polytope:
        :return:
        """
        # insert into rtree
        for new_polytope in new_polytopes:
            if new_polytope.type == 'zonotope':
                lu = zonotope_to_box(new_polytope)
            elif new_polytope.type == 'AH_polytope' or 'H-polytope':
                lu = AH_polytope_to_box(new_polytope)
            else:
                raise NotImplementedError
            self.idx.insert(hash(new_polytope), np.multiply(self.repeated_scaling_matrix, lu))
            # assert (hash(new_polytope) not in self.index_to_polytope_map)
            self.index_to_polytope_map[hash(new_polytope)] = new_polytope

        # add to polytopes
        if isinstance(self.polytopes, np.ndarray):
            self.polytopes = np.concatenate((self.polytopes,np.array(new_polytopes)))
        else:
            self.polytopes.append(new_polytopes)

        # insert into kdtree
        # FIXME: Rebuilding a kDtree should not be necessary
        # self.scaled_key_point_tree, self.key_point_to_zonotope_map = build_key_point_kd_tree(self.polytopes, self.key_vertex_count, self.distance_scaling_array)
        self.scaled_key_point_tree, self.key_point_to_zonotope_map = update_key_point_kd_tree(new_polytopes, \
                                                                                              self.scaled_key_point_tree, \
                                                                                              self.key_point_to_zonotope_map, \
                                                                                              self.key_vertex_count, \
                                                                                              self.distance_scaling_array)

    def delete(self, new_polytopes):
        """
        Delete a new polytope to the tree structure
        :param new_polytope:
        :return:
        """
        # insert into rtree
        for new_polytope in new_polytopes:
            if new_polytope.type == 'zonotope':
                lu = zonotope_to_box(new_polytope)
            elif new_polytope.type == 'AH_polytope' or 'H-polytope':
                lu = AH_polytope_to_box(new_polytope)
            else:
                raise NotImplementedError
            num_idx = self.idx.get_size()
            self.idx.delete(hash(new_polytope), np.multiply(self.repeated_scaling_matrix, lu))
            if self.idx.get_size() > num_idx-1:
                print('PolytopeTree: warning, state {0} was not deleted successfully'.format(lu))
            self.index_to_polytope_map.pop(hash(new_polytope))

            # move from polytopes
            if isinstance(self.polytopes, np.ndarray):
                delete_index = np.where(self.polytopes == np.array(new_polytope))[0]
                self.polytopes = np.delete(self.polytopes, delete_index)
            else:
                self.polytopes.remove(new_polytope)

        # insert into kdtree
        # FIXME: Rebuilding a kDtree should not be necessary
        self.scaled_key_point_tree, self.key_point_to_zonotope_map = build_key_point_kd_tree(self.polytopes, self.key_vertex_count, self.distance_scaling_array)

    def find_closest_polytopes(self, original_query_point, return_candiate_polytopes_only=False, return_intermediate_info=False, return_state_projection=False, may_return_multiple=False):
        """
        Return the closest polytope(s) of the query point
        :param original_query_point: the query point, without psic
        :param return_candiate_polytopes_only: only return the candidate states, no further operations
        :param return_intermediate_info: if true, return evaluated_zonotopes, the evaluated (compute nearest distance) polytopes in order
        :param return_state_projection: if true, return the nearest (projected state) on the nearest polytope(s)
        :param may_return_multiple: if true, return as many as nearest polytopes (unsorted, with less than 1e-5 relative difference in nearest distance)
        :return: Tuple (ndarray of nearest polytopes, nearest distance, nearest projected states) or
                 Tuple intermediate info (ndarray of nearest polytopes, nearest distance, evaluated zonotopes, heuristic box lu)
                 evaluated zonotopes | record polytopes that distance_point_polytope is called, in order
                 heuristic box lu    | record polytopes that are better than the current best
        """
        #find closest centroid
        # try:
        #     query_point.shape[1]
        #     #FIXME: Choose between d*1 (2D) or d (1D) representation of query_point
        # except:
        #     # raise ValueError('Query point should be d*1 numpy array')
        #     query_point=query_point.reshape((-1,1))
        # Construct centroid box
        # EVALUATE THE POLYTOPE OF THE NEAREST KEY POINT
        scaled_query_point = np.multiply(self.distance_scaling_array, original_query_point.flatten())
        _x, ind = self.scaled_key_point_tree.query(np.ndarray.flatten(scaled_query_point))
        scaled_closest_centroid = self.scaled_key_point_tree.data[ind]
        #Use dist(polytope, query) as upper bound
        evaluated_zonotopes = []
        centroid_zonotopes = self.key_point_to_zonotope_map[str(np.divide(scaled_closest_centroid, self.distance_scaling_array))]
        for zonotope in centroid_zonotopes:
            if zonotope.mode_consistent:
                centroid_zonotopes = [zonotope]
                break

        polytope_state_projection = {}
        dist_to_query = {}
        # inf_dist_to_query = {}

        assert(len(centroid_zonotopes)==1)
        evaluated_zonotopes.extend(centroid_zonotopes)
        zd, state = distance_point_polytope(centroid_zonotopes[0], original_query_point, ball='l2', distance_scaling_array=self.distance_scaling_array)
        # zd = distance_point_polytope(cz, query_point, ball='l2')[0]
        best_scaled_distance=zd
        # best_inf_distance=zd
        best_polytope={centroid_zonotopes[0]}
        dist_to_query[centroid_zonotopes[0]] = best_scaled_distance
        polytope_state_projection[centroid_zonotopes[0]] = state

        # inf_dist_to_query[cz] = best_inf_distance
        # scale for numerical reasons
        scaled_aabb_offsets = np.abs(state.flatten()-original_query_point.flatten())*1.001
        u = original_query_point.flatten() - scaled_aabb_offsets
        v = original_query_point.flatten() + scaled_aabb_offsets
        heuristic_box_lu = np.concatenate([u, v])
        # scale the query box
        scaled_heuristic_box_lu = np.multiply(self.repeated_scaling_matrix, heuristic_box_lu)
        #create query box
        #find candidate box nodes

        candidate_ids = list(self.idx.intersection(scaled_heuristic_box_lu))
        # print('PolytopeTree: find {0} candidate polytopes in nearest neighbors search!'.format(len(candidate_ids)))
        if return_candiate_polytopes_only:
            candidate_ids.remove(hash(centroid_zonotopes[0]))
            candidate_states = [self.index_to_polytope_map[polytope_id] for polytope_id in candidate_ids]
            return candidate_states, centroid_zonotopes[0]

        # print('Evaluating %d zonotopes') %len(candidate_boxes)
        #map back to zonotopes
        if len(candidate_ids)==0:
            # This should never happen
            raise ValueError('No closest zonotope found!')
            # When a heuristic less than centroid distance is used,
            # a candidate box does not necessarily exist. In this case,
            # use the zonotope from which the heuristic is generated.
            # '''
            #
            # evaluated_zonotopes = pivot_polytope
            # closest_distance = pivot_distance
            # return evaluated_zonotopes, candidate_boxes, query_box
        else:
            # for cb in candidate_boxes:
            #     print(cb)
            #     evaluated_zonotopes.append(cb.polytope)
            #find the closest zonotope with randomized approach]
            while(len(candidate_ids)>=1):
                if best_scaled_distance < 1e-9:
                    # IF THE QUERY POINT IS EXACTLY CONTAINED IN ONE POLYTOPE
                    # point is contained by polytope, break
                    break
                # RANDOM SAMPLE A CANDIDATE POLYTOPE TO EVALUATE
                sample = np.random.randint(len(candidate_ids))
                #solve linear program for the sampled polytope
                pivot_polytope = self.index_to_polytope_map[candidate_ids[sample]]
                if pivot_polytope in best_polytope:
                    # IF THE CURRENT POLYTOPE HAS BEEN EVALUATED
                    #get rid of this polytope
                    candidate_ids[sample], candidate_ids[-1] = candidate_ids[-1], candidate_ids[sample]
                    candidate_ids = candidate_ids[0:-1]
                    continue
                # EVALUATE THE CANDIDATE POLYTOPE: CALCULATE NEAREST DISTANCE
                if pivot_polytope not in dist_to_query:
                    pivot_distance, state = distance_point_polytope(pivot_polytope, original_query_point, ball="l2",
                                                                    distance_scaling_array=self.distance_scaling_array)
                    # inf_pivot_distance = distance_point_polytope(pivot_polytope, query_point)[0]
                    dist_to_query[pivot_polytope] = pivot_distance
                    polytope_state_projection[pivot_polytope] = state
                    # inf_dist_to_query[pivot_polytope] = inf_dist_to_query
                    if return_intermediate_info:
                        evaluated_zonotopes.append(pivot_polytope)
                else:
                    pivot_distance = dist_to_query[pivot_polytope]
                    # inf_pivot_distance = inf_dist_to_query[pivot_polytope]
                # IF THE POLYTOPE IS WORSE THAN CURRENT BEST
                if pivot_distance>best_scaled_distance:#fixme: >= or >?
                    #get rid of this polytope
                    candidate_ids[sample], candidate_ids[-1] = candidate_ids[-1], candidate_ids[sample]
                    candidate_ids = candidate_ids[0:-1]
                # IF THE POLYTOPE IS COMPARABLE TO CURRENT BEST
                elif np.allclose(pivot_distance, best_scaled_distance):
                    best_polytope.add(pivot_polytope)
                    #get rid of this polytope
                    candidate_ids[sample], candidate_ids[-1] = candidate_ids[-1], candidate_ids[sample]
                    candidate_ids = candidate_ids[0:-1]
                # IF THE POLYTOPE IS BETTER THAN CURRENT BEST
                else:
                    #reconstruct AABB
                    # create query box
                    # scale for numerical reasons
                    scaled_aabb_offsets = np.abs(state.flatten()-original_query_point.flatten())*1.001
                    u = original_query_point.flatten() - scaled_aabb_offsets
                    v = original_query_point.flatten() + scaled_aabb_offsets
                    heuristic_box_lu = np.concatenate([u, v])
                    # scale the query box
                    scaled_heuristic_box_lu = np.multiply(self.repeated_scaling_matrix, heuristic_box_lu)
                    # find new candidates
                    candidate_ids = list(self.idx.intersection(scaled_heuristic_box_lu))
                    best_scaled_distance = pivot_distance
                    # best_inf_distance = inf_pivot_distance
                    best_polytope = {pivot_polytope}
            if return_intermediate_info:
                return np.atleast_1d(list(best_polytope)[0]), best_scaled_distance, evaluated_zonotopes, heuristic_box_lu
            if return_state_projection:
                if not may_return_multiple:
                    return np.atleast_1d(list(best_polytope)[0]), best_scaled_distance, polytope_state_projection[list(best_polytope)[0]]
                else:
                    return np.asarray(list(best_polytope)), best_scaled_distance, np.asarray([polytope_state_projection[bp].flatten() for bp in best_polytope])
            else:
                if not may_return_multiple:
                    return np.atleast_1d(list(best_polytope)[0])
                else:
                    return np.atleast_1d(list(best_polytope))
#
# class PolytopeTree_Old:
#     '''
#     Deprecated implementation. RTree is now used for the underlying data structure
#     '''
#     def __init__(self, polytopes):
#         self.polytopes = polytopes
#         # Create box data structure from zonotopes
#         self.type = self.polytopes[0].type
#         self.box_nodes = []
#         for z in self.polytopes:
#             if self.type == 'zonotope':
#                 box = zonotope_to_box(z)
#             elif self.type == 'AH_polytope' or 'H-polytope':
#                 box = AH_polytope_to_box(z, return_AABB=True)
#             else:
#                 raise NotImplementedError
#             bn = BoxNode(box)
#             self.box_nodes.append(bn)
#         assert (len(self.box_nodes) == len(self.polytopes))
#         self.key_point_tree, self.key_point_to_zonotope_map = build_key_point_kd_tree(self.polytopes)
#         self.root = binary_split(self.box_nodes)
#         # Create kd-tree data structure from zonotopes
#
#     def find_closest_zonotopes(self,query_point, return_intermediate_info=False):
#         #find closest centroid
#         try:
#             query_point.shape[1]
#             #FIXME: Choose between d*1 (2D) or d (1D) represntation of query_point
#         except:
#             # raise ValueError('Query point should be d*1 numpy array')
#             query_point=query_point.reshape((-1,1))
#         _x, ind = self.key_point_tree.query(np.ndarray.flatten(query_point))
#         closest_centroid = self.key_point_tree.data[ind]
#         # print('Closest centroid', closest_centroid)
#
#         #Use dist(centroid, query) as upper bound
#         vector_diff = np.subtract(np.ndarray.flatten(closest_centroid),\
#                                   np.ndarray.flatten(query_point))
#         # pivot_distance = 2*np.linalg.norm(vector_diff)
#
#         #Use dist(polytope, query) as upper bound
#         evaluated_zonotopes = []
#         centroid_zonotopes = self.key_point_to_zonotope_map[closest_centroid.tostring()]
#         best_distance = np.inf
#         best_polytope = None
#         for cz in centroid_zonotopes:
#             evaluated_zonotopes.append(cz)
#             zd = zonotope_distance_point(cz,query_point)[0]
#             if best_distance > zd:
#                 best_distance=zd
#                 best_polytope=cz
#
#         #create query box
#         query_box = AABB_centroid_edge(query_point,2*best_distance)
#         #find candidate box nodes
#         candidate_boxes = []
#         self.root.evaluate_node(query_box,candidate_boxes)
#         # print('Evaluating %d zonotopes') %len(candidate_boxes)
#         #map back to zonotopes
#         closest_distance = np.inf
#         if candidate_boxes is None:
#             # This should never happen
#             raise ValueError('No closest zonotope found!')
#             # When a heuristic less than centroid distance is used,
#             # a candidate box does not necessarily exist. In this case,
#             # use the zonotope from which the heuristic is generated.
#             # '''
#             #
#             # evaluated_zonotopes = pivot_polytope
#             # closest_distance = pivot_distance
#             # return evaluated_zonotopes, candidate_boxes, query_box
#         else:
#             # for cb in candidate_boxes:
#             #     print(cb)
#             #     evaluated_zonotopes.append(cb.polytope)
#             #find the closest zonotope with randomized approach
#             while(len(candidate_boxes)>1):
#                 sample = np.random.randint(len(candidate_boxes))
#                 #solve linear program for the sampled polytope
#                 pivot_polytope = candidate_boxes[sample].polytope
#                 if return_intermediate_info:
#                     evaluated_zonotopes.append(pivot_polytope)
#                 pivot_distance = distance_point_polytope(pivot_polytope, query_point, ball="l2")[0]
#                 if pivot_distance>=best_distance:#fixme: >= or >?
#                     #get rid of this polytope
#                     candidate_boxes[sample], candidate_boxes[-1] = candidate_boxes[-1], candidate_boxes[sample]
#                     candidate_boxes = candidate_boxes[0:-1]
#                 else:
#                     #reconstruct AABB
#                     # create query box
#                     query_box = AABB_centroid_edge(query_point, 2 * pivot_distance)
#                     # find candidate box nodes
#                     candidate_boxes = []
#                     self.root.evaluate_node(query_box, candidate_boxes)
#                     best_distance = pivot_distance
#                     best_polytope = pivot_polytope
#             if return_intermediate_info:
#                 return np.atleast_1d(best_polytope), best_distance, evaluated_zonotopes, query_box
#             return np.atleast_1d(best_polytope)

