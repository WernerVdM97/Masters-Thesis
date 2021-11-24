'''
Contruct a single tree with PyGasope modeling the leaf nodes' output

draws inspiration from https://github.com/ankonzoid/LearningX/tree/master/advanced_ML/model_tree
'''
# %%
from graphviz import Digraph
import numpy as np
import PyGasope
import math


class ModelTree:
    def __init__(self, tree_id, input_data, target_data, 
                 bagged_splits=True, max_depth=3, 
                 min_leaf_samples=20, max_poly_order = 3, 
                 ga_terms=10, ga_pop_size=30, ga_gens=100,
                 forest_size=100, lin_base=False,
                 called_from_ensemble=False, Progress=True
                 ):

        self.__lin_compare = lin_base    
        self.__called_from_tree=False
        self.__called_from_ensemble = called_from_ensemble
        if not called_from_ensemble:
            # code was originally written to accommodate data of the following shape (N x X)
            # this rectifies this:
            input_data = np.transpose(input_data)
            self.__called_from_ensemble = True
            self.__called_from_tree = True
            # default shape is now (X x N)

        if len(input_data.shape) == 1:
            input_data = np.array([input_data])

        self.__max_poly_order = max_poly_order
        self.__id = tree_id
        self.__grow_progress = 0
        self.__actual_depth = 0
        self.__no_nodes = 0
        self.__forest_size = forest_size
        self.__bagged_splits = bagged_splits

        max_nodes = 1
        for i in range(max_depth):
            max_nodes += 2**(i)

        self.__max_nodes = max_nodes - 1
        self.__max_depth = max_depth
        self.__min_leaf_samples = min_leaf_samples

        self.__ga_terms = ga_terms
        self.__ga_popsize = ga_pop_size
        self.__ga_gens = ga_gens

        # visual feedback
        self.__progress = Progress

        # pointer to root
        self.__root = self.__grow(input_data, target_data)

    # grow the tree i.e. induce nodes to describe dataset
    def __grow(self, input_data, target_data):
        global node_index

        def __create_node(input_data, target_data, container, parent=None, is_root_node=False):
            N = target_data.size
            if is_root_node:
                node = {'index': container['node_index'],  # global index of nodes, unique per
                        'depth': 1,  # layer in which node resides - root starts at 1
                        'parent_index': None,
                        'split_index': None,  # feature to split on
                        'split_value': None,  # value to split on
                        'SDR': None,
                        # branch nodes
                        'branch': {'false': None, 'true': None},
                        # True/False - indicates terminal node (both brances = None)
                        'leaf': False,
                        'model': None,
                        # subset of training/target-data on which node is proposed
                        'data_subset': (input_data, target_data),
                        'N': N
                        }
            else:
                node = {'index': container['node_index'],  # global index of nodes, unique per
                        # layer in which node resides - root starts at 1
                        'depth': parent['depth']+1,
                        'parent_index': parent['index'],
                        'split_index': None,  # feature to split on
                        'split_value': None,  # value to split on
                        'SDR': None,
                        # branch nodes depending on split condition
                        'branch': {'false': None, 'true': None},
                        # True/False - indicates terminal node (both brances = None)
                        'leaf': False,
                        'model': None,
                        # subset of training/target-data on which node is proposed
                        'data_subset': (input_data, target_data),
                        'N': N
                        }

            container['node_index'] += 1
            self.__no_nodes = container['node_index']
            # self.__grow_progress += 1
            return node

        def __split(node):
            split = {'halt_splitting': True,
                     'split_index': None,
                     'split_value': None,
                     'SDR': None,
                     'data_left': (None, None),  # split is not satisfied
                     'data_right': (None, None)  # split condition is satisfied
                     }

            (input_data, target_data) = node['data_subset']
            N = target_data.size  # number of samples
            F = input_data.shape[0]  # number of features
            best_split_feature = None
            best_split_value = None
            best_SDR = 0
            best_data_subset = [None, None]
            parent_SD = np.std(target_data)

            ############################feature split subset########################################
            if self.__bagged_splits:
                feature_subset_index = np.random.choice(
                    F, int(F/3+0.5), replace=False)
            else:
                feature_subset_index = np.arange(F)
            ########################################################################################

            # check first top criteria
            if node['depth'] < self.__max_depth:
                for j in range(feature_subset_index.size):  # loop through each feature
                    split_value_candidates = []
                    for i in range(N):  # loop through each data sample
                        split_value_candidates.append(
                            input_data[feature_subset_index[j], i])

                    for split_value in split_value_candidates:

                        # split data
                        left_idx = np.where(
                            input_data[feature_subset_index[j], :] <= split_value)[0]
                        right_idx = np.delete(np.arange(0, N), left_idx)

                        input_data_left = input_data[:, left_idx]
                        target_data_left = target_data[left_idx]
                        input_data_right = input_data[:, right_idx]
                        target_data_right = target_data[right_idx]

                        left_N, right_N = target_data_left.size, target_data_right.size

                        # check min samples stop criteria
                        if not (left_N >= self.__min_leaf_samples and right_N >= self.__min_leaf_samples):
                            continue

                        # compute SDR
                        split_SD = np.std(target_data_left)*left_N / \
                            N + np.std(target_data_right)*right_N/N
                        proposed_SDR = parent_SD - split_SD

                        '''
                        # SDR has to be atleast a 2 percent improvement to be evaluated
                        if (proposed_SDR/parent_SD) > 0.02:
                            continue
                        '''

                        # check if better than previous value, if so, update split
                        if best_SDR < proposed_SDR:
                            best_split_feature = feature_subset_index[j]
                            best_split_value = split_value
                            best_SDR = proposed_SDR
                            split['halt_splitting'] = False
                            split['SDR'] = proposed_SDR
                            best_data_subset = [(input_data_left, target_data_left),
                                                (input_data_right, target_data_right)]

            # update values
            split['split_index'] = best_split_feature
            split['split_value'] = best_split_value
            split['data_left'] = best_data_subset[0]
            split['data_right'] = best_data_subset[1]
            return split

        def __recursively_branch(node, container):
            # evaluate candidate splits
            split = __split(node)

            # Check for stop condition if so, build PyGasope model on data
            #####################################PyGasope################################################################
            if split['halt_splitting']:
                # update depth of tree (in the case that it stops splitting before max depth)
                if self.__actual_depth < node['depth']:
                    self.__actual_depth = node['depth']

                # for visual progress feedback, account for nodes that can no longer be grown in subset
                '''
                max_sub_nodes = 1
                if node['depth'] < self.__max_depth:
                    for i in range(self.__max_depth-node['depth']):
                        max_sub_nodes += 2**(i)
                    self.__grow_progress += max_sub_nodes
                '''
                tree_progress = self.__grow_progress/self.__max_nodes
                self.__grow_progress += 1
                forest_progress = self.__id / self.__forest_size

                # parameters - (original gmpcc used 10 terms, max-order=5)                   
                node['leaf'] = True
                max_terms = self.__ga_terms
                max_poly_order = self.__max_poly_order
                (input_data, target_data) = node['data_subset']

                # Build PyGasope model
                node['model'],_ = PyGasope.Evolve(max_terms,
                                                input_data, target_data,
                                                max_poly_order=max_poly_order, 
                                                generations=self.__ga_gens,
                                                population_size=self.__ga_popsize,
                                                debug=False,
                                                Progress=self.__progress,
                                                Progress_info=[tree_progress, forest_progress],
                                                LinearCompare=self.__lin_compare,
                                                called_from_ensemble = self.__called_from_ensemble)
                # node['model'].TreePrintOut
                return
            ############################################################################################################
            # else proceed with splitting
            else:
                # select split and update node information (split index, split value)
                node['split_index'] = split['split_index']
                node['split_value'] = split['split_value']
                node['SDR'] = split['SDR']

                # delete old partition to free up memory
                del node['data_subset']

                # partition data acording to new split
                (input_data_left, target_data_left) = split['data_left']
                (input_data_right, target_data_right) = split['data_right']

                # create new nodes of split
                node['branch']['false'] = __create_node(
                    input_data_left, target_data_left, container, parent=node)  # left
                node['branch']['true'] = __create_node(
                    input_data_right, target_data_right, container, parent=node)  # right

                # continue branching
                __recursively_branch(node['branch']['false'], container)
                __recursively_branch(node['branch']['true'], container)

        container = {'node_index': 0}
        root_node = __create_node(
            input_data, target_data, container, is_root_node=True)
        __recursively_branch(root_node, container)

        return root_node

    # predict a single sample at a time
    def predict_sample(self, input_sample):
        node = self.__root

        def __predict(node, input_sample):
            if node['leaf'] == True:
                if input_sample.size > 1:
                    input_sample = input_sample.reshape(-1, 1)
                return node['model'].out(input_sample)
            else:
                if input_sample[node['split_index']] <= node['split_value']:
                    return __predict(node['branch']['false'], input_sample)
                else:
                    return __predict(node['branch']['true'], input_sample)
            input_sample
        '''
        def __predictAdjacent(node, input_sample, split_index=0, split_value=0): # for smoothing purposes find neigbour leaf node
            if node['leaf'] == True:
                if input_sample.size > 1:
                    input_sample = input_sample.reshape(-1, 1)
                return node['model'].out(input_sample), split_index, split_value 
            else:
                if input_sample[node['split_index']] <= node['split_value']:
                    next_node = node['branch']['false']
                    if next_node['leaf'] == True:
                        return __predictAdjacent(node['branch']['true'], input_sample, 
                                                 split_index = node['split_index'], 
                                                 split_value = node['split_value'])
                    else:
                        return __predictAdjacent(next_node, input_sample)
                else:
                    next_node = node['branch']['true']
                    if next_node['leaf'] == True:
                        return __predictAdjacent(node['branch']['false'], input_sample, 
                                                 split_index = node['split_index'], 
                                                 split_value = node['split_value'])
                    else:
                        return __predictAdjacent(next_node, input_sample)
            input_sample
        '''

        y = __predict(node, input_sample)
        '''
        y_adj, split_index, split_value  = __predictAdjacent(node, input_sample)
        
        print(split_value)
        
        if (input_sample[split_index] - split_value)**2 < 0.01:
            y = y*0.5 + 0.5*y_adj
        '''
        return y

    # predict output of entire data set
    def predict(self, input_data):
        
        if not self.__called_from_ensemble:
            # code was originally written to accommodate data of the following shape (X x N)
            # this rectifies this:
            input_data = np.transpose(input_data)
            # default shape is now (N x X)

        if self.__called_from_tree:
            # code was originally written to accommodate data of the following shape (X x N)
            # this rectifies this:
            input_data = np.transpose(input_data)
            # default shape is now (N x X)

        if len(input_data.shape) == 1:
            input_data = np.array([input_data])
        predictions = np.zeros(input_data.shape[1])
        for i in range(input_data.shape[1]):
            predictions[i] = self.predict_sample(input_data[:, i])
        return predictions

    def __str__(self):
        return str(self.__root)

    def Visualise(self, file_name, file_path=""):
        graph = Digraph('graph', node_attr={'shape': 'record', 'height': '.1'})

        def __recursively_visualise(node, parent_index=0, parent_depth=0, edge_label=""):

            # Check for empty Node (i.e. parent is leaf)
            if node is None:
                return

            # Subtext of split and node
            node_index = node['index']
            if node['branch']['false'] is None and node['branch']['true'] is None:
                split = ""
            else:
                split = "X_{} <= {:.1f}\\nSDR = {:.6f}\\n ".format(
                    node['split_index'], node['split_value'], node['SDR'])

            label = "{} #observations = {}\n".format(split, node['N'])

            # if root node, append leaf model
            if node['leaf']:
                label += node['model'].TreePrintOut()

            # Visual Configuration of Node
            graph.attr('node', label=label, shape='rectangle')
            graph.node('node{}'.format(node_index),
                       color='black', style='filled',
                       fillcolor='white', fontcolor='black')

            # Visualise Split
            if parent_depth > 0:
                graph.edge('node{}'.format(parent_index),
                           'node{}'.format(node_index), label=edge_label)

            # Recursively visualise child or append leaf value
            __recursively_visualise(node['branch']['false'],
                                    parent_index=node_index,
                                    parent_depth=parent_depth + 1,
                                    edge_label="")
            __recursively_visualise(node['branch']['true'],
                                    parent_index=node_index,
                                    parent_depth=parent_depth + 1,
                                    edge_label="")

        # Start graph at root
        __recursively_visualise(self.__root,
                                parent_index=0,
                                parent_depth=0,
                                edge_label="")

        # Export png
        print("Saving model tree diagram to '{}.png'...".format(
            file_path + file_name))
        graph.format = "png"
        graph.render(filename=(file_path+file_name), view=False, cleanup=True)

    def details(self):
        details = "Tree Index: {}\t#Nodes: {}\tDepth: {}\t ".format(
            self.__id, self.__no_nodes, self.__actual_depth)

        return details
# %%
