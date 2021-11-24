'''
Class to construct ensembles of PyTree.
'''

from PyTree import ModelTree
import numpy as np
import sys
import math
from sklearn.metrics import mean_squared_error


class MTForest:
    def __init__(self, input_data, target_data, score_in = None, score_target=None,
                 forest_size=10, max_poly_order = 3, tree_depth = 3,
                 feature_bagging=False, lin_base=False,
                 ga_pop_size=30, ga_gens=100, ga_terms=10,
                 min_leaf_samples = 20, Progress=True
                 ):

        # code was originally written to accommodate data of the following shape (N x X)
        # this rectifies this:
        input_data = np.transpose(input_data)
        # default shape is now (X x N)
        
        if len(input_data.shape) == 1:
            input_data = np.array([input_data])

        self.__Progress = Progress
        self.__lin_base = lin_base
        self.__ga_pop_size = ga_pop_size
        self.__ga_gens = ga_gens
        self.__ga_terms = ga_terms
        self.__min_leaf_samples = min_leaf_samples

        self.score = np.zeros((2,forest_size))

        self.__tree_depth = tree_depth
        self.__max_poly_order = max_poly_order
        self.__forest_size = forest_size
        self.__forest = []
        self.__grow_forest(input_data, target_data, feature_bagging, score_in, score_target)

    def __grow_forest(self, input_data, target_data, feature_bagging, score_in, score_target):

        def _grow_tree(index, input_data, target_data, feature_bagging):

            # assumes data is a numpy array of the shape (F x N) for input and (1 x N) for output

            # split data
            F = input_data.shape[0]  # Number of feature
            N = target_data.size  # Number of sample observations

            if not feature_bagging:
                # select all features (bagged_splits=True)
                feature_index = np.arange(F)
            else:
                # Feature Aggregation/Bagging (usually of size sqrt(F), bagged_splits=False)
                feature_index = np.random.choice(
                    F, int(F/3+0.5), replace=False)

            # select From N, N number of samples, with replacement, amounting to 66% unique samples (Random Forest inspired)
            data_subset_index = np.random.choice(N, N)

            # Separate OOB from Original
            unique_entries = np.unique(data_subset_index)
            OoB_index = np.delete(np.arange(0, N), unique_entries)
            OoB_input_data = input_data[:, OoB_index]
            OoB_input_data = OoB_input_data[feature_index, :]
            OoB_target_data = target_data[OoB_index]

            # Bootstrap subset
            input_data = input_data[:, data_subset_index]
            input_data = input_data[feature_index, :] # in the case of feature bagging
            target_data = target_data[data_subset_index]

            Tree = {'index': index,
                    'feature_index': feature_index,  # in case feature bagging is used
                    'data_subset_index': data_subset_index,
                    'OOB': -1,
                    'model': ModelTree(index, input_data, target_data, 
                                       bagged_splits=(not feature_bagging),
                                       # np.random.choice(np.arange(2,5),2)[0],  # allow variety in depths
                                       max_depth=self.__tree_depth,
                                       min_leaf_samples= self.__min_leaf_samples, 
                                       max_poly_order = self.__max_poly_order,
                                       forest_size=self.__forest_size,
                                       ga_pop_size=self.__ga_pop_size, 
                                       ga_gens=self.__ga_gens, 
                                       ga_terms=self.__ga_terms,
                                       lin_base=self.__lin_base,
                                       called_from_ensemble=True,
                                       Progress=self.__Progress)
                    }

            # Compute OOB
            Tree['OOB'] = mean_squared_error(
                Tree['model'].predict(OoB_input_data), OoB_target_data)

            return Tree

        # induce trees
        for i in range(self.__forest_size):
            self.__forest.append(_grow_tree(
                i, input_data, target_data, feature_bagging))
            
            if score_in != None and score_target != None:
                self.score[1,i] = mean_squared_error(self.predict(score_in),score_target)
                self.score[0,i] = mean_squared_error(self.predict(input_data,CalledFromSelf=True),target_data)

        # for visualisation purposes
        sys.stdout.write('\n')

    def predict(self, input_data, CalledFromSelf = False):
        
        if not CalledFromSelf:
            # code was originally written to accommodate data of the following shape (N x X)
            # this rectifies this:
            input_data = np.transpose(input_data)
            # default shape is now (X x N)

        y = 0
        if len(input_data.shape) == 1:
            input_data = np.array([input_data])

        for i in range(len(self.__forest)):
            x = input_data[self.__forest[i]['feature_index'], :]
            y += self.__forest[i]['model'].predict(x)

        y = y/len(self.__forest)         # average out of trees

        return y

    # DEPRECATED
    def __predict_given_forest(self, forest, input_data, CalledFromSelf = False):
        
        if not CalledFromSelf:
            # code was originally written to accommodate data of the following shape (N x X)
            # this rectifies this:
            input_data = np.transpose(input_data)
            # default shape is now (X x N)

        y = 0
        if len(input_data.shape) == 1:
            input_data = np.array([input_data])

        for i in range(len(forest)):
            x = input_data[forest[i]['feature_index'], :]
            y += forest[i]['model'].predict(x)

        y = y/len(forest)         # average out of trees

        return y

    # print each tree as text line
    def Forest_details(self):
        for i in range(len(self.__forest)):
            print(self.__forest[i]['model'].details(),
                  "Data Subset:", self.__forest[i]['feature_index'],
                  "\tOoB MSE: {:.4f}".format(self.__forest[i]['OOB']))

    # output each tree structure to PNG
    def Print_Forest(self):
        for i in range(len(self.__forest)):
            self.__forest[i]['model'].Visualise(
                "Tree_ID_{}".format(i), file_path="PyGasope/Forest_Png/")

    # Three separate pruning methods below
    def ForestPruning(self, input_data, target_data, printout=True):
        # code was originally written to accommodate data of the following shape (X x N)
        # this rectifies this:
        input_data = np.transpose(input_data)
        # default shape is now (N x X)

        if self.__forest_size > 1:
            if printout:
                print("Pruning Forest ...")

            # elimenate trees that have poor OOB
            '''
            best_OOB = 99999
            for i in range(self.__forest_size):
                # find best OOB as reference for pruning
                if self.__forest[i]['OOB'] < best_OOB:
                    best_OOB = self.__forest[i]['OOB']
                            
            print("Bad Trees:")
            for i in range(self.__forest_size,0,-1):
                # prune trees that have bad OOB
                if self.__forest[i-1]['OOB'] > best_OOB*2:
                    tree = self.__forest.pop(i-1)
                    print(tree['model'].details(), "Data Subset:", tree['feature_index'],
                    "\tOoB MSE: {:.4f}".format(tree['OOB']))
            print("That is all, thank you\n")
            '''
            ##################################################################################################

            # eleminate trees one by one that contribute to loss 
            # FORMULA: if is loss increases with more than (1/forest_size) %. Delete tree
            original_mse = mean_squared_error(
                self.predict(input_data, CalledFromSelf=True), target_data)
            pop_indexes = []

            for i in range(self.__forest_size):
                # pop tree
                tree = self.__forest.pop(i)

                # calc loss
                candidate_mse = mean_squared_error(
                    self.predict(input_data, CalledFromSelf=True), target_data)

                # add index to be removed if loss is less
                # see formula comment above
                if candidate_mse * (1 + 1/self.__forest_size) < original_mse:
                    pop_indexes.append(i)

                # re-insert ree
                self.__forest.insert(i, tree)

            if (len(pop_indexes)-1) >= self.__forest_size:
                print("we got a problem :(")
            else:
                if printout:
                    print("Bad Trees:")
                    # remove bad trees
                    for i in range(len(pop_indexes), 0, -1):
                        tree = self.__forest.pop(pop_indexes[i-1])
                        print(tree['model'].details(), "Data Subset:", tree['feature_index'],
                            "\tOoB MSE: {:.4f}".format(tree['OOB']))
                    print("That is all, thank you\n")
                self.__forest_size -= len(pop_indexes)

            ##################################################################################################
            # start with best scoring tree and add one by one if loss is decreased
            '''
            smallest_loss = 999999

            # get best tree
            for i in range(self.__forest_size):
                tree_loss = mean_squared_error(self.predict(input_data),target_data)

                if tree_loss < smallest_loss:
                    smallest_loss = tree_loss
                    index = i
            
            new_forest = [self.__forest[index]]
            base_loss = mean_squared_error(self.__predict_given_forest(new_forest, input_data),
                                           target_data)
            bad_trees = []
            j = 0
            # add trees
            for i in range(self.__forest_size):                
                if i!=index:
                    # calculate new loss
                    new_forest.append(self.__forest[i])
                    candidate_loss = mean_squared_error(self.__predict_given_forest(new_forest, input_data),
                                                        target_data)

                    if candidate_loss > base_loss:
                        bad_trees.append(new_forest.pop(j))
                        j-=1
                    else:
                        base_loss = candidate_loss
                    j+=1

            # empty old forest
            for i in range(self.__forest_size, 0, -1):
                self.__forest.pop(i-1)

            # fill forest with new one
            for i in range(len(new_forest)):
                self.__forest.append(new_forest[i])
            self.__forest_size = len(new_forest)

            print("Bad Trees:")
            for i in range(len(bad_trees)):
                print(bad_trees[i]['model'].details(), "Data Subset: ", bad_trees[i]['feature_index'])
            print("That is all, thank you\n")
            '''
            ##################################################################################################
