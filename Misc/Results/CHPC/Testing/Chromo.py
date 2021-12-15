'''
{From original GASOPE:
Each individual is made up of a set I of unique, term-coefficient mappings
tξ is made up of a set T of unique, variable-order mappings}

This class:

A chromosome that represents an array of var-order mappings with their coefficients

    y = C1*T1(x1) + C2 + ... etc.

where each chromosome represents a function approximation through multiple terms.

(stored in two seperate dictionaries, 
one describing the term, T: self.__var_orders
and one the coefficient, C: self.__coefficients
with terms sharing the same key: n ϵ N)
'''
# %%
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import math
import random
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore", message="Covariance of the parameters could not be estimated")

# %%


class Chromosone:
    def __init__(self, num_features, max_terms, max_poly_order, 
                 manual_init=False, var_orders=None, called_from_ensemble=False
                 ):

        self.__called_from_ensemble = called_from_ensemble

        if manual_init:
            self.__coefficients = {}
            for i in range(max_terms):
                self.__coefficients[i] = 1
            self.__num_terms = max_terms
            self.__num_features = num_features
            self.__max_poly_order = max_poly_order
            self.__num_terms = max_terms
            self.__reinit_term_count = max_terms
            self.__var_orders = var_orders
        else:
            self.__coefficients = {}  # calculated using least squares
            self.__var_orders = {}  # randomly calculated, evolved using GA
            self.__num_features = num_features
            self.__max_poly_order = max_poly_order
            self.__num_terms = max_terms
            self.__reinit_term_count = max_terms
            self.__fill(max_terms)

    # random initialisation for var term-order pairs
    # coefficients start as default 1
    def __fill(self, max_terms):
        # first term always bias term
        self.__coefficients[0] = 1
        self.__var_orders[0] = np.zeros(self.__num_features, dtype=int)

        # set var orders
        for i in range(1, max_terms):
            self.__coefficients[i] = 1

            # allow duplicate orders of different variable:
            '''
            self.__var_orders[i] = np.random.randint(0, self.__max_poly_order+1,(self.__num_features)) #allow unique terms vs:
            #self.__var_orders[i] = np.random.randint(1, self.__max_poly_order+1, (self.__num_features))
            '''
            # select number of avialable orders randomly per term
            num_choices = np.random.randint(0, self.__max_poly_order+1)

            # see if term should be left empty
            if num_choices == 0:
                self.__coefficients[i] = 1
                self.__var_orders[i] = np.zeros(self.__num_features, dtype=int)
            else:
                # choose orders that should make up the term
                available_orders = np.arange(self.__max_poly_order)+1
                chosen_orders = np.random.choice(available_orders,
                                                 num_choices,
                                                 replace=False)

                # check how many features there are before assigning var-order
                if chosen_orders.size < self.__num_features:
                    # unique orders variables:
                    self.__var_orders[i] = np.concatenate((chosen_orders,
                                                           np.zeros(self.__num_features-chosen_orders.size)))
                    np.random.shuffle(self.__var_orders[i])
                else:
                    # unique orders variables
                    self.__var_orders[i] = np.random.choice(
                        chosen_orders, self.__num_features)
                    np.random.shuffle(self.__var_orders[i])

        # remove repetitive var terms
        self.optimiseVarTerms()


    # Mutation operators: shrink, expand, petrub and reinit
    def ReInit(self):
        # clear 
        keys = []
        for key in self.__coefficients.keys():
            keys.append(key)
        for i in range(len(keys)):
            del self.__coefficients[keys[i]]
            del self.__var_orders[keys[i]]
        
        #re initialise
        self.__fill(self.__reinit_term_count)


    # Randomly remove an entire term
    def Shrink(self):
        # do not shrink if only one term:
        if len(self.__coefficients) >= 3:
            key = random.choice(list(self.__coefficients.keys()))
            ''' # error catching
            if key not in self.__var_orders.keys():
                print(self)
                print(self.__coefficients)
                print(self.__var_orders)
            '''
            del self.__coefficients[key]
            del self.__var_orders[key]
            self.__num_terms -= 1
        elif len(self.__coefficients) == 2:
            # do not shrink down to a single bias term
            no_bias_term = True
            for value in self.__var_orders.values():
                if (value == np.zeros(self.__num_features)).all():
                    no_bias_term = False
                    break
            if no_bias_term:
                key = random.choice(list(self.__coefficients.keys()))
                del self.__coefficients[key]
                del self.__var_orders[key]
                self.__num_terms -= 1

    # Randomly add an entire term
    def Expand(self):
        r = self.__num_terms
        # find a key value not yet in use
        while r in self.__coefficients:
            r += 1
        # assign coefficient as 1
        self.__coefficients[r] = 1

        # select available orders randomly per term
        num_choices = np.random.randint(1, self.__max_poly_order+1)
        available_orders = np.arange(self.__max_poly_order)+1
        chosen_orders = np.random.choice(available_orders,
                                         num_choices,
                                         replace=False)

        # check how many features there are before adding new term
        if chosen_orders.size < self.__num_features:
            # unique orders variables:
            self.__var_orders[r] = np.concatenate((chosen_orders,
                                                   np.zeros(self.__num_features-chosen_orders.size)))
            np.random.shuffle(self.__var_orders[r])
        else:
            # unique orders variables
            self.__var_orders[r] = np.random.choice(
                chosen_orders, self.__num_features)
            np.random.shuffle(self.__var_orders[r])

        # ensure there are no duplicates due to expand
        self.optimiseVarTerms()

    # Choose randomly a term's feature to be either randomly removed, added, or changed
    # do not perform Peturb on datasets containing a single feature (this is effectively just expand/shrink then)
    def Peturb(self):
        # Select action to be taken
        # action_choice = random.randint(1, 2)  # exclude choice 3
        action_choice = random.randint(1, 3)

        # select a term that is not the bias term
        non_bias_terms = []
        for key, value in self.__var_orders.items():
            if not (value == np.zeros(self.__num_features)).all():
                non_bias_terms.append(key)

        if len(non_bias_terms) > 0:
            term = random.choice(list(non_bias_terms))

            # remove var from term i.e. set var's order = 0
            if action_choice == 1:
                # find features with nonzero order value in term
                candidate_features = []
                for i in range(self.__var_orders[term].size):
                    if self.__var_orders[term][i] != 0:
                        candidate_features.append(i)

                # remove feature order mapping if possible
                if len(candidate_features) != 0:
                    feature = random.choice(candidate_features)
                    self.__var_orders[term][feature] = 0

            # change var order
            if action_choice == 2:
                # new value of var order. allowed to be same value
                feature = random.randint(0, self.__num_features-1)
                self.__var_orders[term][feature] = random.randint(
                    1, self.__max_poly_order)

            # add var to term i.e. var's order != 0
            if action_choice == 3:
                # find features with zero order value in term
                candidate_features = []
                for i in range(self.__var_orders[term].size):
                    if self.__var_orders[term][i] == 0:
                        candidate_features.append(i)

                # add var order mapping to term if possible
                if len(candidate_features) != 0:
                    feature = random.choice(candidate_features)
                    self.__var_orders[term][feature] = random.randint(
                        1, self.__max_poly_order)

            self.optimiseVarTerms()

    # produce offspring from two parents
    # KILLS PARENTS - use deep copy for parents;
    # Also assumes parents do not have duplicate var terms (i.e. optimiseVarTerms())
    def Crossover(self, other):
        # select randomly terms from both and create new individual from these terms
        # term present in both are given higher chances of selection
        inclusive_terms = {}
        exclusive_terms = {}

        # remove intersecting terms
        i = 0
        inclusive_keys1 = []
        for key1, value1 in self.__var_orders.items():
            is_match = False
            for key2, value2 in other.__var_orders.items():
                if (value1 == value2).all():
                    inclusive_terms[i] = self.__var_orders[key1]
                    inclusive_keys1.append(key1)
                    i += 1
                    is_match = True
                    break
            if is_match:
                del other.__var_orders[key2]
        for i in range(len(inclusive_keys1)):
            del self.__var_orders[inclusive_keys1[i]]

        # concatenate remaining outersects
        i = 0
        for value in self.__var_orders.values():
            exclusive_terms[i] = value
            i += 1
        for value in other.__var_orders.values():
            exclusive_terms[i] = value
            i += 1

        # randomly select from intersect and outersect var_orders for offspring
        offspring = {}
        i = 0
        for value in inclusive_terms.values():
            if random.random() < 0.8:
                offspring[i] = value
                i += 1

        for value in exclusive_terms.values():
            if random.random() < 0.2:
                offspring[i] = value
                i += 1

        # If zero genes from parents were passed down, create random child:
        # or if the only gene is the bias term
        if len(offspring) == 0 or (len(offspring) == 1 and (offspring[0] == np.zeros(self.__num_features)).all()):
            youngling = Chromosone(self.__num_features,
                                   self.__num_terms, self.__max_poly_order, called_from_ensemble=self.__called_from_ensemble)
        else:
            youngling = Chromosone(self.__num_features, len(offspring), self.__max_poly_order,
                                   manual_init=True, var_orders=offspring,  called_from_ensemble=self.__called_from_ensemble)

        return youngling

    # print individual's equational representation
    def getPlotTitle(self):
        line_out = ""
        term = ""
        split = False
        if len(self.__coefficients) > 4:
            split = True
        i = 0
        for key, value in self.__var_orders.items():
            term = "({:.3f})".format(self.__coefficients[key])
            for i in range(value.size):
                if value[i] != 0 and value[i] != 1:
                    term += "(x{}^{})".format(i+1, value[i])
                if value[i] == 1:
                    term += "(x{})".format(i+1)
            term += " + "
            if split and i == int(len(self.__coefficients)/2):
                line_out += "\n"
            line_out += term
            i += 1
        line_out = line_out[:-2]

        return line_out

    # print individual's equational representation
    def __str__(self):
        line_out = ""
        term = ""
        for key, value in self.__var_orders.items():
            term = "({:.3f})".format(self.__coefficients[key])
            for i in range(value.size):
                if value[i] != 0 and value[i] != 1:
                    term += "(x{}^{})".format(i+1, value[i])
                if value[i] == 1:
                    term += "(x{})".format(i+1)
            term += " + "
            line_out += term
        line_out = line_out[:-2]

        return line_out

    #lower is better
    def fitness(self, input_data, target_data, Mean_Square_Dev=1, weighted_terms=True):
        
        if not self.__called_from_ensemble:
            # code was originally written to accommodate data of the following shape (X x N)
            # this rectifies this:
            input_data = np.transpose(input_data)
            # default shape is now (N x X)

        if len(input_data.shape) != 1 and input_data.shape[0] != self.__num_features:
            print("Invalid input data array dimension, expected ",
                  self.__num_features, "x", target_data.size)
            print("got: ", input_data.shape)
            return 0

        if len(target_data.shape) != 1:
            print("Invalid target data array dimension, expected 1xN")
            print("got: ", target_data.size)
            return 0

        if weighted_terms:
            # term complexity:
            k = 0
            for value in self.__var_orders.values():
                k += np.sum(value)

                # punish higher orders more severely
                #k += np.sum(value) ** 2

            if k >= target_data.size:
                k = target_data.size - 1
            k = (target_data.size-1)/(target_data.size-k)
        else:
            k = 1

        # calculate predicted out
        w = self.getCoefficients()
        predicted_data = self.__func(input_data, *w)

        # temp fix for chromosomes with single bias term:
        if len(self.__var_orders) == 1:
            for value in self.__var_orders.values():
                if (value == np.zeros(self.__num_features)).all():
                    predicted_data = np.ones(target_data.size)*predicted_data
                    #print("crisis averted")

        # calculate MSE
        mse = mean_squared_error(target_data, predicted_data)
        mse = mse / Mean_Square_Dev

        return mse * k

    # calculate output of individual given a single pattern
    def out(self, input_data):

        if len(input_data.shape) == 1:  # univariate
            y = 0
            for key, value in self.__coefficients.items():
                y += value * input_data ** self.__var_orders[key][0]

        else:  # multivariate
            
            if not self.__called_from_ensemble:
                # code was originally written to accommodate data of the following shape (M x N)
                # this rectifies this:
                input_data = np.transpose(input_data)
                # default shape is now (N x M)

            if self.__num_features != input_data.shape[0]:
                print("num features invalid, expected:", self.__num_features)
                print("got: ", input_data.shape[0])
                return 0

            y = 0
            for key, value in self.__coefficients.items():
                term = value
                for i in range(input_data.shape[0]):
                    term *= input_data[i]**self.__var_orders[key][i]
                y += term

        return y

    # output var term-orders as 2D array, m x p: m-features, p-terms
    def toArray(self):
        array = []
        for value in self.__var_orders.values():
            array.append(value)

        return np.array(array)

    # return array of coefficients
    def getCoefficients(self):
        coefficients = np.zeros(self.__num_terms)
        i = 0
        for value in self.__coefficients.values():
            coefficients[i] = value
            i += 1

        return coefficients

    # groups duplicate var terms
    def optimiseVarTerms(self):
        # find repetitive keys
        repetitive_keys = {}
        skip_keys = []
        for key, value in self.__var_orders.items():
            repetitive_keys[key] = []
            for key2, value2 in self.__var_orders.items():
                if key != key2 and key not in skip_keys and (value == value2).all():
                    repetitive_keys[key].append(key2)
                    skip_keys.append(key2)

        # group repitive keys
        for key, value in repetitive_keys.items():
            if len(value) > 0:
                for i in range(len(value)):
                    self.__coefficients[key] += self.__coefficients[value[i]]
                    del self.__coefficients[value[i]]
                    del self.__var_orders[value[i]]

        # update number of terms
        self.__num_terms = len(self.__coefficients)

    # optimises coefficient value using least squares
    def least_squares_coefficients(self, input_data, target_data):

        if not self.__called_from_ensemble:
            # code was originally written to accommodate data of the following shape (X x N)
            # this rectifies this:
            input_data = np.transpose(input_data)
            # default shape is now (N x X)

        p0 = np.ones(self.__num_terms)
        # get new coefficient
        try:
            if self.__num_terms > target_data.size:
                w, _ = curve_fit(self.__func, input_data,
                                 target_data, p0=p0, method="dogbox")
            else:
                w, _ = curve_fit(self.__func, input_data,
                                 target_data, p0=p0, maxfev=1000)

        except RuntimeError:  # i.e. could not estimate using least squares
            w = p0
            # print('woops')

        # set new coefficients
        i = 0
        for key in self.__coefficients.keys():
            self.__coefficients[key] = w[i]
            i += 1

    # ONLY FOR CURVE_FIT USE, use out() to calculate individual's output
    def __func(self, x, *coefficients):

        '''
        if not self.__called_from_ensemble:
            # code was originally written to accommodate data of the following shape (N x X)
            # this rectifies this:
            x = np.transpose(x)
            print("hos")
            # default shape is now (N x X)
        '''

        # takes values for each feature and coefficient and calculates y = Lambda*x1*x2 + ...
        # ydata = f(xdata, *params) + eps.

        if len(x.shape) == 1:  # univariate
            y = 0
            i = 0
            for value in self.__var_orders.values():  # loop through terms
                y += coefficients[i] * x ** value[0]
                i += 1

        else:  # multivariate
            y = 0
            i = 0
            for value in self.__var_orders.values():  # loop through terms
                term = coefficients[i]
                for j in range(x.shape[0]):  # loop through features
                    if value[j] != 0:
                        term *= x[j]**value[j]
                y += term
                i += 1

        return y

    def TreePrintOut(self):
        equation = ""
        for key, value in self.__coefficients.items():

            if (self.__var_orders[key] == np.zeros(self.__num_features)).all():
                # coefficient of bias term
                equation += "({:.2})".format(value)

            else:
                equation += "({:.2})*".format(value)

            for i in range(self.__var_orders[key].size):
                if self.__var_orders[key][i] != 0:
                    if self.__var_orders[key][i] == 1:
                        equation += "(x_{})".format(i)
                    else:
                        equation += "(x_{}^{})".format(
                            i, int(self.__var_orders[key][i]))
            equation += " +\n"
        return equation[:-3]
