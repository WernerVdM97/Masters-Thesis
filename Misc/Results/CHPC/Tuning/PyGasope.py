'''
Evolve multiple chromosones into a best fit polynomial
'''

# %%
import time
import sys
import math
import numpy as np
from copy import deepcopy
import time
from Chromo import Chromosone
from sklearn.metrics import mean_squared_error

# %%


def InitPop(num_features, max_terms, max_poly_order, population_size, 
            InjectLinear=False, called_from_ensemble = False
            ):
    
    pop = []
    if InjectLinear:
        pop.append(Chromosone(num_features, max_terms, 1, called_from_ensemble=called_from_ensemble))
        for _ in range(population_size-1):
            pop.append(Chromosone(num_features, max_terms, max_poly_order, called_from_ensemble=called_from_ensemble))

            '''
            # inject polys of different orders:
            orders = np.arange(max_poly_order-1)+2
            pop.append(Chromosone(num_features, max_terms, # max_poly_order))
                                np.random.choice(orders,1)[0]))
            '''
    else:
        for _ in range(population_size):
            pop.append(Chromosone(num_features, max_terms, max_poly_order, called_from_ensemble=called_from_ensemble))

            '''
            # inject polys of different orders:
            orders = np.arange(max_poly_order-1)+2
            pop.append(Chromosone(num_features, max_terms, # max_poly_order))
                                np.random.choice(orders,1)[0]))
            '''
    return np.array(pop)


def Evolve(max_terms, input_data, target_data, 
           crossover_rate=0.2, mutation_rate=0.35, elite=0.1, 
           max_poly_order=3, population_size=30, generations=100, 
           debug=False, Progress=False, Progress_info=[1, 1], LinearCompare=False,
           called_from_ensemble = False
           ):
    '''Assumes data is given in numpy array with (N x F)'''
    
    t_lst = 0
    t_fit = 0
    t_new_pop = 0
    t_genetics = 0

    # calculate population breakdown
    num_survivors = int(population_size*elite+0.5)
    num_mutations = int(population_size*mutation_rate+0.5)
    num_younglings = population_size - num_mutations - num_survivors
    parents = int(population_size*crossover_rate+0.5)
    if parents < 2: # incase crossover rate is too small.
        parents=2

    # keep record of best mse score each generation
    score = np.zeros(generations)

    if len(input_data.shape) == 1:  # univariate
        num_features = 1
    else:
        if called_from_ensemble:
            num_features = input_data.shape[0]
        else :
            num_features = input_data.shape[1]

    pop = InitPop(num_features, max_terms, max_poly_order,
                  population_size, InjectLinear=False, called_from_ensemble = called_from_ensemble)
    #HallOfFame = deepcopy(pop[0])
    fit = np.zeros(population_size)

    # Calculate Mean Square Deviation for fitness unification of each chromosone
    average = np.ones(target_data.size) * np.average(target_data)
    Mean_Square_Dev = mean_squared_error(target_data, average)

    # generations
    for g in range(generations):

        if Progress:
            #print("\rPyGasope progress: {:.2F}".format(100*g/population_size))
            sys.stdout.write('\r')
            sys.stdout.write("Forest Growth: [%-10s] %03d%% Tree Growth: [%-10s] %03d%% Building Poly: [%-10s] %03d%%" % (
                '='*int(Progress_info[1]*10), Progress_info[1]*100,
                '='*int(Progress_info[0]*10), Progress_info[0]*100,
                '='*int(10*g/generations), int(100*g/generations)))
            sys.stdout.flush()

        # For each individual
        for i in range(len(pop)):

            t_temp = time.time()

            # re-estimate coefficients
            pop[i].least_squares_coefficients(input_data, target_data)

            t_lst += time.time()-t_temp
            t_temp = time.time()

            # evaluate fitness
            fit[i] = pop[i].fitness(input_data, target_data,
                                    Mean_Square_Dev=Mean_Square_Dev, 
                                    weighted_terms=True)

            t_fit += time.time() - t_temp

        # Sort by fitness and add to Hall of Fame (for use when sampling data)
        sorted_indexes = np.argsort(fit)
        pop = pop[sorted_indexes]
        score[g] = pop[0].fitness(input_data, target_data, weighted_terms=False)
        
        # Hall of fame only necessary for dataset sampling.
        '''        
        if pop[0].fitness(input_data, target_data, weighted_terms=False) < HallOfFame.fitness(input_data, target_data, weighted_terms=False):
            HallOfFame = deepcopy(pop[0])
        '''
        t_temp = time.time()

        # crossover
        younglings = []

        for i in range(0, num_younglings):
            # two unique random parents chosen from top y% candidates
            parent_index = np.random.choice(parents, 2, replace=False)
            p1 = deepcopy(pop[parent_index[0]])
            p2 = deepcopy(pop[parent_index[1]])
            younglings.append(p1.Crossover(p2))
        younglings_index = slice(num_survivors, num_survivors + num_younglings)
        pop[younglings_index] = np.array(younglings)

        # mutation
        mutation_candidates = []
        for i in range(num_mutations):
            # select indiv for mutation from next gen candidates
            mutation_index = np.random.randint(
                0, num_survivors+num_younglings)  # + num_younglings
            mutation_candidates.append(deepcopy(pop[mutation_index]))

            # randomly select type of mutation
            action_choice = np.random.randint(1, 5)
            if action_choice == 1:
                mutation_candidates[i].Shrink()
            if action_choice == 2:
                mutation_candidates[i].Expand()
            if action_choice == 3:
                mutation_candidates[i].Peturb()
            if action_choice == 4:
                mutation_candidates[i].ReInit()

            # add mutation to next gen
            pop[num_survivors+num_younglings+i] = mutation_candidates[i]

        t_genetics += time.time() - t_temp
        t_temp = time.time()
        
        '''
        # evolve new pop
        new_pop = population_size - \
            (num_survivors + num_younglings + num_mutations)
        new_pop_index = slice(
            num_survivors + num_younglings + num_mutations, population_size)
        pop[new_pop_index] = InitPop(
        num_features, max_terms, max_poly_order, new_pop)
        '''

        t_new_pop += time.time()-t_temp

    # add one linear model for comparison
    if LinearCompare:
        var_orders = {}

        # check for one dimensional data
        if called_from_ensemble:
            if len(input_data.shape) == 1:
                F = 1
            else:
                F = input_data.shape[0]
        else :
            if len(input_data.shape) == 1:
                F = 1
            else:
                F = input_data.shape[1]

        # fill base var-orders to get unique linear variable order terms
        var_orders[0] = np.zeros(F)
        for i in range(F):
            var_orders[i+1] = np.zeros(F)
            var_orders[i+1][i] = 1

        lin_model = Chromosone(
            F, F+1, 1, manual_init=True, var_orders=var_orders,called_from_ensemble=called_from_ensemble)
        lin_model.least_squares_coefficients(input_data, target_data)

    # final sort to get best indiv
    for i in range(population_size):
        pop[i].least_squares_coefficients(input_data, target_data)
        fit[i] = pop[i].fitness(input_data, target_data)
    sorted_indexes = np.argsort(fit)
    pop = pop[sorted_indexes]
    score[-1] = pop[0].fitness(input_data, target_data, weighted_terms=False)

    if debug:
        print("least squares: ", t_lst)
        print("fitness: ", t_fit)
        print("crossover/mutation: ", t_genetics)
        print("new population: ", t_new_pop)

    # Hall of fame unnecessary if data set is not subsampled each generation
    '''
    if pop[0].fitness(input_data, target_data, weighted_terms=True) <= HallOfFame.fitness(input_data, target_data, weighted_terms=True):
        pop[0].optimiseVarTerms()
        #print("Chose Non-HOF\nTrue MSE diff: ", pop[0].fitness(input_data, target_data, weighted_terms=False) - HallOfFame.fitness(input_data, target_data, weighted_terms=False))
        return pop[0]
    else:
        HallOfFame.optimiseVarTerms()
        #print("Chose HOF\nTrue MSE diff: ", HallOfFame.fitness(input_data, target_data, weighted_terms=False)- pop[0].fitness(input_data, target_data, weighted_terms=False))
        return HallOfFame
    '''

    # final comparison with simple lin model. Without complexity punishment
    if LinearCompare:
        if pop[0].fitness(input_data, target_data, weighted_terms=False) < lin_model.fitness(input_data, target_data, weighted_terms=False):
            pop[0].optimiseVarTerms()
            return pop[0], score
        else:
            return lin_model, score
    else:
        return pop[0], score

# %%
