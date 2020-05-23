# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:14:09 2020

@author: timha

Test GA / NN Code file
Based on code from: https://github.com/ConorLazarou/GeneticAlgorithms/tree/master/connga

v1. Added various params as described in comments.
v2. TIM Added comments to ecosystem.generation fn to clarify what is going on.
v3. TIM Added 3 node output version with difft fn being learned for each output.

"""

import copy

import numpy as np
import random
from matplotlib import pyplot as plt
from ga_nn import Organism, Ecosystem



def simulate_and_evaluate(organism, actual_f, loss_f, replicates=500):
    """ TIM: creates a large batch of 500 numbers incrementing from 0..1 
             and passes them all into NN & returns 500 preds 
        TIM: Separated from main and expanded such that if you pass in a list of target fns
             instead of a single fn, then it will calculate the overall loss as the sum of 
             individual losses.
             NOTE: Number of target fns assumed to be the same as number of output neurons 
    """
    X = np.linspace(0, 1, replicates).reshape((replicates, 1))
    predictions = organism.predict(X)
    if type(actual_f) != list:    #TIM: Assume single loss calulated against a single output
        loss = loss_f(actual_f(X), predictions)
    else: 
        loss = 0                          #TIM: assume number of outputs = number of target functions
        for i in range(len(actual_f)):
            loss += loss_f(actual_f[i](X), predictions[:, i])                
    return -loss


def plot_fn_approximation(X, y_pred, y_true, title='Plot Title'):
    """ TIM: plot predictions against actual function
    """
    plt.scatter(X, y_pred, label='Predictions')
    plt.plot(X, y_true, label='Target', c='k')
    plt.legend()
    plt.title(title)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.show()
    

def main_single_output(mutate=True, crossover=True, mutation_std=0.05, 
                       crossover_rate=0.5, crossover_method='single_neuron',
                       activation='relu', output='linear', num_generations=1000,
                       architecture = [1, 16,16,16, 1], population_size=100,
                       mutate_single_layer=False, mutate_single_neuron=False,
                       mutate_strategy_turnon=11):
    """Learn and visualize a function from [0,1] to something else
    TIM: added configurable crossover, mutation and layer activation fn
    TIM: Added various extra args to main for more flexible running
    """
    np.random.seed(101)  #TIM Added for reproduceability
    actual_f = lambda x : np.sin(x*6*np.pi)  # the function to learn, y = sin(x * 6 * pi)
    loss_f = lambda y, y_hat : np.mean(np.abs(y - y_hat)**(2))  # the loss function (negative reward)
    X = np.linspace(0, 1, 200)


    scoring_function = lambda organism : simulate_and_evaluate(organism, actual_f, loss_f, replicates=500)

    original_organism = Organism(architecture, output=output, use_bias=True,
                                 activation=activation, mutation_std=mutation_std, 
                                 crossover_rate=crossover_rate, 
                                 mutate_single_layer=False, 
                                 mutate_single_neuron=False)
    
    ecosystem = Ecosystem(original_organism, scoring_function, population_size=population_size, 
                          mating=True, mutate=mutate, crossover=crossover,
                          crossover_method=crossover_method)
    best_organisms = [ecosystem.get_best_organism()]
    for i in range(num_generations):
        if i == mutate_strategy_turnon:
            print(f'{i} Turning on mutation strategy: Single Layer:{mutate_single_layer}  Single Neuron:{mutate_single_neuron}')
            ecosystem.turn_on_mutation_strategy(mutate_single_layer=mutate_single_layer,
                                                mutate_single_neuron=mutate_single_neuron)
        this_generation_best = ecosystem.generation()
        #this_generation_best = ecosystem.get_best_organism(include_reward=True)
        best_organisms.append(this_generation_best[0])
        if i % 10 == 0:
            print(f'{i}: {this_generation_best[1]:.2f}')

    y_pred = best_organisms[-1].predict(X.reshape(-1,1)).flatten()
    y_true = actual_f(X)
    plot_fn_approximation(X, y_pred, y_true, title=f'Sine Wave Generation {i}; Reward={this_generation_best[1]:.2f}')
    return


def main_three_output(mutate=True, crossover=True, mutation_std=0.05, 
                      crossover_rate=0.5, crossover_method='single_neuron',
                      activation='relu', output='linear', num_generations=1000, 
                      architecture = [1, 16,16,16, 3],
                      population_size=100,
                      mutate_single_layer=False, mutate_single_neuron=False,
                      mutate_strategy_turnon=11):
    """Learn and visualize a function from [0,1] to something else
    TIM: added configurable crossover, mutation and layer activation fn
    TIM: Added various extra args to main for more flexible running
    """
    np.random.seed(101)  #TIM Added for reproduceability
    actual_f1 = lambda x : np.sin(x*6*np.pi)             # the function to learn, y = sin(x * 6 * pi)
    actual_f2 = lambda x : np.cos(x*8*np.pi)             # the 2nd function to learn
    actual_f3 = lambda x : np.sqrt(5*x+0.21*x**4)        # the 3rd function to learn
    actual_f = [actual_f1, actual_f2, actual_f3]
    loss_f = lambda y, y_hat : np.mean(np.abs(y - y_hat)**(2))  # the loss function (negative reward)
    X = np.linspace(0, 1, 200)


    scoring_function = lambda organism : simulate_and_evaluate(organism, actual_f, loss_f, replicates=500)

    original_organism = Organism(architecture, output=output, use_bias=True,
                                 activation=activation, mutation_std=mutation_std, 
                                 crossover_rate=crossover_rate, 
                                 mutate_single_layer=False, 
                                 mutate_single_neuron=False)
    ecosystem = Ecosystem(original_organism, scoring_function, population_size=population_size, 
                          mating=True, mutate=mutate, crossover=crossover,
                          crossover_method=crossover_method)
    best_organisms = [ecosystem.get_best_organism()]
    for i in range(num_generations):
        if i == mutate_strategy_turnon:
            print(f'{i} Turning on mutation strategy: Single Layer:{mutate_single_layer}  Single Neuron:{mutate_single_neuron}')
            ecosystem.turn_on_mutation_strategy(mutate_single_layer=mutate_single_layer,
                                                mutate_single_neuron=mutate_single_neuron)
        this_generation_best = ecosystem.generation()
        #this_generation_best = ecosystem.get_best_organism(include_reward=True)
        best_organisms.append(this_generation_best[0])
        if i % 10 == 0:
            print(f'{i}: {this_generation_best[1]:.10f}')

    y_pred = best_organisms[-1].predict(X.reshape(-1,1))
    y_true = actual_f1(X)
    plot_fn_approximation(X, y_pred[:, 0], y_true, title=f'1 Sine Wave Generation {i}; Reward={this_generation_best[1]:.2f}')
    y_true = actual_f2(X)
    plot_fn_approximation(X, y_pred[:, 1], y_true, title=f'2 Cos Wave Generation {i}; Reward={this_generation_best[1]:.2f}')
    y_true = actual_f3(X)
    plot_fn_approximation(X, y_pred[:, 2], y_true, title=f'3 Cos Wave Generation {i}; Reward={this_generation_best[1]:.2f}')
    return



if __name__ == '__main__':
    main_single_output(mutate=True, crossover=True, mutation_std=0.025, crossover_rate=0.05, crossover_method='single_neuron',
         activation='tanh', num_generations=1000, architecture=[1, 16,16,16, 1], population_size=100,
         mutate_single_layer=True, mutate_single_neuron=False, mutate_strategy_turnon=11)

    main_three_output(mutate=True, crossover=False, mutation_std=0.025, crossover_rate=0.5, crossover_method='single_neuron',
         activation='tanh', num_generations=1000, architecture=[1, 32,32,32,32,64, 3], population_size=100,
         mutate_single_layer=True, mutate_single_neuron=False, mutate_strategy_turnon=11)



    #TIM tests below here
    """
    actual_f1 = lambda x : np.sin(x*6*np.pi)  # the function to learn, y = sin(x * 6 * pi)
    actual_f2 = lambda x : np.cos(x*13*np.pi)  # the function to learn, y = sin(x * 6 * pi)
    actual_f3 = lambda x : np.sqrt(5*x+0.21*x**4)        # the function to learn, y = sin(x * 6 * pi)

    loss_f = lambda y, y_hat : np.mean(np.abs(y - y_hat)**(2))  # the loss function (negative reward)

    X = np.linspace(0, 1, 200)
    org = Organism([1, 32,32,32,32,64, 3], output='linear', use_bias=True,
                                 activation='tanh', mutation_std=0.025, 
                                 crossover_rate=0.05, 
                                 mutate_single_layer=True, 
                                 mutate_single_neuron=True)
    
    org.mutate()
    
    y_pred = org.predict(X.reshape(-1,1))
    print(y_pred, 'shape=', y_pred.shape)  #Tim: (200, 3)
    y_pred[0,1]
    loss = simulate_and_evaluate(org, [actual_f1, actual_f2, actual_f3], loss_f)
    y_pred = org.predict(X.reshape(-1,1))
    y_true = actual_f1(X)
    i = 999
    plot_fn_approximation(X, y_pred[:, 0], y_true, title=f'1 Sine Wave Generation')
    y_true = actual_f2(X)
    plot_fn_approximation(X, y_pred[:, 1], y_true, title=f'1 Cos Wave Generation')
    y_true = actual_f3(X)
    plot_fn_approximation(X, y_pred[:, 2], y_true, title=f'1 Polynomial Generation')
    """


    
