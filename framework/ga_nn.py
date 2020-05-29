# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:42:40 2020

@author: timha
"""

import copy

import dill as pickle
import numpy as np
#import pickle
import random
from matplotlib import pyplot as plt


class Organism():
    def __init__(self, dimensions, use_bias=True, output='softmax', activation='relu', 
                 mutation_std=0.05, crossover_rate=0.5, mutate_single_layer=False, mutate_single_neuron=False):
        """ TIM: Added configurable crossover rate
            TIM: Added configurable hidden layer activation fn
        """
        self.layers = []
        self.biases = []
        self.use_bias = use_bias
        self.output = output
        self.output_activation = self._activation(output)
        self.layer_activation = self._activation(activation)  #TIM added
        self.mutation_std = mutation_std
        self.crossover_rate = crossover_rate
        self.mutate_single_layer = mutate_single_layer 
        self.mutate_single_neuron = mutate_single_neuron
        for i in range(len(dimensions)-1):
            shape = (dimensions[i], dimensions[i+1])
            std = np.sqrt(2 / sum(shape))
            layer = np.random.normal(0, std, shape)
            bias = np.random.normal(0, std, (1,  dimensions[i+1])) * use_bias
            self.layers.append(layer)
            self.biases.append(bias)
        self.fitness_score = 0                 #TIM Added for Malmo world

    def _activation(self, output):
        """Return a specified activation function.
        TIM: Added Relu
        
        output - a function, or the name of an activation function as a string.
                 String must be one of softmax, sigmoid, linear, tanh."""
        if output == 'softmax':
            return lambda X : np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == 'sigmoid':
            return lambda X : (1 / (1 + np.exp(-X)))
        if output == 'linear':
            return lambda X : X
        if output == 'tanh':
            return lambda X : np.tanh(X)
        if output == 'relu':  #TIM Added so we can configure hidden layer activations if we want
            return lambda X : np.clip(X, 0, np.inf)            
        else:
            return output
        
    def get_fitness(self):
        """ TIM: Added for Malmo world. Returns the fitness score of an individual
        """
        return self.fitness_score

    def predict(self, X):
        """Apply the function described by the organism to input X and return the output.
        TIM: Added hidden layer activation fn
        TIM: Returns last layer outputs. If last layer size = e.g. 3, it will return a tensor [batch_size, 3]
        """
        if not X.ndim == 2:
            raise ValueError(f'Input has {X.ndim} dimensions, expected 2')
        if not X.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}')
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            X = X @ layer + np.ones((X.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                X = self.output_activation(X) # output activation
            else:
                #X = np.clip(X, 0, np.inf)  # ReLU
                X = self.layer_activation(X)  # TIM added
        
        return X

    def predict_choice(self, X, deterministic=True):
        """Apply `predict` to X and return the organism's "choice".
        
        if deterministic then return the choice with the highest score.
        if not deterministic then interpret output as probabilities and select
        from them randomly, according to their probabilities.
        TIM: This would be used against softmax output. UNUSED BY US
        """
        probabilities = self.predict(X)
        if deterministic:
            return np.argmax(probabilities, axis=1).reshape((-1, 1))
        if any(np.sum(probabilities, axis=1) != 1):
            raise ValueError(f'Output values must sum to 1 to use deterministic=False')
        if any(probabilities < 0):
            raise ValueError(f'Output values cannot be negative to use deterministic=False')
        choices = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            U = np.random.rand(X.shape[0])
            c = 0
            while U > probabilities[i, c]:
                U -= probabilities[i, c]
                c += 1
            else:
                choices[i] = c
        return choices.reshape((-1,1))

    def mutate(self):
        """Mutate the organism's weights in place.
        TIM: Modifed to enable different mutation patterns:
            mutate_single_layer=False, mutate_single_neuron=False: mutate weights of all neurons in all layers
            mutate_single_layer=False, mutate_single_neuron=true: mutate weights of single neuron in all layers
            mutate_single_layer=True, mutate_single_neuron=true: mutate weights of single neuron in single layer
            mutate_single_layer=True, mutate_single_neuron=False: mutate weights of all neurons in single layer            
        """
        if self.mutate_single_layer:
            n_layers = len(self.layers) - 1
            i = random.randint(0, n_layers)
            if self.mutate_single_neuron:   #TIM mutate weights of single neuron in single layer i which will be column n of layers[i] weight matrix
                n = random.randint(0, self.layers[i].shape[1] - 1)
                #print('Single layer', i, 'single neuron', n, 'self.layers[i].shape', self.layers[i].shape ,'self.layers[i][:,n].shape', self.layers[i][:,n].shape) 
                self.layers[i][:,n] += np.random.normal(0, self.mutation_std, self.layers[i][:,n].shape)  
                if self.use_bias:
                    #print('Single layer', i, 'single neuron', n, 'self.biases[i].shape', self.biases[i].shape) 
                    self.biases[i][0, n] += np.random.normal(0, self.mutation_std, 1)  

            else:                   #TIM mutate all neurons in single layer
                self.layers[i] += np.random.normal(0, self.mutation_std, self.layers[i].shape)
                if self.use_bias:
                    self.biases[i] += np.random.normal(0, self.mutation_std, self.biases[i].shape)
        else:                        
            if self.mutate_single_neuron:  #TIM mutate weights of single neuron in each layer which will be column n of layers[i] weight matrix
                for i in range(len(self.layers)):
                    n = random.randint(0, self.layers[i].shape[1] - 1)
                    self.layers[i][:,n] += np.random.normal(0, self.mutation_std, self.layers[i][:,n].shape)  
                    if self.use_bias:
                        self.biases[i][0, n] += np.random.normal(0, self.mutation_std, 1) 
            else:                     #TIM the original code mutate weights of all neurons in all layers
                for i in range(len(self.layers)):
                    self.layers[i] += np.random.normal(0, self.mutation_std, self.layers[i].shape)
                    if self.use_bias:
                        self.biases[i] += np.random.normal(0, self.mutation_std, self.biases[i].shape)
        return
            

    def mate(self, other, mutate=True, crossover=True, crossover_method='single_neuron'):
        """Mate two organisms together, create an offspring, mutate it, and return it.
        TIM: Added ability to turn off crossover as well as configurable crossover rate
        """
        if self.use_bias != other.use_bias:
            raise ValueError('Both parents must use bias or not use bias')
        if not len(self.layers) == len(other.layers):
            raise ValueError('Both parents must have same number of layers')
        if not all(self.layers[x].shape == other.layers[x].shape for x in range(len(self.layers))):
            raise ValueError('Both parents must have same shape')

        
        child = copy.deepcopy(self)
        if crossover:
            if crossover_method != 'single_neuron':    # Same as original code
                for i in range(len(child.layers)):
                    pass_on = np.random.rand(1, child.layers[i].shape[1]) < self.crossover_rate   #TIM 0.5
                    child.layers[i] = pass_on * self.layers[i] + ~pass_on * other.layers[i]
                    child.biases[i] = pass_on * self.biases[i] + ~pass_on * other.biases[i]

            elif crossover_method == 'single_neuron':   # HENRY
                # Retrieve position of neuron to cross over
                n_layers = len(self.layers)
                neuron_layer = random.randint(0, n_layers)
                if neuron_layer != n_layers:
                    neuron_index = random.randint(0, self.layers[neuron_layer].shape[0] - 1)
                else:
                    neuron_index = random.randint(0, self.layers[neuron_layer - 1].shape[1] - 1)

                # Update weights out of neuron
                if neuron_layer != n_layers:
                    child.layers[neuron_layer][neuron_index] = other.layers[neuron_layer][neuron_index]

                # Update weights (and bias) into neuron
                if neuron_layer != 0:
                    child.layers[neuron_layer - 1][:, neuron_index] = other.layers[neuron_layer - 1][:, neuron_index]
                    child.biases[neuron_layer - 1][:, neuron_index] = other.biases[neuron_layer - 1][:, neuron_index]
                    
        if mutate:
            child.mutate()
        return child

    def organism_like(self):
        """Return a new organism with the same shape and activations but reinitialized weights.
        TIM: Added crossover_rate + layer activation fn"""
        dimensions = [x.shape[0] for x in self.layers] + [self.layers[-1].shape[1]]
        return Organism(dimensions, use_bias=self.use_bias, output=self.output, activation=self.layer_activation,
                        mutation_std=self.mutation_std, crossover_rate=self.crossover_rate,
                        mutate_single_layer = self.mutate_single_layer, 
                        mutate_single_neuron = self.mutate_single_neuron)


class Ecosystem():
    def __init__(self, holotype, scoring_function, population_size=100, holdout='sqrt', new_blood=0.1, 
                 mating=True, mutate=True, crossover=True, crossover_method='single_neuron'):
        """ TIM: Added mutate and crossover params so can turn each on/over with rate decided in Organism class
            TIM: Parameterised crossover_method
        """
        self.population_size = population_size
        self.population = [holotype.organism_like() for _ in range(population_size)]
        self.scoring_function = scoring_function
        if holdout == 'sqrt':
            self.holdout = max(1, int(np.sqrt(population_size)))
        elif holdout == 'log':
            self.holdout = max(1, int(np.log(population_size)))
        elif holdout > 0 and holdout < 1:
            self.holdout = max(1, int(holdout * population_size))
        else:
            self.holdout = max(1, int(holdout))
        self.new_blood = max(1, int(new_blood * population_size))
        self.mating = mating
        self.mutate = mutate
        self.crossover = crossover 
        self.crossover_method = crossover_method

        self.avg_fitness = -9e50
        self.max_fitness = -9e50


    def generation(self, verbose=False):
        """Run a single generation, evaluating, mating, and mutating organisms, returning the best one.
        TIM: Assuming population_size=100, holdout=10, new_blood=10, resulting population will be:
            90 mutated/crossover children of mostly fittest 10 parents  ("holdout" parents) 
            plus 9 new random individuals   (i.e new blood)
            plus the fittest individual from last time
        """
        rewards = []
        for index, organism in enumerate(self.population):
            if verbose:
                print(f'{index+1}/{self.population_size}', end='\r')
            reward = self.scoring_function(organism)
            rewards.append(reward)
        self.population = [self.population[x] for x in np.argsort(rewards)[::-1]]  #TIM: Sort population by fitness score descending
        best_organism = self.population[0]
        best_score = max(rewards)

        self.avg_fitness = sum(rewards)/len(rewards)
        self.max_fitness = max(rewards)

        new_population = []
        for i in range(self.population_size):
            parent_1_idx = i % self.holdout    #TIM for self.holdout = 10, parent_1_idx cycles between 0..9 i.e top 10 fittest organisms
            if self.mating:                    #TIM parent_2_idx = min(99, a random number between 0 and inf but weighted towards 0..10). Ie tend to mate with another fit individual
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx
            offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx], 
                                                           self.mutate, self.crossover,
                                                           self.crossover_method)
            new_population.append(offspring)  #TIM: create new pop of 100 mutated children if crossover enabled, or 100 mutated individuals otherwise
        
            
        for i in range(1, self.new_blood+1):  #TIM: self.new_blood = 10
            new_population[-i] = self.population[0].organism_like()  #TIM: Replace 10 mutated individuals with completely random individuals
        new_population[-1] = self.population[0]                      #TIM: Then replace one of those 10 with a direct copy of the fittest individual 
        self.population = new_population  #TIM: so our population now = 90 mutated/crossover children of mostly fittest parents plus 9 new random individuals plus the fittest individual from last time
        return best_organism, best_score


    def get_best_organism(self, include_reward=False):
        """Evaluate the ecosystem's organisms and return the best organism."""
        rewards = [self.scoring_function(x) for x in self.population]
        if include_reward:
            best = np.argsort(rewards)[-1]
            return self.population[best], rewards[best]
        else:
            return self.population[np.argsort(rewards)[-1]]
        
    def turn_on_mutation_strategy(self, mutate_single_layer=False, mutate_single_neuron=False):
        """ TIM: Turn on a particular mutation strategy for the whole population
        """
        for index, organism in enumerate(self.population):
            organism.mutate_single_layer = mutate_single_layer 
            organism.mutate_single_neuron = mutate_single_neuron

        return

    def set_population(self, new_population):
        if len(new_population) != self.population_size:
            raise Exception('New population is incorrect size, must be same size as ecosystem')
        self.population = new_population

def save_brains(ecosystem, path, filename):
    try:
        with open(f'{path}/{filename}.pickle', 'wb') as handle:
            pickle.dump(ecosystem, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except IOError:
        print('IOError')

def load_brains(path, filename):
        with open(f'{path}/{filename}.pickle', 'rb') as handle:
            brains = pickle.load(handle)
        return brains
    # except IOError:
    #     print(f'Loading {path}/{filename}.pickle failed.')
    #     return False

