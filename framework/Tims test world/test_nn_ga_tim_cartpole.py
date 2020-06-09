# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:41:20 2020

@author: timha

GA on cartpole from: https://gist.github.com/ConorLazarou/846e547725ea239c3edfcdcb99a9eb7a#file-connga_cartpole-py

"""

# Set up the environment and collect the observation space and action space sizes
env = gym.make("CartPole-v1")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# The function for creating the initial population
organism_creator = lambda : Organism([observation_space, 16, 16, 16, action_space], output='softmax')

def simulate_and_evaluate(organism, trials=1):
    """
    Run the environment `trials` times, using the organism as the agent
    Return the average number of timesteps survived
    """
    fitness = 0
    for i in range(trials):
        state = env.reset() # Get the initial state
        while True:
            fitness += 1
            action = organism.predict(state.reshape((1,-1)))
            action = np.argmax(action.flatten())
            state, reward, terminal, info = env.step(action)
            if terminal: # break if the agent wins or loses
                break
    return fitness / trials

# Create the scoring function and build the ecosystem
scoring_function = lambda organism : simulate_and_evaluate(organism, trials=5)
ecosystem = Ecosystem(organism_creator, scoring_function, 
                      population_size=100, holdout=0.1, mating=True)

generations = 200
for i in range(generations):
    ecosystem.generation()
    # [Visualization code omitted]
    if this_generation_best[1] == 500: # Stop when an organism achieves a perfect score
        break
    