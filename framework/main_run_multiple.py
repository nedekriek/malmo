""" Tim: modified form of Matthews worksheet_4 code
Dont forget to run the minecraft environment before trying to run this code...
Run this code from within the directory you have put this code in..

Tim:
    23/5: Made all important params part of a dictionary called C for ease of updates
          Added fn to calculate the sensory info shape
          Added GA/NN functions

"""

from __future__ import print_function
from builtins import range
import itertools
#import MalmoPython    #TIM
import malmo.MalmoPython as MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import re
import malmoutils
import cs765_utils
from cs765_utils import cuboid,entity
from drawing import SensoryVisualization
import numpy as np
malmoutils.fix_print()

import matplotlib.pyplot as plt

#TIM:
import skimage.measure
import copy
from ga_nn import Organism, Ecosystem, save_brains, load_brains
np.random.seed(101)  #TIM Added for reproduceability


def get_sensory_info_shape(C):
    """ TIM This is a stripped down version of the actual Agent.get_sensory_info()
            function whose only purpose is to allow calculation of the number of neurons 
            in the first layer of the NN. IF the real get_sensory_info() fn is changed
            in a way that it's output shape is changed then this fn must be correspondingly
            modified.
    """
    arr = np.ones((C['h'], C['w'], 4), dtype=np.float32) / 255  #dummy input for calculating flattened sensory input size..
    if C['grayscale']:
        grey = np.expand_dims(rgb2gray(arr), axis=2)
        arr = np.concatenate((grey, np.expand_dims(arr[:,:,3], axis=2)), axis=2)
    h,w,c = arr.shape
    downsample_h = h // C['s_height']
    downsample_w = w // C['s_width']
    sensory_info = skimage.measure.block_reduce(arr, (downsample_h, downsample_w, 1), C['s_fn'])
    shape = sensory_info.shape
    if C['flatten']:
        sensory_info = sensory_info.flatten()
        shape = sensory_info.shape
        sensory_info = sensory_info.reshape(1,sensory_info.shape[0])  #reshape to [1, n_features] (the 1 is the batch size)
    return shape, sensory_info


#TIM: C is a dictionary of parameters we want an agent to run with.
C = {}
C['image_log'] = "image_log/"
C['save_dir'] = "logs/"
C['label'] = 'default'  # Henry: Distinct labels specify distinct 'runs'
C['XML'] = "./scenarios/scenario1.xml"  #Mitchell:  scenario1.xml , scenario2.xml , scenario3.xml in increasing difficulty
C['h'] = 240                 #minecraft image dims (need to be changed in XML as well as here)
C['w'] = 320
C['actions'] = ['turn', "strafe", "move", "use", "pitch"]
C['num_actions'] = len(C['actions'])
C['turn_thresh'] = 0.5
C['strafe_thresh'] = 0.5
C['move_thresh'] = 0.05
C['use_thresh'] = 0.5
C['pitch_thresh'] = 0.9
C['turn_multiplier'] = 1.0
C['strafe_multiplier'] = 1.0
C['move_multiplier'] = 1.0
C['pitch_multiplier'] = 1.0
C['Goal'] = [0, 9]                      #Mitchell: The X and Z coordinate of the goal, this doesn't between XML files, Y doesn't matter

#Mitchell: Fitness function parameters
C['step_reward'] = 0                    #TIM: Could be -1, 0 or +1. This is the reward returned for surviving a single step.
C["Dist_multiplier"] = 200              #Mitchell: Weight of distance on result
C["Dist_threshold"] = 9                 #Mitchell: How many blocks away before we take away points (9 is start position)
C["Suffocation_penalty"] = -20000       #Mitchell: Penalty for killing itself
C["Goal_reward"] = 20000                #Mitchell: Reward for reaching the goal
C["Lava_penalty"] = -2000               #Mitchell: Penalty for dying in the lava
C["Time_penalty"] = -1800                #Mitchell: Penalty for running out of time
C["Time_multiplier"] = -100             #Mitchell: Multiplier for time taken

#TIM: below are agent.get_sensory_info() params:
C['s_height'] = 10           # was 60 height of sensory representation. Minecraft Image height must be exact multiple of this
C['s_width'] = 10            # was 80 width of sensory representation. Minecraft Image height must be exact multiple of this
C['s_fn'] = np.mean          # any np fn, could try np.max
C['flatten'] = True          # T/F
C['grayscale'] = False       # T/F
C["output_image"] = False    # If True, RGB_output will determine type of output
C['RGB_output'] = True 
C['threshold']  = 0.3        # 0 = turn off
C['normalise'] = True

C['log'] = True              # log=True will log the path and actions of the agent in a csv file in C['save_dir']

dummy_sensory_info_shape, dummy_sensory_info = get_sensory_info_shape(C)
print('NN input vector shape with above params = ', dummy_sensory_info_shape[0])   #4800

# GA / NN Params:
C['mutate']=True, 
C['crossover']=True, 
C['mutation_std']=0.025
C['crossover_rate']=0.25
C['crossover_method']='single_neuron'
C['activation']='tanh'
C['output']='linear'
C['num_generations']=3
# NN architecture - NOTE: the first layer value must be the actual size of the sensory_input
#                          and the last layer must be the number of actions
C['architecture'] = [ dummy_sensory_info_shape[0], 
                      dummy_sensory_info_shape[0] // 2, 
                      dummy_sensory_info_shape[0] // 10, 
                      dummy_sensory_info_shape[0] // 10, 
                      C['num_actions']]       
C['population_size']=10
C['mutate_single_layer']=True
C['mutate_single_neuron']=True
C['mutate_strategy_turnon']=6         #number of generations to initially run with all weights mutated before switching to mutate single/layer/neuron settings.

#Agent brain params - each agent will have a population of brains
C['original_brain'] = Organism(C['architecture'], 
                               output=C['output'], use_bias=True,
                               activation=C['activation'], 
                               mutation_std=C['mutation_std'], 
                               crossover_rate=C['crossover_rate'], 
                               mutate_single_layer=False, 
                               mutate_single_neuron=False)  #last 2 params will be set after C['mutate_strategy_turnon'] generations
dummy_predictions = C['original_brain'].predict(dummy_sensory_info)  #Tim This will throw an error if NN arch doesnt match sensory input shape
print('Shape of NN output  (1, num_actions):', dummy_predictions.shape)
assert dummy_predictions.shape[1] == C['num_actions'], "NN output != number of actions. Something is wrong."

print('Running with parameters:', C)

#Mitchell
from PIL import Image

it = 0


def eval_fitness(brain):
    """ Tim: returns the fitness score of an individual agent brain 
    """
    return brain.get_fitness()


def rgb2gray(rgb):
    """ TIM: from https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class Agent(object):
    newid = itertools.count()
    def __init__(self, C, brains=None):
        self.count = 0                              #Used to keep track of how many inputs it has taken
        self.host = MalmoPython.AgentHost()
        malmoutils.parse_command_line(self.host)
        self.index = 0#next(Agent.newid)              #First agent is 0, second is 1...
        self.iteration = 0                          #Used to generate CSV files
        self.C = copy.deepcopy(C)                   #TIM: dictionary of parameters. The deepcopy is to force it to take a copy of C rather than just pointing to C so we can have multiple agents with different Cs

        #TIM: Note each brain's fitness is set after running the game with that agent/brain, the scoring fn just retrieves it
        self.scoring_function = lambda brain : eval_fitness(brain)  #Will return the current fitness score from this brain     # TODO: Why not just brain.fitness_score ?
        #TIM: Create population of agent brains
        self.brain_population = Ecosystem(self.C['original_brain'], 
                              self.scoring_function,
                              initial_population=brains,
                              population_size=self.C['population_size'], 
                              mating=True, 
                              mutate=self.C['mutate'], 
                              crossover=self.C['crossover'],
                              crossover_method=self.C['crossover_method'],
                                          )
        

    def connect(self, client_pool):
        self.client_pool = client_pool
        self.client_pool.add( MalmoPython.ClientInfo('127.0.0.1',10000+self.index) )

    def safeStartMission(self, mission, client_pool):
        self.recording_spec = MalmoPython.MissionRecordSpec()
        cs765_utils.safeStartMission(self.host, mission, client_pool, self.recording_spec, self.index, '' )

    def is_mission_running(self):
        return self.host.peekWorldState().is_mission_running
    
    def pixels_to_numpy(self, pixels):
        """ TIM convert malmo pixels to np array shape [h,w,c], c = rgbd channel respectively
        Optionally normalize values to between 0.0 & 1.0
        """
        arr = np.array(pixels,dtype=np.uint8).reshape(self.C['h'], self.C['w'], 4).astype(np.float32)
        if self.C['normalise']:
            arr = arr / 255.0
        return arr

    def output_image_RGB(self, sensory_info):
        """Mitchell: Using this as a debug so we can see what the bot is seeing when RGB sensors are used
        Output: Saves a PNG for the RGB output and a PNG for the depth output to a folder in the directory
        """
        image = np.zeros(sensory_info.shape, dtype=np.uint8)
        image[:,:,0] = (sensory_info[:,:,0]*255)
        image[:,:,1] = (sensory_info[:,:,1]*255)
        image[:,:,2] = (sensory_info[:,:,2]*255)
        image[:,:,3] = ((1-sensory_info[:,:,3])*255) #Invert disparity into depth (White = close, Black = far)
        img = Image.fromarray(image[:,:,:3])
        img.save(self.C['image_log']+str(self.count)+'.png')
        img = Image.fromarray(image[:,:,3])
        img.save(self.C['image_log']+str(self.count)+'_d.png')
        self.count += 1

    def output_image_3bit(self, sensory_info):
        """Mitchell: Using this as a debug so we can see what the bot is seeing when 3bit output is used
        Not actually 3 bit, just each RGB channel is a binary value
        - May need improving when in the scene, some pixels are being classified as red and blue
            Solution would be to mask the most important channel, and set all in the other channel that are true in the mask to 0
        Input: sensory_info np array
        Output: Saves a PNG for the RGB output and a PNG for the depth output to a folder in the directory
        """
        image = np.zeros(sensory_info.shape, dtype=np.uint8)
        image[:,:,:3] = sensory_info[:,:,:3]
        image[:,:,3] = ((1-sensory_info[:,:,3])*255) #Invert disparity into depth (White = close, Black = far)
        img = Image.fromarray(image[:,:,:3])
        img.save(self.C['image_log']+str(self.count)+'.png')
        img = Image.fromarray(image[:,:,3])
        img.save(self.C['image_log']+str(self.count)+'_d.png')
        self.count += 1
       
    def get_sensory_info(self, arr):
        """ TIM Convert raw numpy array into sensory information to be input 
                into our agent's algorithm
        Inputs:
        - threshold - if set to 0 no threshold is applied, otherwise set to a value between 0-1
        - RGB_output - True: logs images as RGB, False: logs as 3 binary channels
        NOTE: To work properly: 
            h must be an exact multiple of s_height
            w must be an exact multiple of s_width
            Some valid (s_height, s_width) combinations for h=240,w=320:
                (240,320) - no reduction performed
                (1,1), (3,4), (4,4), (5,5), (6,5), (8,8), (16,16) ...
        Returns: (if grayscale = True, c will be 2)
            tensor shape [s_height, s_width, c] if flatten = False  (in the unlikely event we want to try convolutions)
            vector shape [s_height * s_width * c] if flatten = True (suitable for input into fully connected layer)
        """
        if self.C['grayscale']:
            grey = np.expand_dims(rgb2gray(arr), axis=2)
            arr = np.concatenate((grey, np.expand_dims(arr[:,:,3], axis=2)), axis=2)
        h,w,c = arr.shape
        downsample_h = h // self.C['s_height']
        downsample_w = w // self.C['s_width']
        sensory_info = skimage.measure.block_reduce(arr, (downsample_h, downsample_w, 1), self.C['s_fn'])

        #Threshold values to distinguish parts of the scene - Mitchell
        if self.C['threshold'] != 0:
            sensory_info[:,:,0] = (sensory_info[:,:,0] > self.C['threshold'])
            sensory_info[:,:,1] = (sensory_info[:,:,1] > self.C['threshold'])
            sensory_info[:,:,2] = (sensory_info[:,:,2] > self.C['threshold'])

        if self.C['output_image']:  #TIM Added so can turn off/on
            #Used for debugging, so we can see what the agent sees in a log
            if self.C['RGB_output']:
                #If we provide RGB instead of the compressed information
                self.output_image_RGB(sensory_info)
            else:
                self.output_image_3bit(sensory_info)
            
        if self.C['flatten']:
            sensory_info = sensory_info.flatten()
            sensory_info = sensory_info.reshape(1, sensory_info.shape[0])  #reshape to [1, n_features] (the 1 is the batch size)
        return sensory_info

    def pixels_as_arrays(self, pixels):
        """Takes pixel data and returns four arrays, for 
        RED, GREEN, BLUE, and DEPTH of each pixel.  Values 
        are returned as floats between 0 and 1.

        """
        arr = self.pixels_to_numpy(pixels, normalise=True )  #TIM
        #arr = np.array(pixels,dtype=np.uint8).reshape(240,320,4)
        r = arr[:,:,0]    #.astype(np.float32)/255.0
        g = arr[:,:,1]    #.astype(np.float32)/255.0
        b = arr[:,:,2]    #.astype(np.float32)/255.0
        d = arr[:,:,3]    #.astype(np.float32)/255.0
        return r,g,b,d

    def output_actions(self, actions):
        """Mitchell: Saves a log of the X, Z, Yaw, Pitch, Action of the agent"""
        #save_dir = "logs/"
        with open(self.C['save_dir']+str(self.iteration)+'.csv', 'w') as file:
            for action in actions:
                #Write X, Z, Yaw, Pitch, Action for each decision
                file.write(str(action)[1:-1] + '\n')

    def run(self, brain=0):
        """Mitchell: This function runs the agent on the world, log=True will log the path and actions of the agent in a csv file.
           Tim: added brain argument so now this function will run an agent with a 
                particular brain (out of it's population of brains).
        """
        distance_from_goal = -1
        total_reward = []   # List of all rewards received
        current_reward = 0  # Current reward
        self.iteration += 1
        actions = []        # List of agents past for logging
        
        # Wait for a valid observation
        world_state = self.host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = self.host.peekWorldState()
            
        # Wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = self.host.peekWorldState()
        world_state = self.host.getWorldState()
        for error in world_state.errors:
            print(error)

        self.host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)

        # Check mission is still running
        if not world_state.is_mission_running:
            return 0

        assert len(world_state.video_frames) > 0, 'No video frames!?'

        # Here is the actual agent running
        start_time = time.time()

        # Act once
        action, returned_reward = self.act(brain)
        total_reward.append(returned_reward)
        if world_state.number_of_observations_since_last_state > 0 and self.C['log']:
            msg = world_state.observations[-1].text
            obs = json.loads(msg)
            time_point = [obs["XPos"], obs["ZPos"], obs["Pitch"], obs["Yaw"], action] 
            actions += [time_point]
        
        while world_state.is_mission_running:
            world_state = self.host.peekWorldState()
            action, returned_reward = self.act(brain)        #TIM Added brain
            total_reward.append(returned_reward)             #Step cost

            #Mitchell: Extract relevant information. Can also get "YPos", "Yaw", "Pitch"
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                obs = json.loads(msg)
                distance_from_goal = math.sqrt((obs["XPos"]-C["Goal"][0])**2 + (obs["ZPos"]-C["Goal"][1])**2)  #Mitchell: Eudclidean distance between current pos and goal 
                if self.C['log']:
                    time_point = [obs["XPos"], obs["ZPos"], obs["Pitch"], obs["Yaw"], action]
                    actions += [time_point]

        #Mitchell: Transform current reward into what we have in the fitness function parameters
        current_reward = sum(r.getValue() for r in world_state.rewards) #Current reward is the reward from the world (Death reward)
        if current_reward == 1000:
            current_reward = C["Goal_reward"]
        elif current_reward == 0:
            current_reward = C["Lava_penalty"]
        elif current_reward == -100:
            current_reward = C["Lava_penalty"]
        elif current_reward == 100:
            current_reward = C["Time_penalty"]
        total_reward.append(current_reward)

        #Mitchell: Add time taken to fitness
        time_taken = time.time() - start_time
        time_reward= time_taken*C["Time_multiplier"]
        total_reward.append(time_reward)
        
        #Mitchell: Add distance to goal to the reward
        if distance_from_goal != -1: #Sometimes distance from goal isn't calculated
            distance_from_goal_reward = (C["Dist_threshold"]-distance_from_goal)*C["Dist_multiplier"]
            total_reward.append(distance_from_goal_reward)

        total_reward_sum = sum(total_reward)
        #Log actions and reward
        if self.C['log']:
            #The last line of the CSV shows: Total reward, distance reward, time_reward, current_reward, time taken
            actions += [[total_reward_sum, distance_from_goal_reward, time_reward, current_reward, time_taken]]
            self.output_actions(actions)

        self.brain_population.population[brain].fitness_score = total_reward_sum   #TIM: super important to assign each agent brain a fitness score
        return (total_reward_sum, time_taken)

    
    def act(self, brain=0):
        """Mitchell: This function is called to make one move in the world
           Tim: Added brain
        """
        actions = ['turn', "strafe", "move", "use", "pitch"]
        action = "None"

        world_state = self.host.getWorldState()

        if len(world_state.video_frames) >= 1:
            pixels = world_state.video_frames[-1].pixels
            #red,green,blue,depth = self.pixels_as_arrays(pixels)
            #sv.display_data(blue)
            arr = self.pixels_to_numpy(pixels)
            sensory_info = self.get_sensory_info(arr)
            
            actions = self.brain_population.population[brain].predict(sensory_info)  #should return [1, num_actions]
            
            #TODO TIM THIS PART NEEDS WORK. WE NEED TO DECIDE WHATE TO DO WITH THE 5 values that we now have.
            turnval = actions[0, 0]
            if abs(turnval) > C['turn_thresh']:
                actval = np.clip(turnval*C['turn_multiplier'], -1, 1)
                action = "turn " + str(actval)
                self.host.sendCommand(action)
            strafeval = actions[0, 1]
            if abs(strafeval) > C['strafe_thresh']:
                actval = np.clip(strafeval*C['strafe_multiplier'], -1, 1)
                action = "strafe " + str(actval)
                self.host.sendCommand(action)
            moveval = actions[0, 2]
            if abs(moveval) > C['move_thresh']:
                actval = np.clip(moveval*C['move_multiplier'], -1, 1)
                action = "move " + str(actval)
                self.host.sendCommand(action)
            useval = actions[0, 3]
            if abs(useval) > C['use_thresh']:  #Lets say below thresh means "don't use anything"
                action = "use 1"                
                self.host.sendCommand(action)
            pitchval = actions[0, 4]  # not sure what to do with pitch so ignoring for now
            if abs(pitchval) > C['pitch_thresh']:
                actval = np.clip(pitchval*C['pitch_multiplier'], -1, 1)
                action = "pitch " + str(actval)
                self.host.sendCommand(action)
        
        return action, C['step_reward'] # could be positive number ot reward longevity, 0, or negative number to reward efficiency

def plot_fitness(max_fitness, avg_fitness):
    """ TIM: plot fitness by generation
    Mitchell: plotting seperate plots for max and average
    """
    fig, ax = plt.subplots(2)
    ax[0].plot(max_fitness, label='Max', c='k')
    ax[0].set(xlabel='Generation', ylabel='Fitness', title="Max Fitness By Generation")
    ax[1].plot(avg_fitness, label='Avg')
    ax[1].set(xlabel='Generation', ylabel='Fitness', title="Average Fitness By Generation")
    plt.tight_layout()
    plt.savefig(C['save_dir']+"_Fitness.png")
    #plt.show()


def update_dictionary(C, override):
    """
    Mitchell: Change parameters we want for the experiment
    """
    for key, value in override.items():
        C[key] = value
    return C


def run_incremental_training(C, override_list):
    for idx, override in enumerate(override_list):
        if idx == 0:
            brains = main(C, override=override)
        else:
            brains = main(C, brains=brains, override=override)


def main(C, brains=None, override={}):
    generational_max_fitness = []
    generational_avg_fitness = []
    overall_best_fitness = -9e300
    overall_best_fitness_gen = 0
    overall_best_brain = None
    have_saved_brains = False   # False until loop has saved the initial set of brains, for max/avg stats
    
    C = update_dictionary(C, override)

    agent = Agent(C, brains) #Single agent

    mission_xml = cs765_utils.load_mission_xml(C['XML'])
    print(mission_xml)
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

    client_pool = MalmoPython.ClientPool()
    agent.connect(client_pool)

    #Using code from tabular_q_learning
    for i in range(C['num_generations']):
        if i == C['mutate_strategy_turnon']:
            print(f'{i} Turning on mutation strategy: Single Layer:{C["mutate_single_layer"]}  Single Neuron:{C["mutate_single_neuron"]}')
            agent.brain_population.turn_on_mutation_strategy(mutate_single_layer=C['mutate_single_layer'],
                                                             mutate_single_neuron=C['mutate_single_neuron'])
        
        print("\n Map - Mission %d of %d:" % (i+1, C['num_generations'] ))
        rewards = []
        
        for j in range(C['population_size']):  #TIM: run agent with each brain in it's brain_population
            agent.safeStartMission( my_mission, client_pool)    
    
            print("Waiting for the mission to start", end=' ')
            cs765_utils.safeWaitForStart([agent.host])
    
            # Focus the agent on the item in the hotbar
            agent.host.sendCommand("hotbar.9 1")
            agent.host.sendCommand("hotbar.9 0")

            print(f'Testing brain {j} for the agent...')
            total_reward = agent.run(j)  # returns (total_reward, time_taken)
            print(f'Gen {i} Brain {j} Reward (total_reward, time_taken):', total_reward)
            rewards.append(total_reward)
            # -- clean up -- #
            time.sleep(0.5) # (let the Mod reset)

        print(f'Gen:{i}  Evolving population based on fitness scores...')
        this_generation_best_brain = agent.brain_population.generation()

        gen_avg_reward = sum([r[0] for r in rewards]) / C['population_size']
        gen_max_reward = max([r[0] for r in rewards])
        if gen_max_reward > overall_best_fitness:
            overall_best_fitness = gen_max_reward
            overall_best_fitness_gen = i
            overall_best_brain = this_generation_best_brain

        if have_saved_brains:
            best_avg_brains = load_brains(path='saved_brains', filename=('best_avg_' + C['label']))
            best_max_brains = load_brains(path='saved_brains', filename=('best_max_' + C['label']))
        if not have_saved_brains or (gen_avg_reward > best_avg_brains.avg_fitness):
            save_brains(agent.brain_population, path='saved_brains', filename=('best_avg_' + C['label']))
        if not have_saved_brains or (gen_max_reward > best_max_brains.max_fitness):
            save_brains(agent.brain_population, path='saved_brains', filename=('best_max_' + C['label']))
        have_saved_brains = True


        #TODO Write the max & avg fitness out to a file here!
        generational_max_fitness.append(gen_max_reward)
        generational_avg_fitness.append(gen_avg_reward)
        fitness_stats = np.asarray(rewards)
        np.savetxt(C['save_dir']+f'fitness_gen{i}.csv', fitness_stats, delimiter=',')
        print(f'FINISHED Gen {i}:  Max Reward:{gen_max_reward}  Avg Reward:{gen_avg_reward}')
        print(f'BEST FITNESS TO DATE:{overall_best_fitness} in Gen {overall_best_fitness_gen}')
    plot_fitness(generational_max_fitness, generational_avg_fitness)

    final_gen_brains = agent.brain_population.population    # Henry: Retrieve final brains for next increment in learning
    del agent
    return final_gen_brains


overrides = [{'save_dir' : 'logs/A',
              'label' : 'A',
              'step_reward': 0,
              'Dist_multiplier' : 200,
              'Lava_penalty' : -2500,
              'Time_penalty' : -1800,
              'Time_multiplier' : -100,
              'num_generations': 10
              },
             {'save_dir': 'logs/B',
              'label': 'B',
              'XML' : "./scenarios/scenario2.xml",
              'step_reward': 0,
              'Dist_multiplier': 200,
              'Lava_penalty': -2500,
              'Time_penalty': -1800,
              'Time_multiplier': -100,
              'num_generations': 10
              }
             ]

run_incremental_training(C, overrides)

# override = {'save_dir' : "logs/A",
#             'label': 'A',
#             'step_reward' : 0,
#             'Dist_multiplier' : 200,
#             'Lava_penalty' : -2000,
#             'Time_penalty' : -1800,
#             'Time_multiplier' : 0,
#             'num_generations': 2}
# main(C, override)
#
# np.random.seed(101)
# override = {'save_dir' : "logs/B",
#             'label': 'B',
#             'step_reward' : 0,
#             'Dist_multiplier' : 300,
#             'Lava_penalty' : -2000,
#             'Time_penalty' : -1800,
#             'Time_multiplier' : -100}
# main(C, override)
#
# np.random.seed(101)
# override = {'save_dir' : "logs/C",
#             'label': 'C',
#             'step_reward' : 0,
#             'Dist_multiplier' : 200,
#             'Lava_penalty' : -2500,
#             'Time_penalty' : -1800,
#             'Time_multiplier' : -100}
# main(C, override)
#
# np.random.seed(101)
# override = {'save_dir' : "logs/D",
#             'label': 'D',
#             'step_reward' : 0,
#             'Dist_multiplier' : 200,
#             'Lava_penalty' : -2000,
#             'Time_penalty' : -1750,
#             'Time_multiplier' : -50}
# main(C, override)
#
# np.random.seed(101)
# override = {'save_dir' : "logs/E",
#             'label': 'E',
#             'step_reward' : 0,
#             'Dist_multiplier' : 200,
#             'Lava_penalty' : -2000,
#             'Time_penalty' : -2000,
#             'Time_multiplier' : 0}
# main(C, override)