""" Tim: modified form of Matthews worksheet_4 code
Dont forget to run the minecraft environment before trying to run this code...
Run this code from within the directory you have put this code in..
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

#Mitchell
from PIL import Image

it = 0


def rgb2gray(rgb):
    """ TIM: from https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



class Agent(object):
    newid = itertools.count()
    def __init__(self):
        self.count = 0 #Used to keep track of how many inputs it has taken
        self.host = MalmoPython.AgentHost()
        malmoutils.parse_command_line(self.host)
        self.index = next(Agent.newid) # first agent is 0, 2nd is 1...
        self.queued_commands = [] # this commands are sent the next time that act() is called

        #TIM: Matthew had a separate "Sunwatcher" agent inheriting from this Agent class. for simplicity I've just mergred his Sunwatcher code into this agent so we can call it later if we want to
        self.yaw = 0.0   # your current facing (yaw)
        self.pitch = 0.0 # your current facing (pitch)

        ## NOTE: You may want to add some code here.


    def connect(self, client_pool):
        self.client_pool = client_pool
        self.client_pool.add( MalmoPython.ClientInfo('127.0.0.1',10000+self.index) )

    def safeStartMission(self, mission, client_pool):
        self.recording_spec = MalmoPython.MissionRecordSpec()
        cs765_utils.safeStartMission(self.host, mission, client_pool, self.recording_spec, self.index, '' )

    def is_mission_running(self):
        return self.host.peekWorldState().is_mission_running
    
    def pixels_to_numpy(self, pixels, normalise=True):
        """ TIM convert malmo pixels to np array shape [h,w,c], c = rgbd channel respectively
        Optionally normalize values to between 0.0 & 1.0
        """
        arr = np.array(pixels,dtype=np.uint8).reshape(240,320,4).astype(np.float32)
        if normalise:
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
        img.save("image_log/"+str(self.count)+'.png')
        img = Image.fromarray(image[:,:,3])
        img.save("image_log/"+str(self.count)+'_d.png')
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
        img.save("image_log/"+str(self.count)+'.png')
        img = Image.fromarray(image[:,:,3])
        img.save("image_log/"+str(self.count)+'_d.png')
        self.count += 1
       
    def get_sensory_info(self, arr, s_height=60, s_width=80, s_fn=np.mean, 
                         flatten=True, grayscale=False, RGB_output=True, threshold = 0.3):
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
        if grayscale:
            grey = np.expand_dims(rgb2gray(arr), axis=2)
            arr = np.concatenate((grey, np.expand_dims(arr[:,:,3], axis=2)), axis=2)
        h,w,c = arr.shape
        downsample_h = h // s_height
        downsample_w = w // s_width
        sensory_info = skimage.measure.block_reduce(arr, (downsample_h, downsample_w, 1), s_fn)

        #Threshold values to distinguish parts of the scene - Mitchell
        if threshold != 0:
            sensory_info[:,:,0] = (sensory_info[:,:,0] > threshold)
            sensory_info[:,:,1] = (sensory_info[:,:,1] > threshold)
            sensory_info[:,:,2] = (sensory_info[:,:,2] > threshold)

        #Used for debugging, so we can see what the agent sees in a log
        if RGB_output:
            #If we provide RGB instead of the compressed information
            self.output_image_RGB(sensory_info)
        elif not RGB_output:
            self.output_image_3bit(sensory_info)
            
        if flatten:
            sensory_info = sensory_info.flatten()
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

    def run(self):
        """run the agent on the world"""
        total_reward = []
        current_reward = 0
        
        # wait for a valid observation
        world_state = self.host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = self.host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = self.host.peekWorldState()
        world_state = self.host.getWorldState()
        for err in world_state.errors:
            print(err)

        if not world_state.is_mission_running:
            return 0 # mission already ended

        assert len(world_state.video_frames) > 0, 'No video frames!?'


        #Here is the actual agent running
        start_time = time.time()

        #first action
        total_reward += [self.act(current_reward)]
        
        while world_state.is_mission_running:
            world_state = self.host.peekWorldState()
            time.sleep(0.1)
            current_reward = sum(r.getValue() for r in world_state.rewards)
            total_reward += [self.act(current_reward)]
            
        fitness = time.time() - start_time
        return total_reward

    
    def act(self,current_reward):
        actions = ['turn', "strafe", "move"]

        world_state = self.host.getWorldState()

        if len(world_state.video_frames) >= 1:
            pixels = world_state.video_frames[-1].pixels
            #red,green,blue,depth = self.pixels_as_arrays(pixels)
            #sv.display_data(blue)
            arr = self.pixels_to_numpy(pixels)
            sensory_info = self.get_sensory_info(arr)

            
            ## NOTE: Most of your code will go here.

            #action = our_algorithm(sensory_info)
            #self.queued_commands.append(action)

            self.host.sendCommand(random.choice(actions) + " " + str(random.random()*2-1))
            
        return 0



def main():
    agent = Agent() #Single agent
    num_iterations = 10

    #Decorators drawn between lives
    drawing_decorators = ''
    drawing_decorators += cuboid(-3,25,-2, 3,29,10, 'air') #Clear area
    #Ceiling
    drawing_decorators += cuboid(-3,25,-2, 3,25,10, 'emerald_block') #Green - Floor
    drawing_decorators += cuboid(-3,29,-2, 3,29,10, 'emerald_block') #Green - Ceiling
    #Walls
    drawing_decorators += cuboid(3,25,-2, 3,29,10, 'sea_lantern') #Blue - Wall
    drawing_decorators += cuboid(-3,25,-2, -3,29,10, 'sea_lantern') #Blue - Wall
    drawing_decorators += cuboid(-3,25,-2, 3,29,-2, 'sea_lantern') #Blue - Wall
    drawing_decorators += cuboid(-3,25,10, 3,29,10, 'sea_lantern') #Blue - Wall
    #Divider
    drawing_decorators += cuboid(-3,25,3, 3,29,3, 'sea_lantern') #Blue - Wall
    drawing_decorators += cuboid(-1,25,3, -1,28,3, 'air') #Blue - Wall
    drawing_decorators += cuboid(1,25,3, 1,28,3, 'air') #Blue - Wall
    #Threat
    drawing_decorators += cuboid(-1,25,3,1,25,3, 'lava') #Blue - Wall


    subset = {
        'DrawingDecorators' : drawing_decorators,
        'StartTime' : 6000, #Fixed start time (Midday) - Mitchell
    }

    mission_xml = cs765_utils.load_mission_xml('./desert_island.xml',subset)
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

    client_pool = MalmoPython.ClientPool()
    agent.connect(client_pool)

    #Using code from tabular_q_learning
    rewards = []
    for i in range(num_iterations):
        print("\n Map - Mission %d of %d:" % (i+1, num_iterations ))
        agent.safeStartMission( my_mission, client_pool)    

        print("Waiting for the mission to start", end=' ')
        cs765_utils.safeWaitForStart([agent.host])

        # -- run the agent in the world -- #
        reward = agent.run()

        total_reward = 0
        for r in reward:
            total_reward +=r
        print('Cumulative reward: %d' % total_reward)
        rewards += [total_reward]

        # -- clean up -- #
        time.sleep(0.5) # (let the Mod reset)


main()
