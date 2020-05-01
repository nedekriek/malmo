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


it = 0


def rgb2gray(rgb):
    """ TIM: from https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



class Agent(object):
    newid = itertools.count()
    def __init__(self):
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
        """ TIM convert malmo pixels to np array shape [h,w,c], c = rgbd respectively
        Optionally normaalize values to between 0.0 & 1.0
        """
        arr = np.array(pixels,dtype=np.uint8).reshape(240,320,4).astype(np.float32)
        if normalise:
            arr = arr / 255.0
        return arr
       
    def get_sensory_info(self, arr, s_height=4, s_width=4, s_fn=np.mean, 
                         flatten=True, grayscale=False):
        """ TIM Convert raw numpy array into sensory information to be input 
                into our agent's algorithm
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

    def angle_to_time(self,angle,part_of_day='unspecified'):
        """ TIM: From Matthew's SunWatcher class 
        Takes the angle of the celestial object in the sky 
        (at horizon = 0, at the top of the sky = 90), and estimates 
        the time of day. 

        Without knowing if the object is the sun or moon, and without
        knowing if it is rising or setting our estimate can only be so
        accurate.

        If the part_of_day argument is specified, then
        the estimate can be more accurate. For this argument a numerical 
        value is given that specifies the following.

        0: between     0 and  6000 in minecraft time (dawn--midday)
        1: between  6000 and 12000 in minecraft time (midday--dusk)
        2: between 12000 and 18000 in minecraft time (dusk--midnight)
        3: between 18000 and 24000 in minecraft time (midnight--dawn)
        """
        
        estimate = 6000.0*(-self.pitch / 90.0)
        if part_of_day == 'unspecified' :
            print('Time (%f) is estimated to be one of the following:' %(self.correct_time))
            print('\t %f' %(estimate))
            print('\t %f' %(12000-estimate))
            print('\t %f' %(12000+estimate))
            print('\t %f' %(12000+(12000-estimate)))
        else :
            print('Time (%f) is estimated to be:' %(self.correct_time))
            print('\t %f %s' %(estimate,'<---' if part_of_day==0 else ''))
            print('\t %f %s' %(12000-estimate,'<---' if part_of_day==1 else ''))
            print('\t %f %s' %(12000+estimate,'<---' if part_of_day==2 else ''))
            print('\t %f %s' %(12000+(12000-estimate),'<---' if part_of_day==3 else ''))


    
    def act(self):
        if self.index==0:
            global it
            it += 1
            
        ## when you override this method in a subclass, make sure to
        ## call the parent's class as I did below with `super().act()`
        ## so that queued commands are sent.
        for command in self.queued_commands:
            self.host.sendCommand(command)
        self.queued_commands.clear()

        world_state = self.host.getWorldState()
        
        #Tim commented out Matthew code relating to time of day
        ## gets current positional data and stores it in yaw and pitch
        #if len(world_state.observations) > 0 :
        #    obvsText = world_state.observations[-1].text
        #    data = json.loads(obvsText) 
        #    self.yaw = data.get(u'Yaw', 0)
        #    self.yaw += 360.0*2
        #    self.yaw = self.yaw % 360.0
            # force yaw to lie between 0 and 360.
        #    self.pitch = data.get(u'Pitch', 0)
        #    self.correct_time = data.get(u'WorldTime',0)
        #    self.correct_time = self.correct_time % 24000.0
            # force correct time to lie between 0 and 24000.
            
        if len(world_state.video_frames) >= 1:
            pixels = world_state.video_frames[-1].pixels
            #red,green,blue,depth = self.pixels_as_arrays(pixels)
            #sv.display_data(blue)
            arr = self.pixels_to_numpy(pixels)
            sensory_info = self.get_sensory_info(arr)


            ## NOTE: Most of your code will go here.

            # action = our_algorithm(sensory_info)
            #self.queued_commands.append(action)

            self.queued_commands.append('move 1')
            self.queued_commands.append('turn 1')

        
            
########################################
#### INITIALISE One AGENT
## the order here corresponds to the order of the AgentSections
#agents = [SunWatcher()]  #TIM: Matthew code. His comments said initialising 2 agents 
                          # but when I added a second one I got an error 
                          # which might relate to the environment itself not being set up for >1 agent?
agents = [Agent()]

drawing_decorators = ''
drawing_decorators += cuboid(-5,10,-5,5,25,5, 'sand')
drawing_decorators += cuboid(-5,10,-5,-5,25,-5, 'water')
drawing_decorators += cuboid(-5,10,5,-5,25,5, 'water')
drawing_decorators += cuboid(5,10,5,5,25,5, 'water')
drawing_decorators += cuboid(5,10,-5,5,25,-5, 'water')
subst = {
    'DrawingDecorators' : drawing_decorators,
    'StartTime' : str(np.random.randint(24000)),
}


mission_xml = cs765_utils.load_mission_xml('./desert_island.xml',subst)
print(mission_xml)
my_mission = MalmoPython.MissionSpec(mission_xml, True)

client_pool = MalmoPython.ClientPool()
for agent in agents:
    agent.connect(client_pool)

for agent in agents:
    agent.safeStartMission(my_mission, client_pool)

cs765_utils.safeWaitForStart([agent.host for agent in agents])# agent_a.host, agent_b.host ])

# TIM Add counter to stop agents running
counter = 100

## Main loop
while any([agent.is_mission_running() for agent in agents]):
    time.sleep(0.1)
    for agent in agents:
        agent.act()
    counter -= 1   # TIM added
    if counter == 0:
        print('Counter ran out, breaking..')
        break

agents[0].host.sendCommand('turn 0')
#agents[0].host.sendCommand('move 1')

#TIM Added for testing sensory info extractor..:
world_state = agents[0].host.getWorldState()
world_state.is_mission_running   #True
print(len(world_state.video_frames))   #1
pixels = world_state.video_frames[-1].pixels
arr = np.array(pixels,dtype=np.uint8).reshape(240,320,4)
print(arr.shape)   #(240, 320, 4)
tst = agents[0].get_sensory_info(arr)
print(tst.shape)  # (64,)
tst = agents[0].get_sensory_info(arr,flatten=True)
print(tst.shape)
tst = agents[0].get_sensory_info(arr,flatten=False)
print(tst.shape)
tst = agents[0].get_sensory_info(arr,grayscale=True)
print(tst.shape)
tst = agents[0].get_sensory_info(arr,flatten=False, grayscale=True)
print(tst.shape)
tst = agents[0].get_sensory_info(arr, s_height=240, s_width=320)
print(tst.shape)  #(307200,)
tst = agents[0].get_sensory_info(arr, s_height=240, s_width=320, grayscale=True)
print(tst.shape)  #153600



'''
# Tim: Matthew's Sunwatcher code in case we need it:

sv = SensoryVisualization()
class SunWatcher(Agent):
    def __init__(self):
        super().__init__()
        self.yaw = 0.0   # your current facing (yaw)
        self.pitch = 0.0 # your current facing (pitch)

        ## NOTE: You may want to add some code here.

    def angle_to_time(self,angle,part_of_day='unspecified'):
        """ Takes the angle of the celestial object in the sky 
        (at horizon = 0, at the top of the sky = 90), and estimates 
        the time of day. 

        Without knowing if the object is the sun or moon, and without
        knowing if it is rising or setting our estimate can only be so
        accurate.

        If the part_of_day argument is specified, then
        the estimate can be more accurate. For this argument a numerical 
        value is given that specifies the following.

        0: between     0 and  6000 in minecraft time (dawn--midday)
        1: between  6000 and 12000 in minecraft time (midday--dusk)
        2: between 12000 and 18000 in minecraft time (dusk--midnight)
        3: between 18000 and 24000 in minecraft time (midnight--dawn)
        """
        
        estimate = 6000.0*(-self.pitch / 90.0)
        if part_of_day == 'unspecified' :
            print('Time (%f) is estimated to be one of the following:' %(self.correct_time))
            print('\t %f' %(estimate))
            print('\t %f' %(12000-estimate))
            print('\t %f' %(12000+estimate))
            print('\t %f' %(12000+(12000-estimate)))
        else :
            print('Time (%f) is estimated to be:' %(self.correct_time))
            print('\t %f %s' %(estimate,'<---' if part_of_day==0 else ''))
            print('\t %f %s' %(12000-estimate,'<---' if part_of_day==1 else ''))
            print('\t %f %s' %(12000+estimate,'<---' if part_of_day==2 else ''))
            print('\t %f %s' %(12000+(12000-estimate),'<---' if part_of_day==3 else ''))

        
    def act(self):
        super().act()
        world_state = self.host.getWorldState()
        
        ## gets current positional data and stores it in yaw and pitch
        if len(world_state.observations) > 0 :
            obvsText = world_state.observations[-1].text
            data = json.loads(obvsText) 
            self.yaw = data.get(u'Yaw', 0)
            self.yaw += 360.0*2
            self.yaw = self.yaw % 360.0
            # force yaw to lie between 0 and 360.
            self.pitch = data.get(u'Pitch', 0)
            self.correct_time = data.get(u'WorldTime',0)
            self.correct_time = self.correct_time % 24000.0
            # force correct time to lie between 0 and 24000.
            
        if len(world_state.video_frames) >= 1:
            pixels = world_state.video_frames[-1].pixels
            red,green,blue,depth = self.pixels_as_arrays(pixels)
            #sv.display_data(blue)

            ## NOTE: Most of your code will go here.
'''

