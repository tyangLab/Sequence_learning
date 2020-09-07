#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:09:46 2019

@author: Zhewei Zhang
"""
import copy
import time
import numpy as np

trans_prob  = 0.8 # A1-B1, A2-B2
reward_prob = 0.8 # from B1/B2 to reward
block_size = 50
Double = True

Double = True
n_input = 10;
trial_length = 13
shape_Dur = 3; # period for shape presentation
choice_Dur = 2; #period for shape interval

# In[]

def get_block(nth, numTrials, block_size):
    # nth is a positive number
    # 0: B1 with large reward prob, 1:b2 with large reward prob
    assert nth != 0
    nth = nth%(block_size*2) 
    nth = nth if nth !=0 else block_size*2
    
    numTrials_t = numTrials + nth - 1 
    unit = np.hstack((np.ones((block_size,)), np.zeros((block_size,)) ))
    blocks = np.tile(unit, int(numTrials_t/(block_size*2))) 
    # 
    rest_num = numTrials_t - blocks.size
    if rest_num <= block_size:
        rest = np.ones(rest_num,)
    else:
        rest = np.hstack((np.ones(block_size,), np.zeros(rest_num - block_size,)))
    blocks = np.hstack((blocks, rest))
    blocks = blocks[nth-1:]
    return blocks

# In[]
class generator_t:
    """
    first seven inputs represent visual stimulus 
    
    1st~2nd inputs representing the options
    
    3rd~4th inputs representing the intermeidate outcome
    
    6th~8th inputs representing the movement/choice
    
    9th~10th inpust denotes the reward states
    """

    def __init__(self):
        """
    
        """
        self.nth = 0
        self.prv_input = []
        self.setting = {'block_size':block_size,  'reward_prob':reward_prob,
                        'trans_prob':trans_prob,  'Double'     :Double,
                        'shape_Dur' :shape_Dur,   'choice_Dur' :choice_Dur}
        np.random.seed(int(time.time()))
        
    def generate(self, numTrials = int(1e3), seed = []):
        '''
        Generate the shapes and choice according to DDM
        
        Then, arrange the data into time sequence
        fixation is one time step later than fixation point appear
        fixation stop is one time step later than fixation point disappear
        Time point: 1: fixation point appears; 2: acquire fixation 3 target on;
            4:5:5*shape_num-1:shape on; 7:5:5*shape_num+2 shape off; 
        '''
        
        if seed:
            np.random.seed(int(seed))
                
        blocks  = get_block(self.nth+1, numTrials, block_size)
        choices = np.random.choice([0,1], numTrials) # 0:A1; 1:A2
        
        # probability of transition to the B2 in stage 2
        trans_probs = trans_prob*choices + (1-trans_prob)*(1-choices) 
        
        # 0: B1; 1: B2
        stage2 = trans_probs>np.random.rand(numTrials,) 
        
        # reward probability of the observation in stage 2 
        reward_prob_B1 = (1-reward_prob)*stage2*(1-blocks) + reward_prob*(1-stage2)*(1-blocks)
        reward_prob_B2 = reward_prob*stage2*blocks + (1-reward_prob)*(1-stage2)*blocks 
        reward_prob_B = reward_prob_B1 + reward_prob_B2
        # reward feedback
        rewards = reward_prob_B > np.random.rand(numTrials,)

        stage2 = stage2+1 # 1: B1; 2: B2

        # inputs of the network
        data= []
        for nTrial in range(numTrials):
            inputs = np.zeros((n_input,trial_length))
            # the three-five time points representing the first epoch
            inputs[0:2,2:5] = 1 
            
            if choices[nTrial]==0:
                inputs[6,5:7] = 1
            elif choices[nTrial]==1:
                inputs[7,5:7] = 1
            
            if stage2[nTrial]==1:
                inputs[2,7:10] = 1
            elif stage2[nTrial]==2:
                inputs[3,7:10] = 1
            
            if rewards[nTrial]==1:
                inputs[8,10:12] = 1
            
            inputs[4,:] = inputs[0:4,:].sum(axis = 0)
            inputs[4,np.where(inputs[4,:]!=0)] = 1
            inputs[4,:] = 1 - inputs[4,:]
            inputs[5,:] = 1 - inputs[6:8,:].sum(axis = 0)
            inputs[9,:] = 1 - inputs[8,:]
            
            if Double:
                if self.nth == 0:
                    prv_input = np.zeros((n_input,trial_length))
                    prv_input[[4,5,9],:] = 1
                else:
                    prv_input = self.prv_input
                data.append([np.hstack((prv_input, inputs)).T])
                self.prv_input = copy.deepcopy(inputs)
            else:
                data.append([inputs.T])
        
        training_guide = np.zeros((3, numTrials))
        training_guide[0,:] = rewards # training or not
        if Double:
            training_guide[1,:] = trial_length  # first trianing time step
            training_guide[2,:] = 2*trial_length # last training time step
        else:
            training_guide[1,:] = 0
            training_guide[2,:] = trial_length
        training_guide = training_guide.astype(np.int).T.tolist()
            
        data_Brief = {'block' :blocks, 'choices':choices,
                      'stage2':stage2, 'reward' :rewards, 
                      'training_guide':training_guide}
        
        self.nth += numTrials
        return data, data_Brief

    def reset(self):
        self.nth = 0
        self.prv_input = []
    
# In[]    
class generator_v:
    def __init__(self):
        """
    
        """
        self.nth = 0
        self.setting = {'block_size':block_size,  'reward_prob':reward_prob,
                        'trans_prob':trans_prob,  'Double'     :Double,
                        'shape_Dur' :shape_Dur,   'choice_Dur' :choice_Dur}
        np.random.seed(int(time.time()))
        
    def generate(self, numTrials = int(1e3), seed = []):
        '''
        Generate validation dataset
        
        '''
        if seed:
            np.random.seed(int(seed))
        
        blocks = get_block(self.nth+1, numTrials, block_size)
        
        trans_probs  = trans_prob*np.ones(numTrials,)
        reward_probs = reward_prob*blocks + (1-reward_prob)*(1-blocks)
        
        inputs = [
                [0., 0., 1., 1., 1., 0.],
                [0., 0., 1., 1., 1., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 0.]
                ]
        
        data = [[np.array(inputs).T]]*numTrials

        data_Brief = {'rwd_probs':reward_probs, 'tran_probs':trans_probs,
                      'block_size':block_size, 'block':blocks}
        
        self.nth += numTrials
        
        return data, data_Brief

    def reset(self):
        self.nth = 0












