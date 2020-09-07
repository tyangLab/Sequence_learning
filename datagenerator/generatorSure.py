#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:13:54 2019


@author: Zhewei Zhang

Reproduce fake data of the sure target task(Kiani, 2009))

in half of trials, it is the typical random dots task
in the other half, a sure target appear, the sure target lead to a small reward

V2: different duration of the random dots stimuli

1st-14th: about the visual stimili
    first input represents the fixation point
    second/third inputs denote the two choice targets
    the fourth one is the sure target
    5th-9th: are the neurons prefer the left direction
    10th-14th: are the neurons prefer the right direction
    15th: no visual stimuli

16th-20th: about the movement; fixation/left tar/right tar/sure target/break

21th-22th: reward/no reward

"""
import os
import time
import datetime
import numpy as np
import scipy.io as sio

# In[0]: hyperparameters

n_input = 22

# dots
num_neurons_dots = 10
rdmdots_dur_list = [1,2,3,4,6,8]
coherence_list = [-51.2, -25.6, -12.8, -6.4, -3.2, 0, 3.2, 6.4, 12.8, 25.6, 51.2]
# sure targets
sure_prop = 0.5
sure_reward = 0.5
# response and choice
std_con = 2.5
threshold = 7.5
linear_inc = 2.5

# In[0]: helper function

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def stimuli2resp(coherences, rdmdots_dur, numTrials):
    
    neuron_resp, acc_evi = [], []
    for nT in range(numTrials):
        resp = std_con*np.random.randn(num_neurons_dots, rdmdots_dur[nT])
        resp[:int(num_neurons_dots/2),:] += coherences[nT]/10
        resp[int(num_neurons_dots/2):,:] += -coherences[nT]/10
        
        evi_left  = resp[:int(num_neurons_dots/2),:].sum()
        evi_right = resp[int(num_neurons_dots/2):,:].sum()
            
        acc_evi.append(evi_left-evi_right)
        neuron_resp.append(resp)

    acc_evi = np.array(acc_evi)
    return neuron_resp, acc_evi


# In[1]: 
    
class generator_t:
    def __init__(self):
        """
    
        """
        self.setting = {'sure_propt':sure_prop, 'std_cont'   :std_con, 
                        'threshold' :threshold, 'sure_reward':sure_reward,
                        'coherence_list'  :coherence_list,
                        'num_neurons_dots':num_neurons_dots }
        np.random.seed(int(time.time()))
        
    def generate(self, numTrials = int(1e3), seed = []):
        '''
        Generate the choice according to bayesian inference
        
        '''
        if seed:
            np.random.seed(int(seed))

        coherences  = np.random.choice(coherence_list,  numTrials)
        rdmdots_dur = np.random.choice(rdmdots_dur_list,numTrials)
        sure_trials = np.random.rand(numTrials,) < sure_prop
        
        neuron_resp, acc_evi = stimuli2resp(coherences, rdmdots_dur, numTrials)
        # choice and reward 
        choices, rewards = [], []
        for nT in range(numTrials):
            choice = None
            if sure_trials[nT]==1:
                if acc_evi[nT] > threshold + linear_inc*(rdmdots_dur[nT]-1):
                    choice = 1
                elif acc_evi[nT] < -(threshold + (linear_inc*(rdmdots_dur[nT]-1))):
                    choice = 2
                else:
                    choice = 3
            else :
                choice = 1 if acc_evi[nT] > 0 else 2
                
            choices.append(choice)
        
            if choice==3:
                reward = sure_reward
            elif (coherences[nT]>0) == (choice == 1):
                reward = 1
            elif (coherences[nT]<0) == (choice == 1):
                reward = 0
            elif coherences[nT]==0 and choice!=3:
                reward = np.random.rand()>0.5
                
            rewards.append(reward)

        # rearrange
        data = []
        n_timepoint = 17 + np.max(rdmdots_dur)
        for nT in range(numTrials):
            inputs = np.zeros((n_input,n_timepoint)) # inputs*time points
            # sensory stimili
            inputs[0,1:9+rdmdots_dur[nT]] = 1 # fixation period
            inputs[1:3,3:11+rdmdots_dur[nT]] = 1 # two choice targets
            inputs[4:14,5:5+rdmdots_dur[nT]] = sigmoid(neuron_resp[nT])
            if sure_trials[nT]==1:
                inputs[3,5+rdmdots_dur[nT]+2:5+rdmdots_dur[nT]+6] = 1
            
            # action inputs
            inputs[15,2:10+rdmdots_dur[nT]] = 1
            choice_start, choice_end = 5+rdmdots_dur[nT]+5, 5+rdmdots_dur[nT]+8
            if choices[nT]==1:
                inputs[16, choice_start:choice_end] = 1
                inputs[1, 5+rdmdots_dur[nT]+6] = 1
            elif choices[nT]==2:
                inputs[17, choice_start:choice_end] = 1
                inputs[2, 5+rdmdots_dur[nT]+6] = 1
            elif choices[nT]==3:
                inputs[18, choice_start:choice_end] = 1
                inputs[3, 5+rdmdots_dur[nT]+6] = 1
            # reward inputs
            if rewards[nT]==1:
                inputs[20,5+rdmdots_dur[nT]+8:5+rdmdots_dur[nT]+10] = 1
            if rewards[nT]==sure_reward:
                inputs[20,5+rdmdots_dur[nT]+8:5+rdmdots_dur[nT]+10] = sure_reward
            
            inputs[14,:] = 1-inputs[0:4,:].sum(axis=0)
            inputs[14,inputs[14,:]!=1] = 0
            inputs[19,:] = 1-inputs[15:19,:].sum(axis = 0)
            inputs[21,:] = 1
            inputs[21,inputs[20,:]!=0] = 0
            
            data.append([inputs.T])
        
        # training_guide[0,:]: training or not; 
        #               [1,:]: first trianing time step
        #               [2,:]: last training time step
        training_guide = np.zeros((3, numTrials))
        training_guide[0,:] = [1 if reward!=0 else 0 for reward in rewards]
        training_guide[1,:] = 0
        training_guide[2,:] = 17 + rdmdots_dur
        training_guide = training_guide.astype(np.int).T.tolist()
        
        
        data_Brief = {'coherences' :coherences,  'rdmdots_dur':rdmdots_dur,
                      'sure_trials':sure_trials, 'acc_evi':acc_evi,
                      'choices':choices,'reward': rewards,
                      'training_guide':training_guide}
        
        return data, data_Brief

# In[]:        
class generator_v:
    def __init__(self):
        """
    
        """
        self.setting = {'sure_propt':sure_prop, 'std_cont'   :std_con, 
                        'threshold' :threshold, 'sure_reward':sure_reward,
                        'coherence_list'  :coherence_list,
                        'num_neurons_dots':num_neurons_dots }
        np.random.seed(int(time.time()))
    
    def generate(self, numTrials = int(1e3), seed = []):
        '''
        Generate validation dataset
        
        '''
        if seed:
            np.random.seed(int(seed))

        coherences  = np.random.choice(coherence_list,  numTrials)
        rdmdots_dur = np.random.choice(rdmdots_dur_list,numTrials)
        sure_trials = np.random.rand(numTrials,) < sure_prop
        
        neuron_resp, acc_evi = stimuli2resp(coherences, rdmdots_dur, numTrials)
        
        trial_length = 11 + rdmdots_dur
        data = []
        for nTrial in range(numTrials):
            inputs = np.zeros((n_input,trial_length[nTrial]))
            inputs[0,1:9+rdmdots_dur[nTrial]] = 1 # fixation period
            inputs[1:3,3:11+rdmdots_dur[nTrial]] = 1 # two choice targets
            inputs[4:14,5:5+rdmdots_dur[nTrial]] = sigmoid(neuron_resp[nTrial])
            
            if sure_trials[nTrial]==1:
                inputs[3,5+rdmdots_dur[nTrial]+2:5+rdmdots_dur[nTrial]+6] = 1
                
            inputs[15,2:10+rdmdots_dur[nTrial]] = 1
        
            inputs[14,:] = 1-inputs[0:4,:].sum(axis=0)
            inputs[14,inputs[14,:]!=1] = 0
            inputs[19,:] = 1-inputs[15:19,:].sum(axis = 0)
            inputs[21,:] = 1
            inputs[21,inputs[20,:]!=0] = 0
            data.append([inputs.T])

        
    
        data_Brief = {'coherences':coherences, 'acc_evi':acc_evi,
                     'sure_trials':sure_trials,'rdmdots_dur':rdmdots_dur}
        return data, data_Brief














