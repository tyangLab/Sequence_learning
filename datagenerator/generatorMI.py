#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:41:47 2019

@author: Zhewei Zhang


Reproduce fake data of the multisensory integration task

1st-12th: about the visual stimili
    first input represents the fixation point
    second/third inputs denote the two choice targets
    4th-11th: are the neurons with different perfer orientation for modality 1
    12th-19th: are the neurons with different perfer orientation for modality 1
    20th: no visual stimuli

21th-24th: about the movement; fixation/left tar/right tar/break

25th-26th: reward/no reward


The difference from version 1 is the way of simulating the neural response; AND 
    add the training guide for variable trial length


"""

import time
import numpy as np
import scipy.stats as sts

# In[0]: hyperparameters
n_input = 26


s_vi=90
s_ve=90
g_vi=300
g_ve=300
NumNeurons = 8
tuning_curve_disc=np.linspace(0, 180, NumNeurons, endpoint = True)


NumTimes = 5 # the maximum value of NumTimes is 5
per_ori_set = tuning_curve_disc

NumStimuli = 180
direction_range = [0, 180]


# all possible motion direction
dir_set = np.linspace(direction_range[0], direction_range[1], 
                      NumStimuli, endpoint = True)

# In[0]: tools

def sigmoid(x):
  return 1 / (1 + np.exp(1-x))

#def sigmoid(x):
#  return 1 / (1 + np.exp(-x))

def norm_pdf(x, mean, std):
    return np.exp(-(x-mean)**2/(2*std**2))/(std*np.sqrt(2*np.pi))

def rates_2(s):
    vi_rates=g_vi*norm_pdf(s,tuning_curve_disc,s_vi)
    ve_rates=g_ve*norm_pdf(s,tuning_curve_disc,s_ve)
    return vi_rates, ve_rates


def rates(s):
    vi_rates=g_vi*sts.norm.pdf(s,tuning_curve_disc,s_vi)
    ve_rates=g_ve*sts.norm.pdf(s,tuning_curve_disc,s_ve)
    return vi_rates, ve_rates

def counts(s):
#    vi_rates,ve_rates=rates(s)  
    vi_rates,ve_rates=rates_2(s)  # faster
    vi_sp=np.random.poisson(vi_rates)
    ve_sp=np.random.poisson(ve_rates)
    return vi_sp,ve_sp


# generate the rates_list for shorter running time
rates_list = []
post_len=180
ss=np.linspace(0,180,post_len)
for sc in range(post_len):
    s=ss[sc]
    rates_list.append(rates_2(s))

def get_posterior(r_counts, post_len=180):
        
#    ss=np.linspace(0,180,post_len) 
    log_posterior_com=np.zeros(post_len)
    log_posterior_ve=np.zeros(post_len)
    log_posterior_vi=np.zeros(post_len)

    for sc in range(post_len):
#        s=ss[sc]
        rates_s = rates_list[sc]
#        rates_s = rates(s) # slower, but more general 
        log_posterior_vi[sc]=np.sum(-rates_s[0]+r_counts[0]*np.log(rates_s[0]))
        log_posterior_ve[sc]=np.sum(-rates_s[1]+r_counts[1]*np.log(rates_s[1]))
    log_posterior_com = log_posterior_vi+log_posterior_ve

    log_p = [log_posterior_vi,log_posterior_ve, log_posterior_com]
    for i, k in enumerate(log_p):
        k=np.exp(k)/sum(np.exp(k))
        log_p[i]=k
        
    estimates=[]
    for k in enumerate(log_p):
        estimate = k[1][:int(len(k[1])/2)].sum() > k[1][int(len(k[1])/2):].sum()
        estimates.append(estimate)
    return log_p, estimates

def package(resp, signal = True):
    baseline = 0.5
    if signal:
        return sigmoid(resp)
    else:
        return baseline + np.zeros(resp.shape)

def stimuli2resp(modality, direction, numTrials):
    resp = np.zeros((numTrials, NumNeurons*2, NumTimes))
    estimates = np.zeros((numTrials, 3))
    #posteriors = []
    for nTrial in range(numTrials):
        r_counts_vi, r_counts_ve = [], []
        for nTime in range(NumTimes):
            r_count_vi, r_count_ve = counts(direction[nTrial])
            r_counts_vi.append(r_count_vi)
            r_counts_ve.append(r_count_ve)
            if modality[nTrial]==0:
                resp[nTrial,:NumNeurons,nTime] = package(r_count_vi)
                resp[nTrial,NumNeurons:,nTime] = package(r_count_ve, signal = False)
            elif modality[nTrial]==1:
                resp[nTrial,:NumNeurons,nTime] = package(r_count_vi, signal = False)
                resp[nTrial,NumNeurons:,nTime] = package(r_count_ve)
            elif modality[nTrial]==2:
                resp[nTrial,:NumNeurons,nTime] = package(r_count_vi)
                resp[nTrial,NumNeurons:,nTime] = package(r_count_ve)
        r_counts = [r_counts_vi, r_counts_ve]
        _, estimate = get_posterior(r_counts)
        estimates[nTrial,:] = estimate
    return resp, estimates
    
# In[]
class generator_t:
    def __init__(self):
        """
    
        """
        self.trial_length = 17
        self.setting = {'NumStimuli':NumStimuli, 'NumNeurons':NumNeurons,
                        'direction_range':direction_range, 'per_ori_set':per_ori_set,
                        's_vi':s_vi,'s_ve':s_ve,'g_vi':g_vi,'g_ve':g_ve}
        np.random.seed(int(time.time()))
        
    def generate(self, numTrials = int(1e3), seed = []):
        '''
        Generate the choice according to bayesian inference
        
        '''
        if seed:
            np.random.seed(int(seed))

        # parameters
        # visual or vestibular
        modalities = np.random.choice([0,1], numTrials)
        directions = np.random.choice(dir_set, numTrials)
        
        # neural response and bayesian inference
        resp, estimates = stimuli2resp(modalities, directions, numTrials)    

        rewards, choices, = [], []
        for nTrial in range(numTrials):
            choices.append(estimates[nTrial, modalities[nTrial]])
            if directions[nTrial] < 90:
                reward = 1 if choices[-1]==1 else 0
            elif directions[nTrial] > 90:
                reward = 1 if choices[-1]==0 else 0
            else:
                reward = 1 if np.random.rand()>0.5 else 0
            rewards.append(reward)
            
        data = []
        for nTrial in range(numTrials):
            inputs = np.zeros((n_input,self.trial_length))
            # visual inputs
            inputs[0,1:10] = 1 
            inputs[1:3,3:12] = 1 
            inputs[3:19,5:5+NumTimes] = resp[nTrial,:,:]
            
            inputs[20,1:11] = 1
            # choice and reward feednacl
                
            if choices[nTrial]==0:
                inputs[21,11:14] = 1
                inputs[1,12] = 1
            elif choices[nTrial]==1:
                inputs[22,11:14] = 1
                inputs[2,12] = 1
                
            if rewards[nTrial]==1:
                inputs[24,14:16] = 1
            if rewards[nTrial]==0:
                inputs[25,14:16] = 1
            
            inputs[19,:] = 1-inputs[0:3,:].sum(axis=0)
            inputs[19,inputs[19,:]!=1] = 0
            inputs[23,:] = 1-inputs[20:23,:].sum(axis = 0)
            inputs[25,:] = 1-inputs[24,:]
            data.append([inputs.T])

        # training_guide[0,:]: training or not; 
        #               [1,:]: first trianing time step
        #               [2,:]: last training time step
        training_guide = np.zeros((3, numTrials))
        training_guide[0,:] = rewards
        training_guide[1,:] = 0
        training_guide[2,:] = self.trial_length
        training_guide = training_guide.astype(np.int).T.tolist()

        data_Brief = {'direction':directions, 'modality':modalities,
                      'choice':choices, 'reward':rewards, 
                      'estimate':estimates, 'training_guide':training_guide}
        
        return data, data_Brief
    
# In[]
class generator_v:
    def __init__(self):
        """
    
        """
        
        self.trial_length = 13
        self.setting = {'NumStimuli':NumStimuli, 'NumNeurons':NumNeurons,
                        'direction_range':direction_range, 'per_ori_set':per_ori_set,
                        's_vi':s_vi,'s_ve':s_ve,'g_vi':g_vi,'g_ve':g_ve}
        np.random.seed(int(time.time()))
        
    def generate(self, numTrials = int(1e3), seed = []):
        '''
        Generate validation dataset
        
        '''
        if seed:
            np.random.seed(int(seed))

        modalities = np.random.choice([0, 1, 2], numTrials)
        directions = np.random.choice(dir_set, numTrials)
        
        # neural response and bayesian inference
        resp, estimates = stimuli2resp(modalities, directions, numTrials)
        
        data = []
        for nTrial in range(numTrials):
            inputs = np.zeros((n_input, self.trial_length))
            # sensory stimuli
            inputs[0,1:10] = 1 # fixation period
            inputs[1:3,3:12] = 1 # 
            inputs[3:19,5:5+NumTimes] = resp[nTrial,:,:]
            
            inputs[20,1:11] = 1
            
            inputs[19,:] = 1-inputs[0:3,:].sum(axis=0)
            inputs[19,inputs[19,:]!=1] = 0
            inputs[23,:] = 1-inputs[20:23,:].sum(axis = 0)
            inputs[25,:] = 1-inputs[24,:]
            data.append([inputs.T])  
    
        data_Brief = {'direction':directions, 'modality':modalities, 'estimate':estimates}
    
        return data, data_Brief