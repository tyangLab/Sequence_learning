#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017

@author: Huzi Cheng, Zhewei Zhang

configuration file for reaction time version of the shape task

"""
from generatorRT import generator_t, generator_v

settimg = {
        "name": "shape_rt",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "generator_t": generator_t(),
        "generator_v": generator_v(),
        "log_path"  : "../log/RT/",
        "model_path": "../save/RT/model-shape_rt_",
        "load_model": "",
        "train_num": 75e4,
        "input_size": 20,
        "batch_size": 128,
        "reset_hidden":True,
        "lr": 1e-3,
        "record": True,
        "train" : True,
        "validate": True,
        "lesion": "",
        "anlys_file":"",
    }

n_runs = 3
task_configurations = [settimg]*n_runs

# In[] ========================================================================
# for lesion simulation
import os
import glob
import copy
os.chdir('../holmes')
from toolkits_whenwhich import model_loading
from toolkits_whenwhich import when_which_constructer, datasaving
os.chdir('../seqrnn_multitask')

model_files = glob.glob('../save/RT/*.pt')
n_runs = len(model_files)
neuron_proportion = [0.5, 0.4, 0.3, 0.2, 0.1]

if n_runs != 0:
    # loop over each model
    for file_path in model_files:
        # load the .pt files
        ho_weight, w_hr, _, _ = model_loading(file_path)
        for prop in neuron_proportion:
            # using hidden-output connections to find out the when/which units
            when, which = when_which_constructer(ho_weight, prop)
            # save the identity of when/which model
            datasaving(file_path, when, which, w_hr, ho_weight, prop)

settimg_lesion = {
        "name": "shape_rt",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "generator_t": generator_t(),
        "generator_v": generator_v(),
        "log_path"  : "../log/RT/lesion/",
        "model_path": "../save/RT/lesion/model-shape_rt_",
        "load_model": "",
        "train_num": 0,
        "input_size": 20,
        "batch_size": 128,
        "reset_hidden":True,
        "lr": 1e-3,
        "record": True,
        "train" : False,
        "validate": True,
        "lesion": ['output', 'when_pos', 'when_neg', 'which_pos', 'which_neg'],
        "anlys_file":"",
    }

task_configurations_lesion = []
for prop in neuron_proportion:
    wh_files = glob.glob('../save/RT/WhenWhich_*threshold'+str(prop)+'*.yaml')
    for i in range(n_runs):
        task_configuration = copy.deepcopy(settimg_lesion)
        ## for each run
        if prop != 0.5:
            task_configuration["lesion"] = ['output', 'when_pos', 'when_neg']
        task_configuration["anlys_file"] = wh_files[i]
        task_configuration["load_model"] = model_files[i]
        
        task_configurations_lesion.append(task_configuration)


