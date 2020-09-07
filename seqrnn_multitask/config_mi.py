#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:05:37 2019

@author: Zhewei Zhang 

configuration file for multisensory integration task
    
"""
from generatorMI import generator_t, generator_v

settimg = {
        "name": "mult_int",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "generator_t": generator_t(),
        "generator_v": generator_v(),
        "log_path": "../log/MI/",
        "model_path": "../save/MI/model-mult_int_",
        "load_model": "",
        "train_num": 75e4,
        "input_size": 26,
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
    