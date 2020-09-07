#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:07:45 2019

@author: Zhewei Zhang

configuration file for simplfied two step task

"""
from generatorST import generator_t, generator_v
print('simple two')
setting = {
        "name": "smp_ts",
        "model_mode":'GRU',
        "optimizer":'Adam',        
        "generator_t": generator_t(),
        "generator_v": generator_v(),
        "log_path": "../log/ST/",
        "model_path": "../save/ST/model-sp_ts_",
        "load_model": "",
        "train_num":75e4,
        "input_size": 10,
        "batch_size": 1,
        "reset_hidden":False,
        "lr": 1e-4,
        "record": True,
        "train" : True,
        "validate": True,
        "lesion": "",
        "anlys_file":""
    }

n_runs = 3
task_configurations = [setting]*n_runs
