#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:12:59 2019

@author: Zhewei Zhang

cofiguration file for the sure task adapted from Kiani 2009

"""
from generatorSure import generator_t, generator_v

setting = {
        "name": "sure_task",
        "model_mode":'GRU',
        "optimizer":'Adam',
        "generator_t": generator_t(),
        "generator_v": generator_v(),
        "log_path": "../log/Sure/",
        "model_path": "../save/Sure/model-sure_",
        "load_model": "",
        "train_num": 75e4,
        "input_size": 22,
        "batch_size": 128,
        "reset_hidden":True,
        "lr": 1e-3,
        "record": True,
        "train" : True,
        "validate": True,
        "lesion": "",
        "anlys_file":""
    }

n_runs = 3
task_configurations = [setting]*n_runs
