#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:26:41 2019

@author: Zhewei Zhang

"""

from __future__ import division

import sys
import os
import time
import copy
import argparse
import importlib
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import data_util
from model import GRUNetwork
from lesion import lesion_rnn
from evaluator import SeqAgent, Calculator, interactive_sample, match_rate

"""
Main script of training and validating RNN with GRU.
"""
training_interval = 12800
t_interval = 1000
save_interval = 25000
clip_setting = 0.25

def train(model, calculator, trial_data, hidden, tmp_reward, clip=clip_setting, bsz = None):
    """
    Training

    trial_data -> state -> encode -> rnn-> output-> decode -> state
    trial_data.shape: (?,20)
    """
    if bsz is None:
        bsz = len(tmp_reward)
        
    model.zero_grad()  # reset the grad
    pred_trial, total_loss, hidden = calculator.get_output(model, trial_data, hidden, tmp_reward, bsz = bsz)
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # gradient clip

    # apply gradient descent
    model.optimizer.step()

    # correct rate
    cr = []
    for i in range(bsz):
        rwd = tmp_reward[i]
#        if not rwd[0]:
#            continue
        cr.append(match_rate(trial_data[rwd[1]+1:rwd[2], i, :], 
                             pred_trial[rwd[1]:rwd[2]-1, i, :])
                 )
    
    correct_rate = np.array(cr).mean()
    
    return  total_loss.item(), correct_rate, hidden

def model_save(model, save_path, model_name):
    if not os.path.isfile(save_path + model_name):
        torch.save(model.state_dict(), save_path + model_name)
    else:
        n = 0
        while 1: # save the model
            n += 1
            if not os.path.isfile(save_path + model_name.split('.')[0] + '-' + str(n) + '.pt'):
                torch.save(model.state_dict(), save_path + model_name.split('.')[0] + '-' + str(n) + '.pt')
                break
    print("_" * 36)
    print("model saved")

def repackage_hidden(h):
    """Wrap hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def validate_stage(trial, trial_settings, task_env, sqa):
    
    
    task_env.configure(trial, trial_settings)
    
    raw_rec, tmp_loss, correct_rate = interactive_sample(sqa, task_env)  
    trial_abstract = task_env.extract_trial_abstractb64()

    wining = task_env.is_winning()
    completed = task_env.is_completed()
    task_env.reset_configuration()

    return wining, completed, tmp_loss, correct_rate, trial_abstract, raw_rec

def validate(dp, model, sqa, generator_v, task, record, truncate=1e800, to_console=True,
             cuda_enabled=False, suffix = ''):
    
    wining_counts, completed_counts, trial_counts = 0, 0, 0
    
    v_dataset, v_conditions = generator_v.generate(truncate)
    v_helper     = task.ValidationHelper(v_dataset, v_conditions)
    v_conditions = v_helper.validation_conditions

    global_low = 0
    record_saved = 0
    total_loss_v, correct_rate_v = 0, 0
    try:
        neuron_shape = list(map(lambda x: int(x), list(model.init_hidden(bsz = 1).data.shape))) # bsz is equal to 1 for validation, for GRU
#        neuron_shape = list(map(lambda x: int(x), list(model.init_hidden(bsz = 1)[0].data.shape))) # for LSTM
        behavior_shape = list(v_dataset[0][0].shape)
        behavior_shape.pop(0)
        if record and (not record_saved):
            print("creating raw records...")
            dp.create_hdf5raw(behavior_shape, neuron_shape, suffix = suffix)
        
        task_env = task.Task()
        for step, trial in enumerate(v_dataset):
            if step >= truncate:
                break
            if reset_hidden:
                sqa.init_hidden()

            trial_settings = v_conditions[step]
            wining, completed, loss, correct_rate, trial_abstract, raw_rec = validate_stage(trial, trial_settings, task_env, sqa)
            
            trial_counts = trial_counts + 1
            wining_counts = wining_counts + wining
            completed_counts = completed_counts + completed

            task_env.reset_configuration()

            total_loss_v   += loss
            correct_rate_v +=  correct_rate

            if record:
                # Notice: all these arrays is np.ndarray
                behavior_data = raw_rec["sensory_sequence"]
                prediction_data = raw_rec["predicted_trial"]
                raw_output = raw_rec["raw_records"]
                neuron_data = raw_rec["hidden_records"]
                tmp_high = global_low + behavior_data.shape[0]

                index_data = np.array([global_low, tmp_high]).reshape(1, 1, 2)
                dp.append_hdf5raw(behavior_data, prediction_data, raw_output,
                                  neuron_data, index_data)

                # save experiment result
                global_low = tmp_high
                dp.write_log(step, correct_rate, loss, training=0)
                dp.write_validation_brief(step, trial_abstract)
            
            if (step + 1) % t_interval == 0 and step > 0 and to_console:
                print("(Validate)STEP: {:6d} | AVERAGE CORRECT RATE: {:6f} | AVERAGE LOSS(without loss): {:.6f} ".format(
                    step + 1,
                    correct_rate_v / t_interval,
                    total_loss_v / t_interval
                ))
                total_loss_v, correct_rate_v = 0, 0
                
    except KeyboardInterrupt:
        print("KeyboardInterrupt Detected!\n")
        if record and (not record_saved):
            print("Saving validating set...")
            record_saved = dp.close_hdf5raw()

    if record and (not record_saved):
        record_saved = dp.close_hdf5raw()

    wining_rate = wining_counts / trial_counts
    completed_rate = completed_counts / trial_counts
    return wining_rate, completed_rate


def train_stage(dp, model, sqa, save_path, generator_t, generator_v, log_path,
                task, record=1, train_truncate=int(1e8), batch_size=1, 
                clip=clip_setting, cuda_enabled=False):
    
    calculator = Calculator(batch_size = batch_size, cuda_enabled=cuda_enabled)
    
    total_loss, correct_rate = 0, 0
    try:
        hidden = model.init_hidden(batch_size)
        for step in range(int(train_truncate)):
            
            if step >= train_truncate-1:
                model_name = str(step+1) + '-' + dp.get_logname() + ".pt"
                model_save(model, save_path, model_name)
                break
            
            if (step+1) % batch_size == 0:
                if reset_hidden:
                    hidden = model.init_hidden(batch_size)
                # generate the training dataset
                t_dataset, t_conditions = generator_t.generate(numTrials=batch_size)
                # trials: time step by batch size by input number
                if batch_size == 1:
                    trials = np.array(t_dataset)[0,:,:,:].transpose([1,0,2])
                else:
                    trials = np.array(t_dataset).squeeze().transpose([1,0,2])
                    
                reward = np.array(t_conditions["training_guide"])
                # training
                loss, cr, hidden = train(model, calculator, trials, hidden, reward, clip=clip)
            else:
                continue
            
            # necessary for the task keep the hidden unreset
            hidden = repackage_hidden(hidden) 
            total_loss   += loss
            correct_rate += cr

            # saving loss and correct rate during training
            if record:
                dp.write_log(step, cr, loss, training=1)
            
            if (step + 1) % training_interval == 0 and step > 0:
                # print training loss
                print("STEP: {:6d} | AVERAGE CORRECT RATE: {:6f} | AVERAGE LOSS: {:.6f} ".format(
                        step + 1,
                        correct_rate / training_interval * batch_size,
                        total_loss / training_interval * batch_size
                        ))

                total_loss, correct_rate = 0, 0
                
                # online validation
                set_size = 100
                rwd_rate, cpt_rate = validate(dp, model, sqa, generator_v, task,
                                              record=0, truncate=set_size, 
                                              to_console=False)
                print("(Test)WINING RATE:{:6f} | COMPLETED RATE:{:6f} ON {} SAMPLES".format(
                                rwd_rate, cpt_rate, set_size))
            
            # save models
            if (step+1) % save_interval == 0 and step > 0 and record:
                model_name = str(step+1) + '-' + dp.get_logname() + ".pt"
                model_save(model, save_path, model_name)

    except KeyboardInterrupt:
        if record:
            model_save(model, save_path, model_name)
        print("KeyboardInterrupt Detected!\n Now save model and exit the script......")
        sys.exit(0)

def main(task_configurations, args):

    cuda_enabled = args.cuda

    model_parameters = {
        "nhid": 128,
        "nlayers": 1,
        "clip": clip_setting,
        "lr": task_configurations[0]["lr"],
        "input_size": task_configurations[0]["input_size"],
        "batch_size": task_configurations[0]["batch_size"],
        }
    print("Model Parameters: {}".format(model_parameters))
    
    for nth, task_setting in enumerate(task_configurations):
        # model parameters
        model_parameters["lr"] = task_setting["lr"]
        if task_setting['model_mode'] == 'GRU':
            rnn_model = GRUNetwork(model_parameters["input_size"],
                               model_parameters["nhid"],
                               model_parameters["batch_size"],
                               model_parameters["nlayers"],
                               model_parameters["lr"],
                               cuda_enabled  = cuda_enabled,
                               )
        else:
            raise Exception('unknown model mode')
            
        # load model
        if len(task_setting["load_model"]) > 0:
            rnn_model.load_state_dict(torch.load(task_setting["load_model"]))
        print("{}th repeat:".format(nth+1))
        print(rnn_model)
        
        # inactivate output connection if necessary
        rnn_model = lesion_rnn(rnn_model, 
                               task_setting['lesion'],
                               task_setting['anlys_file'])
        
        # train and/or test models
        for label, model in rnn_model.items():
            # global settings
            log_path = task_setting["log_path"] # "../log/"
            
            data_name = time.strftime("%Y%m%d_%H%M", time.localtime())
            if label: 
                data_name = label + '-'+ data_name
            print(data_name)
            
            # data processor
            dp = data_util.DataProcessor(log_name = data_name)
            
            # a generator generating training dataset
            generator_t = task_setting["generator_t"]
            # a generator generating validate dataset
            generator_v = task_setting["generator_v"]
            
            if task_setting["record"]:
                # create log first
                dp.create_log(log_path, task_setting["name"]) 
                dp.save_condition(model_parameters, task_setting)
            
            task = importlib.import_module('tasks.' + task_setting["name"])
            # test the trial with the batch_size being 1
            sqa = SeqAgent(model, batch_size = 1, cuda=cuda_enabled) 
            
            if task_setting["train"] > 0:
                print('-' * 89)
                print("START TRAINING: {}".format(task_setting["name"]))
                print('-' * 89)

                train_stage(dp, model, sqa, task_setting["model_path"], 
                            generator_t, generator_v, 
                            log_path, task, 
                            cuda_enabled    = cuda_enabled,
                            clip            = model_parameters["clip"], 
                            record          = task_setting["record"],
                            batch_size      = model_parameters["batch_size"],
                            train_truncate  = task_setting["train_num"], 
                            )
                    
            if task_setting["validate"]:
                print('-' * 89)
                print("START VALIDATION: {}".format(task_setting["name"]))
                print('-' * 89)
                rwd_rate, cpt_rate = validate(dp, model, sqa, generator_v, task,
                                              task_setting["record"], 
                                              truncate=int(5e3))
                print("wining_rate :{}| completed_rate :{}".format(rwd_rate,
                                                                   cpt_rate))
                
if __name__ == '__main__':

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(path)
    sys.path.append(os.path.join(path, 'datagenerator'))
    sys.path.append(os.path.join(path, 'holmes'))
    os.chdir(os.path.join(path, 'seqrnn_multitask'))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="if cuda acceleration is enabled",
                        default=1,
                        type=bool)
    parser.add_argument("--config", help="specify the configuration file ",
                        default='config_shape', # config_shape/config_sure/config_mi/config_st
                        type=str)
    parser.add_argument("--inactivation", help=" inactivate the output connection or not",
                        default=1, 
                        type=bool)
    args = parser.parse_args()
    
    config_file  = args.config
    inactivation = args.inactivation
    
    config = importlib.import_module(config_file)
    if not inactivation:
        task_configurations = config.task_configurations
    else:
        task_configurations = config.task_configurations_lesion
        
    reset_hidden = task_configurations[0]['reset_hidden']
    
    main(task_configurations, args)
