# coding: utf-8
# Data Loader
# @Author: Huzi Cheng
# @Date: 2017.08.25
# pylint: disable=C0103
# pylint: disable=E1101
"""
This Module is used to load the data from the .mat file.
"""
from __future__ import division

import copy
import csv
import os
import pickle
import time

import numpy as np
import scipy.io
import tables
import torch
from torch.autograd import Variable
# from yaml import CDumper as Dumper
from yaml import Dumper
from yaml import dump

INPUT_CHANNEL = 16 

class WinningQueue:
    """
    ...
    """
    def __init__(self, size=3):
        self.winning_rate_list = list([0.0 for i in range(size)])
        self.size = size

    def update(self, new_v):
        new_v = float(new_v)
        print("new_v: {}".format(new_v))
        #assert type(new_v) == float
        self.winning_rate_list.append(new_v)
        self.winning_rate_list.pop(0)

    def get_kw(self):
        return sum(self.winning_rate_list)/self.size, min(self.winning_rate_list)

    def expand(self):
        pass



class Dictionary(object):# could we delete this class, it seems not necessary
    """
    the map from state_vector to state
    """

    def __init__(self, create_state=True):
        self.state2idx = {}
        self.idx2state = []

        if create_state:
            s_n = 0
            for f in [0, 1]:  # fixation
                for t in [0, 1, 2]:  # target
                    for c in [0, 1, 2]:  # choice
                        for r in [0, 1]:  # reward
                            for s in [i for i in range(11)]:  # shape
                                state = copy.deepcopy(self.pa2state(f, t, s, c, r))  # nopep8
                                self.add_state(state)
                                s_n = s_n + 1
            print("State Dictionary Initialized!\n Number of States: ", s_n)

    def __len__(self):
        return len(self.idx2state)

    def pa2state(self, fixation, target, shape, choice, reward):
        """turn experiment condition to state
        """
        state = np.zeros(INPUT_CHANNEL)
        if fixation == 1:
            state[0] = 1
        else:
            state[0] = 0

        if reward == 1:
            state[15] = 1
        else:
            pass

        if target == 2:  # right
            state[2] = 1
        elif target == 1:  # left
            state[1] = 1
        else:
            pass

        if choice == 2:  # right
            state[14] = 1
        elif choice == 1:  # left
            state[13] = 1
        else:
            pass

        if not shape == 0:  # 0, [1,10]
            state[2 + shape] = 1
        else:  # no shape
            pass
        return state

    def add_state(self, state):
        """state.shape (INPUT_CHANNEL,) type np.int
        """
        state = state.astype(np.int)
        state_st = "".join([str(i) for i in state])
        if state_st not in self.state2idx:
            self.idx2state.append(state_st)
            self.state2idx[state_st] = len(self.idx2state) - 1
        return self.state2idx[state_st]

    def encode(self, state):
        """from state to s_v tensor
        return state_vector(tensor) and index
        """
        state_vector = torch.zeros(len(self.idx2state))
        state = state.astype(np.int)
        state_st = "".join([str(i) for i in state])
        index = self.state2idx[state_st]
        state_vector[index] = 1
        return state_vector, index

    def decode(self, state_vector):
        """from tensor to ndarry
        state_vector size (396)
        return state(ndarry) and index
        """
        index = torch.max(state_vector, 0)[1][0]
        state_st = self.idx2state[index]
        state = np.zeros(INPUT_CHANNEL)
        for i, j in enumerate(state_st):
            state[i] = int(j)
        return state.astype(np.int), index


class DataProcessor(object):
    """
    load data, encode data, decode data
    """

    def __init__(self,log_name = time.strftime("%Y%m%d_%H%M", time.localtime()), cud=0):
        if cud:
            self.cud = 1
        else:
            self.cud = 0
        self.data_attr = ""
        self.data_path = ""
        self.log_file = None
        self.raw_data_path = ""
        self.cond_file = ""
        self.data_set = None
        self.writer = None
        self.params = {}
        self.log_path = None
        self.log_name = log_name
        self.task_name = None
        self.raw_file = None
        self.hdf5_path = None
        self.validation_brief_file = None
        self.validation_brief_writer = None

    def load_data_v2(self, data_path, data_attr, data_brief_attr):
        """
        Turn data into the numpy format.
        """
        print("Current Path: ", os.getcwd())
        self.data_path = data_path
        self.data_attr = data_attr
        mat = scipy.io.loadmat(data_path)
#        self.data_set = np.array([np.transpose(t[0]).astype(np.int) for t in mat[data_attr]])
#        self.data_set = [np.transpose(t[0]) for t in mat[data_attr]]
        self.data_set = [t[0] for t in mat[data_attr]]
        self.data_conditions = mat[data_brief_attr][0,0]
        return self.data_set, self.data_conditions

    def get_target(self, state, n2t=True):
        """extract the target index(Zero Indexing) from a state.
        """
        index = 0
        for i, val in enumerate(state[::-1]):
            index = index + np.power(2, i) * val
        if n2t:
            index_t = torch.Tensor(1).type(torch.LongTensor)
            index_t[0] = int(index)
            if self.cud:
                return Variable(index_t.cuda())
            else:
                return Variable(index_t)
        else:
            return int(index)

    def create_log(self, log_path, suffix):
        """
        create a file object named with the local time.
        """
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            
        self.task_name = suffix
#        self.log_name = self.log_name + '-' + suffix
        self.raw_file = log_path + self.log_name + '-' + suffix + '.raw' + '.p'
        self.hdf5_path = log_path + self.log_name + '-' + suffix
        self.log_file = open(log_path + self.log_name + '-' + suffix + '.csv', 'w')

        self.writer = csv.writer(self.log_file)
        headers = ['trial_num', 'correct_rate', 'loss', 'training']
        self.writer.writerow(headers)


    def write_log(self, count, correct_rate, loss, training=True):
        """Write line into log file"""
        record_line = [count, correct_rate, loss, training]
        self.writer.writerow(record_line)

    def write_validation_brief(self, step, trial_brief_base64):
        record_line = [step, trial_brief_base64]
        self.validation_brief_writer.writerow(record_line)

    def save_raw(self, raw_data):
        """save raw data (real and generated)"""
        with open(self.raw_file, 'wb') as fp:
            pickle.dump(raw_data, fp)
            print('Raw Data Saved!')

    def create_hdf5raw(self, behavior_shape, neuron_shape, suffix = ''):

        self.validation_brief_file = open(self.log_path + self.log_name + suffix + '-' + self.task_name +'.validation_brief.csv', 'w')
        self.validation_brief_writer = csv.writer(self.validation_brief_file)
        validation_brief_headers = ['step', 'trial_brief_base64']
        self.validation_brief_writer.writerow(validation_brief_headers)

        self.raw_hdf5file = tables.open_file(self.log_path + self.log_name + suffix + '-' + self.task_name + '.hdf5', mode='w')
        content_type = tables.Float64Atom()
        index_dtype = tables.UInt64Atom()
        label_shape = (0, 1, 2)
        neuron_shape.insert(0, 0)
        behavior_shape.insert(0, 0)

        self.behavior_storage = self.raw_hdf5file.create_earray(self.raw_hdf5file.root, 'behavior', content_type, shape=behavior_shape)
        self.prediction_storage = self.raw_hdf5file.create_earray(self.raw_hdf5file.root, 'prediction', content_type, shape=behavior_shape)
        self.rawoutput_storage = self.raw_hdf5file.create_earray(self.raw_hdf5file.root, 'rawoutput', content_type, shape=behavior_shape)
        self.neuron_storage = self.raw_hdf5file.create_earray(self.raw_hdf5file.root, 'neuron', content_type, shape=neuron_shape)
        self.index_storage = self.raw_hdf5file.create_earray(self.raw_hdf5file.root, 'index', index_dtype, shape=label_shape)
        
    def append_hdf5raw(self, behavior_data, prediction_data, raw_output, neuron_data, index_data):
        self.behavior_storage.append(behavior_data)
        self.prediction_storage.append(prediction_data)
        self.rawoutput_storage.append(raw_output)
        self.neuron_storage.append(neuron_data)
        self.index_storage.append(index_data)

    def close_hdf5raw(self, ):
        self.raw_hdf5file.close()
        return 1

    def save_condition(self, model_params, task_setting):
        """Save the experiment condition into yaml"""
        print("saving conditions...")
        
#        self.cond_file = save_path + self.log_name + '-' + self.task_name + '.yaml'
        save_path, file = os.path.split(task_setting["model_path"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        self.cond_file = save_path + '/' + self.log_name + '-' + self.task_name + '.yaml'
        self.params = {
            "log name":             self.log_name + '-' + self.task_name,
            "log path":             task_setting["log_path"],
            "model path":           task_setting["model_path"],
            "load_model":           task_setting["load_model"],

            "RNN type":             task_setting["model_mode"],
            "optimizer":            task_setting["optimizer"],
            "Network hidden cells": model_params["nhid"],
            "Network layers":       model_params["nlayers"],
            "Input size":           model_params["input_size"],
            "Batch size":           model_params["batch_size"],

            "train_num":            task_setting["train_num"],
            "record":               task_setting["record"],
            "training":             task_setting["train"],
            "validate":             task_setting["validate"],
            "lesion":               task_setting["lesion"],
            "anlys_file":           task_setting["anlys_file"],

            "Init learning rate":   model_params["lr"],
            "reset_hidden":         task_setting["reset_hidden"],
            
#            "task_setting":         task_setting# save all as an insurance policy
        }
        
        with open(self.cond_file, 'w') as fj:
            output = dump(self.params, Dumper=Dumper, default_flow_style=False)
            fj.write(output)
            fj.close()
        print("Experiment Condition saved")
    def get_logname(self):
        return self.log_name
    def name_update(self,new_name):
        self.log_name = new_name + self.raw_log_name

def main():
    """Class test"""
#    dt = Dictionary()


if __name__ == '__main__':
    main()
