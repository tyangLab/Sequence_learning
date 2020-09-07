#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:04:04 2019

@author: Zhewei Zhang

two step task adapted from Daw et.al, 2011
"""

from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from .tools import obj2base64, base642obj


class Task:
    """
    reversal learning
    """

    def __init__(self):
        """
        Warning:
            1. here we assume that the model will have a perfect after-choice behavior!
                changing choice behavior is excluded from analysis!
        """
        self.states_pool = []
        self.time_step = -1

        self.trial_length = 12
        self.completed = None
        self.trial_end = None
        self.reward = None
        self.state = None
        self.choice = None
        self.chosen = None
        self.common = None

        self.block = None
        self.tran_probs = None
        self.rwd_probs = None
        
        self.action_step = [5,6]
        self.about_state  = [2,3]
        self.about_choice = [5,6,7]
        self.about_reward = [8,9]
        
        self.interrupt_states = [[0, 0, 0, 0, 1, 0, 0, 1, 0, 1]]
        self.chosen_states = [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        ]        


    def configure(self, trial_data, settings):
        """
         specific for each trial, or reset the parameters at the beginning of each trial
        :param trial_data: np.ndarry with shape (time, channels)
        :param settings:
        :return:
        """
        self.states_pool = trial_data[0].tolist()
        self.block  = settings['block']
        self.rwd_probs  = settings['rwd_probs']
        self.tran_probs = settings['tran_probs']

        self.trial_end = False
        self.reward = None
        self.time_step = -1
        self.completed = None
        
        self.choice = None
        self.chosen = None
        self.state = None#
        self.common = None
        
    def step(self, action):
        """
        :param action: a number in (0,1,2,3): fix on fixation point, left, right, other positions
        :return: trial_end, next sensory_inputs
        """
        self.time_step += 1
#        print(self.time_step)
#        print(action)
        if len(action) != 1 or not(action[0] in (0, 1, 2)):
            self.states_pool = copy.deepcopy(self.interrupt_states)
        else:
            action = action[0]
            if not (self.time_step in self.action_step):# choice has not been made
                if action == 0:  # fixate, keep silent
                    pass
                else:
#                    print('response in wrong time window')
                    self.states_pool = copy.deepcopy(self.interrupt_states)
            elif self.time_step == self.action_step[0]:# choice has not been made

                if action == 0:  # fixate, keep silent
#                    print('no response')
                    self.states_pool = copy.deepcopy(self.interrupt_states)
                elif action == 1:
                    state = 1 if self.tran_probs>np.random.rand() else 2
                elif action == 2:
                    state = 1 if (1- self.tran_probs)>np.random.rand() else 2
                    
                if action != 0:
#                    print("block: {:6f} | choice: {:6f} | reward: {:.6f} ".format(self.block,action,reward))
                    reward = self.rwd_probs>np.random.rand() if state==1 else (1-self.rwd_probs)>np.random.rand()
                    self.common = state==action
                    self.state = state
                    self.choice = action
                    self.reward = reward                
                    self.chosen = True

                    state_ = copy.deepcopy(self.chosen_states)
                    state_[0][self.about_choice[action]]=1
                    state_[1][self.about_choice[action]]=1
                    
                    state_[2][self.about_state[state-1]] = 1
                    state_[3][self.about_state[state-1]] = 1
                    state_[4][self.about_state[state-1]] = 1

                    state_[5][self.about_reward[1-reward]] = 1
                    state_[6][self.about_reward[1-reward]] = 1
                    
                    self.states_pool = copy.deepcopy(state_)

            elif self.time_step == self.action_step[1]:# choice has not been made
                if action == self.choice:
                    pass
                else:
#                    print('change choices')
                    self.states_pool = copy.deepcopy(self.interrupt_states)
            else:
                raise BaseException("Wrong time step index!")
        try:
#            print(self.states_pool)
            next_sensory_inputs = self.states_pool.pop(0)
            if len(self.states_pool) == 0:
                self.trial_end = True
                if self.time_step == self.trial_length:
                    self.completed = True
                else:
                    self.reward = None
#            return self.trial_end, next_sensory_inputs, self.completed
            return self.trial_end, next_sensory_inputs
        except IndexError:
            print("Wrong States Pool! Find Out Your Fucking Task Dynamic Error & Fix It!!!")

    def reset_configuration(self):
        self.time_step = -1
        self.states_pool = []
        self.completed = None
        self.trial_end = False
        self.rwd_probs = None
        self.block = []
        
        self.reward = None
        self.choice = None
        self.chosen = None
        self.state = None

    def extract_trial_abstractb64(self):
        info = {
            "block": self.block,
            "choice": self.choice,
            "state": self.state,
            "chosen": self.chosen,
            "reward": self.reward,
            "completed": self.completed,
            "common": self.common

        }
        return obj2base64(info)

    def is_winning(self):
        """
        This allows the agent to know if this trial is wined
        :return: 1 if win else 0
        """
#        print(self.reward)
        return 1 if self.reward else 0

    def is_completed(self):
        """
        This allows the agent to know if this trial is wined
        :return: 1 if win else 0
        """
        return 1 if self.completed else 0

class TrainingHelper:
    """
    helper in training process
    """

    def __init__(self, training_set, training_conditions):
        self.training_set = training_set
        self.training_guide = training_conditions["training_guide"]

class ValidationHelper:
    """
    helper in training process
    """

    def __init__(self, validation_set, validation_conditions):
        self.validation_set = validation_set
        self.validation_conditions = ValidationConditionList(validation_conditions)


class ValidationConditionList:
    def __init__(self, conditions):
        self.block = conditions["block"]
        self.tran_probs = conditions["tran_probs"]
        self.rwd_probs   = conditions["rwd_probs"]


    def __getitem__(self, item):
        if isinstance(item, int):
            if item <= self.block.shape[0]:
                return {"block": self.block[item] ,
                        "rwd_probs": self.rwd_probs[item],
                        "tran_probs": self.tran_probs[item]
                        }
            else:
                raise IndexError("Out of range!")
            pass
        else:
            raise TypeError("Wrong index type")


class TaskAnalytics:
    """
    task analysis tools
    """

    def __init__(self):
        self.index = None
        pass

    def decode_trial_brief(self, index, log_file):
        try:
            data_brief = base642obj(log_file['trial_brief_base64'][index])
            return data_brief
        except KeyError:
            raise KeyError("Index out of the range!")

    def test(self, **kwargs):
        return "hello" + kwargs["name"]
