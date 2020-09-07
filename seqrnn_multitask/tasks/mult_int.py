#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:36:02 2019

@author: Zhewei Zhang

Sure task adapted from Kiani 2009

"""

from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from .tools import obj2base64, base642obj


class Task:
    """
    Daw two step task
    """

    def __init__(self):
        """
        Warning:
            1. here we assume that the model will have a perfect after-choice behavior!
                changing choice behavior is excluded from analysis!
        """
        self.states_pool = []
        self.time_step = -1
        self.trial_length = 16
#        self.trial_length = 17
        self.trial_end = None
        self.reward = None
        self.completed = None
        self.choice = None
        self.chosen = None

        self.modality = None
        self.directions = None
        self.estimates = None

        self.action_step = [11,12,13]
        self.targets = [1,2]
        self.about_choice = [20,21,22,23]
        self.about_reward = [24,25]
        
        self.interrupt_states = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]]
        self.completed_states = [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        ]
        self.corr_num = 0
        self.wrong_num = 0


    def configure(self, trial_data, settings):
        """
         specific for each trial, or reset the parameters at the beginning of each trial
        :param trial_data: np.ndarry with shape (time, channels)
        :param settings:
        :return:
        """
        self.states_pool = trial_data[0].tolist()
        self.completed = None
        self.trial_end = False
        self.reward = None
        self.time_step = -1
        
        self.choice = None
        self.chosen = None
        self.modality = settings["modality"]
        self.directions = settings["directions"]
        self.estimates = settings["estimates"]

    def step(self, action):
        """
        :param action: a number in (0,1,2,3): fix on fixation point, left, right, other positions
        :return: trial_end, next sensory_inputs
        """
        self.time_step += 1
#        print(self.time_step)
#        print(action)
        if len(action) != 1 or not(action[0] in (0, 1, 2, 3)):
            self.states_pool = copy.deepcopy(self.interrupt_states)
        else:
            action = action[0]
            if not (self.time_step in self.action_step):# choice has not been made
                if action == 0:  # fixate, keep silent
                    pass
                elif action == 3 and (self.time_step in [0, 1] or self.time_step > self.action_step[-1]):
                    pass
                else:
#                    print(self.time_step)
                    self.states_pool = copy.deepcopy(self.interrupt_states)
                    
            elif self.time_step == self.action_step[0]:# choice has not been made
                if action == 0 or action == 3:  # fixate, keep silent
                    self.states_pool = copy.deepcopy(self.interrupt_states)
                elif action == 1:
                    reward = True if self.directions>90 else False
                elif action == 2:
                    reward = True if self.directions<90 else False
                if self.directions == 90:
                    reward = True if np.random.rand()>0 else False
                    
                if action == 1 or action == 2:
                    self.choice = action
                    self.reward = reward
                    self.chosen = True
                    state_ = copy.deepcopy(self.completed_states)
                    state_[1][self.targets[action-1]]=1
                    for i in [0,1,2]:# about action
                        state_[i][self.about_choice[action]]=1
                    if reward == 1:# about reward
                        for i in [2,3]:
                            state_[i][self.about_reward[0]]=1
                    else:
                        for i in [2,3]:
                            state_[i][self.about_reward[1]]=1
                    self.states_pool = copy.deepcopy(state_)
                    
            elif self.time_step in self.action_step[1:]:# choice has not been made
                if action == self.choice:
                    pass
                else:
                    self.reward = None
                    self.states_pool = copy.deepcopy(self.interrupt_states)
            else:
                raise BaseException("Wrong time step index!")
        try:
            next_sensory_inputs = self.states_pool.pop(0)
            if len(self.states_pool) == 0:
                self.trial_end = True
                if self.time_step == self.trial_length:
                    self.completed = True
                else:
                    self.reward = None
            return self.trial_end, next_sensory_inputs
        except IndexError:
            print("Wrong States Pool! Find Out Your Fucking Task Dynamic Error & Fix It!!!")

    def reset_configuration(self):
        
        self.time_step = -1
        self.trial_end = False
        self.completed = None
        self.reward = None
        self.states_pool = []
        
        self.choice = None
        self.chosen = None
        self.modality = None
        self.directions = None
        self.estimates = None



    def extract_trial_abstractb64(self):
        info = {
            "choice": self.choice,
            "chosen": self.chosen,
            "reward": self.reward,
            "completed": self.completed,
            "modality": self.modality,
            "directions": self.directions,
            "estimates": self.estimates

        }
        return obj2base64(info)

    def is_winning(self):
        """
        This allows the agent to know if this trial is wined
        :return: 1 if win else 0
        """
#        print(self.trial_reward)
        if self.reward:
            return 1
        else:
            return 0

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
    def __init__(self, validation_conditions):
        self.directions = validation_conditions["direction"]
        self.estimates = validation_conditions["estimate"]
        self.modality = validation_conditions["modality"]
    def __getitem__(self, item):
        if isinstance(item, int):
            if item <= self.directions.size:
                return {"directions": self.directions[item],
                        "estimates": self.estimates[item,:],
                        "modality": self.modality[item]
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
