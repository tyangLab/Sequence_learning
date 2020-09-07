# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:32:05 2019

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
        self.trial_length = None
        self.trial_end = None
        self.reward = None
        self.completed = None
        self.choice = None
        self.chosen = None
        
        self.sure_trial = None
        self.randots_dur = None
        self.coherence = None
        self.sure_reward = 0.5
        self.acc_evi = None

        self.action_step = []
        self.targets = [1,2,3]
        self.about_choice = [15,16,17,18,19]
        self.about_reward = [20,21]
        
        self.interrupt_states = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]]
        self.completed_states = [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
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
        self.acc_evi = settings["acc_evi"]
        self.sure_trial = settings["sure_trial"]
        self.coherence = settings["coherence"]
        self.randots_dur = settings["randots_dur"]

        self.trial_length = 17 + settings["randots_dur"]
        self.action_step = [10,11,12] + settings["randots_dur"]
        
    def step(self, action):
        """
        :param action: a number in (0,1,2,3): fix on fixation point, left, right, other positions
        :return: trial_end, next sensory_inputs
        """
        self.time_step += 1
#        print(self.time_step)
#        print(action)
        if len(action) != 1 or not(action[0] in (0, 1, 2, 3, 4)):
            self.states_pool = copy.deepcopy(self.interrupt_states)
        else:
            action = action[0]
            if not (self.time_step in self.action_step):# choice has not been made
                if action == 0:  # fixate, keep silent
                    pass
                elif action == 4 and (self.time_step in [0, 1] or self.time_step > self.action_step[-1]):
                    pass
                else:
#                    print(self.time_step)
#                    if self.time_step < self.action_step[0]:
#                        print('early response')
                    self.states_pool = copy.deepcopy(self.interrupt_states)
                    
            elif self.time_step == self.action_step[0]:# choice has not been made
                if action == 0 or action == 4:  # fixate, keep silent
                    reward = None
#                    print('no response')
                    self.states_pool = copy.deepcopy(self.interrupt_states)
                elif action == 1:
                    reward = 1 if self.coherence>0 else 0
                    reward = np.random.rand()>0.5 if self.coherence==0 else reward                    
                elif action == 2:
                    reward = 1 if self.coherence<0 else 0
                    reward = np.random.rand()>0.5 if self.coherence==0 else reward
                elif action == 3: # sure target
                    if self.sure_trial == 1:
                        reward = self.sure_reward
                    else:
                        reward = None
#                       print('choose sure target in no sure target trial')
                        self.states_pool = copy.deepcopy(self.interrupt_states)
                
                if reward != None:
                    self.choice = action
                    self.reward = reward
                    self.chosen = True
                    state_ = copy.deepcopy(self.completed_states)
                    if self.sure_trial:
                        state_[0][self.targets[-1]]=1
                    state_[1][self.targets[action-1]]=1
                    for i in [0,1,2]:# about action
                        state_[i][self.about_choice[action]]=1
                    if reward >0:# about reward
                        for i in [3,4]:
                            state_[i][self.about_reward[0]]=reward
                            state_[i][self.about_reward[1]]=0

                    self.states_pool = copy.deepcopy(state_)
                    
            elif self.time_step in self.action_step[1:]:# choice has not been made
                if action == self.choice:
                    pass
                else:
#                    print('change mind')
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
            return self.trial_end, next_sensory_inputs#, self.completed
        except IndexError:
            print("Wrong States Pool! Find Out Your Fucking Task Dynamic Error & Fix It!!!")

    def reset_configuration(self):
        
        self.time_step = -1
        self.trial_end = False
        self.completed = None
        self.reward = None
        self.states_pool = []
        self.action_step = []


        self.choice = None
        self.chosen = None
        self.sure_trial = None
        self.coherence = None
        self.randots_dur = None
        self.trial_length = None
        
    def extract_trial_abstractb64(self):
        info = {
            "choice": self.choice,
            "chosen": self.chosen,
            "reward": self.reward,
            "sure_trial":self.sure_trial,
            "coherences":self.coherence,
            "completed": self.completed,
            "acc_evi": self.acc_evi,
            "randots_dur":self.randots_dur
        }
        return obj2base64(info)

    def is_winning(self):
        """
        This allows the agent to know if this trial is wined
        :return: 1 if win else 0
        """
#        print(self.trial_reward)
        if self.reward == None:
            return 0
        elif self.reward > 0:
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
    def __init__(self, conditions):
        self.coherences  = conditions["coherences"]
        self.sure_trials = conditions["sure_trials"]
        self.randots_dur = conditions["rdmdots_dur"]
        self.acc_evi     = conditions["acc_evi"]
                        
    def __getitem__(self, item):
        if isinstance(item, int):
            if item <= self.coherences.size:
                return {"coherence": self.coherences[item],
                        "sure_trial": self.sure_trials[item],
                        "randots_dur": self.randots_dur[item],
                        "acc_evi": self.acc_evi[item],
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
