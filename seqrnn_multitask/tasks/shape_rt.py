# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:21:58 2018

@author: Zhewei Zhang, Huzi Cheng
"""

from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from .tools import obj2base64, base642obj


class Task:
    """
    Reaction time task
    """

    def __init__(self):
        """
        Warning:
            1. here we assume that the model will have a perfect after-choice behavior!
                changing choice behavior is excluded from analysis!
        """
        self.states_pool = []
#        self.time_step = 0
        self.time_step = -1

        self.shape_offset = 3
        self.shape_interval = 5
        self.temp_wegihts = None
        self.experienced_shape_num = 0
        self.experienced_temp_weights = 0
        self.interrupt_shape_bound = 30

        self.trialtype = None
        self.trial_end = None
        
        self.reward = None
        self.chosen = None
        self.choice = None # choice for left and right
        self.completed = False
        self.choice_time = None

        self.trial_length = None
        self.action_step = np.arange(7,200,self.shape_interval).tolist()
        self.about_target = [1,2]
        self.about_choice = [14,15,16,17]
        self.about_reward = [18,19]

        self.interrupt_states = [  # no visual inputs, fixate on other position, and no reward
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        ]
        self.failed_states = [  # no visual inputs, fixate on other position, and no reward by zzw
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
        ]
        self.successful_states = [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
        ]
        

    def configure(self, trial_data, settings):
        """

        :param trial_data: np.ndarry with shape (time, channels)
        :param settings:
        :return:
        """
        self.states_pool = trial_data[0].tolist()
        self.trialtype = settings["trialtype"]  # 0: green is rewarded; 1: red is rewarded
        self.temp_wegihts = settings["temp_wegihts"]

        self.time_step = -1
        self.trial_end = False
        self.experienced_shape_num = 0
        self.experienced_temp_weights = 0
        
        self.chosen = None
        self.choice = None
        self.reward = None
        self.completed = False
        self.choice_time = None
        
        self.trial_length = None
        self.action_step = np.arange(7,200,self.shape_interval).tolist()
        
        
        
    def step(self, action):
        """
        :param action: a number in (0,1,2,3): fix on fixation point, left, right, other positions
        :return: trial_end, next sensory_inputs
        """
        self.time_step = self.time_step + 1
        
        if len(action) != 1 or not(action[0] in (0, 1, 2, 3)):
            self.choice = action # zzw 20200315
            self.states_pool = copy.deepcopy(self.interrupt_states)
        else:
            action = action[0]
            if not (self.time_step  in self.action_step):
                if action == 0:  # fixate, keep silent
                    pass
                else:
                    if self.time_step > self.action_step[-1]:
                        # fication on target has been done
                        pass
                    else:
                        self.choice = action # zzw 20200315
#                       print('response in wrong time window or break fixation')
                        self.states_pool = copy.deepcopy(self.interrupt_states)
            else:
                if not self.chosen:
                    if action == 0:  # fixate, keep silent
                        pass
                    elif action == 3:
    #                    print('response in wrong time window')
                        self.states_pool = copy.deepcopy(self.interrupt_states)
                    elif action in (1, 2):  # made a choice
                        # action 1: left, action 2: right
                        # change the states_pool according to the time step and settings.
                        # check if there will be reward
                        self.chosen = True
                        self.choice = action
                        
                        self.trial_length = self.time_step+5
                        self.action_step = [self.time_step, self.time_step+1, self.time_step+2]
                        self.experienced_shape_num = int(np.ceil((self.time_step - self.shape_offset) / self.shape_interval))
                        self.experienced_temp_weights = self.temp_wegihts[:self.experienced_shape_num]
    
                        self.reward = (2-action == self.trialtype)
                        
                        if self.reward:
                            state_ = copy.deepcopy(self.successful_states)
                        else:
                            state_ = copy.deepcopy(self.failed_states)
                            
                        # customize the choice position
                        state_[0][self.choice + 14] = 1
                        state_[1][self.choice + 14] = 1
                        state_[2][self.choice + 14] = 1
                        # customize the target
                        state_[1][self.choice] = 1

                        self.states_pool = copy.deepcopy(state_)
                    else:
                        raise BaseException("Wrong action index!")
                else:
                    # decision has been made, and fixation on target has to been kept for a while
                    if self.choice == action:
                        pass
                    else:
                        self.choice = np.inf # zzw 20200315
                        self.states_pool = copy.deepcopy(self.interrupt_states)
        try:
            next_sensory_inputs = self.states_pool.pop(0)
            if len(self.states_pool) == 0:
                self.trial_end = True
                if self.time_step == self.trial_length:
                    self.completed = True
                else:
                    self.reward = None
            return self.trial_end, next_sensory_inputs
#            return self.trial_end, next_sensory_inputs
        except IndexError:
            print("Wrong States Pool! Find Out Your Fucking Task Dynamic Error & Fix It!!!")


    def reset_configuration(self):
        
        self.states_pool = []
        self.trialtype = None  # 0: green is rewarded; 1: red is rewarded
        self.temp_wegihts = None

        self.time_step = -1
        self.trial_end = False
        self.experienced_shape_num = 0
        self.experienced_temp_weights = 0
        
        self.chosen = None
        self.choice = None
        self.reward = None
        self.completed = False
        self.choice_time = None
        
        self.trial_length = None
        self.action_step = np.arange(7,200,self.shape_interval).tolist()

    def extract_trial_abstractb64(self):
        info = {
            "tmp_weight_sequence": self.experienced_temp_weights,
            "rt":self.experienced_shape_num,
            "chosen": self.chosen,
            "choice": self.choice,
            "reward": self.reward,
            "trialtype": self.trialtype,
            "completed": self.completed,
            "choice_time": self.action_step
        }
        
        return obj2base64(info)

    def is_winning(self):
        """
        This allows the agent to know if this trial is wined
        :return: 1 if win else 0
        """
        return 1 if self.reward else 0

    def is_completed(self):
        
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
        self.trialtypes      = np.array(conditions["trialtype"]).reshape(-1)
        self.temp_wegihts    = np.array(conditions["temp_wegihts"]).squeeze()
        self.shape_sequences = np.array(conditions["trialCondi"]).squeeze()


    def __getitem__(self, item):
        if isinstance(item, int):
            if item <= self.temp_wegihts.shape[0]:
                return {"trialtype" : self.trialtypes[item],
                        "temp_wegihts": self.temp_wegihts[item].reshape(-1),
                        "shape_sequences": self.shape_sequences[item].reshape(-1),
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
