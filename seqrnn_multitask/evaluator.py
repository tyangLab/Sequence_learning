# -*- coding: utf-8 -*-
# @Author: Huzi Cheng
# @Date: 09/11/2017, 14:38

from __future__ import division
from __future__ import print_function

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""
trial: time * channel matrix

transfer this matrix to generate a generative model.

If the network do not made a meaningful choice in 20 shapes the trial will be cut off.

The whole trial is simulated as a interactive environment.

INFO
==========================================================
raw_prediction:

tmp_loss:

correct_rate:

raw_rec:[, , , ,]

"""


def match_rate(t1, t2):
    """Compute the correct rate of prediction"""
    if t1.shape == t2.shape:
        diff = np.sum(np.abs(t1 - t2))  # trial is binary
        return 1 - (diff / (t1.size))
    else:
        print("Matched shape? No!")
        return 0
    
def softmax(x,beta = 8):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(beta*x) / np.sum(np.exp(beta*x), axis=0) 


def cons_np2c(bsz, cuda=False):
    if cuda:
        def func(state_vector, require_g=True):
            return Variable(torch.Tensor(state_vector).type(torch.FloatTensor).view(1, bsz, -1).cuda(),
                            requires_grad=require_g)

        def func2(t_variable):
            return t_variable.data.cpu().numpy() # for GRU
    else:
        def func(state_vector, require_g=True):
            return Variable(torch.Tensor(state_vector).type(torch.FloatTensor).view(1, bsz, -1),
                            requires_grad=require_g)
            pass

        def func2(t_variable):
            return t_variable.data.numpy()

    return func, func2


class Calculator:
    def __init__(self, batch_size = 1, cuda_enabled=False):
        self.cuda_enabled = cuda_enabled
        self.np2t, self.t2np = cons_np2c(batch_size, self.cuda_enabled)

    def get_output(self, model, trial_data, hidden, tmp_reward, 
                   bsz = None, tau=np.inf):
        """
        cal the output tensor of specific input.
        model.train_sensory[0]: whether sensory nodes are trained
                           [1]: the index of sensory output units 
        
        model.train_action [0]: whether action nodes are trained
                           [1]: the index of action output units 
                
        temp_reward: tmp_reward[:,0]: whether this trial is needed for training
                     tmp_reward[:,1]: the first time point is used for
                                         training, if necessary
                     tmp_reward[:,2]: the last time point is used for
                                         training, if necessary
        
        """
        total_loss = 0
        criterion = model.criterion
        prediction_trial_length = trial_data.shape[0] - 1
        
        pred_trial = np.zeros((prediction_trial_length, model.batch_size, model.ninp))
        
        #___________________________________________________________________
        # reward_guide is used to indicating which outputs are required
        #     for training
        reward_guide = np.tile(tmp_reward[:,0], (trial_data.shape[2],1))
        reward_guide = np.tile(reward_guide, (trial_data.shape[0],1,1)).transpose(0,2,1)
        for i in range(model.batch_size):
            reward_guide[:tmp_reward[i,1], i, :] = 0
            reward_guide[tmp_reward[i,2]:, i, :] = 0

        # turn it into torch Variable
        if self.cuda_enabled:
            nstep = Variable(torch.Tensor(np.array([0])).type(torch.FloatTensor).cuda(),requires_grad=False)
            reward_guide = Variable(torch.Tensor(reward_guide).type(torch.FloatTensor).cuda(),requires_grad=False)
        else:
            nstep = Variable(torch.Tensor(np.array([0])).type(torch.FloatTensor),requires_grad=False)
            reward_guide = Variable(torch.Tensor(reward_guide).type(torch.FloatTensor),requires_grad=False)
        #___________________________________________________________________
        # the time step when the hidden response is saved and as the initial 
        #     response in the next trial or training_trial
        save_step = int(trial_data.shape[0]/2-1)
        
        for i in range(prediction_trial_length):
            inputs = self.np2t(trial_data[i  ,:,:])
            n_put  = self.np2t(trial_data[i+1,:,:], require_g=False)
            output, hidden = model(inputs, hidden, bsz = bsz)
            
            if i == save_step: #TODO: for now, it is only for two step task
                hidden_ = copy.deepcopy(Variable(hidden))#for GRU
            
            n_put = n_put.reshape(bsz, -1) # 1 by batch size by number of inputs
            
            if reward_guide[i].sum() == 0:
                loss = criterion(output*0.0, n_put*0.0)

            else:
                loss  = criterion((output[reward_guide[i]==1]).reshape([1,-1])[0], 
                                  (n_put[reward_guide[i]==1]).reshape([1,-1])[0])

            if not np.isnan(loss.item()):
                nstep += reward_guide[i].sum()
                total_loss = total_loss + loss*reward_guide[i].sum()
                
            raw_output = self.t2np(output)
            pred_trial[i,:,:] = np.around(raw_output)
        
        total_loss = total_loss/nstep if nstep!=0 else total_loss
        return pred_trial, total_loss, hidden_

    
class SeqAgent:
    """
    
    
    """
    def __init__(self, model, batch_size=1, cuda=False):
        self.cuda = cuda
        self.batch_size = batch_size
        self.model = model  # an instance of CPU or GPU nn module
        self.hidden = self.model.init_hidden(bsz = batch_size)
        self.raw_records = []  # ndarry
        # self.proposed_records = []
        self.hidden_records = []  # ndarry
        self.sensory_sequence = []
        self.input_gate = []
        self.reset_gate = []
        self.new_gate = []
        self.np2t, self.t2np = cons_np2c(batch_size, self.cuda)
        
    def init_hidden(self, batch_size=1):
        self.hidden = self.model.init_hidden(bsz = batch_size)
        
    def get_action(self, state_vector, choice_position):
        """
        Here the output is a [1, channel_num] tensor variable
        :param state_vector:
        :return:
        """
        processed_input = self.np2t(state_vector, require_g=False)
        
        model = self.model
        output, self.hidden = model(processed_input, self.hidden, bsz = self.batch_size)
        
        if model.cuda_enabled :
            action_options = output[0, choice_position].cpu().detach().numpy() 
        else:
            action_options = output[0, choice_position].detach().numpy()
        
        pro_soft = softmax(action_options)
        idx = torch.tensor(np.random.choice(pro_soft.size, 1, p=pro_soft))
        selected_action = [idx.data.item()] # 0: fixation point, 1: left, 2: right, 3: other position
        
        self.raw_records.append(self.t2np(output).reshape(-1))
        self.hidden_records.append(self.t2np(self.hidden))
        self.sensory_sequence.append(state_vector)
        
        return selected_action

    def reset(self):
        self.raw_records = []
        self.hidden_records = []
        self.sensory_sequence = []
        self.input_gate = []
        self.reset_gate = []
        self.new_gate = []

    def summary(self):
        self.sensory_sequence = np.array(self.sensory_sequence)
        self.raw_records = np.array(self.raw_records)
        self.hidden_records = np.array(self.hidden_records)


def interactive_sample(sqa, task_env):
    """
    :param sqa: Sequence prediction agent
    :param task_env: Task object
    :return:
    ========================================
    assume that trial_data is time * channel matrix

    when the action terminates the trial, the env will return a stop signal 
        and a last input to the agent
    ========================================
    
    """
    raw_rec = {}
    sqa.reset()
    
    action = [0]  # init action: fixate on fixation point
    trial_end = False
    planaction_record = []
    while not trial_end:
        trial_end, sensory_inputs = task_env.step(action)
        action = sqa.get_action(sensory_inputs,task_env.about_choice)
        planaction_record.append(action) 

            
    sqa.summary()
    # trial end, collect data
    predicted_trial = np.around(sqa.raw_records).astype(np.int)
    for i in range(predicted_trial.shape[0]):  
        if len(planaction_record[i])!=0:
            action = planaction_record[i][0]
            predicted_trial[i, task_env.about_choice] = 0
            predicted_trial[i, task_env.about_choice[0] + action] = 1
        elif len(planaction_record[i]) == 0:
            predicted_trial[i, task_env.about_choice] = 0

    raw_rec["sensory_sequence"] = copy.deepcopy(sqa.sensory_sequence)  # validation set
    raw_rec["predicted_trial"] = copy.deepcopy(predicted_trial)  # predicted set
    raw_rec["raw_records"] = copy.deepcopy(sqa.raw_records)  # raw output
    raw_rec["hidden_records"] = copy.deepcopy(sqa.hidden_records)  # raw hidden
    
    # when calculate correct ratio, we need to truncate the trial
    correct_rate = match_rate(sqa.sensory_sequence[1:,:], predicted_trial[:-1,:])
    total_loss = np.mean(np.power(sqa.sensory_sequence[1:] - sqa.raw_records[:-1], 2))
    
    return raw_rec, total_loss, correct_rate
