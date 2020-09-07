# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 18:19:25 2018

@author: Zhewei Zhang
"""
import os
import sys
import copy
import numpy as np

path = os.getcwd()
os.chdir('../holmes')
from toolkits_whenwhich import wh_loading
os.chdir(path)

def lesion_rnn(rnn_model,lesion_method,anlys_file):
    """
    # 
    """
    model_all = {}
    if not anlys_file:
        model_all[''] = rnn_model
        return model_all
    if isinstance(anlys_file,tuple):
        anlys_file = anlys_file[0]
    y = wh_loading(anlys_file)
    when = y["when"]
    which = y["which"]
    numNeuron_con = 0
    model = []
    lesion_prob = '-'+anlys_file[-8:-5]
    if 'when_pos' in lesion_method:
        neurons = when.pos
        numNeuron_con = np.max((numNeuron_con, neurons.shape[0]))
        model.append(lesion_each('when_pos',lesion_method,rnn_model, neurons, numNeuron_con))
    if 'when_neg' in lesion_method:
        neurons = when.neg
        numNeuron_con = np.max((numNeuron_con, neurons.shape[0]))
        model.append(lesion_each('when_neg',lesion_method,rnn_model, neurons, numNeuron_con))
    if 'which_pos' in lesion_method:
        neurons = which.pos
        numNeuron_con = np.max((numNeuron_con, neurons.shape[0]))
        model.append(lesion_each('which_pos',lesion_method,rnn_model, neurons, numNeuron_con))
    if 'which_neg' in lesion_method:
        neurons = which.neg
        numNeuron_con = np.max((numNeuron_con, neurons.shape[0]))
        model.append(lesion_each('which_neg',lesion_method,rnn_model, neurons, numNeuron_con))
    if 'when' in lesion_method:
        neurons = np.append(when.pos,when.neg)
        numNeuron_con = np.max((numNeuron_con, neurons.shape[0]))
        model.append(lesion_each('when',lesion_method,rnn_model, neurons, numNeuron_con))
    if 'which' in lesion_method:
        neurons = np.append(which.pos,which.neg)
        numNeuron_con = np.max((numNeuron_con, neurons.shape[0]))
        model.append(lesion_each('which',lesion_method,rnn_model, neurons, numNeuron_con))

    if not model:
        model_all[''] = rnn_model
    else:
        model.append(lesion_control(lesion_method, rnn_model, numNeuron_con))
        for model_ in model:
            for key,value in model_.items():
                model_all[key+lesion_prob] = value
    
    return model_all


def lesion_each(name,lesion_method,rnn_model, neurons,numNeuron_les_con):
    model = {}
    if 'output' in lesion_method:
        model[ name + '_output_les'] = set_zeroweight_output(rnn_model, neurons)

    return model


def set_zeroweight_output(rnn_model, pos_neg):
    model_les = copy.deepcopy(rnn_model)
    for i in pos_neg:
        model_les.decoder.weight[:,int(i)] = 0
    model_les.decoder2 = rnn_model.decoder
    
    return model_les

def lesion_control(lesion_method, rnn_model, numNeuron_con):
    numNeuron = rnn_model.decoder.weight.shape[1]
    Neuron_les = np.random.choice(numNeuron,numNeuron_con, replace=False)
    model = {}
    if 'rnn' in lesion_method:
        model['rnn_les_con'] = set_zeroweight_rnn(rnn_model, Neuron_les)

    if 'output' in lesion_method:
        model['output_les_con'] = set_zeroweight_output(rnn_model, Neuron_les)
    
    return model
    
    
    
    
    
