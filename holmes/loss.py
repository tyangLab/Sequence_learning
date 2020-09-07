#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:46:11 2020

@author: Zhewei Zhang
"""


import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


testTrialNum = 5000 
def plot_loss(file_paths):
    # file_paths = glob.glob('/home/tyang/Desktop/temp/*shape_lr.csv')
    # file_paths = np.sort(glob.glob('../log/RT/GRU/*shape_rt.csv'))
    cr_train, loss_train = [], []
    cr_test , loss_test  = [], []
    for file_path in file_paths:
        file_path = file_path[:-4] + 'csv'
        obj = pd.read_csv(file_path)
        
        allTrialNum = obj.correct_rate.shape[0]
        trainTrialNum = allTrialNum - testTrialNum
        
        cr   = obj.correct_rate.values[:-testTrialNum] # last 5000 trials are test trials
        
        loss = obj.loss.values[:-testTrialNum]
    
        cr_train.append(cr)
        loss_train.append(loss)
    
        cr_test.append(obj.correct_rate.values[-testTrialNum:])
        loss_test.append(obj.loss.values[-testTrialNum:])
    
    cr_test,  loss_test  = np.array(cr_test),  np.array(loss_test)
    cr_train, loss_train = np.array(cr_train), np.array(loss_train)
    
    fig = plt.figure()
    plt.subplot(211)
    plt.errorbar(range(trainTrialNum), cr_train.mean(axis=0), stats.sem(cr_train, axis=0),label='loss')
    plt.errorbar(trainTrialNum*1.1, cr_test.mean(), stats.sem(cr_test.reshape(-1)), fmt = 'k*', MarkerSize = 10)
    plt.ylabel('correct rate')
    plt.subplot(212)
    plt.errorbar(range(trainTrialNum), loss_train.mean(axis=0), stats.sem(loss_train, axis=0),label='loss')
    plt.errorbar(trainTrialNum*1.1, loss_test.mean(), stats.sem(loss_test.reshape(-1)), fmt = 'k*', MarkerSize = 10)
    plt.yscale('log')
    plt.ylabel('loss (log scale)')
    fig.savefig('../figs/loss.eps', format='eps', dpi=1000)
    







