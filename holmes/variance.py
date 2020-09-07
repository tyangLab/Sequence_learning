#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:36:57 2019

@author: Zhewei Zhang

test the activities variability 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import tkinter as tk
from tkinter import filedialog

from toolkits_2 import load, get_hidden_resp_all, get_bhvinfo
from selectivity import regress_resp


def variance(resp):
    var_ce = np.nanstd(resp, axis=2)
    return var_ce.T

def resp_grouping(resp_hidden, choice, rt):
    
    numTrials = len(resp_hidden)
    resp_group = {"choice_left":[],"choice_right":[],"start_all":[]}
    
    for nT in range(numTrials):
        resp = resp_hidden[nT]
        if rt[nT] < 6:
            continue
        if choice[nT] == 1.0:
            resp_group["choice_left"].append(resp[-30:,:])
        elif choice[nT] == -1.0:
            resp_group["choice_right"].append(resp[-30:,:])
        else:
            raise Exception ('unknown choice')
        resp_group["start_all"].append(resp[:28,:])

    return resp_group


def plot_variance(df_VarCE, df_signN):
    
    # combine the data from each file, and only the significant neurons 
    #   are included
    
    
    VarCE_s = np.vstack(df_VarCE.start_all)
    VarCE_choL  = np.vstack(df_VarCE.choice_left)
    VarCE_choR = np.vstack(df_VarCE.choice_right)

    signLeft  = np.hstack(df_signN.evid_pos)
    signRight = np.hstack(df_signN.evid_neg)
    # varinace CE, sorted according to their prefer choice
    VarCE_s = np.vstack((VarCE_s[signLeft, 3:], VarCE_s[signRight, 3:]))
    VarCE_Pref    = np.vstack((VarCE_choL[signLeft, :], VarCE_choR[signRight,:]))
    VarCE_NonPref = np.vstack((VarCE_choL[signRight,:], VarCE_choR[signLeft, :]))
    
    # rearrange; nEpoch by nNeuron by nTime in each epoch
    #  and get the mean averaged acorss the time point
    nNeuron, nTime = VarCE_s.shape[0], 5
    VarCE_s = VarCE_s.reshape(nNeuron, -1, nTime)[:,:,-2:].mean(axis=2)
    VarCE_Pref    = VarCE_Pref.reshape(nNeuron, -1, nTime)[:,:-1,-2:].mean(axis=2)
    VarCE_NonPref = VarCE_NonPref.reshape(nNeuron, -1, nTime)[:,:-1,-2:].mean(axis=2)
        
    fig = plt.figure()
    # VarCE aligned to the shape onset, and averaged acorss each neuron
    nEpoch = VarCE_s.shape[1]
    plt.errorbar(range(nEpoch),
                 VarCE_s.mean(axis=0), 
                 yerr  = stats.sem(VarCE_s, axis=0), 
                 label = 'start_all');
    
    nEpoch = VarCE_Pref.shape[1]
    plt.errorbar(np.arange(8,8+nEpoch,1),
                 VarCE_Pref.mean(axis=0), 
                 yerr  = stats.sem(VarCE_Pref, axis=0), 
                 label = 'choice_pref');
            
    plt.errorbar(np.arange(8,8+nEpoch,1),
                 VarCE_NonPref.mean(axis=0), 
                 yerr  = stats.sem(VarCE_NonPref, axis=0), 
                 label = 'choice_nonpref');

    plt.legend()
    fig.savefig('../figs/var_CE.eps', format='eps', dpi=1000)
    plt.show()
    

def variance_extract(file_paths):
    """
    calculating the variance of neurons that are selective to evidence
    
    """
    df_VarCE = pd.DataFrame([], columns = {'label',       'choice_left',
                                           'choice_right','start_all'})
    df_signN = pd.DataFrame([], columns = {'label','evid_pos','evid_neg'})

    for i, file in enumerate(file_paths):
        ## load files
        paod, trial_briefs = load(file)
        trial, choice, shape, _, _= get_bhvinfo(paod,trial_briefs)

        resp_hidden = get_hidden_resp_all(paod, trial_briefs)
    
        ## group the response
        resp_group = resp_grouping(resp_hidden, choice.left, shape.rt)
        
        # variance
        var_ce = {}
        for key, value in resp_group.items():
            var_ce[key] = variance(np.dstack(value))
        
        df_VarCE.loc[i] = {'label': file, 
                           'choice_left':  var_ce['choice_left'],
                           'choice_right': var_ce['choice_right'],
                           'start_all':    var_ce['start_all']
                           }
        
        results = regress_resp(resp_hidden, trial, choice, shape)
                
        # 128 neurons by 5 time point by 5 parameters (1 bias term + 4 factors)
        # index 1 represent evidence selectivity,
        params = np.array(results['params'])[:,:,1]
        pvalue = np.array(results['p_values'])[:,:,1]

        p_threshold = 0.05/pvalue.size # correction 
        
        np.warnings.filterwarnings('ignore')
        signNeuron = np.all(pvalue < p_threshold, axis =1)
        posNeuron  = np.all(params > -1e-10, axis =1)
        negNeuron  = np.all(params <  1e-10, axis =1)

        df_signN.loc[i] = {'label': file, 
                            'evid_pos': np.logical_and(signNeuron, posNeuron),
                            'evid_neg': np.logical_and(signNeuron, negNeuron)
                            }
        
    return df_VarCE, df_signN


def main():
    print("start")
#    %matplotlib auto
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
            parent=root,title='Choose a file',
            filetypes=[("HDF5 files", "*.hdf5")]
            )
    print("select the files")
    
    df_VarCE, df_signN = variance_extract(file_paths)
    
    plot_variance(df_VarCE, df_signN)

    

if __name__ == '__main__':
    main()

