#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:49:27 2019

@author: Zhewei Zhang

plot 1. the probability of choosing sure target
     2. the correct rate in the trials with/o sure targets
     3. the psth of neurons with choice prediction
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

import tkinter as tk
from tkinter import filedialog

from toolkits_2 import get_sureinfo, load, get_hidden_resp_sure

"""
representing the relation between the logRT and rt; 

fit the psychmetric curve

fig1 a/c
"""

coherence_list = [-51.2, -12.8, -6.4, -3.2, 0, 3.2, 6.4, 12.8, 51.2]
    

def selectivity_test(resp, trials):
    """
    whether neurons encode the motion direction during the motion viewing period
        or predict the choices in the no sure target trials
    """
    numTrials  = resp.shape[0]
    numNeurons = resp[0].shape[1]
    
    dots_dur = trials.randots_dur.values
    # actually, it is one step eariler than the the event happen, 
    # which are predictions, be careful about the time step
    cho_on = 4+dots_dur+5
    
    # trials choosing left/right when sure target is not available
    nosure_left = np.intersect1d(np.where(trials.choice==1), 
                                 np.where(trials.sure_trial==0)) 
    nosure_right = np.intersect1d(np.where(trials.choice==2),
                                  np.where(trials.sure_trial==0))  
    
    choice_sel = {'pos':[],'neg':[]}
    for nNeuron in range(numNeurons):
        # group trials according to choices
        rvs_c1, rvs_c2 = [], []
        for nT in range(numTrials):
            if nT in nosure_left:
                rvs_c1.append(resp[nT][cho_on[nT], nNeuron])
            elif nT in nosure_right:
                rvs_c2.append(resp[nT][cho_on[nT], nNeuron])
        # compare the neural response
        rvs_c1, rvs_c2 = np.array(rvs_c1), np.array(rvs_c2)        
        t, p = stats.ttest_ind(rvs_c1, rvs_c2)
        if p < 0.05/(2*128):
            if rvs_c1.mean() > rvs_c2.mean():
                choice_sel['pos'].append(nNeuron)
            else:
                choice_sel['neg'].append(nNeuron)

    return choice_sel
        

def psth(resp_all, df_detail, Neurons=None, mean=False, label=''):
    """
    plot the psth of Neurons, and trials are sorted by the choice and the 
        appearance of sure target
    """    
    if np.all(Neurons == None):
        Neurons = range(numNeurons)
    # the trials with random dots duration smaller than min_motion are excluded
    min_motion = 4
    # number of files and neurons
    numFile = len(resp_all)
    numNeurons = resp_all[0][0].shape[1]
    
    if numFile ==1:
        path, file = os.path.split(file_paths[0])
        task_time = file.split('-')[0]
    else:
        task_time = 'combined'

    resp_plot = {'sure':    {'left': {'mov':[],'cho':[]}, 
                             'right':{'mov':[],'cho':[]}, 
                             'sure_left': {'mov':[],'cho':[]},
                             'sure_right':{'mov':[],'cho':[]}},
                 'no_sure': {'left':{'mov':[],'cho':[]}, 
                             'right':{'mov':[],'cho':[]}}
                 }
    
    for resp, trials, neurons_curr in zip(resp_all, df_detail, Neurons):
        i = 0
        numTrials = resp.shape[0]
        dots_dur = trials.randots_dur.values.astype(np.int)
        num_tartrials = np.sum(dots_dur>=min_motion)
        # neural response
        resp_mov = np.zeros((neurons_curr.size, np.sum(num_tartrials), 6))
        resp_cho = np.zeros((neurons_curr.size, np.sum(num_tartrials), 7))
        # time point of events happening
        moive_on, moive_off, cho_on = 5, 5+dots_dur, 5+dots_dur+5
        # conditions and choices
        choice    = trials["choice"][    dots_dur>=min_motion].values
        coherence = trials["coherence"][ dots_dur>=min_motion].values
        sure_trial= trials["sure_trial"][dots_dur>=min_motion].values
        
        no_sure = {'left' : np.logical_and(sure_trial==0, choice==1),
                   'right': np.logical_and(sure_trial==0, choice==2)}
        
        sure = {'left'      : np.logical_and(sure_trial==1, choice==1),
                'right'     : np.logical_and(sure_trial==1, choice==2),
                'sure_left' : np.logical_and(coherence>0  , choice==3),
                'sure_right': np.logical_and(coherence<0  , choice==3)}

        for nT, dur in zip(range(numTrials), dots_dur):
            if dur < min_motion:
                    continue
            for nth, nNeuron in enumerate(neurons_curr):
                resp_mov[nth,i,:] = resp[nT][moive_on-2:moive_on+4 ,nNeuron]
                resp_cho[nth,i,:] = resp[nT][cho_on[nT]-5:cho_on[nT]+2 ,nNeuron]
            i += 1
        
        for indexs, label_ in zip([sure, no_sure],['sure', 'no_sure']):
            for key, value in indexs.items():
                resp_mov_curr = resp_mov[:,value,:].reshape(-1, 6)
                resp_cho_curr = resp_cho[:,value,:].reshape(-1, 7)
                resp_plot[label_][key]['mov'].append(resp_mov_curr)
                resp_plot[label_][key]['cho'].append(resp_cho_curr)
    
    # plot setting
    colorset = ['r','g','c','y']

    for labels, value__s in resp_plot.items():
        title_label = 'averaged-' + labels + '-' + label
        fig = plt.figure()
        i=0
        for key, value in value__s.items():
            resp_mov = np.concatenate(value['mov'])
            resp_cho = np.concatenate(value['cho'])
            plot_errorbar(range(6), resp_mov, color = colorset[i], label = key)
            plot_errorbar(np.linspace(7,13,7), resp_cho, color = colorset[i])
            i+=1
    
        plt.axvline(2,linestyle='-.',color='k',label = 'moive onset')
        plt.axvline(12,linestyle='-.',color='k',label = 'choice')
        plt.legend()
        plt.title(title_label)
        fig.savefig(savepath + title_label + '.eps', format='eps', dpi=1000)
        plt.show()
    

def plot_errorbar(x, response,  color = 'b', label=None):
    plt.errorbar(x, np.nanmean(response,axis=0), 
                 np.nanstd(response, axis=0)/np.sqrt(response.shape[0]),
                 color = color,
                 label = label)
    

def data_extract(file_paths):
    """
    
    """
    df_detail = []
    df_summary = pd.DataFrame([], columns = {'choice'})
    sign_neurons = {'pos_choice':[], 'neg_choice':[]}
    n_resp_all = []
    for i, file in enumerate(file_paths):

        paod, trial_briefs = load(file)
        trials = get_sureinfo(paod,trial_briefs)

        files_pd = pd.DataFrame([trials["choice"],trials["reward"],trials["randots_dur"],
                                 trials["sure_trial"],trials["coherence"]],
                                ['choice','reward','randots_dur','sure_trial','coherence'])
        files_pd = files_pd.T
        df_detail.append(files_pd)
        
        choice =[]
        for ii in coherence_list:
            choice.append([
                    ii,
                    np.where(trials["choice"][trials["coherence"]==ii]== 1)[0].shape[0],
                    np.where(trials["choice"][trials["coherence"]==ii]== 2)[0].shape[0],
                    np.where(trials["choice"][trials["coherence"]==ii]== 3)[0].shape[0]
                    ])
        choice = np.array(choice)
        df_summary.loc[i] = {'choice':choice}        

        # response of neurons in hidden layer
        n_resp = get_hidden_resp_sure(paod, trial_briefs) 
        n_resp_all.append(n_resp)
        # test the choice selectivity of each neuron 
        choice_sel = selectivity_test(n_resp, trials)
        sign_neurons['pos_choice'].append(np.array(choice_sel['pos']))
        sign_neurons['neg_choice'].append(np.array(choice_sel['neg']))

    return df_detail, df_summary, n_resp_all, sign_neurons


def bhv_plot(df_detail, savepath = './'):

    prob_sure_all, cr_w_all, cr_o_all = [], [], []
    for files_pd in df_detail:
        # the list of possible coherence and duration of rdm dots
        coherence_list = np.unique(files_pd["coherence"]).tolist()
        randots_dur_list = np.unique(files_pd["randots_dur"]).tolist()

        # choices sorted by coherence and duration
        choice_CohDur = files_pd.choice.groupby([files_pd.coherence, 
                                                 files_pd.randots_dur])
        # prob of choosing sure target
        prob_sure = np.zeros((len(coherence_list), len(randots_dur_list)))
        for i, coh_dur in enumerate(choice_CohDur):
            coh = coh_dur[0][0]
            dur = coh_dur[0][1]
            x = np.where(np.array(coherence_list) == coh)[0][0]
            y = np.where(np.array(randots_dur_list) == dur)[0][0]
            
            prob_sure[x,y] = np.mean(coh_dur[1]==3)
        
        # choices sorted by coherence, duration and with/o sure target
        choice_CohDurSure = files_pd.choice.groupby([files_pd.coherence, 
                                                      files_pd.randots_dur,
                                                      files_pd.sure_trial])
        # correct rate in the trials w/o sure target
        cr_w = np.zeros((len(coherence_list), len(randots_dur_list)))
        cr_o = np.zeros((len(coherence_list), len(randots_dur_list)))
        
        for i, coh_dur_sure in enumerate(choice_CohDurSure):
            coh  = coh_dur_sure[0][0]
            dur  = coh_dur_sure[0][1]
            sure = coh_dur_sure[0][2]
            choice = coh_dur_sure[1]
            x = np.where(np.array(coherence_list)   == coh)[0][0]
            y = np.where(np.array(randots_dur_list) == dur)[0][0]
            numTrials = np.sum(choice==1) + np.sum(choice==2)
            if sure:
                cr_w[x,y] = np.sum(choice==1)/numTrials if coh>0 else np.sum(choice==2)/numTrials
            else:
                cr_o[x,y] = np.sum(choice==1)/numTrials if coh>0 else np.sum(choice==2)/numTrials

        cr_w_all.append(cr_w)
        cr_o_all.append(cr_o)
        prob_sure_all.append(prob_sure)

    cr_w_all = np.array(cr_w_all)
    cr_o_all = np.array(cr_o_all)
    prob_sure_all = np.array(prob_sure_all)

    color = ['g','r','y','c','b','m','k']
    fig = plt.figure('probability sure target')
    for i, coh in enumerate(coherence_list):
        if i <5:
            prob_sure_all[:,10-i,:] = (prob_sure_all[:,i,:]+prob_sure_all[:,10-i,:])/2
        else:
            plt.errorbar([0,1,2,3,5,7], np.mean(prob_sure_all[:,i,:], axis=0), 
                         yerr = stats.sem(prob_sure_all[:,i,:], axis=0),
                         fmt= '-', color = color[10-i], label = np.abs(coh))
    plt.xticks([0,1,2,3,5,7], {100,200,300,400,600,800})
    plt.ylabel('probability sure target')
    plt.legend()
    fig.savefig(savepath+'/prob_sure.eps', format='eps', dpi=1000)
    
    fig2 = plt.figure('probability correct')
    for i, coh in enumerate(coherence_list):
        if i<5:
            cr_w_all[:,10-i,:] = (cr_w_all[:,i,:]+cr_w_all[:,10-i,:])/2
            cr_o_all[:,10-i,:] = (cr_o_all[:,i,:]+cr_o_all[:,10-i,:])/2
        else:
            plt.errorbar([0,1,2,3,5,7], np.mean(cr_w_all[:,i,:], axis=0), 
                         yerr = stats.sem(cr_w_all[:,i,:], axis=0),
                         fmt='-', color = color[10-i], label = np.abs(coh))
            plt.errorbar([0,1,2,3,5,7], np.mean(cr_o_all[:,i,:], axis=0),
                         yerr = stats.sem(cr_o_all[:,i,:], axis=0),
                         fmt='-.', color = color[10-i])
    plt.legend()
    plt.ylabel('probability correct')
    plt.xticks([0,1,2,3,5,7], {100,200,300,400,600,800})
    fig2.savefig(savepath +'/prob_cr.eps', format='eps', dpi=1000)
    
    plt.show()



def main(file_paths=None):
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    if file_paths == None:
        # file_paths = filedialog.askopenfilenames(parent = root,
        #                                         title = 'Choose a file',
        #                                         filetypes = [("HDF5 files", "*.hdf5")]
        #                                         )
        file_paths = glob.glob('../log/Sure/*.hdf5')

    ##
    df_detail, df_summary, n_resp_all, sign_neurons = data_extract(file_paths)


    psth(n_resp_all, df_detail, Neurons = sign_neurons['pos_choice'], 
         label = 'pos_choice')

    psth(n_resp_all, df_detail, Neurons = sign_neurons['neg_choice'], 
         label = 'neg_choice')
    bhv_plot(df_detail, savepath = savepath)

    return df_summary

if __name__ == '__main__':
    savepath = '../figs/'
    main()
