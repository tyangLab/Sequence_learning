#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:36:20 2019

@author: Zhewei Zhang

plot the psychometric curve and threshold in multisensory integration tasl
"""
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from toolkits_2 import get_multinfo, load



def cum_gaussian(x, x0, k):
    return stats.norm.cdf((x-x0)/k)

def psych_curve(direction, choice, modality):
    """
    fitting with a cumulative Gaussian function
    return: fitted parameters ()
    """
    fit_params = []
    for i in np.unique(modality):
        trial = np.where(modality == i)[0]
        prpt, pcov = curve_fit(cum_gaussian, direction[trial], 2-choice[trial])
        fit_params.append(prpt)
    return fit_params


def data_extract(file_paths):
    """
    
    """
    df_detail = []
    # theo denotes the bayesian inference
    df_summary = pd.DataFrame([], columns = {'cho_prob','prpt','prpt_Baye'})
    for nth, file in enumerate(file_paths):

        paod, trial_briefs = load(file)
        trials = get_multinfo(paod,trial_briefs)
        files_pd = pd.DataFrame([trials["choice"],trials["reward"],
                                 trials["chosen"], trials["modality"],
                                 trials["direction"],trials["estimates"]],
                                ['choice','reward','chosen',
                                 'modality','direction','estimates']
                                )
        files_pd = files_pd.T
        df_detail.append(files_pd)
        # choice probability in each directions and modalities
        cho_prob = [[],[],[]]
        for i in np.unique(trials["direction"]):
            trial = np.where(trials["direction"]==i)[0]
            for ii in range(3):
                temp = np.intersect1d(trial, np.where(trials["modality"]==ii))
                cho_prob[ii].append(np.mean(trials["choice"][temp]==1))
        
        modality = trials["modality"]
        Baye_choice = np.diag(np.vstack(trials["estimates"].values)[:,modality])+1
        
        # the std of fitted Gaussian curve is viewed as threshold 
        prpt = psych_curve(trials["direction"], trials["choice"], trials["modality"])
        prpt_Baye = psych_curve(trials["direction"], Baye_choice, trials["modality"])
        df_summary.loc[nth] = {'cho_prob':cho_prob,'prpt':prpt,'prpt_Baye':prpt_Baye}
    return df_detail, df_summary
    

def bhv_plot(df_summary):
    fig_w, fig_h = (10, 7)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    
    numfiles = df_summary.cho_prob.values.shape[0]
    cho_prob = np.array([df_summary.cho_prob.values[i] for i in range(numfiles)])
    
    # psychometric curve
    fig1 = plt.figure('choices') 
    x = np.arange(1,181,1)
    labels = ['modality 1','modality 2','combined']
    color = ['r','g','k']
    
    x_ = np.tile(x, len(df_summary))   
    for i in range(3):
        y_  = np.hstack(cho_prob[:,i,:].reshape(-1,))
        y_[np.where(np.isnan(y_))]=1
        prpt, pcov = curve_fit(cum_gaussian, x_, y_)
        plt.plot(np.mean(cho_prob[:,i,:], axis=0), '.', label = labels[i], color = color[i])
        plt.plot(x, cum_gaussian(x, prpt[0], prpt[1]), color = color[i])
    fig1.legend()
    plt.title('psychmetric curve')
    plt.xlabel('motion direction')        
    plt.ylabel('probability of choosing left')        
    fig1.savefig('../figs/MI-psych_curve.eps', format='eps', dpi=1000)
    plt.show()
    
    
    # behavior threshold (std) in each modalities
    prpt = np.vstack([(df_summary.prpt.values[i]) for i in range(numfiles)])
    prpt_Baye = np.vstack([(df_summary.prpt_Baye.values[i]) for i in range(numfiles)])
    fig2 = plt.figure('threshold') 
    
    thres_vi, thres_ve, thres_bi = prpt[::3,1], prpt[1::3,1], prpt[2::3,1]
    plt.bar([0,2,4], 
            [np.mean(thres_vi), np.mean(thres_ve), np.mean(thres_bi)],
            yerr = [stats.sem(thres_vi), stats.sem(thres_ve), stats.sem(thres_vi)],
            label = 'model')
    
    thres_vi, thres_ve, thres_bi = prpt_Baye[::3,1], prpt_Baye[1::3,1], prpt_Baye[2::3,1]
    plt.bar([1,3,5],
            [np.mean(thres_vi), np.mean(thres_ve),np.mean(thres_bi)],
            yerr = [stats.sem(thres_vi), stats.sem(thres_ve), stats.sem(thres_vi)],
            label = 'bayesian')
    
    plt.legend()
    plt.ylabel('threshold ')
    plt.xticks([0.5,2.5,4.5], {'visual','vestibular','combined'})
    fig2.savefig('../figs/MI-threshold.eps', format='eps', dpi=1000)
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
        file_paths = glob.glob('../log/MI/*.hdf5')
    ##
    df_detail, df_summary = data_extract(file_paths)
    bhv_plot(df_summary)
    return df_summary

if __name__ == '__main__':
    savepath = '../figs/'
    sel_psth = False
    df_summary = main()
