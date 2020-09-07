#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:11:24 2019

@author: Zhewei Zhang
    
plotting psth and sort trials by the finnal choice and logLR

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import tkinter as tk
from tkinter import filedialog

from toolkits_2 import load, get_bhvinfo, get_hidden_resp_all
from selectivity import regress_resp



def list2array(*kwargs):
    re = []
    for var in kwargs:
        re.append(np.array(var))
    return re

def sort(resp, label):
    numGroup = 4
    resp, label = list2array(resp, label)
    min_, max_ = np.min(label), np.max(label)

    resp_group = []
    for ngroup in range(numGroup):
        lowB  = min_ + (max_ - min_)/numGroup*ngroup
        highB = max_ - (max_ - min_)/numGroup*(numGroup-ngroup-1)
        lowIndex  = np.where(label >= lowB) 
        highIndex = np.where(label <= highB)
        index = np.intersect1d(lowIndex, highIndex)
        if not index.size:
            continue
        resp_group.append(resp[index,:,:].mean(axis=0))
    return np.array(resp_group).transpose([2,0,1])

def psth_align(respH, choice, shape, unitUrg, unitEvi):
    '''
    prepare the response for plotting
    '''
    numTrials = choice.count().status
    numUnits  = respH[0].shape[1]
    
    ## urgency signal
    nshape_max = 18 
        
    resp_Nshape = {'left' : np.zeros((numUnits, nshape_max)), 
                   'right': np.zeros((numUnits, nshape_max))}
    # average the response of each neuron in each epoch and sorted based on 
    #   final choices
    for nshape in range(nshape_max):
        # loop over epochs
        start_time, end_time = 3+5*nshape, 3+5*(1+nshape)
        resp_left, resp_right = [], []
        for i in range(numTrials):
            # iterate over trials
            if shape.rt[i] < nshape:
                continue
            if choice.left[i] == 1.0:
                resp_left.append(respH[i][ start_time:end_time,:].mean(axis=0))
            elif choice.left[i] == -1.0:
                resp_right.append(respH[i][start_time:end_time,:].mean(axis=0))
            else:
                raise Exception('unknown choices')
        resp_Nshape['left' ][:,nshape] = np.mean(np.array(resp_left),  axis=0)
        resp_Nshape['right'][:,nshape] = np.mean(np.array(resp_right), axis=0)

    # the responses of neurons with selectivity
    resp_urg = {'pos':{'left':[],'right':[]}, 'neg':{'left':[],'right':[]}}
    for label, Neurons in unitUrg.items():
        for neuron  in Neurons:
            resp_urg[label]['left' ].append(resp_Nshape['left' ][neuron,:])
            resp_urg[label]['right'].append(resp_Nshape['right'][neuron,:])

    # resp_pos_urg = np.array(resp_pos_urg).transpose(1,2,0)
    
# =============================================================================
    # numTime stands for the number of time point in each epoch    
    numEpoch, numGroup, numTime = 3, 4, 5
    
    resp = {'start' : np.zeros((numUnits, numEpoch, numGroup, numTime*2)),
            'choice': np.zeros((numUnits, numEpoch, numGroup, numTime*2))}
    # align the response of each neuron to the first shape onset and choice
    #   Then group the response based the sum weight
    for nepoch in range(numEpoch):
        ind_S1, ind_S2 =  3+numTime*nepoch, 3+numTime*(1+nepoch)
        ind_C1, ind_C2 = -6-numTime*(nepoch+1)+1, -6-numTime*nepoch+1
        if nepoch == 0:
            ind_S1, ind_C2 = 0, -1
        # get the reponses in first three/last numEpochs
        respStart,  respChoice  = [], []
        AcEviStart, AcEviChoice = [], []
        for ntrial in range(numTrials):
            nshape  = shape.rt.iloc[ntrial]
            tempEvi = shape.tempweight.iloc[ntrial]
            if nshape < numEpoch*2:
                continue
            respStart.append(respH[ntrial][ind_S1:ind_S2,:])
            AcEviStart.append(tempEvi[:nepoch+1].sum())
            
            respChoice.append(respH[ntrial][ind_C1:ind_C2,:])
            AcEviChoice.append(tempEvi[:nshape-nepoch].sum())
                                             
        # sort trials
        num1, num2 = ind_S2-ind_S1, ind_C2-ind_C1
        resp['start' ][:,nepoch,:,:num1] = sort(respStart,  AcEviStart)
        resp['choice'][:,nepoch,:,:num2] = sort(respChoice, AcEviChoice)
    
    resp_evi = {'start':[],'choice':[]}
    # the responses of neurons with selectivity
    for label, Neurons in unitEvi.items():
        for neuron  in Neurons:
            if label == 'pos':
                resp_evi['start' ].append(resp['start' ][neuron])
                resp_evi['choice'].append(resp['choice'][neuron])
            elif label == 'neg':
                resp_evi['start' ].append(resp['start' ][neuron,:,::-1,:])
                resp_evi['choice'].append(resp['choice'][neuron,:,::-1,:])
    
    return resp_urg, resp_evi

def psth_plot(psth_resp, savepath = './'):
    # psth_resp: a list; 
    #   psth_resp[i][0]: sort by the urgency signal, averaged over epochs
    #   psth_resp[i][1]: sort by the sum weight/accumulated evidence
    
    # plot the psth sorted by choice, neurons that are selective to urgency 
    #    signal are included
    psth_urg = {'pos':{'left':[],'right':[]}, 'neg':{'left':[],'right':[]}}
    urgency_pos, urgency_neg = [], []
    for psth in psth_resp:
        for sign in ['pos', 'neg']: # 
            for choice in ['left', 'right']:
                psth_urg[sign][choice].append(psth[0][sign][choice])
                
    nshape = range(psth_urg[sign][choice][0][0].size)    
    
    plt.figure()
    plt.title('urgency signal')
    colorset = ['r','m','c','g']
    for sign in ['pos', 'neg']: # 
        for choice in ['left', 'right']:
            value = np.concatenate(psth_urg[sign][choice])
            plt.errorbar(nshape, 
                         np.nanmean(value, axis=0),
                         yerr = stats.sem(value, axis=0 ,nan_policy = 'omit'), 
                         color = colorset.pop(0),
                         label = sign +'- choosing '+ choice)

    plt.legend()
    plt.xticks([0,4,9,14,19],('1','5','10','15','20'))
    plt.savefig('../figs/psth_urgency.eps', format='eps', dpi=1000)
    plt.show()
    
    # plot the psth sorted by sum weight, only neurons that are selective to it 
    #    are included
    psth_evi = {'start':[],'choice':[]}
    for psth in psth_resp:
        for epoch in ['start','choice']:
            psth_evi[epoch].append(psth[1][epoch])

    fig2 = plt.figure()
    ind_S1, ind_S2 = [0, 8, 13], [8, 13, 18] 
    ind_C1, ind_C2 = [35, 30, 25], [44, 35, 30]
    nT_start, nT_choice = [8,5,5], [9,5,5]
    colorset = ['r','m','c','g']
    for nepoch in range(3):
        for ngroup in range(4):
            # plot the psth align to the shape onset, in each group/epoch
            start_time, end_time = ind_S1[nepoch], ind_S2[nepoch]
            psth_start = np.concatenate(psth_evi['start'])
            psth_start = psth_start[:,nepoch, ngroup,:nT_start[nepoch]]
            plt.errorbar(np.arange(start_time,end_time,1), 
                         psth_start.mean(axis=0),
                         yerr = stats.sem(psth_start, axis=0), 
                         color = colorset[ngroup])
            # plot the psth align to the choice, in each group/epoch
            start_time, end_time = ind_C1[nepoch], ind_C2[nepoch]
            psth_choice = np.concatenate(psth_evi['choice'])
            psth_choice = psth_choice[:,nepoch, ngroup,:nT_choice[nepoch]]
            plt.errorbar(np.arange(start_time, end_time,1), 
                         psth_choice.mean(axis=0),
                         yerr = stats.sem(psth_choice, axis=0), 
                         color = colorset[ngroup])

    plt.title('PSTH - log LR')
    plt.axvline(3,linestyle='-.',color='k',label = 'shape on')
    plt.axvline(39,linestyle='-.',color='k',label = 'choice')
    bottom, top = plt.ylim()
    plt.ylim([bottom, 1])
    plt.legend()
    fig2.savefig('../figs/psth-choice.eps', format='eps', dpi=1000)
    plt.show()

def psth_extract(path_file):

    psth_resp = []
    unitUrg = {'pos':[],'neg':[]}
    unitEvi = {'pos':[],'neg':[]}
    findPos = lambda x, y: np.where(np.all(x > y, axis =1))[0]
    findNeg = lambda x, y: np.where(np.all(x < y, axis =1))[0]
    
    for i, file in enumerate(path_file):
        # load behavioral data
        paod, trial_briefs = load(file)
        trial, choice, shapes, _, _ = get_bhvinfo(paod, trial_briefs)
        
        # load neural response
        resp_hidden = get_hidden_resp_all(paod,trial_briefs)
        
        for resp in resp_hidden:
            np.any(np.isnan(resp))
        
        # neuronal selectivity test
        results = regress_resp(resp_hidden, trial, choice, shapes)
        
        # find units that are selective to evidence/urgency
        # 128 neurons by 5 time point by 5 parameters (1 bias term + 4 factors)
        params  = np.array(results['params'])
        p_value = np.array(results['p_values'])
        
        threshold = 0.05/p_value.size
        np.warnings.filterwarnings('ignore')
        signUrg = np.where(np.all(p_value[:,:,3] < threshold, axis =1))[0]
        signEvi = np.where(np.all(p_value[:,:,4] < threshold, axis =1))[0]
        
        # neurons with positive/negative selectivity to urgency and evidence
        unitUrg['pos'] = np.intersect1d(signUrg, findPos(params[:,:,3],-1e-10))
        unitUrg['neg'] = np.intersect1d(signUrg, findNeg(params[:,:,3], 1e-10))
        
        unitEvi['pos'] = np.intersect1d(signEvi, findPos(params[:,:,4],-1e-10))
        unitEvi['neg'] = np.intersect1d(signEvi, findNeg(params[:,:,4], 1e-10))
        
        resp_urg, resp_evi = psth_align(resp_hidden, choice, shapes,
                                        unitUrg, unitEvi)
        psth_resp.append([resp_urg, resp_evi])
        
    return psth_resp


def main():
    print("start")
#    %matplotlib auto
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames(
            parent=root,title='Choose a file',
            filetypes=[("HDF5 files", "*.hdf5")]
            )
    print("select the files")
    #
    psth_resp = psth_extract(file_path)
    psth_plot(psth_resp)
    
if __name__ == '__main__':
    fig_w, fig_h = (10, 7)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    main()


















