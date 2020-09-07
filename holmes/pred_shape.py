# -*- coding: utf-8 -*-
# @Author: Zhewei Zhang
# @Date: 09/01/2018, 23:39

"""
preditive coding about the visual information analysis
"""

import numpy as np
import pandas as pd
from scipy import stats

import tkinter as tk
from tkinter import filedialog

from toolkits_bhv import shape_config
from toolkits_2 import get_bhvinfo,load

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


"""
using the subthreshold activity of the shape output units to predict the 
next presenting shape

fig5
"""


WEIGHTS = (-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9)
prob_L = np.array((0.0056, 0.0166, 0.0480, 0.0835, 0.1771, 
                   0.2229, 0.1665, 0.1520, 0.0834, 0.0444))
prob_R = np.flipud(prob_L)
time_len = 10

input_setting = shape_config()
shapes_pos = input_setting['shapes']
reward_pos = input_setting['reward']
choice_pos = input_setting['choice']

def shape_prediction(path_files):
    
    df_pred = pd.DataFrame([], columns = {'label','dist_neuro',
                           'weight','kl_div'})

    for i, file in enumerate(path_files):
        ## load files
        paod, trial_briefs = load(file)
        trial, _, shape, _, _ = get_bhvinfo(paod,trial_briefs)
        numTrials = shape.rt.values
        #======================== shape prediction ========================
        dist_neuro, weight = [], []
        for i, rt in enumerate(numTrials):
            _, _, rd, _= paod.get_neuron_behavior_pair(index = trial.num.iloc[i])
            for j in range(rt):
                #  the shapes in all the epoches are included
                weight.append(shape.tempweight.iloc[i][:j].sum())
                dist_neuro.append(rd[2 + j * 5, shapes_pos])            
        
        # the sum weight before each shape onset
        weight = np.array(weight)
        # the response of neurons before the shape onset
        dist_neuro = np.asarray(dist_neuro)
        
        sega = np.linspace(weight.min(),weight.max(),num=11, endpoint=True)        
        kl_div = []
        for n_group in range(len(sega)-1):
            index = (weight >= sega[n_group]) & (weight < sega[n_group+1])
            if n_group == len(sega)-1-1:
                index = (weight >= sega[n_group]) & (weight <= sega[n_group+1])
                
            pred_dist = np.mean(dist_neuro[index], axis=0)
            pred_dist = pred_dist/np.sum(pred_dist)
            
            kl_div.append([stats.entropy(pred_dist, qk = prob_L),
                           stats.entropy(pred_dist, qk = prob_R)])
            
        df_pred.loc[i] = {'label' : file,  'weight': weight,
                          'kl_div': np.array(kl_div),
                          'dist_neuro': dist_neuro
                          }
    return df_pred
            

def plot_shapePrediction(df_pred):
    
    # plot the KL divergence between the sampling distirbution and 
    # neural activities
    kl_div = np.hstack(df_pred.kl_div.values)

    fig = plt.figure()
    plt.errorbar(range(kl_div.shape[0]), 
                 np.nanmean(kl_div[:,0::2], axis = 1),
                 yerr = stats.sem(kl_div[:,0::2], axis = 1),
                 label = 'left')
    plt.errorbar(range(kl_div.shape[0]), 
                 np.nanmean(kl_div[:,1::2], axis = 1),
                 yerr = stats.sem(kl_div[:,1::2], axis = 1), 
                 label = 'right')
    plt.title('KL divergence')
    
    plt.legend()
    plt.xticks(np.linspace(0,kl_div.shape[0]-1,10),
               ('0~10%','10~20%','20~30%','30~40%','40~50%',
               '50~60%','60~70%','70~80%','80~90%','90~100%'))
    plt.xlabel('sorted by the summed weight ')
    plt.ylabel('kl divergence')
    fig.savefig('../figs/kl divergence.eps', format='eps', dpi=1000)
    

    # plot the neural activities along with the sum weight

    weight = np.hstack(df_pred.weight.values)
    # sort the neural activities by the magnitude of weight
    sort_index = np.argsort(weight)
    dist_neuro = np.vstack(df_pred.dist_neuro.values)
    dist_neuro = dist_neuro[sort_index,:]

    X = np.linspace(0, 1, endpoint=True, num = dist_neuro.shape[1])
    Y = np.arange(1, dist_neuro.shape[0]+1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = dist_neuro

    X = np.hstack((X, (2*X[:,-1] - X[:,-2]).reshape(-1,1)))
    Y = np.hstack((Y, Y[:,-1].reshape(-1,1)))
    Z = np.hstack((Z, Z[:,-1].reshape(-1,1)))

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d') 
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.view_init(azim=1, elev=270)
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('shape weight')
    plt.ylabel('predicted distribution')
    plt.title('neuronal response')
    plt.xticks(np.linspace(0,1,num=10,endpoint = True),np.arange(-0.9,1,0.2))
    fig2.savefig('../figs/shape prediction.eps', format='eps', dpi=1000)
    plt.show()



def main():
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(parent = root,
                                            title = 'Choose a file',
                                            filetypes = [("HDF5 files", "*.hdf5")]
                                            )
    ##
    fig_w, fig_h = (10, 4)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    # =========================================================================
    df_pred = shape_prediction(file_paths)
    plot_shapePrediction(df_pred)



if __name__ == '__main__':
    main()

