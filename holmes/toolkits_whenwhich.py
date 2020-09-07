# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:16:38 2018

@author: Zhewei Zhang


find the when and which neurons in the hidden layer of the network
"""

## load model
import os
import yaml
import copy
import glob
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

import tkinter as tk
from tkinter.filedialog import askopenfilenames as file_selection

from toolkits_bhv import shape_config

path = os.getcwd()
os.chdir('../seqrnn_multitask')
from model import GRUNetwork
os.chdir(path)

numNeuron = 128
input_setting = shape_config()
choice_pos = input_setting['choice']
input_size = input_setting['input_size']


# In[]: tools finding the when and which units
class neurons_inNetwork:
    """
    
    """
    def __init__(self, value, threshold):
        self.threshold = threshold
        self.value = value
        self.pos = []
        self.neg = []
        self.neuron_constructer()
            
    def __getitem__(self,key):
        return self.dict[key]
    
    def __setitem__(self,key,value):
        self.dict[key] = value
        
    def neuron_constructer(self):
        index = np.argsort(self.value)[::-1]
        # Get neurons with positive values and their sum is more than a half of 
        #  the sum of positive values. In the mean time, we minmize the number 
        #  the neurons
        pos_v = self.value[np.where(self.value > 0)]
        pos_v_cumsum = np.sort(pos_v)[::-1].cumsum()
        necePosProportion = self.threshold*pos_v.sum()
        numPosSelected = np.where(pos_v_cumsum >= necePosProportion)[0][0]+1
        self.pos = index[:numPosSelected]
            
        # Get neurons with positive values and their sum is more than a half of 
        #  the sum of positive values. In the mean time, we minmize the number 
        #  the neurons
        neg_v_abs = np.abs(self.value[np.where(self.value < 0)])
        neg_v_cumsum = np.sort(neg_v_abs)[::-1].cumsum()
        neceNegProportion = self.threshold*neg_v_abs.sum()
        numNegSelected = np.where(neg_v_cumsum >= neceNegProportion)[0][0]+1
        self.neg = index[-numNegSelected:]

def model_select():
    """
     GUI , select the model files
    """
    root = tk.Tk()
    root.withdraw()
    file_paths = file_selection(initialdir = '../save/RT',
                               parent=root, 
                               title='Choose the model',
                               filetypes=[("model files", ".pt")]
                               )
    return file_paths

def model_loading(file_path):
    # load the model parameters
    # it doesn't matter whether it is true setting 
    model_parameters = {
                        "nhid": 128,
                        "nlayers": 1,
                        "input_size": input_size,
                        "batch_size": 1,
                        "clip": 0.25,
                        "lr": 0.6
                        }
    # load model
    rnn_model = GRUNetwork(model_parameters["input_size"],
                           model_parameters["nhid"],
                           model_parameters["batch_size"],
                           model_parameters["nlayers"],
                           model_parameters["lr"],
                           )
    rnn_model.load_state_dict(torch.load(file_path))
    
    # get the connection weight of the trained model
    ho_weight = rnn_model.decoder.weight.detach().numpy()
    w_hr, w_hi, w_hn = rnn_model.rnn.weight_hh_l0.chunk(3, 0)
    
    return ho_weight, w_hr, w_hi, w_hn
    
    
def when_which_constructer(ho_weight, neuron_proportion): 
    """
    when neuron: when should I make a choice
    which neuron: which target should I choose
    """
    when_value  = ho_weight[choice_pos[1]:choice_pos[-1],:].mean(axis = 0) - ho_weight[choice_pos[0],:]    
    which_value = ho_weight[choice_pos[1],:] - ho_weight[choice_pos[2],:]

    when  = neurons_inNetwork(when_value, neuron_proportion)
    which = neurons_inNetwork(which_value, neuron_proportion)

    return when, which

def datasaving(filepath, when, which, rnn_weight, ho_weight, prop):
    # save the results
    path, model = os.path.split(filepath)
    saving_name = path + '/WhenWhich_' + model[:-3] + '_threshold' + str(prop) +'.yaml'
                    
    data = {'model':model,'when':when,'which':which, 'threshold':prop,
            'rnn_weight':rnn_weight,'output_weight':ho_weight}
        
    with open(saving_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print('lesion file saved')
    print('file path, file name:',saving_name)
    

# In[]: 
# tools calculating geodesic distance, maxmimum flow, and connection pattern

def wh_loading(saving_name):
    # load the file storing the identity of when/which units
    f = open(saving_name)
    yaml_load = lambda x: yaml.load(x, Loader=yaml.Loader)
    y = yaml_load(f)
    f.close()
    return y


def cal_mat_thresholded(mat, prob_largest = 0.3):
    # find the connection with the highest prob_largest of the 
    #    absolute values of the mat
    
    num_largest = np.round(mat.size*prob_largest).astype(int)
    value_sorted = np.sort(np.abs(mat), axis = None)[::-1]

    mat_binary = np.abs(mat)>=value_sorted[num_largest]
    
    return mat_binary


def get_others(when, which, zero_node):
    others = []
    for i in range(numNeuron):
        if not(any(i == when) or any(i == which) or any(i == zero_node)):#
            others.append(i)
            
    return others

def grouping(type1,type2,value):
    value_group = []
    for i in type1:
        for j in type2:
            if i!=j:
                value_group.append(value[i,j])
    return value_group

def interaction(when, which, others, value):
    """
        classifying value between each groups
    """
    when_when   = grouping(when, when, value)
    when_which  = grouping(when, which,value)
    which_which = grouping(which,which,value)
    
    control_1 = grouping(when, others,value)
    control_2 = grouping(which,others,value)
    control = control_1 + control_2
    
    value_mean = {'control':    np.mean(control),
                  'when_when':  np.mean(when_when),
                  'when_which': np.mean(when_which),
                  'which_which':np.mean(which_which)
                  }
    
    value_sem = {'control':    stats.sem(control),
                 'when_when':  stats.sem(when_when),
                 'when_which': stats.sem(when_which),
                 'which_which':stats.sem(which_which)
                 }
    
    num = {'control':   len(control),    'when_when':  len(when_when),
           'when_which':len(when_which), 'which_which':len(which_which)
           }

    return value_mean, value_sem, num

def gesDistanceInv(graph, when, which, zero_node):
    """
    get the shortest path length
    """
    g = copy.deepcopy(graph)
    
    # geodesic distance between two nodes
    path_length = nx.all_pairs_dijkstra_path_length(g) # for weighted network
    gd = 1e10 + np.zeros([numNeuron, numNeuron])
    for i in path_length:
        for key, value in i[1].items():
            gd[i[0],key] = value
            
    # using the inverse of the distance
    gd = 1/gd
    others = get_others(when, which, zero_node)
    # get the inverse of the distance between each groups 
    dInv_mean, dInv_sem, dInv_num = interaction(when, which, others, gd)
    
    return dInv_mean, dInv_sem, dInv_num

def maxFlow(graph, when, which, zero_node):
    """
    maximun flow between two nodes
    """
    g = copy.deepcopy(graph)
    max_flow = np.zeros([numNeuron,numNeuron])
    for i in range(numNeuron):
        for j in range(numNeuron):
            if i==j or np.any(zero_node==i) or np.any(zero_node==j):
                continue
            max_flow[i,j] = nx.maximum_flow_value(g,i,j)
    
    others = get_others(when, which, zero_node)
    
    flow_mean, flow_sem, flow_num = interaction(when, which, others, max_flow)
    
    return flow_mean, flow_sem, flow_num

def gesdes_plot(df_dInv):
    # plot the inverse of geodesic distance between each groups
    nFile, nGroup = df_dInv.label.count(), 4
    gdInv = np.zeros((nFile, nGroup))
    for i in range(nFile):
        gdInv[i,:] = np.array([df_dInv['mean'][i]['control'], 
                               df_dInv['mean'][i]['when_when'], 
                               df_dInv['mean'][i]['which_which'],
                               df_dInv['mean'][i]['when_which'],
                              ])
    fig = plt.figure()
    plt.boxplot(gdInv)
    plt.ylabel('1 / geodesic distance')
    plt.xticks(np.arange(1, nFile+1),
              ('when/which-others','when-when','which-which','when-which'))
    fig.savefig('../figs/invGeoDis.eps', format='eps', dpi=1000)
    plt.show()

def maxflow_plot(df_mflow, title = ''):
    # plot the maximum flow between each groups
    nFile, nGroup = df_mflow.label.count(), 4
    maxflow = np.zeros((nFile, nGroup))
    for i in range(df_mflow.label.count()):
        maxflow[i,:] = np.array([df_mflow['mean'][i]['control'], 
                                 df_mflow['mean'][i]['when_when'], 
                                 df_mflow['mean'][i]['which_which'], 
                                 df_mflow['mean'][i]['when_which']])
    fig2 = plt.figure()
    plt.boxplot(maxflow)
    plt.ylabel('maximun flow')
    plt.xticks(np.arange(1, nFile+1),
              ('when/which-others','when-when','which-which','when-which'))
    fig2.savefig('../figs/maxFlow.eps', format='eps', dpi=1000)
    plt.show()

def conpattern_plot(df_output):
    """
        plot connection pattern of when/which units
    """
    fig_w, fig_h = (10, 7)
    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    
    nFile, nOutput = df_output.label.count(), 3
    # connection pattern of when/which units
    when_p,  when_n  = np.zeros((nFile, nOutput)), np.zeros((nFile, nOutput))
    which_p, which_n = np.zeros((nFile, nOutput)), np.zeros((nFile, nOutput))
    for i in range(nFile):
        when  = df_output.when.iloc[i]
        which = df_output.which.iloc[i]
        # output connection matrix 
        output_C = df_output.output.iloc[i][choice_pos[:-1], :] 
        
        when_p[i,:] =  output_C[:, when.pos].mean(axis=1)
        when_n[i,:] =  output_C[:, when.neg].mean(axis=1)
        which_p[i,:] = output_C[:, which.pos].mean(axis=1)
        which_n[i,:] = output_C[:, which.neg].mean(axis=1)
    
    # plot
    f = plt.figure()
    titles = ['when units','which units']
    for i, units_connection in enumerate([[when_p, when_n],[which_p, which_n]]):
        plt.subplot(2,1,i+1)
        pos, neg = units_connection[0], units_connection[1]
        plt.errorbar(range(nOutput), y = np.mean(pos, axis=0),
                     yerr = stats.sem(neg,axis=0), label = 'pos')
        plt.errorbar(range(nOutput),  y = np.mean(neg, axis=0),
                     yerr = stats.sem(neg,axis=0), label = 'neg')
        plt.legend()
        plt.title(titles[i])
        plt.xticks(range(nOutput),('fixtaion', 'left target','right target'))
    
    f.savefig('../figs/weight pattern.eps', format='eps', dpi=1000)
    plt.show()

def graph_construct(rnn_weight, prob_largest = 0.3):
    """
        construct a structure graph
    """
    
    connectivity = np.abs(rnn_weight)
    # get the index matrix representing the connections are largest 30%
    net_mat = cal_mat_thresholded(connectivity, prob_largest = prob_largest)
    # construct a directed weighted graph
    graph = nx.DiGraph()
    norms = np.mean(connectivity[net_mat!=0])
    for i in range(net_mat.shape[0]):
        for j in range(net_mat.shape[1]):
            if net_mat[i,j] != 0:
                graph.add_edge(i, j,
                               weight   = norms/np.abs(connectivity[i,j]),
                               capacity = np.abs(connectivity[i,j]) 
                               )
                net_mat[i,j] = norms/np.abs(connectivity[i,j])
    
    # find the nodes that are not connected with any other
    zero_node = np.intersect1d(np.where(net_mat.sum(axis=0)==0),
                               np.where(net_mat.sum(axis=1)==0))
    return graph, zero_node


def graph_extract(file_paths):
    # the inverse of the geodesic distance
    df_dInv  = pd.DataFrame([], columns = {'label','mean','num'})
    # the maximum flow
    df_mflow = pd.DataFrame([], columns = {'label','mean','num'})
    # the output connection pattern
    df_output = pd.DataFrame([], columns = {'label','when','which','output'})
    
    for i, file in enumerate(file_paths):
        print(file)
        y = wh_loading(file)
        when = y['when']
        which = y['which']
        when_units  = np.append(when.pos, when.neg)
        which_units = np.append(which.pos,which.neg)
        # using the recurrent connection to construct a directed weighted graph
        rnn_weight = y['rnn_weight'].detach().numpy()
        graph, island = graph_construct(rnn_weight) 
        # get the inverse distance between each groups
        dInv_mean, dInv_sem, dInv_num = gesDistanceInv(graph, when_units, 
                                                       which_units, island)

        df_output.loc[i] = {'label': file,'when':when,'which':which,
                             'output': y['output_weight']}
        
        df_dInv.loc[i] = {'label': file, 'mean': dInv_mean,'num':dInv_num}

        print('calcuating maximum flow takes a long time, please be patient')
        flow_mean, flow_sem, flow_num = maxFlow(graph, when_units, 
                                                which_units, island)
        df_mflow.loc[i] = {'label': file, 'mean': flow_mean,'num':flow_num}
    
    return df_output, df_dInv, df_mflow

def wh_select():
    print("start")
    print("select the model files")
    root = tk.Tk()
    root.withdraw()
    file_path = file_selection(initialdir = '../save/RT',
                               parent=root,
                               title='Choose the when/which file',
                               filetypes=[("model files", "*0.5.yaml")]
                               )
    print('*'*49)
    
    return file_path

# In[]: tools about basic bhv analysis

from toolkits_bhv import bhv_extract

def groups_files(file_paths):
    
    # group the files based on the lesion type
    group = []
    for nfile in file_paths:
        path, file = os.path.split(nfile)
        group.append(file.split('-')[0])
        
    files_pd = pd.DataFrame([list(file_paths),group],['name','lesion'])
    files_pd = files_pd.T
    files_groups = files_pd.name.groupby([files_pd.lesion])
    ncondition = files_pd.lesion.nunique()
    condition = files_pd.lesion.unique()

    return files_groups, condition, ncondition

def lesionBhv_selection(file_paths = None):
    print("start")
    print("select the files")
    root = tk.Tk()
    root.withdraw()
    if file_paths == None:
        file_paths = file_selection(initialdir ='../log/RT_lesion',
                                    parent = root,
                                    title  = 'Choose a file',
                                    filetypes=[("HDF5 files", "*-0.5-*.hdf5")]
                                    )
    print('*'*49)
    
    return file_paths

def lesion_extract(file_paths = None):
    files_groups, condition, ncondition = groups_files(file_paths)
    
    df = pd.DataFrame([], columns = {'label','rt','rt_sem','cr','cr_sem',
                                     'fr','fr_sem','cr_log','cr_log_sem',
                                     'bias','bias_sem'})
    
    for i, files_group in enumerate(files_groups):
        df_basic, _, _ = bhv_extract(files_group[1])

        df.loc[i] = {'label':      files_group[0],
                    'rt':          df_basic.rt_mean.mean(),
                    'rt_sem':      df_basic.rt_mean.sem(),
                    'cr':          df_basic.cr.mean(),
                    'cr_sem':      df_basic.cr.sem(),
                    'fr':          df_basic.fr.mean(),
                    'fr_sem':      df_basic.fr.sem(),
                    'cr_log':      df_basic.cr_log.mean(),
                    'cr_log_sem':  df_basic.cr_log.sem(),
                    'bias':        df_basic.choice_prop.mean(),
                    'bias_sem':    df_basic.choice_prop.sem()
                    }
        
    return df, condition, ncondition

def plot_basicBhv_lesion(df, condition, ncondition):
    # plot the reaction time
    # plot the correct rate, consistency with evidence and choice bias
    x = np.arange(ncondition)
    # reaction time changes
    fig = plt.figure()
    plt.boxplot(np.vstack(df.rt.values).T)
    plt.xticks(x+1, df.label.values)
    plt.ylabel('reaction time')
    fig.savefig('../figs/lesion_effect_rt.eps', format='eps', dpi=1000)
    
    # choices
    fig2 = plt.figure()
    plt.bar(x-0.2, df.cr,     yerr = df.cr_sem,     width=0.15,label = 'cr')
    plt.bar(x,     df.cr_log, yerr = df.cr_log_sem, width=0.15,label = 'cr_log')
    plt.bar(x+0.2, df.bias,   yerr = df.bias_sem,   width=0.15,label = 'choice_prop')
   
    plt.legend()
    plt.ylabel('proportion(%)')
    # plt.xticks(x,('control','when neg','when pos','which neg','which pos'))
    plt.xticks(x, condition)
    fig2.savefig('../figs/lesion_effect_choice.eps', format='eps', dpi=1000)
    plt.show()
    
# In[] speed-accuracy trade off

def speed_accuracy_extract():
    
    cr, cr_sem = np.zeros(11,), np.zeros(11,)
    rt, rt_sem = np.zeros(11,), np.zeros(11,)
    cr_log, cr_log_sem = np.zeros(11,),  np.zeros(11,)
    
    for i, prop in enumerate(['0.5','0.4','0.3','0.2','0.1']):
        print('label:  ', prop)
        # bhv files in which when units are inactivated
        files = glob.glob('../log/RT/lesion/when*'+prop+'*.hdf5')
        
        df, _, _ = lesion_extract(file_paths = files)
        
        cr[i] = df.loc[df.label == 'when_neg_output_les'].cr
        rt[i] = df.loc[df.label == 'when_neg_output_les'].rt
        cr_sem[i] = df.loc[df.label == 'when_neg_output_les'].cr_sem
        rt_sem[i] = df.loc[df.label == 'when_neg_output_les'].rt_sem
        cr_log[i] = df.loc[df.label == 'when_neg_output_les'].cr_log
        cr_log_sem[i] = df.loc[df.label == 'when_neg_output_les'].cr_log_sem
        
        cr[10-i] = df.loc[df.label == 'when_pos_output_les'].cr
        rt[10-i] = df.loc[df.label == 'when_pos_output_les'].rt.mean() 
        cr_sem[10-i] = df.loc[df.label == 'when_pos_output_les'].cr_sem
        rt_sem[10-i] = df.loc[df.label == 'when_pos_output_les'].rt_sem
        cr_log[10-i] = df.loc[df.label == 'when_pos_output_les'].cr_log
        cr_log_sem[10-i] = df.loc[df.label == 'when_pos_output_les'].cr_log_sem
        
    # no lesion
    files = glob.glob('../log/RT/*.hdf5') 
    df,_,_ = bhv_extract(file_paths = files)
    cr[5] = df.cr.mean()
    rt[5] = df.rt_mean.mean()
    cr_sem[5] = df.cr.sem()
    rt_sem[5] = df.rt_mean.sem()
    cr_log[5] = df.cr_log.mean()
    cr_log_sem[5] = df.cr_log.sem()

    return cr, cr_sem, rt, rt_sem, cr_log, cr_log_sem

def plot_speed_accuracy(cr, cr_sem, rt, rt_sem, cr_log, cr_log_sem):
    
    
    fig0, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # plot the lesion effect on choice
    ax1.errorbar(range(11), cr, cr_sem, label = 'cr')
    ax1.errorbar(range(11), cr_log, cr_log_sem, label = 'cr_log')
    plt.ylim([0.7,1.25])
    ax1.plot(3, 1.02)
    ax1.legend()
    # plot lesion effect on reaction time
    ax2.errorbar(range(11), rt, rt_sem,color = 'k', label = 'rt')
    plt.ylim([2,10.5])
    ax2.legend()
    
    fig0.savefig('../figs/speed_accuracy_all.eps', format='eps', dpi=1000)
    plt.show()


