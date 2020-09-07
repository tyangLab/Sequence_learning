# -*- coding: utf-8 -*-
# @Author: Zhewei Zhang
# @Date: 07/27/2018, 23:39

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegressionCV

from toolkits_2 import get_bhvinfo, load

# In[]
def shape_config():
    fp = 0
    input_size = 20
    target = [1,2]
    shapes = [3,4,5,6,7,8,9,10,11,12]
    novisual = 13
    choice = [14,15,16,17]
    reward = [18,19]
    input_setting = {'fp':fp,
                     'target':target,
                     'shapes':shapes, 
                     'novisual':novisual,
                     'choice':choice,
                     'reward':reward,
                     'input_size':input_size
                     }
    return input_setting

# In[] 
"""
representing the relation between the logRT and rt; 

fit the psychmetric curve
"""

num_shape = 18
psycv_range = 2.

def sigmoid(x, x0, k):
    return -1 + 2/(1+np.exp(-k*(x-x0)))


def log_rt(shapes, choices):
    """
    transfer it into category to get the logLR in all rt values, 
        and it is convenient to save and plot in matlab
    retrun: right_rt_log: logLR varies with rt when the right is choosing
            left_rt_log: logLR varies with rt when the left is choosing
    """
    shape_num_category = pd.Categorical(shapes['rt'],ordered = True,
                                        categories = np.arange(1.,num_shape+.1,1)
                                        )
    left_rt_log = shapes['sumweight'][choices['left'] == 1].groupby(
            shape_num_category[choices['left'] == 1])
    right_rt_log = shapes['sumweight'][choices['left'] == -1].groupby(
            shape_num_category[choices['left'] == -1])
    
    return right_rt_log.describe()[['count','mean','std']], left_rt_log.describe()[['count','mean','std']]


def psych_curve(shapes, choices, groups = np.linspace(-2,2,num=40)):
    """
    return: psychmetric curve
    """
    psy_curve = choices.left.groupby(pd.cut(shapes['sumweight'],groups))
    if shapes['sumweight'].shape[0] > 2:
        prpt = np.array([np.nan, np.nan])    
        # prpt, pcov = curve_fit(sigmoid, shapes['sumweight'], choices.left)
        prpt = np.array([-0.10974141,  6.40010796]), 1 
        pcov = np.array([[ 8.44701144e-05, -8.38683285e-06],[-8.38683285e-06,  3.55214019e-02]])
    else:
        prpt = np.array([np.nan, np.nan])    
    return psy_curve.mean(), prpt



def bhv_extract(file_paths):
    """
    
    """
    df_basic = pd.DataFrame([], columns = {'rt_mean','rt_sem','choice_prop',
                                           'cr','cr_log','fr','label'})
    df_logRT = pd.DataFrame([], columns = {'right_rt_log','left_rt_log',
                                           'rt', 'choice'})
    df_psycv = pd.DataFrame([], columns = {'cr','fr','psy_curve',
                                           'fitting_x0', 'fitting_k'})
    for i, file in enumerate(file_paths):

        paod, trial_briefs = load(file)
        _, choice, shape, _, finish_rate = get_bhvinfo(paod,trial_briefs)
        right_rt_log , left_rt_log = log_rt(shape, choice)
        psy_curve, prpt = psych_curve(shape, choice, 
                                      groups = np.linspace(-psycv_range,psycv_range,num=20))
        
        # keep summary data of each file for plotting
        df_basic.loc[i] = {'rt_mean':     shape.rt.mean(),
                           'rt_sem':      shape.rt.sem(),
                           'cr':          choice.correct.mean(),
                           'fr':          finish_rate,
                           'choice_prop': (choice.left.mean()+1)/2,
                           'cr_log':      choice.correct_logLR.mean(),
                           'label': file
                           }
        
        df_logRT.loc[i] = {'right_rt_log': right_rt_log['mean'].values, 
                           'left_rt_log':   left_rt_log['mean'].values,
                           'rt':           shape.rt,
                           'choice':       choice.left,
                       }
        
        df_psycv.loc[i] = {'psy_curve': psy_curve, 
                           'cr':          np.round(choice.correct.mean(),3),
                           'fr':          np.round(finish_rate,3),
                           'fitting_x0':  prpt[0], 
                           'fitting_k':   prpt[1]
                       }

    return df_basic, df_logRT, df_psycv

def plot_bhvBasic(df_logRT, df_psycv):

    rt_left, rt_right = [], []
    for i in range(df_logRT.rt.count()):
        rt_left.extend( df_logRT.rt[i][df_logRT.choice[i]== 1])
        rt_right.extend(df_logRT.rt[i][df_logRT.choice[i]==-1])
        
    bins_ = np.arange(0.5,25.6,1)
    left_rt_dist  = np.histogram(rt_left,  bins = bins_, density = True)[0]
    right_rt_dist = np.histogram(rt_right, bins = bins_, density = True)[0]
    
    # ignore the rt happens less than 0.1%
    tar_Point = np.intersect1d(np.where(left_rt_dist>1e-3), np.where(right_rt_dist>1e-3)) 
    left_rt_dist = left_rt_dist[tar_Point]
    right_rt_dist = right_rt_dist[tar_Point]
    

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    x = np.arange(1,df_logRT.right_rt_log[0].shape[0]+1)[tar_Point]
    
    RT_left = np.vstack(df_logRT.left_rt_log.values)
    RT_right = np.vstack(df_logRT.right_rt_log.values)
    
    ax1.errorbar(x, np.nanmean(RT_left,axis=0)[tar_Point],
                    yerr = stats.sem(RT_left,axis=0,nan_policy = 'omit')[tar_Point],
                    color = 'r')
    ax1.errorbar(x, np.nanmean(RT_right,axis=0)[tar_Point],
                    yerr = stats.sem(RT_right,axis=0,nan_policy = 'omit')[tar_Point],
                    color = 'g')
    plt.ylim(-1.7,1.7)

    left_rt_dist = np.append(0, np.append(np.array([0]*tar_Point[0]),left_rt_dist))
    right_rt_dist= np.append(0, np.append(np.array([0]*tar_Point[0]),right_rt_dist))
    
    rt_dist = pd.DataFrame({'left': left_rt_dist,'right':-right_rt_dist})
    rt_dist.left.plot(kind='bar', color='red', ax=ax2)
    rt_dist.right.plot(kind='bar', color='green', ax=ax2)
    plt.xticks(np.arange(0,num_shape,5), ('0', '5', '10', '15','20'))
    plt.ylim(-0.28,0.28)
    plt.xlim(0,num_shape)
    fig.savefig('../figs/logLR vs rt.eps', format='eps', dpi=1000)

    
    #### psychometric curve
    num_x = len(df_psycv.psy_curve[0].index.values)
    x_ = np.linspace(-psycv_range,psycv_range, num = num_x, endpoint = True)
    x_ = np.tile(x_, len(df_psycv))
    y_ = np.hstack(df_psycv.psy_curve.values)
    # fit with a sigmoid function
    prpt, pcov = curve_fit(sigmoid, x_[~np.isnan(y_)], y_[~np.isnan(y_)])
    prpt_x0, prpt_k = prpt[0], prpt[1]
    plt.figure('psychmetric curve')
    ## psy_curve for plot
    x = range(num_x)
    psy_values = []
    for i in df_psycv.psy_curve.values:
        psy_values.append(i.values)
    psy_values = np.array(psy_values)
    
    plt.errorbar(x, np.nanmean(psy_values, axis=0),
                 yerr = stats.sem(psy_values, axis=0, nan_policy = 'omit'),
                 fmt = 'o', 
                 label = 'correct rate:{}||completion rate:{}'.format(
                     np.round(df_psycv['cr'].mean(),3),
                     np.round(df_psycv['fr'].mean(),3)
                     )
                 )
    # plot fitted line    
    x = np.linspace(-psycv_range, psycv_range, 100, endpoint = True)
    y = sigmoid(x, prpt_x0, prpt_k) 
    x = x*num_x/(psycv_range*2) + (num_x-1)/2
    plt.plot(x,y)
    plt.xlabel('')
    plt.xticks(np.array([0,(num_x-1)/2,num_x-1]), (-psycv_range, '0.0', psycv_range))#
    plt.legend(loc=2,fontsize = 'large')
    foo_fig = plt.gcf()
    foo_fig.savefig('../figs/psychmetric curve.eps', format='eps', dpi=1000)
    plt.show()


# In[]

"""
with logistci regression, we show the leverage of each shape on the choice

fig1 b/d 
"""

max_rt = 25
def shape_extract(path_files):
    df_temporal = pd.DataFrame([], columns = {'label','bais','coef'})
    df_subweight = pd.DataFrame([], columns = {'label','bais','coef'})

    for i, file in enumerate(path_files):
        ## load files
        paod, trial_briefs = load(file)
        _, choice, shape, _, finish_rate = get_bhvinfo(paod,trial_briefs)
        reg_temporal  = temporal_weight(shape,choice)
        reg_subweight = subject_weight(shape, choice)
        
        df_temporal.loc[i] = {'label': file,
                              'bais':  reg_temporal['bais'],
                              'coef':  reg_temporal['coef'], 
                           }
        df_subweight.loc[i] = {'label': file,
                               'bais': reg_subweight['bais'],
                               'coef': reg_subweight['coef'], 
                               }
    
    return df_temporal, df_subweight

def temporal_weight(df_shape, df_choice):
    
    # calculating the weight of shapes in first three, last three and 
    #   intermediate epochs. 
    # Trials with shapes less than 6 are excluded

    nth = 3
    numtrials = df_shape.count().rt
    first, last, inter, choice = [], [], [], []
    
    temp_weight = np.concatenate(df_shape.tempweight.values)
    temp_weight = temp_weight.reshape(-1,df_shape.tempweight[0].shape[0])

    for i in range(numtrials):
        rt = df_shape.rt.iloc[i]
        if rt < nth*2+0.5:
            continue
        first.append( temp_weight[i, :nth])
        last.append(  temp_weight[i, rt-nth:rt])
        inter.append((temp_weight[i, nth:rt-nth]).sum())
        choice.append(df_choice.left.iloc[i])
        
    inter = np.array(inter)
    X = np.hstack((first, inter.reshape(-1,1), last))

    choice= np.array(choice)
    choice[choice==-1] = 0
    
    lm = LogisticRegressionCV(cv=10, fit_intercept=True)
    lm.fit(X[:,:-1], choice)    

    return {'bais':lm.intercept_[0],'coef': lm.coef_[0]}

def subject_weight(shape, choice):
    ## calculating subjective weight
    numTrials, numshapes = shape.shape[0], 10
    
    shape_order = np.concatenate(shape.order.values)
    shape_order = shape_order.reshape(-1, shape.tempweight[0].shape[0])
    # the number of each shape in each trial
    shapeNum_percon = np.zeros((numTrials, numshapes))
    for numT in range(numTrials):
        for i in range(10):
            shapeNum_percon[numT,i] = np.sum(shape_order[numT,:]==i+1)
    choice_left = choice.left.values
    choice_left[choice_left==-1]=0
    
    lm = LogisticRegressionCV(cv=10,fit_intercept=True)
    lm.fit(shapeNum_percon, choice_left)

    return {'bais':lm.intercept_[0], 'coef': lm.coef_[0]}

def plot_subweight(df_epoch, df_subweight):
    fig = plt.figure('regression weight on each epoch') 
    bais_epoch = df_epoch.bais.values
    coef_epoch = np.vstack(df_epoch.coef.values)
    coefs_epoch = np.hstack((bais_epoch.reshape(-1,1),coef_epoch))
    plt.errorbar([0,1,2,3,5,7,8], 
                 coefs_epoch.mean(axis=0), 
                 yerr = stats.sem(coefs_epoch, axis=0)
                 )
    plt.plot([0,1,2,3,5,7,8], coefs_epoch.mean(axis=0), 'o-')
    plt.legend()
    plt.ylabel('coef')
    plt.xlabel('nth shape')
    fig.savefig('../figs/regression_on_epochs.eps', format='eps', dpi=1000)

    f2 = plt.figure('subjective value')
    coefs_subweight = np.vstack(df_subweight.coef.values)
    plt.errorbar(range(10),
                 coefs_subweight.mean(axis=0),
                 yerr = stats.sem(coefs_subweight,axis=0)
                 )
    plt.plot(range(10), coefs_subweight.mean(axis=0), 'o-')
    f2.savefig('../figs/subjective_weight.eps', format='eps', dpi=1000)
    print('figurs are saved')

