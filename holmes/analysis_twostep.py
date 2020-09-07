#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:36:29 2019

@author: Zhewei Zhang

stay probability, RL model fitting, history effect test
decode Q value from hidden layer, the correlation between q value and output

"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV

from lmfit import Model
from scipy import stats
from scipy.special import erf
from scipy.signal import savgol_filter
from scipy.optimize import minimize, curve_fit
        

import tkinter as tk
from tkinter import filedialog

from toolkits_2 import get_stinfo, load, get_resp_ts, stayprob


def lin(x, k, b):
    return x*k + b

def sigmoid(x, a, b, c, d):
     return d + c*1 / (1 + np.exp(-b*(x-a)))


"""
plot the stay probability in the two step task

"""
def softmax(state, q, beta, rep = None):
    """
    Softmax policy: selects action probabilistically depending on the value.
    Args:
        state: an integer corresponding to the current state.
        q: a matrix indexed by state and action.
        params: a dictionary containing the default parameters.
    Returns:
        an array corresponding to the probability of choosing each options.
    """
    value = q[state,:]
    prob = np.exp(value * beta)# beta is the inverse temperature parameter
    if rep != None:
        prob[rep[0]] = np.exp(value[rep[0]] * beta + rep[1]) # tendency of repeating the previous choice
        
    prob = prob / np.sum(prob)  # normalize            
    return prob

def cr_block(trials):
    """
    input: common: the status of transition from the first to the second stage
           reward: reward feedback
           choice: the choice in the first stage
           prob  : the reward probability of four targets in the second stage
    test how the choices is affected by the correct/stay/outcome/
        transition:common or rate/outcome*transition
    """
    pre, post = 10, 30
    numTrials = trials["choice"].size
    trial_sep = np.where(np.diff(trials["block"])!=0)[0]
    numBlocks = np.sum(np.diff(trials["block"])!=0)+1
    correct_rate = []
    for nblock in range(numBlocks):
        if nblock == 0:
            continue
        # the trials in current block
        trial_start = trial_sep[nblock-1]+1 if nblock > 0 else 0
        # Current block
        curr_block = trials["block"][trial_start]
        # correct rate ten trials before/30 trials after the reversal
        choice = trials["choice"][trial_start-pre:trial_start+post].values
        cr = choice-1 if curr_block == 0 else 2-choice
        cr[:pre] = 1-cr[:pre]
        correct_rate.append(cr)
    correct_rate = np.vstack(correct_rate)
    
    return np.nanmean(correct_rate, axis = 0)


def hist_effect(common, reward, choice, hist_len = 10):
    """
    input: common: the status of transition from the first to the second stage
           reward: reward feedback
           choice: the choice in the first stage
    test how the choices is affected by the interaction between the outcome and
           the transition status(common or rare) in the last 10 trials 
    """
    numTrials = common.size
    regressors = np.zeros((hist_len*4, numTrials))

    stay = np.diff(choice)
    stay[stay!=0] = 1
    stay = 1-stay
    
    states = reward*2 + 1-common
    for i in range(4):
        target_trials_base = np.where(states==i)[0]
        for ii in range(hist_len):
            temp = np.zeros((1, numTrials))
            target_trials = target_trials_base + ii
            temp[0,target_trials[target_trials<numTrials-1]] = 1
            regressors[i*hist_len+ii,:] = temp
    
#    lm = LogisticRegression(fit_intercept=False, penalty='l1', solver='liblinear')
    lm = LogisticRegression(fit_intercept=True, solver='liblinear')
    lm.fit(regressors[:,:-1].T, stay)
    return lm.coef_

def factors_eff(common, reward, choice, block):
    """
    input: common: the status of transition from the first to the second stage
           reward: reward feedback
           choice: the choice in the first stage
           prob  : the reward probability of four targets in the second stage
    test how the choices is affected by the correct/stay/outcome/
        transition:common or rate/outcome*transition
    """
    correct = block[:-1] == choice[:-1]
    repeat = np.ones(choice[:-1].size)
    outcome = reward[:-1]*1/2
    outcome[outcome==0] = -1/2
    trans = common[:-1]-0.5
    trans_out = trans*outcome*2
    
    stay = np.diff(choice)
    stay[stay!=0] = 1
    stay = 1-stay
    
    regressors = np.vstack((correct, repeat, outcome, trans, trans_out))
    # l1 is used owing to the correlation between the outcome and the trans_outcome
#    lm = LogisticRegressionCV(cv=10, fit_intercept=False, penalty='l1', solver='liblinear')
    lm = LogisticRegression(fit_intercept=True, solver='liblinear')
    lm.fit(regressors.T, stay)
    return lm.coef_

def cost_mfmb(params, choices, blocks, states, rewards, ReturnQ = False):
    """
    return the negative log likelihood given the choices and reward feedback
    
    """
    nll = 0
    gama = 1
    trans_prob = [0.8, 0.2]
    n_states, n_arms = 2, 2

    w1, lr_MB, lr_MF, lr_lambda, beta1, rep = params[0], params[1], params[2], params[3], params[4]*10, params[5]*10
    w2 = 1 - w1

    NumTrials = choices.size
    Q_value_based = np.zeros((n_states,n_arms))
    Q_value_free = np.zeros((n_states,n_arms))

    Q_value_all = []
    action_probs = []
#    max_nll = 0
#    nll_all = np.zeros(NumTrials,)
    for ntrial in range(1,NumTrials):
        ######## True action and states
        state = states[ntrial]
        action = choices[ntrial]
        reward = rewards[ntrial]
        
        ######## choice probability based on MB/MF-RL
        Q_value = w1*Q_value_based + w2*Q_value_free
        act_prob = softmax(0, Q_value, beta1, [choices[ntrial-1], rep])
        
        ######## calculating the log likelihood 
        act_prob += 1e-10
        nll += -((1-action) * np.log(act_prob[0]) + action * np.log(act_prob[1]))
        
        ####### model-based update
        Q_value_based[1, state] += lr_MB*(reward - Q_value_based[1, state])
                
        Q_value_based[0, 0] = np.matmul(trans_prob, Q_value_based[1, :])
        Q_value_based[0, 1] = np.matmul(trans_prob, Q_value_based[1, ::-1])
        
        ####### model free update
        second_q_error = reward - Q_value_free[1, state]
        Q_value_free[1, state] += lr_MF*lr_lambda*(second_q_error)
       
        first_q_error = Q_value_free[1,state] - Q_value_free[0, action]
        Q_value_free[0, action] += gama*lr_MF*first_q_error
        
        # store values
        Q_value_all.append(Q_value)
        action_probs.append(act_prob)

        
    if ReturnQ:
        return nll, np.array(Q_value_all), action_probs
    else:
        return nll

    
def data_extract(file_paths):
    """
    
    """
    df_details = pd.DataFrame([], columns = {'state', 'choice', 'reward',
                                             'block','Q_diff','R_diff',
                                             'Q_diff_pred'})
    df_summary = pd.DataFrame([], columns = {'stay_num', 'type_num','cr',
                                             'history','factors',
                                             'fit_RL','fit_lm','fit_sm'})
    
    for i, file in enumerate(file_paths):
        paod, trial_briefs = load(file)
        # load the choice information,         
        trials = get_stinfo(paod, trial_briefs, completedOnly = False)
        # the correct rate
        cr = cr_block(trials)
        # load the choice information, only complete trials are included
        trials = get_stinfo(paod, trial_briefs)
        # the stay probability after each trial type
        stay_num, type_num, transition = stayprob(trials["choice"].values-1,
                                                  trials["state"].values, 
                                                  trials["reward"])
        # how the history affect choice
        coef_hist    = hist_effect(transition, trials["reward"], 
                                   trials["choice"], hist_len = 5)
        # how several factors affect choice
        coef_factors = factors_eff(transition, trials["reward"], 
                                   trials["choice"], trials["block"])

        ## fit the bhv with a reinforcement learning model
        bounds=[[0,1],[0,1],[0,1],[0,1],[0,3],[0,3]]
        params = np.array([0.7, 0.5 , 0.8, 0.5, 0.1 , 0.1])
        cons = [] #construct the bounds in the form of constraints
        for factor in range(len(bounds)):
            l = {'type': 'ineq','fun': lambda x: x[factor]-bounds[factor][0]}
            u = {'type': 'ineq','fun': lambda x: bounds[factor][1]-x[factor]}
            cons.append(l)
            cons.append(u)
        
        nll_wrapper = lambda parameters: cost_mfmb(parameters,
                                                   trials["choice"]-1,
                                                   trials["block"], 
                                                   trials["state"]-1,
                                                   trials["reward"]
                                                   )
        res = minimize(nll_wrapper, x0=params, method='SLSQP' , bounds=bounds, 
                        constraints=cons)
        
        # estimate the q value based on the fitted parameters
        _, Q_value, _ = cost_mfmb(res.x,
                                  trials["choice"].values-1, 
                                  trials["block"].values,  
                                  trials["state"].values-1,
                                  trials["reward"].values, 
                                  ReturnQ = True
                                  )

        
        # the response of units in hidden layer and output layer
        resp_hidden, resp_output = get_resp_ts(paod, trial_briefs) 

        # the differnce of response of two action units before choice
        b4_cho, left_u, right_u = 4, 6, 7
        Q_diff = Q_value[:,0,0]-Q_value[:,0,1]
        R_diff = resp_output[1:,b4_cho, left_u] - resp_output[1:,b4_cho, right_u]
        
        # fit with a linear model
        lmodel = Model(lin)
        result_l = lmodel.fit(Q_diff, x=R_diff, k=1, b=0)
        # fit with a sigmoid model
        smodel = Model(sigmoid)
        result_s = smodel.fit(Q_diff, x=R_diff, a=0, b=1, c=1, d=0)
        
        # fit the response of units in the hidden layer
        reg = LinearRegression().fit(resp_hidden[1:,b4_cho,:], Q_diff)
        Q_diff_pred = reg.predict(resp_hidden[1:,b4_cho,:])
        

        df_summary.loc[i] = {
                            'stay_num': stay_num, 
                            'type_num': type_num,
                            'history':  coef_hist,
                            'factors':  coef_factors,
                            'fit_RL' :  res.x, # params, #
                            'fit_lm' :  result_l, 
                            'fit_sm' :  result_s,
                            'cr'     :  cr
                            }

        df_details.loc[i] = {
                            'block':  trials["block"].values, 
                            'state':  trials["state"].values,
                            'choice': trials["choice"].values,
                            'reward': trials["reward"].values,
                            'R_diff': R_diff,
                            'Q_diff': Q_diff,
                            'Q_diff_pred':Q_diff_pred
                            }

    return df_details, df_summary


def plot_value_coding(df_details, df_summary):

    """
    test the linear relation between the Q value and activities of the units 
    in the hidden layer and output layer

    """
    # parameters for smooth
    window_length, polyorder = 101, 1
    smooth = lambda x: savgol_filter(x, window_length, polyorder)
    
    R_diff = df_details["R_diff"].values
    Q_diff = df_details["Q_diff"].values
    Q_diff_pred = df_details["Q_diff_pred"].values
    
    # plot the qvalue difference against the output activities difference
    R_diff_list = [i for i in R_diff] + [np.hstack(R_diff)]
    Q_diff_list = [i for i in Q_diff] + [np.hstack(Q_diff)]
    Q_diff_pred_list = [i for i in Q_diff_pred] + [np.hstack(Q_diff_pred)]

    fig0 = plt.figure('q value vs output')
    for i, (Q_diff_, R_diff_) in enumerate(zip(Q_diff_list, R_diff_list)):
        argsort = np.argsort(Q_diff_)
        R_diff_ = smooth(R_diff_[argsort])
        Q_diff_ = smooth(Q_diff_[argsort])
        
        if i == len(Q_diff_list)-1:
            plt.plot(Q_diff_, R_diff_, 'k')
        else:
            plt.plot(Q_diff_, R_diff_, 'grey')
    plt.show()
    fig0.savefig('../figs/q value vs output.eps', format='eps', dpi=1000)
    
    fig1 = plt.figure('q value decoding')
    for i, (Q_diff_pred_, Q_diff_) in enumerate(zip(Q_diff_pred_list, Q_diff_list)):
        argsort = np.argsort(Q_diff_pred_)
        
        Q_diff_ = smooth(Q_diff_[argsort])
        Q_diff_pred_ = smooth(Q_diff_pred_[argsort])

        if i == len(Q_diff_pred_list)-1:
            plt.plot(Q_diff_pred_, Q_diff_, 'k')
        else:
            plt.plot(Q_diff_pred_, Q_diff_, 'grey')
    plt.show()
    fig1.savefig('../figs/predicted Q vs Q.eps', format='eps', dpi=1000)
        
def plot_bhv(df_summary):
    '''
    plot the stay probability ; historical effect;  
        factors affect the choices and the correct rate
    '''
#    fig_w, fig_h = (10, 7)
#    plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
    stay_num = np.vstack(df_summary.stay_num.values)
    type_num = np.vstack(df_summary.type_num.values)
    
    stay_prob = stay_num/type_num
    # plot stay probability
    fig = plt.figure('stay probability')
    plt.bar([1,4], np.mean(stay_prob[:,[0,2]], axis=0), 
            yerr = stats.sem(stay_prob[:,[0,2]], axis=0), label = 'Common')
    plt.bar([2,5], np.mean(stay_prob[:,[1,3]], axis=0),
            yerr = stats.sem(stay_prob[:,[1,3]], axis=0), label = 'Rare')
    plt.legend()
    plt.xticks([1.5,4.5], ('Rewarded', 'Unrewarded'))
    plt.ylabel('stay probability')
    plt.yticks(np.linspace(0,1,num=6,endpoint = True))
    # fig.savefig('../figs/stay probability.eps', format='eps', dpi=1000)
    plt.show()
    
    # history effect
    coef = np.vstack(df_summary.history.values)
    labels = ['common unrewarded','rare unrewarded','common rewarded','rare rewarded']
    fig2 = plt.figure('hist effect')
    hist_len = int(coef.shape[1]/4) # four groups
    for i in range(4):# four groups
        coef_ = coef[:,hist_len*i:hist_len*(i+1)]
        plt.errorbar(range(hist_len),
                     np.mean(coef_, axis=0), yerr = stats.sem(coef_, axis=0),
                     marker = 'o',  markersize = 20,  markerfacecolor='none',
                     label = labels[i])
        
    plt.xlabel('lag (trials)');  plt.ylabel('log odds')
    plt.xticks(range(5), ('-1', '-2', '-3', '-4', '-5'))
    plt.yticks(np.linspace(-2, 2, num=9,endpoint = True))
    fig2.legend()
    fig2.savefig('../figs/hist effect.eps', format='eps', dpi=1000)
    plt.show()
    # signficance test 
    _, pvalues_hist = stats.ttest_1samp(coef,0)
    print(pvalues_hist)
    
    # how transition, reward, correct, repeat affect chocies    
    coef = np.vstack(df_summary.factors.values)
    fig3 = plt.figure('coef_factors')
    for i in range(5):
        plt.plot(i, np.mean(coef[:,i]), 'o')
        plt.errorbar(i, np.mean(coef[:,i]), yerr = stats.sem(coef[:,i]))
    plt.ylabel('log odds')
    plt.xticks(range(5), ['correct', 'repeat', 'outcome', 'trans', 'trans_out'])
    plt.yticks(np.linspace(0,3.5,num=8,endpoint = True))
    fig3.savefig('../figs/factors effect.eps', format='eps', dpi=1000)
    plt.show()
    # signficance test 
    _, pvalues_factors = stats.ttest_1samp(coef,0)
    print(pvalues_factors)
    
    # the correct before and after reversal 
    cr = np.vstack(df_summary.cr.values)
    fig4 = plt.figure('cr')
    plt.errorbar(range(cr.shape[1]), 
                 np.mean(cr, axis=0),
                 yerr = stats.sem(cr, axis=0))
    plt.ylabel('correct rate');  plt.xlabel('lag (trials)')
    plt.yticks(np.linspace(0, 1, num=6, endpoint = True))
    plt.show()
    
def main(file_paths = None):
    print("start")
    print("select the files")
    if file_paths == None:
         root = tk.Tk()
         root.withdraw()
         # file_paths = filedialog.askopenfilenames(parent = root,
         #                                         title = 'Choose a file',
         #                                         filetypes = [("HDF5 files", "*.hdf5")]
         #                                         )
         file_paths = glob.glob('../log/ST/*.hdf5')
    ##
    df_details, df_summary = data_extract(file_paths)
    plot_bhv(df_summary)
    plot_value_coding(df_details, df_summary)


if __name__ == '__main__':
    main()

