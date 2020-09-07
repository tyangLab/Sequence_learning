# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:24:16 2018

@author: Zhewei Zhang

demonstrate selectivity on the logLR, abs(logLR), urgency and choice
"""
import numpy as np

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# import warnings

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()



def lm_Regression(X, y):
    """
        linear regression. but with significance test
        X: matrix, No. observations by No. factors
    """    
    newX = np.append(np.ones((len(X), 1)), X, axis=1)

    lm = LinearRegression(fit_intercept=True)
    lm.fit(X, y)

    params = np.append(lm.intercept_, lm.coef_)
    pred_y = lm.predict(X)
    SSresidual = ((y - pred_y) ** 2).sum()
    MSE = SSresidual / (len(newX) - len(newX[0]))
    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    np.warnings.filterwarnings('ignore')
    ts_b = params / sd_b
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
    
    return {'params': params, 'p_values': np.array(p_values)}


def linearReg(x, resp, loc = None):
    
    x = np.asarray(x,dtype = 'float64')
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    if loc is None:
        y = resp
    else:
        y = resp[loc]
    result = lm_Regression(x_std,y)
    return result


def get_factors(trials, choices, shapes):

    numTrials = trials.count().num

    total_length = 0
    sum_weight, time_interest, urgence, choice = [], [], [], []
    
    for i in range(numTrials):
        if not choices.status.iloc[i]: # complete trials only
            continue

        shape_ontime = shapes.ontime.iloc[i]
        time_interest.append(shape_ontime+3+total_length)
        # the number of shapes presented in each trial        
        shape_num = shapes.on.iloc[i]
        # urgency
        urgence.append(shape_num)
        # sum weight
        sum_weight.append(shapes.tempweight.iloc[i][shape_num-1].cumsum())
        # choice
        choice_opportunity = np.zeros(shape_num.shape).tolist()
        if choices.left.iloc[i]==1:
            choice_opportunity[-1] = 1
            choice.extend(choice_opportunity)
        else:
            choice_opportunity[-1] = -1
            choice.extend(choice_opportunity)
        
        total_length += trials.length.iloc[i]
        
    choice = np.array(choice)
    urgence = np.concatenate(np.array(urgence))
    sum_weight = np.concatenate(np.array(sum_weight))
    time_interest = np.concatenate(np.array(time_interest))
    
    return sum_weight, urgence, choice, time_interest


def lm_fit(sum_weight, urgence, choice, resp, time_interest):
    params = []
    p_values = []
    # summed weight/abs(summed weight)/urgency/choice, all in one regression
    x = np.hstack([sum_weight.reshape(-1,1), np.abs(sum_weight.reshape(-1,1)), 
                   urgence.reshape(-1,1),    choice.reshape(-1,1)])
    # perform the regression on each time point of epoches
    for i in range(5):
        result = linearReg(x,resp,time_interest+i)
        params.append(result['params'].tolist())
        p_values.append(result['p_values'].tolist())
        
    return params, p_values

def regress_resp(resp_hidden, trial, choice, shapes):
    """
    perfrom a linear regression
        X: sum weight, absoulte value of sum weight; urgency, choice
        y: response of units in the hidden layer
    """    
    # neural response
    resp_hidden = np.concatenate(resp_hidden, axis=0)
    num_Neuron = resp_hidden.shape[1]
    # variables
    sum_weight, urgence, choice, time_interest = get_factors(trial, choice, shapes)

    lm_result = {'params':[],'p_values':[]}

    for i in range(num_Neuron):
        resp = resp_hidden[:,i]
        params, p_values = lm_fit(sum_weight, urgence, choice, 
                                  resp,
                                  time_interest)
        lm_result['params'].append(params)
        lm_result['p_values'].append(p_values)
        
        if (i+1)%(2**4) == 0:
            print('regression:{}th neuron'.format(i+1))
    return lm_result

