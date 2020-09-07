# -*- coding: utf-8 -*-
# @Author: Huzi Cheng
# @Date: 08/01/2018, 10:39

"""
provide tools for extracting data.
"""

import os
import base64
import pickle
import tables
import importlib

import numpy as np
import pandas as pd

from itertools import islice

WEIGHTS = (-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9)

def base642obj3(bobj):
    """
    decoder for py3
    """
    sobj = base64.b64decode(bobj)
    obj = pickle.loads(sobj, encoding='latin1')
    return obj

def getone(li):
    """
    get the index of 1 from the right side of the array
    """
    try:
        # pdb.set_trace()
        indexs = np.where(li == 1)
        return indexs[0][0]
    except:
        # means error
        return -1



def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def match_rate(t1, t2):
    """Compute the correct rate of prediction"""
    if t1.shape == t2.shape:
        diff = np.sum(np.abs(t1 - t2))  # trial is binary
        return 1 - (diff / float(t1.shape[0] * t1.shape[1]))
    else:
        print("Matched shape? No!")
        return 0


def shape_reduce(seq):
    """Get the cumulative evidence seq from weights evidence"""
    ac_seq = [0, ]  # 0 added since we line it.
    cu_w = 0
    for w in seq:
        cu_w = cu_w + w
        ac_seq.append(cu_w)
    return np.array(ac_seq), cu_w


def get_shape_count(seq):
    """Get the shape count according to the shape sequence"""
    sc = np.zeros(10)
    for s in seq:
        sc[s] = sc[s] + 1
    return sc


def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]


def extract_data(weights_choices):
    n_weights_choices = []
    for w_c in weights_choices:
        if w_c[0] < 200 and w_c[0] > -200:
            n_weights_choices.append(w_c)

    n_weights_choices = np.array(n_weights_choices)
    evidence_basket = []
    one_basket = []
    num = 10
    for i in range(-90, 90):
        evidence_l = i / float(num)
        evidence_r = (i + 1) / float(num)
        count = 0
        count_one = 0
        for j in range(n_weights_choices.shape[0]):
            if n_weights_choices[j, 0] > evidence_l and n_weights_choices[j, 0] <= evidence_r:
                count = count + 1
                if n_weights_choices[j, 1] == 1:
                    count_one = count_one + 1
        if count_one >= 0 and count > 0:
            evidence_basket.append(evidence_l)
            one_basket.append(count_one / float(count))
    return evidence_basket, one_basket


def get_dis_val(c, X_val, y_val):
    """
    c=[c0, ..., c9]
    """
    err = 0
    base = 10
    cs = np.array(c)
    nf = []
    for f in X_val:
        nf.append(np.dot(f, cs))
    nf = np.array(nf)
    p = np.power(base, nf) / (1. + np.power(base, nf))
    one = y_val == 1
    zero = y_val == 0
    err = -np.sum(np.log(p * one + (zero - (p * zero))))
    return err


def get_dis(c, X, y):
    """
    c=[c0, ..., c9]
    """
    err = 0
    base = 10
    cs = np.array(c)
    nf = []
    for f in X:
        nf.append(np.dot(f, cs))
    nf = np.array(nf)
    p = np.power(base, nf) / (1. + np.power(base, nf))
    one = y == 1
    zero = y == 0
    err = -np.sum(np.log(p * one + (zero - (p * zero))))
    return err


def cq(q):
    return np.power(10, q) / (1. + np.power(10, q))


def get_state(fd, vali_set=True):
    """
    check if the trial get the reward, the position of target, and the direction
    of choice.
    """
    no_choice = 0
    target_right_val = 0
    reward_val = 0
    if fd[-3, -1] == 1:  # rewards
        reward_val = 1
    if fd[4, 2] == 1:  # target one in right.
        # notice no matter the cases belongs to fake data or predicted data, we
        # only trace the 2nd mark of target position, since the 1st mark in pre-
        # dicted data is random (though this might be useful in other analysis)
        target_right_val = 1
    try:
        choice_right_val = 0
        pos_choice_val = map(lambda x: np.sum(x), fd[:, 13:15]).index(1)
        # print pos_choice_val
        nums_val = (pos_choice_val - 3) / 5  # num of shapes
        # print "nums(val): ",nums_val
        if fd[pos_choice_val, -3] == 1:  # choose right
            # print "choose right"
            choice_right_val = 1
    except ValueError:
        no_choice = 1
        choice_right_val = -1  # no choice
        pos_choice_val = 0
        nums_val = 0
    return reward_val, target_right_val, choice_right_val, pos_choice_val, nums_val, no_choice


class PaoDing:
    """
    Data preprocessor.
    """

    def __init__(self, task_name):
        self.plt = None
        self.task_name = task_name
        os.chdir("../")  # TODO: find out a more elegant solution for this import issue
        self.task = importlib.import_module('seqrnn_multitask.tasks.' + self.task_name)
        os.chdir("./holmes")
        self.analyst = self.task.TaskAnalytics()

        self.available_methods = dir(self.analyst)

        self.log = None
        self.hdf5_file = None
        self.index_records = None
        self.behavior_records = None
        self.prediction_records = None
        self.rawoutput_records = None
        self.neuron_records = None
        self.fig_path = None
        self.trial_briefs = None
        self.inputdate_records = None
        self.resetgate_records = None
        self.newgate_records = None

    def load_material(self, log_path, records_time, fig_path="../figs_m/"):
#        self.log = pd.read_csv(log_path + records_time + '-' + self.task_name + '.csv', index_col=0)
        self.hdf5_file = tables.open_file(log_path + records_time + '-' + self.task_name + '.hdf5', mode='r')
        self.index_records = self.hdf5_file.root.index
        self.behavior_records = self.hdf5_file.root.behavior
        self.prediction_records = self.hdf5_file.root.prediction
        self.rawoutput_records = self.hdf5_file.root.rawoutput
        self.neuron_records = self.hdf5_file.root.neuron
        self.fig_path = fig_path
        self.trial_briefs = pd.read_csv(log_path + records_time + '-' + self.task_name + '.validation_brief.csv', index_col=0)
        try:
            self.inputdate_records = self.hdf5_file.root.inputgate
            self.resetgate_records = self.hdf5_file.root.resetgate
            self.newgate_records = self.hdf5_file.root.newgate
        except:
            pass

    def get_neuron_behavior_pair(self, index, gates = False):
        """
        :param index: trial_index
        :return: fd: fake data by sensory sequence; gd: generated sensory prediction
        rd: raw output sequence of output neuron; nd: neuron activiy data
        """
        index_low, index_high = self.index_records[index, 0]
        fd = self.behavior_records[index_low: index_high]
        gd = self.prediction_records[index_low: index_high]
        rd = self.rawoutput_records[index_low: index_high]
        nd = self.neuron_records[index_low: index_high].squeeze()
        if gates:
            rg = self.resetgate_records[index_low: index_high].squeeze()
            ig = self.inputdate_records[index_low: index_high].squeeze()
            ng = self.newgate_records[index_low: index_high].squeeze()
            return fd, gd, rd, nd, rg, ig, ng
        else:
            return fd, gd, rd, nd

    def apply_method(self, method_name, kwargs):
        """
        may deprecated
        :param method_name:
        :param kwargs:
        :return:
        """
        if method_name in self.available_methods and isinstance(method_name, str):
            return getattr(self.analyst, method_name)(kwargs)
        else:
            raise AttributeError("The analyst does not support this method!")
