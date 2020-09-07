#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:46:05 2019

@author: Zhewei Zhang

all analysis about shape task except lesion studies 
"""

import glob
import tkinter as tk
from tkinter import filedialog

print("start")
print("select the files")
# root = tk.Tk()
# root.withdraw()
# file_paths = filedialog.askopenfilenames(parent = root,
#                                         title = 'Choose a file',
#                                         filetypes = [("HDF5 files", "*.hdf5")]
#                                         )
file_paths = glob.glob('../log/RT/*.hdf5')
# In[]: plot loss and prediction accuracy
# Fig 3 a
from loss import plot_loss
plot_loss(file_paths)

# In[]: plot reaction time distribution and psychometric curve
# Fig 3 b/c

from toolkits_bhv import bhv_extract, plot_bhvBasic

# basic behavioral analysis
df_basic, df_logRT, df_psycv = bhv_extract(file_paths)
plot_bhvBasic(df_logRT, df_psycv)

# In[]: plot subjective value and shape order effect
# Fig 3 d/e
from toolkits_bhv import shape_extract, plot_subweight

# subjective value
df_epoch, df_subweight = shape_extract(file_paths)
plot_subweight(df_epoch, df_subweight)


# In[]
# Population responses of the units that are selective to accumulated evidence 
#   and urgency
# Fig 5 b/c
from PSTH import psth_extract, psth_plot

psth_resp = psth_extract(file_paths)
psth_plot(psth_resp)

# In[]: variance CE
# Fig 5 d
from variance import variance_extract, plot_variance

df_VarCE, df_signN = variance_extract(file_paths)
plot_variance(df_VarCE, df_signN)

# In[]: plot the shape preidction
# Fig 7
from pred_shape import shape_prediction, plot_shapePrediction

# shape prediction
df_pred = shape_prediction(file_paths)
plot_shapePrediction(df_pred)




