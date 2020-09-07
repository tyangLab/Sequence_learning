# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:24:20 2020

@author: Zhewei Zhang
"""
import glob


# In[0]: find the when and which units, and save the results
from toolkits_whenwhich import model_loading, model_select
from toolkits_whenwhich import when_which_constructer, datasaving

neuron_proportion = 0.5
# file_paths = model_select()
file_paths = glob.glob('../save/RT/*.pt')

# loop over each model
for file_path in file_paths:
    # load the .pt files
    ho_weight, w_hr, _, _ = model_loading(file_path)
    # using hidden-output connections to find out the when/which units
    when, which = when_which_constructer(ho_weight,neuron_proportion)
    # save the identity of when/which model
    datasaving(file_path, when, which, w_hr, ho_weight, neuron_proportion)
    
# In[1]: plot geodesic distance, maxmimum flow, and output connection pattern
from toolkits_whenwhich import wh_select, graph_extract
from toolkits_whenwhich import gesdes_plot, maxflow_plot, conpattern_plot
# load the previous files
# wh_filepath = wh_select()
wh_filepath = glob.glob('../save/RT/*WhenWhich_*threshold0.5*.yaml')
# calulating geodesic distance, maxmimum flow, and output connection pattern
df_output, df_dInv, df_mflow = graph_extract(wh_filepath)
# plot geodesic distance
gesdes_plot(df_dInv)
# plot maxmimum flow
maxflow_plot(df_mflow)
# plot output connection pattern
conpattern_plot(df_output)

# In[2] when and which lesion 
# from toolkits_whenwhich import lesionBhv_selection

from toolkits_whenwhich import lesion_extract, plot_basicBhv_lesion
# load lesion files
# lesion_files = lesionBhv_selection()
lesion_files = glob.glob('../log/RT/lesion/*-0.5-*.hdf5')

df_lesion, condition, ncondition = lesion_extract(lesion_files)
# plot the lesion effect on reaction time and choices
plot_basicBhv_lesion(df_lesion, condition, ncondition)

# In[3] speed-accuracy trade-off 
from toolkits_whenwhich import speed_accuracy_extract, plot_speed_accuracy

cr, cr_sem, rt, rt_sem, cr_log, cr_log_sem = speed_accuracy_extract()
plot_speed_accuracy(cr, cr_sem, rt, rt_sem, cr_log, cr_log_sem)

