# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:38:07 2021

@author: jesse
"""

import os
import datetime

# Third-party imports
import numpy as np
import netCDF4
from scipy.stats.stats import pearsonr

# Local imports

# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')
now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')

mlp_mean_tas = netCDF4.Dataset('Data/Weight_maps/MLP_tas_1unit_weights.nc')
bnn_mean_tas = netCDF4.Dataset('Data/Weight_maps/BNN_tas_1unit_meanweights_v2.nc')
bnn_var_tas = netCDF4.Dataset('Data/Weight_maps/BNN_tas_1unit_varweights_v2.nc')

pear_tas = pearsonr(mlp_mean_tas.variables['weight'][:].data.flatten(),bnn_mean_tas.variables['mean'][:].data.flatten())
ratio_tas = np.mean(bnn_var_tas.variables['var'][:]) / np.mean(np.abs(bnn_mean_tas.variables['mean']))

mlp_mean_pr = netCDF4.Dataset('Data/Weight_maps/MLP_pr_1unit_weights.nc')
bnn_mean_pr = netCDF4.Dataset('Data/Weight_maps/BNN_pr_1unit_meanweights_v2.nc')
bnn_var_pr = netCDF4.Dataset('Data/Weight_maps/BNN_pr_1unit_varweights_v2.nc')

pear_pr = pearsonr(mlp_mean_pr.variables['weight'][:].data.flatten(),bnn_mean_pr.variables['mean'][:].data.flatten())
ratio_pr = np.mean(bnn_var_pr.variables['var'][:]) / np.mean(np.abs(bnn_mean_pr.variables['mean']))
