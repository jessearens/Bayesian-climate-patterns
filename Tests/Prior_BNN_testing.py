# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:57:33 2022

@author: jesse
"""


## General imports
import os
import datetime
import time

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import pygmt
import netCDF4
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfd = tfp.distributions

# Local imports
import supportfuncs as sf
from BNN_testing import BNN


# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')
now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')
load_old = False # describes if an existing (trained) network can be loaded, or a new network needs to be trained.

# A. Temperature
# 1. Testing BNN class on CMIP5 temperature data. > normal unit gaussian prior
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10,10]
sample_size = 200
n_iterations = 500
datestr = 'unit_gaussian_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'prior')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())
ll = {}

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)
ll.update({datestr:network.ll_score})

# 2. Testing BNN class on CMIP5 temperature data. > simple MLP-altered gaussian prior
# Define and create BNN network.
datestr = 'simple_MLPtas_gaussian_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'MLPtas')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)
ll.update({datestr:network.ll_score})

# 3. Testing BNN class on CMIP5 temperature data. > complex MLP-altered gaussian prior
# Define and create BNN network.
datestr = 'complex_MLPtas_gaussian_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'tas_complex')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll.update({datestr:network.ll_score})

# 4. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'unit_laplace_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'laplace')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll.update({datestr:network.ll_score})

# 5. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'complex_MLPtas_laplace_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'laplace_complex')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll.update({datestr:network.ll_score})

# 6. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'unit_stutent_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'Student')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.

Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll.update({datestr:network.ll_score})

# 7. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'complex_student_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'student_complex')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll.update({datestr:network.ll_score})

# B Precipitation
# 1. Testing BNN class on CMIP5 precipitation data. > normal unit gaussian prior
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10,10]
sample_size = 200
n_iterations = 500
datestr = 'unit_gaussian_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'prior')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())
ll_pr = {}

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll_pr.update({datestr:network.ll_score})

# 2. Testing BNN class on CMIP5 temperature data. > simple MLP-altered gaussian prior
# Define and create BNN network.
datestr = 'simple_MLPpr_gaussian_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'MLPpr')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)
ll_pr.update({datestr:network.ll_score})

# 3. Testing BNN class on CMIP5 temperature data. > complex MLP-altered gaussian prior
# Define and create BNN network.
datestr = 'complex_MLPpr_gaussian_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'pr_complex')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll_pr.update({datestr:network.ll_score})

# 4. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'unit_laplace_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'laplace')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll_pr.update({datestr:network.ll_score})

# 5. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'complex_MLPpr_laplace_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'laplace_complex_pr')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll_pr.update({datestr:network.ll_score})

# 6. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'unit_stutent_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'Student')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.

Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll_pr.update({datestr:network.ll_score})

# 7. Testing BNN class on CMIP5 temperature data. > Unit Laplace prior
# Define and create BNN network.
datestr = 'complex_student_prior_1unit_'
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'student_complex_pr')
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size, compute_ll = True)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
ll_pr.update({datestr:network.ll_score})

with open("Data/tas_prior_results_v2.txt", 'w') as f:
    for key, value in ll.items():
        f.write('%s:%s\n' % (key, value))

with open("Data/pr_prior_results_v2.txt", 'w') as f:
    for key, value in ll_pr.items():
        f.write('%s:%s\n' % (key, value))

