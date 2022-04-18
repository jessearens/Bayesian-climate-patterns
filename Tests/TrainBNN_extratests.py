# -*- coding: utf-8 -*-
"""
Script which trains and tests the CMIP5 datasets with a Bayesian Neural Network (BNN).
Runs on some alternative testing conditions to validate some of the general results.

Created on Thu Dec 16 2021
Last edit on Thu Dec 16 2021

@author: Jesse Arens
"""

## General imports
import os
import datetime

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
import Core.supportfuncs as sf
from Core.BNN import BNN

# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')
now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')
load_old = False # describes if an existing (trained) network can be loaded, or a new network needs to be trained.

kl_tas = 10**-4
prior_tas = 'Laplace_tas'
kl_pr = 10**-3
prior_pr = 'Laplace_pr'
n_iterations = 2000
sample_size = 1000

# 1. Testing BNN class on CMIP5 temperature data, with removed global mean.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)
A_tr = A_tr - A_tr.mean(axis = 1)[:, np.newaxis]
A_te = A_te - A_te.mean(axis = 1)[:, np.newaxis]

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size

Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 2. Testing BNN class on CMIP5 precipitation data, with removed global mean.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)
A_tr = A_tr - A_tr.mean(axis = 1)[:, np.newaxis]
A_te = A_te - A_te.mean(axis = 1)[:, np.newaxis]

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 3.a. Testing BNN class on CMIP5 temperature data, with different test/train subsets.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'subseta_'

# Define and create testing and training subsets.
train_mask = [1, 1, 1, 1, 1, 0, 0 ,1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 3.b. Testing BNN class on CMIP5 temperature data, with different test/train subsets.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'subsetb_'

# Define and create testing and training subsets.
train_mask = [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 3.c. Testing BNN class on CMIP5 temperature data, with different test/train subsets.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'subsetc_'

# Define and create testing and training subsets.
train_mask = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 3.d. Testing BNN class on CMIP5 temperature data, with different test/train subsets.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'subsetd_'

# Define and create testing and training subsets.
train_mask = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 3.e. Testing BNN class on CMIP5 temperature data, with different test/train subsets.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'subsete_'

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 3.f. Testing BNN class on CMIP5 temperature data, with different test/train subsets.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'subsetf_'

# Define and create testing and training subsets.
train_mask = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 4.a. Testing BNN class on CMIP5 precipitation data, with different test/train subsets.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'subseta_'
kl_weight = 10**-3
prior = 'Laplace_pr'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 4.b. Testing BNN class on CMIP5 precipitation data, with different test/train subsets.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'subsetb_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 4.c. Testing BNN class on CMIP5 precipitation data, with different test/train subsets.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'subsetc_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 4.d. Testing BNN class on CMIP5 precipitation data, with different test/train subsets.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'subsetd_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 4.e. Testing BNN class on CMIP5 precipitation data, with different test/train subsets.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'subsete_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 4.f. Testing BNN class on CMIP5 precipitation data, with different test/train subsets.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'subsetf_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 5.a. Testing BNN class on CMIP5 temperature data, with different network architectures.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'archa_'
kl_weight = 10**-4
prior = 'Laplace_tas'

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 5.b. Testing BNN class on CMIP5 temperature data, with different network architectures.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'archb_'

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10, 10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 5.c. Testing BNN class on CMIP5 temperature data, with different network architectures.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'archc_'

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [5, 5]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 5.d. Testing BNN class on CMIP5 temperature data, with different network setups.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'archd_'
# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [20,20]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 5.e. Testing BNN class on CMIP5 temperature data, with different network setups.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'arche_'
# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [50,50]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 5.f. Testing BNN class on CMIP5 temperature data, with different network setups.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)
datestr = 'archf_'
# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [100, 100]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description =  datestr, kl_weight = kl_tas, prior = prior_tas)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 6.a. Testing BNN class on CMIP5 precipitation data, with different network architecture.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'archa_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 6.b. Testing BNN class on CMIP5 precipitation data, with different network architecture.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'archb_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10, 10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 6.c. Testing BNN class on CMIP5 precipitation data, with different network architecture.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'archc_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [5, 5]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 6.d. Testing BNN class on CMIP5 precipitation data, with different network architecture.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'archd_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [20, 20]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()

# 6.e. Testing BNN class on CMIP5 precipitation data, with different network architecture.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'arche_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [50,50]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()


# 6.f. Testing BNN class on CMIP5 precipitation data, with different network architecture.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)
datestr = 'archf_'

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [100, 100]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description =  datestr, kl_weight = kl_pr, prior = prior_pr)
network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

# Compute and plot network predictions on testing & training subsets.
sample_size = sample_size
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)

network.save_weights()