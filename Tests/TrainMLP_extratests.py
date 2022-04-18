# -*- coding: utf-8 -*-
"""
Script which trains and tests the CMIP5 datasets with a Multi-Layer Perceptron (MLP).
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
import pygmt
import netCDF4
import tensorflow as tf
tfk = tf.keras

# Local imports
import supportfuncs as sf
from MLP import MLP

# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')
now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')
load_old = False # describes if an existing (trained) network can be loaded, or a new network needs to be trained.

#5. Testing MLP class on CMIP5 temperature data with removed global mean.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)
A_tr = A_tr - A_tr.mean(axis = 1)[:, np.newaxis]
A_te = A_te - A_te.mean(axis = 1)[:, np.newaxis]

# Define and create MLP network.
ridge = 10**4
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = MLP(n_inputs, n_outputs, hidden_layers, ridge_penalty = ridge, param = param, date = datestr)

network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adadelta(learning_rate=0.001), overwrite = load_old)

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'tas', 'MLP',datestr, closeplot = False)

network.save_weights()

#6. Testing MLP class on CMIP5 precipitation data with removed global mean.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)
A_tr = A_tr - A_tr.mean(axis = 1)[:, np.newaxis]
A_te = A_te - A_te.mean(axis = 1)[:, np.newaxis]

# Define and create MLP network.
ridge = 10**7
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = MLP(n_inputs, n_outputs, hidden_layers, ridge_penalty = ridge, param = param, date = datestr)

network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adadelta(learning_rate = 0.001), overwrite = load_old)

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'pr', 'MLP',datestr,closeplot = False)

network.save_weights()
