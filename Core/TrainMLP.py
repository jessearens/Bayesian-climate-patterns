# -*- coding: utf-8 -*-
"""
Script which trains and tests the CMIP5 datasets with a Multi-Layer Perceptron (MLP).

Also visualises prediction results of the MLP.

Created on Tue Aug 17 2021
Last edit on Fri Sep 3 2021

@author: Jesse Arens
"""

## General imports
import os
import datetime
import time

# Third-party imports
import numpy as np
import pygmt
import netCDF4
import tensorflow as tf
tfk = tf.keras

# Local imports
import Core.supportfuncs as sf
from Core.MLP import MLP

# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')
now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')
load_old = True # describes if an existing (trained) network can be loaded, or a new network needs to be trained.


#1. Testing MLP class on CMIP5 temperature data.
# Load CMIP5 temperature data.
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create MLP network.
ridge = 10**4
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = MLP(n_inputs, n_outputs, hidden_layers, ridge_penalty = ridge, param = param, date = datestr)
tic = time.process_time()
network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adadelta(learning_rate=0.001), overwrite = load_old)
toc = time.process_time()
timestr = f"1,{toc-tic}\n"

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'tas', 'MLP',datestr, closeplot = False)

network.save_weights()

#2. Testing MLP class on CMIP5 precipitation data.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create MLP network.
ridge = 10**7
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = MLP(n_inputs, n_outputs, hidden_layers, ridge_penalty = ridge, param = param, date = datestr)
tic = time.process_time()
network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adadelta(learning_rate = 0.001), overwrite = load_old)
toc = time.process_time()
timestr += f"2,{toc-tic}\n"

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'pr', 'MLP',datestr,closeplot = False)

network.save_weights()


# 3. Visualising a 1 unit 1 layer relevance plot for temperature.
# Load CMIP5 temperature data
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')

#Subsets
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create MLP network.
ridge = 10**4
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [1]
network = MLP(n_inputs,n_outputs,hidden_layers,ridge_penalty = ridge, param = param, date = datestr)
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.98,
    staircase=True)
tic = time.process_time()
network.train(A_tr,Y_tr,n_iterations=2000,method=tfk.optimizers.Adam(learning_rate=lr_schedule),overwrite=load_old)
toc = time.process_time()
timestr +=f"3,{toc-tic}\n"

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'tas', 'MLP',datestr, closeplot = False)

network.save_weights()

w = network.model.layers[0].weights[0].numpy().reshape(45,90)

weightdata = netCDF4.Dataset('Data/Weight_maps/MLP_tas_1unit_weights.nc','w',format='NETCDF4')
weightdata.createDimension("lat",45)
lat = weightdata.createVariable("lat","float64",dimensions="lat");
lat[:] = models['lats']
weightdata.createDimension("lon",90)
lon = weightdata.createVariable("lon","float64",dimensions="lon")
lon[:] = models['lons']
weight = weightdata.createVariable("weight","float32",dimensions=("lat","lon"))
weight[:] = w
weightdata.close()

fig = pygmt.Figure()
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 temperature: weight for 1 hidden unit"'])
pygmt.makecpt(cmap='Code/BuRd.cpt',series=[-.007,.007])
fig.grdimage(grid='Data/Weight_maps/MLP_tas_1unit_weights.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.002"])
fig.show()
fig.savefig('Figures/Weight_maps/MLP_tas_1unit_weights.png')

# 4. Visualising a 1 unit 1 layer relevance plot for precipitation
# Load CMIP5 precipitation data
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')

#Subsets
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create MLP network.
ridge = 10**7
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [1]
network = MLP(n_inputs,n_outputs,hidden_layers,ridge_penalty = ridge, param = param, date = datestr)
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)
tic = time.process_time()
network.train(A_tr,Y_tr,n_iterations=2000,method=tfk.optimizers.Adam(learning_rate=lr_schedule),overwrite=load_old)
toc = time.process_time()
timestr += f"4,{toc-tic}\n"

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'pr', 'MLP',datestr, closeplot = False)

network.save_weights()

w = network.model.layers[0].weights[0].numpy().reshape(45,90)

weightdata = netCDF4.Dataset('Data/Weight_maps/MLP_pr_1unit_weights.nc','w',format='NETCDF4')
weightdata.createDimension("lat",45)
lat = weightdata.createVariable("lat","float64",dimensions="lat");
lat[:] = models['lats']
weightdata.createDimension("lon",90)
lon = weightdata.createVariable("lon","float64",dimensions="lon")
lon[:] = models['lons']
weight = weightdata.createVariable("weight","float32",dimensions=("lat","lon"))
weight[:] = w
weightdata.close()

fig = pygmt.Figure()
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight for 1 hidden unit"'])
pygmt.makecpt(cmap='Code/BrBG.cpt',series=[-.0004,.0004])
fig.grdimage(grid='Data/Weight_maps/MLP_pr_1unit_weights.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.0001"])
fig.show()
fig.savefig('Figures/Weight_maps/MLP_pr_1unit_weights.png')

file = 'Data/MLP_CPUtime.txt'
f = open(file,'w')
f.write(timestr)
f.close()