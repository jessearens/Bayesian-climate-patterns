# -*- coding: utf-8 -*-
"""
Script which trains and tests the CMIP5 datasets with a Bayesian Neural Network (BNN).

Also visualises prediction results of the BNN.
Finally, this script also trains a 1-unit BNN for visualisation of indicator patterns and uncertainty.

Created on Fri Aug 20 2021
Last edit on Wed Dec 15 3 2021

@author: Jesse Arens
"""

## General imports
import os
import datetime
import time

# Third-party imports
import numpy as np
#import matplotlib.pyplot as plt
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


# 1. Testing BNN class on CMIP5 temperature data.
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
hidden_layers = [10, 10]
network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description = datestr, kl_weight = 10**-4, prior = 'Laplace_tas')
tic = time.process_time()
network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())
toc = time.process_time()
timestr = f"1,{toc-tic}\n"

# Compute and plot network predictions on testing & training subsets.
sample_size = 10000
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)

network.save_weights()

# 2. Testing BNN class on CMIP5 precipitation data.
# Load CMIP5 precipitation data.
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets, with leaky-relu and MSE loss.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [10, 10]
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description = datestr, kl_weight = 10**-3, prior = 'Laplace_pr')
tic = time.process_time()
network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())
toc = time.process_time()
timestr += f"2,{toc-tic}\n"

# Compute and plot network predictions on testing & training subsets.
sample_size = 10000
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)

network.save_weights()

# 3. Visualising a 1 unit 1 layer relevance plot for temperature.
# Load CMIP5 temperature data
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')+'_1unit_'

# Define and create testing and training subsets.
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [1]

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps = 100,
    decay_rate= 0.96,
    staircase = True)
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description = datestr, kl_weight=1/4050, prior = 'Student_tas')
tic = time.process_time()
network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adam(lr_schedule), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())
toc = time.process_time()
timestr += f"3,{toc-tic}\n"

sample_size = 10000
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)

network.save_weights()

dummy_input = np.array([0])
model_posterior = network.model.layers[0]._posterior(dummy_input)
w_mean = model_posterior.mean().numpy()[:-1].reshape(45,90)
w_var = model_posterior.variance().numpy()[:-1].reshape(45,90)

meandata = netCDF4.Dataset('Data/Weight_maps/BNN_tas_1unit_meanweights.nc','w',format='NETCDF4')
meandata.createDimension("lat",45)
lat = meandata.createVariable("lat","float64",dimensions="lat");
lat[:] = models['lats']
meandata.createDimension("lon",90)
lon = meandata.createVariable("lon","float64",dimensions="lon")
lon[:] = models['lons']
grd_mean = meandata.createVariable("mean","float32",dimensions=("lat","lon"))
grd_mean[:] = w_mean
meandata.close()

vardata = netCDF4.Dataset('Data/Weight_maps/BNN_tas_1unit_varweights.nc','w',format='NETCDF4')
vardata.createDimension("lat",45)
lat = vardata.createVariable("lat","float64",dimensions="lat");
lat[:] = models['lats']
vardata.createDimension("lon",90)
lon = vardata.createVariable("lon","float64",dimensions="lon")
lon[:] = models['lons']
grd_mean = vardata.createVariable("var","float32",dimensions=("lat","lon"))
grd_mean[:] = w_var
vardata.close()

fig = pygmt.Figure()
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 temperature: weight mean for 1 hidden unit"'])
pygmt.makecpt(cmap='Code/Core/BuRd.cpt',series=[-.3,.3])
fig.grdimage(grid='Data/Weight_maps/BNN_tas_1unit_meanweights.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.1"])
fig.show()
fig.savefig('Figures/Weight_maps/BNN_tas_1unit_meanweights.png')

fig = pygmt.Figure()
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 temperature: weight variance for 1 hidden unit"'])
pygmt.makecpt(cmap='Code/Core/Greys.cpt', series=[0.0,.9])
fig.grdimage(grid='Data/Weight_maps/BNN_tas_1unit_varweights.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.1"])
fig.show()
fig.savefig('Figures/Weight_maps/BNN_tas_1unit_varweights.png')

# 4. Visualising a 1 unit 1 layer relevance plot for precipitation.
# Load CMIP5 precipitation data
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

# Define and create testing and training subsets.
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [1]

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps = 100,
    decay_rate = 0.96,
    staircase = True)
network = BNN(n_inputs, n_outputs, hidden_layers, param = param, description = datestr, kl_weight = 10**-4, prior = 'Student_pr')
tic = time.process_time()
network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adam(lr_schedule), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())
toc = time.process_time()
timestr += f"4,{toc-tic}\n"

sample_size = 10000
Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)

network.save_weights()

dummy_input = np.array([0])
model_posterior = network.model.layers[0]._posterior(dummy_input)
w_mean = model_posterior.mean().numpy()[:-1].reshape(45,90)
w_var = model_posterior.variance().numpy()[:-1].reshape(45,90)

meandata = netCDF4.Dataset('Data/Weight_maps/BNN_pr_1unit_meanweights.nc','w',format='NETCDF4')
meandata.createDimension("lat",45)
lat = meandata.createVariable("lat","float64",dimensions="lat");
lat[:] = models['lats']
meandata.createDimension("lon",90)
lon = meandata.createVariable("lon","float64",dimensions="lon")
lon[:] = models['lons']
grd_mean = meandata.createVariable("mean","float32",dimensions=("lat","lon"))
grd_mean[:] = w_mean
meandata.close()

vardata = netCDF4.Dataset('Data/Weight_maps/BNN_pr_1unit_varweights.nc','w',format='NETCDF4')
vardata.createDimension("lat",45)
lat = vardata.createVariable("lat","float64",dimensions="lat");
lat[:] = models['lats']
vardata.createDimension("lon",90)
lon = vardata.createVariable("lon","float64",dimensions="lon")
lon[:] = models['lons']
grd_mean = vardata.createVariable("var","float32",dimensions=("lat","lon"))
grd_mean[:] = w_var
vardata.close()

fig = pygmt.Figure()
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight mean for 1 hidden unit"'])
pygmt.makecpt(cmap='Code/Core/BrBG.cpt',series=[-.4,.4])
fig.grdimage(grid='Data/Weight_maps/BNN_pr_1unit_meanweights_v2.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.2"])
fig.show()
fig.savefig('Figures/Weight_maps/BNN_pr_1unit_meanweights.png')

fig = pygmt.Figure()
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight variance for 1 hidden unit"'])
pygmt.makecpt(cmap='Code/Core/Greys.cpt', series=[0.0,.8])
fig.grdimage(grid='Data/Weight_maps/BNN_pr_1unit_varweights_v2.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.1"])
fig.show()
fig.savefig('Figures/Weight_maps/BNN_pr_1unit_varweights.png')

file = 'Data/BNN_CPUtime.txt'
f = open(file,'w')
f.write(timestr)
f.close()
