# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:27:27 2021

@author: Jesse
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

# 1. Visualising a 0 hidden layer (i.e. Linear Regression model) output for temperature.
# Load CMIP5 temperature data
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')

#Subsets
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
ridge = 10**4
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [0]
L2_norm = 0.5*ridge / (n_inputs*1)
model = tfk.Sequential([
      tfk.layers.Dense(1,
                        input_shape=(4050,),
                        kernel_regularizer = tfk.regularizers.L2(l2=L2_norm),
                        name='Linear_Regression_layer'
                        )
    ])
network = MLP(n_inputs,n_outputs,hidden_layers,ridge_penalty = ridge, param = param, date = datestr, model = model)

network.train(A_tr,Y_tr,n_iterations=1000,method=tfk.optimizers.Adam())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'tas', 'LinReg', datestr, closeplot = False)

network.save_weights()

w = network.model.layers[0].weights[0].numpy().reshape(45,90)

weightdata = netCDF4.Dataset('Data/Weight_maps/LinReg_tas_weights_10kL2.nc','w',format='NETCDF4')
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
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 temperature: weight for linear regression"'])
pygmt.makecpt(cmap='Code/BuRd.cpt',series=[-0.008,0.008])
fig.grdimage(grid='Data/Weight_maps/LinReg_tas_weights_10kL2.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.002"])
fig.show()
fig.savefig('Figures/Weight_maps/LinReg_tas_weights_10kL2.png')

# 2. Visualising a 0 hidden layer (i.e. Linear Regression model) output for precipitation.
# Load CMIP5 precipitation data
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')

#Subsets
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# Define and create BNN network.
ridge = 10**7
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [0]
L2_norm = 0.5*ridge / (n_inputs*1)
model = tfk.Sequential([
      tfk.layers.Dense(1,
                        input_shape=(4050,),
                        kernel_regularizer = tfk.regularizers.L2(l2=L2_norm),
                        name='Linear_Regression_layer'
                        )
    ])
network = MLP(n_inputs,n_outputs,hidden_layers,ridge_penalty = ridge, param = param, date = datestr, model = model)

network.train(A_tr,Y_tr,n_iterations=1000,method=tfk.optimizers.Adam())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'pr', 'LinReg',datestr, closeplot = False)

network.save_weights()

w = network.model.layers[0].weights[0].numpy().reshape(45,90)

weightdata = netCDF4.Dataset('Data/Weight_maps/LinReg_pr_weights_10ML2.nc','w',format='NETCDF4')
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
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight for linear regression"'])
pygmt.makecpt(cmap='Code/BuRd.cpt',series=[-.0006,.0006])
fig.grdimage(grid='Data/Weight_maps/LinReg_pr_weights_10ML2.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.0002"])
fig.show()
fig.savefig('Figures/Weight_maps/LinReg_pr_weights_10ML2.png')

# 3. Visualising a 0 hidden layer (i.e. Linear Regression model) output for temperature with removed global mean.
# Load CMIP5 temperature data
path = 'Data/CMIP5/'
param = 'tas'
models = sf.load_CMIP(path, param)

now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')

#Subsets
train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)
A_tr = A_tr - A_tr.mean(axis = 1)[:, np.newaxis]
A_te = A_te - A_te.mean(axis = 1)[:, np.newaxis]

# Define and create BNN network.
ridge = 10**4
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [0]
L2_norm = 0.5*ridge / (n_inputs*1)
model = tfk.Sequential([
      tfk.layers.Dense(1,
                        input_shape=(4050,),
                        kernel_regularizer = tfk.regularizers.L2(l2=L2_norm),
                        name='Linear_Regression_layer'
                        )
    ])
network = MLP(n_inputs,n_outputs,hidden_layers,ridge_penalty = ridge, param = param, date = datestr, model = model)

network.train(A_tr,Y_tr,n_iterations=1000,method=tfk.optimizers.Adam())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'tas', 'LinReg', datestr, closeplot = False)

network.save_weights()

w = network.model.layers[0].weights[0].numpy().reshape(45,90)

weightdata = netCDF4.Dataset('Data/Weight_maps/LinReg_tas_weights_10kL2_removedglobalmean.nc','w',format='NETCDF4')
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
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 temperature: weight for linear regression"'])
pygmt.makecpt(cmap='Code/BuRd.cpt',series=[-0.008,0.008])
fig.grdimage(grid='Data/Weight_maps/LinReg_tas_weights_10kL2_removedglobalmean.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.002"])
fig.show()
fig.savefig('Figures/Weight_maps/LinReg_tas_weights_10kL2_removedglobalmean.png')

# 4. Visualising a 0 hidden layer (i.e. Linear Regression model) output for precipitation with removed global mean.
# Load CMIP5 precipitation data
path = 'Data/CMIP5/'
param = 'pr'
models = sf.load_CMIP(path, param)

now = datetime.datetime.now()
datestr = now.strftime('%Y-%m-%d_%H%M')

#Subsets
train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)
A_tr = A_tr - A_tr.mean(axis = 1)[:, np.newaxis]
A_te = A_te - A_te.mean(axis = 1)[:, np.newaxis]

# Define and create BNN network.
ridge = 10**7
n_inputs = A_tr.shape[1]
n_outputs = 1
hidden_layers = [0]
L2_norm = 0.5*ridge / (n_inputs*1)
model = tfk.Sequential([
      tfk.layers.Dense(1,
                        input_shape=(4050,),
                        kernel_regularizer = tfk.regularizers.L2(l2=L2_norm),
                        name='Linear_Regression_layer'
                        )
    ])
network = MLP(n_inputs,n_outputs,hidden_layers,ridge_penalty = ridge, param = param, date = datestr, model = model)

network.train(A_tr,Y_tr,n_iterations=1000,method=tfk.optimizers.Adam())

# Compute and plot network predictions on testing & training subsets.
Yhat_tr = network.predict(A_tr)
Yhat_te = network.predict(A_te)
sf.plot_prediction(Y_tr, Yhat_tr, Y_te, Yhat_te, years, 'pr', 'LinReg',datestr, closeplot = False)

network.save_weights()

w = network.model.layers[0].weights[0].numpy().reshape(45,90)

weightdata = netCDF4.Dataset('Data/Weight_maps/LinReg_pr_weights_10ML2_removedglobalmean.nc','w',format='NETCDF4')
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
fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight for linear regression"'])
pygmt.makecpt(cmap='Code/BuRd.cpt',series=[-.0006,.0006])
fig.grdimage(grid='Data/Weight_maps/LinReg_pr_weights_10ML2_removedglobalmean.nc')
fig.coast(shorelines=True)
fig.colorbar(frame=["a0.0002"])
fig.show()
fig.savefig('Figures/Weight_maps/LinReg_pr_weights_10ML2_removedglobalmean.png')