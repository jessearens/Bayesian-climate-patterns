# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:15:54 2022

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
import Core.supportfuncs as sf
from Tests.BNN_testing import BNN

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
sample_size = 1000
n_iterations = 2000

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps = 100,
    decay_rate = 0.96,
    staircase = True)

#KL_list =  [0.000001, 0.00001, 0.0001, 1/4050, 0.001, 0.01, 0.1, 1, 10, 100]
KL_list =  [0.0001, 1/4050, 0.001, 0.01]
hidden_layers = [1]
ll_tas_1unit = {}
n_train = 23; n_test = 6;
MAE_tas_1unit = np.zeros((len(KL_list),4)); i = 0

for kl_weight in KL_list:
    datestr = 'tas_KLtest_' +str(kl_weight)+'_1unit_'
    network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'student_complex', kl_weight = kl_weight)
    network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(lr_schedule), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

    # Compute and plot network predictions on testing & training subsets.
    Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
    Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, Y_te, sample_size, compute_ll = True)
    sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
    MAE_tas_1unit[i,:] = sf.compute_stats(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, n_train, n_test)
    ll_tas_1unit.update({datestr:network.ll_score})
    i +=1


    dummy_input = np.array([0])
    model_posterior = network.model.layers[0]._posterior(dummy_input)
    w_mean = model_posterior.mean().numpy()[:-1].reshape(45,90)
    w_var = model_posterior.variance().numpy()[:-1].reshape(45,90)

    meandata = netCDF4.Dataset('Data/Weight_maps/'+datestr+'_meanweights.nc','w',format='NETCDF4')
    meandata.createDimension("lat",45)
    lat = meandata.createVariable("lat","float64",dimensions="lat");
    lat[:] = models['lats']
    meandata.createDimension("lon",90)
    lon = meandata.createVariable("lon","float64",dimensions="lon")
    lon[:] = models['lons']
    grd_mean = meandata.createVariable("mean","float32",dimensions=("lat","lon"))
    grd_mean[:] = w_mean
    meandata.close()

    vardata = netCDF4.Dataset('Data/Weight_maps/'+datestr+'_varweights.nc','w',format='NETCDF4')
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
    pygmt.makecpt(cmap='Code/Core/BuRd.cpt',series=[-.1,.1])
    fig.grdimage(grid='Data/Weight_maps/'+datestr+'_meanweights.nc')
    fig.coast(shorelines=True)
    fig.colorbar(frame=["a0.2"])
    fig.show()
    fig.savefig('Figures/Weight_maps/'+datestr+'_meanweights.png')

    fig = pygmt.Figure()
    fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight variance for 1 hidden unit"'])
    pygmt.makecpt(cmap='Code/Core/Greys.cpt', series=[0.0,.1])
    fig.grdimage(grid='Data/Weight_maps/'+datestr+'_varweights.nc')
    fig.coast(shorelines=True)
    fig.colorbar(frame=["a0.1"])
    fig.show()
    fig.savefig('Figures/Weight_maps/'+datestr+'_varweights.png')

# np.savetxt('Data/BNN_'+param+'_kltest_MAE_1unit_v2.txt',MAE_tas_1unit,
#            header='forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE',
#            fmt = '%.2f',
#            delimiter=',')

# with open("Data/tas_kltest_results_1unit_v2.txt", 'w') as f:
#     for key, value in ll_tas_1unit.items():
#         f.write('%s:%s\n' % (key, value))


# # A. Temperature

# # Define and create BNN network.

# hidden_layers = [10,10]
# ll_tas = {}
# MAE_tas = np.zeros((len(KL_list),4)); i = 0

# for kl_weight in KL_list:
#     datestr = 'tas_KLtest_' +str(kl_weight)
#     network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'Laplace_complex', kl_weight = kl_weight)
#     network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

#     # Compute and plot network predictions on testing & training subsets.
#     Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
#     Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, Y_te, sample_size, compute_ll = True)
#     sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
#     MAE_tas[i,:] = sf.compute_stats(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, n_train, n_test)
#     ll_tas.update({datestr:network.ll_score})
#     i +=1

# np.savetxt('Data/BNN_'+param+'_kltest_MAE_v2.txt',MAE_tas,
#                header='forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE',
#                fmt = '%.2f',
#                delimiter=',')

# with open("Data/tas_kltest_results_v2.txt", 'w') as f:
#     for key, value in ll_tas.items():
#         f.write('%s:%s\n' % (key, value))

# # B Precipitation
# # 1. Testing BNN class on CMIP5 precipitation data. > normal unit gaussian prior
# # Load CMIP5 temperature data.
# path = 'Data/CMIP5/'
# param = 'pr'
# models = sf.load_CMIP(path, param)

# # Define and create testing and training subsets.
# train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
# A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# # Define and create BNN network.
# n_inputs = A_tr.shape[1]
# n_outputs = 1

# hidden_layers = [10,10]

# ll_pr = {}
# n_train = 18; n_test = 4;
# MAE_pr = np.zeros((len(KL_list),4)); i = 0

# for kl_weight in KL_list:
#     datestr = 'pr_KLtest_' + str(kl_weight)
#     network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'laplace_complex_pr', kl_weight = kl_weight)
#     network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

#     # Compute and plot network predictions on testing & training subsets.
#     Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
#     Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, Y_te, sample_size, compute_ll = True)
#     sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
#     MAE_pr[i,:] = sf.compute_stats(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, n_train, n_test)
#     ll_pr.update({datestr:network.ll_score})
#     i +=1

# np.savetxt('Data/BNN_'+param+'_kltest_MAE_v2.txt',MAE_pr,
#                 header='forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE',
#                 fmt = '%.2f',
#                 delimiter=',')

# with open("Data/pr_kltest_results_v2.txt", 'w') as f:
#     for key, value in ll_pr.items():
#         f.write('%s:%s\n' % (key, value))

# # B Precipitation
# # 1. Testing BNN class on CMIP5 precipitation data. > normal unit gaussian prior
# # Load CMIP5 temperature data.
# path = 'Data/CMIP5/'
# param = 'pr'
# models = sf.load_CMIP(path, param)

# # Define and create testing and training subsets.
# train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
# A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

# # Define and create BNN network.
# n_inputs = A_tr.shape[1]
# n_outputs = 1

# hidden_layers = [1]
# ll_pr_1unit = {}
# n_train = 18; n_test = 4;
# MAE_pr_1unit = np.zeros((len(KL_list),4)); i = 0

# for kl_weight in KL_list:
#     datestr = 'pr_KLtest_' + str(kl_weight)+'_1unit_'
#     network = BNN(n_inputs,n_outputs,hidden_layers,param = param, date = datestr, prior = 'student_complex_pr', kl_weight = kl_weight)
#     network.train(A_tr, Y_tr, n_iterations = n_iterations, method = tfk.optimizers.Adam(lr_schedule), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())

#     # Compute and plot network predictions on testing & training subsets.
#     Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
#     Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, Y_te, sample_size, compute_ll = True)
#     sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = True)
#     MAE_pr_1unit[i,:] = sf.compute_stats(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, n_train, n_test)
#     ll_pr_1unit.update({datestr:network.ll_score})
#     i +=1

#     dummy_input = np.array([0])
#     model_posterior = network.model.layers[0]._posterior(dummy_input)
#     w_mean = model_posterior.mean().numpy()[:-1].reshape(45,90)
#     w_var = model_posterior.variance().numpy()[:-1].reshape(45,90)

#     meandata = netCDF4.Dataset('Data/Weight_maps/'+datestr+'_meanweights.nc','w',format='NETCDF4')
#     meandata.createDimension("lat",45)
#     lat = meandata.createVariable("lat","float64",dimensions="lat");
#     lat[:] = models['lats']
#     meandata.createDimension("lon",90)
#     lon = meandata.createVariable("lon","float64",dimensions="lon")
#     lon[:] = models['lons']
#     grd_mean = meandata.createVariable("mean","float32",dimensions=("lat","lon"))
#     grd_mean[:] = w_mean
#     meandata.close()

#     vardata = netCDF4.Dataset('Data/Weight_maps/'+datestr+'_varweights.nc','w',format='NETCDF4')
#     vardata.createDimension("lat",45)
#     lat = vardata.createVariable("lat","float64",dimensions="lat");
#     lat[:] = models['lats']
#     vardata.createDimension("lon",90)
#     lon = vardata.createVariable("lon","float64",dimensions="lon")
#     lon[:] = models['lons']
#     grd_mean = vardata.createVariable("var","float32",dimensions=("lat","lon"))
#     grd_mean[:] = w_var
#     vardata.close()

#     fig = pygmt.Figure()
#     fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight mean for 1 hidden unit"'])
#     pygmt.makecpt(cmap='Code/BuRd.cpt',series=[-.2,.2])
#     fig.grdimage(grid='Data/Weight_maps/'+datestr+'_meanweights.nc')
#     fig.coast(shorelines=True)
#     fig.colorbar(frame=["a0.2"])
#     fig.show()
#     fig.savefig('Figures/Weight_maps/'+datestr+'_meanweights.png')

#     fig = pygmt.Figure()
#     fig.basemap(region="g",projection="N20c",frame=['a','+t"CMIP5 precipitation: weight variance for 1 hidden unit"'])
#     pygmt.makecpt(cmap='Code/Greys.cpt', series=[0.0,2])
#     fig.grdimage(grid='Data/Weight_maps/'+datestr+'_varweights.nc')
#     fig.coast(shorelines=True)
#     fig.colorbar(frame=["a0.1"])
#     fig.show()
#     fig.savefig('Figures/Weight_maps/'+datestr+'_varweights.png')


# np.savetxt('Data/BNN_'+param+'_kltest_MAE_1unit_v2.txt',MAE_pr_1unit,
#                 header='forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE',
#                 fmt = '%.2f',
#                 delimiter=',')

# with open("Data/pr_kltest_results_1unit_v2.txt", 'w') as f:
#     for key, value in ll_pr_1unit.items():
#         f.write('%s:%s\n' % (key, value))
