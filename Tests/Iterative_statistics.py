# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:04:48 2022

@author: jesse
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
from Core.BNN import BNN

# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')
now = datetime.datetime.now()
datestr = 'iterative'+ now.strftime('%Y-%m-%d_%H%M')
load_old = False # describes if an existing (trained) network can be loaded, or a new network needs to be trained.


def iterate_MLP(param,n_iterations):
    # Load and prepare CMIP5 data and used dataset-dependent NN parameters.
    models = sf.load_CMIP('Data/CMIP5/',param)
    if (param == 'tas'):
        n_test = 6;
        n_train = 23;
        ridge = 10**4
        train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    else:
        n_test  = 4
        n_train = 18;
        ridge = 10**7
        train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
    A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

    # Initiate storage arrays
    YOD = np.zeros((n_iterations,n_test)); runtime = np.zeros((n_iterations,2));
    MAE_stats = np.zeros((n_iterations,4));
#    w0 = np.zeros((4050,10,n_iterations)); #b0 = np.zeros((10,n_iterations));
#    w1 = np.zeros((10,10,n_iterations)); #b1 = np.zeros((10,n_iterations));
#    w2 = np.zeros((10,1,n_iterations)); #b2 = np.zeros((n_iterations))

    # Run iteration
    for i in range(n_iterations):
        print('Now working on iteration ' + str(i))

        # Define and create MLP network.
        n_inputs = A_tr.shape[1]
        n_outputs = 1
        hidden_layers = [10, 10]
        network = MLP(n_inputs, n_outputs, hidden_layers, ridge_penalty = ridge, param = param, date = datestr)
        tic = time.process_time()
        network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adadelta(learning_rate=0.001))
        toc = time.process_time()

        # Compute and plot network predictions on testing & training subsets.
        Yhat_tr = network.predict(A_tr)
        Yhat_te = network.predict(A_te)
        tac = time.process_time()

        # Compute the desired statistics
        YOD[i,:] = sf.find_YOD(Y_te,Yhat_te)
        runtime[i,0] = toc-tic;
        MAE_stats[i,:] = sf.compute_stats(Y_tr, Yhat_tr, Y_te, Yhat_te, n_train, n_test)
#        w0[:,:,i] = network.weights[0][:]
#        w1[:,:,i] = network.weights[2][:]
#        w2[:,:,i] = network.weights[4][:]

    # Save data
    path = 'Data/iterative_stats/'
    np.savetxt(path+'MLP_'+param+'_runtime.txt',runtime,
               header = 'runtime (s)',
               delimiter=',')
    np.savetxt(path+'MLP_'+param+'_YOD.txt',YOD,
               delimiter=',')
    np.savetxt(path+'MLP_'+param+'_MAE.txt',MAE_stats,
               header='forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE',
               fmt = '%.2f',
               delimiter=',')
#    np.savetxt(path+'MLP_'+param+ '_EWD.txt',
#              [np.array([np.mean(w0),np.std(w0),np.mean(w1),np.std(w1),np.mean(w2),np.std(w2)])],
#               header='Layer_0_mean, Layer_0_std, Layer_1_mean, Layer_1_std, Layer_2_mean, Layer2_std',
#               delimiter=',')

iterate_MLP('tas',25)
iterate_MLP('pr',25)

def iterate_BNN(param,n_iterations):
    # Load and prepare CMIP5 data and used dataset-dependent NN parameters.
    models = sf.load_CMIP('Data/CMIP5/',param)
    if (param == 'tas'):
        n_test = 6;
        n_train = 23;
        kl_weight = 10**-4
        prior_f = 'Laplace_tas'
        train_mask = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    else:
        n_test  = 4
        n_train = 18;
        kl_weight = 10**-3
        prior_f = 'Laplace_pr'
        train_mask = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
    A_tr, A_te, Y_tr, Y_te, years = sf.prepare_inputs(models, train_mask)

    # Initiate storage arrays
    YOD = np.zeros((n_iterations,n_test));
    runtime = np.zeros((n_iterations,2));
    MAE_stats = np.zeros((n_iterations,4));

    # Define and create BNN network.
    n_inputs = A_tr.shape[1]
    n_outputs = 1
    hidden_layers = [10, 10]

    # Run iteration
    for i in range(n_iterations):
        print('Now working on iteration ' + str(i))
        network = BNN(n_inputs,n_outputs,hidden_layers,param = param, description = datestr, kl_weight = kl_weight, prior = prior_f)
        tic = time.process_time()
        network.train(A_tr, Y_tr, n_iterations = 2000, method = tfk.optimizers.Adam(), overwrite = load_old, lossf = tfk.losses.MeanSquaredError())
        toc = time.process_time()


        # Compute and plot network predictions on testing & training subsets.
        sample_size = 1000
        Yhat_tr, Yhat_tr_mean, Yhat_tr_std = network.predict(A_tr, sample_size)
        Yhat_te, Yhat_te_mean, Yhat_te_std = network.predict(A_te, sample_size)
        tac = time.process_time()

        sf.plot_prediction(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, years, param, 'BNN', datestr, closeplot = False)

        YOD[i,:] = sf.find_YOD(Y_te,Yhat_te_mean)
        runtime[i,0] = toc-tic; runtime[i,1] = tac-toc;
        MAE_stats[i,:] = sf.compute_stats(Y_tr, Yhat_tr_mean, Y_te, Yhat_te_mean, n_train, n_test)

    # Save data
    path = 'Data/iterative_stats/'
    np.savetxt(path+'BNN_'+param+'_runtime.txt',runtime,
               header = 'runtime (s)',
               delimiter=',')
    np.savetxt(path+'BNN_'+param+'_YOD.txt',YOD,
               delimiter=',')
    np.savetxt(path+'BNN_'+param+'_MAE.txt',MAE_stats,
               header='forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE',
               fmt = '%.2f',
               delimiter=',')

# iterate_BNN('tas',25)
# iterate_BNN('pr',25)