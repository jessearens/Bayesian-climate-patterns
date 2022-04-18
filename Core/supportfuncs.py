# -*- coding: utf-8 -*-
"""
Module containing a few supporting functions.

These are used for handling data that is fed into or retrieved from the Tensorflow models, including:
    - Converting a netcdf file to a npz file
    - Importing & preparing a netcdf file for ingestion into a Neural Network
    - Creating prediction scatter plots

Created on Tue May 18 2021
Last edit on Fri Aug 27 2021

@author: Jesse Arens
"""
# General imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from tensorflow import keras as tfk # Base framework for MLP object implementation

# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')

# Functions
def load_CMIP(path, param, r = '90x45', netcdfs=False):
    """
    I/O function used to load CMIP5 data into Python, for eventual feeding into a network (after preparing the inputs).

    Parameters
    ----------
    path : str
        Path location of the CMIP5 data folder.
    param : str
        CMIP5-parameter to import. In this project either 'tas' or 'pr'.
    r : str, optional
        Resolution of data to be loaded. The default is '90x45'.
    netcdfs : bool, optional
        Flag which described whether the dataformat is a netCDF-type or NpzFile.
        If the data is in netCDF-format, this function will call @netcdf2npz() The default is False.

    Raises
    ------
    Exception
        If parameter is different than 'tas' or 'pr', as function cannot handle these.

    Returns
    -------
    models : NpzFile obj
        Contains the CMIP5 data, stored in a NpzFile archive.

    """
    # Path for netcdfs: ends at param
    if netcdfs:
        if (param == 'pr'):
            mdl_list  = ['ACCESS1-0','ACCESS1-3','CanESM2','CMCC-CMS','CNRM-CM5',
              'CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','GISS-E2-H',
              'GISS-E2-H-CC','GISS-E2-R','GISS-E2-R-CC','HadGEM2-CC',
              'HadGEM2-ES','inmcm4','MIROC5','MIROC-ESM','MIROC-ESM-CHEM',
              'MRI-CGCM3','NorESM1-M','NorESM1-ME']
        elif (param == 'tas'):
            mdl_list = ['ACCESS1-0','ACCESS1-3','CanESM2','CCSM4','CESM1-BGC','CESM1-CAM5',
             'CMCC-CMS','CNRM-CM5','CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G',
             'GFDL-ESM2M','GISS-E2-H','GISS-E2-H-CC','GISS-E2-R','GISS-E2-R-CC',
             'HadGEM2-AO','HadGEM2-CC','HadGEM2-ES','inmcm4','IPSL-CM5A-LR',
             'IPSL-CM5A-MR','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MPI-ESM-MR',
             'MRI-CGCM3','NorESM1-M','NorESM1-ME']
        else:
            raise Exception('Param needs to be either "tas" or "pr"')
        models = netcdf2npz(mdl_list,param)
    else:
        models = np.load(path+'CMIP5_hist_rcp_'+param+'_annualmean_r'+ r + '.npz')

    return models

def prepare_inputs(models,train_mask,startyear=1920,stopyear=2100):
    """
    Prepare CMIP5 data in order to be fed into a neural network.

    Parameters
    ----------
    models : NpzFile obj
        Contains the CMIP5 data, stored in a NpzFile archive.
    trainmask : list of int
        Describes which models are chosen for testing and wwhich for training.
        For each model there are two options: 0 (testing) or 1 (training).
    startyear : int, optional
        The first year of the input dataset. The default is 1920.
    stopyear : TYPE, optional
        The last year of the input dataset. The default is 2100.

    Returns
    -------
    A_train : Array of float32
        Input data of training subset. Array shape = (number of CMIP5 models * years per model, map size per model input)
    A_test : Array of float32
        Input data of testing subset. Array shape = (number of CMIP5 models * years per model, map size per model input)
    Y_train : Array of float32
        Target data of training subset. Array shape = (number of CMIP5 models * years per model, 1)
    Y_test : Array of float32
        Target data of testing subset. Array shape = (number of CMIP5 models * years per model, 1)
    years : Array of int32
        Array containing all years for which data is used. For further use in plotting.

    """
    data = models['data']
    mask = np.ma.make_mask(train_mask)

    mapsize = data.shape[-1]*data.shape[-2]
    A_train = data[mask,startyear-1920:stopyear-1920,:,:].reshape(-1,mapsize).astype('float32')
    A_test = data[~mask,startyear-1920:stopyear-1920,:,:].reshape(-1,mapsize).astype('float32')

    years = np.arange(startyear,stopyear,1)
    n_train = sum(mask)
    n_test = data.shape[0] - n_train

    Y_train  = np.tile(years, n_train).reshape(1, n_train * years.size).T.astype('float32')
    Y_test  = np.tile(years, n_test).reshape(1, n_test * years.size).T.astype('float32')

    return A_train, A_test, Y_train, Y_test, years

def netcdf2npz(mdl_list,param):
    """
    Convert a (CMIP5-derived) netCDF4 file into a python-based NpzFile and save it.

    Parameters
    ----------
    mdl_list : list of str
        List of the name of all the models to load and include in the NpzFile.
    param : str
        CMIP5-parameter to convert. In this project either 'tas' or 'pr'.

    Returns
    -------
    npz : NpzFile obj
        Contains the CMIP5 data, stored in a NpzFile archive.

    """
    i = len(mdl_list)
    j = 180; k = 45; l = 90;
    data = np.zeros((i,j,k,l),dtype='float32')
    m = 0
    for mdl in mdl_list:

        dataset = netCDF4.Dataset('Data/CMIP5/' + param + '/' + mdl + '/cdo_results/' + mdl + '_yearmean_regrid.nc')
        data[m,:,:,:] = dataset.variables[param][:]
        m += 1

    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]

    dataset.close()

    file = 'Data/CMIP5/CMIP5_hist_rcp_' + param + '_annualmean_r90x45.npz'
    np.savez(file,data=data,lats=lats,lons=lons)
    npz = np.load(file)

    return npz

def plot_prediction(Y_train, Yhat_train, Y_test, Yhat_test, years, param, network_name, date = 'undefined', closeplot=True):
    """
    Create a prediction scatterplot, of truth versus predicted years.

    Parameters
    ----------
    Y_train : Array of float32
        Truth target data for training.
    Yhat_train : Array of float32
        Target data for training, as predicted by the neural network after training.
    Y_test : Array of float32
        Truth target data in testing.
    Yhat_test : Array of float32
        Target data for testing, as predicted by the neural network after training.
    years : Array of int32
        Array containing all years for which data is used.
    param : str
        CMIP5-parameter used. In this project either 'tas' or 'pr'.
    network_name : str
        Descriptor for the type of neural network used for the plot, used for version control in saving.
    date : str, optional
        Date string, used for version control in saving. The default is 'undefined'.
    closeplot : bool, optional
        Whether the plot needs to be closed (i.e. only saved but not shown by the IDE). The default is True.
    r : str, optional
        Used resolution of the data, used for version control in saving. The default is '90x45'.

    Returns
    -------
    None.

    """
    plt.figure()
    plt.scatter(Y_train, Yhat_train, s = 25, marker = '.', color = 'gray')
    j = 0;

    n_years = years.size
    n_train = int(Y_train.size / n_years)
    n_test = int(Y_test.size / n_years)

    for _ in range(n_test):
        plt.scatter(Y_test[j:j + n_years, :], Yhat_test[j:j + n_years, :], s = 25, marker = '.')
        j += n_years
    plt.plot([1880, 2140], [1880, 2140], 'k')
    plt.xlim([1880, 2140]); plt.ylim([1880, 2140])

    if param  ==  'tas':
        plt.title('Near-surface air temperature')
    else: plt.title('Precipitation')

    plt.xlabel('actual year'); plt.ylabel('predicted year');
    ticks = np.arange(years[0]-20, years[-1]+20, 20); plt.xticks(ticks); plt.yticks(ticks)

    statspath = 'Data/Networks/' + date + '_' + network_name + param + '_stats.txt'
    forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE = compute_stats(
                Y_train,
                Yhat_train,
                Y_test,
                Yhat_test,
                n_train,
                n_test,
                statspath)

    textstr = "Post Year 2000 \nMean Absolute Error\n"
    plt.text(2055,1900, textstr,fontweight='bold',fontsize='small')
    textstr = f"Training: {forecast_train_MAE:0.2f}\n"
    textstr += f"Testing: {forecast_test_MAE:0.2f}"
    plt.text(2055,1885,textstr,fontsize='small')

    plt.savefig('Figures/' + network_name + '/' + date + '_' + param + '.png')
    if closeplot:
        plt.close('all')

    print('Prediction scatter plot created & saved as ' + network_name + '/'+ date + '_' + param + '.png file')


def find_YOD(Y_test,Yhat_test):
    """
    Compute the Year of Climate Departure.

    The year found is the first year after which all predicted testing data exceeds
    the predictions of a baseline period of 20 years.

    Parameters
    ----------
    Y_test : Array of float32
        Truth target data in testing.
    Yhat_test : Array of float32
        Target data for testing, as predicted by the neural network after training.

    Returns
    -------
    YOD : Array of float32
        Contains the first year for each CMIP5 model for which climate departure is found.

    """
    n_test = int(Y_test.size/180)
    j = 0; YOD = np.zeros(n_test)

    for i in range(n_test):
        max_base = np.max(Yhat_test[j:j+19])
        idx = np.where(Yhat_test[j:j+180] < max_base)[0]
        YOD[i]= Y_test[np.max(idx)+1]
        j+= 180

    return YOD


def compute_stats(Y_train,Yhat_train,Y_test,Yhat_test,n_train,n_test,path = []):
    """
    Compute predictive statistics for the trained neural network, and save them for future reference.

    Parameters
    ----------
    Y_train : Array of float32
        Truth target data for training.
    Yhat_train : Array of float32
        Target data for training, as predicted by the neural network after training.
    Y_test : Array of float32
        Truth target data in testing.
    Yhat_test : Array of float32
        Target data for testing, as predicted by the neural network after training.
    n_train : int
        Number of models used for training.
    n_test : int
        Number of models used for testing.
    path : str
        File path for saving the statistics.

    Returns
    -------
    forecast_train_MAE : float32
        Mean Absolute Error post-2000 for the training dataset.
    total_train_MAE : float32
        Mean Absoulte Error for the whole timeseries for the training dataset.
    forecast_test_MAE : float32
        Mean Absolute Error post-2000 for the testing dataset.
    total_test_MAE : float32
        Mean Absolute Error for the whole timeseries for the testing dataset.
    """
    Y_train_forecast = Y_train.reshape(n_train, -1)[:,80:]
    Yhat_train_forecast = Yhat_train.reshape(n_train, -1)[:,80:]
    forecast_train_MAE = np.mean(np.abs(Yhat_train_forecast - Y_train_forecast))
    total_train_MAE = np.mean(np.abs(Yhat_train - Y_train))

    Y_test_forecast = Y_test.reshape(n_test, -1)[:,80:]
    Yhat_test_forecast = Yhat_test.reshape(n_test, -1)[:,80:]
    forecast_test_MAE = np.mean(np.abs(Yhat_test_forecast - Y_test_forecast))
    total_test_MAE = np.mean(np.abs(Yhat_test - Y_test))

    if(path):
        np.savetxt(path,
               [np.array([forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE])],
               header='forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE',
               fmt = '%.2f',
               delimiter=','
               )

    return forecast_train_MAE, total_train_MAE, forecast_test_MAE, total_test_MAE
