# -*- coding: utf-8 -*-
"""
Implementation of a Multi-Layer Perceptron (MLP).

Uses the Tensorflow & Tensorflow Keras libraries.

A Multi-Layer Perceptron, often also called a normal feedforward neural network,
consists of a number of hidden layers, each containing a few nodes, followed by
an output layer of one or more nodes. All nodes of a subsequent layer are connected
to the previous one through weights.

A prediction is made by computing the value of each node through summation of
the multiplication of previous node values with their corresponding weights and
bias, followed by a non-linear 'activation' function.
The values of the weights are computed during training, by minimizing the loss,
i.e. the cumulative error for all training inputs, of the model.

Created on Wed May 5 2021
Last edit on Mon Oct 18 2021

@author: Jesse Arens
"""

## Imports
# General imports
import os
import copy
import pickle

# Third-party imports
import tensorflow as tf
tfk = tf.keras

# I/O  settings.
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')

## Network implementation
# Multi-Layer Perceptron class definition
class MLP(tfk.Model):
    """
    Implementation of a Multi-Layer Perceptron (MLP).

    Uses the Tensorflow & Tensorflow Keras libraries.
    """

    # Initialization: this is what happens when you create an object of the class MLP
    def __init__(self,input_size,output_size,hidden_layers,ridge_penalty = 0, param = 'tas', r = '90x45', date = 'undefined', model = None):
        """
        Implement a Multi-Layer Perceptron (MLP).

        Uses the Tensorflow & Tensorflow Keras libraries.

        Parameters
        ----------
        input_size : int
            Size of each individual input samples (i.e. map size).
        output_size : int
            Size of each output target (i.e. year).
        hidden_layers : list of int
            List of the number of units per hidden layer. Each list entry corresponds to one hidden layer.
        ridge_penalty : int, optional
            Ridge regression penalty, to be applied as kernel regularizer in the first layer. The default is 0.
        param : str
            CMIP5-parameter to import. In this project either 'tas' or 'pr'.
        r : str, optional
            Resolution of data to be loaded. The default is '90x45'.
        date : str, optional
            Date string, used for version control in saving. The default is 'undefined'.
        model : None or Keras Sequential obj, optional
            Allows passing of a predefined, deviant network setup, which overwrites regular MLP setup.

        Raises
        ------
        Exception
            If hidden_layers is not a list of units.

        Returns
        -------
        None.

        Notes
        -----
        A Multi-Layer Perceptron, often also called a normal feedforward neural network,
        consists of a number of hidden layers, each containing a few nodes, followed by
        an output layer of one or more nodes. All nodes of a subsequent layer are connected
        to the previous one through weights.

        A prediction is made by computing the value of each node through summation of
        the multiplication of previous node values with their corresponding weights and
        bias, followed by a non-linear 'activation' function.
        The values of the weights are computed during training, by minimizing the loss,
        i.e. the cumulative error for all training inputs, of the model.

        """
        super(MLP,self).__init__() # MLP inherits properties from a Keras Model Class

        # Check if there are hidden layers & define number of layers
        # Check if hidden_layers is a list.
        if not isinstance(hidden_layers, list):
            raise Exception('hidden layers must be a list, example: hidden_layers = [10, 10], meaning there are two layers of 10 units each.')

        # Check if hidden layers list is empty. When empty, there are no layers.
        if len(hidden_layers) == 0:
            self.n_hidden_layers = 0
        # Check if the first hidden layer has zero units, this equals to having no layers.
        elif hidden_layers[0] == 0:
            self.n_hidden_layers = 0
        # If you end up here, there is at least one hidden layer with one of more units, so we can count the amount of layers.
        else:
            self.n_hidden_layers = len(hidden_layers)

        # Define some object metadata variables
        self.param = param
        self.r = r
        self.date = date
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.ridge_penalty = ridge_penalty

        # Build network architecture using Tensorflow Keras
        # Save the Keras model architecture as part of the MLP object.
        self.model = self._create_model(model)

        # Predefine transformation constants (normalization)
        self.A_mean = None
        self.A_std = None
        self.Y_mean = None
        self.Y_std = None

        # Raise metadata flags for inspecting network state
        self.optimizer = None # Whether the network already has an optimizer
        self.istrained = False # Has network been trained already or not?

        # Print MLP object & Keras model architecture in the console after creation.
        print(self.__repr__())
        print(self._get_model())

        return


    def __repr__(self):
       """Represent (display) the current object state."""
       str = f"{type(self).__name__}({self.input_size},{self.output_size},{self.hidden_layers})."

       if self.istrained:
            str += f"\n Network was trained for {self.n_epochs} epochs."
            str += f"\n Resulting final loss was {self.metadata['loss'][-1]}."
            str += f"\n Resulting final RMSE was {self.metadata['root_mean_squared_error'][-1]}.\n"
       else:
            str += "\n Network has not been trained.\n"
       return str


    def _get_model(self):
        """Retrieve the Keras model architecture of the MLP object."""
        return self.model.summary()

    def _create_model(self,network = None):
        """
        Create a Keras Sequential Network model and add Keras layers to the sequential model.

        Parameters
        ----------
        network : Keras Sequential obj, optional
            Allows for passing a deviant network set-up to the network. The default is None.

        Returns
        -------
        model : Keras Sequential obj
            The instance of a Keras Sequential model object, which represents
            the full model as specified by the init parameters.

        Notes
        -----
        Keras layers consist of weights and a possible subsequent activation.
        Therefore, this deviates from a common conception of neural network layers,
        in which the layers are the outputs after each propagation step (and the
        input layer). The difference results in a MLP of 4 layers (input, 2 hidden
        layers and output) being represented by 3 Keras layers.

        For MLP, we use the 'Dense' layer object, for which all layer units are
        connected to all units of the previous layer. This is identical to the
        staple fully-connected layer in most common feed-forward neural networks.

        Regularization can be applied on certain aspects of a layer:
            kernal (i.e. weights),
            bias, or
            activity (i.e. outputs).

        Adding a layer is done using the function add() in the (Sequential) model API:
        model.add(Dense(output size,
                        input size,
                        activation='activation function name',
                        kernel_regularizer='weight regularization type',
                        name = 'insert name'
                        )
                 )

        See Also
        --------
        tfk.Sequential : Sequential model generator from Tensorflow Keras
        tfk.layers.Dense : Dense layer implementation in Tensorflow Keras
        tfk.regularizers.L2 : regularization implementation in Tensorflow Keras
        """
        if network == None:
            model = tfk.Sequential(name = 'MLP')

            # Correction for ridge regression parameter for L2-norm implementation in Keras.
            self.L2_norm = 0.5*self.ridge_penalty / (self.input_size*self.hidden_layers[0])

            # Add the first layer, connecting the inputs and the first hidden layer.
            # Regularization in our case only applied on weights of the first layer.
            model.add(tfk.layers.Dense(self.hidden_layers[0],
                                   input_shape = (self.input_size,),
                                   activation = 'tanh',
                                   kernel_regularizer = tfk.regularizers.L2(l2 = self.L2_norm),
                                   name = 'Input_layer'))

            # Now add extra layers
            for i in range(self.n_hidden_layers):
                # Check if the current layer is the last layer or not.
                if(i+1<self.n_hidden_layers): # There is still a next hidden layer after the current one, so we can safely add one more hidden layer with activation function.
                    model.add(tfk.layers.Dense(self.hidden_layers[i],
                                           activation = 'tanh',
                                           name = 'Hiddenlayer_'+str(i+1)))
                else:
            # This is the last hidden layer, before output. We do not apply activation on this layer.
                    model.add(tfk.layers.Dense(self.output_size,
                                           name = 'Hiddenlayer_'+str(i+1)))
        else: model = network

        return model

    def _normalization_constants(self,A,Y):
        """
        Compute normalization constants (mean, standard deviation).

        These are used to transform the data to normalized data, or back.
        Resulting constants are saved as variables inside the class instance.

        Parameters
        ----------
        A : Array of float32
            Input data. Array shape = (number of CMIP5 models * years per model, map size per model input).
        Y : Array of float32
            Target (output) data. Array shape = (number of CMIP5 models * years per model, 1).

        Returns
        -------
        None.
        """
        # Input variables A
        if self.A_mean is None:
            self.A_mean = A.mean(axis=0)
            self.A_std = A.std(axis=0)
            self.A_constant = self.A_std == 0
            self.A_stdFixed = copy.copy(self.A_std)
            self.A_stdFixed[self.A_constant] = 1

        # Output variables Y
        if self.Y_mean is None:
            self.Y_mean = Y.mean(axis=0)
            self.Y_std = Y.std(axis=0)
            self.Y_constant = self.Y_std == 0
            self.Y_stdFixed = copy.copy(self.Y_std)
            self.Y_stdFixed[self.Y_constant] = 1
        return

    def _normalize_input(self,A):
        """
        Normalize input variables.

        Based on previously defined normalization constants (mean and std).

        Parameters
        ----------
        A : Array of float32
            Input data. Array shape = (number of CMIP5 models * years per model, map size per model input)

        Returns
        -------
        result : Array of float32
            Normalized input data. Array shape is the same as input.

        """
        result = (A - self.A_mean) / self.A_stdFixed
        result[:, self.A_constant] = 0.0
        return result

    def _denormalize_input(self,A):
        """
        Revert normalized input variables to their original state.

        Uses the normalization constants which have been saved as variables to
        the class instance.

        Parameters
        ----------
        A : Array of float32
            Normalized input data. Array shape  = (number of CMIP5 models * years per model, map size per model input)

        Returns
        -------
        Array of float32
            De-normalized (i.e. original) input data. Array shape is the same as input.

        """
        return self.A_std * A  + self.A_mean

    def _normalize_output(self,Y):
        """
        Normalize output variables.

        Based on previously defined normalization constants (mean and std).

        Parameters
        ----------
        Y : Array of float32
            Target or output data. Array shape =  (number of CMIP5 models * years per model, 1)

        Returns
        -------
        result : Array of float32
            Normalized target or output data. Array shape is the same as input.

        """
        result = (Y - self.Y_mean)/self.Y_stdFixed
        result[:, self.Y_constant] = 0.0
        return result

    def _denormalize_output(self,Y):
        """
        Revert normalized output variables to their original state.

        Mainly used after prediction. Uses the normalization constants which have been saved as variables to
        the class instance.

        Parameters
        ----------
        Y : Array of float32
            Normalized target or output data. Array shape =  (number of CMIP5 models * years per model, 1)

        Returns
        -------
        Array of float32
            De-normalized (i.e. original) target or ouput data. Array shape is the same as input.

        """
        return self.Y_std * Y + self.Y_mean


    def _select_optimizer(self,method, **kwargs):
        """
        Set a default optimizers from the Keras API, with default parameters.

        Parameters
        ----------
        method : str
            Descriptor of the Keras optimizer class, as defined in the Keras API.
        **kwargs : dict, optional
            Allows for passing arguments to the Keras optimizer class,
            if non-default behavior is required. Check the desired Keras
            optimizer for which arguments are available.

        Raises
        ------
        Exception
            When the requested optimizer is not available through Keras, or is misspelled.

        Returns
        -------
        optimizer : Keras optimizers obj
            Instance of the chosen optimizer, with provided arguments.

        See Also
        --------
        tfk.optimizers : List of available Keras optimizers and their arguments.
        """
        opt_dict = {'SGD': tfk.optimizers.SGD(**kwargs),
                    'RMSprop': tfk.optimizers.RMSprop(**kwargs),
                    'Adam': tfk.optimizers.Adam(**kwargs),
                    'Adadelta':tfk.optimizers.Adadelta(**kwargs),
                    'Adagrad':tfk.optimizers.Adagrad(**kwargs),
                    'Adamax':tfk.optimizers.Adamax(**kwargs),
                    'Nadam':tfk.optimizers.Nadam(**kwargs),
                    'Ftrl':tfk.optimizers.Ftrl(**kwargs)
                    }

        try:
            optimizer = opt_dict[method]
        except:
            raise Exception("Training optimizer algorithm not known, please use one of the available Keras optimizer classes")

        return optimizer

    def train(self, A, Y, n_iterations=500,batch_size = None, method='Adam',overwrite=False):
        """
        Train the MLP, by ingesting it with training input & output data.

        If an identical, trained, network is already found in the networks folder, it will load this network instead.
        Model weights and metadata parameters are saved in the class instance and not returned directly.

        Parameters
        ----------
        A : Array of float32
            Original (non-normalized) input data. Array shape = (number of CMIP5 models * years per model, map size per model input)
        Y : Array of float32
            Original (non-normalized) target data.  Array shape =  (number of CMIP5 models * years per model, 1)
        n_iterations : int, optional
            The amount of full training loops performed. Each training loop, all the weights are updated once.The default is 500.
        batch_size : int or None, optional
            Amount of samples included in each gradient update. The default is None, which in Keras implementation then results in a batch size of 32.
        method : str or Keras optimizers obj, optional
            Defines the desired Keras optimizer to be used for the gradient update phase in training. The default is 'Adam'.
        overwrite : bool, optional
            Whether to overwrite an existing network, if one already exists. The default is False.

        Raises
        ------
        Exception
            If the provided input data does not have the same shape as the network's input size'

        Returns
        -------
        None.
        """
        # Check dimensions with input/output size of network.
        if A.shape[1] != self.input_size:
            raise Exception(f' Training: number of columns in A ({A.shape[1]}) not equal to number of network inputs ({self.input_size})')

        # Before training, we normalize the input and outputs as this results in more optimal training.
        self._normalization_constants(A,Y)
        A = self._normalize_input(A)
        Y = self._normalize_output(Y)

        # Define & initialize the optimization (gradient descent) method.
        if type(method) == str:
            # If method is a string, set the optimizer as defined by Keras API
            self.optimizer = self._select_optimizer(method)
        else:
            # If method is an object, directly allocate it to the optimizer variable.
            self.optimizer = method

        # Compile the model, i.e. prepares it for training by specifying the
        # optimization algorithms and loss & tracking metric parameters.
        self.model.compile(
            optimizer = self.optimizer,
            loss = tfk.losses.MeanSquaredError(),
            metrics = [ tfk.metrics.RootMeanSquaredError() ]
            )

        # Check if weights already exist
        if(self.date):
            path = 'Data/Networks/' + self.date + '_MLP_' + self.param + '.index'
        else:
            path =  'Data/Networks/MLP_' + self.param + '.index'
        if (os.path.exists(path) and overwrite):
            print('Network has previously been trained, retrieving weights')
            self._load_weights(path = path)

        else:
            print("Now training the neural network, with ridge penalty "
                  + str(self.ridge_penalty) +".")
            # Fit the model. Save resulting tracked metadata as history.
            history = self.model.fit(
                x = A,
                y = Y,
                epochs = n_iterations,
                batch_size = batch_size
                )
            self.metadata = history.history

        # Set some parameters used later.
        self.istrained = True
        self.n_epochs = n_iterations

        # Print MLP object & training result to console.
        print("Neural network is now trained.")
        print(self.__repr__())
        print(self._get_model())

    def predict(self,A):
        """
        Compute the prediction of a given input A, provided the network has been trained.

        Parameters
        ----------
        A : Array of float32
            Original (non-normalized) input data to perform the prediction on.
            Array shape = (number of CMIP5 models * years per model, map size per model input)

        Returns
        -------
        Yhat : Array of float32
            Prediction result, as denormalized output data.
            Array shape = (number of CMIP5 models * years per model, 1)
        """
        # Computes the prediction of a given input A, provided the network has been trained.
        if self.istrained == False:
            print('Warning: Neural Network has not been trained yet. \n Prediction is based on random initial weights.')

        # First prepare the input by normalizing, as network is trained on normalized inputs and outputs.
        A = self._normalize_input(A)

        # Then make a prediction using the built-in Keras model prediction function.
        Yhat = self.model.predict(A)

        # Transform normalized prediction back to years.
        Yhat = self._denormalize_output(Yhat)

        return Yhat

    def save_weights(self):
        """
        Save the MLP object instance, including training information when available.

        The instance is saved as two files: a Keras model and a pickle dump of the metadata.
        Required parameters for file saving are provided as variables in the MLP object instance.

        Returns
        -------
        None.
        """
        # Save the Keras model architecture (including possible training information)
        self.model.save_weights(
            'Data/Networks/' + self.date + '_MLP_' + self.param + '_r' + self.r)
        print('Network weights saved as MLP_' + self.param + ' Keras file.')

        f = open('Data/Networks/' + self.date + '_MLP_' +
                 self.param + '_r' + self.r + '_metadata.pkl', "wb")
        pickle.dump(self.metadata, f)
        f.close()
        print('Network metadata saved as MLP_' + self.param + ' pickle file')
        return

    def _load_weights(self, path = None):
        """
        Load a previously saved MLP object including training information when available.

        The instance is loaded from two files: a Keras model and a pickle dump of the metadata.
        Required parameters for file loading are provided as variables in the MLP object instance.

        Returns
        -------
        None.
        """
        if not(path):
            path = 'Data/Networks/' + str(self.date) +'_MLP_'+self.param + '_r' + self.r

        self.model.load_weights(path)
        print('Network weights from MLP_' + self.param + ' retrieved')

        f = open(path + '_metadata.pkl', 'rb')
        self.metadata = pickle.load(f)
        f.close()


