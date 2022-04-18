# -*- coding: utf-8 -*-
"""
Implementation of a Bayesian Neural Network (BNN).

Uses the Tensorflow, Tensorflow Keras & Tensorflow Probability libraries.

Created on Wed May 26 2021
Last edit on Mon Aug 30 2021

@author: Jesse Arens
"""

# General imports
import os
import copy
import pickle
import numpy as np

# Third-party imports
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfd = tfp.distributions

# I/O settings
os.chdir('C:/Users/Jesse/Documents/Universiteit/MasterThesis/')

## Network implementation
# Bayesian By Backprop Neural Network class definition
class BNN(tfk.Model):
    """
    Implementation of a Bayesian Neural Network (BNN).

    Uses the Tensorflow & Tensorflow Keras libraries.

    """

    # Initialization: this is what happens when you create an object of the class BNN
    def __init__(self, input_size, output_size, hidden_layers, param, description = 'undefined', kl_weight = None, prior = 'Normal', model = None):
        """
        Implement a Bayesian Neural Network (BNN).

        Uses the Tensorflow & Tensorflow Keras libraries.

        Parameters
        ----------
        input_size : int
            Size of each individual input samples (i.e. map size).
        output_size : int
            Size of each output target (i.e. year).
        hidden_layers : list of int
            List of the number of units per hidden layer. Each list entry corresponds to one hidden layer.
        param : str
            CMIP5-parameter to import. In this project either 'tas' or 'pr'.
        description : str, optional
            Description string, used for version control in saving. The default is 'undefined'.
        kl_weight : float, optional
            Value for weighing the importance of the KL-divergence term in the the neural network loss. The default is None, resulting in a standard weight of 1/input_size.
        prior : str, optional
            Description of prior function(s) to be used. The default is None, resulting in a unit Gaussian prior.
        model : None or Keras Sequential obj, optional
            Allows passing of a predefined, deviant network setup, which overwrites regular BNN setup.

        Raises
        ------
        Exception
            If hidden_layers is not a list of units.

        Returns
        -------
        None.

        """
        super(BNN, self).__init__()  # BNN inherits properties from a Keras Model Class

        # Check if there are hidden layers & define number of layers
        # Check if hidden_layers is a list.
        if not isinstance(hidden_layers, list):
            raise Exception(
                'hidden layers must be a list, example: hidden_layers = [10, 10], meaning there are two layers of 10 units each.')

        # Check if hidden layers list is empty. When empty, there are no layers.
        if len(hidden_layers) == 0:
            self.n_hidden_layers = 0
        # Check if the first hidden layer has zero units, this equals to having no layers.
        elif hidden_layers[0] == 0:
            self.n_hidden_layers = 0
        # If you end up here, there is at least one hidden layer with one of more units, so we can count the amount of layers.
        else:
            self.n_hidden_layers = len(hidden_layers)

        # Define some object variables
        self.param = param
        self.description = description
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.prior = prior
        if not(kl_weight):
              self.kl_weight = 1/self.input_size
        else: self.kl_weight = kl_weight

        # Build network architecture using Tensorflow Keras
        # Save the Keras model architecture as part of the BNN object.
        self.model = self._create_model(self.prior,self.posterior, model)

        # Predefine transformation constants (normalization)
        self.A_mean = None
        self.A_std = None
        self.Y_mean = None
        self.Y_std = None

        # Raise metadata flags for inspecting network state
        self.optimizer = None  # The specified optimizer used in training.
        self.istrained = False  # Has network been trained already or not?

        # Print MLP object & Keras model architecture in the console after creation.
        print(self.__repr__())
        print(self._get_model())

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
        """Retrieve the Keras model architecture of the BNN object."""
        return self.model.summary()

    def _create_model(self, prior, posterior, network = None):
        """
        Create a Keras Sequential Network model and add Keras layers to the sequential model.

        Parameters
        ----------
        prior : Python callable
            Returns a callable which takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.
        posterior : Python callable
            Returns a callable which takes an input and produces the posterior distribution,
            provided as a Tensorflow Probability Distribution instance.
        network : Keras Sequential obj, optional
            Allows for passing a deviant network set-up to the network. The default is None.

        Returns
        -------
        model : Keras Sequential obj
            The instance of a Keras Sequential model object, which represents
            the full model as specified by the init parameters.

        Notes
        -----
        The base model is a sequential model, which means that the layers follow one another in propagation.
        Keras layers consist of weights and a possible subsequent activation.
        Therefore, this deviates from a common conception of neural network layers,
        in which the layers are the outputs after each propagation step (and the
        input layer). The difference results in a MLP of 4 layers (input, 2 hidden
        layers and output) being represented by 3 Keras layers.

        For MLP, we use the Dense layer object, for which all layer units are
        connected to all units of the previous layer. This is identical to the
        staple fully-connected layer in most common feed-forward neural networks.

        Regularization can be applied on certain aspects of a layer:
        kernal (i.e. weights),
        bias, or
        activity (i.e. outputs).
        Adding a layer is done using the function add() in the (Sequential) model API, for example:
        model.add(Dense(output size,
                        input size,
                        activation function name,
                        weight regularization type,
                        name
                    )
                 )

        To implement a BNN instead of an MLP, Tensorflow probability is used
        to implement DenseVariational layers instead of Keras Dense layers.
        The difference is that DenseVariational layers require a prior &
        posterior distribution, which is expressed as a function, and created
        using a Variable Layer (layer containng a variable), which contains
        the distributions to be approximated. After initialisation,a DenseVariational
        layer interfaces identical to a normal Dense layer:
        model.add(DenseVariational(output size,
                                   input size
                                   prior function
                                   posterior function
                                   kl_weighting
                                   kl_approximation
                                   activation function
                                   name
                    )
                 )

        Here, we also define the kl_weighting, which allows a tweaking the importance
        of the KL-divergence in the loss function. Furthermore, we define whether
        the KL-divergence is approximated using Monte Carlo approximation or
        through a pre-defined built-in registration.

        See Also
        --------
        tfk.Sequential : Sequential model generator from Tensorflow Keras.
        tfp.layers.DenseVariational : Variational inference implementation in Tensorflow Probability
        tfp.layers.IndependentNormal : A Keras layer representing an independent normal distribution.
        prior : function that generates the DenseVariational prior distribution.
        posterior : function that generates the DenseVariational posterior distribution.

        """
        if network == None:
            model = tfk.Sequential(name = 'BNN')

            # We have several options for prior, here we select which one is to be used.
            prior_funcs = {
                'Normal' : [self.prior_Normal, self.prior_Normal, self.prior_Normal, True],
                'Laplace_tas' : [self.prior_Laplace_tas_w0, self.prior_Laplace_tas_w1, self.prior_Laplace_tas_w2, False],
                'Laplace_pr' : [self.prior_Laplace_pr_w0, self.prior_Laplace_pr_w1, self.prior_Laplace_pr_w2, False],
                'Student_tas' : [self.prior_Student_tas_w0, self.prior_Student_tas_w1, self.prior_Student_tas_w2, False],
                'Student_pr' : [self.prior_Student_pr_w0, self.prior_Student_pr_w1, self.prior_Student_pr_w2, False],
                }



            model.add(tfp.layers.DenseVariational(units = self.hidden_layers[0],
                                             input_shape = (self.input_size,),
                                             make_prior_fn = prior_funcs[self.prior][0],
                                             make_posterior_fn = posterior,
                                             kl_weight = self.kl_weight,
                                             kl_use_exact = prior_funcs[self.prior][-1],
                                             activation='leaky_relu',
                                             name = 'Input_layer'
                                             )
                  )

            # Now add extra layers
            for i in range(self.n_hidden_layers):
                # Check if the current layer is the last layer or not.
                if(i+1 < self.n_hidden_layers):
                    # There is still a next hidden layer after the current one, so we can safely add one more hidden layer with activation function.
                    model.add(tfp.layers.DenseVariational(units = self.hidden_layers[i],
                                                     make_prior_fn = prior_funcs[self.prior][i],
                                                     make_posterior_fn = posterior,
                                                     kl_weight = self.kl_weight,
                                                     kl_use_exact = prior_funcs[self.prior][-1],
                                                     activation = 'leaky_relu',
                                                     name = 'Hiddenlayer_'+str(i+1)
                                                     )
                          )
                else:
                    print(i)
                    # This is the last layer, i.e. output layer. We do not apply activation on this layer.
                    model.add(tfp.layers.DenseVariational(units = tfp.layers.IndependentNormal.params_size(1),
                                                     make_prior_fn = prior_funcs[self.prior][i-1],
                                                     make_posterior_fn = posterior,
                                                     kl_weight = self.kl_weight,
                                                     kl_use_exact = prior_funcs[self.prior][-1],
                                                     name = 'Hiddenlayer_'+str(i+1)
                                                     )
                          )
                    model.add(tfp.layers.IndependentNormal(1, name ='Output_distribution'))

        else: model = network

        return model

    def _normalization_constants(self, A, Y):
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

    def _normalize_input(self, A):
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

    def _denormalize_input(self, A):
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
        return self.A_std * A + self.A_mean

    def _normalize_output(self, Y):
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

    def _denormalize_output(self, Y):
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

    def _selectoptimizer(self, method, **kwargs):
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
        tfk.optimizers : List of available Keras optimizers, and their arguments.
        """
        opt_dict = {'SGD': tfk.optimizers.SGD(**kwargs),
                    'RMSprop': tfk.optimizers.RMSprop(**kwargs),
                    'Adam': tfk.optimizers.Adam(**kwargs),
                    'Adadelta': tfk.optimizers.Adadelta(**kwargs),
                    'Adagrad': tfk.optimizers.Adagrad(**kwargs),
                    'Adamax': tfk.optimizers.Adamax(**kwargs),
                    'Nadam': tfk.optimizers.Nadam(**kwargs),
                    'Ftrl': tfk.optimizers.Ftrl(**kwargs)
                    }

        try:
            optimizer = opt_dict[method]
        except:
            raise Exception(
                "Training optimizer algorithm not known, please use one of the available Keras optimizer classes")

        return optimizer

    def train(self, A, Y, n_iterations=500, batch_size=None, method=tfk.optimizers.Adam(learning_rate=0.005), overwrite=False, lossf=tfk.losses.MeanSquaredError()):
        """
        Train the BNN, by ingesting it with training input & output data.

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
            Defines the desired Keras optimizer to be used for the gradient update phase in training. The default is tfk.optimizers.Adam(learning_rate=0.005).
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
            raise Exception(
                f' Training: number of columns in A ({A.shape[1]}) not equal to number of network inputs ({self.input_size})')

        # Before training, we normalize the input and outputs as this results in more optimal training.
        self._normalization_constants(A, Y)
        A = self._normalize_input(A)
        Y = self._normalize_output(Y)

        # Define & initialize the optimization (gradient descent) method.
        if type(method) == str:
            # If method is a string, set the optimizer as defined by Keras API
            self.optimizer = self._selectoptimizer(method)
        else:
            # If method is an object, directly allocate it to the optimizer variable.
            self.optimizer = method

        # Compile the model, i.e. prepares it for training by specifying the
        # optimization algorithms and loss & tracking metric parameters.
        loss = lossf

        self.model.compile(
            optimizer=self.optimizer,
            loss=loss,
            metrics=[tfk.metrics.RootMeanSquaredError()]
        )


        # Check if network weights already exist.
        if (os.path.exists('Data/Networks/' + self.description + '_BNN_' + self.param + '.index') and overwrite):
            print('Network has previously been trained, retrieving weights')
            self._loadweights()

        else:
            print("Now training the neural network.")
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

        # Print BNN object & training result to console.
        print("Neural network is now trained.")
        print(self.__repr__())
        print(self._get_model())

    def predict(self, A,sample_size=1):
        """
        Compute the prediction of a given input A, provided the network has been trained.

        As the BNN does not give a deterministic prediction, sampling can be used to derive a distribution.

        Parameters
        ----------
        A : Array of float32
            Original (non-normalized) input data to perform the prediction on.
            Array shape = (number of CMIP5 models * years per model, map size per model input)
        sample_size : int
            Defines how many prediction samples are made. The default is 1, resulting in a single prediction.
            If more than 1, each individual sample is stored in the array.

        Returns
        -------
        Yhat : Array of float32
            Prediction result, as denormalized output data.
            Array shape = (number of CMIP5 models * years per model, sample_size)
        Yhat_mean : Array of float32
            Mean prediction value, based on the predictions as defined by sample_size.
            Array shape = (number of CMIP5 models * years per model, 1)
        Yhat_std : Array of float32
            Standard deviation value, based on the predictions as defined by sample_size.
            Array shape = (number of CMIP5 models * years per model, 1)
        """
        if self.istrained == False:
            print('Warning: Neural Network has not been trained yet. \n Prediction is based on random initial weights.')

        # First prepare the input by normalizing, as network is trained on normalized inputs and outputs.
        A = self._normalize_input(A)

        # Then make a prediction using the built-in Keras model prediction function.
        Yhat = np.zeros((A.shape[0],sample_size), dtype = 'float32')
        ss = str(sample_size)
        print('Now computing the prediction of ' + ss + ' samples from the network.')
        for i in range(sample_size):
            Yhat[:,i] = self.model.predict(A).reshape(A.shape[0])
            print('Computed prediction sample ' + str(i) + '/' + ss)
        # Transform normalized prediction back to years.
        Yhat = self._denormalize_output(Yhat)
        Yhat_mean = Yhat.mean(axis=1).reshape(A.shape[0],1)
        Yhat_std = Yhat.std(axis=1).reshape(A.shape[0],1)
        return Yhat, Yhat_mean, Yhat_std

    def save_weights(self):
        """
        Save the BNN object instance, including training information when available.

        The instance is saved as two files: a Keras model and a pickle dump of the metadata.
        Required parameters for file saving are provided as variables in the BNN object instance.

        Returns
        -------
        None.
        """
        self.model.save_weights(
            'Data/Networks/' + self.description + self.param + '_BNN')
        print('Network weights saved as BNN_' + self.param + ' Keras file.')

        f = open('Data/Networks/' + self.description + self.param + '_BNN_metadata.pkl', "wb")
        pickle.dump(self.metadata, f)
        f.close()
        print('Network metadata saved as BNN_' + self.param + ' pickle file')


    def _load_weights(self):
        """
        Load a previously saved BNN object including training information when available.

        The instance is loaded from two files: a Keras model and a pickle dump of the metadata.
        Required parameters for file loading are provided as variables in the BNN object instance.

        Returns
        -------
        None.
        """
        self.model.load_weights(
            'Data/Networks/' + self.description + self.param + '_BNN')
        print('Network weights from ' + self.description + self.param + '_BNN retrieved')

        f = open('Data/Networks/' + self.description + self.param +'_BNN_metadata.pkl', 'rb')
        self.metadata = pickle.load(f)
        f.close()

    @staticmethod
    def prior_Normal(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale = 1),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Laplace_tas_w0(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Laplace(loc=tf.zeros(n, dtype=dtype)-7.04600133758504e-05, scale = 0.2770994628747123),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Laplace_tas_w1(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Laplace(loc=tf.zeros(n, dtype=dtype)-0.006575932027772069, scale = 3.7708420294564293),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Laplace_tas_w2(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Laplace(loc=tf.zeros(n, dtype=dtype)+0.12325552105903625, scale = 1.397196636057831),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Laplace_pr_w0(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Laplace(loc=tf.zeros(n, dtype=dtype)-2.1929042304691393e-05, scale = 0.17925633823616097),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Laplace_pr_w1(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Laplace(loc=tf.zeros(n, dtype=dtype)-0.04217399284243584, scale = 2.45063008081159),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Laplace_pr_w2(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Laplace(loc=tf.zeros(n, dtype=dtype)-0.02366732046357356, scale = 0.8781665796565905),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Student_tas_w0(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.StudentT(df= 2.018587782817862, loc=tf.zeros(n, dtype=dtype)-8.320658029026934e-05, scale = 0.0012642358989995973),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Student_tas_w1(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.StudentT(df= 9549418.76484643, loc=tf.zeros(n, dtype=dtype)+0.016591635677198013, scale = 0.33136260515375976),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Student_tas_w2(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.StudentT(df= 3233341.6638192795, loc=tf.zeros(n, dtype=dtype)+0.03929842830500162, scale = 0.4620396859536655),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Student_pr_w0(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.StudentT(df= 2.0028306325106566, loc=tf.zeros(n, dtype=dtype)-4.1298135866761645e-05, scale = 0.0013265594460017019),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Student_pr_w1(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.StudentT(df= 7549080.463531487, loc=tf.zeros(n, dtype=dtype)+0.004352458988547579, scale = 0.3404959705372761),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def prior_Student_pr_w2(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the prior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for DenseVariational. The default is None.

        Returns
        -------
        Function
            Takes an input and produces the prior distribution,
            provided as a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.StudentT(df= 8548865.495691143, loc=tf.zeros(n, dtype=dtype)+0.010639840353200692, scale = 0.4481909934656285),
            reinterpreted_batch_ndims=1)

    @staticmethod
    def posterior(kernel_size, bias_size=1, dtype=None):
        """
        Return a callable which takes an input and produces the posterior distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (i.e. the weights).
        bias_size : int, optional
            Size of the bias term. The default is 1.
        dtype : dtype obj, optional
            Data type. Requested for the VariableLayer. The default is None.

        Returns
        -------
        Keras Sequential obj
            A callable which takes the input and produces the posterior distribution,
            provided through a Keras Sequential model containing a Tensorflow Probability Distribution instance.

        """
        n = kernel_size + bias_size
        return tfk.Sequential([
            tfp.layers.VariableLayer(
                tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
            tfp.layers.IndependentNormal(n)])
