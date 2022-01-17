# -*- coding: utf-8 -*-
"""
This package is an auxiliary library used to preprocess data and interface with
TensorFlow (tf.keras).

List of functions:
    . pandas2array()
        - Converts pandas DataFrame into numpy array while spitting the data into
        training and testing sets.
    . 
    
Sections:
    . Imports
    . Main Functions
    
Last edit: 2021-11-27
Author: Dinis Abranches
"""


            
    
######################################################
# Imports
######################################################

# General
import os # Basic
import random # Basic

# Specific
import pandas # Specific
import numpy # Specific
import tensorflow as tf # Specific
from sklearn.model_selection import train_test_split # Specific
from sklearn import preprocessing # Specific

######################################################
# Main Functions
######################################################

def config(seed=None,CUDA='-1',nThreads=14):
    """
    config() configures the universal seed for Python, TensorFlow, numpy, and
    random, sets the number of CUDA devices visible to TensorFlow, and sets the
    number of CPU threads to be used by TensorFlow.
    
    Parameters
    ----------
    seed : int or None, optional
        Seed to be used in TensorFlow, numpy, and random.
        Default: None
    CUDA : string, optional
        Number of CUDA devices visbile to Tensorflow.
        Default: '-1'
    nThreads : int, optional
        Number of CPU threads available to TensorFlow.
        Default: 1
    
    Returns
    -------
    None.

    """
    os.environ["CUDA_VISIBLE_DEVICES"]=CUDA
    tf.config.threading.set_inter_op_parallelism_threads(nThreads)
    tf.config.threading.set_intra_op_parallelism_threads(nThreads)
    if seed is not None:
        os.environ['PYTHONHASHSEED']=str(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)

def pandas2array(mlDatabase,stratType='Standard',splitFrac=0.8,seed=None):
    """
    pandas2array() converts the mlDatabase into numpy arrays and performs a 
    stratefied splitting, originating:
        X_Train - features of the training set
        Y_Train - labels of the training set
        X_Test - features of the testing set
        Y_Test - labels of the testing set

    Parameters
    ----------
    mlDatabase : pandas DataFrame
        mlDatabase for a given property; can be of size (N,56) or (N,57), depending
        on wether the database contains a temperature feature.
    stratType : string
        Stratification on the labels ('Standard') or the natural log of the labels ('Log')
    splitFrac : float, optional
        Fraction of the dataset to be used as training data.
        Default: 0.8
    seed : int, optional
        Random seed (equal to pandas random_state)
        when spliting the data.
        Default: None

    Returns
    -------
    X_Train : numpy array
        Array of training features of size (N*splitFrac,51) or (N*splitFrac,52),
        depending on wether temperature is used as a feature.
    Y_Train : numpy array
        Array of training labels of size (N*splitFrac,)
    X_Test : numpy array
        Array of testing features of size (N*(1-splitFrac),51) or (N*(1-splitFrac),52),
        depending on wether temperature is used as a feature.
    Y_Test : numpy array
        Array of testing labels of size (N*(1-splitFrac),)
    """
    # Get columns of interest from mlDatabase (drop index, name, CAS, and notes)
    A=mlDatabase.iloc[:,3:-1]
    # Convert to numpy as float64
    A=A.to_numpy(dtype='float64')
    # Split into features and labels, and into training and testing sets
    if A.shape[1]==52: # Data does not contain a temperature feature
        X=A[:,1:] # Features
        Y=A[:,0] # Labels
    elif A.shape[1]==53: # Data contains a temperature feature
        Y=A[:,1] # Labels
        X=numpy.delete(A,1,axis=1) # Features
    # Stratify Y
    # Iterate over number of bins, trying to find the larger number of bins that
    # guarantees at least 5 values per bin
    for n in range(1,100):
        # Bin Y using n bins
        stratifyVector=pandas.cut(Y,n,labels=False)
        # Define isValid (all bins have at least 5 values)
        isValid=True
        # Check that all bins have at least 5 values
        for k in range(n):
            if numpy.count_nonzero(stratifyVector==k)<5:
                isValid=False
        #If isValid is false, n is too large; nBins must be the previous iteration
        if not isValid:
            nBins=n-1
            break
    # Generate vector for stratified splitting based on labels
    stratifyVector=pandas.cut(Y,nBins,labels=False)
    # Perform splitting
    X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,
                                                   Y,
                                                   train_size=splitFrac,
                                                   random_state=seed,
                                                   stratify=stratifyVector)
    # Return
    return X_Train,Y_Train,X_Test,Y_Test

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method specified
    and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, Array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . LogStand - standardization on the log of the variable, (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be zero; uses a small buffer, (log(x+b)-mean(log(x+b)))/std(log(x+b))
        Defalt: 'Standardization'
    reverse : bool
        Wether to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1: inputArray=inputArray.reshape((-1,1))
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        skScaler=preprocessing.StandardScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        inputArray=skScaler.inverse_transform(inputArray) # Rescale the data back to its original distribution
        # Check method
        if method=='LogStand': inputArray=numpy.exp(inputArray)
        elif method=='Log+bStand': inputArray=numpy.exp(inputArray)-10**-3
    elif not reverse:
        # Check method
        if method=='Standardization': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler

def generateCNN(architecture,CNN_type):
    """
    generateCNN() generates the CNN (tf.keras model) with the desired architecture. 

    Parameters
    ----------
    architecture : dictionary
        Dictionary containing all the information related to architecture and fitting
        of the desired CNN.
        Items:
            actFunction - Activation function for all nodes except the output
            conv1_Filters - Number of filters in the 1st Conv layer
            conv1_kernelSize - Kernel size in the 1st Conv layer
            conv1_strides- Number of kernel strides in the 1st Conv layer
            pool1_size - Kernel size of the first pooling layer
            pool1_type - Pool type operation of the 1st Pooling layer (average or max)
            conv2_Filters - Number of filters in the 2nd Conv layer (can be zero)
            conv2_kernelSize - Kernel size in the 2nd Conv layer
            conv2_strides- Number of kernel strides in the 2nd Conv layer
            pool2_size - Kernel size of the second pooling layer
            pool2_type - Pool type operation of the 2nd Pooling layer (average or max)
            dense1_Nodes - Number of nodes in the 1st dense layer
            dense2_Nodes - Number of nodes in the 2nd dense layer (can be zero)
            alpha - Learning rate for the Adam optimizer
            beta_1 - Parameter beta_1 of the Adam optimizer
            beta_2 - Parameter beta_2 of the Adam optimizer
            regL2 - L2 regularizer weight
    CNN_type : int
        Type of CNN (1 or 2). Type 1 has no temperature input while Type 2 does.

    Returns
    -------
    model : tf.keras object
        tf.keras model object of the CNN.

    """
    # Define CNN architecture from the input
    alpha=architecture.get('alpha')
    beta_1=architecture.get('beta_1')
    beta_2=architecture.get('beta_2')
    actFunction=architecture.get('actFunction')
    conv1_Filters=architecture.get('conv1_Filters')
    conv1_kernelSize=architecture.get('conv1_kernelSize')
    conv1_strides=architecture.get('conv1_strides')
    pool1_size=architecture.get('pool1_size')
    pool1_type=architecture.get('pool1_type')
    conv2_Filters=architecture.get('conv2_Filters')
    conv2_kernelSize=architecture.get('conv2_kernelSize')
    conv2_strides=architecture.get('conv2_strides')
    pool2_size=architecture.get('pool2_size')
    pool2_type=architecture.get('pool2_type')
    dense1_Nodes=architecture.get('dense1_Nodes')
    dense2_Nodes=architecture.get('dense2_Nodes')
    regL2=architecture.get('regL2')
    # Define weight initializer method
    kernelInit='he_uniform'
    # Layer number
    k=0 
    # Input layer
    inputLayer_1=tf.keras.Input(shape=(51,1),name='Layer_'+str(k))
    # Define second input if temperature is a feature (CNN_type is 2)
    if CNN_type==2: inputLayer_2=tf.keras.Input(shape=(1,),name='Layer_T')    
    k=k+1 # Update layer number
    # 1st 1D Convolution Layer
    auxLayers=tf.keras.layers.Conv1D(conv1_Filters,
                                     conv1_kernelSize,
                                     strides=conv1_strides,
                                     padding='same',
                                     activation=actFunction,
                                     kernel_initializer=kernelInit,
                                     kernel_regularizer=tf.keras.regularizers.L2(regL2),
                                     bias_regularizer=tf.keras.regularizers.L2(regL2),
                                     name='Layer_'+str(k))(inputLayer_1)
    k=k+1 # Update layer number
    # 1st Pooling
    if pool1_type=='max':
        auxLayers=tf.keras.layers.MaxPooling1D(pool_size=pool1_size,
                                               padding='same',
                                               data_format='channels_first',
                                               name='Layer_'+str(k))(auxLayers)
    elif pool1_type=='avg':
        auxLayers=tf.keras.layers.AveragePooling1D(pool_size=pool1_size,
                                                   padding='same',
                                                   data_format='channels_first',
                                                   name='Layer_'+str(k))(auxLayers)
    k=k+1 # Update layer number
    # 2nd 1D Convolution Layer
    if conv2_Filters>0:
        auxLayers=tf.keras.layers.Conv1D(conv2_Filters,
                                         conv2_kernelSize,
                                         strides=conv2_strides,
                                         padding='same',
                                         activation=actFunction,
                                         kernel_initializer=kernelInit,
                                         kernel_regularizer=tf.keras.regularizers.L2(regL2),
                                         bias_regularizer=tf.keras.regularizers.L2(regL2),
                                         name='Layer_'+str(k))(auxLayers)
        k=k+1 # Update layer number
        # 2nd Pooling (only exists if 2nd 1D Conv exists)
        if pool2_type=='max':
            auxLayers=tf.keras.layers.MaxPooling1D(pool_size=pool2_size,
                                                   padding='same',
                                                   data_format='channels_first',
                                                   name='Layer_'+str(k))(auxLayers)
        elif pool2_type=='avg':
            auxLayers=tf.keras.layers.AveragePooling1D(pool_size=pool2_size,
                                                       padding='same',
                                                       data_format='channels_first',
                                                       name='Layer_'+str(k))(auxLayers)
        k=k+1 # Update layer number
    # Flatten Layer
    auxLayers=tf.keras.layers.Flatten(data_format='channels_first',
                                      name='Layer_'+str(k))(auxLayers)
    k=k+1 # Update layer number
    # Combine temperature input with auxLayers if CNN_type is of type 2
    if CNN_type==2: auxLayers=tf.keras.layers.concatenate([auxLayers,inputLayer_2],
                                                          name='Layer_Concatenate')
    # 1st Dense
    auxLayers=tf.keras.layers.Dense(dense1_Nodes,
                                    activation=actFunction,
                                    kernel_initializer=kernelInit,
                                    kernel_regularizer=tf.keras.regularizers.L2(regL2),
                                    bias_regularizer=tf.keras.regularizers.L2(regL2),
                                    name='Layer_'+str(k))(auxLayers)
    k=k+1 # Update layer number
    # 2nd Dense
    if dense2_Nodes>0:
        auxLayers=tf.keras.layers.Dense(dense2_Nodes,
                                        activation=actFunction,
                                        kernel_initializer=kernelInit,
                                        kernel_regularizer=tf.keras.regularizers.L2(regL2),
                                        bias_regularizer=tf.keras.regularizers.L2(regL2),
                                        name='Layer_'+str(k))(auxLayers)
        k=k+1 # Update layer number
    # Output
    outputLayer=tf.keras.layers.Dense(1,
                                      activation='linear',
                                      kernel_initializer=kernelInit,
                                      kernel_regularizer=tf.keras.regularizers.L2(regL2),
                                      bias_regularizer=tf.keras.regularizers.L2(regL2),
                                      name='Layer_'+str(k))(auxLayers)
    # Build model
    if CNN_type==1: model=tf.keras.Model(inputs=inputLayer_1,outputs=outputLayer)
    elif CNN_type==2: model=tf.keras.Model(inputs=[inputLayer_1,inputLayer_2],outputs=outputLayer)
    # Generate optimizer
    optimizer=tf.keras.optimizers.Adam(learning_rate=alpha,
                                       beta_1=beta_1,
                                       beta_2=beta_2)
    # Compile model with mean_squared_error loss function
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics='mean_absolute_error')
    # Output
    return model

def modelFit(model,X_Train,Y_Train,X_Test,Y_Test,batchSize=32,verbose=0):
    """
    modelFit() fits the CNN model using the training data provided. Early stopping
    is employed using the testing data provided.

    Parameters
    ----------
    model : tf.keras object
        tf.keras model object of the CNN.
    X_Train : numpy array
        Features of the training set.
    Y_Train : numpy array
        Labels of the training set.
    X_Test : numpy array
        Features of the testing set set.
    Y_Test : numpy array
        Labels of the testing set.
    batchSize : int
        Batch size for the fitting. 
        Default: 32
    verbose : int
        Verbose option for tf.keras model.fit. 
        Default: 0

    Returns
    -------
    model : tf.keras object
        tf.keras model object of the CNN.
    hist : TYPE
        DESCRIPTION.
    trainMetric : TYPE
        DESCRIPTION.
    testMetric : TYPE
        DESCRIPTION.

    """
    # Define early stopping
    earlyStop=tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',
                                               min_delta=0,
                                               patience=500,
                                               verbose=0,
                                               mode='min',
                                               baseline=None,
                                               restore_best_weights=True)
    # Fit the model
    hist=model.fit(x=X_Train,
                   y=Y_Train,
                   batch_size=batchSize,
                   epochs=200000, # Arbitrarily large number (fitting is stopped by earlyStop)
                   verbose=verbose,
                   validation_data=(X_Test,Y_Test),
                   callbacks=earlyStop)
    # Get metrics
    trainMetric=model.evaluate(X_Train,Y_Train)[1]
    testMetric=model.evaluate(X_Test,Y_Test)[1]
    # Output
    return model,hist,trainMetric,testMetric
