# -*- coding: utf-8 -*-
"""
Sherpa_Trial_Script.py is the trial script for Sherpa Bayesian Optimization and
Sherpa Local Search. It gets called by a Sherpa runner script. 
Using the information provided by the Sherpa runner script, namely
database information and hyperparameter set, Sherpa_Trial_Script.py preprocesses
the dataset, builds the requested neural network, and fits it by interfacing
with the mlAuxiliary.py library.

Information contained in a Sherpa trial object:
    Architecture Hyperparameters:
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
    Fitting Hyperparameters:
        alpha - Learning rate for the Adam optimizer
        beta_1 - Parameter beta_1 of the Adam optimizer
        beta_2 - Parameter beta_2 of the Adam optimizer
        regL2 - L2 regularizer weight
        batchSize - Batch size to be used during fitting
    Other:
        database - Dummy hyperparameter: placeholder for the database name
        seed - Seed for all randomness
        nRepetitions - Number of repetitions per trial

Sections:
    Imports
    Main Script

Version: 1.0
Last edit: 2021-11-15
Author: Dinis Abranches
"""

######################################################
# Imports
######################################################

# General
import os

# Specific
import pandas
import sherpa
import numpy
from sklearn.metrics import r2_score

# Local
from . import mlAuxiliary as ml

######################################################
# Main Script
######################################################

# Get trial object from MongoDB
client=sherpa.Client(host='127.0.0.1') # '127.0.0.1' is the local host
trial=client.get_trial()

# Configure seed
ml.config(seed=trial.parameters['seed'])

# Build databasePath from the trial object and load database
databasePath=os.path.join(os.path.dirname(__file__),
                          '..',
                          '..',
                          'Databases',
                          trial.parameters['database']+'.csv')
mlDatabase=pandas.read_csv(databasePath,dtype=str)

# Define label splitting and normalization to be used
if trial.parameters['database']=='VP_mlDatabase' or trial.parameters['database']=='S_mlDatabase' or trial.parameters['database']=='S_25_mlDatabase':
    labelNorm='LogStand'
    stratType='Log'
else:
    labelNorm='Standardization'
    stratType='Standard'
   
# Check wether mlDatabase is of type 1 (no temperature input) or type 2 (temperature input)
if mlDatabase.shape[1]==56: CNN_type=1
elif mlDatabase.shape[1]==57: CNN_type=2

# Convert mlDatabase to arrays of training and testing features and labels
X_Train,Y_Train,X_Test,Y_Test=ml.pandas2array(mlDatabase,stratType=stratType,splitFrac=0.8,seed=trial.parameters['seed'])

# Normalize Labels
Y_Train,scaler_Y=ml.normalize(Y_Train,method=labelNorm)
Y_Test=ml.normalize(Y_Test,method=labelNorm,skScaler=scaler_Y)[0]

# Normalize Features
if CNN_type==1: # Normalize all features with Log+bStand
    X_Train,scaler_X=ml.normalize(X_Train,method='Log+bStand')
    X_Test=ml.normalize(X_Test,method='Log+bStand',skScaler=scaler_X)[0]
elif CNN_type==2:
    # Normalize sigma profile features with logTransform and temoperature feature with Standardization
    X_Train[:,0:1],scaler_X1=ml.normalize(X_Train[:,0:1],method='Standardization')
    X_Train[:,1:],scaler_X2=ml.normalize(X_Train[:,1:],method='Log+bStand')
    X_Test[:,0:1]=ml.normalize(X_Test[:,0:1],method='Standardization',skScaler=scaler_X1)[0]
    X_Test[:,1:]=ml.normalize(X_Test[:,1:],method='Log+bStand',skScaler=scaler_X2)[0]

# Reshape for convolution (and for temperature input)
X_Train_Input=X_Train.reshape(X_Train.shape[0],X_Train.shape[1],1)
X_Test_Input=X_Test.reshape(X_Test.shape[0],X_Test.shape[1],1)
if CNN_type==2: # Separate temperature feature from sigma profile features
    X_Train_Input=[X_Train_Input[:,1:,0],X_Train_Input[:,0,0]]
    X_Test_Input=[X_Test_Input[:,1:,0],X_Test_Input[:,0,0]]

# Define CNN architecture
architecture={'actFunction' : trial.parameters['actFunction'],
              'conv1_Filters' : trial.parameters['conv1_Filters'],
              'conv1_kernelSize' : trial.parameters['conv1_kernelSize'],
              'conv1_strides' : trial.parameters['conv1_strides'],
              'pool1_size' : trial.parameters['pool1_size'],
              'pool1_type' : trial.parameters['pool1_type'],
              'conv2_Filters' : trial.parameters['conv2_Filters'],
              'conv2_kernelSize' : trial.parameters['conv2_kernelSize'],
              'conv2_strides' : trial.parameters['conv2_strides'],
              'pool2_size' : trial.parameters['pool2_size'],
              'pool2_type' : trial.parameters['pool2_type'],
              'dense1_Nodes' : trial.parameters['dense1_Nodes'],
              'dense2_Nodes' : trial.parameters['dense2_Nodes'],
              'alpha' : trial.parameters['alpha'],
              'beta_1' : trial.parameters['beta_1'],
              'beta_2' : trial.parameters['beta_2'],
              'regL2' : trial.parameters['regL2']}

# Report artificial metric if data to parameters ratio is smaller than 3
model=ml.generateCNN(architecture,CNN_type)
if model.count_params()>mlDatabase.shape[0]/3:
    metric=[numpy.log(1/0.1)]
# Perform model fitting
else: 
    metric=[]
    for n in range(trial.parameters['nRepetitions']): # N repetitions per trial
        # Generate CNN model
        model=ml.generateCNN(architecture,CNN_type)
        # Fit model
        model,hist,trainMetric,testMetric=ml.modelFit(model,
                                                      X_Train_Input,
                                                      Y_Train,
                                                      X_Test_Input,
                                                      Y_Test,
                                                      batchSize=trial.parameters['batchSize'],
                                                      verbose=0)
        # Predict
        Y_Train_Predicted=model.predict(X_Train_Input)
        Y_Test_Predicted=model.predict(X_Test_Input)
        # Calculate R^2 and append to metric
        r2=r2_score(Y_Test,Y_Test_Predicted)
        if r2<0.1: r2=0.1
        metric.append(numpy.log(1/r2))
# Calculate average metric
if len(metric)>1: # If there were N repetitions, report the average of the best N-1 results
    metric=numpy.average(sorted(metric)[:-1])
else:
    metric=metric[0]

# Finish trial by sending metrics
client.send_metrics(trial=trial,iteration=1,objective=metric)