# -*- coding: utf-8 -*-
"""
Sherpa_Bayesian_Optimization.py is the Sherpa runner script for the hyperparameter
optimization of the neural networks developed in this work using BayesianOptimization.
The script is universal for all databases, and the user only needs to change 
the name of the database to perform the optimization.

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
    Configuration
    Auxiliary Functions
    Main Script

Last edit: 2021-11-28
Author: Dinis Abranches
"""

######################################################
# Imports
######################################################

# General
import os
import shutil
import glob
import time

# Specific
import sherpa

######################################################
# Configuration
######################################################

# Database
database='D_mlDatabase'

# Defining Hyperparameter space
hyperParameters=[sherpa.Choice(name='actFunction',range=['swish','relu']),
                 sherpa.Ordinal(name='conv1_Filters',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Ordinal(name='conv1_kernelSize',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Ordinal(name='conv1_strides',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Ordinal(name='pool1_size',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Choice(name='pool1_type',range=['max','avg']),
                 sherpa.Ordinal(name='conv2_Filters',range=[0,1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Ordinal(name='conv2_kernelSize',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Ordinal(name='conv2_strides',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Ordinal(name='pool2_size',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Choice(name='pool2_type',range=['max','avg']),
                 sherpa.Ordinal(name='dense1_Nodes',range=[1,2,3,4,5,6,7,8,9,10]),
                 sherpa.Ordinal(name='dense2_Nodes',range=[0,1,2,3,4,5,6,7,8,9,10]),
                 # Dummy hyperparameters to communicate with Sherpa_Trial_Script.py:
                 sherpa.Choice(name='alpha',range=[0.001]),
                 sherpa.Choice(name='beta_1',range=[0.99]),
                 sherpa.Choice(name='beta_2',range=[0.999]),
                 sherpa.Choice(name='regL2',range=[0.001]),
                 sherpa.Choice(name='batchSize',range=[16]),
                 sherpa.Choice(name='database',range=[database]),
                 sherpa.Choice(name='seed',range=[None]),
                 sherpa.Choice(name='nRepetitions',range=[3])
                 ]

######################################################
# Auxiliary Functions
######################################################

def cleanLog(output_dir):
    """
    cleanLog() removes unnecessary files from the output dir of a Sherpa optimization.

    Parameters
    ----------
    output_dir : string
        Path to the output_dir of the Sherpa optimization.

    Returns
    -------
    None.

    """
    for file in glob.glob(os.path.join(output_dir,'WiredTiger*')): os.remove(file)
    for file in glob.glob(os.path.join(output_dir,'*.wt')): os.remove(file)
    for file in glob.glob(os.path.join(output_dir,'*.lock')): os.remove(file)
    for file in glob.glob(os.path.join(output_dir,'*.turtle')): os.remove(file)
    os.remove(os.path.join(output_dir,'config.pkl'))
    os.remove(os.path.join(output_dir,'storage.bson'))
    os.remove(os.path.join(output_dir,'log.txt'))
    shutil.rmtree(os.path.join(output_dir,'diagnostic.data'))
    shutil.rmtree(os.path.join(output_dir,'journal'))
    shutil.rmtree(os.path.join(output_dir,'jobs'))
    # Output    
    return
    

######################################################
# Main Script
######################################################

# Build output folder
output_dir=os.path.join(os.path.dirname(__file__),
                        '..',
                        'Sherpa_Logs',
                        database+'_1_Bayesian')

# Defining optimizer and run
results=sherpa.optimize(parameters=hyperParameters,
                        algorithm=sherpa.algorithms.GPyOpt(max_num_trials=5000,
                                                           max_concurrent=30,
                                                           verbosity=True),
                        lower_is_better=True,
                        scheduler=sherpa.schedulers.LocalScheduler(),
                        command='python -m lib.Sherpa_Trial_Script', # Run with -m for local imports inside trial script
                        output_dir=output_dir,
                        max_concurrent=30,
                        disable_dashboard=True)

# Kill leftover mongod process
os.system(r'pkill -f mongod')
time.sleep(10)

# Clean log folder
cleanLog(output_dir)