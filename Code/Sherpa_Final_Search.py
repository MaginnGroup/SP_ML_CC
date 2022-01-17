# -*- coding: utf-8 -*-
"""
Sherpa_Final_Search is the Sherpa runner script for the final fitting of the neural
networks. It runs through each seed and saves the best model obtained.
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
database='VP_mlDatabase'

# Defining Hyperparameter space
hyperParameters=[sherpa.Choice(name='actFunction',range=['swish']),
                 sherpa.Choice(name='conv1_Filters',range=[5]),
                 sherpa.Choice(name='conv1_kernelSize',range=[6]),
                 sherpa.Choice(name='conv1_strides',range=[3]),
                 sherpa.Choice(name='pool1_size',range=[3]),
                 sherpa.Choice(name='pool1_type',range=['avg']),
                 sherpa.Choice(name='conv2_Filters',range=[1]),
                 sherpa.Choice(name='conv2_kernelSize',range=[5]),
                 sherpa.Choice(name='conv2_strides',range=[5]),
                 sherpa.Choice(name='pool2_size',range=[5]),
                 sherpa.Choice(name='pool2_type',range=['max']),
                 sherpa.Choice(name='dense1_Nodes',range=[7]),
                 sherpa.Choice(name='dense2_Nodes',range=[6]),
                 sherpa.Choice(name='alpha',range=[0.0012]),
                 sherpa.Choice(name='beta_1',range=[0.99]),
                 sherpa.Choice(name='beta_2',range=[0.999]),
                 sherpa.Choice(name='regL2',range=[0.001]),
                 sherpa.Choice(name='batchSize',range=[32]),
                 sherpa.Choice(name='database',range=[database]),
                 sherpa.Discrete(name='seed',range=[0,2000]),
                 sherpa.Choice(name='nRepetitions',range=[1])
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
                        database+'_3_SeedSearch')

# Defining optimizer and run
results=sherpa.optimize(parameters=hyperParameters,
                        algorithm=sherpa.algorithms.GridSearch(num_grid_points=2001),
                        lower_is_better=True,
                        scheduler=sherpa.schedulers.LocalScheduler(),
                        command='python -m lib.Sherpa_Trial_Script_Final', # Run with -m for local imports inside trial script
                        output_dir=output_dir,
                        max_concurrent=10,
                        disable_dashboard=True)

# Kill leftover mongod process
os.system(r'pkill -f mongod')
time.sleep(10)

# Clean log folder
cleanLog(output_dir)