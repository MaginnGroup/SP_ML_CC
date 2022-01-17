# -*- coding: utf-8 -*-
"""
Sherpa_Local_Search is the Sherpa runner script for the hyperparameter optimization
of the neural networks developed in this work using the Local Search algorithm. 
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
database='S_25_mlDatabase'

# Defining Initial Hyperparameter space for Architecture Tuning



hyperParameter=[sherpa.Choice(name='actFunction',range=['swish','relu']),
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
                 sherpa.Choice(name='alpha',range=[0.001]),
                 sherpa.Choice(name='beta_1',range=[0.99]),
                 sherpa.Choice(name='beta_2',range=[0.999]),
                 sherpa.Choice(name='regL2',range=[0.001]),
                 sherpa.Choice(name='batchSize',range=[16]),
                 sherpa.Choice(name='database',range=[database]),
                 sherpa.Choice(name='seed',range=[42]),
                 sherpa.Choice(name='nRepetitions',range=[1])]

# Defining Initial Seed Configuration
seed_configuration={'actFunction' : 'relu',
                    'conv1_Filters' : 3,
                    'conv1_kernelSize' : 10,
                    'conv1_strides' : 7,
                    'pool1_size' : 4,
                    'pool1_type' : 'max',
                    'conv2_Filters' : 4,
                    'conv2_kernelSize' : 1,
                    'conv2_strides' : 1,
                    'pool2_size' : 3,
                    'pool2_type' : 'max',
                    'dense1_Nodes' : 1,
                    'dense2_Nodes' : 7,
                    'alpha' : 0.001,
                    'beta_1' : 0.99,
                    'beta_2' : 0.999,
                    'regL2' : 0.001,
                    'batchSize' : 16,
                    'database' : database,
                    'seed' : 42,
                    'nRepetitions' : 1,
                    }

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

for n in range(5): # 10 iterations
# First Local Search (Architecture)
    # Define output folder
    output_dir=os.path.join(os.path.dirname(__file__),
                            '..',
                            'Sherpa_Logs',
                            database+'_2_LocalSearch',
                            'Architecture_'+str(n))
    # Run Run Local Search (Architecture)
    results=sherpa.optimize(parameters=hyperParameter,
                            algorithm=sherpa.algorithms.LocalSearch(seed_configuration),
                            lower_is_better=True,
                            scheduler=sherpa.schedulers.LocalScheduler(),
                            command='python -m lib.Sherpa_Trial_Script', # Run with -m for local imports inside trial script
                            output_dir=output_dir,
                            max_concurrent=40,
                            disable_dashboard=True)
    # Defining new Hyperparameter space for Fitting
    hyperParameter=[sherpa.Choice(name='actFunction',range=[results.get('actFunction')]),
                     sherpa.Choice(name='conv1_Filters',range=[results.get('conv1_Filters')]),
                     sherpa.Choice(name='conv1_kernelSize',range=[results.get('conv1_kernelSize')]),
                     sherpa.Choice(name='conv1_strides',range=[results.get('conv1_strides')]),
                     sherpa.Choice(name='pool1_size',range=[results.get('pool1_size')]),
                     sherpa.Choice(name='pool1_type',range=[results.get('pool1_type')]),
                     sherpa.Choice(name='conv2_Filters',range=[results.get('conv2_Filters')]),
                     sherpa.Choice(name='conv2_kernelSize',range=[results.get('conv2_kernelSize')]),
                     sherpa.Choice(name='conv2_strides',range=[results.get('conv2_strides')]),
                     sherpa.Choice(name='pool2_size',range=[results.get('pool2_size')]),
                     sherpa.Choice(name='pool2_type',range=[results.get('pool2_type')]),
                     sherpa.Choice(name='dense1_Nodes',range=[results.get('dense1_Nodes')]),
                     sherpa.Choice(name='dense2_Nodes',range=[results.get('dense2_Nodes')]),
                     sherpa.Continuous(name='alpha',range=[10**-5,10**-1],scale='log'),
                     sherpa.Continuous(name='beta_1',range=[0.9,0.9999],scale='log'),
                     sherpa.Continuous(name='beta_2',range=[0.9,0.99999],scale='log'),
                     sherpa.Continuous(name='regL2',range=[10**-5,10**-1],scale='log'),
                     sherpa.Ordinal(name='batchSize',range=[8,16,32,64]),
                     sherpa.Choice(name='database',range=[database]),
                     sherpa.Choice(name='seed',range=[42]),
                     sherpa.Choice(name='nRepetitions',range=[1])]
    # Defining new Seed Configuration
    seed_configuration={'actFunction' : results.get('actFunction'),
                        'conv1_Filters' : results.get('conv1_Filters'),
                        'conv1_kernelSize' : results.get('conv1_kernelSize'),
                        'conv1_strides' : results.get('conv1_strides'),
                        'pool1_size' : results.get('pool1_size'),
                        'pool1_type' : results.get('pool1_type'),
                        'conv2_Filters' : results.get('conv2_Filters'),
                        'conv2_kernelSize' : results.get('conv2_kernelSize'),
                        'conv2_strides' : results.get('conv2_strides'),
                        'pool2_size' : results.get('pool2_size'),
                        'pool2_type' : results.get('pool2_type'),
                        'dense1_Nodes' : results.get('dense1_Nodes'),
                        'dense2_Nodes' : results.get('dense2_Nodes'),
                        'alpha' : results.get('alpha'),
                        'beta_1' : results.get('beta_1'),
                        'beta_2' : results.get('beta_2'),
                        'regL2' : results.get('regL2'),
                        'batchSize' : results.get('batchSize'),
                        'database' : database,
                        'seed' : 42,
                        'nRepetitions' : results.get('nRepetitions'),
                        }
    # Kill leftover mongod process
    os.system(r'pkill -f mongod')
    time.sleep(10)
    # Clean log
    cleanLog(output_dir)
# Second Local Search (Fitting)
    # Define output folder
    output_dir=os.path.join(os.path.dirname(__file__),
                            '..',
                            'Sherpa_Logs',
                            database+'_2_LocalSearch',
                            'Fitting_'+str(n))
    # Run Local Search (Fitting)
    results=sherpa.optimize(parameters=hyperParameter,
                            algorithm=sherpa.algorithms.LocalSearch(seed_configuration),
                            lower_is_better=True,
                            scheduler=sherpa.schedulers.LocalScheduler(),
                            command='python -m lib.Sherpa_Trial_Script', # Run with -m for local imports inside trial script
                            output_dir=output_dir,
                            max_concurrent=40,
                            disable_dashboard=True)
    # Defining new Hyperparameter space for architecture
    hyperParameter=[sherpa.Choice(name='actFunction',range=['swish','relu']),
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
                     sherpa.Choice(name='alpha',range=[results.get('alpha')]),
                     sherpa.Choice(name='beta_1',range=[results.get('beta_1')]),
                     sherpa.Choice(name='beta_2',range=[results.get('beta_2')]),
                     sherpa.Choice(name='regL2',range=[results.get('regL2')]),
                     sherpa.Choice(name='batchSize',range=[results.get('batchSize')]),
                     sherpa.Choice(name='database',range=[database]),
                     sherpa.Choice(name='seed',range=[42]),
                     sherpa.Choice(name='nRepetitions',range=[1])]
    # Defining new Seed Configuration
    seed_configuration={'actFunction' : results.get('actFunction'),
                        'conv1_Filters' : results.get('conv1_Filters'),
                        'conv1_kernelSize' : results.get('conv1_kernelSize'),
                        'conv1_strides' : results.get('conv1_strides'),
                        'pool1_size' : results.get('pool1_size'),
                        'pool1_type' : results.get('pool1_type'),
                        'conv2_Filters' : results.get('conv2_Filters'),
                        'conv2_kernelSize' : results.get('conv2_kernelSize'),
                        'conv2_strides' : results.get('conv2_strides'),
                        'pool2_size' : results.get('pool2_size'),
                        'pool2_type' : results.get('pool2_type'),
                        'dense1_Nodes' : results.get('dense1_Nodes'),
                        'dense2_Nodes' : results.get('dense2_Nodes'),
                        'alpha' : results.get('alpha'),
                        'beta_1' : results.get('beta_1'),
                        'beta_2' : results.get('beta_2'),
                        'regL2' : results.get('regL2'),
                        'batchSize' : results.get('batchSize'),
                        'database' : database,
                        'seed' : 42,
                        'nRepetitions' : results.get('nRepetitions'),
                        }
    # Kill any leftovers
    os.system(r'pkill -f mongod')
    time.sleep(10)
    # Clean log
    cleanLog(output_dir)