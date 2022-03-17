# SP_ML_CC
GitHub Repository for "Sigma Profiles in Deep Learning: Towards a Universal Molecular Descriptor"

## Code
This folder contains the Python code used to generate and fit the neural networks developed in this work, including the hyperparameter optimization. It also contains an example Jupyter Notebook.

## Databases
This folder contains all ML databases (csv files) used in this work:
  MM - Molar Mass
  BP - Normal Boiling Temperature
  VP - Vapor Pressure at 25 ºC
  D_20 - Density at 20 ºC
  RI_20 - Refractive Index at 20 ºC
  S_25 - Aqueous Solubility at 25 ºC
  D - Density
  RI - Refractive Index
  S - Aqueous Solubility
It also contains the index file of the VT-2005 sigma profile database and a log file documenting the changes made in this work.

## Models
This folder contains the TensorFlow models of each neural network developed (.h5 files), as well as the normalization weights of features and labels (sklearn scaler objects stored in pkl files). CSV files with the results for the training, validation, and testing sets for each ML database are also included.

## Sherpa_Logs
This folder contains the intermediate results of the hyperparameter optimization for each neural network.
