# SP_ML_CC
GitHub Repository for the manuscript "Sigma Profiles in Deep Learning: Towards a Universal Molecular Descriptor". <br />
Citation: <br />
[https://doi.org/10.1039/D2CC01549H](https://doi.org/10.1039/D2CC01549H) <br />
Dinis O. Abranches, Yong Zhang, Edward J. Maginn, Yamil J. Colón. "Sigma profiles in deep learning: towards a universal molecular descriptor". Chem. Commun., 2022,58, 5630-5633

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

## Dependencies
The following Python packages are needed:

  Flask==2.0.2
  
  GPy==1.10.0
  
  GPyOpt==1.2.6
  
  keras==2.6.0
  
  Keras-Preprocessing==1.1.2
  
  numpy==1.19.5
  
  pandas==1.3.3
  
  parameter-sherpa==1.0.6
  
  paramz==0.9.5
  
  pymongo==3.12.0
  
  scikit-learn==1.0
  
  scipy==1.4.1
  
  tensorflow==2.6.0
  
  tensorflow-estimator==2.6.0
  
