# GNN-gs-properties
Implementation of graphs neural networks to learn ground-state properties.

## Data collection
Choose your settings (dimensions, distribution, bond dimension) in the dmrg.py file and run it to compute the ground state properties with DMRG . The data are written as graphs in the data_preparation.py module. You will need to install the dmrg_env.txt packages to run DMRG.

## ML model
The GNN model is implemented in the model.py file. The training is performed in the training.py file, where you can also choose the number of hidden layers, number of epochs, learning rate and everything related to optimization. The whole pipeline is started through the main.py file, and you will need to install torch_env.txt. In the main file, you can also choose the degree of noise, and the ratio training vs validation split. 
