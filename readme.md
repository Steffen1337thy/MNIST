This is the MNIST-digit recognizer project.

Best accuracy: 0.994327

NN-architecture:
 -  3 convolutional layers
 -  2 maxpool
 -  1 dropout
 -  2 linear layers

Optimizer: SGD

Data size:
- 54.000 training 
- 6.000 validation
- 10.000 test

Best params:
- epochs: 30
- batch size: 64
- learning rate: 0.01
- momentum: 0.9

Special features:
- progress bars
- graphics for train/val accuracy over epochs
- auto-protocol for the models to documentate parameters 

Contents of the folder:

- the MNIST.xslx contains the documentation
- the models directory contains the different models testet
- the figures directory contains the figures for the accuracy/loss development of each model
