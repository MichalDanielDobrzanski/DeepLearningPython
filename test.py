"""
    Testing code for different neural network configurations.
    Adapted for Python 3.4.3

    Usage in shell:
        python3 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""

# ----------------------
# - read the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
# - network.py example:

import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# ----------------------
# - network2.py example:

# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# #net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
#     monitor_evaluation_accuracy=True)


# chapter 3 - Overfitting and regularization example
# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
#     monitor_evaluation_accuracy=True,
#     monitor_training_cost=True)


# ----------------------
# - network3.py example:

"""
    This deep network uses Theano with GPU acceleration support.
    I am using Ubuntu 16.04 with CUDA 7.5

"""

# import network3
# from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
# training_data, validation_data, test_data = network3.load_data_shared()
# mini_batch_size = 10
# net = Network([
#     FullyConnectedLayer(n_in=784, n_out=100),
#     SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
# net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
