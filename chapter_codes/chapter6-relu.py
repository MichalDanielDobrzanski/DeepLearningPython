import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

from elements import network3
from elements.network3 import Network
from elements.network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

# read data:
training_data, validation_data, test_data = network3.load_data_shared('../mnist.pkl.gz')

# mini-batch size:
mini_batch_size = 10

net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)