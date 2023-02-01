import sys
sys.path.append('..')

from elements import network3
from elements.network3 import Network
from elements.network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

# read data:
training_data, validation_data, test_data = network3.load_data_shared('../mnist.pkl.gz')

# mini-batch size:
mini_batch_size = 10

net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)