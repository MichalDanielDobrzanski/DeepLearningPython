import sys
sys.path.append('..')

from elements import network
from elements import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('../mnist.pkl.gz')
training_data = list(training_data)
test_data = list(test_data)

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)