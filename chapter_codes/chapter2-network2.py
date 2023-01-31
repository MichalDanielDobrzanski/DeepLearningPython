import sys
sys.path.append('..')

from elements import network2
from elements import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('../mnist.pkl.gz')
training_data = list(training_data)
test_data = list(test_data)

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
    net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)