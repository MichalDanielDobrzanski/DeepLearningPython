import sys
sys.path.append('..')

from elements import network2
from elements import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('../mnist.pkl.gz')
training_data = list(training_data)
test_data = list(test_data)

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
    net.SGD(training_data[:1000], 30, 10, 0.5,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    early_stopping_n=10)


plt.figure(figsize=(6,5))
plt.plot(np.array(evaluation_accuracy)/10000)
plt.grid()
plt.xlabel('Epoch')
plt.legend(['Accuracy on the test data'])
plt.show()