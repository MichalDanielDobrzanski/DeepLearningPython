import sys
sys.path.append('..')

from elements import network2
from elements import mnist_loader
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('../mnist.pkl.gz')
training_data = list(training_data)
test_data = list(test_data)

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
    net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)


plt.figure(figsize=(6,5))
plt.plot(training_cost[200:])
plt.grid()
plt.title('Cost on the training data')
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(6,5))
plt.plot(evaluation_accuracy[200:])
plt.grid()
plt.title('Accuracy (%) on the test data')
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(6,5))
plt.plot(evaluation_cost)
plt.grid()
plt.title('Cost on the test data')
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(6,5))
plt.plot(training_accuracy)
plt.grid()
plt.title('Accuracy (%) on the training data')
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(6,5))
plt.plot(np.array(training_accuracy[:30])/1000)
plt.plot(np.array(evaluation_accuracy[:30])/10000)
plt.grid()
plt.xlabel('Epoch')
plt.legend(['Accuracy on the training data','Accuracy on the test data'])
plt.show()
