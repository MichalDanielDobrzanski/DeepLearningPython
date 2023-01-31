import sys
sys.path.append('..')

from elements import mnist_loader
import numpy as np
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('../mnist.pkl.gz')
training_data = list(training_data)
test_data = list(test_data)

plt.figure(figsize=(15,5))
for i in range(1,17):
    plt.subplot(2,8,i)
    plt.imshow(training_data[i][0].reshape(28,28), cmap='Greys')
    plt.xlabel(f'label: {np.argmax(training_data[i][1])}')
plt.show()