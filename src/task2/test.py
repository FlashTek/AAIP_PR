from network import Network

from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

def one_hot(x, n_classes):
    x_oh = np.zeros((len(x), n_classes))
    x_oh[np.arange(len(x)), x] = 1.0

    return x_oh

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

def get_test_score():
    prediction = net.predict(x_test)
    pred = np.zeros_like(prediction)
    pred[np.arange(len(pred)), np.argmax(prediction, axis=1)] = 1.0
    prediction = pred

    return np.mean((y_test - prediction)**2)*10/2.0

epochs = 30
batch_sizes = [1, 5, 10, 25, 50, 100, 200, 400, 600, 800, 1000]
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]

results = []

for batch_size in batch_sizes:
    n_batches = int(len(x_train) / batch_size)

    for learning_rate in learning_rates:
        net = Network(28*28, 10)

        print("network initialized - test score: {0}".format(get_test_score()))

        epoch_trajectory = [get_test_score()]

        for epoch in range(epochs):
            #shuffle data randomly
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]

            for batch in range(n_batches):
                net.fit(x_train[batch*batch_size:(batch+1)*batch_size],
                        y_train[batch*batch_size:(batch+1)*batch_size],
                        eta=learning_rate)

            epoch_trajectory.append(get_test_score())
            print("epoch {0} finished - test score: {1}".format(epoch, epoch_trajectory[-1]))

        results.append((batch_size, learning_rate, epoch_trajectory))

import pickle

with open("result.pkl", "wb") as f:
    pickle.dump(results, f)