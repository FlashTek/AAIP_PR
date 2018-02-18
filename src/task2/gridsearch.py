"""
    Test the infleunce of the hyperparameters of the network by trying different values for the batch size and the learning rate

    Creates a pickled file called "result.pkl" which can be analyzed by calling
        python analyze.py
    to create plots. This file contains the used batch size, learning rate, the elapsed time and the development of the trainin & test loss over time
"""

from network import Network
import utility as ut
import numpy as np
import pickle
import time

def get_test_loss():
    """
        Evalutes the trained network net on the test data set

        returns:
            float - test loss
    """

    return ut.get_network_loss(net, x_test, y_test)

def get_train_loss():
    """
        Evalutes the trained network net on the train data set

        returns:
            float - train loss
    """

    return ut.get_network_loss(net, x_train, y_train)

(x_train, y_train), (x_test, y_test) = ut.load_data(reshape=True)

#define hypterparameters which will be tested
#number of epochs to train the model
epochs = 30
#batch size for the SGD
batch_sizes = [1, 5, 10, 25, 50, 100, 200, 400, 600, 800, 1000]
#learning rate for the update step
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]

#storage for the results, which are tuples (batch size, learning rate, train loss as a function of the epoch, test loss)
results = []

#loop over all hyperparameters
for i, batch_size in enumerate(batch_sizes):
    #number of batches in each epoch
    n_batches = int(len(x_train) / batch_size)

    for j, learning_rate in enumerate(learning_rates):
        #initialize the network
        net = Network(28*28, 10)

        print("start run {0} of {1}".format(i*len(batch_sizes)+j, len(batch_sizes) * len(learning_rates)))
        print("\tnetwork initialized - test score: {0}".format(get_test_loss()))

        #store the development of the loss as a function of the epoch here
        epoch_train_trajectory = [get_train_loss()]
        epoch_test_trajectory = [get_test_loss()]

        start_time = time.time()

        for epoch in range(epochs):
            #shuffle data randomly
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]

            #loop through data set
            for batch in range(n_batches):
                #perform SGD
                net.fit(x_train[batch*batch_size:(batch+1)*batch_size],
                        y_train[batch*batch_size:(batch+1)*batch_size],
                        eta=learning_rate)

            #store the loss
            epoch_train_trajectory.append(get_train_loss())
            epoch_test_trajectory.append(get_test_loss())
            print("\tepoch {0} finished - test score: {1}".format(epoch, epoch_test_trajectory[-1]))

        results.append((batch_size, learning_rate, time.time() - start_time, epoch_train_trajectory, epoch_test_trajectory))

#save the results to analyze them later
with open("result.pkl", "wb") as f:
    pickle.dump(results, f)