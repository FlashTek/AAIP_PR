from network import Network
import utility as ut
import numpy as np
import pickle

def get_test_score():
    """
        Evalutes the trained network net on the test data set

        returns:
            float - test score
    """

    return ut.get_test_score(net, x_test, y_test)

(x_train, y_train), (x_test, y_test) = ut.load_data()

#define hypterparameters which will be tested
#number of epochs to train the model
epochs = 30
#batch size for the SGD
batch_sizes = [1, 5, 10, 25, 50, 100, 200, 400, 600, 800, 1000]
#learning rate for the update step
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]

#storage for the results, which are tuples (batch size, learning rate, loss as a function of the epoch)
results = []

#loop over all hyperparameters
for batch_size in batch_sizes:
    #number of batches in each epoch
    n_batches = int(len(x_train) / batch_size)

    for learning_rate in learning_rates:
        #initialize the network
        net = Network(28*28, 10)

        print("network initialized - test score: {0}".format(get_test_score()))

        #store the development of the loss as a function of the epoch here
        epoch_trajectory = [get_test_score()]

        for epoch in range(epochs):
            #shuffle data randomly
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]

            #loop through data se
            for batch in range(n_batches):
                #perform SGD
                net.fit(x_train[batch*batch_size:(batch+1)*batch_size],
                        y_train[batch*batch_size:(batch+1)*batch_size],
                        eta=learning_rate)

            #store the loss
            epoch_trajectory.append(get_test_score())
            print("epoch {0} finished - test score: {1}".format(epoch, epoch_trajectory[-1]))

        results.append((batch_size, learning_rate, epoch_trajectory))

#save the results to analyze them later
with open("result.pkl", "wb") as f:
    pickle.dump(results, f)