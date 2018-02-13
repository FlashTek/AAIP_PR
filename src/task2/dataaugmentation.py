from network import Network
import utility as ut
import numpy as np
import pickle
from matplotlib import pyplot as plt

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
task1dir = os.path.dirname(currentdir) + "/task1/"
sys.path.insert(0, task1dir)

import morphological_operations as mo
import structuring_elements as se 

def apply_noise(x, morph=False):
    #performs data augmentation

    #apply noise
    noise = np.random.randn(*x.shape)

    x = np.clip(x + noise, 0.0, 1.0)

    if morph:
        #apply morphologcical operation, aka closing
        SE = se.square(3)
        for i in range(len(x)):
            x[i] = mo.closing(x[i], SE)

    return x.reshape((-1, 28*28))

(x_train, y_train), (x_test, y_test) = ut.load_data()

a = apply_noise(x_train, morph=True)[0]
plt.imshow(a)
plt.show()

def perform_train_evaluation(x_train, y_train, x_test, y_test):
    #define hypterparameters which will be used
    #number of epochs to train the model
    epochs = 30

    #batch size for the SGD
    batch_size = 10
    #learning rate for the update step
    learning_rate = 0.001

    #number of batches in each epoch
    n_batches = int(len(x_train) / batch_size)

    #initialize the network
    net = Network(28*28, 10)

    def get_test_score():
        return ut.get_test_score(net, x_test, y_test)

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

    return epoch_trajectory

#train and test without noise
no_noise_no_noise_result = perform_train_evaluation(x_train, y_train, x_test, y_test)

#train without noise and test noise
no_noise_noise_result = perform_train_evaluation(x_train, y_train, x_test, y_test)

#train without noise and test noise
no_noise_noise_morph_result = perform_train_evaluation(x_train, y_train, apply_noise(x_test, True), y_test)

#train with noise and test noise
noise_noise_result = perform_train_evaluation(apply_noise(x_train), y_train, x_test, y_test)

#train with noise and morphology and test with noise and morphology
noise_morph_noise_morph_result = perform_train_evaluation(apply_noise(x_train, True), y_train, apply_noise(x_test, True), y_test)

plt.plot(no_noise_no_noise_result, label="Case 1")
plt.plot(no_noise_noise_result, label="Case 2")
plt.plot(no_noise_noise_morph_result, label="Case 3")
plt.plot(noise_noise_result, label="Case 4")
plt.plot(noise_morph_noise_morph_result, label="Case 5")

total_data = (no_noise_no_noise_result, no_noise_noise_result, no_noise_noise_morph_result, noise_noise_result, noise_morph_noise_morph_result)

#save the results to analyze them later
with open("total_data.pkl", "wb") as f:
    pickle.dump(total_data, f)

plt.show()