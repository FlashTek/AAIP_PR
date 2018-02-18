import numpy as np
import urllib.request
import os.path

def load_data(reshape=False, one_hot_encoding=True):
    """
        Loads the MNIST data set using the keras library, normalizes the images from [0,255]->[0,1].
        The labels can be one hot encoded.

        parameters:
            one_hot - bool, use one hot encoding

        returns:
            ((array, array), (array, array)) - ((train input, train output), (test input, test output))
    """

    #check if dataset exists
    if not os.path.isfile("dataset.npz"):
        #if not, download the data set
        urllib.request.urlretrieve("https://s3.amazonaws.com/img-datasets/mnist.npz", "dataset.npz")
    #load the data set
    files = np.load("dataset.npz")

    x_train, y_train, x_test, y_test = [files[x] for x in ["x_train", "y_train", "x_test", "y_test"]] 

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if reshape:
        x_train = x_train.reshape((-1, 28*28))
        x_test = x_test.reshape((-1, 28*28))

    if one_hot_encoding:
        y_train = one_hot(y_train, 10)
        y_test = one_hot(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def one_hot(x, n_classes):
    """
        Performs a one hot encoding of the input.

    parameters:
        x - array, array of labels which shall be encoded

    returns:
        array, one hot encoded input array
    """

    x_oh = np.zeros((len(x), n_classes))
    x_oh[np.arange(len(x)), x] = 1.0

    return x_oh

def get_network_loss(net, x_test, y_test):
    """"
        Calculates the loss of the network on the test data set.

        parameters:
            net - Network, network to be evaluated
            x_test - array, input data
            y_test - array, one hot encoded output data

        returns:
            float - network's loss
    """

    #calculate the prediction
    prediction = net.predict(x_test)

    #set the maximum of the prediction as the real one hot encoded prediction
    pred = np.zeros_like(prediction)
    pred[np.arange(len(pred)), np.argmax(prediction, axis=1)] = 1.0
    prediction = pred

    #calculate the number of wrong classified rows
    return np.mean((y_test - prediction)**2)*y_test.shape[1]/2.0