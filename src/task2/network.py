from neuron import Neuron
import numpy as np

class Network(object):
    """
        A network consisting out of sigmoid neurons
    """

    def __init__(self, n_input, n_output):
        """
            Creates a new sigmoid network

            parameters:
                n_input - int, number of input variables
                n_output - int, number of output variables (classes)
        """

        self._n_output = n_output
        self._n_input = n_input
        
        #initializes the neurons for the output classes
        self._neurons = [Neuron(n_input) for _ in range(n_output)]

    def predict(self, x):
        """
            Makes a predicition y_hat(x) for the input x

            parameters:
                x - array, input data

            returns:
                array - prediction y_hat(x)
        """

        #reshape the input data
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        #obtain predictions for each of the neurons (classes)
        predictions = [neuron.forward(x) for neuron in self._neurons]

        #return the results as one big array
        return np.array(predictions)[:, :, 0].T

    def fit(self, x, y, eta=1e-3):
        """
            Trains the network by changing its weight according to the Stochastic Gradient Descent (SGD)

            parameters:
                x - array, input data of the shape (number of data, n_input)
                y - array, output classes (one hot encoded) of the shape (number of data, n_output)
        """

        #reshape the input
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        #obtain current predictions
        predictions = self.predict(x)

        #calculate the derivative of the MSE loss function
        diff = (predictions - y) / len(x)

        #perform a SGD step for each of the output neurons
        for i, neuron in enumerate(self._neurons):
            #obtain gradient wrt the weights by applying the chain rule
            gradient = np.dot(diff[:, i], neuron.backward(x))
            
            #reshape gradient
            gradient = gradient.reshape((-1, 1))

            #apply the weight change
            neuron.update_weight(-eta * gradient)