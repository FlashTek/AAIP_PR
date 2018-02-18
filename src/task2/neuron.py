import numpy as np

class Neuron(object):
    """
        A sigmoid neuron
    """

    def __init__(self, n_input):
        """
            Initializes a new sigmoid neuron
            
            parameters:
                n_input - int, number of input variables
        """

        self._W = np.random.normal(0.0, 2.0/n_input, size=(n_input+1, 1))

    def forward(self, x):
        """
            Performs a foward pass, i.e. calculates the prediction/output y_hat(x)

            parameters:
                x - array, input data
        """

        x = self._expand_input(x)

        return 1.0 / (1.0 + np.exp(-np.dot(x, self._W)))

    def backward(self, x):
        """
            Calculates a backward pass, i.e. calculate the derivative of the acitvation function

            parameters:
                x - array, input data
        """

        x = self._expand_input(x)

        return np.exp(-np.dot(x, self._W)) * x /(1+np.exp(-np.dot(x, self._W)))**2 

    def update_weight(self, gradient):
        """
            Updates the weights of the neuron

            parameters:
                gradient - array, gradient of the weights which is added to the current weight
        """

        #check the shapes
        if gradient.shape != self._W.shape:
            raise ValueError("Dimensions {0} and {1} do not match".format(gradient.shape, self._W.shape))

        self._W += gradient

    def _expand_input(self, x):
        """
            Expands the input by appending a constant 1.0 at the end to take care of the bias in the neuron

            parameters:
                x - array, the input to be expaned
        """
        
        extension = np.repeat(1.0, x.shape[0]).reshape((-1, 1))
        return np.hstack((extension, x))
