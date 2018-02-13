import autograd.numpy as np
import autograd.numpy.random as rnd
from autograd import grad, elementwise_grad 

class Layer(object):
    def __init__(self, n_input, n_hidden, activation=lambda x: x):
        """

        """
        self._activation = activation
        self._d_activation = elementwise_grad (activation)

        self._W = rnd.normal(0.0, 2.0/n_input, size=(n_input+1, n_hidden))
        self._n_input = n_input
        self._n_hidden = n_hidden


    def forward(self, input):
        """
            Performs a forward pass of this layer, i.e. calculates the layer's output for the given input.
        """

        if len(input.shape) == 1:
            input = input.reshape((1, -1))

        if self._n_input != input.shape[1]:
            raise ValueError("The input does not match the layer's weight. The shapes do not match: {0} != {1}".format(self._W.shape, input.shape))

        #add a plain 1.0 as a first entry of the vector for the bias
        bias_dummy = np.repeat(0.0, input.shape[0]).reshape((-1, 1))
        extended_input = np.hstack((bias_dummy, input))

        #calculate the linear value
        value = np.dot(extended_input, self._W)

        #calculate the non-linear value
        nl_value = self._activation(value)

        #return the non-linear value as the output
        return nl_value, value

    def backward(self, input, gradient, prev_weight):
        """
            Performs a backward pass of this layer, i.e. calculates the layer's gradient.
        """
        bias_dummy = np.repeat(0.0, input.shape[0]).reshape((-1, 1))
        extended_input = np.hstack((bias_dummy, input))

        #do not calculate delta for bias values
        #according to http://briandolhansky.com/blog/2014/10/30/artificial-neural-networks-matrix-form-part-5
        W_nobias = prev_weight[0:-1, :]

        a = np.dot(gradient, W_nobias.T)
        #a = np.dot(gradient, prev_weight.T)
        #print(a.shape)
        b = np.dot(extended_input, self._W)
        b = self._d_activation(b)
        #print(b.shape)


        c = np.multiply(a, b)
        #print(c.shape)

        return c