import numpy as np

class Neuron(object):
    def __init__(self, n_input):
        self._W = np.random.normal(0.0, 2.0/n_input, size=(n_input+1, 1))

    def forward(self, x):
        x = self._expand_input(x)

        return 1.0 / (1.0 + np.exp(-np.dot(x, self._W)))

    def backward(self, x):
        x = self._expand_input(x)

        return np.exp(-np.dot(x, self._W)) * x /(1+np.exp(-np.dot(x, self._W)))**2 

    def update_weight(self, gradient):
        if gradient.shape != self._W.shape:
            raise ValueError("Dimensions {0} and {1} do not match".format(gradient.shape, self._W.shape))

        self._W += gradient

    def _expand_input(self, x):
        extension = np.repeat(1.0, x.shape[0]).reshape((-1, 1))
        return np.hstack((extension, x))
