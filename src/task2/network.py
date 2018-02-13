from neuron import Neuron
import numpy as np

class Network(object):
    def __init__(self, n_input, n_output):
        self._n_output = n_output
        self._n_input = n_input
        
        self._neurons = [Neuron(n_input) for _ in range(n_output)]

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        predictions = [neuron.forward(x) for neuron in self._neurons]

        return np.array(predictions)[:, :, 0].T

    def fit(self, x, y, eta=1e-3):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        predictions = self.predict(x)

        diff = (predictions - y) / len(x)

        for i, neuron in enumerate(self._neurons):
            gradient = np.dot(diff[:, i], neuron.backward(x))
            
            #average gradient now
            #gradient = np.mean(gradient, axis=0, keepdims=True).T
            gradient = gradient.reshape((-1, 1))
            neuron.update_weight(-eta * gradient)