import autograd.numpy as np
from autograd import grad, elementwise_grad 

class Network(object):
    def __init__(self, loss=lambda o,y: np.mean((-o+y)**2)):
        """

        """

        self._layers = []
        self._loss = loss
        self._d_loss = lambda o, y: 2.0*(y-o) #grad(self._loss)

    def add(self, layer):
        self._layers.append(layer)

    def forward(self, input):
        """

        """
        #pass output of layer i as value of layer i+1
        nl_value = input
        data = {}, {}
        for layer in self._layers:
            nl_value, value = layer.forward(nl_value)
            data[1][layer] = value
            data[0][layer] = nl_value

        return value, data

    def backward(self, x, y, forwardpass):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        output_values, linear_values = forwardpass

        #dictionary to store the layer and the weight gradients
        delta = {}

        b = self._layers[-1]._d_activation(linear_values[self._layers[-1]])
        a = self._d_loss(output_values[self._layers[-1]], y)

        delta[self._layers[-1]] = np.multiply(a, b)


        for i in range(len(self._layers)-2, -1, -1):
            next_layer = self._layers[i+1]
            layer = self._layers[i]

            next_delta = delta[next_layer]

            input_value = None
            if i > 0:
                input_value = output_values[self._layers[i-1]]
            else:
                input_value = x

            delta[self._layers[i]] = layer.backward(input_value, next_delta, self._layers[i+1]._W)

        

        def addbias(val):
            bias_dummy = np.repeat(0.0, val.shape[0]).reshape((-1, 1))
            return np.hstack((bias_dummy, val))

        extended_x = addbias(x)

        gradients = {}
        gradients[self._layers[0]] = np.dot(extended_x.T, delta[self._layers[0]])
        for i in range(1, len(self._layers)):
            gradients[self._layers[i]] = np.dot(addbias(output_values[self._layers[i-1]]).T, delta[self._layers[i]]) 

        return gradients
