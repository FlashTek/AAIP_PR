from layer import Layer
from network import Network
import autograd.numpy as np

net = Network()#loss=lambda o, y: np.mean(-(y*np.log(o) + (1-y) * np.log(1-o))))
layer1 = Layer(2, 4)
layer2 = Layer(4, 2)
layer3 = Layer(2, 1)

net.add(layer1)
net.add(layer2)
net.add(layer3)

#print(layer3._d_activation(np.dot(x2, layer3._W)))
#print("yap")

#print(y)
#print(data)

#layer1.backward(x)

eta = 0.001
steps = 4000

X = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]])
Y = np.asarray([[0], [0], [1], [1]])

for n in range(steps):
    for i in range(4):
        x = X#[i]
        y = Y#[i]

        #x = np.array([[0, 1]])#X[i]
        #y = np.array([[1.0]])#Y[:, i]

        prediction, forwardpass_result = net.forward(x)

        #mse = np.mean((y-prediction)**2)
        #print(mse)

        print(prediction)

        gradients = net.backward(x, y, forwardpass_result)

        for layer in net._layers:
            #print(gradients[layer])
            #print(layer._W)
            #print(gradients[layer].shape)
            #print(layer._W.shape)
            layer._W -= eta * gradients[layer]