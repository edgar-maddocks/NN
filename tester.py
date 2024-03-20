import numba
import numpy as np
import time

from numpy.core.multiarray import array as array


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: np.array):
        raise NotImplementedError

    def backwards(self, output_error_grad: np.array):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_features, n_nodes):
        self.weights = np.random.randn(n_nodes, n_features)
        self.biases = np.zeros((n_nodes, 1))

    def forward(self, input: np.array):
        self.input = input
        self.output = np.dot(self.weights, input.T) + self.biases
        return self.output

    def backwards(self, output_error_grad: np.array, lr):
        weights_grad = np.dot(output_error_grad, self.input)
        input_grad = np.dot(self.weights.T, output_error_grad)

        self.weights -= lr * weights_grad
        self.biases -= lr * output_error_grad
        return input_grad


class Activation(Layer):
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, input: np.array):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backwards(self, output_error_grad: np.array):
        return np.multiply(output_error_grad, self.d_activation(self.input))


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        d_tanh = lambda x: 1 - (np.tanh(x) ** 2)
        super().__init__(tanh, d_tanh)


class Loss:
    def __init__(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss


class MSE(Loss):
    def __init__(self):
        mse = lambda y, y_hat: (1 / y.shape[0]) * np.dot((y - y_hat).T, (y - y_hat))
        d_mse = lambda y, y_hat, inputs: (-2 / y.shape[0]) * np.dot(inputs, (y - y_hat))
        super().__init__(mse, d_mse)


class Sequential:
    def __init__(self, layers: list[Layer], loss: Loss, lr=0.001):
        self.layers = layers
        self.loss = loss

    def predict(self, input):
        result = input.T
        for layer in self.layers:
            result = layer.forward(result)
        return result


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

model = Sequential([Dense(2, 3), Tanh(), Dense(3, 1)], loss=MSE())

print(model.predict(x))
