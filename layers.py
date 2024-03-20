import numpy as np
import numba
from numpy.core.multiarray import array as array


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: np.array):
        raise NotImplementedError

    def backwards(self, output_error_grad: np.array, lr):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_features, n_nodes):
        self.weights = np.random.randn(n_nodes, n_features)
        self.biases = np.zeros((n_nodes, 1))

    def forward(self, input: np.array):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output

    def backwards(self, output_error_grad: np.array, lr):
        weights_grad = np.dot(output_error_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_error_grad)

        self.weights -= lr * weights_grad
        self.biases -= lr * np.mean(output_error_grad)
        return input_grad


class Activation(Layer):
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, input: np.array):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backwards(self, output_error_grad: np.array, lr):
        return np.multiply(output_error_grad, self.d_activation(self.input))


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        d_tanh = lambda x: 1 - (np.tanh(x) ** 2)
        super().__init__(tanh, d_tanh)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: x * (x > 0)
        d_relu = lambda x: 1.0 * (x > 0)
        super().__init__(relu, d_relu)


class SoftMax(Layer):
    def forward(self, input: np.array):
        e_x = np.exp(input)
        self.output = e_x / np.sum(e_x)
        return self.output

    def backwards(self, output_error_grad: np.array, lr):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_error_grad)
