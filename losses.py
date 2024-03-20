import numpy as np


class Loss:
    def __init__(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss


class MSE(Loss):
    def __init__(self):
        mse = lambda y, y_hat: np.mean(np.power(y - y_hat, 2))
        d_mse = lambda y, y_hat: 2 * (y_hat - y) / np.size(y)
        super().__init__(mse, d_mse)
