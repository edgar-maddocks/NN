import numpy as np
import time
import numba
import os

from layers import Layer
from losses import Loss


class Sequential:
    def __init__(self, layers: list[Layer], loss: Loss, lr=0.001):
        self.layers = layers
        self.loss = loss
        self.lr = lr

    def predict(self, x: np.array):
        result = x.T
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def fit(
        self,
        x_train,
        y_train,
        epochs=100,
        gd="batch",
        batch_size=0,
        verbose=0,
    ):
        start_timer = time.time()
        y_train_t = y_train.T
        for epoch in range(0, epochs):
            loss = 0
            if gd == "batch":
                y_hat = self.predict(x_train)
                output_error_grad = self.loss.d_loss(y_train_t, y_hat)
                loss = self.loss.loss(y_train_t, y_hat)
            elif gd == "stoch":
                if batch_size < 1:
                    raise ValueError("batch_size must be > 0")
                indices = np.random.permutation(x_train.shape[0])
                x_shuffled = x_train[indices]
                y_shuffled = y_train[indices]

                for i in range(0, x_train.shape[0], batch_size):
                    x_batch = x_shuffled[i : i + batch_size]
                    y_batch = y_shuffled[i : i + batch_size]
                    y_batch_t = y_batch.T
                    y_hat = self.predict(x_batch)
                    output_error_grad = self.loss.d_loss(y_batch_t, y_hat)
                    loss += self.loss.loss(y_batch_t, y_hat)
            for layer in reversed(self.layers):
                output_error_grad = layer.backwards(output_error_grad, self.lr)
            if verbose:
                print(
                    f"Current Epoch: {epoch}     Training Loss: {round(loss, 9)}     Elapsed: {round(time.time() - start_timer, 3)}s",
                    end="\r",
                )
        os.system("cls" if os.name == "nt" else "clear")
        print(
            "######################### TRAINING COMPLETE ###############################"
        )
