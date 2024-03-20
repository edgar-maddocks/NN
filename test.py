import numpy as np
import time

from layers import Dense, Tanh
from losses import MSE
from models import Sequential

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential([Dense(2, 3), Tanh(), Dense(3, 1), Tanh()], loss=MSE(), lr=0.1)

start = time.time()
model.fit(x, y, 500, gd="batch", verbose=1)
end = time.time()

print(model.predict(x))
print(f"Model fitted in {end-start}s")
