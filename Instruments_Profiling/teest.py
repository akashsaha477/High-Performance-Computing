# test_compute.py

import numpy as np

print("START")

A = np.random.rand(2000, 2000)
B = np.random.rand(2000, 2000)

for i in range(10):
    print("Iteration", i)
    C = A @ B   # heavy matrix multiplication

print("DONE")