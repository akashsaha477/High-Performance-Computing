import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20, w_init=None, b_init=None):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = np.array(w_init, dtype=float)
        self.b = float(b_init)

    def step(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        return self.step(z)

    def train(self, X, T):
        for epoch in range(self.epochs):
            for x, t in zip(X, T):
                y = self.predict(x)
                error = t - y
                self.w += self.lr * error * x
                self.b += self.lr * error



X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

T_AND = np.array([0, 0, 0, 1])


p1 = Perceptron(learning_rate=0.1, epochs=25,
                w_init=[0.2, -0.1], b_init=0.0)

p1.train(X, T_AND)
print("AND Gate Solution 1:")
print("Weights:", p1.w, "Bias:", p1.b)


p2 = Perceptron(learning_rate=0.1, epochs=25,
                w_init=[-0.5, 0.7], b_init=-0.2)

p2.train(X, T_AND)
print("AND Gate Solution 2:")
print("Weights:", p2.w, "Bias:", p2.b)


T_OR = np.array([0, 1, 1, 1])


p3 = Perceptron(learning_rate=0.1, epochs=25,
                w_init=[0.1, 0.1], b_init=-0.1)

p3.train(X, T_OR)
print("OR Gate Solution 1:")
print("Weights:", p3.w, "Bias:", p3.b)


p4 = Perceptron(learning_rate=0.1, epochs=25,
                w_init=[-0.3, 0.8], b_init=0.2)

p4.train(X, T_OR)
print("OR Gate Solution 2:")
print("Weights:", p4.w, "Bias:", p4.b)


print("\nTesting AND Gate:")
for x in X:
    print(x, "->", p1.predict(x))

print("\nTesting OR Gate:")
for x in X:
    print(x, "->", p3.predict(x))


p1 = Perceptron(learning_rate=0.1, epochs=20,
                w_init=[0.5], b_init=-0.1)

p1.train(X_NOT, T_NOT)

print("NOT Gate Solution 1")
print("Weight:", p1.w, "Bias:", p1.b)

p2 = Perceptron(learning_rate=0.1, epochs=20,
                w_init=[-0.8], b_init=0.3)

p2.train(X_NOT, T_NOT)

print("NOT Gate Solution 2")
print("Weight:", p2.w, "Bias:", p2.b)


print("\nTesting NOT Gate (Solution 1):")
for x in X_NOT:
    print(x[0], "->", p1.predict(x))

print("\nTesting NOT Gate (Solution 2):")
for x in X_NOT:
    print(x[0], "->", p2.predict(x))