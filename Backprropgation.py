import numpy as np

# Data
np.random.seed(42)
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)

# Normalization
X = X / np.max(X, axis=0)
y = y / 100

# Activation + derivative
sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)

# Network structure
lr = 0.1
epochs = 7000

w1 = np.random.rand(2, 3)
b1 = np.random.rand(1, 3)
w2 = np.random.rand(3, 1)
b2 = np.random.rand(1, 1)

# Training
for _ in range(epochs):
    # Forward
    h = sigmoid(X @ w1 + b1)
    out = sigmoid(h @ w2 + b2)

    # Backprop
    error = y - out
    d_out = error * dsigmoid(out)
    d_h = d_out @ w2.T * dsigmoid(h)

    w2 += h.T @ d_out * lr
    b2 += np.sum(d_out, axis=0, keepdims=True) * lr
    w1 += X.T @ d_h * lr
    b1 += np.sum(d_h, axis=0, keepdims=True) * lr

# Results
print("Input:\n", X)
print("\nActual Output:\n", y)
print("\nPredicted Output:\n", out)
