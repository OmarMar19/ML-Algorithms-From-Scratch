import numpy as np

def train_perceptron(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    # Convert y to {-1, 1}
    y_bin = np.where(y == 0, -1, 1)

    for _ in range(epochs):
        for idx, x in enumerate(X):
            if y_bin[idx] * (np.dot(x, w) + b) <= 0:
                w += lr * y_bin[idx] * x
                b += lr * y_bin[idx]
    return w, b

def predict_perceptron(X, w, b):
    return np.where(np.dot(X, w) + b >= 0, 1, 0)
