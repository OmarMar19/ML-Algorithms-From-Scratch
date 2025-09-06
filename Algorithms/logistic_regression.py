import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    for _ in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
        db = (1/n_samples) * np.sum(y_hat - y)
        w -= lr * dw
        b -= lr * db
    return w, b

def predict_logistic_regression(X, w, b, threshold=0.5):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= threshold).astype(int)

def train_logistic_regression_ovr(X, y, lr=0.1, epochs=1000):
    classes = np.unique(y)
    w_list, b_list = [], []
    for c in classes:
        y_binary = (y == c).astype(int)
        w, b = train_logistic_regression(X, y_binary, lr, epochs)
        w_list.append(w)
        b_list.append(b)
    return np.array(w_list), np.array(b_list), classes

def predict_logistic_regression_ovr(X, w_list, b_list, classes):
    probs = np.array([sigmoid(np.dot(X, w) + b) for w, b in zip(w_list, b_list)])
    return classes[np.argmax(probs, axis=0)]
