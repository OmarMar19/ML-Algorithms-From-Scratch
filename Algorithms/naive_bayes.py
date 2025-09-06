import numpy as np

def train_naive_bayes(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    priors = {}
    likelihoods = {}

    for c in classes:
        X_c = X[y == c]
        priors[c] = len(X_c) / len(X)
        likelihoods[c] = {
            "mean": np.mean(X_c, axis=0),
            "var": np.var(X_c, axis=0) + 1e-6
        }
    return priors, likelihoods

def gaussian_pdf(x, mean, var):
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(-(x - mean) ** 2 / (2 * var))
    return coeff * exponent

def predict_naive_bayes(X, priors, likelihoods):
    y_pred = []
    for x in X:
        posteriors = {}
        for c in priors:
            prior = np.log(priors[c])
            class_likelihood = np.sum(np.log(gaussian_pdf(x, likelihoods[c]["mean"], likelihoods[c]["var"])))
            posteriors[c] = prior + class_likelihood
        y_pred.append(max(posteriors, key=posteriors.get))
    return np.array(y_pred)
