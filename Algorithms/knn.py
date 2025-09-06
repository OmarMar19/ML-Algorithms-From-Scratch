import numpy as np
from collections import Counter

def predict_knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)
