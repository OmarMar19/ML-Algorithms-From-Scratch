import numpy as np

def entropy(y):
    classes = np.unique(y)
    e = 0
    for c in classes:
        p = np.sum(y == c) / len(y)
        e -= p * np.log2(p) if p > 0 else 0
    return e

def best_split(X, y):
    n_samples, n_features = X.shape
    best_gain, split = -1, None
    parent_entropy = entropy(y)
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_idx = X[:, feature] <= t
            right_idx = X[:, feature] > t
            if sum(left_idx) == 0 or sum(right_idx) == 0:
                continue
            e_left = entropy(y[left_idx])
            e_right = entropy(y[right_idx])
            n_left, n_right = sum(left_idx), sum(right_idx)
            gain = parent_entropy - (n_left/n_samples)*e_left - (n_right/n_samples)*e_right
            if gain > best_gain:
                best_gain = gain
                split = {"feature": feature, "threshold": t,
                         "left_idx": left_idx, "right_idx": right_idx}
    return split

def build_tree(X, y, max_depth=3, depth=0):
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return {"value": np.bincount(y).argmax()}
    split = best_split(X, y)
    if split is None:
        return {"value": np.bincount(y).argmax()}
    left = build_tree(X[split["left_idx"]], y[split["left_idx"]], max_depth, depth+1)
    right = build_tree(X[split["right_idx"]], y[split["right_idx"]], max_depth, depth+1)
    return {"feature": split["feature"], "threshold": split["threshold"],
            "left": left, "right": right}

def predict_sample(tree, x):
    if "value" in tree:
        return tree["value"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_sample(tree["left"], x)
    else:
        return predict_sample(tree["right"], x)

def predict_tree(tree, X):
    return np.array([predict_sample(tree, x) for x in X])

def print_tree(node, depth=0):
    if "value" in node:
        print("  " * depth + f"Leaf: Class={node['value']}")
        return
    print("  " * depth + f"[X{node['feature']} <= {node['threshold']:.2f}]")
    print_tree(node['left'], depth+1)
    print_tree(node['right'], depth+1)
