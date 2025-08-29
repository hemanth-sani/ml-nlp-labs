from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def calculate_error(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * (y_true != y_pred)) / np.sum(weights))

def calculate_alpha(error: float) -> float:
    eps = 1e-12
    error = np.clip(error, eps, 1 - eps)
    return 0.5 * np.log((1.0 - error) / error)

def update_weights(weights: np.ndarray, alpha: float, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    w = weights * np.exp(-alpha * y_true * y_pred)
    return w / np.sum(w)

def adaboost(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, max_depth: int = 1, seed: int = 42):
    n = X.shape[0]
    weights = np.ones(n) / n
    classifiers, alphas = [], []
    rng = np.random.RandomState(seed)
    for _ in range(n_estimators):
        stump = DecisionTreeClassifier(max_depth=max_depth, random_state=rng.randint(0, 1_000_000))
        stump.fit(X, y, sample_weight=weights)
        pred = stump.predict(X)
        err = calculate_error(y, pred, weights)
        alpha = calculate_alpha(err)
        weights = update_weights(weights, alpha, y, pred)
        classifiers.append(stump)
        alphas.append(alpha)
    return classifiers, np.array(alphas, dtype=float)

def predict_ensemble(classifiers, alphas, X: np.ndarray) -> np.ndarray:
    scores = np.sum([a * clf.predict(X) for clf, a in zip(classifiers, alphas)], axis=0)
    return np.sign(scores).astype(int)

def evaluate_curve(classifiers, alphas, X: np.ndarray, y: np.ndarray):
    errs = []
    for t in range(1, len(classifiers) + 1):
        pred = predict_ensemble(classifiers[:t], alphas[:t], X)
        errs.append(float(np.mean(pred != y)))
    return np.array(errs, dtype=float)
