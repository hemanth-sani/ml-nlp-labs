import numpy as np
from sklearn.datasets import make_classification
from adaboost_binary.core import adaboost, evaluate_curve

def test_adaboost_learns_on_easy_data():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=0, random_state=7)
    y[y == 0] = -1
    clfs, alphas = adaboost(X, y, n_estimators=8, max_depth=1, seed=0)
    train_err = evaluate_curve(clfs, alphas, X, y)
    assert train_err[-1] < 0.25
