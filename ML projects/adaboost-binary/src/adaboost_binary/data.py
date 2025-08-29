from __future__ import annotations
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_txt(path: str) -> np.ndarray:
    
    return np.loadtxt(path)



def preprocess_labelled_matrix(m: np.ndarray, class1: int, class2: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isin(m[:, 0], [class1, class2])
    sub = m[mask]
    y = sub[:, 0].copy()
    X = sub[:, 1:]
    y[y == class1] = -1
    y[y == class2] = 1
    return X.astype(float), y.astype(int)

def load_digits_pair(class1: int, class2: int) -> Tuple[np.ndarray, np.ndarray]:
    d = load_digits()
    mask = np.isin(d.target, [class1, class2])
    X = d.data[mask].astype(float)
    y = d.target[mask].copy()
    y[y == class1] = -1
    y[y == class2] = 1
    return X, y

def maybe_standardize(X_train, X_test, enabled: bool = True):
    if not enabled:
        return X_train, X_test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if X_test is not None else None
    return X_train_s, X_test_s
