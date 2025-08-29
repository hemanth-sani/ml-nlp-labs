from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path

def plot_error_curve(train_err, test_err, title: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_err)+1), train_err, label="Train Error")
    plt.plot(range(1, len(test_err)+1), test_err, label="Test Error")
    plt.title(title)
    plt.xlabel("Number of Weak Classifiers")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
