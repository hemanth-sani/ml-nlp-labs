from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from adaboost_binary.core import adaboost, evaluate_curve
from adaboost_binary.data import load_txt, preprocess_labelled_matrix, load_digits_pair, maybe_standardize
from adaboost_binary.plot import plot_error_curve

def parse_args():
    p = argparse.ArgumentParser(description="AdaBoost binary classifier")
    p.add_argument("--class1", type=int, required=True, help="Negative class label (-1)")
    p.add_argument("--class2", type=int, required=True, help="Positive class label (+1)")
    p.add_argument("--train", type=str, help="Path to train.txt (label in col 0)")
    p.add_argument("--test", type=str, help="Path to test.txt (label in col 0)")
    p.add_argument("--use-sklearn", action="store_true", help="Use sklearn digits dataset")
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--standardize", dest="standardize", action="store_true", default=True)
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    if args.use_sklearn:
        X_all, y_all = load_digits_pair(args.class1, args.class2)
        n = X_all.shape[0]
        idx = np.random.RandomState(args.seed).permutation(n)
        cut = int(0.7 * n)
        tr, te = idx[:cut], idx[cut:]
        X_train, y_train = X_all[tr], y_all[tr]
        X_test,  y_test  = X_all[te], y_all[te]
    else:
        if not (args.train and args.test):
            raise SystemExit("Provide --train and --test or use --use-sklearn")
        Xy_tr = load_txt(args.train)
        Xy_te = load_txt(args.test)
        X_train, y_train = preprocess_labelled_matrix(Xy_tr, args.class1, args.class2)
        X_test,  y_test  = preprocess_labelled_matrix(Xy_te, args.class1, args.class2)

    X_train, X_test = maybe_standardize(X_train, X_test, enabled=args.standardize)

    clfs, alphas = adaboost(X_train, y_train, n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed)
    train_err = evaluate_curve(clfs, alphas, X_train, y_train)
    test_err  = evaluate_curve(clfs, alphas, X_test,  y_test)

    outdir = Path(args.outdir or f"reports/{args.class1}vs{args.class2}")
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "class_pair": f"{args.class1}vs{args.class2}",
        "n_estimators": int(args.n_estimators),
        "final_train_error": float(train_err[-1]),
        "final_test_error": float(test_err[-1]),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_error_curve(train_err, test_err, f"{args.class1} vs {args.class2}", outdir / "error_curves.png")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
