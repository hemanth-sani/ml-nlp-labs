# 🧪 ML & NLP Labs

A tidy collection of small, well-documented ML/NLP projects. Each project is self-contained with a README, reproducible scripts, and (when applicable) tests.

## 📚 Projects

### 🔶 Machine Learning
| Project | What it is | Tech | Links |
|---|---|---|---|
| 🍰 **AdaBoost Binary** | From-scratch AdaBoost (decision stumps) with train/test error curves (e.g., 1 vs 3, 3 vs 5). | Python, NumPy, scikit-learn, Matplotlib | [Folder](./ML%20projects/adaboost-binary) · [Quickstart](./ML%20projects/adaboost-binary#quickstart) |

### 🔷 NLP
| Project | What it is | Tech | Links |
|---|---|---|---|
| 🧠 **MiniBERT** | Minimal BERT in pure PyTorch (no transformers), with pretrain/finetune on SST & CFIMDB. | PyTorch | [Folder](./NLP%20projects/Minibert) · [Usage](./NLP%20projects/Minibert#usage-command-pattern) |

## How to run a project
See the project README inside each folder. Example:
```bash
cd "ML projects/adaboost-binary"
pip install -r requirements.txt
python train.py --class1 1 --class2 3 --n_estimators 100 --use-sklearn
