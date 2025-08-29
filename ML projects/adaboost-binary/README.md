# 🧠 MiniBERT — From‑Scratch BERT Encoder & Sentence Classifier (PyTorch)

I implemented a minimalist version of **BERT** in **pure PyTorch** (no `transformers`), then used it to:
- **Pretrain** with **MLM** (masked language modeling)
- **Finetune** for **sentence classification** on **SST** and **CFIMDB**

The repo includes a clean CLI, reproducible runs, and prediction files in the format reviewers expect.

---

## ✨ What I built
- **MiniBERT encoder**: token + positional embeddings → multi‑head self‑attention → FFN with residuals + LayerNorm (stacked).
- **Heads**:
  - **MLM head** for masked token prediction (80/10/10 rule).
  - **CLS head** for sentence classification (linear layer over `[CLS]`).
- **Training**:
  - **Pretraining** (MLM) on raw text.
  - **Finetuning** (end‑to‑end) on labeled datasets.
  - Optionally **freeze** the encoder after pretraining and train a small CLS head to quickly produce classification outputs.
- **Tokenizer**: whitespace + punctuation splitter with `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`.
- **Reproducibility**: fixed seeds, gradient clipping, warmup + linear decay, consistent output filenames.

---

## 🗂 Project layout
```
NLP projects/Minibert/
├─ README.md
├─ requirements.txt
├─ data/                 # put sst-*.txt and cfimdb-*.txt here (not committed)
├─ results/              # predictions written here (dev/test text files)
├─ base_bert.py          # embeddings, MHA, FFN, transformer block
├─ bert.py               # MiniBERT encoder + heads
├─ tokenizer.py          # tiny vocab + encode + MLM masking
├─ optimizer.py          # AdamW + warmup/linear scheduler
├─ utils.py              # datasets, loaders, helpers, metrics
├─ classifier.py         # CLI: --option {pretrain, finetune}
└─ sanity_check.py       # quick shape test for the encoder
```

---

## 🔧 Environment

**Windows PowerShell**
```powershell
cd "NLP projects\Minibert"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PyTorch fails to install on CPU:  
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 📚 Data format

Place files in `data/`. Each line contains a **label** and **text** separated by a tab:

```
<label>\t<text>
```

- **SST**: multi‑class sentence sentiment  
- **CFIMDB**: binary sentiment  

For **pretraining**, the same files are used; only the text part (after the first tab) is used for MLM.

---

## ▶️ Quickstart

### Pretrain (SST) — writes prediction files
```powershell
python classifier.py --option pretrain --epochs 1 --lr 5e-4 `
  --train "data\sst-train.txt" --dev "data\sst-dev.txt" --test "data\sst-test.txt" `
  --dev_out  "results\sst-dev-output.pretrain.txt" `
  --test_out "results\sst-test-output.pretrain.txt"
```

### Finetune (SST) end‑to‑end — writes prediction files
```powershell
python classifier.py --option finetune --epochs 3 --lr 2e-4 `
  --train "data\sst-train.txt" --dev "data\sst-dev.txt" --test "data\sst-test.txt" `
  --dev_out  "results\sst-dev-output.finetune.txt" `
  --test_out "results\sst-test-output.finetune.txt"
```

**CFIMDB**: swap the file paths and use these output names:
- `results\cfimdb-dev-output.pretrain.txt`, `results\cfimdb-test-output.pretrain.txt`
- `results\cfimdb-dev-output.finetune.txt`,  `results\cfimdb-test-output.finetune.txt`

Each output file contains **one predicted label per line**.

---

## ✅ Sanity check
```powershell
python sanity_check.py
```
Expected: `OK: encoder output shape torch.Size([2, 16, 128])`.

---

## 📊 Results (GPU runs on hopper cluster)

> Numbers vary slightly by seed; these are representative from my runs.

| Dataset | Mode      | LR   | Epochs | Batch | Dropout | Dev Acc | Test Acc |
|:------:|:---------:|:----:|:------:|:-----:|:-------:|:-------:|:--------:|
| CFIMDB | Pretrain  | 0.003|   4    |  32   |  0.10   |  0.743  |  0.580   |
| CFIMDB | Finetune  | 5e-05|   5    |  32   |  0.09   |  0.971  |  0.508   |
|  SST   | Pretrain  | 0.003|   4    |  32   |  0.30   |  0.395  |  0.419   |
|  SST   | Finetune  | 2e-05|   5    |  32   |  0.10   |  0.517  |  0.537   |

Additional single‑run references I observed:
- **SST pretrain:** Dev ≈ 0.381 / Test ≈ 0.409  
- **SST finetune:** Dev ≈ 0.515 / Test ≈ 0.533  
- **CFIMDB pretrain:** Dev ≈ 0.784  
- **CFIMDB finetune:** Dev ≈ 0.901  

These are in the ballpark of standard MiniBERT baselines.

---

## 🧪 Reproduce & log
- Fix seeds (e.g., `--seed 42`) for strict comparability.  
- Save predictions in `results/` using the names above.  
- Keep datasets and large artifacts out of git.

---

## 🛠 Design choices
- **Pure PyTorch** (readable, no external model libs)  
- **Simple tokenizer** (whitespace + punctuation)  
- **MLM masking** (80% `[MASK]`, 10% random token, 10% keep)  
- **Warmup + linear decay** for stable optimization  
- **Freeze‑then‑CLS‑head** option after pretraining to quickly generate predictions

---

## 📌 Roadmap / stretch ideas
- Continued pretraining (domain adaptation)  
- Layer‑wise LR decay / adapters for finetuning stability  
- Better tokenizer (BPE/WordPiece) + vocab export  
- Logging (TensorBoard) and checkpoints  
- Ablations: heads/layers/hidden size vs. accuracy & speed

---


