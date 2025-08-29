# 🧠 MiniBERT — From-Scratch BERT Encoder & Sentence Classifier (PyTorch)

I implemented a minimalist version of **BERT** in **pure PyTorch** (no `transformers`), then used it to:
- **Pretrain** with **MLM** (masked language modeling)
- **Finetune** for **sentence classification** on **SST** and **CFIMDB**

The repo includes clean CLI scripts, reproducible runs, and prediction files in the exact format graders/reviewers expect.

---

## ✨ What I built

- **MiniBERT encoder**: token + positional embeddings → multi-head self-attention → FFN with residuals + LayerNorm (stacked L×).
- **Heads**:
  - **MLM head** for masked token prediction (with 80/10/10 masking rule).
  - **CLS head** for sentence classification (linear layer on top of `[CLS]`).
- **Training loops**:
  - **Pretraining** (MLM) on raw text.
  - **Finetuning** (end-to-end) on labeled datasets.
  - After pretraining, I optionally **freeze** the encoder and train a small CLS head to quickly produce classification outputs.
- **Tokenizer**: simple whitespace+punctuation splitter, with `[PAD]/[UNK]/[CLS]/[SEP]/[MASK]`.
- **Reproducibility**: fixed seeds, gradient clipping, warmup + linear decay scheduler, predictable output file names.

---

## 🗂 Project layout
```
NLP projects/Minibert/
├─ README.md
├─ requirements.txt
├─ data/ # put sst-.txt and cfimdb-.txt here (not committed)
├─ results/ # predictions written here (dev/test text files)
├─ base_bert.py # embeddings, MHA, FFN, transformer block
├─ bert.py # MiniBERT encoder + heads
├─ tokenizer.py # tiny vocab + encode + MLM masking
├─ optimizer.py # AdamW + warmup/linear scheduler
├─ utils.py # datasets, loaders, helpers, metrics
├─ classifier.py # CLI: --option {pretrain,finetune}
└─ sanity_check.py # quick shape test for the encoder
```
---

## 🔧 Environment

```powershell
# Windows PowerShell
cd "NLP projects\Minibert"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
📚 Data format

Put all files in data/. Each line has a label and text:

<label>\t<text>


SST: multi-class sentence sentiment.

CFIMDB: binary sentiment.

For pretraining, I reuse the same files; only the text portion (after the first tab) is used for MLM.

▶️ Quickstart (commands)
Pretrain (SST) → write prediction files
python classifier.py --option pretrain --epochs 1 --lr 5e-4 `
  --train "data\sst-train.txt" --dev "data\sst-dev.txt" --test "data\sst-test.txt" `
  --dev_out  "results\sst-dev-output.pretrain.txt" `
  --test_out "results\sst-test-output.pretrain.txt"

Finetune (SST) end-to-end → write prediction files
python classifier.py --option finetune --epochs 3 --lr 2e-4 `
  --train "data\sst-train.txt" --dev "data\sst-dev.txt" --test "data\sst-test.txt" `
  --dev_out  "results\sst-dev-output.finetune.txt" `
  --test_out "results\sst-test-output.finetune.txt"


Swap the paths/filenames for CFIMDB:

cfimdb-dev-output.pretrain.txt, cfimdb-test-output.pretrain.txt

cfimdb-dev-output.finetune.txt, cfimdb-test-output.finetune.txt

Each output file contains one predicted label per line.

✅ Sanity check
python sanity_check.py


Expected: OK: encoder output shape torch.Size([2, 16, 128]).

📊 Results (my GPU runs on campus cluster)

Numbers vary a bit by seed; these are representative from my runs.

Dataset	Mode	LR	Epochs	Batch	Dropout	Dev Acc	Test Acc
CFIMDB	Pretrain	0.003	4	32	0.10	0.743	0.580
CFIMDB	Finetune	5e-05	5	32	0.09	0.971	0.508
SST	Pretrain	0.003	4	32	0.30	0.395	0.419
SST	Finetune	2e-05	5	32	0.10	0.517	0.537

Additional single-run references I observed:

SST pretrain Dev ≈ 0.381 / Test ≈ 0.409

SST finetune Dev ≈ 0.515 / Test ≈ 0.533

CFIMDB pretrain Dev ≈ 0.784

CFIMDB finetune Dev ≈ 0.901

These are in the ballpark of standard MiniBERT baselines.

🧪 Reproduce & log

Fix seeds (--seed 42, if you want strict comparability).

Save your prediction files in results/ using the names above.

Keep datasets and large artifacts out of git.

🛠 Design choices

Pure PyTorch: easy to read and grade; avoids external model libs.

Simple tokenizer: whitespace+punct; works well enough for the task.

MLM masking: 80% [MASK], 10% random token, 10% keep.

Warmup + linear decay: stable optimization for both pretrain and finetune.

Freezing after pretrain (optional): train a small CLS head to quickly produce classification outputs.

📌 Roadmap / stretch ideas

Continued pretraining (domain adaptation on in-domain text).

Layer-wise LR decay / adapters for finetuning stability.

Better tokenizer (BPE/WordPiece) + vocab export.

Logging (tensorboard) and checkpoints for long runs.

Ablations: heads/layers/hidden size vs. accuracy and speed.