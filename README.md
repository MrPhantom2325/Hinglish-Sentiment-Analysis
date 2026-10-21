# Hinglish-Sentiment-Analysis

3-class sentiment classification (negative / neutral / positive) of code-mixed Hindi-English (Hinglish)
tweets from SemEval-2020 Task 9 (SentiMix). Compares a TF-IDF + LinearSVC baseline against
mBERT, MuRIL and HingBERT, and analyses *why* they differ (tokenizer fragmentation of romanized
Hindi, performance vs amount of code-mixing).

> Status: work in progress. Sections marked TODO are filled in as milestones are completed.

## 1. Environment setup (Mac, Apple Silicon)

Use a native arm64 Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -c "import platform; print(platform.machine())"   # must print arm64
pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"  # should print True
```

## 2. Data

The raw SentiMix files go in `data/raw/` (never committed). To download and verify them:

```bash
python scripts/download_data.py
```

Provenance (URLs, file names, sha256, verification counts) is recorded in `data/raw/SOURCE.md`.

## 3. Pipeline

TODO: exact command sequence (Mac milestones 1–3 → Drive → Colab → Mac milestones 6–7).

## 4. Results

TODO: generated from `results/tables/main_results.md`.

## Repository layout

```
configs/default.yaml   paths, label map, hyperparameters, seeds
data/raw/              original SentiMix files (gitignored)
data/processed/        train/dev/test CSV + JSONL (gitignored)
src/                   parsing, preprocessing, models, evaluation, analysis
scripts/               data download, Colab packaging
notebooks/             Colab training notebook
tests/                 pytest suite with a synthetic CoNLL fixture
results/               preds/ (gitignored), metrics/, figures/, tables/
```
