"""Tests for src/train.py helpers (no model download needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from src.train import KeepBestInMemory, TweetDataset, build_parser, metrics_from_logits, stratified_subsample


def test_stratified_subsample_keeps_proportions_and_is_deterministic():
    df = pd.DataFrame({"label": ["negative"] * 300 + ["neutral"] * 400 + ["positive"] * 300})
    a = stratified_subsample(df, 0.1, seed=0)
    b = stratified_subsample(df, 0.1, seed=0)
    assert len(a) == 100
    assert a["label"].value_counts().to_dict() == {"neutral": 40, "negative": 30, "positive": 30}
    assert a.index.equals(b.index)
    assert stratified_subsample(df, 1.0, seed=0) is df


def test_metrics_from_logits():
    logits = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0], [0, 3.0, 0]])
    m = metrics_from_logits((logits, np.array([0, 1, 2, 2])))
    assert m["accuracy"] == pytest.approx(0.75)


def test_tweet_dataset_items():
    def fake_tokenizer(texts, truncation, max_length):
        return {"input_ids": [[1, 2, 3][:max_length] for _ in texts], "attention_mask": [[1, 1, 1] for _ in texts]}

    ds = TweetDataset(["a b", "c"], [0, 2], fake_tokenizer, max_length=2)
    assert len(ds) == 2
    assert ds[1] == {"input_ids": [1, 2], "attention_mask": [1, 1, 1], "labels": 2}


def test_parser_defaults():
    args = build_parser().parse_args(["--model", "mbert"])
    assert (args.lr, args.seed, args.train_fraction, args.max_steps) == (2e-5, 42, 1.0, -1)


def test_keep_best_in_memory_restores_best_weights():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=30, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=16, num_labels=3)
    model = BertForSequenceClassification(cfg)
    cb = KeepBestInMemory()
    state = SimpleNamespace(epoch=1.0, global_step=10)

    cb.on_evaluate(None, state, None, metrics={"eval_macro_f1": 0.6}, model=model)
    best = {k: v.clone() for k, v in model.state_dict().items()}

    with torch.no_grad():  # "epoch 2" changes every weight, including LayerNorms, and is worse
        for p in model.parameters():
            p.add_(1.0)
    cb.on_evaluate(None, SimpleNamespace(epoch=2.0, global_step=20), None,
                   metrics={"eval_macro_f1": 0.5}, model=model)
    assert cb.best_epoch == 1.0 and cb.best_score == 0.6

    cb.restore(model)
    for k, v in model.state_dict().items():
        assert torch.equal(v, best[k]), k
