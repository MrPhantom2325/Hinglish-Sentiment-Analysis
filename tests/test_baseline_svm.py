"""Tests for the SVM baseline and the shared output-file helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.baseline_svm import run_svm, tagged_text
from src.utils import PRED_COLUMNS, compute_metrics, load_config, predictions_frame, run_name

WORDS = {
    "negative": ["bahut", "bura", "worst", "bekar", "hate", "ganda"],
    "neutral": ["kal", "meeting", "office", "news", "aaj", "update"],
    "positive": ["accha", "best", "love", "pyaar", "great", "mast"],
}


def _toy_split(n_per_class: int, seed: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, words in WORDS.items():
        for i in range(n_per_class):
            text = " ".join(rng.choice(words, size=5))
            rows.append({"id": f"{prefix}{label[:3]}{i}", "text": text, "label": label})
    df = pd.DataFrame(rows)
    df["char_text"] = df["text"]
    df["word_text"] = df["text"]
    return df


@pytest.fixture
def toy():
    return {"train": _toy_split(20, 0, "tr"), "dev": _toy_split(6, 1, "dv"), "test": _toy_split(6, 2, "te")}


def test_run_name_and_metrics():
    assert run_name("svm", "C0.1", 42) == "svm_C0.1_42"
    assert run_name("mbert", "2e-05", 43, 0.25) == "mbert_2e-05_43_frac0.25"
    m = compute_metrics([0, 1, 2, 2], [0, 1, 2, 1])
    assert m["accuracy"] == pytest.approx(0.75)


def test_predictions_frame_format():
    df = predictions_frame(["a", "b"], "dev", [0, 2], np.array([[2.0, 0.1, 0.0], [0.0, 0.1, 3.0]]))
    assert list(df.columns) == PRED_COLUMNS
    assert df["pred"].tolist() == [0, 2]


def test_tagged_text():
    assert tagged_text(["Ladki", "bahuuut", "cute", "😂"], ["HIN", "HIN", "ENG", "EMT"]) == (
        "ladki_HIN bahuut_HIN cute_ENG face_with_tears_of_joy_EMT"
    )


def test_run_svm_writes_standard_files(toy, tmp_path):
    cfg = load_config()
    res = run_svm(toy, cfg, tmp_path, seeds=[42, 43])
    assert len(res) == 2
    preds = sorted((tmp_path / "preds").glob("preds_svm_C*_4*.csv"))
    runs = sorted((tmp_path / "metrics").glob("run_svm_C*_4*.json"))
    assert len(preds) == 2 and len(runs) == 2
    df = pd.read_csv(preds[0], dtype={"id": str})
    assert list(df.columns) == PRED_COLUMNS
    assert sorted(df["split"].unique()) == ["dev", "test"]
    assert len(df) == len(toy["dev"]) + len(toy["test"])
    run = json.loads(runs[0].read_text())
    for key in ("config", "dev_macro_f1", "dev_accuracy", "test_macro_f1", "test_accuracy",
                "train_time_sec", "device"):
        assert key in run
    search = json.loads((tmp_path / "metrics" / "search_svm.json").read_text())
    assert len(search["grid"]) == len(cfg["svm"]["C_grid"])
    # toy data is trivially separable
    assert res["dev_macro_f1"].min() > 0.9
