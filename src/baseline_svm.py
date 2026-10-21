"""TF-IDF (word + char n-grams) + LinearSVC baseline.

Procedure (mirrors the transformer setup; the test set is never used for selection):
  1. Fit the TF-IDF features on train.
  2. Tune C over the config grid on dev macro-F1, using the LR-search seed (42).
  3. With the best C, train on train only for each seed (42, 43, 44) and predict dev + test.

Usage:
    python -m src.baseline_svm               # model key "svm"
    python -m src.baseline_svm --lang_tags   # model key "svm_tags": word features become e.g. "ladki_HIN"
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from src.preprocess import clean_for_svm
from src.utils import (
    compute_metrics,
    label_map,
    load_config,
    now_iso,
    pred_path,
    predictions_frame,
    read_jsonl,
    resolve_path,
    run_json_path,
    run_name,
    set_seed,
    setup_logging,
    write_json,
    write_predictions,
)

log = logging.getLogger(__name__)

SPLITS = ("train", "dev", "test")
# The data is already tokenized (punctuation is split off), so word features split on whitespace.
# sklearn's default pattern would drop one-letter Hinglish words such as "h" (hai) and "k" (ke).
WORD_TOKEN_PATTERN = r"\S+"


# --------------------------------------------------------------------------- inputs


def tagged_text(tokens: list[str], tags: list[str]) -> str:
    """SVM-cleaned tokens with their language tag appended: 'ladki_HIN bahut_HIN cute_ENG'."""
    pieces = []
    for tok, tag in zip(tokens, tags, strict=True):
        pieces.extend(f"{p}_{tag}" for p in clean_for_svm(tok).split())
    return " ".join(pieces)


def load_split(proc_dir: Path, split: str, lang_tags: bool) -> pd.DataFrame:
    """Processed split with `char_text` (cleaned) and `word_text` (cleaned, optionally tag-suffixed)."""
    df = pd.read_csv(proc_dir / f"{split}.csv", dtype={"id": str})
    df["char_text"] = df["text"].map(clean_for_svm)
    if lang_tags:
        recs = {r["id"]: r for r in read_jsonl(proc_dir / f"{split}.jsonl")}
        df["word_text"] = [tagged_text(recs[i]["tokens"], recs[i]["tags"]) for i in df["id"]]
    else:
        df["word_text"] = df["char_text"]
    return df


# --------------------------------------------------------------------------- model


def select_word_text(df: pd.DataFrame) -> pd.Series:
    """Column selector for the word TF-IDF branch."""
    return df["word_text"]


def select_char_text(df: pd.DataFrame) -> pd.Series:
    """Column selector for the char TF-IDF branch."""
    return df["char_text"]


def build_features(svm_cfg: dict[str, Any]) -> FeatureUnion:
    """FeatureUnion of word (1-2)-gram and char_wb (2-5)-gram TF-IDF."""
    word = TfidfVectorizer(
        ngram_range=tuple(svm_cfg["word_ngram"]),
        min_df=svm_cfg["min_df"],
        sublinear_tf=True,
        token_pattern=WORD_TOKEN_PATTERN,
        lowercase=False,  # text is already lower-cased; keeps the "_HIN" suffix as is
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(svm_cfg["char_ngram"]),
        min_df=svm_cfg["min_df"],
        sublinear_tf=True,
    )
    return FeatureUnion(
        [
            ("word", Pipeline([("select", FunctionTransformer(select_word_text)), ("tfidf", word)])),
            ("char", Pipeline([("select", FunctionTransformer(select_char_text)), ("tfidf", char)])),
        ]
    )


def make_svm(C: float, seed: int) -> LinearSVC:
    """LinearSVC with balanced class weights."""
    return LinearSVC(C=C, class_weight="balanced", random_state=seed, max_iter=10000)


def fit_quietly(clf: LinearSVC, X: Any, y: np.ndarray) -> tuple[LinearSVC, bool]:
    """Fit and report whether liblinear raised a ConvergenceWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, y)
    return clf, any(issubclass(w.category, ConvergenceWarning) for w in caught)


# --------------------------------------------------------------------------- run


def run_svm(
    dfs: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    results_dir: Path,
    lang_tags: bool = False,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Tune C on dev, run all seeds at the best C, write standard outputs. Returns per-seed results."""
    model = "svm_tags" if lang_tags else "svm"
    svm_cfg = cfg["svm"]
    seeds = seeds or list(svm_cfg["seeds"])
    search_seed = int(cfg["training"]["lr_search_seed"])
    labels = label_map(cfg)
    y = {s: dfs[s]["label"].str.lower().map(labels).to_numpy() for s in SPLITS}

    t0 = time.perf_counter()
    features = build_features(svm_cfg)
    X = {"train": features.fit_transform(dfs["train"])}
    X["dev"], X["test"] = features.transform(dfs["dev"]), features.transform(dfs["test"])
    feat_time = time.perf_counter() - t0
    n_word = len(features.transformer_list[0][1].named_steps["tfidf"].vocabulary_)
    n_char = len(features.transformer_list[1][1].named_steps["tfidf"].vocabulary_)
    log.info("%s: features word=%d char=%d (%.1fs)", model, n_word, n_char, feat_time)

    # --- C search on dev (seed = LR-search seed) ---
    search = []
    for C in svm_cfg["C_grid"]:
        clf, warned = fit_quietly(make_svm(C, search_seed), X["train"], y["train"])
        m = compute_metrics(y["dev"], clf.predict(X["dev"]))
        search.append({"C": C, "dev_macro_f1": m["macro_f1"], "dev_accuracy": m["accuracy"],
                       "converged": not warned})
        log.info("%s: C=%-5g dev macro-F1=%.4f acc=%.4f%s", model, C, m["macro_f1"], m["accuracy"],
                 "" if not warned else "  (ConvergenceWarning)")
    best = max(search, key=lambda r: (r["dev_macro_f1"], -r["C"]))  # ties -> smaller C
    best_C = best["C"]
    lr_tag = f"C{best_C:g}"
    log.info("%s: best C=%g (dev macro-F1=%.4f)", model, best_C, best["dev_macro_f1"])
    search_summary = {
        "model": model, "search_seed": search_seed, "grid": search, "best_C": best_C, "timestamp": now_iso(),
    }
    write_json(search_summary, results_dir / "metrics" / f"search_{model}.json")

    # --- final runs: train only, one per seed ---
    rows = []
    for seed in seeds:
        set_seed(seed)
        t1 = time.perf_counter()
        clf, warned = fit_quietly(make_svm(best_C, seed), X["train"], y["train"])
        train_time = time.perf_counter() - t1
        assert list(clf.classes_) == [0, 1, 2]
        frames, metrics = [], {}
        for split in ("dev", "test"):
            scores = clf.decision_function(X[split])
            frames.append(predictions_frame(dfs[split]["id"], split, y[split], scores))
            for k, v in compute_metrics(y[split], scores.argmax(1)).items():
                metrics[f"{split}_{k}"] = v
        name = run_name(model, lr_tag, seed)
        write_predictions(pred_path(results_dir, name), frames)
        write_json(
            {
                "model": model,
                "lr": lr_tag,
                "seed": seed,
                "fraction": 1.0,
                "phase": "final",
                "config": {
                    "C": best_C,
                    "class_weight": "balanced",
                    "lang_tags": lang_tags,
                    **{k: svm_cfg[k] for k in ("word_ngram", "char_ngram", "min_df")},
                    "word_token_pattern": WORD_TOKEN_PATTERN,
                    "n_features_word": n_word,
                    "n_features_char": n_char,
                    "converged": not warned,
                },
                **metrics,
                "train_time_sec": round(train_time + feat_time, 2),
                "device": "cpu",
                "n_train": len(dfs["train"]),
                "timestamp": now_iso(),
            },
            run_json_path(results_dir, name),
        )
        rows.append({"model": model, "C": best_C, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def log_results(res: pd.DataFrame) -> None:
    """Log per-seed results and mean ± std."""
    for r in res.itertuples(index=False):
        log.info("%s seed=%d  dev F1=%.4f acc=%.4f | test F1=%.4f acc=%.4f",
                 r.model, r.seed, r.dev_macro_f1, r.dev_accuracy, r.test_macro_f1, r.test_accuracy)
    for col in ("dev_macro_f1", "dev_accuracy", "test_macro_f1", "test_accuracy"):
        log.info("%s %-14s mean=%.4f std=%.4f (n=%d)", res["model"].iloc[0], col,
                 res[col].mean(), res[col].std(ddof=1) if len(res) > 1 else 0.0, len(res))


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + LinearSVC baseline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None, help="processed data dir")
    parser.add_argument("--out_dir", default=None, help="results dir")
    parser.add_argument("--lang_tags", action="store_true", help="append language tags to word features")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    proc_dir = resolve_path(args.data_dir or cfg["paths"]["processed_dir"])
    results_dir = resolve_path(args.out_dir or cfg["paths"]["results_dir"])
    dfs = {s: load_split(proc_dir, s, args.lang_tags) for s in SPLITS}
    res = run_svm(dfs, cfg, results_dir, lang_tags=args.lang_tags, seeds=args.seeds)
    log_results(res)


if __name__ == "__main__":
    main()
