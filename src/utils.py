"""Shared helpers: config, logging, paths, seeding, metrics and the standard output files.

Every model (SVM and transformers) writes the same two files per run:
    results/preds/preds_{model}_{lr}_{seed}[_frac{f}].csv
        columns: id, split, gold, pred, score_neg, score_neu, score_pos   (dev and test rows)
    results/metrics/run_{model}_{lr}_{seed}[_frac{f}].json
        config, dev/test macro-F1 and accuracy, training time, device
`gold` and `pred` are label ids from the fixed label map (negative=0, neutral=1, positive=2).
For the SVM the `lr` slot holds the regularization constant, e.g. "C0.1".
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
SCORE_COLUMNS = ["score_neg", "score_neu", "score_pos"]
PRED_COLUMNS = ["id", "split", "gold", "pred", *SCORE_COLUMNS]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- config + logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Hugging Face downloads log every HTTP request at INFO; keep them quiet.
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config (defaults to configs/default.yaml)."""
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Return `p` as an absolute path; relative paths are taken from the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def label_map(cfg: dict[str, Any]) -> dict[str, int]:
    """Return the fixed label map {name: id} with lower-case names."""
    return {k.lower(): int(v) for k, v in cfg["labels"].items()}


# --------------------------------------------------------------------------- seeding


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- metrics


def compute_metrics(gold: Sequence[int], pred: Sequence[int]) -> dict[str, float]:
    """Macro-F1 over the three classes and accuracy."""
    return {
        "macro_f1": float(f1_score(gold, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "accuracy": float(accuracy_score(gold, pred)),
    }


# --------------------------------------------------------------------------- run names + files


def format_lr(lr: float) -> str:
    """Learning rate as used in file names: 2e-05 -> '2e-05'."""
    return f"{lr:.0e}"


def run_name(model: str, lr: str, seed: int, fraction: float = 1.0) -> str:
    """'{model}_{lr}_{seed}' plus '_frac{f}' for learning-curve runs (fraction < 1)."""
    name = f"{model}_{lr}_{seed}"
    if fraction < 1.0:
        name += f"_frac{fraction:g}"
    return name


def pred_path(results_dir: Path, name: str) -> Path:
    """results/preds/preds_{name}.csv"""
    return results_dir / "preds" / f"preds_{name}.csv"


def run_json_path(results_dir: Path, name: str) -> Path:
    """results/metrics/run_{name}.json"""
    return results_dir / "metrics" / f"run_{name}.json"


def predictions_frame(
    ids: Sequence[str], split: str, gold: Sequence[int], scores: np.ndarray
) -> pd.DataFrame:
    """Standard prediction rows for one split; `scores` is (n, 3) in label-id order."""
    scores = np.asarray(scores, dtype=float)
    assert scores.shape == (len(ids), 3), f"scores shape {scores.shape} != ({len(ids)}, 3)"
    df = pd.DataFrame({"id": list(ids), "split": split, "gold": list(gold), "pred": scores.argmax(1)})
    df[SCORE_COLUMNS] = scores
    return df[PRED_COLUMNS]


def write_predictions(path: Path, frames: Iterable[pd.DataFrame]) -> None:
    """Concatenate per-split prediction frames and write the standard CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(list(frames), ignore_index=True)
    assert list(df.columns) == PRED_COLUMNS
    df.to_csv(path, index=False)


def read_predictions(path: Path) -> pd.DataFrame:
    """Read a standard prediction CSV (ids as strings)."""
    return pd.read_csv(path, dtype={"id": str, "split": str})


def write_json(obj: dict[str, Any], path: Path) -> None:
    """Write pretty JSON (creating the folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def now_iso() -> str:
    """Current local time, ISO 8601, seconds precision."""
    return dt.datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- jsonl


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write one JSON object per line (UTF-8, non-ASCII kept as is)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
