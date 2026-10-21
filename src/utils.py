"""Shared helpers: config loading, logging, paths and simple I/O."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
