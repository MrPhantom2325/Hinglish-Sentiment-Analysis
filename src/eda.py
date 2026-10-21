"""Dataset statistics and figures for the processed SentiMix splits.

Writes figures to results/figures/ and a summary to results/tables/eda_summary.md.

Usage:
    python -m src.eda                      # includes tokenizer subword lengths (downloads tokenizers)
    python -m src.eda --skip_tokenizers    # offline / quick
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.plotting import INK_2, LABEL_COLORS, LABEL_ORDER, SINGLE, SURFACE, apply_style, plt, save
from src.preprocess import clean_for_svm, clean_for_transformer
from src.utils import load_config, read_jsonl, resolve_path, setup_logging

log = logging.getLogger(__name__)

SPLITS = ("train", "dev", "test")
WORD_START_RE = re.compile(r"^\w")


# --------------------------------------------------------------------------- loading + tables


def load_splits(proc_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, Any]]]]:
    """Load the processed CSV and JSONL for every split."""
    dfs = {s: pd.read_csv(proc_dir / f"{s}.csv", dtype={"id": str}) for s in SPLITS}
    recs = {s: read_jsonl(proc_dir / f"{s}.jsonl") for s in SPLITS}
    return dfs, recs


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency)."""

    def fmt(v: Any) -> str:
        if isinstance(v, float | np.floating):
            return floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def label_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts and percentages of each label per split."""
    rows = []
    for s, df in dfs.items():
        c = df["label"].value_counts()
        row: dict[str, Any] = {"split": s, "n": len(df)}
        for lab in LABEL_ORDER:
            row[lab] = f"{c.get(lab, 0)} ({100 * c.get(lab, 0) / len(df):.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def describe_table(dfs: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """Mean / median / p95 / min / max of one numeric column per split."""
    rows = []
    for s, df in dfs.items():
        x = df[col].dropna()
        rows.append(
            {
                "split": s,
                "mean": x.mean(),
                "median": x.median(),
                "p95": x.quantile(0.95),
                "min": x.min(),
                "max": x.max(),
                "share == 0": f"{100 * (x == 0).mean():.1f}%",
            }
        )
    return pd.DataFrame(rows)


def cmi_by_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """CMI statistics per label for one split."""
    rows = []
    for lab in LABEL_ORDER:
        x = df.loc[df["label"] == lab, "cmi"]
        rows.append(
            {
                "label": lab,
                "n": len(x),
                "mean CMI": x.mean(),
                "median CMI": x.median(),
                "share CMI == 0": f"{100 * (x == 0).mean():.1f}%",
                "mean hin_ratio": df.loc[df["label"] == lab, "hin_ratio"].mean(),
            }
        )
    return pd.DataFrame(rows)


def top_tokens(records: list[dict[str, Any]], tag: str, k: int = 30) -> list[tuple[str, int]]:
    """Most frequent lower-cased word tokens with a given tag (mentions, hashtags, URLs excluded)."""
    c: Counter[str] = Counter()
    for r in records:
        for tok, t in zip(r["tokens"], r["tags"], strict=True):
            if t == tag and WORD_START_RE.match(tok) and not tok.lower().startswith("http"):
                c[tok.lower()] += 1
    return c.most_common(k)


def overlap_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How many dev/test tweets have an exact duplicate (after SVM-style cleaning) in train."""
    train_texts = set(dfs["train"]["text"].map(clean_for_svm))
    rows = []
    for s in ("dev", "test"):
        cleaned = dfs[s]["text"].map(clean_for_svm)
        dup = cleaned.isin(train_texts)
        rows.append({"split": s, "tweets also in train": int(dup.sum()), "share": f"{100 * dup.mean():.2f}%"})
    return pd.DataFrame(rows)


def subword_table(texts: list[str], models: dict[str, str], max_length: int) -> pd.DataFrame:
    """Subword sequence length (with [CLS]/[SEP]) per tokenizer on cleaned texts."""
    from transformers import AutoTokenizer

    rows = []
    for key, model_id in models.items():
        tok = AutoTokenizer.from_pretrained(model_id)
        lengths = np.array([len(ids) for ids in tok(texts, add_special_tokens=True)["input_ids"]])
        rows.append(
            {
                "model": key,
                "tokenizer": model_id,
                "vocab size": tok.vocab_size,
                "mean": lengths.mean(),
                "median": float(np.median(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "p99": float(np.percentile(lengths, 99)),
                "max": int(lengths.max()),
                f"share > {max_length}": f"{100 * (lengths > max_length).mean():.2f}%",
            }
        )
        log.info("subwords %-8s p95=%.0f max=%d", key, rows[-1]["p95"], rows[-1]["max"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


def plot_label_distribution(dfs: dict[str, pd.DataFrame], path: Path) -> None:
    """Grouped bars: share of each label within each split."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(dfs))
    width = 0.26
    for i, lab in enumerate(LABEL_ORDER):
        shares = [100 * (df["label"] == lab).mean() for df in dfs.values()]
        bars = ax.bar(
            x + (i - 1) * width, shares, width, color=LABEL_COLORS[lab], label=lab,
            edgecolor=SURFACE, linewidth=2,
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x, [f"{s}\n(n={len(df):,})" for s, df in dfs.items()])
    ax.set_ylabel("% of tweets in split")
    ax.set_ylim(0, 45)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Label distribution per split")
    save(fig, path)


def plot_token_lengths(df: pd.DataFrame, path: Path) -> None:
    """Histogram of tokens per tweet (train) with median and p95 marked."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    n = df["n_tokens"]
    ax.hist(n, bins=np.arange(0.5, n.max() + 1.5, 1), color=SINGLE, edgecolor=SURFACE, linewidth=0.5)
    # median label sits left of its line, p95 label right of its line, so they never collide
    for q, name, dx, ha in ((n.median(), "median", -4, "right"), (n.quantile(0.95), "p95", 4, "left")):
        ax.axvline(q, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(f"{name} = {q:.0f}", (q, ax.get_ylim()[1] * 0.92), xytext=(dx, 0),
                    textcoords="offset points", fontsize=8, color=INK_2, ha=ha)
    ax.set_xlabel("tokens per tweet (after rejoining mentions/hashtags/URLs)")
    ax.set_ylabel("tweets")
    ax.grid(axis="x", visible=False)
    ax.set_title("Tweet length (train)")
    save(fig, path)


def plot_mixing(df: pd.DataFrame, path: Path) -> None:
    """Two panels: hin_ratio and CMI histograms (train)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(df["hin_ratio"].dropna(), bins=np.linspace(0, 1, 21), color=SINGLE,
                 edgecolor=SURFACE, linewidth=1)
    axes[0].set_xlabel("hin_ratio = n_hin / (n_hin + n_eng)")
    axes[0].set_ylabel("tweets")
    axes[0].set_title("Share of Hindi-tagged words (train)")
    axes[1].hist(df["cmi"], bins=np.arange(0, 52.5, 2.5), color=SINGLE, edgecolor=SURFACE, linewidth=1)
    axes[1].set_xlabel("Code-Mixing Index (0 = monolingual, 50 = even mix)")
    axes[1].set_title("Code-Mixing Index (train)")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    save(fig, path)


def plot_cmi_by_label(df: pd.DataFrame, path: Path) -> None:
    """Box plot of CMI per label (train), with the mean marked."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    data = [df.loc[df["label"] == lab, "cmi"] for lab in LABEL_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showmeans=True,
                    medianprops={"color": SURFACE, "linewidth": 2},
                    meanprops={"marker": "D", "markerfacecolor": SURFACE, "markeredgecolor": INK_2,
                               "markersize": 6},
                    whiskerprops={"color": INK_2}, capprops={"color": INK_2},
                    flierprops={"marker": ".", "markersize": 3, "markeredgecolor": INK_2, "alpha": 0.3})
    for patch, lab in zip(bp["boxes"], LABEL_ORDER, strict=True):
        patch.set_facecolor(LABEL_COLORS[lab])
        patch.set_edgecolor(LABEL_COLORS[lab])
    ticks = [f"{lab}\nmean {x.mean():.1f}" for lab, x in zip(LABEL_ORDER, data, strict=True)]
    ax.set_xticks(range(1, 4), ticks)
    ax.set_ylabel("CMI")
    ax.grid(axis="x", visible=False)
    ax.set_title("Code-Mixing Index by label (train)")
    save(fig, path)


def plot_top_tokens(hin: list[tuple[str, int]], eng: list[tuple[str, int]], path: Path) -> None:
    """Side-by-side horizontal bars of the most frequent Hin- and Eng-tagged tokens."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    for ax, items, name in ((axes[0], hin, "Hin"), (axes[1], eng, "Eng")):
        words = [w for w, _ in items][::-1]
        counts = [c for _, c in items][::-1]
        ax.barh(words, counts, color=SINGLE, edgecolor=SURFACE, linewidth=1, height=0.75)
        ax.set_title(f"Top {len(items)} {name}-tagged tokens (train)")
        ax.set_xlabel("occurrences (lower-cased)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    save(fig, path)


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for the processed SentiMix splits.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None, help="processed data dir")
    parser.add_argument("--out_dir", default=None, help="results dir")
    parser.add_argument("--skip_tokenizers", action="store_true")
    args = parser.parse_args()
    setup_logging()
    apply_style()

    cfg = load_config(args.config)
    proc_dir = resolve_path(args.data_dir or cfg["paths"]["processed_dir"])
    res_dir = resolve_path(args.out_dir or cfg["paths"]["results_dir"])
    fig_dir, tab_dir = res_dir / "figures", res_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    dfs, recs = load_splits(proc_dir)
    train = dfs["train"]

    plot_label_distribution(dfs, fig_dir / "eda_label_distribution.png")
    plot_token_lengths(train, fig_dir / "eda_token_length.png")
    plot_mixing(train, fig_dir / "eda_hin_ratio_cmi.png")
    plot_cmi_by_label(train, fig_dir / "eda_cmi_by_label.png")
    hin, eng = top_tokens(recs["train"], "HIN"), top_tokens(recs["train"], "ENG")
    plot_top_tokens(hin, eng, fig_dir / "eda_top_tokens.png")
    log.info("figures written to %s", fig_dir)

    max_length = cfg["training"]["max_length"]
    if args.skip_tokenizers:
        sub_md = "_Skipped (`--skip_tokenizers`)._"
    else:
        texts = train["text"].map(clean_for_transformer).tolist()
        sub = subword_table(texts, cfg["models"], max_length)
        sub.to_csv(tab_dir / "eda_subword_lengths.csv", index=False)
        sub_md = md_table(sub, "{:.1f}")

    top_df = pd.DataFrame(
        {
            "rank": range(1, 31),
            "Hin token": [w for w, _ in hin],
            "Hin count": [c for _, c in hin],
            "Eng token": [w for w, _ in eng],
            "Eng count": [c for _, c in eng],
        }
    )
    n_czech_like = int(train["text"].str.contains("[ěůřčšž]", regex=True).sum())

    sections = [
        "# EDA summary (SentiMix Hinglish)",
        "",
        "Generated by `python -m src.eda` from `data/processed/`. Tokens are counted after rejoining "
        "mentions, hashtags and URLs (see `src/data.py`). Figures are in `results/figures/eda_*.png`.",
        "",
        "## Label distribution",
        "",
        md_table(label_table(dfs)),
        "",
        "## Tokens per tweet",
        "",
        md_table(describe_table(dfs, "n_tokens"), "{:.1f}"),
        "",
        "## Code-mixing",
        "",
        "hin_ratio = n_hin / (n_hin + n_eng):",
        "",
        md_table(describe_table(dfs, "hin_ratio"), "{:.3f}"),
        "",
        "CMI = 100 · (1 − max(n_hin, n_eng) / (n − u)), u = language-independent tokens (O, EMT):",
        "",
        md_table(describe_table(dfs, "cmi")),
        "",
        "CMI by label (train):",
        "",
        md_table(cmi_by_label_table(train)),
        "",
        "## Most frequent Hin- and Eng-tagged tokens (train, lower-cased)",
        "",
        md_table(top_df),
        "",
        f"## Subword length per tokenizer (train, `clean_for_transformer`, incl. [CLS]/[SEP]; "
        f"max_length = {max_length})",
        "",
        sub_md,
        "",
        "## Train/dev/test overlap (exact duplicates after `clean_for_svm`)",
        "",
        md_table(overlap_table(dfs)),
        "",
        "## Notes",
        "",
        f"- {n_czech_like} train tweet(s) contain Czech-only characters (e.g. id 4330); they are kept "
        "as released.",
        "- Mojibake in the test file was repaired in `src/data.py` "
        f"(`preprocessing.fix_mojibake: {cfg.get('preprocessing', {}).get('fix_mojibake', True)}`).",
        "- Mentions take the language tag of the username; rejoined URLs are tagged `O`.",
        "",
    ]
    out = tab_dir / "eda_summary.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
