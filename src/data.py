"""Parse SentiMix CoNLL files into processed CSV/JSONL splits.

Pipeline per split:
    parse_conll -> (test only) attach_labels -> rejoin -> tweet_features -> write CSV + JSONL

Usage:
    python -m src.data                     # paths from configs/default.yaml
    python -m src.data --out_dir some/dir
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import label_map, load_config, resolve_path, setup_logging, write_jsonl

log = logging.getLogger(__name__)

SPLITS = ("train", "dev", "test")
HINDI_TAGS = frozenset({"HIN"})
ENGLISH_TAGS = frozenset({"ENG"})
INDEPENDENT_TAGS = frozenset({"O", "EMT"})  # language-independent tokens (u in CMI)
KNOWN_TAGS = HINDI_TAGS | ENGLISH_TAGS | INDEPENDENT_TAGS
MISSING_TAG = "O"  # tag given to token lines that have no tag
URL_TAG = "O"  # a rejoined URL is language-independent

WORD_RE = re.compile(r"^\w+$")
UNDERSCORES_RE = re.compile(r"^_+$")
MENTION_START_RE = re.compile(r"^@_*$")  # "@" or "@_" / "@__" (handle starting with underscores)
HASHTAG_START_RE = re.compile(r"^#_*$")
URL_SCHEME_RE = re.compile(r"^https?$", re.IGNORECASE)
URL_SEPARATOR_RE = re.compile(r"^[:/]+$")  # ":", "//", "/", "://"
T_CO_PARTS = frozenset({"t", ".", "co", "tco", "tc"})
ELLIPSIS = "…"
# Lead characters of UTF-8 text that was wrongly decoded as Windows-1252 ("â€¦" for "…",
# "ðŸ˜‚" for 😂). None of these occur in the train/dev files.
MOJIBAKE_HINT_RE = re.compile("[ÂÃâð]")


@dataclass
class Tweet:
    """One tweet: id, optional label, and token-aligned language tags."""

    id: str
    label: str | None
    tokens: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- parsing


def normalize_tag(tag: str) -> str:
    """Upper-case and strip a language tag ('Hin' / 'hin ' -> 'HIN'); empty -> MISSING_TAG."""
    tag = tag.strip().upper()
    return tag if tag else MISSING_TAG


def _split_token_line(line: str) -> tuple[str, str | None]:
    """Split a token line into (token, tag-or-None). Tolerates spaces instead of a tab.

    The token may come back empty: the raw data has lines like '\\xa0<TAB>O' whose token is
    only whitespace. The caller skips those.
    """
    if "\t" in line:
        token, _, rest = line.partition("\t")
        tag = rest.strip().split("\t")[-1].strip() if rest.strip() else None
        return token.strip(), tag
    # No tab: maybe "token   Hin" separated by spaces.
    pieces = line.split()
    if len(pieces) >= 2 and pieces[-1].upper() in KNOWN_TAGS:
        return " ".join(pieces[:-1]), pieces[-1]
    return line.strip(), None


def parse_conll_lines(lines: Iterable[str]) -> list[Tweet]:
    """Parse CoNLL lines. A `meta` line always starts a new tweet, so missing blank lines are fine."""
    tweets: list[Tweet] = []
    current: Tweet | None = None
    n_missing_tag = n_orphan = n_empty_token = 0
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        first = line.strip().split(None, 1)[0]
        if first.lower() == "meta":
            fields = line.strip().split()
            if len(fields) < 2:
                raise ValueError(f"meta line without id: {line!r}")
            label = fields[2].lower() if len(fields) > 2 else None
            current = Tweet(id=fields[1], label=label)
            tweets.append(current)
            continue
        if current is None:
            n_orphan += 1
            continue
        token, tag = _split_token_line(line)
        if not token:
            n_empty_token += 1
            continue
        if tag is None:
            n_missing_tag += 1
            tag = MISSING_TAG
        current.tokens.append(token)
        current.tags.append(normalize_tag(tag))
    if n_empty_token:
        log.info("%d token lines had an empty/whitespace-only token (skipped)", n_empty_token)
    if n_missing_tag:
        log.warning("%d token lines had no tag (set to %s)", n_missing_tag, MISSING_TAG)
    if n_orphan:
        log.warning("%d token lines appeared before any meta line (ignored)", n_orphan)
    return tweets


def parse_conll(path: str | Path) -> list[Tweet]:
    """Parse one CoNLL file."""
    with Path(path).open(encoding="utf-8") as f:
        return parse_conll_lines(f)


def read_test_labels(path: str | Path) -> dict[str, str]:
    """Read a `Uid,Sentiment` CSV into {id: lower-case label}."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in reader.fieldnames or []}
        id_col, lab_col = cols["uid"], cols["sentiment"]
        return {row[id_col].strip(): row[lab_col].strip().lower() for row in reader}


def attach_labels(tweets: list[Tweet], labels: dict[str, str]) -> None:
    """Set each tweet's label from `labels`; every tweet must receive one."""
    missing = [t.id for t in tweets if t.id not in labels]
    assert not missing, f"{len(missing)} tweets have no label, e.g. {missing[:5]}"
    for t in tweets:
        t.label = labels[t.id]


# --------------------------------------------------------------------------- encoding repair


def fix_mojibake(token: str) -> str:
    """Repair UTF-8 that was decoded as Windows-1252 ('â€¦' -> '…', 'ðŸ˜‚' -> '😂').

    Each character is mapped back to its byte (cp1252, falling back to Latin-1 for the five
    bytes cp1252 leaves undefined) and the bytes are decoded as UTF-8. Tokens without a
    mojibake lead character, or that do not decode cleanly, are returned unchanged.
    """
    if not MOJIBAKE_HINT_RE.search(token):
        return token
    buf = bytearray()
    for ch in token:
        try:
            buf += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) >= 256:
                return token
            buf.append(ord(ch))
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return token


# --------------------------------------------------------------------------- rejoining


def _url_starts_at(tokens: list[str], i: int) -> bool:
    """True if tokens[i] is 'http(s)' followed by a separator ('//', ':') or a truncation '…'."""
    return (
        i + 1 < len(tokens)
        and bool(URL_SCHEME_RE.match(tokens[i]))
        and bool(URL_SEPARATOR_RE.match(tokens[i + 1]) or ELLIPSIS in tokens[i + 1])
    )


def _is_handle_word(tokens: list[str], j: int) -> bool:
    """A word token that can be part of a handle (not underscores only, not a URL start)."""
    return (
        j < len(tokens)
        and bool(WORD_RE.match(tokens[j]))
        and not UNDERSCORES_RE.match(tokens[j])
        and not _url_starts_at(tokens, j)
    )


def _consume_handle(tokens: list[str], j: int) -> int:
    """Consume a user/hashtag handle starting at index j; return the index after it.

    A handle is: word ( '_'+ word? )*  — the original tokenizer split handles at underscores,
    e.g. "@ abhisar _ sharma" is the handle "abhisar_sharma". A following URL is never absorbed.
    """
    if not _is_handle_word(tokens, j):
        return j
    j += 1
    while j < len(tokens) and UNDERSCORES_RE.match(tokens[j]):
        j += 1
        if _is_handle_word(tokens, j):
            j += 1
    return j


def _consume_url(tokens: list[str], i: int) -> int:
    """Consume a split t.co-style URL starting at the scheme token at index i.

    Handles e.g. "https // t . co / AbC123", "https // tco / AbC123", "https // t co / AbC",
    and truncated forms ending in "…". Returns the index after the URL.
    """
    j = i + 1
    n = len(tokens)
    # scheme separator(s): ":" "//" "://"  (or a truncation "…", "//…")
    while j < n and (URL_SEPARATOR_RE.match(tokens[j]) or ELLIPSIS in tokens[j]):
        if ELLIPSIS in tokens[j]:
            return j + 1
        j += 1
    # domain: t . co | tco | t co | tc
    while j < n and tokens[j].lower() in T_CO_PARTS:
        j += 1
    # path: "/" ID ["…"]   (or a truncated "/…")
    if j < n and ELLIPSIS in tokens[j] and tokens[j].strip("/" + ELLIPSIS) == "":
        return j + 1
    if j < n and tokens[j] == "/":
        j += 1
        if j < n and WORD_RE.match(tokens[j]):
            j += 1
        if j < n and tokens[j] == ELLIPSIS:
            j += 1
    return j


def rejoin(tokens: list[str], tags: list[str]) -> tuple[list[str], list[str]]:
    """Rejoin split mentions, hashtags and URLs, keeping a token-aligned tag list.

    - "@" + handle -> "@handle", tag of the first handle word
    - "#" + handle -> "#handle", tag of the first handle word
    - "https" "//" "t" "." "co" "/" "ID" -> one URL token, tag URL_TAG ("O")
    """
    out_tok: list[str] = []
    out_tag: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if MENTION_START_RE.match(tok) or HASHTAG_START_RE.match(tok):
            end = _consume_handle(tokens, i + 1)
            if end > i + 1:
                out_tok.append("".join(tokens[i:end]))
                out_tag.append(tags[i + 1])
                i = end
                continue
        elif _url_starts_at(tokens, i):
            end = _consume_url(tokens, i)
            out_tok.append("".join(tokens[i:end]))
            out_tag.append(URL_TAG)
            i = end
            continue
        out_tok.append(tok)
        out_tag.append(tags[i])
        i += 1
    return out_tok, out_tag


# --------------------------------------------------------------------------- features


def compute_cmi(n_hin: int, n_eng: int, n: int, u: int) -> float:
    """Code-Mixing Index (Das & Gambäck): 100 * (1 - max(n_hin, n_eng) / (n - u)); 0 if n == u."""
    if n == u:
        return 0.0
    return 100.0 * (1.0 - max(n_hin, n_eng) / (n - u))


def tweet_features(tags: list[str]) -> dict[str, Any]:
    """Counts, hin_ratio (NaN if no Hin/Eng tokens) and CMI from a tag list."""
    n = len(tags)
    n_hin = sum(t in HINDI_TAGS for t in tags)
    n_eng = sum(t in ENGLISH_TAGS for t in tags)
    n_other = sum(t in INDEPENDENT_TAGS for t in tags)
    hin_ratio = n_hin / (n_hin + n_eng) if (n_hin + n_eng) else math.nan
    return {
        "n_tokens": n,
        "n_hin": n_hin,
        "n_eng": n_eng,
        "n_other": n_other,
        "hin_ratio": hin_ratio,
        "cmi": compute_cmi(n_hin, n_eng, n, n_other),
    }


def build_records(tweets: list[Tweet], repair_encoding: bool = True) -> list[dict[str, Any]]:
    """Turn parsed tweets into records with rejoined text, tokens, tags and features.

    `raw_tokens` keep the file's tokens exactly; `tokens` are encoding-repaired (optional) and rejoined.
    """
    records = []
    n_fixed = 0
    for t in tweets:
        tokens = t.tokens
        if repair_encoding:
            tokens = [fix_mojibake(tok) for tok in t.tokens]
            n_fixed += sum(a != b for a, b in zip(tokens, t.tokens, strict=True))
        tokens, tags = rejoin(tokens, t.tags)
        records.append(
            {
                "id": t.id,
                "label": t.label,
                "text": " ".join(tokens),
                "tokens": tokens,
                "tags": tags,
                "raw_tokens": t.tokens,
                "raw_tags": t.tags,
                **tweet_features(tags),
            }
        )
    if repair_encoding:
        log.info("encoding repair: %d tokens fixed (mojibake)", n_fixed)
    return records


# --------------------------------------------------------------------------- checks + main


def check_split(name: str, records: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    """Assert size, label counts, unique ids, valid labels and non-empty texts."""
    labels = label_map(cfg)
    expected = cfg.get("expected", {})
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
    assert all(r["text"].strip() for r in records), f"{name}: empty text"
    bad = {r["label"] for r in records} - set(labels)
    assert not bad, f"{name}: unknown labels {bad}"
    if name in expected.get("sizes", {}):
        assert len(records) == expected["sizes"][name], (
            f"{name}: {len(records)} tweets, expected {expected['sizes'][name]}"
        )
    counts = Counter(r["label"] for r in records)
    if name in expected.get("label_counts", {}):
        assert counts == Counter(expected["label_counts"][name]), (
            f"{name}: label counts {dict(counts)} != {expected['label_counts'][name]}"
        )


def log_summary(name: str, records: list[dict[str, Any]]) -> None:
    """Log counts and simple feature statistics for one split."""
    df = pd.DataFrame(records)
    tag_counts = Counter(t for r in records for t in r["tags"])
    log.info("%-5s n=%d labels=%s", name, len(df), dict(sorted(Counter(df["label"]).items())))
    log.info(
        "%-5s tokens: raw=%d rejoined=%d  tags=%s",
        name,
        sum(len(r["raw_tokens"]) for r in records),
        int(df["n_tokens"].sum()),
        dict(sorted(tag_counts.items())),
    )
    log.info(
        "%-5s mean n_tokens=%.1f  mean cmi=%.2f  mean hin_ratio=%.3f  hin_ratio NaN (no Hin/Eng)=%d",
        name,
        df["n_tokens"].mean(),
        df["cmi"].mean(),
        df["hin_ratio"].mean(),
        int(df["hin_ratio"].isna().sum()),
    )
    n_url = sum(tok.lower().startswith("http") for r in records for tok in r["tokens"])
    n_mention = sum(tok.startswith("@") and len(tok) > 1 for r in records for tok in r["tokens"])
    n_hashtag = sum(tok.startswith("#") and len(tok) > 1 for r in records for tok in r["tokens"])
    leftover = sum(
        tok == "//" or URL_SCHEME_RE.match(tok) is not None for r in records for tok in r["tokens"]
    )
    log.info(
        "%-5s rejoined: urls=%d mentions=%d hashtags=%d  leftover '//' or bare 'http(s)' tokens=%d",
        name, n_url, n_mention, n_hashtag, leftover,
    )


def show_examples(records: list[dict[str, Any]], k: int = 3) -> None:
    """Log k tweets (one with a mention, a hashtag, a URL if possible) as raw vs rejoined."""
    wanted = [
        lambda r: any(t.startswith("@") and len(t) > 1 and "_" in t for t in r["tokens"]),
        lambda r: any(t.startswith("#") and len(t) > 1 for t in r["tokens"]),
        lambda r: any(t.lower().startswith("http") for t in r["tokens"]),
    ]
    picked: list[dict[str, Any]] = []
    for cond in wanted[:k]:
        for r in records:
            if cond(r) and r not in picked and len(r["raw_tokens"]) <= 25:
                picked.append(r)
                break
    for r in picked:
        raw = " ".join(f"{t}/{g}" for t, g in zip(r["raw_tokens"], r["raw_tags"], strict=True))
        new = " ".join(f"{t}/{g}" for t, g in zip(r["tokens"], r["tags"], strict=True))
        log.info("example id=%s label=%s cmi=%.1f", r["id"], r["label"], r["cmi"])
        log.info("  raw     : %s", raw)
        log.info("  rejoined: %s", new)
        log.info("  text    : %s", r["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SentiMix CoNLL files into processed splits.")
    parser.add_argument("--config", default=None, help="config YAML (default configs/default.yaml)")
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    raw_dir = resolve_path(args.raw_dir or cfg["paths"]["raw_dir"])
    out_dir = resolve_path(args.out_dir or cfg["paths"]["processed_dir"])
    files = cfg["paths"]["raw_files"]
    repair = cfg.get("preprocessing", {}).get("fix_mojibake", True)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids: dict[str, set[str]] = {}
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        tweets = parse_conll(raw_dir / files[split])
        if split == "test" and any(t.label is None for t in tweets):
            attach_labels(tweets, read_test_labels(raw_dir / files["test_labels"]))
            log.info("test: joined labels from %s", files["test_labels"])
        records = build_records(tweets, repair_encoding=repair)
        check_split(split, records, cfg)
        log_summary(split, records)
        split_records[split] = records
        all_ids[split] = {r["id"] for r in records}

        csv_cols = ["id", "text", "label", "hin_ratio", "cmi", "n_tokens"]
        df = pd.DataFrame(records)[csv_cols].round({"hin_ratio": 4, "cmi": 4})
        df.to_csv(out_dir / f"{split}.csv", index=False)
        write_jsonl(records, out_dir / f"{split}.jsonl")
        log.info("%-5s wrote %s and %s", split, out_dir / f"{split}.csv", out_dir / f"{split}.jsonl")

    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not (all_ids[a] & all_ids[b]), f"{a} and {b} share ids"
    log.info("checks passed: sizes, label counts, unique ids, no empty texts, no cross-split ids")
    show_examples(split_records["train"])


if __name__ == "__main__":
    main()
