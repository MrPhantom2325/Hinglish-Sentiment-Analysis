"""Download and verify the SemEval-2020 Task 9 (SentiMix) Hinglish files.

Downloads the four original files from a pinned Hugging Face commit into data/raw/,
checks their sha256 hashes, re-checks tweet and label counts against the published
statistics, joins the test labels, and writes data/raw/SOURCE.md.

Usage:
    python scripts/download_data.py            # download missing files, verify everything
    python scripts/download_data.py --force    # re-download even if files exist
    python scripts/download_data.py --verify-only

Uses only the standard library so it runs before `pip install`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import logging
import sys
import urllib.request
from collections import Counter
from pathlib import Path

log = logging.getLogger("download_data")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

HF_REPO = "RTT1/SentiMix"
HF_COMMIT = "205f0391fc1f10320ec3c10708eaa27e88db04c7"
BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_COMMIT}"

# split -> (original file name, sha256)
FILES: dict[str, tuple[str, str]] = {
    "train": (
        "train_14k_split_conll.txt",
        "9b1118ad0a3e46a909e0e380bfa289b8fc748f4e7fca24871129e3a22dd36b80",
    ),
    "dev": (
        "dev_3k_split_conll.txt",
        "e4dc583ed95e892669527b864aac01b86d990aee59d5fdc91a5e615e0f5f00a0",
    ),
    "test": (
        "Hindi_test_unalbelled_conll_updated.txt",
        "3bfb1c33d2429db79e33a3eab0823e2c3597eada2afbbed5070468c8f4262fcd",
    ),
    "test_labels": (
        "test_labels_hinglish.txt",
        "6ce8340f8c8b20e81cccb40265ebfcde00b4de0423651a0d7ebdd5a75fc884ed",
    ),
}

EXPECTED_LABELS: dict[str, dict[str, int]] = {
    "train": {"positive": 4634, "negative": 4102, "neutral": 5264},
    "dev": {"positive": 982, "negative": 890, "neutral": 1128},
    "test": {"positive": 1000, "negative": 900, "neutral": 1100},
}


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, expected_sha: str, force: bool) -> None:
    """Download one file to RAW_DIR unless it already exists with the right hash."""
    dest = RAW_DIR / name
    if dest.exists() and not force:
        if sha256(dest) == expected_sha:
            log.info("present, hash ok: %s", name)
            return
        log.warning("present but hash differs, re-downloading: %s", name)
    url = f"{BASE_URL}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        out.write(resp.read())
    got = sha256(tmp)
    if got != expected_sha:
        tmp.unlink()
        raise SystemExit(f"sha256 mismatch for {name}: got {got}, expected {expected_sha}")
    tmp.replace(dest)


def read_meta(path: Path) -> list[tuple[str, str | None]]:
    """Return (id, label-or-None) for every `meta` line in a CoNLL file."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0].strip().lower() == "meta":
                tid = parts[1].strip()
                label = parts[2].strip().lower() if len(parts) > 2 and parts[2].strip() else None
                rows.append((tid, label))
    return rows


def read_test_labels(path: Path) -> dict[str, str]:
    """Read the `Uid,Sentiment` CSV into {id: label}."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {r["Uid"].strip(): r["Sentiment"].strip().lower() for r in reader}


def verify() -> dict[str, dict[str, int]]:
    """Check hashes, counts, uniqueness and the test-label join. Returns label counts."""
    for split, (name, expected_sha) in FILES.items():
        got = sha256(RAW_DIR / name)
        assert got == expected_sha, f"{split}: sha256 {got} != {expected_sha}"
    log.info("all sha256 hashes match")

    metas = {s: read_meta(RAW_DIR / FILES[s][0]) for s in ("train", "dev", "test")}
    test_labels = read_test_labels(RAW_DIR / FILES["test_labels"][0])

    test_ids = [tid for tid, _ in metas["test"]]
    missing = [tid for tid in test_ids if tid not in test_labels]
    extra = set(test_labels) - set(test_ids)
    assert not missing, f"{len(missing)} test tweets have no label, e.g. {missing[:5]}"
    assert not extra, f"{len(extra)} labels have no test tweet, e.g. {sorted(extra)[:5]}"
    metas["test"] = [(tid, test_labels[tid]) for tid in test_ids]

    counts: dict[str, dict[str, int]] = {}
    for split, rows in metas.items():
        ids = [tid for tid, _ in rows]
        assert len(ids) == len(set(ids)), f"{split}: duplicate ids"
        c = Counter(label for _, label in rows)
        counts[split] = dict(c)
        log.info("%-5s n=%d %s", split, len(rows), dict(sorted(c.items())))
        assert c == Counter(EXPECTED_LABELS[split]), (
            f"{split}: label counts {dict(c)} != expected {EXPECTED_LABELS[split]}"
        )

    id_sets = {s: {tid for tid, _ in rows} for s, rows in metas.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = id_sets[a] & id_sets[b]
        assert not overlap, f"{a}/{b} share {len(overlap)} ids"
    log.info("no id overlap between splits; every test tweet has a label")
    return counts


def write_source_md(counts: dict[str, dict[str, int]]) -> None:
    """Write data/raw/SOURCE.md with URLs, hashes and verified counts."""
    today = dt.date.today().isoformat()
    lines = [
        "# SentiMix Hinglish — data provenance",
        "",
        "- Dataset: SemEval-2020 Task 9 (SentiMix), Hinglish track",
        f"- Source: Hugging Face dataset `{HF_REPO}`, pinned commit `{HF_COMMIT}`",
        f"- Repo page: https://huggingface.co/datasets/{HF_REPO}",
        f"- Verified on: {today} (by `scripts/download_data.py`)",
        "- Why this source: the official CodaLab pages (competitions.codalab.org/competitions/20654,",
        "  codalab.lisn.upsaclay.fr) refused connections. This mirror has the original file names",
        "  and matches the published split sizes and label counts exactly.",
        "",
        "## Files",
        "",
        "| split | original file name | URL | sha256 |",
        "|---|---|---|---|",
    ]
    for split, (name, digest) in FILES.items():
        lines.append(f"| {split} | `{name}` | {BASE_URL}/{name} | `{digest}` |")
    lines += [
        "",
        "## Verification counts",
        "",
        "| split | tweets | positive | negative | neutral |",
        "|---|---|---|---|---|",
    ]
    for split, c in counts.items():
        n = sum(c.values())
        lines.append(
            f"| {split} | {n} | {c.get('positive', 0)} | {c.get('negative', 0)} | {c.get('neutral', 0)} |"
        )
    lines += [
        "",
        "Test labels come from `test_labels_hinglish.txt` (CSV `Uid,Sentiment`), joined on tweet id;",
        "all 3000 test tweets received a label. No duplicate ids within a split, no overlap across splits.",
        "",
    ]
    (RAW_DIR / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", RAW_DIR / "SOURCE.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--verify-only", action="store_true", help="skip downloading")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        for name, digest in FILES.values():
            download(name, digest, args.force)
    counts = verify()
    write_source_md(counts)


if __name__ == "__main__":
    main()
