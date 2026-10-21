"""Tests for src/data.py: CoNLL parsing, rejoining, encoding repair and CMI."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.data import (
    attach_labels,
    build_records,
    compute_cmi,
    fix_mojibake,
    parse_conll,
    parse_conll_lines,
    read_test_labels,
    rejoin,
    tweet_features,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.conll"


@pytest.fixture(scope="module")
def tweets():
    return parse_conll(FIXTURE)


@pytest.fixture(scope="module")
def records(tweets):
    return {r["id"]: r for r in build_records(tweets)}


# --------------------------------------------------------------------------- parser


def test_parses_all_tweets_with_lowercase_labels(tweets):
    assert [t.id for t in tweets] == ["1", "2", "3", "4", "5", "6"]
    assert [t.label for t in tweets] == ["positive", "negative", "neutral", "positive", "negative", "neutral"]


def test_tags_are_uppercased(tweets):
    t2 = tweets[1]
    assert set(t2.tags) <= {"HIN", "ENG", "O", "EMT"}
    assert t2.tags[t2.tokens.index("decision")] == "ENG"  # was "eng"


def test_stray_whitespace_missing_tag_and_empty_token(tweets):
    t3 = tweets[2]
    # "  kya  \t HIN" is stripped; the whitespace-only token line is skipped;
    # "missingtag" has no tag and gets "O".
    assert t3.tokens == ["kya", "scene", "hai", "missingtag"]
    assert t3.tags == ["HIN", "ENG", "HIN", "O"]


def test_missing_blank_line_still_splits_tweets(tweets):
    t3, t4 = tweets[2], tweets[3]
    assert "@" not in t3.tokens
    assert t4.tokens[0] == "@" and t4.tokens[-1] == "yaar"


def test_unlabelled_meta_line():
    tw = parse_conll_lines(["meta\t99\n", "hello\tEng\n"])
    assert tw[0].id == "99" and tw[0].label is None


def test_meta_with_spaces_instead_of_tabs():
    tw = parse_conll_lines(["meta 7 Positive\n", "dost Hin\n"])
    assert (tw[0].id, tw[0].label, tw[0].tokens, tw[0].tags) == ("7", "positive", ["dost"], ["HIN"])


# --------------------------------------------------------------------------- test labels


def test_read_and_attach_test_labels(tmp_path):
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text("Uid,Sentiment\n1,Positive\n2,neutral", encoding="utf-8")  # no trailing newline
    labels = read_test_labels(labels_file)
    assert labels == {"1": "positive", "2": "neutral"}

    tw = parse_conll_lines(["meta\t1\n", "a\tEng\n", "meta\t2\n", "b\tEng\n"])
    attach_labels(tw, labels)
    assert [t.label for t in tw] == ["positive", "neutral"]


def test_attach_labels_fails_on_missing_id():
    tw = parse_conll_lines(["meta\t1\n", "a\tEng\n", "meta\t3\n", "b\tEng\n"])
    with pytest.raises(AssertionError):
        attach_labels(tw, {"1": "positive"})


# --------------------------------------------------------------------------- rejoin


def test_rejoin_mention_with_underscore(records):
    r = records["1"]
    assert r["tokens"][0] == "@abhisar_sharma"
    assert r["tags"][0] == "HIN"  # tag of the handle's first word


def test_rejoin_hashtag_and_url(records):
    r = records["2"]
    assert r["tokens"] == ["#videsh_mantri", "worst", "decision", "https//t.co/AbC123xYz"]
    assert r["tags"] == ["HIN", "ENG", "ENG", "O"]
    assert r["text"] == "#videsh_mantri worst decision https//t.co/AbC123xYz"


def test_handle_does_not_swallow_following_url(records):
    r = records["4"]
    assert r["tokens"] == ["@KylieD_", "https//tco/Yh9F71", "yaar"]
    assert r["tags"] == ["ENG", "O", "HIN"]


def test_rejoin_variants():
    toks = ["https", "//", "t", "co", "/", "Xy1", "…", "next"]
    out, tags = rejoin(toks, ["ENG", "O", "ENG", "ENG", "O", "HIN", "O", "ENG"])
    assert out == ["https//tco/Xy1…", "next"] and tags == ["O", "ENG"]

    out, _ = rejoin(["https", "…", "bas"], ["ENG", "O", "HIN"])
    assert out == ["https…", "bas"]

    # bare "@", a lone "#" and "http" used as a word are left alone
    out, _ = rejoin(["@", "…", "#", "!", "http", "hai"], ["O"] * 6)
    assert out == ["@", "…", "#", "!", "http", "hai"]


def test_rejoin_keeps_tokens_and_tags_aligned(records):
    for r in records.values():
        assert len(r["tokens"]) == len(r["tags"]) == r["n_tokens"]
        assert len(r["raw_tokens"]) == len(r["raw_tags"])


# --------------------------------------------------------------------------- encoding repair


@pytest.mark.parametrize(
    ("broken", "fixed"),
    [
        ("â€¦", "…"),
        ("â€™", "’"),
        ("ðŸ˜‚", "😂"),
        ("ðŸ™\x8f", "🙏"),  # contains a byte cp1252 leaves undefined
        ("bahut", "bahut"),  # normal text unchanged
        ("…", "…"),  # already-correct text unchanged
        ("😂", "😂"),
        ("Â", "Â"),  # cannot be decoded -> unchanged
    ],
)
def test_fix_mojibake(broken, fixed):
    assert fix_mojibake(broken) == fixed


def test_encoding_repair_in_records(records, tweets):
    r = records["6"]
    assert tweets[5].tokens == ["Ohho", "https", "â€¦", "ðŸ˜‚"]  # raw tokens untouched
    assert r["raw_tokens"] == ["Ohho", "https", "â€¦", "ðŸ˜‚"]
    assert r["tokens"] == ["Ohho", "https…", "😂"]


# --------------------------------------------------------------------------- CMI and features


def test_compute_cmi_values():
    assert compute_cmi(n_hin=3, n_eng=1, n=5, u=1) == pytest.approx(25.0)
    assert compute_cmi(n_hin=2, n_eng=2, n=4, u=0) == pytest.approx(50.0)
    assert compute_cmi(n_hin=4, n_eng=0, n=4, u=0) == 0.0  # monolingual
    assert compute_cmi(n_hin=0, n_eng=0, n=3, u=3) == 0.0  # only language-independent tokens


def test_features_on_fixture(records):
    # tweet 1: @abhisar_sharma/HIN bahut/HIN accha/HIN movie/ENG hai/HIN 😂/EMT
    r = records["1"]
    assert (r["n_tokens"], r["n_hin"], r["n_eng"], r["n_other"]) == (6, 4, 1, 1)
    assert r["hin_ratio"] == pytest.approx(0.8)
    assert r["cmi"] == pytest.approx(20.0)

    r = records["3"]  # kya/HIN scene/ENG hai/HIN missingtag/O
    assert r["cmi"] == pytest.approx(100 / 3)
    assert r["hin_ratio"] == pytest.approx(2 / 3)

    assert records["4"]["cmi"] == pytest.approx(50.0)


def test_hin_ratio_nan_without_language_tokens(records):
    r = records["5"]  # "!!!"/O "😡"/EMT
    assert math.isnan(r["hin_ratio"])
    assert r["cmi"] == 0.0
    assert tweet_features([])["cmi"] == 0.0
