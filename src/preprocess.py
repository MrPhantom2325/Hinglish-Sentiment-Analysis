"""Text cleaning for the transformer models and the TF-IDF + SVM baseline.

Romanized Hindi spellings (nahi / nai / nhi) are deliberately NOT normalized:
spelling variation is one of the things this project measures.
"""

from __future__ import annotations

import re

import emoji

# The rejoined URLs look like "https//t.co/AbC123" (the colon is missing in the raw data).
URL_RE = re.compile(r"\b(?:https?|www\.)\S*", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
# Any non-digit character repeated 3+ times -> 2 ("bahuuuut" -> "bahuut", "!!!!" -> "!!").
# Digits are excluded so numbers such as "1000" are not changed.
ELONGATION_RE = re.compile(r"(\D)\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")


def clean_for_transformer(text: str) -> str:
    """Mentions -> '@user', URLs -> 'http', drop '#', cap elongation at 2, normalize spaces. Keeps emojis."""
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = ELONGATION_RE.sub(r"\1\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_for_svm(text: str) -> str:
    """`clean_for_transformer`, then emojis -> words (e.g. 'face_with_tears_of_joy') and lowercase."""
    text = clean_for_transformer(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    return WHITESPACE_RE.sub(" ", text).strip().lower()
