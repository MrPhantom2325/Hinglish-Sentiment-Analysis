"""Tests for src/preprocess.py."""

from __future__ import annotations

import pytest

from src.preprocess import clean_for_svm, clean_for_transformer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("@abhisar_sharma bahut accha", "@user bahut accha"),
        ("dekho https//t.co/AbC123xYz yaar", "dekho http yaar"),
        ("link https//tco/Xy1… end", "link http end"),
        ("https… bas", "http bas"),
        ("#videsh_mantri zindabad", "videsh_mantri zindabad"),
        ("bahuuuut accha", "bahuut accha"),
        ("nooooo!!!!!", "noo!!"),
        ("year 2000 or 1111", "year 2000 or 1111"),  # digits are not shortened
        ("  lots   of\tspace \n", "lots of space"),
        ("so funny 😂", "so funny 😂"),  # emojis kept
        ("😂😂😂😂", "😂😂"),  # repeated emoji capped like any character
        ("nahi nai nhi nahin", "nahi nai nhi nahin"),  # spellings NOT normalized
        ("Bahut ACCHA", "Bahut ACCHA"),  # case kept for cased models
    ],
)
def test_clean_for_transformer(raw, expected):
    assert clean_for_transformer(raw) == expected


def test_clean_for_transformer_full_tweet():
    raw = "@user1 @Shining__star77 Ohho Very beautiful eyes yarrrrr #love https//t.co/RINOmqkPsZ"
    assert clean_for_transformer(raw) == "@user @user Ohho Very beautiful eyes yarr love http"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bahut ACCHA 😂", "bahut accha face_with_tears_of_joy"),
        ("@SomeOne #Modi https//t.co/x", "@user modi http"),
        ("NAHIIIII yaar 😡😡😡", "nahii yaar enraged_face enraged_face"),
        ("nahi nai nhi", "nahi nai nhi"),
    ],
)
def test_clean_for_svm(raw, expected):
    assert clean_for_svm(raw) == expected


def test_cleaning_is_idempotent():
    raw = "@a_b bahuuut #x https//t.co/y 😂😂😂"
    once = clean_for_transformer(raw)
    assert clean_for_transformer(once) == once
    assert clean_for_svm(clean_for_svm(raw)) == clean_for_svm(raw)
