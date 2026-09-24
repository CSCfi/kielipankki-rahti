"""The writers must reproduce the layout of the production files captured
in test/reference/ (see test/capture_reference.py)."""

import os
import xml.etree.ElementTree as ET

import pytest

from asr import formats

WORDS = [
    {"start": 0.0, "end": 0.61, "label": "pohjantuuli"},
    {"start": 0.61, "end": 0.71, "label": "ja"},
    {"start": 0.71, "end": 1.33, "label": "aurinko"},
    {"start": 2.37, "end": 2.97, "label": "pohjantuuli"},
]
LETTERS = [
    {"start": 0.0, "end": 0.3, "label": "p"},
    {"start": 0.3, "end": 0.61, "label": "o"},
]
REFERENCE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "test", "reference")


def test_ctm_layout():
    assert formats.to_ctm(WORDS).splitlines()[:3] == [
        "0.0 0.61 pohjantuuli",
        "0.61 0.71 ja",
        "0.71 1.33 aurinko",
    ]


def test_ctm_matches_reference_style():
    path = os.path.join(REFERENCE, "align_pohjantuuli.ctm")
    if not os.path.exists(path):
        pytest.skip("test/reference not present")
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    assert first == "0.0 0.61 pohjantuuli\n"


def test_textgrid_layout_and_gap_filling():
    tg = formats.to_textgrid([("words", WORDS)], 3.0)
    lines = tg.splitlines()
    assert lines[:8] == [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0.000000",
        "xmax = 3.000000",
        "tiers? <exists>",
        "size = 1",
        "item []:",
    ]
    assert lines[8:14] == [
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "words"',
        "        xmin = 0.000000",
        "        xmax = 3.000000",
        "        intervals: size = 6",
    ]
    assert lines[14:18] == [
        "        intervals [1]:",
        "            xmin = 0.000000",
        "            xmax = 0.610000",
        '            text = "pohjantuuli"',
    ]
    # The pause between 1.33 and 2.37 and the tail up to xmax are empty.
    assert '            text = ""' in lines
    assert lines[-2] == "            xmax = 3.000000"
    assert tg.endswith('"\n')


def test_textgrid_reference_header():
    path = os.path.join(REFERENCE, "align_pohjantuuli.TextGrid")
    if not os.path.exists(path):
        pytest.skip("test/reference not present")
    with open(path, encoding="utf-8") as f:
        ref = f.read().splitlines()
    ours = formats.to_textgrid([("words", WORDS)], 36.67).splitlines()
    assert ours[:4] == ref[:4]
    assert ours[5:13] == ref[5:13]
    assert ours[14:18] == ref[14:18]


def test_two_tiers():
    tg = formats.to_textgrid([("words", WORDS), ("letters", LETTERS)], 3.0)
    assert "size = 2" in tg
    assert '        name = "letters"' in tg


def test_eaf_structure():
    eaf = formats.to_eaf([("words", WORDS), ("letters", LETTERS)])
    root = ET.fromstring(eaf.encode("utf-8"))
    tiers = [t.get("TIER_ID") for t in root.findall("TIER")]
    assert tiers == ["default", "words", "letters"]
    assert root.find("HEADER/MEDIA_DESCRIPTOR") is None
    values = [a.text for a in root.findall("TIER[@TIER_ID='words']//ANNOTATION_VALUE")]
    assert sorted(values) == sorted(w["label"] for w in WORDS)
    slots = {s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE")) for s in root.findall("TIME_ORDER/TIME_SLOT")}
    assert 610 in slots.values() and 2370 in slots.values()


def test_all_formats_keys():
    result = formats.all_formats({"words": WORDS, "letters": LETTERS}, 3.0)
    assert set(result) == {"ctm", "eaf", "TextGrid", "intervals"}
    assert result["intervals"]["words"] == WORDS
