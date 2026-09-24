"""Interval tiers as ctm, Praat TextGrid and ELAN eaf.

A tier is a list of ``{"start", "end", "label"}`` with times in seconds. The
Kaldi service wrote a single ``words`` tier; the writers here keep its exact
layout and add further tiers after it.
"""

import os
import tempfile

import pympi


def _num(x):
    return str(round(float(x), 3))


def to_ctm(words):
    """Three columns, ``start end label``, one line per word."""
    return "".join(f"{_num(w['start'])} {_num(w['end'])} {w['label']}\n" for w in words)


def _filled(intervals, xmax):
    """Intervals with the gaps filled by empty ones, covering [0, xmax]."""
    out = []
    pos = 0.0
    for iv in intervals:
        if iv["start"] > pos:
            out.append({"start": pos, "end": iv["start"], "label": ""})
        out.append(iv)
        pos = iv["end"]
    if xmax > pos:
        out.append({"start": pos, "end": xmax, "label": ""})
    return out


def to_textgrid(tiers, xmax):
    """Long-form TextGrid with one IntervalTier per (name, intervals)."""
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {0:.6f}",
        f"xmax = {xmax:.6f}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for n, (name, intervals) in enumerate(tiers, 1):
        filled = _filled(intervals, xmax)
        lines += [
            f"    item [{n}]:",
            '        class = "IntervalTier"',
            f'        name = "{name}"',
            f"        xmin = {0:.6f}",
            f"        xmax = {xmax:.6f}",
            f"        intervals: size = {len(filled)}",
        ]
        for i, iv in enumerate(filled, 1):
            text = iv["label"].replace('"', '""')
            lines += [
                f"        intervals [{i}]:",
                f"            xmin = {iv['start']:.6f}",
                f"            xmax = {iv['end']:.6f}",
                f'            text = "{text}"',
            ]
    return "\n".join(lines) + "\n"


def to_eaf(tiers):
    """ELAN file written by pympi, with its default empty tier first."""
    eaf = pympi.Elan.Eaf()
    for name, intervals in tiers:
        eaf.add_tier(name)
        for iv in intervals:
            start = int(round(iv["start"] * 1000))
            end = int(round(iv["end"] * 1000))
            if end <= start:
                end = start + 1
            eaf.add_annotation(name, start, end, iv["label"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.eaf")
        pympi.Elan.to_eaf(path, eaf)
        with open(path, encoding="utf-8") as f:
            return f.read()


def all_formats(tiers, xmax):
    """The alignment result: a dict with ``ctm``, ``eaf``, ``TextGrid`` and
    ``intervals``. ``tiers`` is an ordered dict of name to intervals whose
    first entry is the word tier."""
    items = list(tiers.items())
    return {
        "ctm": to_ctm(items[0][1]),
        "eaf": to_eaf(items),
        "TextGrid": to_textgrid(items, xmax),
        "intervals": dict(items),
    }
