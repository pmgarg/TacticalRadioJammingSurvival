"""Strict decision parsing.

Design commitment: NO hidden retries. One repair attempt on malformed JSON, recorded as
such, then abstain. An abstention is an outcome we score, not an error we hide.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class ParseOutcome:
    ok: bool
    data: dict
    mode: str          # "clean" | "repaired" | "failed"
    detail: str = ""


def _first_object(text: str) -> str | None:
    """Brace-balanced scan. rfind('}') breaks on trailing prose containing braces."""
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def parse_decision(raw: str) -> ParseOutcome:
    t = (raw or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    blob = _first_object(t)
    if blob is None:
        return ParseOutcome(False, {}, "failed", "no JSON object in reply")
    try:
        return ParseOutcome(True, json.loads(blob), "clean")
    except json.JSONDecodeError as e:
        repaired = _repair(blob)
        if repaired is not None:
            try:
                return ParseOutcome(True, json.loads(repaired), "repaired", str(e))
            except json.JSONDecodeError:
                pass
        return ParseOutcome(False, {}, "failed", f"unparseable: {e}")


def _repair(blob: str) -> str | None:
    """The three malformations a model actually produces, and nothing speculative:
    trailing commas, single quotes around keys, and bare NaN/Infinity."""
    s = re.sub(r",(\s*[}\]])", r"\1", blob)
    s = re.sub(r"'([A-Za-z_][A-Za-z_0-9]*)'\s*:", r'"\1":', s)
    s = re.sub(r"\b(NaN|Infinity|-Infinity)\b", "0.0", s)
    return s if s != blob else None
