"""
The canonical train / val / test split. One function, one definition, no second opinion.

AUDIT F4.1: the project had TWO corpora. `corpus/` held the `sweep` family out entirely;
`corpus_all/` did not. Training defaulted to `corpus_all` and evaluation defaulted to
`corpus`, so 50 of the 125 scenarios in `corpus/test` -- every one of them sweep -- were
in the cause head's training or validation data. That silently inverted the project's own
headline generalisation claim: `sweep` was reported as "the held-out family, never trained
on, now 90%", when it had 1,742 training rows.

The split is a pure function of the scenario NAME, so it is identical everywhere it is
computed and cannot drift between two call sites. Nothing may override it except the
documented env var, which exists only so the "what if we DO train on sweep?" ablation can
be run deliberately -- and any artefact produced that way is stamped, not silently mixed in.
"""
from __future__ import annotations

import hashlib
import os

HELD_OUT_FAMILY = "sweep"          # the novel-attack generalisation claim rests on this
TRAIN_PCT, VAL_PCT = 70, 85        # hash buckets: <70 train, <85 val, else test


def family_of(name: str) -> str:
    """`sweep_10203` -> `sweep`; `hidden_term_4` -> `hidden_term`."""
    stem = name.rsplit("#", 1)[0]
    parts = stem.split("_")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) or stem


def held_out_family() -> str:
    """Empty string disables the family holdout -- for the deliberate ablation only."""
    return os.environ.get("HOLD_OUT_FAMILY", HELD_OUT_FAMILY)


def split_of(name: str, family: str | None = None) -> str:
    """train | val | test for a scenario, from its name alone."""
    fam = family or family_of(name)
    hold = held_out_family()
    if hold and fam == hold:
        return "test"
    r = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 100
    return "train" if r < TRAIN_PCT else "val" if r < VAL_PCT else "test"


def is_trainable(name: str, family: str | None = None) -> bool:
    return split_of(name, family) != "test"


def assert_no_leak(train_names, test_names, where: str = "") -> None:
    """Raise rather than report a contaminated number. This is a build failure."""
    bad = sorted(set(train_names) & set(test_names))
    if bad:
        raise AssertionError(
            f"LEAKAGE{' in ' + where if where else ''}: {len(bad)} scenario(s) appear in "
            f"both training and test. First few: {bad[:5]}. "
            f"Run `python3 -m train.filter_traces` to clean existing trace files."
        )
