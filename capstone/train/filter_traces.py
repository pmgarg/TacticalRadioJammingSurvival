#!/usr/bin/env python3
"""
Drop leaked rows from existing trace files, so the contamination can be fixed WITHOUT
re-running ns-3 (AUDIT F4.1).

Every trace row carries the `scenario` it came from, and `scenario/split.py` decides the
split from the name alone. So a contaminated corpus is repairable by filtering: any row
whose scenario belongs to the test split is removed from train and val. 540 ns-3 episodes
cost about six hours to regenerate; this costs seconds and produces exactly the same
training set that a clean generation run would have.

    python3 -m train.filter_traces                       # report only, changes nothing
    python3 -m train.filter_traces --write               # write *.clean.jsonl beside them
    python3 -m train.filter_traces --write --in-place    # overwrite (keeps a .leaked backup)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario.split import split_of, family_of, held_out_family   # noqa: E402

DEFAULT = ["../data/traces_ns3/train.jsonl", "../data/traces_ns3/val.jsonl",
           "../data/traces_all/train.jsonl", "../data/traces_all/val.jsonl",
           "../data/traces_bridge/train.jsonl"]


def scan(path: str):
    keep, drop = [], []
    fams = collections.Counter()
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(r.get("scenario", "")).split("#")[0]
        fam = r.get("family") or family_of(name)
        if name and split_of(name, fam) == "test":
            drop.append(line)
            fams[fam] += 1
        else:
            keep.append(line)
    return keep, drop, fams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--in-place", action="store_true")
    a = ap.parse_args()

    files = a.files or [p for p in DEFAULT if os.path.exists(p)]
    print(f"held-out family: {held_out_family()!r}\n")
    total_drop = 0
    for p in files:
        keep, drop, fams = scan(p)
        total_drop += len(drop)
        pct = 100.0 * len(drop) / max(1, len(keep) + len(drop))
        flag = "LEAK" if drop else "ok  "
        print(f"[{flag}] {p:38s} keep={len(keep):6d}  drop={len(drop):5d} ({pct:5.1f}%)"
              + (f"  {dict(fams)}" if drop else ""))
        if drop and a.write:
            out = p if a.in_place else p.replace(".jsonl", ".clean.jsonl")
            if a.in_place:
                os.replace(p, p.replace(".jsonl", ".leaked.jsonl"))
            with open(out, "w") as fh:
                fh.write("\n".join(keep) + ("\n" if keep else ""))
            print(f"         -> wrote {out}"
                  + (f"  (original kept as {p.replace('.jsonl', '.leaked.jsonl')})"
                     if a.in_place else ""))

    print(f"\ntotal rows dropped: {total_drop}")
    if total_drop and not a.write:
        print("re-run with --write (add --in-place to replace the originals)")
    if not total_drop:
        print("no leakage: every training row is outside the test split")


if __name__ == "__main__":
    main()
