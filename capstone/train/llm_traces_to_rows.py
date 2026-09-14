#!/usr/bin/env python3
"""
Turn LLM-teacher traces into training rows — the edge that closes Path B to Path C.

This is the step the brief actually asks for: *"run a large model in simulation as a
teacher, capture the situation-to-action traces, and distil them into a small model that
fits the drone."* Until now the student was distilled from `OracleLabeller`, which is handed
the answer and cannot reason about a situation nobody anticipated. These rows come from a
teacher that saw exactly what the student sees and no ground truth.

Two things are deliberately NOT taken from the teacher:

  * the CAUSE label. ns-3 knows the true cause for free at every tick, so using the
    teacher's guess as a classification target would be strictly worse supervision.
    The teacher is needed for what the simulator cannot label -- WHICH TEST TO RUN and
    WHEN TO STOP INVESTIGATING.
  * any row where the teacher's parse failed or it abstained with no evidence. A trace is
    a record of what happened, including the failures; a training set is not.

    python3 -m train.llm_traces_to_rows --traces ../data/traces/llm_v7 --out ../data/traces_llm/train.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario.split import split_of, family_of   # noqa: E402

CAUSES = {"barrage", "spot", "reactive", "sweep", "fading",
          "node_loss", "congestion", "hidden_term"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="../data/traces/llm")
    ap.add_argument("--out", default="../data/traces_llm/train.jsonl")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "any"])
    ap.add_argument("--min-confidence", type=float, default=0.0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.traces, "*.jsonl")))
    if not files:
        print(f"no traces under {a.traces}")
        sys.exit(2)

    rows, skipped = [], collections.Counter()
    fams = collections.Counter()
    for p in files:
        episode = os.path.basename(p)[:-len(".jsonl")]
        fam = family_of(episode)
        if a.split != "any" and split_of(episode, fam) != a.split:
            skipped["wrong split"] += 1
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped["unparseable trace line"] += 1
                continue
            f = r.get("features") or []
            if not f:
                skipped["no features (trace predates the features field)"] += 1
                continue
            if r.get("parse_mode") == "failed" or r.get("error"):
                skipped["teacher failed to answer"] += 1
                continue
            call = r.get("call")
            if not call:
                skipped["no call"] += 1
                continue
            if float(r.get("confidence", 0.0)) < a.min_confidence:
                skipped["below --min-confidence"] += 1
                continue
            cause = fam if fam in CAUSES else "fading"
            rows.append({
                "scenario": episode, "family": fam,
                # cause: the simulator's free label, NOT the teacher's guess (see docstring)
                "cause": cause,
                "t": r.get("t_sim", 0.0),
                "features": f,
                # call: THIS is what the teacher is for
                "call": call,
                "args": r.get("args", {}),
                "available": r.get("available", []),
                "source": "llm",
                "teacher_belief_top": max(r.get("belief", {"?": 0}),
                                          key=r.get("belief", {"?": 0}).get),
                "teacher_confidence": r.get("confidence", 0.0),
            })
            fams[fam] += 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"traces read : {len(files)} files")
    print(f"rows written: {len(rows)}  -> {a.out}")
    print(f"by family   : {dict(sorted(fams.items()))}")
    if skipped:
        print("skipped     :")
        for k, v in skipped.most_common():
            print(f"    {v:6d}  {k}")
    # the teacher's own agreement with the free label, for the report
    agree = sum(1 for r in rows if r["teacher_belief_top"] == r["cause"])
    if rows:
        print(f"\nteacher agreed with the simulator's label on {agree}/{len(rows)} "
              f"= {agree/len(rows):.1%} of rows (it never saw the label)")


if __name__ == "__main__":
    main()
