"""The failure-analysis loop: why did the teacher get it wrong, and what is MISSING?

WHAT THIS IS FOR
Everything else in this project measures whether the agent was right. Nothing asked the
question that actually improves it: *when it was wrong, what evidence would it have needed?*
Without that, every improvement is a human noticing something by hand -- which is how the
reactive family was fixed (the event gate never consulted the TX-side statistics, so the
teacher was never even asked; fixing it took that family from 0 decisions per episode to
11-14, and the student from 0/3 to 2/3 on the live bridge). This makes that search
systematic.

THREE STAGES, and the first two cost nothing

  1. CONFUSE   For every wrong decision, record (true cause -> what it said). The teacher's
               own `why` string is kept, because it says which features it leaned on.

  2. SEPARATE  For each confused pair, compute which of the 56 features actually
               discriminates those two classes on the labelled corpus, by area under the
               ROC curve. AUC 0.5 means the feature is useless for that pair; 1.0 means it
               separates them perfectly. Then check whether the discriminating features
               were even IN the panel the teacher saw (agent/llm_teacher.PANEL is 23 of 56).
               This alone identifies missing evidence with no model calls at all.

  3. ASK       Optionally show the LLM its own failure plus the features that would have
               separated the classes, and ask it to classify the failure as:
                 missing_evidence   - the panel did not contain what was needed
                 unhelpful_evidence - it was there but did not discriminate
                 misread            - it was there, it discriminated, the reasoning erred
               Each verdict implies a different fix: add a feature or a diagnostic call,
               redesign the statistic, or change the prompt/rule layer.

OUTPUT
  data/failure_analysis.json   machine-readable, and
  a ranked table of FEATURES NOT IN THE PANEL that would have separated the most failures.
  That table is the input to the next corpus run -- see --emit-panel-patch.

    python3 train/analyse_failures.py --traces ../data/traces/llm_full \\
        --corpus ../data/traces_ns3/train.jsonl --top 12
    python3 train/analyse_failures.py ... --ask 40      # add the LLM verdicts
"""
from __future__ import annotations

import argparse, collections, glob, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
from percept.features import FEATURE_NAMES                    # noqa: E402
from agent.llm_teacher import PANEL                           # noqa: E402

PANEL_IDX = {idx for _n, idx, _d in PANEL}

# Bookkeeping, not evidence. These count what the AGENT has spent or where it is; in a
# passive corpus rollout they are constant within a family, so they "separate" classes at
# AUC 1.00 while carrying no physical information whatsoever. A student that learned them
# would be reading the scenario generator, not the radio. They are excluded by name, and
# the degenerate-variance guard below catches anything similar that is not on this list.
NOT_EVIDENCE = {"hops_used", "scans_used", "actions_used", "t_in_episode", "scan_age"}


def degenerate(pos: np.ndarray, neg: np.ndarray) -> bool:
    """True when a feature is constant within each class -- a label in disguise.

    Such a feature reports AUC 1.0 and means nothing. It is how a corpus artefact
    masquerades as the decisive statistic."""
    return float(pos.std()) < 1e-6 and float(neg.std()) < 1e-6


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC, and |2*AUC-1| is the separability. No sklearn needed."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r1 = ranks[: len(pos)].sum()
    return (r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_corpus(path: str):
    X, y = [], []
    for line in open(path):
        r = json.loads(line)
        if r.get("features"):
            X.append(r["features"]); y.append(r["cause"])
    return np.asarray(X, float), np.asarray(y)


def load_failures(trace_dir: str):
    """(true_cause, said, why) for every cleanly-parsed decision that was wrong."""
    fails, total = [], 0
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        fam = os.path.basename(f).rsplit("_", 1)[0]
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            b = r.get("belief") or {}
            if not b or r.get("parse_mode") != "clean":
                continue
            total += 1
            said = max(b, key=b.get)
            truth = r.get("cause_truth") or fam
            if said != truth:
                fails.append((truth, said, str(r.get("why", "")), r.get("features")))
    return fails, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="../data/traces/llm_full")
    ap.add_argument("--corpus", default="../data/traces_ns3/train.jsonl")
    ap.add_argument("--out", default="../data/failure_analysis.json")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-pair", type=int, default=25,
                    help="ignore confusions rarer than this")
    ap.add_argument("--ask", type=int, default=0,
                    help="ask the LLM to triage this many failures (0 = free mode)")
    ap.add_argument("--emit-panel-patch", action="store_true",
                    help="print a ready-to-paste PANEL addition for llm_teacher.py")
    a = ap.parse_args()

    # ---- 1. confuse -----------------------------------------------------------------
    fails, total = load_failures(a.traces)
    pairs = collections.Counter((t, s) for t, s, _w, _f in fails)
    print(f"decisions: {total}   wrong: {len(fails)} ({100*len(fails)/max(1,total):.1f}%)")
    print(f"\nTOP CONFUSIONS (true -> said)")
    for (t, s), c in pairs.most_common(12):
        print(f"  {t:<12} -> {s:<12} {c:>5}")

    # ---- 2. separate ----------------------------------------------------------------
    X, y = load_corpus(a.corpus)
    print(f"\ncorpus for separability: {len(X)} labelled ns-3 windows")
    report, missing_score = {}, collections.Counter()
    print(f"\n{'confusion':<28}{'best features that DO separate them (AUC)':<52}")
    for (t, s), c in pairs.most_common():
        if c < a.min_pair:
            continue
        pos, neg = X[y == t], X[y == s]
        if len(pos) < 10 or len(neg) < 10:
            continue
        seps = []
        for i in range(X.shape[1]):
            if FEATURE_NAMES[i] in NOT_EVIDENCE or degenerate(pos[:, i], neg[:, i]):
                continue
            sep = abs(2 * auc(pos[:, i], neg[:, i]) - 1)
            seps.append((sep, i))
        seps.sort(reverse=True)
        best = seps[: a.top]
        shown = ", ".join(
            f"{FEATURE_NAMES[i]}{'' if i in PANEL_IDX else '*'}={s_:.2f}"
            for s_, i in best[:4])
        print(f"  {t[:11]:<11}->{s[:11]:<12} {shown}")
        # a feature that separates well but is NOT in the panel is missing evidence
        for sep, i in best:
            if i not in PANEL_IDX and sep >= 0.30:
                missing_score[i] += c * sep
        report[f"{t}->{s}"] = {
            "count": c,
            "separating": [{"feature": FEATURE_NAMES[i], "auc_sep": round(sp, 3),
                            "in_panel": i in PANEL_IDX} for sp, i in best],
        }
    print("\n  (* = the feature is NOT in the 23-row panel the teacher sees)")

    # ---- the headline: what should the panel gain? ----------------------------------
    print(f"\n{'='*70}\nMISSING EVIDENCE -- separates failures but is NOT shown to the teacher")
    print(f"{'='*70}")
    if not missing_score:
        print("  none: every discriminating feature is already in the panel.")
        print("  Then the failures are reasoning, not evidence -- fix the prompt or rules.")
    for i, sc in missing_score.most_common(a.top):
        print(f"  {FEATURE_NAMES[i]:<26} impact={sc:>8.0f}   (feature index {i})")

    if a.emit_panel_patch and missing_score:
        print(f"\n--- paste into agent/llm_teacher.py PANEL ---")
        for i, _sc in missing_score.most_common(a.top):
            print(f'    ("{FEATURE_NAMES[i]}", {i}, "TODO: one-line meaning"),')

    # ---- 3. ask (optional) -----------------------------------------------------------
    verdicts = {}
    if a.ask:
        from gateway.provider import make_provider, extract_json, LlmError
        prov = make_provider("claude")
        buckets = collections.Counter()
        picked = fails[: a.ask]
        print(f"\nasking the model to triage {len(picked)} failures ...")
        for t, s, why, _f in picked:
            key = f"{t}->{s}"
            sep = report.get(key, {}).get("separating", [])
            ev = ", ".join(f"{d['feature']}({'shown' if d['in_panel'] else 'NOT SHOWN'})"
                           for d in sep[:5])
            q = (f"You diagnosed a failing radio link as '{s}'. The truth was '{t}'.\n"
                 f"Your stated reasoning was: {why[:300]}\n\n"
                 f"On the labelled corpus, the features that best separate '{t}' from "
                 f"'{s}' are: {ev}\n\n"
                 "Classify YOUR failure as exactly one of: missing_evidence (the panel did "
                 "not contain what was needed), unhelpful_evidence (it was shown but does "
                 "not discriminate), misread (it was shown and discriminates, you erred).\n"
                 'Reply with ONLY {"verdict":"...","fix":"<one concrete sentence>"}')
            try:
                d = extract_json(prov.complete(q))
                v = str(d.get("verdict", "?"))
                buckets[v] += 1
                verdicts.setdefault(key, []).append({"verdict": v, "fix": d.get("fix")})
            except (LlmError, ValueError, json.JSONDecodeError):
                buckets["unparseable"] += 1
        print("\nTRIAGE")
        for k, v in buckets.most_common():
            print(f"  {k:<20} {v:>4}")

    json.dump({"decisions": total, "wrong": len(fails),
               "confusions": {f"{t}->{s}": c for (t, s), c in pairs.most_common()},
               "separability": report,
               "missing_evidence": [{"feature": FEATURE_NAMES[i], "impact": round(sc, 1),
                                     "index": i} for i, sc in missing_score.most_common()],
               "triage": verdicts},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
