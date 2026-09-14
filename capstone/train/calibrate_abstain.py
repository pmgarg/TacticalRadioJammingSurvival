"""Calibrate the conformal abstain threshold of an ALREADY-TRAINED bundle, and measure
whether abstention does the job the design claims for it.

WHY THIS IS A SEPARATE TOOL
Conformal calibration is post-hoc by construction: it reads the trained model's posteriors
on a held-out set and picks a cut. It changes no weight. Keeping it out of the training
script means the threshold can be re-derived, and re-argued, without a retrain -- and it
means the calibration set is an explicit choice rather than whatever `train_mixed.py`
happened to pool.

THE BUG THIS EXISTS TO FIX
v9 shipped `abstain_threshold = 0.0`, which the report read as "abstention is off". The
code was right and the calibration SET was wrong: `train_mixed.py` calibrates on
refsim-val + ns3-val pooled. refsim is the easy world (97.6% accurate); ns-3 is the world
the agent is actually scored in (88.4%). Pooled, the error at q=0 is 9.3% -- just under the
10% target -- so the smallest valid cut is 0.0 and abstention never fires. Calibrate on the
DEPLOYMENT distribution alone and the honest cut is 0.70.

THE HARDER RESULT
Run with --ood and this also reports the abstention rate on a family held out of training.
On v9 that number is the most useful thing in this file, and it is negative: see
docs/REVIEW.md section 3.1.
"""
from __future__ import annotations

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def forward(bundle: dict, head: str, X: np.ndarray) -> np.ndarray:
    h = X
    Ws = [np.asarray(w, np.float32) for w in bundle[head]["W"]]
    bs = [np.asarray(b, np.float32) for b in bundle[head]["b"]]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    e = np.exp(h - h.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cost_decision(bundle: dict, P: np.ndarray):
    """Minimum-expected-cost class and ITS posterior -- the exact pair agent/student.py
    thresholds on. Calibrating on max(posterior) instead would give a guarantee that does
    not transfer to the deployed rule (AUDIT F4.4)."""
    classes = bundle["cause"]["classes"]
    order = bundle["cost_order"]
    COST = np.asarray(bundle["cost_matrix"], float)
    idx = [order.index(c) for c in classes]
    ec = P @ COST[np.ix_(idx, idx)]
    k = ec.argmin(axis=1)
    return np.asarray(classes)[k], P[np.arange(len(P)), k]


def load(path: str):
    rows = [json.loads(l) for l in open(path)]
    X = np.asarray([r["features"] for r in rows], np.float32)
    y = np.asarray([r["cause"] for r in rows])
    fam = np.asarray([r.get("family", "?") for r in rows])
    return X, y, fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--calib", required=True,
                    help="validation rows from the DEPLOYMENT world (ns-3), not pooled")
    ap.add_argument("--ood", default=None,
                    help="test rows containing the held-out family, for the novelty check")
    ap.add_argument("--held-out-family", default="sweep")
    ap.add_argument("--target-err", type=float, default=0.10)
    ap.add_argument("--write", action="store_true",
                    help="patch abstain_threshold into the bundle in place")
    a = ap.parse_args()

    d = json.load(open(a.bundle))
    X, y, _ = load(a.calib)
    pred, conf = cost_decision(d, forward(d, "cause", X))
    correct = pred == y
    print(f"calibration set: {a.calib}  n={len(X)}  "
          f"min-cost accuracy={correct.mean():.4f}  err={1 - correct.mean():.4f}")

    thr = None
    print(f"\n  {'q':>5s} {'retained':>9s} {'err':>8s}")
    for q in np.linspace(0.0, 0.95, 96):
        keep = conf >= q
        if keep.sum() < 20:
            break
        err = 1.0 - correct[keep].mean()
        if q * 100 % 10 < 1e-6:
            print(f"  {q:5.2f} {100 * keep.mean():8.1f}% {err:8.4f}")
        if thr is None and err <= a.target_err:
            thr = float(q)
    if thr is None:
        thr = 0.5
        print("\n  no cut reached the target error; falling back to 0.50")
    keep = conf >= thr
    print(f"\n  CONFORMAL THRESHOLD = {thr:.2f}  "
          f"(retains {100 * keep.mean():.1f}% at {1 - correct[keep].mean():.4f} error, "
          f"target {a.target_err})")
    if thr == 0.0:
        print("  WARNING: a threshold of 0.00 means the model never abstains. That is a")
        print("  legitimate conformal answer when the calibration error is already under")
        print("  target -- but it also means the calibration set is easier than deployment.")
        print("  Check that --calib is the world the agent is actually scored in.")

    report = {"calib": a.calib, "n_calib": int(len(X)),
              "calib_accuracy": float(correct.mean()),
              "target_err": a.target_err, "threshold": thr,
              "retained_frac": float(keep.mean()),
              "retained_err": float(1 - correct[keep].mean())}

    # ---- the novelty check: does a low posterior actually mean "I have not seen this"? --
    if a.ood:
        Xo, yo, famo = load(a.ood)
        predo, confo = cost_decision(d, forward(d, "cause", Xo))
        corro = predo == yo
        held = famo == a.held_out_family
        seen = ~held
        print(f"\n  --- novelty check on {a.ood} ---")
        print(f"  {'family':12s} {'n':>6s} {'acc':>7s} {'mean p':>8s} {'abstains':>9s}")
        for f in sorted(set(famo)):
            m = famo == f
            mark = "   <== HELD OUT" if f == a.held_out_family else ""
            print(f"  {f:12s} {m.sum():6d} {corro[m].mean():7.3f} {confo[m].mean():8.3f} "
                  f"{100 * (confo[m] < thr).mean():8.1f}%{mark}")
        a_in = float((confo[seen] < thr).mean()) if seen.sum() else float("nan")
        a_out = float((confo[held] < thr).mean()) if held.sum() else float("nan")
        print(f"\n  in-distribution : accuracy {corro[seen].mean():.3f}, abstains {100*a_in:.1f}%")
        print(f"  held-out family : accuracy {corro[held].mean():.3f}, abstains {100*a_out:.1f}%")
        if held.sum() and a_out <= a_in:
            print("\n  RESULT: the abstention rate is NOT higher on the family the model has")
            print("  never seen. The softmax posterior carries no novelty signal -- the model")
            print("  is confidently wrong out of distribution. Conformal abstention controls")
            print("  IN-DISTRIBUTION risk (it does remove the worst of the ambiguous families)")
            print("  and nothing else. Detecting the unseen family is the induced rules' job,")
            print("  which is why agent/hybrid.py exists. Do not advertise abstention as an")
            print("  out-of-distribution guard.")
        report.update({"ood": a.ood, "abstain_in_dist": a_in, "abstain_held_out": a_out,
                       "acc_in_dist": float(corro[seen].mean()),
                       "acc_held_out": float(corro[held].mean()),
                       "detects_novelty": bool(held.sum() and a_out > a_in)})

    out = os.path.join(os.path.dirname(a.bundle), "abstain_calibration.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"\n  wrote {out}")

    if a.write:
        d["abstain_threshold"] = thr
        json.dump(d, open(a.bundle, "w"))
        print(f"  patched abstain_threshold={thr:.2f} into {a.bundle} (no weight changed)")


if __name__ == "__main__":
    main()
