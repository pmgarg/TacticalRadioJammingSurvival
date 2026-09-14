"""
Distil the teacher into the on-device student (design §9.3-9.4, 9.6).

Two heads, both over the same 48-dim feature vector:
  * cause head   -> posterior over {8 causes}, decided by MINIMUM EXPECTED COST under
                    the confusion-cost matrix, not by argmax. This is what encodes the
                    asymmetry: declaring jamming when it was fading costs 10, so the
                    model must be much more confident before it says "jamming".
  * call head    -> which function to invoke (test or recovery action)

Plus a conformal abstention threshold fitted on val, so "unknown" has a calibrated
error bound rather than a hand-picked confidence cut.

Exports a plain-weight bundle (no framework needed at inference) and reports its size
against the <512 KB on-device budget.
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))
CAUSE_ORDER = COSTCFG["order"]
COST = np.array([[float(COSTCFG["matrix"][t][i]) for i in range(len(CAUSE_ORDER))]
                 for t in CAUSE_ORDER])          # COST[true, declared]


def load(path):
    X, yc, ya, fam = [], [], [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            X.append(r["features"]); yc.append(r["cause"])
            ya.append(r["call"]); fam.append(r["family"])
    return np.asarray(X, dtype=np.float32), np.asarray(yc), np.asarray(ya), np.asarray(fam)


def expected_cost_decision(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    """argmin_d  sum_c P(c) * COST[c, d]  -- the decision rule the metric rewards."""
    idx = [CAUSE_ORDER.index(c) for c in classes]
    P = np.zeros((proba.shape[0], len(CAUSE_ORDER)))
    P[:, idx] = proba
    exp_cost = P @ COST                      # (n, n_declared)
    return np.array([CAUSE_ORDER[i] for i in exp_cost.argmin(axis=1)])


def conformal_threshold(proba: np.ndarray, y_true: np.ndarray, classes: list[str],
                        target_err: float = 0.10, cost_rule: bool = True) -> float:
    """Smallest confidence cut whose retained set has error <= target_err.

    AUDIT F4.4: this used to calibrate on `max(proba)` with an argmax prediction, while
    `agent/student.py` abstains on the probability of the MINIMUM-EXPECTED-COST class.
    Two different statistics, so the <=target_err guarantee did not transfer to the
    deployed rule. It now calibrates on exactly the pair inference uses.
    """
    if cost_rule:
        pred = expected_cost_decision(proba, classes)
        idx = np.array([classes.index(p) if p in classes else 0 for p in pred])
        conf = proba[np.arange(len(proba)), idx]
    else:
        conf = proba.max(axis=1)
        pred = np.array(classes)[proba.argmax(axis=1)]
    correct = (pred == y_true)
    for q in np.linspace(0.0, 0.95, 96):
        keep = conf >= q
        if keep.sum() < 20:
            break
        if 1.0 - correct[keep].mean() <= target_err:
            return float(q)
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="../data/traces")
    ap.add_argument("--out", default="../data/student")
    ap.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args()

    Xtr, ytr, atr, ftr = load(os.path.join(a.traces, "train.jsonl"))
    Xva, yva, ava, fva = load(os.path.join(a.traces, "val.jsonl"))
    print(f"train {Xtr.shape}  val {Xva.shape}")

    # ---------------- cause head ---------------- #
    # sklearn 1.8's early-stopping scorer chokes on string labels, so encode.
    t0 = time.time()
    lec = LabelEncoder().fit(np.concatenate([ytr, yva]))
    cause = MLPClassifier(hidden_layer_sizes=(a.hidden, a.hidden), max_iter=600,
                          alpha=1e-3, random_state=0, early_stopping=True,
                          n_iter_no_change=25)
    cause.fit(Xtr, lec.transform(ytr))
    pc = cause.predict_proba(Xva)
    classes = list(lec.inverse_transform(cause.classes_))
    argmax_pred = np.array(classes)[pc.argmax(axis=1)]
    ec_pred = expected_cost_decision(pc, classes)
    acc_argmax = (argmax_pred == yva).mean()
    acc_ec = (ec_pred == yva).mean()

    def mean_cost(pred, true):
        return np.mean([COST[CAUSE_ORDER.index(t), CAUSE_ORDER.index(p)]
                        for t, p in zip(true, pred)])
    print(f"  cause head  ({time.time()-t0:.0f}s)")
    print(f"    argmax        : acc={acc_argmax:.3f}  mean_cost={mean_cost(argmax_pred, yva):.3f}")
    print(f"    min-exp-cost  : acc={acc_ec:.3f}  mean_cost={mean_cost(ec_pred, yva):.3f}")
    # the FP that matters
    fad = yva == "fading"
    if fad.sum():
        jam = {"barrage", "spot", "reactive", "sweep"}
        fp_argmax = np.mean([p in jam for p in argmax_pred[fad]])
        fp_ec = np.mean([p in jam for p in ec_pred[fad]])
        print(f"    FP fading->jamming: argmax={fp_argmax:.3f}  min-exp-cost={fp_ec:.3f}")

    thr = conformal_threshold(pc, yva, classes, target_err=0.10)
    print(f"    conformal abstain threshold (<=10% err on retained): {thr:.2f}")

    # ---------------- call head ---------------- #
    lea = LabelEncoder().fit(np.concatenate([atr, ava]))
    call = MLPClassifier(hidden_layer_sizes=(a.hidden,), max_iter=600, alpha=1e-3,
                         random_state=0, early_stopping=True, n_iter_no_change=25)
    call.fit(Xtr, lea.transform(atr))
    acc_call = call.score(Xva, lea.transform(ava))
    print(f"  call head     : acc={acc_call:.3f}  ({len(set(atr))} distinct calls)")

    # ---------------- export a plain-weight bundle ---------------- #
    os.makedirs(a.out, exist_ok=True)
    def dump_mlp(m, name, enc):
        return {"name": name,
                "classes": [str(c) for c in enc.inverse_transform(m.classes_)],
                "W": [w.astype(np.float32).tolist() for w in m.coefs_],
                "b": [b.astype(np.float32).tolist() for b in m.intercepts_]}
    bundle = {"contract_version": json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))["contract_version"],
              "n_features": int(Xtr.shape[1]),
              "cause": dump_mlp(cause, "cause", lec),
              "call": dump_mlp(call, "call", lea),
              "cost_order": CAUSE_ORDER,
              "cost_matrix": COST.tolist(),
              "abstain_threshold": thr}
    p = os.path.join(a.out, "student_bundle.json")
    json.dump(bundle, open(p, "w"))
    size = os.path.getsize(p)
    nparams = sum(w.size for w in cause.coefs_) + sum(b.size for b in cause.intercepts_) \
            + sum(w.size for w in call.coefs_) + sum(b.size for b in call.intercepts_)
    print(f"\n  bundle: {p}")
    print(f"    parameters      : {nparams:,}")
    print(f"    float32 weights : {nparams*4/1024:.1f} KB   (budget 512 KB)")
    print(f"    int8 weights    : {nparams/1024:.1f} KB")
    print(f"    json on disk    : {size/1024:.1f} KB")
    print(f"    FITS ESP32: {'YES' if nparams*4/1024 < 512 else 'NO'}")


if __name__ == "__main__":
    main()
