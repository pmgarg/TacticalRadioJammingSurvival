"""
Train the final student on refsim + authoritative ns-3 features.

The cause head learns from BOTH simulators, so it cannot come to depend on a feature
that only refsim produces (the S1 trap documented in FIDELITY.md). ns-3 states carry no
teacher action, so they are relabelled here by running the privileged teacher over each
ns-3 feature vector -- the teacher is a function of (features, true cause), and the true
cause is known for every corpus scenario.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, yaml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from agent.teacher import OracleLabeller
from train.train_student import conformal_threshold
from agent.api import Context

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))
CAUSE_ORDER = COSTCFG["order"]
COST = np.array([[float(COSTCFG["matrix"][t][i]) for i in range(len(CAUSE_ORDER))]
                 for t in CAUSE_ORDER])
ALL_CALLS = ["no_op", "spectrum_scan", "silent_listen", "neighbor_probe", "load_test",
             "mobility_test", "hop_channel", "set_tx_power", "change_tdma_slot",
             "reroute", "fallback_to_lora", "declare_link_lost"]


def load(path, tag):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            r["source"] = r.get("source", tag)
            rows.append(r)
    return rows


def relabel_calls(rows):
    """Give ns-3 states a teacher action (they arrive with a placeholder)."""
    out = 0
    by_scn = {}
    for r in rows:
        by_scn.setdefault(r["scenario"], []).append(r)
    for scn, rs in by_scn.items():
        rs.sort(key=lambda x: x["t"])
        t = OracleLabeller(rs[0]["cause"], recoverable=("refusal" not in scn))
        t.reset()
        for r in rs:
            ctx = Context(t=r["t"], channel=6, n_channels=8, peers=[], budget={},
                          available=ALL_CALLS, last_scan=None)
            r["call"] = t.decide(r["features"], ctx).call
            out += 1
    return out


def expected_cost_decision(proba, classes):
    idx = [CAUSE_ORDER.index(c) for c in classes]
    P = np.zeros((proba.shape[0], len(CAUSE_ORDER)))
    P[:, idx] = proba
    return np.array([CAUSE_ORDER[i] for i in (P @ COST).argmin(axis=1)])


def mean_cost(pred, true):
    return float(np.mean([COST[CAUSE_ORDER.index(t), CAUSE_ORDER.index(p)]
                          for t, p in zip(true, pred)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refsim", default="../data/traces_all")
    ap.add_argument("--ns3", default="../data/traces_ns3")
    ap.add_argument("--out", default="../data/student_mixed")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bridge", default="../data/traces_bridge/train.jsonl")
    ap.add_argument("--llm", default="../data/traces_llm/train.jsonl",
                    help="LLM-teacher rows (train/llm_traces_to_rows.py). These supply the "
                         "CALL targets the simulator cannot label -- which test to run and "
                         "when to stop investigating. DESIGN v2.0 section 9.5 step 2.")
    ap.add_argument("--llm-weight", type=int, default=3,
                    help="LLM rows are few and expensive; upweight them like bridge rows")
    ap.add_argument("--bridge-weight", type=int, default=4,
                    help="repeat live ns-3 on-policy states this many times")
    a = ap.parse_args()

    tr = load(os.path.join(a.refsim, "train.jsonl"), "refsim")
    # AUDIT F4.7: `_r1/train.jsonl` is BY CONSTRUCTION `base + round-1 rows` (see
    # train/dagger.py), so `tr + d1` contained every base row twice and silently
    # doubled the weight of the refsim distribution. Keep only what round 1 ADDED.
    d1_all = load(os.path.join(a.refsim, "_r1", "train.jsonl"), "dagger")
    seen = {json.dumps(r, sort_keys=True) for r in tr}
    d1 = [r for r in d1_all if json.dumps(r, sort_keys=True) not in seen]
    if d1_all:
        print(f"  dagger round 1: {len(d1_all)} rows, {len(d1)} of them new "
              f"({len(d1_all) - len(d1)} were duplicates of the base corpus)")
    n3 = load(os.path.join(a.ns3, "train.jsonl"), "ns3")
    if n3:
        relabel_calls(n3)
    va = load(os.path.join(a.refsim, "val.jsonl"), "refsim")
    va3 = load(os.path.join(a.ns3, "val.jsonl"), "ns3")
    if va3:
        relabel_calls(va3)

    br = load(a.bridge, "bridge")
    lm = load(a.llm, "llm")
    # On-policy states from LIVE ns-3 are few but they are the only ones drawn from the
    # distribution the agent actually visits at deployment. Upweight them.
    train = tr + d1 + n3 + br * a.bridge_weight + lm * a.llm_weight
    val = va + va3
    print(f"train rows: refsim={len(tr)} dagger={len(d1)} ns3={len(n3)} "
          f"bridge={len(br)}x{a.bridge_weight} llm={len(lm)}x{a.llm_weight}  "
          f"total={len(train)}")
    if not lm:
        print("  NOTE: no LLM-teacher rows. The student is being distilled from the ORACLE "
              "only, which is the gap DESIGN v2.0 section 9.5 names as the thesis.")
    print(f"val   rows: refsim={len(va)} ns3={len(va3)}  total={len(val)}")

    X = np.array([r["features"] for r in train], np.float32)
    yc = np.array([r["cause"] for r in train])
    ya = np.array([r["call"] for r in train])
    Xv = np.array([r["features"] for r in val], np.float32)
    yvc = np.array([r["cause"] for r in val])
    yva = np.array([r["call"] for r in val])
    srcv = np.array([r["source"] for r in val])

    lec = LabelEncoder().fit(np.concatenate([yc, yvc]))
    cause = MLPClassifier(hidden_layer_sizes=(a.hidden, a.hidden), max_iter=800,
                          alpha=1e-3, random_state=0, early_stopping=True,
                          n_iter_no_change=30)
    t0 = time.time()
    cause.fit(X, lec.transform(yc))
    classes = list(lec.inverse_transform(cause.classes_))
    pv = cause.predict_proba(Xv)
    ec = expected_cost_decision(pv, classes)
    print(f"\ncause head ({time.time()-t0:.0f}s)  overall acc={np.mean(ec==yvc):.3f} "
          f"cost={mean_cost(ec, yvc):.3f}")
    for s in sorted(set(srcv)):
        m = srcv == s
        if m.sum():
            print(f"    on {s:7s}: acc={np.mean(ec[m]==yvc[m]):.3f} "
                  f"cost={mean_cost(ec[m], yvc[m]):.3f}  (n={m.sum()})")
    fad = yvc == "fading"
    if fad.sum():
        jam = {"barrage", "spot", "reactive", "sweep"}
        print(f"    FP fading->jamming: {np.mean([p in jam for p in ec[fad]]):.4f}")

    lea = LabelEncoder().fit(np.concatenate([ya, yva]))
    call = MLPClassifier(hidden_layer_sizes=(a.hidden,), max_iter=800, alpha=1e-3,
                         random_state=0, early_stopping=True, n_iter_no_change=30)
    call.fit(X, lea.transform(ya))
    print(f"call head   acc={call.score(Xv, lea.transform(yva)):.3f}")

    os.makedirs(a.out, exist_ok=True)
    def dump(m, enc):
        return {"classes": [str(c) for c in enc.inverse_transform(m.classes_)],
                "W": [w.astype(np.float32).tolist() for w in m.coefs_],
                "b": [b.astype(np.float32).tolist() for b in m.intercepts_]}
    n = sum(w.size for w in cause.coefs_) + sum(b.size for b in cause.intercepts_) \
        + sum(w.size for w in call.coefs_) + sum(b.size for b in call.intercepts_)
    # AUDIT F4.4 / F4: the version was hardcoded to 1.0.0 while the contract was 1.1.0,
    # which defeats the version check the field exists for; and the abstain threshold was
    # hardcoded to 0.0, so the calibrated abstention the design describes never fired.
    _contract = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
    _abstain = (conformal_threshold(pv, yvc, classes, target_err=0.10)
                if len(Xv) else 0.0)
    print(f"  abstain threshold (conformal, on the min-cost statistic): {_abstain:.3f}")
    json.dump({"contract_version": _contract["contract_version"],
               "n_features": int(X.shape[1]),
               "cause": dump(cause, lec), "call": dump(call, lea),
               "cost_order": CAUSE_ORDER, "cost_matrix": COST.tolist(),
               "abstain_threshold": float(_abstain)},
              open(os.path.join(a.out, "student_bundle.json"), "w"))
    print(f"\nparameters={n:,}  fp32={n*4/1024:.1f} KB  int8={n/1024:.1f} KB  "
          f"(budget 512 KB) -> {'FITS' if n*4/1024 < 512 else 'TOO BIG'}")


if __name__ == "__main__":
    main()
