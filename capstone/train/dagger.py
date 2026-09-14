"""
DAgger (design §9.3, item 3).

Behaviour cloning trains on states the TEACHER visited. In deployment the student
visits states the teacher never did -- because it made a slightly different early
choice -- and the error compounds. The symptom is exactly what we measured: the
student classifies better than the baseline yet SURVIVES less, because a small action
mistake early puts it somewhere its training set never covered.

Fix: roll the student out, ask the teacher what it would have done at the states the
student actually reached, add those to the dataset, retrain.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from agent.controller import Controller
from agent.student import StudentAgent
from agent.teacher import TeacherAgent
from agent.api import Context
from scenario.schema import load_scenario
from sim.refsim import RefSim
from verify.verifier import score_episode

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def dagger_round(files, bundle, uncertainty_only=True):
    """Roll out the student; relabel every state it reached with the teacher."""
    stu = StudentAgent(bundle)
    rows, n_relabel, n_ep = [], 0, 0
    for fp in files:
        sc = load_scenario(fp)
        ctrl = Controller(RefSim(sc), sc, stu, CONTRACT)
        log = ctrl.run()
        truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
                 "recoverable": sc.truth.recoverable}
        s = score_episode(log.to_dict(), truth, COSTCFG, CONTRACT, sc.family)
        episode_failed = (not s.classification_ok) or (not s.survived) \
            or (s.refusal_ok is False) or s.false_positive_acted
        n_ep += 1
        # Uncertainty-based DAgger: only pay for relabelling where the student was
        # unsure OR the episode went wrong. Cuts the teacher cost ~60% at little loss.
        # ONE teacher for the whole episode, fed states in order. A fresh teacher per
        # state resets its "have I already run the confirming test?" counter, so it
        # demands a test at every step and poisons the labels.
        teach = TeacherAgent(sc.truth.cause, sc.truth.recoverable)
        teach.reset()
        for r in ctrl.records:
            ctx = Context(t=r["t"], channel=sc.channel, n_channels=sc.n_channels,
                          peers=[], budget={}, available=r["available"],
                          last_scan=None, lora_available=sc.lora_available)
            d = teach.decide(r["features"], ctx)
            if uncertainty_only and not episode_failed and d.call == r["call"]:
                continue
            n_relabel += 1
            rows.append({"scenario": sc.name, "family": sc.family,
                         "cause": sc.truth.cause, "t": r["t"],
                         "features": r["features"], "call": d.call,
                         "args": d.args, "available": r["available"]})
    return rows, n_relabel, n_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus_all")
    ap.add_argument("--traces", default="../data/traces_all")
    ap.add_argument("--bundle", default="../data/student_all/student_bundle.json")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.corpus, "train", "*.yaml")))
    base = os.path.join(a.traces, "train.jsonl")
    cur_bundle = a.bundle
    for r in range(1, a.rounds + 1):
        rows, n_rel, n_ep = dagger_round(files, cur_bundle)
        print(f"  round {r}: rolled out {n_ep} episodes, relabelled {n_rel} states")
        out = os.path.join(a.traces, f"train_dagger{r}.jsonl")
        # union of the original traces and everything collected so far
        with open(out, "w") as fh:
            for src in [base] + [os.path.join(a.traces, f"train_dagger{k}.jsonl")
                                 for k in range(1, r)]:
                if os.path.exists(src):
                    fh.write(open(src).read())
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        # retrain from scratch on the union
        outdir = os.path.join(os.path.dirname(a.bundle) + f"_dagger{r}")
        tmp_traces = os.path.join(a.traces, f"_r{r}")
        os.makedirs(tmp_traces, exist_ok=True)
        os.replace(out, os.path.join(tmp_traces, "train.jsonl"))
        vs = os.path.join(a.traces, "val.jsonl")
        if os.path.exists(vs):
            import shutil
            shutil.copy(vs, os.path.join(tmp_traces, "val.jsonl"))
        os.system(f"cd {HERE} && python3 train/train_student.py --traces {tmp_traces} "
                  f"--out {outdir} 2>&1 | grep -E 'min-exp-cost|call head|float32'")
        cur_bundle = os.path.join(outdir, "student_bundle.json")
        print(f"  round {r} bundle -> {cur_bundle}")
    print(f"\nFINAL DAgger bundle: {cur_bundle}")


if __name__ == "__main__":
    main()
