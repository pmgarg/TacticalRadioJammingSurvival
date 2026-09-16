# Jamming Survival — On-Device Mesh Agent

Capstone (EAG V3, Project 9) · Prateek Mohan Garg · Jatin Pahuja

**Design:** `docs/DESIGN.md` (merged authoritative design, 4-week plan in §12)

---

## What works today

| Component | Status |
|---|---|
| Frozen interface contract | `capstone/contract/` — actions, budgets, cost matrix |
| Scenario engine (**arbitrary mesh**) | `capstone/scenario/` — any N nodes, any topology, any jammers |
| Scenario corpus (9 families) | `scenarios/*.yaml` |
| Percept feature layer (48 features, S1–S10) | `capstone/percept/` |
| Reference channel sim (fast) | `capstone/sim/refsim.py` |
| Agent controller + safety envelope | `capstone/agent/controller.py` |
| Classical baseline diagnoser | `capstone/agent/baseline.py` |
| Verifier + gates | `capstone/verify/verifier.py` |
| **ns-3 world (authoritative)** | `sim_ns3/jamming-sim.cc` — compiles & runs on ns-3.45 |
| Teacher / student / DAgger | not started (Week 3) |

Current baseline scores on the 9-family corpus: **78% classification, expected cost 0.33,
100% survival, FP(fading→hop) = 0.0, refusal gate 100%.**

---

## Quick start (fast reference sim — no ns-3 needed)

```bash
cd capstone

# one scenario, verbose
python3 run_episode.py --scenario ../scenarios/spot_single_channel.yaml

# the whole corpus + scorecard + gates
python3 run_episode.py --corpus --out ../data/runs

# list / regenerate the corpus
python3 -m scenario.library --list
python3 -m scenario.library --emit ../scenarios --seeds 5
```

## Running the real ns-3 world

Your ns-3.45 tree: `/Users/prateekgarg/Documents/NS3/ns-3-dev`

```bash
# 1. install the scenario program
# one command does install + build + run-all + row/exit-code checks:
bash sim_ns3/build_and_check.sh
python3 capstone/tests/test_world_ns3.py     # asserts the fixes took effect

# 2. export any scenario to the flat config ns-3 reads
cd <repo>/capstone
python3 -m sim.export_ns3 ../scenarios/barrage_all_channels.yaml -o /tmp/b.cfg

# 3. run it
cd ~/Documents/NS3/ns-3-dev
./ns3 run "jamming-sim --config=/tmp/b.cfg --out=/tmp/barrage"
# -> /tmp/barrage.percept.csv   (what the agent may see)
# -> /tmp/barrage.truth.csv     (verifier only)
```

Verified working on ns-3.45 (optimized build) with modules:
`wifi spectrum propagation olsr mobility applications internet energy flow-monitor stats`.

**Rebuild ns-3 optimized for corpus generation** — a debug build is ~10x slower:
```bash
./ns3 configure --build-profile=optimized --disable-examples --disable-tests
```

---

## Instructor demo: throw any scenario at it

The engine is fully data-driven — nothing is hardcoded to 6 nodes. Write a YAML and run it:

```yaml
name: instructor_demo
family: spot
seed: 42
duration_s: 60
n_channels: 8
channel: 3
comm_range_m: 260
nodes:
  - {id: D1, pos: [0, 0, 40], role: agent}
  - {id: P1, pos: [180, 0, 30]}
  - {id: P2, pos: [360, 0, 30]}
  - {id: P3, pos: [180, 180, 30]}
traffic:
  - {src: D1, dst: P2, rate_kbps: 250}
jammers:
  - {id: JX, type: spot, pos: [120, 40, 20], eirp_dbm: 20, channels: [3]}
events:
  - {t: 20.0, type: jammer_on, target: JX}
truth: {cause: spot, onset_t: 20.0, recoverable: true}
```

Prebuilt topology helpers: `line_topology(n)`, `grid_topology(r,c)`, `ring_topology(n)`,
`random_topology(n, seed)` in `capstone/scenario/library.py`.

---

## Known limitations (honest list)

1. **ns-3 noise floor is contaminated by own TX.** The SpectrumAnalyzer sits at the agent
   and hears its own 16 dBm transmissions. Mitigated with a min-hold estimator (real radios
   do the same), so the *per-channel profile* discriminates correctly, but absolute levels
   are not a true noise floor. Proper fix: gate sampling on `own_tx_active`.
2. **refsim is an approximation.** Per design A2 it is for development and bulk training
   only; ns-3 is authoritative for every reported number. The two are not yet cross-validated
   (Week 2 fidelity gate).
3. **Sweep is unsolved by the baseline** (0% accuracy) — it needs ≥3 scans to see the hot
   channel move. This is genuine headroom for the learned student, not a bug.
4. **AgentBridge socket not yet wired** — ns-3 currently writes CSV logs; the live
   Python-agent-in-the-loop path is the next step.
5. No teacher, student, or DAgger yet (Week 3 of the plan).
