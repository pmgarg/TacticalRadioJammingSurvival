# DAG flow — every module, both modes, and the five paths

**What this is for:** explaining the system out loud. Each section is one path you can walk
an examiner through on its own, with the module names that actually exist in the repo.

Two modes throughout:

- **MODE S — simulation.** The world is ns-3 (authoritative) or refsim (fast approximation).
  This is where training and all reported numbers come from.
- **MODE H — hardware.** The world is real 2.4 GHz radio: ESP32 mesh nodes plus a jammer.
  Same student, same envelope, same features. Only the adapter underneath changes.

**The one sentence that makes it all fit together:** `RawObs` is the boundary. Everything
upstream of it is *the world*; everything downstream *ships to the drone*. Modes S and H are
two implementations of the same port.

---

## 1. The block DAG (both modes on one picture)

```
                     ┌──────────────────────── MODE S ────────────────────────┐
                     │                                                        │
  scenarios/*.yaml   │   sim/export_ns3.py ──►  sim_ns3/jamming-sim.cc        │
  scenario/corpus.py │        (.cfg)              ├─ SpectrumWifiPhy + OLSRv1  │
        │            │                            ├─ mesh-node-app.cc          │
        └────────────┤                            │    beacons @10 Hz +        │
                     │                            │    LinkReport KPI exchange │
                     │                            └─ jammers: barrage / spot / │
                     │                               sweep / reactive          │
                     │                                   │                     │
                     │        ┌──────────────────────────┤                     │
                     │        ▼                          ▼                     │
                     │  *.percept.csv            AF_UNIX socket                │
                     │  (offline corpus)         (live, 10 Hz)                 │
                     │        │                          │                     │
                     │  sim/ns3_adapter.py      sim/bridge_server.py           │
                     │        │                          │                     │
                     │  sim/refsim.py  ─────────────────┐│   (fast dev world)  │
                     └────────┼──────────────────────────┼┼─────────────────────┘
                              │                          ││
                     ┌──────── MODE H ──────────┐        ││
                     │ ESP-NOW mesh (4× ESP32)  │        ││
                     │ + 1× ESP32 jammer        │        ││
                     │ hal_sense() [TO BUILD]   │        ││
                     └────────┬─────────────────┘        ││
                              │                          ││
                              ▼                          ▼▼
                    ╔═══════════════════════════════════════════════╗
                    ║   RawObs   ◄── THE PORT. Same struct, 4 sources ║
                    ╚═══════════════════════════╤═══════════════════╝
                                                │
                            percept/features.py │  56 features, pure stdlib
                            percept/ring.py     │  EWMA · CUSUM · run-length · corr
                            percept/norm.py     │  FIXED constants, identical everywhere
                                                ▼
                    ┌───────────────────────────────────────────────┐
                    │  agent/controller.py   THE SAFETY ENVELOPE    │
                    │   • action mask (_available)                  │
                    │   • budget: 6 costly actions                  │
                    │   • dead-man failsafe (sensing only)          │
                    │   • anomaly gate (CUSUM) -> 1 Hz decisions    │
                    └───────┬───────────────┬───────────────┬───────┘
                            │               │               │
              ┌─────────────┘               │               └─────────────┐
              ▼                             ▼                             ▼
   ┌────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
   │ agent/baseline.py  │   │ TEACHER SIDE (lab only)  │   │ STUDENT (ships)      │
   │ hand-written rules │   │                          │   │                      │
   │ THE CONTROL        │   │ harness/loop.py          │   │ agent/student.py     │
   └────────────────────┘   │  observe→render→ask→     │   │  2 MLP heads, numpy  │
                            │  parse→validate→record   │   │ agent/rule_agent.py  │
                            │ harness/registry.py      │   │  induced rules       │
                            │  tools FROM the contract │   │ agent/policy_table   │
                            │ harness/parser.py        │   │  belief -> action    │
                            │ harness/trace.py         │   │                      │
                            │ gateway/provider.py      │   │ NO LLM. <512 KB.     │
                            │  claude -p + SHA256 cache│   │ <20 ms. deterministic│
                            │ agent/llm_teacher.py     │   └──────────────────────┘
                            │  ** THE TEACHER **       │
                            │  sees the same 56 feats, │
                            │  NO ground truth         │
                            │                          │
                            │ agent/teacher.py         │
                            │  OracleLabeller — hand-  │
                            │  ed the answer. A        │
                            │  CEILING, never a result │
                            └──────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────────────┐
                    │ verify/verifier.py   the ONLY thing that may  │
                    │ see ground truth. Deterministic predicates,   │
                    │ never reads the agent's prose.                │
                    │   -> data/runs/*.json -> dash/serve.py        │
                    └───────────────────────────────────────────────┘
```

**Read the shape, not the boxes.** Four worlds feed one struct; one feature layer feeds one
envelope; three agents sit inside that envelope and are scored by one verifier. Nothing in
the student's column knows which world is underneath it. That is the whole architecture.

---

## 2. Path A — the world produces evidence (MODE S)

Walk this first: it is the part that is real physics rather than our code.

```
scenarios/spot_single_channel.yaml
  └─ sim/export_ns3.py            YAML -> flat key=value .cfg
      └─ jamming-sim.cc
          ├─ MultiModelSpectrumChannel + SpectrumWifiPhy    <- why: a WaveformGenerator
          │                                                    jammer is not an 802.11
          │                                                    device, so under YansWifiPhy
          │                                                    it would be INVISIBLE
          ├─ OLSRv1 (stock ns-3, + our UB patch)
          ├─ mesh-node-app.cc: 10 Hz beacons, each carrying up to 16 LinkReports
          │                    ("this is how I hear YOU") -> reciprocal link quality
          └─ jammer: barrage | spot | sweep | reactive
              └─ SpectrumAnalyzer co-located with the agent -> per-channel band power
                  ├─ <out>.percept.csv   what the agent MAY see
                  └─ <out>.truth.csv     verifier only
```

**Talking point:** the ground truth is ns-3's own scenario configuration, not something we
wrote. That is the reason any number here means anything.

---

## 3. Path B — the teacher reasons, and writes the policy (MODE S, lab only)

```
RawObs -> 56 features
   └─ harness/loop.py
       ├─ render_panel()          23 curated features as a TABLE with units and meanings
       │                          (handing a reasoning model a float vector wastes it)
       ├─ registry.render_catalogue()   the tool list, GENERATED FROM agent_contract.json
       │                                so the tools the LLM may call and the tools the
       │                                world implements cannot drift apart
       ├─ _should_ask()           the EVENT GATE: re-reason when the situation CHANGES,
       │                          not on a 1 Hz metronome  (37 of 176 calls suppressed)
       ├─ gateway/provider.py     claude -p, SHA-256 prompt cache
       │                          -> a finished experiment REPLAYS with zero model calls
       ├─ harness/parser.py       strict JSON, ONE repair attempt, then abstain
       ├─ registry.validate()     call + args checked against the contract
       └─ harness/trace.py        append-only JSONL: prompt, response, latency, cache hit
                                   -> every decision replayable offline, no API key
           │
           ├──► per-step traces  ────────────────────────► distillation (Path C)
           └──► harness/induce.py
                   LLM proposes rules in a tiny DSL
                     IF scan_bad_frac >= 0.9 AND scan_noise_spread <= 0.3 THEN barrage
                   -> measured on a HELD-OUT split
                   -> kept only if precision >= 0.75 (>= 0.90 for a jamming conclusion)
                   -> data/policy_v3.json   +   policy_v3_audit.json (every proposal,
                                                 accepted or not, with its numbers)
```

**The result worth showing:** induction accepted rules at **1.00 held-out precision for
barrage, spot and sweep** and **rejected** them for fading (0.60), node_loss (0.36) and
hidden_term (0.41). *The attacker half of this problem is expressible as threshold
conjunctions and the benign half is not.* That is why the student is a hybrid and not a
rule table — and why the rule table is still worth having: it is the auditable half.

---

## 4. Path C — training the student

```
                 free labels                       reasoning
  ns-3 knows the true cause              agent/llm_teacher.py decides
  at every tick                          WITHOUT ground truth
        │                                        │
        │  agent/teacher.py                      │  harness/run_llm.py
        │  OracleLabeller (privileged)           │
        ▼                                        ▼
  train/gen_traces.py  (refsim)          data/traces/llm_v*/    <-- STILL TO RUN
  train/gen_ns3_corpus.py (ns-3)             ** this is the edge the brief grades **
        │
        ├─ scenario/split.py       ONE canonical split. split_of(name) is a pure function,
        │                          so training and evaluation cannot disagree.
        │                          HELD_OUT_FAMILY = "sweep" -> test only, never trained.
        ├─ train/filter_traces.py  drops any row whose scenario is in the test split
        │                          (this is how the 2,274 leaked sweep rows were removed
        │                           WITHOUT re-running ns-3)
        ▼
  train/train_mixed.py
        cause head  56 -> 64 -> 64 -> 8      trained on free ns-3 labels
        call  head  56 -> 64 -> 12           distilled from teacher CALL choices
        + cost matrix (fading->jamming = 10, the reverse = 1)
        + conformal abstain threshold, calibrated on the SAME statistic inference uses
        ▼
  train/dagger.py / bridge_dagger.py
        roll the student out CLOSED LOOP -> ask the expert at the states it reached
        -> add (student state, expert action) -> retrain
        (pure behaviour cloning demonstrably collapses closed-loop; this is the fix)
        ▼
  data/student_v8_clean/student_bundle.json     12,691 params · 49.6 KB fp32
        ▼
  train/export_int8.py  -> student_weights.h    the C header the ESP32 build compiles in
        (int8 only ships if argmax agreement >= 99.5% AND posterior L1 < 0.02)
```

---

## 5. Path D — setting the student up to run (both modes)

The student is **not** just the .json. Per decision, on device:

```
  56 features
     ├─► rule set (data/policy_v3.json)        auditable, ports to C as a table + for-loop
     │        agent/rule_agent.py
     ├─► net (student_bundle.json)             agent/student.py, numpy forward pass
     │        cause head -> posterior
     │        call  head -> action preference
     ├─► min-EXPECTED-COST decision            not argmax. Declaring jamming on fading
     │    Decision.declared                    costs 10; being wrong the other way costs 1
     ├─► abstain if below the conformal cut    "unknown" is a legal, scored answer
     └─► cheapest unspent distinguishing test if not confident
                       │
                       ▼
     agent/controller.py ENVELOPE  (identical in MODE S and MODE H)
       mask -> budget -> repeat-suppressor -> dead-man failsafe
```

**Rules where they are precise, the net where they are not.** The disagreement between the
two is itself an uncertainty signal.

---

## 6. Path E — validation, five separate gates

Each answers a different question. `make gates` runs them in this order.

| Gate | Question | Mechanism | Status |
|---|---|---|---|
| **G0** | is the agentic code even in the repo? | imports + compileall | green |
| **G4** | is any test scenario in the training data? | `scenario/split.py` + `filter_traces` | green (was 50) |
| **G1(sim)** | does every call in the contract DO something? | `tests/test_contract_effects.py` — issue one call repeatedly, diff the telemetry against a no_op control | green, 2 known-open |
| **G1(ns-3)** | did the world fixes take effect in the REAL simulator? | `tests/test_world_ns3.py` — reads the measured CSVs | run `sim_ns3/build_and_check.sh` |
| **G3** | is the safety property structural or learned? | `tests/test_rogue_control.py` — a deliberately malicious agent | 10/10 |
| **G2** | does refsim agree with ns-3? | `tests/xval_refsim_ns3.py` | needs cached CSVs |

**The G3 result is the one to say out loud.** A rogue agent that demands `hop_channel`
forever also scores FP(fading→hop) = 0.000 — because on a clean spectrum the mask never
offers hopping. So that number measures the *envelope*, not the model. Report it as a
structural guarantee and quote the belief-level FP as the model's number. Most projects
would have quietly claimed the 0.000 as a learning result.

---

## 7. The headline, honestly stated

| | baseline | oracle\* | student (v8, clean) |
|---|---|---|---|
| classification, seen families | 67.7% | 100% | **98.5%** |
| expected cost | 1.431 | 0.000 | **0.031** |
| **sweep (never trained on)** | 0% | 100% | **0%** |
| FP fading→jamming (acted) | 0.000 | 0.000 | 0.000 (structural) |
| refusal gate | PASS | PASS | PASS |

\* handed the true cause. A ceiling, not a competitor.

**Say the 0% out loud.** On the eight families it has seen, the student is a 46× cost
improvement over hand-written rules. On a genuinely novel attack it fails completely. That
is the real answer to "does distillation generalise", and the earlier 90% was an artefact of
1,742 leaked training rows. A measured failure you can explain beats a number you cannot
defend.

---

## 8. What is NOT done, and where it sits on this DAG

1. **Path B → Path C is not connected.** The student is still distilled from the *oracle*,
   not from the *LLM teacher*. `harness/run_llm.py` also still runs against refsim
   (`RefSim`), never the ns-3 bridge. That edge is the thesis the brief grades.
2. **MODE H has no adapter.** `hal_sense()` and friends are specified (`CODE_FLOW.md` §4.2)
   and not written. ~300 lines of ESP-IDF glue; everything above `RawObs` already exists.
3. **`set_tx_power` cannot move the metric it is judged on** — inert by physics, yet it is
   `PLAYBOOK["fading"]`. A decision, not a bug.
4. **The ns-3 corpus must be regenerated** after the C++ fixes: enabling TDMA changes beacon
   timing corpus-wide, so old traces are not comparable to new ones.
