# Code flow — teacher, student, ns-3, hardware

**Companion to** `DESIGN.md` v2.0 (what the system should be) and `AUDIT.md` (what it currently is).
This document answers four questions:

1. Which module belongs to which block, and how the blocks connect.
2. What exactly runs on real hardware, and how it gets there.
3. How each component is proven correct **before** we spend LLM tokens on it.
4. How to run everything by hand from the CLI, and how to see it in a dashboard.

Everything here was checked against the code. Where something does not exist yet it is marked
**[TO BUILD]**; where it exists but is broken it is marked **[BROKEN]** with the reference.

---

## 1. The one idea that makes the hardware question easy

> **`RawObs` is the hardware boundary.**

`percept/features.py` defines a single struct, `RawObs` (+ `LinkObs`, `ChannelObs`, `ScanResult`).
Everything **upstream** of it is *the world*. Everything **downstream** of it is *device code*.

```
                    ┌──────────── the WORLD (never ships) ─────────────┐
  refsim.py ────────┤                                                   │
  ns3_adapter.py ───┼──►  RawObs  ──────────────────────────────────────┼──►  device code
  bridge_server.py ─┤    (the port)                                     │     (ships)
  esp32 HAL  [TO BUILD]                                                 │
                    └───────────────────────────────────────────────────┘
```

There are already **three** implementations of that port and they all work:

| Adapter | Source of truth | Used by |
|---|---|---|
| `sim/refsim.py` | our fast approximate channel model | bulk generation, development |
| `sim/ns3_adapter.py` | `*.percept.csv` written by ns-3 | offline corpus training |
| `sim/bridge_server.py` | live ns-3 over an AF_UNIX socket | closed-loop episodes |

The ESP32/SDR build is simply **a fourth implementation of the same port**. That is the whole
answer to "what code do we run on real hardware": *the same code that is downstream of `RawObs`
today, plus one new adapter.* Nothing in `agent/`, `percept/` or `verify/` needs to know which
adapter is underneath it.

This is not aspiration — it is how the code is already factored. It is also the reason §2.2 of
`DESIGN.md` rejects hand-writing an SDR stack: the port is at the *telemetry* level, not at the
PHY level, so the radio stack underneath can be ns-3, ESP-NOW, or anything else that can report
RSSI, noise floor and per-peer delivery.

---

## 2. The four blocks

```
╔════════════════════════════════════════════════════════════════════════════╗
║ BLOCK A — WORLD                                    lab only, never ships   ║
║                                                                            ║
║  sim_ns3/jamming-sim.cc      ns-3 scenario program: PHY, jammers, events   ║
║  sim_ns3/mesh-node-app.cc    beacons + reciprocal LinkReport KPI exchange  ║
║  sim_ns3/*.PATCHED.cc        OLSR UB hardening (NS3_BUG.md)                ║
║  capstone/scenario/          scenario schema, corpus generator, topologies ║
║  capstone/sim/export_ns3.py  scenario YAML → flat .cfg for the C++         ║
║  capstone/sim/refsim.py      fast approximate channel model                ║
║  capstone/sim/ns3_adapter.py *.percept.csv  → RawObs                       ║
║  capstone/sim/bridge_server.py  live socket → RawObs, and actions back     ║
╚════════════════════════════════╤═══════════════════════════════════════════╝
                                 │  RawObs  (the port)
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
╔════════════════════════════════╗  ╔════════════════════════════════════════╗
║ BLOCK B — TEACHER   lab only   ║  ║ BLOCK C — STUDENT      SHIPS TO DEVICE ║
║                                ║  ║                                        ║
║  gateway/provider.py  [TARBALL]║  ║  percept/features.py   RawObs → 56 f.  ║
║    claude -p + SHA-256 cache   ║  ║  percept/ring.py       EWMA/CUSUM/corr ║
║  harness/loop.py      [TARBALL]║  ║  percept/norm.py       fixed constants ║
║    observe→render→ask→parse→   ║  ║  agent/api.py          Decision/Context║
║    validate→decide→record      ║  ║  agent/controller.py   ENVELOPE:       ║
║  harness/registry.py  [TARBALL]║  ║    masking, budget, dead-man failsafe  ║
║    tool schema from contract   ║  ║  agent/student.py      net forward pass║
║  harness/parser.py    [TARBALL]║  ║  agent/rule_agent.py   induced rules   ║
║  harness/trace.py     [TARBALL]║  ║                              [TARBALL] ║
║  agent/llm_teacher.py [TARBALL]║  ║  data/student_*/student_bundle.json    ║
║  harness/induce.py    [TARBALL]║  ║  data/policy_v3.json         [TARBALL] ║
║    LLM proposes rules,         ║  ║                                        ║
║    corpus disposes             ║  ║  ── nothing here imports ns-3, numpy   ║
║  agent/teacher.py              ║  ║     is the only non-stdlib dep, and    ║
║    OracleLabeller — NOT the    ║  ║     only in student.py ──              ║
║    teacher (DESIGN §9.3)       ║  ╚════════════════════════════════════════╝
╚════════════════════════════════╝
                 │ traces, policies, thresholds        ▲
                 └─────────────── distillation ────────┘
                                 │
╔════════════════════════════════▼═══════════════════════════════════════════╗
║ BLOCK D — MEASUREMENT                              lab only, never ships   ║
║                                                                            ║
║  verify/verifier.py     deterministic predicates + gates                   ║
║  train/*.py             gen_traces, dagger, train_student, train_mixed,    ║
║                         gen_ns3_corpus, evaluate, export_int8              ║
║  harness/mutate.py + sweep.py    mutation corpus + robustness  [TARBALL]   ║
║  harness/verify_links.py         executable architecture check [TARBALL]   ║
║  tests/                 agent safety suite, refsim↔ns-3 cross-validation   ║
║  dash/                  live dashboard                          [NEW]      ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**[TARBALL]** = the file exists in `Claude outputs/capstone_final.tar.gz` and **not** in the
repository. See `AUDIT.md` §1.2 — merging these is stage S0 and blocks everything else.

### 2.1 Module ownership, file by file

| File | Block | Ships to device? | Note |
|---|---|---|---|
| `percept/features.py` | C | **yes** | pure stdlib (`math` only). Port to C99 verbatim |
| `percept/ring.py` | C | **yes** | pure stdlib. Ring buffers, EWMA, CUSUM, run-length, Pearson |
| `percept/norm.py` | C | **yes** | fixed constants — must be byte-identical on sim and device |
| `agent/api.py` | C | **yes** | pure stdlib. `Decision`, `Context`, cause list |
| `agent/controller.py` | C | **yes** | the safety envelope. **[BROKEN]** reads `sc.lora_jammed` (ground truth) at `:130` and computes `survived` from truth at `:298` — both must go before this ships |
| `agent/student.py` | C | **yes** | numpy forward pass. **[BROKEN]** imports `PLAYBOOK` from `teacher.py` — the deployed student depends on the module that exists only to cheat. Move `PLAYBOOK` into `agent/policy_table.py` |
| `agent/rule_agent.py` | C | **yes** | **[TARBALL]** induced-rule executor |
| `agent/baseline.py` | C/D | no (comparator) | hand-written rules, the control |
| `agent/teacher.py` | B | **never** | `OracleLabeller`. Rename it. It is handed `true_cause` |
| `agent/llm_teacher.py` | B | **never** | **[TARBALL]** the actual teacher |
| `gateway/provider.py` | B | **never** | **[TARBALL]** `claude -p`, SHA-256 cache |
| `harness/*` | B/D | **never** | **[TARBALL]** the agent loop and the eval machinery |
| `sim/refsim.py` | A | never | **[BROKEN]** three features are synthesised from ground truth (`:370-371`, `:381-382`, `:420-422`) |
| `sim/ns3_adapter.py` | A | never | CSV → RawObs |
| `sim/bridge_server.py` | A | never | live socket; also does feature extraction **lab-side**, which is fine because it is standing in for the device |
| `sim/export_ns3.py` | A | never | YAML → `.cfg`. **[BROKEN]** writes `flow.0.start` which the C++ never reads |
| `scenario/*` | A | never | schema, corpus, topologies |
| `sim_ns3/*.cc` | A | never | ~88% ns-3 API by line count |
| `verify/verifier.py` | D | never | the only thing that may see ground truth |
| `train/*` | D | never | offline pipeline |

**Read the "ships" column as the port list.** It is eight files, all but one pure stdlib.

---

## 3. How it runs today — the three paths

### 3.1 Path 1 — fast loop, no ns-3 (development)

```
scenario YAML ──► scenario/schema.load_scenario ──► sim/refsim.RefSim
                                                        │ step() 10 Hz
                                                        ▼
                                                     RawObs
                                                        │
                            agent/controller.Controller │ FeatureExtractor.update()
                                                        ▼
                                                  56 features
                                                        │ every 10th step (1 Hz)
                                                        ▼
                                     _available()  ──► action mask
                                                        ▼
                                            agent.decide(features, ctx)
                                                        ▼
                                     mask enforcement + budget + failsafe
                                                        ▼
                                                    _apply() → refsim
                                                        │
                                                        ▼ end of episode
                                                  EpisodeLog ──► verify/verifier
```

Entry point: `capstone/run_episode.py`. Seconds per episode.

### 3.2 Path 2 — offline ns-3 corpus (how the model is actually trained)

```
corpus YAML  ──► sim/export_ns3.to_config ──► /tmp/x.cfg
                                                 │
                        ns3 run "jamming-sim --config=... --out=/tmp/x"
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
                 x.percept.csv                                      x.truth.csv
                 (what the agent may see)                          (verifier only)
                        │
                 sim/ns3_adapter ──► RawObs ──► FeatureExtractor ──► rows
                        │
                 train/gen_ns3_corpus.py ──► data/traces_ns3/{train,val,test}.jsonl
                        │
                 train/train_mixed.py ──► data/student_*/student_bundle.json
```

Note what this path **does not** do: the agent never acts. The CSV is a passive recording of one
fixed world. That is why closed-loop numbers trail offline numbers — the agent's own actions move
it into states this corpus does not contain.

### 3.3 Path 3 — live closed loop (the one that matters)

```
 bridge_server.py                                    jamming-sim.cc
 ────────────────                                    ──────────────
 bind AF_UNIX socket
 spawn ns3 binary with --sock ────────────────────►  BridgeConnect()
                                                     Simulator::Run()
                        ◄──── {t, pdr[], rssi[], noise, rev_*, tx_*} ── every 100 ms
 build RawObs                                        (ns-3 BLOCKS here; sim time frozen,
 FeatureExtractor.update()                            so agent latency costs nothing)
 1 Hz: agent.decide()
 controller mask + budget
                        ──── {"call": "...", "args": {...}} ────────►
                                                     hop_channel / set_tx_power /
                                                     silent_listen / move /
                                                     declare_link_lost
                                                     [8 other calls fall through
                                                      and do NOTHING — AUDIT F2]
```

**Cadence, measured:** the C++ `decide` lambda re-arms every **100 ms** (`jamming-sim.cc:1413`),
and Python decimates to 1 Hz (`bridge_server.py:187-190`). So 10 Hz socket round-trips, 10 Hz
feature updates, 1 Hz decisions.

### 3.4 Where the LLM teacher sits — and where it does not

Today: `harness/run_llm.py:31,49` constructs `Controller(RefSim(sc), ...)`. **The LLM teacher has
never run against ns-3.** Given that a refsim-only *student* scores 96% on refsim and 50% on ns-3
(`RESULTS.md` §3), every LLM number currently in `RESULTS_LLM.md` carries the same risk. Pointing
`run_llm.py` at `bridge_server` is one day's work and is stage S3 in `DESIGN.md` §12.

---

## 4. Hardware — what actually runs, and how

### 4.1 What you do NOT write

You do not write a PHY, a MAC, an ARQ, or OLSR. On ESP32 the radio stack is **ESP-NOW** (or
painlessMesh); on a Linux SDR host it would be the kernel's 802.11 stack or the vendor SDK. The
brief's Tier 2 is *"4 or 5 ESP32 boards running painlessMesh or ESP-NOW, plus one more ESP32 as
the interferer"* — the stack is given. `DESIGN.md` §2.2 records why hand-writing one is rejected.

### 4.2 What you DO write: one adapter, about six functions

This is the only genuinely new code the hardware needs. **[TO BUILD]**

```c
/* hal.h — the fourth implementation of the RawObs port (§1). */

typedef struct {
    uint64_t t_us;                      /* esp_timer_get_time()                    */
    float    rssi_dbm[MAX_PEERS];       /* rx_ctrl.rssi, last frame per peer       */
    float    noise_floor_dbm;           /* rx_ctrl.noise_floor  <- the key enabler */
    uint32_t rx_ok[MAX_PEERS];          /* beacons received, per peer              */
    uint32_t rx_expected[MAX_PEERS];    /* beacons due (they are periodic)         */
    float    rev_rssi[MAX_PEERS];       /* from the peer's LinkReport in its beacon*/
    float    rev_pdr [MAX_PEERS];
    uint32_t tx_attempts, tx_acked;     /* esp_now_send callback                   */
    uint32_t tx_defer_us;               /* time waiting for a clear channel        */
    uint8_t  channel;
} hal_obs_t;

void hal_sense       (hal_obs_t *out);                       /* free, continuous   */
int  hal_scan        (float noise_dbm[N_CH], uint32_t dwell_ms); /* COSTS deafness */
int  hal_set_channel (uint8_t ch);                           /* ~2 ms on ESP32     */
int  hal_set_tx_power(int8_t dbm);
int  hal_set_slot    (uint8_t slot, uint8_t n_slots);        /* TDMA gate          */
int  hal_set_tx      (bool enabled);                         /* silent_listen      */
int  hal_declare_lost(void);                                 /* failsafe / RTH     */
```

Everything above that line is the code you already have. Everything below it is ~300 lines of
ESP-IDF glue. **That is the entire sim-to-real gap**, and it is small precisely because the port
is drawn at the telemetry level.

### 4.3 What ports, and in what form

| What | Today | On device |
|---|---|---|
| Feature extraction | `percept/*.py`, pure stdlib, ~640 lines | **port to C99, no malloc.** Same fixed constants from `norm.py` — this is the single highest-leverage sim-to-real decision. Two feature implementations means the model trains on one distribution and deploys on another |
| Safety envelope | `agent/controller.py` | port to C. Masking is four lines; the dead-man failsafe is a timer |
| Induced rules | `data/policy_v3.json` + `agent/rule_agent.py` | a C table plus a `for` loop. **Fully auditable and certifiable** — this is why the hybrid student of `DESIGN.md` §9.4 matters for hardware, not just for the report |
| Learned net | `data/student_*/student_bundle.json` (45.8 KB fp32, 11.7 K params) | `student_weights.h` + a 30-line matmul. **[BROKEN]** the current header is a **162-byte stub** — `export_int8.py:109` crashes on a head that fails the gate (`AUDIT.md` F4.6) |
| Wire format | `MeshPktHdr` + `LinkReport`, `mesh-node-app.h:36-54` | 12 bytes + 3 bytes per report. An ESP-NOW payload carries it unchanged. **This already ports** |
| Slot arithmetic | `mesh-node-app.cc:60-93` | needs only a monotonic clock. `esp_timer_get_time()` |
| TX-shadow estimator | `mesh-node-app.cc:231-283` | pure integer bookkeeping over `(seq, arrival_time)`. We own the TX timestamps via the `esp_now_send` callback, and beacons are periodic by construction. **No extra hardware, no extra radio time** (`TELEMETRY.md` §1) |

### 4.4 Timing and sync — the honest answer

You asked about syncing timing off the host OS clock. For Tier 2 that is the right call and it is
cheap, because **nothing here needs tight sync**:

- The **decision loop is 1 Hz.** A few ms of clock skew is irrelevant.
- The **TX-shadow statistic** (τ ≈ 2–3 ms) uses **only the local node's own clock** — our TX
  timestamp and our own beacon-arrival timestamp. It never compares clocks across nodes. This is
  deliberate and it is why the statistic survives on cheap hardware.
- **TDMA slots** need cross-node agreement. The simulator cheats here — every node reads one
  perfect global clock, there is no guard interval and no drift (`AUDIT.md` F2). On hardware, use
  the beacon as the sync reference (a 100 ms frame with a 5 ms guard tolerates ±2.5 ms drift
  comfortably; ESP32 crystals drift far less than that between beacons). **Write this down as a
  known sim-to-real divergence** rather than discovering it on the bench.

### 4.5 The jamming rig

One extra ESP32 running continuous TX, with the four modes that mirror the corpus: barrage
(hop every few ms across the band), spot (pin one channel), sweep (walk the channel on a
`dwell_ms` timer), reactive (promiscuous RX, fire on detecting a frame from the agent's MAC).
The reactive one is the interesting build and it is also the one the simulator currently gets
wrong (`AUDIT.md` F3a) — building it on hardware would likely have caught that bug.

---

## 5. Proving each component before spending tokens

This is the heart of your question, and it is the process failure that let `AUDIT.md` F3 hide for
weeks: the world, the agent and the training loop were integrated before any of them was
independently verified, so their bugs concealed each other.

**The rule: no block may be wired to the next until it passes its own gate with the others
stubbed. No LLM token is spent until G1–G4 are green.**

### G0 — the repo is the truth *(blocks everything)*

```bash
# merge the tarballs, then:
git ls-files | grep -E 'harness/|gateway/|llm_teacher|rule_agent'   # must be non-empty
python3 -c "import sys; sys.path.insert(0,'capstone'); import harness.loop, gateway.provider"
```
Nothing measured outside the repository counts. **[TO BUILD]** `requirements.txt` and a `Makefile`
— neither exists today, so "it works on my machine" is currently unfalsifiable.

### G1 — the WORLD does what it says *(blocks all training)*

For each of the 9 families, run a **scripted** action sequence — no agent, no model — and assert
the documented telemetry change. This is the gate that catches F2 and F3.

| Assertion | Catches |
|---|---|
| `spectrum_scan` produces a gap in RX for its dwell | the false "scans are charged" comment (`jamming-sim.cc:455`) |
| `change_tdma_slot` changes measured duty cycle | TDMA never enabled (`tdmaSlots` defaults 0) |
| `silent_listen` drives TX count to zero | — |
| `hop_channel` moves the reported hot channel by the same delta | the 5 MHz analyzer offset |
| noise floor **rises** post-onset for every jammer family and **does not** for fading | the reactive jammer sitting on the wrong channel from t=0 |
| pre-onset window is clean for **every** family | `flow.0.start` ignored; reactive armed in `main()` |
| the TX-shadow counters are non-zero in **live** runs, not just offline | the `tick`/`decide` reset race |
| every call in `agent_contract.json` changes *something* | 8 inert calls |

**[TO BUILD]** as `capstone/tests/test_world_contract.py`. It is perhaps 200 lines and it is the
highest-value test in the project.

### G2 — the PERCEPT layer separates the families *(blocks all training)*

Per-family feature means, printed as a table, before any model is trained. A feature that is
identical across families carries no information — this is exactly how `S10 loss_load_corr` was
caught measuring 0.00 everywhere, and how the stale-binary run that produced "every feature at its
default" was caught (`RESULTS_LLM.md` §7). **Check the per-class feature means before trusting any
model.**

Existing: `tests/xval_refsim_ns3.py` (the refsim↔ns-3 fidelity gate). Extend it to print the
separation table and fail on a degenerate feature.

### G3 — the ENVELOPE holds against a rogue agent *(blocks any safety claim)*

`tests/test_agent_safety.py` exists and reports 14/14. **[BROKEN]** two of those assert nothing
(`:107` checks `ctrl.rung >= 0`, vacuously true because `self.rung` is written and never read;
`:113` is a hardcoded `True`). Fix those, then add the control that makes the headline honest:

```
RogueAgent — always demands hop_channel, always declares jamming
  assert hops <= budget            (masking)
  assert 0 hops on fading          (hop-usefulness)
  assert refusal within N steps    (dead-man)
```

When the rogue scores `FP(fading→hop) = 0.000` — and it will — you have proved the property is
**structural**. Report it that way (`DESIGN.md` §8.2), and report the *belief-level* FP as the
model's number.

### G4 — the EVALUATION is honest *(blocks every reported number)*

```bash
python3 - <<'PY'
import os
tr = set(os.listdir('data/corpus_all/train')) | set(os.listdir('data/corpus_all/val'))
te = set(os.listdir('data/corpus/test'))
assert not (tr & te), f"CONTAMINATION: {len(tr & te)} scenarios in both"
PY
```
Today this **fails with 50**. Also: ground truth out of `controller.py`; `survived` computed by the
verifier; `Decision.top` using the cost rule or the cost-rule claim deleted.

### G5 — the HARNESS loop works with the model stubbed

Replay a fixed telemetry file through `harness/loop.py` with a canned provider response. Assert:
the tool schema generated from the contract matches `agent/api.py`; the parser handles clean /
repaired / failed; the trace store round-trips append → reload → summary. `harness/verify_links.py`
already does most of this (`H1`, `H2`, `H3`) — run it as a gate, not an afterthought.

### G6 — only now, spend tokens

With G0–G5 green, the LLM costs are bounded by three mechanisms that already exist and are
measured: the **event gate** (37 of 176 calls suppressed), **SHA-256 prompt caching** (a repeated
state is free; a finished experiment replays with *zero* model calls), and **teacher-on-disagreement**
(invoke the LLM only where the student and the oracle disagree or confidence is low — roughly a
10× reduction). Budget a **200-episode golden set** fully LLM-labelled for headline numbers, and
run the cheap oracle everywhere else.

> **The cost of skipping these gates is already measured in your own repo.** 23 of 30 ns-3
> scenarios were silently crashing while the generator reported `ok=375 failed=0`, because it
> checked only that an output file existed. Every number derived from them had to be retracted.
> A validation that only proves a file exists proves nothing.

---

## 6. CLI runbook

Run everything from `capstone/`. Verified against this repo.

```bash
cd ~/Documents/Projects/EAG_V3/EAG_V3_final_project/capstone
```

### 6.1 One episode, fast sim, verbose

```bash
python3 run_episode.py --scenario ../scenarios/spot_single_channel.yaml -v
```
Prints the tests run, the actions taken, the declared cause and the scorecard. Takes seconds.
Good for "did I break the agent".

### 6.2 The whole corpus + gates

```bash
python3 run_episode.py --corpus --seeds 5 --out ../data/runs
```
Writes one JSON record per episode to `data/runs/` **before** scoring (records-first, A14), then
prints per-family accuracy, expected cost, FP rates, survival and the two gates.

> Note: on a single non-fading, non-refusal scenario both gates print `value=None … FAIL`, because
> there is nothing to measure. That is a reporting wart, not a failure — treat gates as meaningful
> only over the full corpus.

### 6.3 Regenerate the scenario corpus

```bash
python3 -m scenario.library --list                      # the 9 families
python3 -m scenario.library --emit ../scenarios --seeds 5
python3 -m scenario.corpus --n 540 --out ../data/corpus  # validated split
```

### 6.4 Run the real ns-3 world

```bash
# install + build + run every family + check exit codes and row counts:
bash sim_ns3/build_and_check.sh
python3 capstone/tests/test_world_ns3.py

# NOTE: the sources MUST go in scratch/jamming/ (a subdirectory), not scratch/ itself.
# ns-3 builds one executable per scratch directory from every .cc in it, and aborts if a
# directory holds a source with no main() -- so a flat copy of mesh-node-app.cc kills the
# build. build_and_check.sh does this correctly.
#
# For bulk corpus generation, build optimized first (~10x faster than the debug default):
# cd ~/Documents/NS3/ns-3-dev && ./ns3 configure --build-profile=optimized --disable-examples --disable-tests

# one scenario
cd -
python3 -m sim.export_ns3 ../scenarios/barrage_all_channels.yaml -o /tmp/b.cfg
cd ~/Documents/NS3/ns-3-dev
./ns3 run "jamming-sim --config=/tmp/b.cfg --out=/tmp/barrage"
#   -> /tmp/barrage.percept.csv   what the agent may see
#   -> /tmp/barrage.truth.csv     verifier only
#   -> /tmp/barrage.routes.csv    OLSR route table over time
```

**Apply the OLSR patch first** (`sim_ns3/ns3-olsr-robustness.patch`) or 23 of 30 scenarios die
silently with SIGILL. Add a pre-flight check that refuses to run on an unpatched tree.

### 6.5 Generate the ns-3 training corpus

```bash
python3 -m train.gen_ns3_corpus \
  --ns3 ~/Documents/NS3/ns-3-dev/build/scratch/jamming/ns3.45-jamming-sim-optimized \
  --corpus ../data/corpus_all --split train --jobs 2 --out ../data/traces_ns3
```
~6 h on two cores for 540 episodes. **Use the binary under `build/scratch/jamming/`, not the one
a level up** — the stale one produced 375 traces with every feature at its default and a model at
8.7% (`RESULTS_LLM.md` §7). If only percept code changed, use `train/reextract_ns3.py` against the
cached CSVs instead; the simulator is deterministic, so re-running produces byte-identical output.

### 6.6 Train

```bash
python3 -m train.gen_traces   --split train --corpus ../data/corpus --out ../data/traces
python3 -m train.train_student --traces ../data/traces --out ../data/student --hidden 64
python3 -m train.train_mixed  --refsim ../data/traces_all --ns3 ../data/traces_ns3 \
                              --bridge ../data/traces_bridge/train.jsonl --out ../data/student_mixed
python3 -m train.dagger --rounds 2 --corpus ../data/corpus_all --traces ../data/traces_all
```
**[BROKEN] before you run DAgger:** round 2 silently discards round 1 (`dagger.py:84-95`), and
`train_mixed.py:84-97` double-counts the entire refsim base corpus.

### 6.7 Live closed loop against ns-3

```bash
python3 -m sim.bridge_server \
  --ns3 ~/Documents/NS3/ns-3-dev/build/scratch/jamming/ns3.45-jamming-sim-optimized \
  --scenario ../scenarios/reactive_on_tx.yaml \
  --agent student --bundle ../data/student_final/student_bundle.json
```
Python binds the socket and spawns ns-3 as the client. ns-3 blocks while sim time is frozen, so
agent latency does not distort the experiment.

### 6.8 Evaluate on the held-out split

```bash
python3 -m train.evaluate --corpus ../data/corpus --split test \
                          --bundle ../data/student_final/student_bundle.json
```
**[BROKEN]** the split is contaminated (G4). Fix before believing the output.

### 6.9 Tests

```bash
python3 tests/test_agent_safety.py       # 14 checks; 2 currently assert nothing
python3 tests/xval_refsim_ns3.py         # refsim vs ns-3 fidelity gate
```

### 6.10 After merging the tarballs

```bash
python3 -m harness.verify_links                  # 9 non-LLM architecture edges
python3 -m harness.verify_links --llm            # + live LLM edges (slow)
python3 -m harness.run_llm --per-family 2 --workers 8 \
        --trace-dir ../data/traces/llm --out ../data/llm_golden.json
python3 -m harness.induce --per-family 3 --out ../data/policy_v1.json
python3 -m harness.mutate --per-family 3 --manifest ../data/mutations.jsonl
python3 -m harness.sweep  --manifest ../data/mutations.jsonl --agent student \
        --bundle ../data/student_final/student_bundle.json --out ../data/sweep_results.json
```

---

## 7. Dashboard

`capstone/dash/` **[NEW — built, see below]**. Standard library only, no build step, no CDN.

```bash
cd capstone
python3 -m dash.serve                 # then open http://127.0.0.1:8765
python3 -m dash.serve --port 9000 --runs ../data/runs
```

It reads the artefacts the pipeline already writes — nothing new has to be logged:

| Panel | Source |
|---|---|
| Episode list, family, agent, pass/fail | `data/runs/*.json` |
| **Belief timeline** — posterior over 8 causes vs time, with true onset and the true cause marked | `episode_log.classification_trace` |
| **Delivery trace** — PDR vs time with onset, actions and recovery overlaid | `episode_log.pdr_series`, `.actions` |
| Tests and actions, with the agent's stated `why` and `expect` | `episode_log.tests_run`, `.actions` |
| Scorecard + gates, per family | recomputed by `verify/verifier.py` over the records |
| LLM decisions — prompt, response, latency, cache hit | `harness/trace.py` JSONL **[after S0]** |

Two things make this worth building rather than reading JSON:

1. **The belief timeline is the single most persuasive artefact you have.** Watching the posterior
   move as evidence arrives — and seeing it *not* move to "jamming" on a fading episode — is the
   whole thesis in one picture.
2. It is the natural place to put the **teacher's `why` next to the student's belief next to
   ground truth**, which is the comparison the brief asks you to measure.

**Verified working** against this repo: `serve.py` imports `verify/verifier.py` and scores the 9
records in `data/runs/` — the sweep episode reads *truth sweep / declared spot / wrong / cost 1.00*,
which is the documented baseline failure on the held-out family, now visible in one glance instead
of in a JSON file. The page renders clean in light and dark and scrolls its charts horizontally on a
phone. It also immediately surfaced something worth a look: on `fading_no_attacker` the controller
recovers at 11 s and then still fires `declare_link_lost` at 37 s on `recovery_timeout` — correct by
the letter of the failsafe, probably not by its intent.

**Live mode [TO BUILD]:** `bridge_server.py` already holds every decision in memory during an
episode. Adding a `--dash-port` that appends each decision to a JSONL the page tails turns this
from a post-hoc viewer into the live demo. That is a small change and it should come *after* S3.
