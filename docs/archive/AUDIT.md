# Audit — where the project actually is, what is missing, what must be decided

**Date:** 2026-09-13 · **Scope:** full repo + `Claude outputs/` + `docs/DESIGN.md` v1.0 + HLD v0.6
**Method:** read every source file in `capstone/` and `sim_ns3/`, every doc, and the code inside
the tarballs. Every claim below is traceable to a file and line.

---

## 0. Verdict in one paragraph

The **world** (ns-3) and the **evaluation machinery** are genuinely good and are the strongest
parts of the project. The **agentic layer is real but is not in the repository** — it lives in
`Claude outputs/capstone_final.tar.gz` and has never been merged. What *is* in the repository
and what `RESULTS.md` reports as the headline result is a **privileged-expert distillation**
system: a dictionary lookup on the ground-truth answer teaching a 12 K-parameter MLP. That is a
legitimate ML result and it is **not what the brief grades**, it is **not what `docs/DESIGN.md`
§9.1 says you built**, and your instinct that "our design is not right" is correct — though the
flaw is narrower and more fixable than a rewrite. Roughly four working days of merging, deleting
and re-measuring converts what already exists into what the brief asks for. The SDR-in-C++ plan
in your note would, by contrast, add several weeks and is explicitly ruled out by the brief.

---

## 1. Inventory — what exists, and where

### 1.1 In the repository, working

| Component | File | Assessment |
|---|---|---|
| ns-3 world | `sim_ns3/jamming-sim.cc` (1434 ln) | Correct choices. `MultiModelSpectrumChannel` + `SpectrumWifiPhy` + `WaveformGenerator` is the *right* way to model an out-of-standard jammer; under `YansWifiPhy` the jammers would be invisible (`:648-688`) |
| Mesh app + reciprocal KPI | `sim_ns3/mesh-node-app.cc` | A real custom ns-3 `Application`. Beacons at 10 Hz carry up to 16 `LinkReport{peer, rssi, pdr}` records — genuine reverse-link reporting, analogous to batman-adv TQ (`:142-161`, `:286-302`). This is a real contribution |
| OLSR UB patch | `sim_ns3/ns3-olsr-robustness.patch` | A real upstream-worthy bug. `NS_ASSERT(false)` in a `default:` branch compiles to `ud2` → SIGILL under corrupted frames. Genuinely worth reporting to ns-3 |
| Live bridge | `capstone/sim/bridge_server.py` | AF_UNIX + newline JSON, ns-3 blocks while sim time is frozen so agent latency does not distort the experiment. Sound design |
| Scenario engine | `capstone/scenario/` | Fully data-driven, arbitrary N nodes/topology/jammers. Good |
| Percept layer | `capstone/percept/features.py` | 56 features, **no ground-truth leakage by construction** (`:62-89`). Verified |
| Safety envelope | `capstone/agent/controller.py` | Action masking, budget, dead-man failsafe — the best-engineered part of the agent side |
| Verifier | `capstone/verify/verifier.py` | Deterministic predicates, never reads prose. Correct |

### 1.2 Built, measured, written up — but **not in the repository**

`Claude outputs/capstone_final.tar.gz` contains 12 Python files that exist nowhere in the working
tree. `HARNESS.md`, `RESULTS_LLM.md` and `AGENTIC_PLAN.md` describe them in detail as finished
work:

| File (in tarball only) | What it is |
|---|---|
| `capstone/gateway/provider.py` | LLM gateway — `claude -p` subprocess, SHA-256 prompt cache so runs replay byte-identically |
| `capstone/agent/llm_teacher.py` | **The actual teacher the brief asks for.** Sees the same 56 features as the student, no ground truth, returns `{belief, call, args, why, confidence}` as JSON |
| `capstone/harness/loop.py` | The agent loop: observe → render → ask → parse → validate → decide → record. ~230 lines, no framework |
| `capstone/harness/registry.py` | Tool schema **generated from `contract/agent_contract.json`** so tools the LLM may call and tools the sim implements cannot drift |
| `capstone/harness/parser.py` | Strict JSON extraction, exactly one repair attempt, then abstain |
| `capstone/harness/trace.py` | Append-only JSONL: prompt, raw response, parse mode, latency, cache hit |
| `capstone/harness/policy.py` + `induce.py` | Rule DSL + LLM policy induction, accepted only on held-out precision |
| `capstone/agent/rule_agent.py` | Runs **only** the induced rules through the same envelope — makes "are the LLM's rules any good?" a number |
| `capstone/harness/mutate.py` + `sweep.py` | 629-mutation corpus + robustness curve (brief deliverable #5) |
| `capstone/harness/verify_links.py` | Executable check of every edge in the architecture DAG |

Also missing from the repo: `data/policy_v3.json`, `data/mutations.jsonl`, `data/student_v7/`,
`data/llm_cache/`, `data/traces/llm_v*/`. `capstone/teacher/` is an **empty directory**.

**This is the single most consequential problem in the project.** The work that answers the brief
is not in the artefact you will submit, and `README.md` still says "Teacher / student / DAgger —
not started (Week 3)" while `RESULTS.md` reports a shipped student. An examiner reading the repo
sees a distillation project with no agent in it.

---

## 2. Four structural flaws

### F1 — The shipped "teacher" is an oracle, and your own design document forbids it

`docs/DESIGN.md` §9.1 states, in bold:

> **Critical rule: the teacher never sees ground truth.** If it does, its traces encode a
> capability the student can never have, and the distillation silently teaches the student to
> guess. The teacher's advantage must be *reasoning*, not information.

`capstone/agent/teacher.py:56` is `TeacherAgent(true_cause)`. Its belief is literally
`b[self.true] = 0.9` (`:72-75`); its action is `PLAYBOOK[self.true]` (`:131`); its test choice is
`CONFIRMING_TEST[self.true]` (`:122`). It is constructed with the answer at every call site —
`dagger.py:51`, `gen_traces.py:27`, `bridge_dagger.py:48`, `train_mixed.py:51`.

The design doc then contradicts itself in §9.3 refinement 1, which licenses ground-truth labels
for the belief head "because ns-3 labels are free". Both statements cannot stand. The result is
that `RESULTS.md`'s headline "teacher: 100.0% accuracy, 0.000 cost" is **not a result** — it is a
definition. `RESULTS_LLM.md` §5 says this plainly and correctly; `RESULTS.md` does not.

Consequence, and it is the one you intuited: the oracle **cannot reason about a scenario the
instructor invents on the spot**, because it does not reason at all. Add "the jammer also spoofs
beacons from a dead node" and there is no dictionary entry.

### F2 — Roughly half the agent's action surface does nothing

The brief's whole premise is *"run the cheapest distinguishing test."* Measured against the C++:

| Agent call | Reaches ns-3? | Effect |
|---|---|---|
| `hop_channel`, `set_tx_power`, `silent_listen`, `move`, `declare_link_lost` | yes | real (`jamming-sim.cc:1360-1405`) |
| `change_tdma_slot` | yes | **inert** — `tdmaSlots` defaults to 0 (`:566`) and neither `bridge_server.py:112` nor `gen_ns3_corpus.py:27` passes `--tdmaSlots`, so `InMySlot()` always returns true (`mesh-node-app.cc:63-66`) |
| `spectrum_scan`, `reroute`, `listen_test`, `transmit_probe`, `neighbor_probe`, `load_test`, `mobility_test`, `channel_hop_probe` | yes | **fall through the `if/else` chain with no effect whatsoever** |

The comment at `jamming-sim.cc:455-456` — *"Costs are charged by actually suppressing TX/RX for
the action's duration, so a spectrum scan really does make the drone deaf"* — is false. No such
suppression exists; scan cost is Python-side bookkeeping only (`bridge_server.py:161`).

Additionally the "TDMA" is not a MAC. `InMySlot()` gates the **application send call**
(`mesh-node-app.cc:134`); the packet then descends into the 802.11 DCF queue with backoff and
retries, so nothing constrains actual airtime to the slot. There is no guard interval, no slot-sync
beacon and no clock model — every node reads one perfect global simulator clock.

**Three of the eight recovery actions in the playbook are `change_tdma_slot`** (reactive,
congestion, hidden_term — `teacher.py:33-44`). Those three families are also the three the project
reports as weakest. That is not a coincidence.

### F3 — A cluster of measurement bugs corrupts exactly the weak families

| # | Defect | Effect |
|---|---|---|
| a | Reactive jammer never gets `JamOn()` called (`jamming-sim.cc:868-871`), so its PSD keeps the construction-time channel 1 while the mesh runs on channel 6; `g_jam[k].on = true` executes during `main()` **before** `Simulator::Run()`, so it is armed from t=0 with no pre-onset baseline; and `WaveformGenerator::Stop()` cannot truncate a wave already radiating, so each trigger emits 10 ms not the configured 1.5 ms | The "reactive" jammer is on ~60% of the time, on the wrong channel, from t=0. It is effectively a weak barrage. **This is almost certainly why reactive scores 0–50%** |
| b | `tick` (`:1168`) and `decide` (`:1413`) are both scheduled at t=1.0 s +100 ms; `tick` runs first and calls `ResetShadow()` (`:1126`) | The TX-shadow statistic S4′ — `TELEMETRY.md`'s headline mechanism for detecting reactive jamming — is **identically zero in every live episode**, while working correctly offline. Training and live inference see different distributions for four features |
| c | `m_rxCount` increments before the `kind` test (`mesh-node-app.cc:222-226`), counting DATA as beacons | In `hidden_terminal.yaml` the agent's most-loaded link reports PDR = 1.0 constantly regardless of actual loss |
| d | `flow.0.start`/`stop` written by `export_ns3.py:77-78`, never read by the C++ | `hidden_terminal.yaml` declares `start_s: 18.0`; the 1200 kbps interferer actually runs from t≈0.55 s. No pre-onset baseline, again |
| e | Per-link PDR in the offline CSV is quantised to {0,1} (`:1015`); the bridge works around it with a 1 s window (`:1236-1247`) but the **corpus that trains the model does not** | Offline and live features disagree systematically |
| f | `SpectrumValue5MhzFactory` centres the block 5 MHz above the true channel centre; `AnalyzerReport` integrates around the correct centres | A jammer configured on ch 6 is reported hot on ch 7. **S5 — one of the two features `FIDELITY.md` calls decisive — is shifted one channel relative to ground truth** |
| g | The "co-located" spectrum analyzer is a separate node pinned at the agent's *initial* position (`:950-974`) | After any `move`, the noise-floor reading is measured from the wrong place |

Several telemetry fields are also fabricated rather than measured: `tx_retry_depth` is a hardcoded
3× multiplier (`:1136`), `tx_duty` is the literal `g_agentTx ? 1.0 : 0.35` (`:1343`), `retry` is a
*receive*-side drop fraction surfaced as a per-link *transmit* retry rate (`:1296`), and
`own_tx_active` alternates artificially on **different periods** in the two code paths
(`bridge_server.py:83` uses `int(t)%2`, `ns3_adapter.py:70` uses `(t*10)%3`).

### F4 — Evaluation integrity

These matter most, because the project's credibility rests on the verifier being honest.

1. **Test-set contamination.** `corpus/` holds `sweep` out entirely (`corpus.py:29,176-178`);
   `corpus_all/` does not. Training defaults to `corpus_all` (`dagger.py:70`,
   `gen_ns3_corpus.py:53`) while evaluation defaults to `corpus` (`evaluate.py:39`).
   **50 of the 125 `corpus/test` scenarios appear in the cause head's training or validation
   data.** `RESULTS.md:22` says the numbers come from "a held-out test split the models never
   trained on". They do not.
2. **The controller reads ground truth.** `controller.py:130` reads `sc.lora_jammed` — a truth
   field set together with `recoverable=False` for the refusal family (`corpus.py:124-130`) —
   inside the failsafe that *produces the refusal-gate pass*. And `controller.py:298` computes
   `survived` from `sc.truth.recoverable` and writes it into the episode log, which the verifier
   then reads back (`verifier.py:130`). This contradicts `verifier.py:6` ("the only component
   that sees ground truth").
3. **The advertised decision rule is inert.** `student.py:83-94` computes a minimum-expected-cost
   class, but `Decision.top` is plain argmax (`api.py:29-32`) and the verifier scores `row["top"]`
   (`verifier.py:74`). Every reported accuracy, FP and expected-cost figure reflects **argmax**,
   not the cost rule the design is built around.
4. **Abstention is disabled.** `train_mixed.py:148` hardcodes `abstain_threshold: 0.0`; the shipped
   bundle carries 0.0, so `student.py:104` never abstains. The conformal calibration in
   `train_student.py:53-65` is also fitted on `max(proba)` while inference uses the min-cost class,
   so the guarantee would not transfer even if enabled.
5. **The headline FP gate cannot fail.** `hop_channel` only enters the action set when a fresh scan
   shows the channel hot or a ≥6 dB quieter alternative exists (`controller.py:94-109`). On fading
   the spectrum is clean everywhere, so `fp_fading_acted = 0.000` **by construction, for any agent
   including a rogue one**. This is good engineering and a bad headline metric — it measures the
   mask, not the model. Report it as a structural guarantee, not as a score.
6. **int8 export is broken and its artefacts are inconsistent.** `export_int8.py:109` reads
   `W_int8` unconditionally while `:88-91` stores fp32 for a failing head → `KeyError`. On disk:
   `student_weights.h` is **162 bytes** (the preamble and nothing else), `student_int8.json` is
   byte-size-identical to the fp32 bundle and contains no `W_int8`, and `int8_equivalence.json` is
   stale (Sep 11 03:59 vs the bundles' 17:59) with values contradicting `RESULTS.md:38-39`.
   `RESULTS.md:35` says the C header is "emitted"; it is a stub.
7. **DAgger round 2 silently discards round 1.** `dagger.py:84-88` looks for
   `train_dagger{k}.jsonl`, but `os.replace` at `:95` already moved those files. And
   `train_mixed.py:84-97` double-counts the entire refsim base corpus, because `_r1/train.jsonl`
   is by construction `base + round-1 rows`.
8. Two of the "14/14 PASS" safety tests assert nothing (`test_agent_safety.py:107` checks
   `ctrl.rung >= 0`, vacuously true because `self.rung` is written and never read anywhere;
   `:113-114` is a hardcoded `check(..., True)`).
9. `run_episode.py:34` registers only `{"baseline": BaselineAgent}` — the documented entry point
   cannot run the student at all.
10. `RESULTS.md:176-178` says closed-loop bridge DAgger "was not run"; §2b and §4 of the same file
    report it as shipped variant C.

---

## 3. On the cited research papers — none of them are for this project

The brief's Sources section lists four papers. Read the brief's own annotations:

| Paper | The brief says |
|---|---|
| Olfati-Saber 2006, *Flocking for Multi-Agent Dynamic Systems* | "the source of the three algorithms in **project 1**" |
| Olfati-Saber & Murray 2004, *Consensus Problems…* | listed alongside, same multi-agent-consensus family |
| Choi, Brunet & How 2009, *Consensus-Based Decentralized Auctions* | "The CBBA algorithm used in **project 2**" |
| Gerkey & Matarić 2004, *Taxonomy of Task Allocation* | multi-robot task allocation |

This is **Project 9**. All four are flocking / consensus / multi-robot task-allocation papers
attached to other projects in the same handout, plus a "three papers that get confused"
disambiguation note aimed at those projects. **None concerns jamming, RF diagnosis, MANET
robustness, or teacher–student distillation.** Nothing in the repo cites them and nothing should
— there is no consensus problem here (A13/§3 explicitly make neighbours cooperative but not
jointly optimised, so there is no distributed agreement to reach).

**Do not spend time force-fitting them.** If you want a literature section, the relevant bodies of
work are (a) jamming detection and classification in wireless networks, (b) reactive-jamming
detection via transmission-correlated statistics, (c) DAgger / imitation learning under
distribution shift, and (d) LLM-as-teacher distillation for constrained function calling. Ask and
I will pull specific, verified citations for each — I have not invented any here.

---

## 4. Your four-section proposal, graded

> **§1.** Full agentic harness for the teacher → LLM decides → policies and tools improve
> continuously → those policies and tools are used in the student's agentic architecture
> (no LLM on device) → test the student on the same scenarios.

**Correct, and already ~70% built — in a tarball.** `harness/loop.py`, `registry.py`,
`policy.py`, `induce.py`, `rule_agent.py` are exactly this. What is genuinely missing: the student
is currently distilled from the **oracle**, not from the LLM (`RESULTS_LLM.md` §5 names this as
edge L12, "the next piece of work, not a completed one"), and the induced policy has never been
used to shape the student's head.

> **§2.** Write SDR code, test as 2 processes on Linux/macOS, then attach it to ns-3 and
> connect ns-3 live to the teacher.

**This is the part I think is wrong, and it is the biggest risk in your note.** Four reasons:

1. **The brief rules it out.** *"You need no radio hardware to do this project… Build tier 1 first
   and completely."* Its stack list is *"ns-3 or GNU Radio for the channel and the attacker,
   batman-adv or OLSR as the mesh protocol, ESP32 with painlessMesh for the physical rig."*
   Tier 2 is **₹2,000 of ESP32s**, not a hand-written SDR stack.
2. **ns-3 already is that stack.** You cannot "attach your SDR C++ to ns-3" — ns-3 supplies the
   PHY, the MAC, the ARQ and OLSR. A hand-written OLSR and MAC would *replace* the authoritative,
   peer-reviewed component that A2 makes the entire ground-truth argument rest on. Your own design
   doc puts this out of scope twice (§1.2, §3.2, A3, A9).
3. **Writing a correct OLSR and a slotted MAC is a capstone by itself.** You have already found
   one undefined-behaviour bug in *mature, peer-reviewed* OLSR. A fresh implementation would have
   dozens, and none of them would be about agents.
4. **The course is about agentic systems.** You said this yourself and you are right. A hand-rolled
   MAC earns zero marks against the brief's verifier, which measures detection latency,
   classification, false positives, recovery and survival.

What is *legitimate* in §2 and should be kept: **the live ns-3 ↔ teacher loop**. That bridge
exists and works; the LLM teacher has simply never been run through it — `run_llm.py:31,49` uses
`RefSim`, never the ns-3 bridge. Closing that is one day of work and it is worth far more than an
SDR rewrite.

If you want real RF, the brief's own Tier 2 is the answer: 4–5 ESP32-S3 running ESP-NOW, one more
as the jammer. Your `MeshPktHdr` + `LinkReport` wire format and your slot arithmetic port to it
almost directly (`TELEMETRY.md` already maps each KPI to an ESP32 source). That is a weekend, not
a month, and it gives you the demo video the brief says makes tier 2 worth watching.

> **§3.** Live dashboard explaining what is happening in the scenario and in teacher/student
> decisions.

**Keep it, schedule it last.** It is genuinely persuasive — the LLM teacher's `why` string next
to the student's belief next to ground truth, live — and it is cheap once the traces exist,
because `harness/trace.py` already records exactly that. But it is presentation, not result.

> **"The course is about agentic systems, not ML/LLM design."**

**Right, with one correction.** The brief does explicitly ask for distillation: *"run a large
model in simulation as a teacher, capture the situation-to-action traces, and distil them into a
small model that fits the drone. Then measure the gap between teacher and student."* So a small
learned model on the device is **in** the brief. The error is not that you trained a net. The
errors are that (a) the thing teaching it was a dictionary rather than a reasoner, and (b) the
student is a bare classifier where it should be a **function-calling agent** — tools, thresholds,
policies, an evidence-gathering loop and an abstain path — of which a classifier is one component.
That second point is exactly what you described as "defining tool, threshold, policies and
required situation-awareness capabilities", and `rule_agent.py` is already half of it.

---

## 5. Decisions you must make (I recommend one in each case)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| **D1** | What is the teacher? | (a) oracle (b) LLM (c) both | **(c), clearly labelled.** LLM is *the* teacher for every headline. Keep the oracle, rename it `OracleLabeller`, and use it only as a free-label source and a ceiling. Never report it as a result |
| **D2** | What is the student? | (a) MLP (b) induced rule set (c) hybrid | **(c).** Your own data decides this: induction hit **1.00 held-out precision on barrage/spot/sweep** and **failed on fading/node_loss/hidden_term** (0.60/0.36/0.41). Rules for the half that is expressible, net for the half that is not, one envelope over both. That split *is* a finding and it is more interesting than either alone |
| **D3** | Where does the LLM teacher run? | (a) refsim (b) ns-3 live | **(b) for all headline numbers.** `RESULTS.md` §3 already proves refsim-only training collapses 96% → 50% on ns-3. The same argument applies to the teacher |
| **D4** | Hardware | (a) none (b) ESP32 tier 2 (c) full SDR C++ | **(a) now, (b) only if §1–§3 hit their gates.** Never (c) |
| **D5** | Single source of truth | repo vs tarballs | **Repo.** Merge `capstone_final.tar.gz` + `student_v7_final.tar.gz` this week, delete `Claude outputs/`, commit. Nothing measured outside the repo counts |
| **D6** | Held-out family | contaminated vs clean | **Clean.** Train on `corpus_all` minus `corpus/test`, or make both derive from one hash split. Then re-measure. Sweep held out is your generalisation claim; it is currently void |
| **D7** | `change_tdma_slot` | fix vs remove | **Fix** (`--tdmaSlots 4` in both call sites) — three of eight playbook entries depend on it. If it cannot be made real, delete it from the contract and change those three entries, rather than shipping an action that does nothing |
| **D8** | Inert diagnostic calls | implement vs restrict | **Implement `spectrum_scan` cost and `neighbor_probe` / `load_test`** (they are cheap in ns-3 and they carry the brief's "cheapest distinguishing test" thesis). Remove `listen_test`, `transmit_probe`, `channel_hop_probe` from the contract if they will not be implemented — a contract listing calls the world ignores is worse than a smaller contract |

---

## 6. Ordered plan

Ordered so the thing the brief grades lands first, and so nothing is measured twice.

**Day 1 — make the repo the truth (D5).** Merge both tarballs into `capstone/`. Delete
`Claude outputs/`. Rewrite `README.md` — it currently understates the project by three weeks.
Commit. Delete `data/student_v2`, `v5`, `vA`, `vB`, `vC`, `vT1`, `vT2`; keep `vT3`/`final` and
`v7`.

**Day 1–2 — fix the evaluation before touching the agent (F4).** One hash split feeding both
training and eval (D6). Remove `sc.lora_jammed` and the `survived` computation from
`controller.py`. Make `Decision.top` use the cost rule, or stop claiming the cost rule. Re-run and
expect every number to move; that is the point. Fix or delete `export_int8.py` and `student_weights.h`.

**Day 2–3 — fix the world (F2, F3).** In priority order: reactive jammer channel + arming +
burst (a); the `tick`/`decide` shadow-reset race (b); enable TDMA in both call sites (D7); the
5 MHz channel-centre offset (f); `flow.0.start` (d); the beacon/DATA count bug (c). Re-generate the
ns-3 corpus once, after all of them — it is ~6 h on two cores.

**Day 3–4 — close the loop the brief grades (D1, D3).** Point `run_llm.py` at the ns-3 bridge
instead of `RefSim`. Run the golden set. Then the measurement that is currently missing and is the
whole thesis: **train the student on LLM-teacher traces** and report the three-way gap —
LLM teacher vs distilled student vs classical baseline — with the oracle shown separately and
labelled as a ceiling, not a competitor.

**Day 5 — the honest table.** Re-run everything from one command. Every number regenerated.
Every regression listed. Then, and only then, the dashboard and the ESP32 stretch.

**What must never be cut:** the ns-3 world with ground truth, the deterministic verifier, the
mutation corpus, records-first scoring, one distilled student under budget, the refusal case at
100%, and — new — **an LLM in the teacher loop with its traces on disk**.

---

## 7. What is genuinely good, and should be said loudly

The project's instinct for honesty is its best feature and it should be foregrounded, not buried:

- You found and patched a real undefined-behaviour bug in ns-3.45's OLSR, then **retracted your
  own previously reported numbers** because they came from truncated episodes (`NS3_BUG.md`).
- You ran a **prompt ablation** and reported that three separate attempts to add information all
  made the agent *worse* (`RESULTS_LLM.md` §1). That is a real, transferable finding.
- You measured that a refsim-only student scores 96% on refsim and **50% on ns-3**, and used it to
  justify the design (`RESULTS.md` §3).
- You discovered that S1 — the feature your design named "the primary false-positive defence" — is
  **largely unmeasurable** because of survivor bias, and moved the defence to S2/S5 (`FIDELITY.md`).
- Policy induction found that the attacker half of the problem is expressible as threshold
  conjunctions and the benign half is not. That is the most interesting result in the project and
  it is currently buried in a tarball.

Five negative results, each measured. That is worth more than a higher accuracy number, and it is
the strongest evidence that the positive claims are real. Put them in the presentation.
