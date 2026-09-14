# Jamming Survival — On-Device Mesh Agent
## Design Document v2.0 (authoritative)

| | |
|---|---|
| **Project** | Jamming Survival — On-Device Mesh Agent (EAG V3 Capstone, Project 9) |
| **Authors** | Prateek Mohan Garg · Jatin Pahuja |
| **Version** | **2.0** — supersedes v1.0 (`archive/DESIGN_v1.0_merged_superseded.md`), HLD v0.6, DESIGN v1 |
| **Status** | corrected after full code audit (`AUDIT.md`) · re-frozen for the remaining build |

> **Why v2 exists.** v1.0 described a system in which the teacher reasons and the harness is
> borrowed. The code that was built has a teacher that does not reason (it is handed the answer)
> and a harness that is entirely our own (nothing is borrowed). v2 resolves both — it keeps v1's
> physics, its safety architecture and its evaluation discipline, which are sound, and corrects
> the three places where the document and the code disagreed. Every change from v1 is listed in
> §0.2 so a reviewer can diff the intent, not just the prose.

---

## 0. What changed, and why

### 0.1 The one-sentence thesis

*An LLM that reasons about radio telemetry, inside a harness we wrote, produces the tools,
thresholds and policies that a 50 KB on-device agent then executes without it — and we measure,
honestly, exactly how much is lost in that transfer.*

### 0.2 Corrections from v1.0

| # | v1.0 said | Reality | v2.0 says |
|---|---|---|---|
| C1 | §9.1: "the teacher never sees ground truth… its advantage must be reasoning"; §9.3: "the belief head is trained on free ns-3 labels" | Both were implemented as one thing: an oracle handed `true_cause` | **Two distinct components with different names and different roles** (§9). `LlmTeacher` reasons and is the teacher. `OracleLabeller` is a free-label source and a performance ceiling. It is never called "teacher" and never reported as a result |
| C2 | §5.3, §1.3, App. C: the harness is a "capstone profile" of a reused EAG V3 platform (S17Code + glc_v5 pinned submodules), with every deliverable mapped to a borrowed module | No submodules exist. `gateway/provider.py` and `harness/*` are ours, ~900 lines | **The harness is ours, written from scratch, no framework** (§5.3). This is a *stronger* answer to the brief's deliverable #1, not a weaker one. All platform-reuse claims are deleted |
| C3 | §9.3/A5: the student is "a classifier + test planner + policy head" | Shipped as a bare 2-head MLP; the induced rule set exists but is not part of it | **The student is a function-calling agent** (§9.4): induced rules where they are precise, a net where they are not, a shared evidence loop and abstain path, one safety envelope over all of it |
| C4 | §6.2/6.3: 16 calls in the contract | 8 of them reach ns-3 and do nothing | **The contract lists only calls the world implements** (§6). Anything not implemented is removed, not left as a stub |
| C5 | §10.6 targets measured on a "held-out split" | 50 of 125 test scenarios were in training | **One hash split, one seed, both sides** (§10.2). Contamination is a build failure, not a caveat |
| C6 | Tiers 2/3 = ESP32 + RTL-SDR stretch | unchanged, but a hand-written SDR/MAC/OLSR stack was under consideration | **Explicitly out of scope** (§3.2, A9). ns-3 supplies PHY/MAC/ARQ/OLSR; replacing it destroys the ground-truth argument A2 rests on |

---

## 1. Problem

The mesh is failing. The drone must work out **why** and act, with no ground link to ask for help.
Several very different causes look almost identical from inside the radio.

| What the drone sees | What it might be |
|---|---|
| Delivery collapses, noise floor high on every channel | Barrage jamming |
| Delivery collapses on one channel only | Spot jamming |
| Delivery "fine" but everything is a retransmission | Reactive jamming (fires on your own TX) |
| Per-channel collapse migrating across scans | Sweep jamming |
| Intermittent, correlated with position | **Fading / multipath — no attacker** |
| Link up, no traffic from one peer | Dead peer |
| Collisions rising with offered load | Congestion |
| Collisions with low local load | Hidden terminal |

The goal is **not** detecting jamming. It is **avoiding the wrong recovery action** — above all
declaring jamming when the cause is fading, because the response (hop channel) destroys the mesh
that was still working. The cost matrix weights that error 8–10× the reverse.

**The refusal case:** broadband jamming on every channel with no line of sight to any peer. There
is no escape. The correct action is `declare_link_lost` and the failsafe (return-to-home). An
agent that hops forever has failed, and in reality would fly until the battery ran out.

---

## 2. Goals and non-goals

**Goals.** Classify into {barrage, spot, reactive, sweep, fading, node_loss, congestion,
hidden_term} or abstain; keep FP(fading→jamming) near zero and *say whether it is the model or
the envelope that achieved it*; recover with the fewest costly actions; refuse when refusal is
correct; fit the drone budget (A6); beat the classical baseline on the cost-weighted scorecard;
and report the **LLM-teacher → student** gap honestly, per family, with every regression listed.

**Non-goals.** Beating a human RF engineer · attackers outside the corpus · signal-level DSP on
device · **modifying or re-implementing MAC/PHY/ARQ/OLSR** (§3.2) · network scale · multi-drone
joint optimisation · free-form generation on the device.

### 2.2 Why we do not write our own radio stack

This is a decision, recorded because it was reconsidered and rejected.

1. The brief: *"You need no radio hardware to do this project… Build tier 1 first and completely."*
   Its stack list names ns-3/GNU Radio for the channel, OLSR or batman-adv for the mesh, and ESP32
   for the physical rig. A hand-written SDR stack appears nowhere in it.
2. A2 makes the entire ground-truth argument rest on a **peer-reviewed, externally maintained**
   simulator. Replacing ns-3's PHY/MAC/routing with our own code makes us the authority on our
   own ground truth, which is the one thing the design exists to avoid.
3. Implementing OLSR and a slotted MAC correctly is a capstone by itself. We already found one
   undefined-behaviour bug in *mature* OLSR (`NS3_BUG.md`); a fresh implementation would have many,
   and none of them would be about agents.
4. The course grades agentic systems. The verifier measures detection latency, classification,
   false positives, recovery and survival — a hand-rolled MAC moves none of those numbers.

**What we do instead for real RF:** the brief's Tier 2 — 4–5 ESP32-S3 on ESP-NOW plus one
interferer (§11.2). The `MeshPktHdr` + `LinkReport` wire format and the slot arithmetic in
`mesh-node-app.cc` port to it nearly directly; `TELEMETRY.md` already maps each KPI to its ESP32
source. Stretch only, behind the §12 gates.

---

## 3. Assumptions (frozen)

| ID | Assumption |
|---|---|
| A1 | Tier 1 (ns-3) is mandatory and is a complete capstone alone. Tier 2 is stretch. Tier 3 is a written design |
| A2 | Ground truth comes from ns-3's own scenario configuration, never from code we wrote |
| A3 | The fast radio loop stays classical: stock ns-3 PHY/MAC/ARQ/OLSRv1. The agent runs at 1 Hz on top |
| A4 | Both agents are function callers over one fixed contract. Which call, which arguments, why. Only the controller touches ns-3 |
| A5 | **The teacher is an LLM** reached through `gateway/provider.py`, offline and in simulation only. It sees exactly what the student sees and **no ground truth**. Never on the drone |
| A5b | **The oracle is not the teacher.** `OracleLabeller` is handed `true_cause`; it supplies free classification labels and a ceiling. It is never reported as an agent result |
| A6 | On-device: no GPU, no network, < 512 KB, < 20 ms, deterministic. Must also fit an ESP32-S3 |
| A7 | 8 logical RF channels, one active per node, hopping supported, plus a low-rate sub-GHz LoRa fallback |
| A8 | Mesh routing: ns-3 OLSRv1, **unmodified** except for the upstream UB hardening in `NS3_BUG.md`, which changes no routing logic |
| A9 | **No hand-written PHY, MAC, ARQ or routing.** See §2.2 |
| A10 | Hard action budget (6 costly actions) and recovery timeout, enforced by the controller, not the model |
| A11 | The refusal decision must be reachable from sensing alone — never dependent on `move` |
| A12 | 6 nodes (1 agent + 5 peers), 8 channels, 9 scenario families. Frozen before implementation (stops the slide into a generic MANET project) |
| A13 | Every call in the contract is implemented in the world. A call the world ignores is removed from the contract |
| A14 | Records first, scoring second. Every run is a complete record on disk before any predicate runs |
| A15 | Every verdict is a deterministic predicate over tool output + oracle. It never reads the agent's prose or stated confidence |
| A16 | **The repository is the only artefact.** Nothing measured outside it counts |

---

## 4. System architecture — four tracks

This replaces v1's single pipeline. The tracks have explicit interfaces and explicit exit gates,
so each can be built and *tested standing alone* before it is wired to the next. That is the
correction to the process problem: previously the agent, the world and the training loop were
integrated before any of them was independently verified, and the resulting bugs (`AUDIT.md` F3)
hid inside each other for weeks.

```
  T1  TEACHER TRACK  ────────────────────────────────────────────────┐
      harness/loop.py ── gateway/provider.py ── LlmTeacher           │
      tools from contract · trace store · policy induction           │
      exit: LLM diagnoses live ns-3 episodes, traces on disk         │
                                                                      │ policies,
  T2  WORLD TRACK   ─────────────────────────────────────────────┐   │ thresholds,
      ns-3: SpectrumWifiPhy · OLSRv1 · MeshNodeApp · jammers     │   │ tools,
      bridge_server.py (AF_UNIX, 10 Hz percept / 1 Hz decision)  │   │ traces
      exit: every contract call has a measurable effect          │   │
                                                                 ▼   ▼
  T3  STUDENT TRACK ──────────────────────────────────────────────────────
      rules (induced) + net (distilled) + evidence loop + abstain
      one safety envelope · no LLM · < 512 KB · < 20 ms
      exit: runs the same corpus through the same verifier, alone
                                                                 │
  T4  MEASUREMENT   ◄──────────────────────────────────────────────
      verifier · mutation corpus · three-way table · live dashboard
      exit: every number regenerated by one command
```

**Interfaces frozen between tracks:** `contract/agent_contract.json` (the tool schema, T1↔T2↔T3),
`percept/norm.py` constants (T2↔T3), `verify/metrics.md` (everything↔T4). Changing one is a
deliberate, versioned act, not a side effect.

### 4.1 Test-standing-alone rule

No track may be integrated until it passes its own gate **with the other tracks stubbed**:

- **T2 alone:** for each of the 9 families, a scripted action sequence produces the documented
  change in telemetry. `spectrum_scan` costs deafness; `change_tdma_slot` changes duty;
  `silent_listen` stops TX. If an action produces no measurable change, it is not implemented,
  whatever the code says (this is precisely what `AUDIT.md` F2 caught).
- **T1 alone:** the LLM loop diagnoses a fixed replayed telemetry file with no simulator attached,
  and its traces reload and re-score from disk with the model unavailable.
- **T3 alone:** the student runs the corpus with the LLM absent and the oracle absent.
- **T4 alone:** the verifier scores a hand-written synthetic episode log to a known answer.

---

## 5. Harness (brief deliverable #1)

**Ours. No LangChain, no LangGraph, no agent framework of any kind.** ~900 lines of plain Python
in `capstone/harness/` plus `capstone/gateway/provider.py`. This is a design commitment, not a
compromise: every control-flow branch is in one call stack, there is no hidden retry, no hidden
prompt mutation and no hidden state, and a grader can read the entire decision path.

| File | Responsibility |
|---|---|
| `registry.py` | Typed tool catalogue, **generated from `contract/agent_contract.json`**, so the tools the LLM may call and the tools the world implements cannot drift apart |
| `parser.py` | Strict JSON extraction: brace-balanced scan, **exactly one** repair attempt, then abstain |
| `trace.py` | Append-only JSONL — prompt, raw response, parse mode, latency, cache hit, tool result |
| `loop.py` | observe → render → ask → parse → validate → decide → record, plus the event gate |
| `policy.py` | Rule DSL, deliberately small enough that a rule cannot hide a second model inside it |
| `induce.py` | Policy induction: the LLM proposes, the held-out corpus disposes |
| `mutate.py` / `sweep.py` | Mutation corpus + detection-rate sweep (deliverable #5) |
| `run_llm.py` | Episode runner, **against the ns-3 bridge** (v1 ran it against refsim; see §9.2) |
| `verify_links.py` | Executable check of every edge in the architecture DAG |

**Commitments.** No hidden retries — one repair, recorded as `parse_mode: "repaired"`, then
abstain, and an abstention is a scored outcome not a swallowed error. Every LLM call replayable —
the provider caches on SHA-256 of the prompt, so a finished experiment re-runs byte-identically
with no model call and no API key. Failure is typed — `provider_error`, `unparseable`,
`masked_call`, `budget_exhausted` are four distinct recorded states. The loop is synchronous;
parallelism lives at the episode level only.

**The event gate.** A 1 Hz poll over a 60 s episode would ask the same question sixty times. The
gate re-reasons when the situation *changes*: `quiet` (delivery healthy, no diagnosis pending) →
no call; `stable` (panel moved < 0.15 L∞ and the legal-move set is unchanged) → hold the previous
belief and downgrade the call to `no_op`, because re-spending budget on an unchanged situation is
the exact pathology it exists to stop. Measured: 37 of 176 calls suppressed.

### 5.1 Mapping to the five required deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Own harness, no framework | `capstone/harness/` — §5, ours entirely |
| 2 | Open-source tool wrapped, vendored, pinned | ns-3.45 + our UB patch; `sim/bridge_server.py`, `sim/export_ns3.py`, `sim/ns3_adapter.py` |
| 3 | Task set with deterministic verifiers | 540 validated scenarios + `verify/verifier.py` predicates over the run record |
| 4 | Mutation corpus + detection rate | `harness/mutate.py` → `data/mutations.jsonl` (629), `harness/sweep.py` |
| 5 | Raw run records, scorer swappable | `data/runs/<run_id>.json` written before any predicate; `harness/trace.py` for LLM decisions |
| — | Refusal task | §8.3 — structural, 100% |

---

## 6. The contract (A13)

One file, `contract/agent_contract.json`, is the single source of truth for: what may be called,
what it costs, what it returns, and what it separates. `harness/registry.py` generates the LLM
tool schema from it; `agent/controller.py` enforces it; `sim_ns3/jamming-sim.cc` implements it.

**A13 in practice:** a call stays in the contract only if T2's standing-alone test shows it
produces a measurable change in telemetry. At the time of writing, `listen_test`,
`transmit_probe` and `channel_hop_probe` do not, and are therefore **removed** unless implemented
during T2. This shrinks the advertised action space and makes every remaining entry true.

### 6.1 Sense (free, continuous)

`rssi` · `noise_floor` (per channel) · `pdr` (per link, rolling) · `retry_rate` ·
`spectrum` (last snapshot) · `neighbor_status` · `route_status` · reciprocal link reports
(`rev_rssi`, `rev_pdr`, `rev_age` — how peers say they hear *us*) · TX-side KPIs
(`tx_success_ratio`, `tx_defer_time`, TX-shadow loss) · own position and budget.

### 6.2 Diagnose (buys information, costs budget)

| Call | Cost | Separates | Implemented in ns-3 |
|---|---|---|---|
| `spectrum_scan` | 1 | jamming vs congestion; barrage vs spot vs sweep (over ≥2 scans) | **must gate TX/RX for the dwell** — the deafness is the cost |
| `silent_listen` | 2 | **reactive vs steady** | yes (`SetTxEnabled(false)`) |
| `neighbor_probe` | 2 | dead peer vs channel/routing | **to implement in T2** |
| `load_test` | 2 | congestion vs jamming | **to implement in T2** |
| `mobility_test` | 8 | **fading vs attacker** | yes (`SetPosition`) — analyzer must move with the agent |

### 6.3 Act (recovery)

| Call | Cost class | Notes |
|---|---|---|
| `no_op` | free | the correct answer under the abstain threshold |
| `set_tx_power` | mild | the observed `tx_power` must reflect the change (it currently does not) |
| `reroute` | mild | **to implement in T2** |
| `change_tdma_slot` | mild | requires `--tdmaSlots 4` in both call sites; three of eight playbook entries depend on it |
| `hop_channel` | costly, −1 | retune all PHYs; the coordinated hop is currently free and instantaneous, which flatters it |
| `fallback_to_lora` | costly, −1 | trade rate for a link |
| `move` | expensive, −2 | |
| `declare_link_lost` | terminal | ends the episode, triggers RTH |

### 6.4 Every call records

function · arguments · the hypothesis it acted on · **the expected observable change** ("if this
was spot jamming, PDR on the new channel should exceed 0.8 within 3 s"). §8.4's recovery verifier
checks the outcome against that expectation, and a failed expectation is evidence folded back into
the posterior — "hopping to ch 6 did not help" is strong evidence against spot.

---

## 7. Percept layer

56 features from `RawObs`, **no ground truth by construction** (`percept/features.py:62-89`),
normalised with fixed committed constants in `percept/norm.py` — the same constants in ns-3, in
refsim and (for Tier 2) on the ESP32. Two feature implementations is the classic sim-to-real trap;
one implementation with three call sites is the defence.

Grouped: delivery (0–7) · signal (8–15) · interference (16–23) · MAC/ARQ (24–29) · spectral memory
(30–35) · temporal (36–39) · self/platform (40–43) · bookkeeping (44–47) · reciprocal link (48–51)
· TX-side KPI (52–55).

### 7.1 The discrimination statistics, and what measurement taught us

| Stat | Definition | Separates | Status |
|---|---|---|---|
| **S2** noise-floor Δ | noise now − pre-onset baseline | **attacker vs no attacker** | **Decisive.** barrage +24.0 dB, spot +25.8 dB, **fading +0.0 dB** |
| **S5** cross-channel profile | spread of noise across scanned channels | barrage (flat, all hot) vs spot (peaked, partial) vs clean | **Decisive** — but the analyzer is currently offset one channel (`AUDIT.md` F3f); fix before relying on it |
| **S3** energy w/o preamble | busy time with no decodable preamble | jamming (energy) vs congestion (packets) | works |
| **S4′** TX-shadow loss | loss just after our own TX vs while silent | **reactive** | strongest discriminator in the corpus offline, **identically zero live** (`AUDIT.md` F3b). Fix first |
| **S6** cross-link agreement | fraction of peers degraded / `pdr_spread` | one peer (node_loss) vs all (channel) | works; the *early* node_loss signal |
| **S7** fade-run statistics | run-length histogram below threshold | Rayleigh fades are short and bounded; a jammer is square and long | works |
| **S9** heartbeat gap | time since any frame from a peer | dead peer (emits nothing ever) vs jammed peer (still tries) | works, but **accumulates** — reading it at 4 s means nothing |
| **S11** link asymmetry | forward minus reverse PDR | our RX vs our TX fault; hidden terminal | works (reciprocal reports) |
| **S1** RSSI/PDR corr | `corr(RSSI, PDR)` | v1 called this "the primary false-positive defence" | **Largely unmeasurable.** Survivor bias: RSSI is only observed on frames that decode, and deeply faded frames do not. The fading call rests on **S2 and S5** instead (`FIDELITY.md`) |
| **S10** loss–load corr | `corr(loss, own offered load)` | congestion | **Measures 0.00 on every family.** Carries no information on this radio. Excluded from teacher guidance; candidate for deletion |

Two of v1's named mechanisms turned out not to work. Both are reported, not quietly dropped —
they are among the project's better findings.

---

## 8. Controller and safety envelope

The controller is deterministic, is not learned, ships with the agent, and owns everything the
model is not allowed to decide.

### 8.1 Loop

10 Hz percept → anomaly gate (two-sided CUSUM on short-window PDR vs a slow baseline) → 1 Hz
decision while an anomaly is live → action masking → execute → record. Detection latency is
measured from true onset, so the gate's own latency counts against the headline metric.

### 8.2 Action masking (mechanism 1)

Any call not in `ctx.available` is rewritten to `no_op`. `hop_channel` enters the set only with a
fresh scan (< 10 s) **and** hop-usefulness (current channel hot, or a ≥ 6 dB quieter alternative).
Repeating the same `(hypothesis, call)` pair is suppressed.

> **Honesty requirement.** Because the spectrum is clean everywhere under fading, `hop_channel` is
> *never available* there, so `FP(fading→hop) = 0.000` holds **for any agent, including a rogue
> one**. This is excellent engineering and a worthless score. v2 requires it be reported as a
> **structural guarantee** with the rogue-agent control alongside, never as a model result. The
> model-level number to report instead is the **belief-level** FP rate.

### 8.3 Dead-man failsafe (mechanism 2) — the refusal guarantee

Forces `declare_link_lost` when: no link for ≥ 20 s · time since onset ≥ 30 s · costly budget
exhausted with the link down · or every channel bad with no escape. It runs whether or not the
model produces output, including if inference hangs. The model can trigger RTH *early* — the skill
we want — and can never prevent it.

**Two fixes required (from `AUDIT.md` F4.2):** the failsafe currently reads `sc.lora_jammed`, a
ground-truth field, and `survived` is computed from `sc.truth.recoverable` inside the controller.
Ground truth must not cross into anything that ships or into anything the verifier reads back.
The escape test must come from the agent's own sensing (A11); survival must be computed by the
verifier.

### 8.4 Recovery verification

After acting, watch PDR/retry over a recovery window. Restored → nominal. Not restored with budget
left → fold the failure into the posterior and continue. Budget or timeout exhausted → failsafe.

---

## 9. Teacher → student

### 9.0 What runs where

| Component | Runs | In the deployed loop? |
|---|---|---|
| **`LlmTeacher`** (the teacher) | offline, simulation only, via `gateway/provider.py` | **No** |
| `OracleLabeller` (not a teacher) | offline only | **No** |
| Harness, trace store, induction, distillation | offline only | **No** |
| **Student** (rules + net + evidence loop) | **on the drone** | **Yes — the only learned thing there** |
| Controller, percept, contract adapters | on the drone | Yes |
| Verifier, ground-truth oracle | offline only | No |

### 9.1 The teacher is an LLM (A5)

`agent/llm_teacher.py` receives the same 56 features the student receives, rendered as a readable
23-row panel with units and plain-English meanings — handing a reasoning model a float vector
throws away the only thing it is good at — plus the current budget, the legal call set, and the
tests already spent. It returns one JSON object: belief over 8 causes, one call, arguments, a
one-sentence rationale, confidence, and an `unrecoverable` score.

**It never sees ground truth.** Its whole advantage must be reasoning; a teacher given the answer
teaches the student to guess, which is precisely the failure v1.0 named and v1.0's code committed.

Cost control, all three necessary: the event gate (§5), SHA-256 prompt caching (a repeated state
is free and a finished experiment replays with zero model calls), and episode-level parallelism.
Measured: ~10 calls per episode, 0 parse failures over 76 episodes.

### 9.2 The teacher runs against ns-3, not refsim

v1 ran the LLM teacher closed-loop against `refsim` only. `RESULTS.md` §3 already proves what that
costs: a student trained only on refsim scores **96% on refsim and 50% on ns-3**. The same
argument applies to the teacher, so **every headline teacher number comes from the ns-3 bridge.**
refsim remains legitimate for bulk generation and for development, subject to the §10.4 fidelity
gate.

### 9.3 `OracleLabeller` — what it is for, and what it is not (A5b)

It is handed `true_cause` and plays a fixed playbook. Legitimate uses, both offline:

1. **Free classification labels.** ns-3 knows the true cause at every tick; supervised learning on
   an unlimited free label is not cheating and removes most of the teacher cost.
2. **A ceiling.** "What would perfect classification have scored?" is a useful number.

**Illegitimate uses, now forbidden:** calling it the teacher; reporting its 100% as a result;
comparing it to the student as though they were competing agents; or using it to select actions
in DAgger without also reporting the LLM-supervised variant. The honest comparison is
**LLM teacher vs student vs classical baseline**, all three without ground truth.

### 9.4 The student is a function-calling agent, not a classifier (C3)

This is the correction that matters most for the brief. On-device, per decision:

```
percept(56) ──► rule set (induced, auditable)  ──┐
           └──► net: cause head + call head     ──┤
                                                  ├─► belief ─► min-expected-cost rule
                     evidence sufficiency check ──┘             │
                                                                ├─ confident ──► policy → call
                                                                └─ not ───────► cheapest unspent
                                                                                 distinguishing test
                                                                                 ─► else abstain
                                        everything above passes through the safety envelope (§8)
```

**Why a hybrid, and it is a measured decision, not a preference.** Policy induction over the
corpus accepted rules at **1.00 held-out precision for barrage, spot and sweep** — the spectral
half of the problem, which is genuinely a conjunction of thresholds — and **rejected** rules for
fading (0.60), node_loss (0.36) and hidden_term (0.41). The rule-only agent scores 43.8%
standalone with **zero false positives acted on**; the net scores far higher but is not auditable.

*The attacker half of this problem is expressible as thresholds and the benign half is not.* That
is the most interesting finding in the project. The architecture should embody it: rules where
they are precise (auditable, portable to C, certifiable), the net where they are not, and the
disagreement between them is itself an uncertainty signal that routes to abstain.

### 9.5 Training

1. **Cause head** on free ns-3 labels (§9.3 use 1) — ordinary supervised learning, unlimited
   labels.
2. **Call and test selection** distilled from **LLM-teacher traces** (§9.1). This is the edge that
   has never been run and is the whole thesis; without it the project has no teacher→student gap
   to report.
3. **Cost-sensitive objective.** `L = Σ_c C(y,c)·p_c` with `C` from `contract/cost_matrix.yaml`.
   v1 specified this; the code trains plain cross-entropy and applies the matrix only at decision
   time. Either implement it or delete the claim.
4. **DAgger**, through the **ns-3 bridge**, uncertainty-filtered. Pure behaviour cloning
   demonstrably collapses closed-loop (acc 90.7 → 96.0, cost 0.253 → 0.093, survival 62.7 → 78.7
   after one round; round 2 over-fits). Two bugs must be fixed first: round 2 silently discards
   round 1's data (`dagger.py:84-95`), and `train_mixed.py:84-97` double-counts the base corpus.
5. **Calibrated abstention.** Conformal threshold on a held-out calibration set, fitted on the
   *same* statistic used at inference. Currently fitted on `max(proba)` while inference uses the
   min-cost class, and then hardcoded to 0.0 so it never fires. Fix both or remove the claim.

### 9.6 Export

fp32 weight arrays + a reference C header. int8 is gated at ≥ 99.5% argmax agreement and posterior
L1 < 0.02; if a head fails the gate it ships fp32, and `export_int8.py` must **not** crash on that
path (it currently does, leaving a 162-byte stub header and an `int8` bundle containing fp32
weights). Since fp32 already fits the 512 KB budget 11×, int8 is an experiment, not a requirement —
say so, and make the artefacts match the claim.

---

## 10. Evaluation

### 10.1 Records first, scoring second (A14)

Every run writes a complete record to `data/runs/<run_id>.json` — run_id, task_id, agent, exact
scenario + seed, code SHAs, episode log, ground truth — **before any predicate executes**. Scoring
is a pure offline pass. Changing a threshold means re-running the scorer, never the simulator.
LLM decisions additionally write prompt + raw response + latency + cache status to
`harness/trace.py` so any reviewer can re-derive a decision with no API key.

### 10.2 One split (C5)

`corpus/` and `corpus_all/` are replaced by a **single deterministic hash split** consumed by both
training and evaluation. A scenario id appearing on both sides is a **build failure**, checked in
CI. `sweep` is held out entirely — it is the novel-attack generalisation claim, and that claim is
currently void because 43 sweep scenarios leaked into training.

### 10.3 Metrics

Classification accuracy (8-way + sub-type) · **FP(fading→jamming) at belief level** *and*
`FP_acted` with the structural caveat of §8.2 · detection latency from onset, **censored not
dropped**, with the censoring rate reported beside it (medians are not comparable across agents
with 24% vs 4% censoring) · recovery time · packets lost · actions consumed · tests before correct
· survival, **computed by the verifier** · mutation-detection rate with a false-flag cap.
Mean ± 95% CI by paired bootstrap over shared seeds, per family and overall, tied to a git SHA.

### 10.4 Fidelity gate (refsim vs ns-3)

refsim may be used for bulk generation only while its post-onset feature marginals agree with ns-3
within 0.5 normalised units. Currently 37/40 pairs pass; the three failures are documented in
`FIDELITY.md` and refsim is conservative on all three, so a student trained on it is not flattered.
**Known refsim leakage to fix:** `shadow_missed` is synthesised from the true jammer type
(`refsim.py:381-382`) and `offered_load_norm`/`decodable_busy` switch on `self.hidden`
(`:370-371`, `:420-422`), making three features near-oracular in refsim and nowhere else. This is
a large part of the 96% → 50% collapse and it is a simulator modelling bug, not a feature-layer bug.

### 10.5 Mutation corpus (deliverable #4)

629 validated mutations over 7 physical axes (power, channel, geometry, duty, load, onset,
severity) plus the broken-input families (out-of-range, contradictory, stale, truncated, NaN,
degenerate config). Each carries parent, axis and signed magnitude so a failure is attributable.
Report the robustness curve per axis. The `power` curve degrading monotonically as the jammer
weakens is the *correct* failure mode — the agent stops seeing an attacker roughly when the
attacker stops being detectable; a flat `geometry` curve is the load-bearing result, because it
shows the agent is not keying on the frozen topology.

### 10.6 The headline table

| | classical baseline | **LLM teacher** | **student** | *(OracleLabeller)* |
|---|---|---|---|---|
| classification | | | | *ceiling, not a competitor* |
| expected cost | | | | |
| detection latency (+ censoring) | | | | |
| FP belief / FP acted | | | | |
| recovery, packets lost, actions | | | | |
| survival | | | | |
| refusal gate | | | | |

Two gaps are the thesis: **teacher → student small** (distillation kept the reasoning) and
**student ≫ baseline** (the learning was necessary). Every scenario where the student regresses is
listed explicitly.

### 10.7 Ablations

Remove S1 (is it really the FP defence? — we predict no) · remove S3 · **passive-only, no epistemic
actions** (what is the *agentic* part worth over a classifier?) · no `silent_listen` · BC-only vs
+DAgger · plain CE vs cost matrix · fp32 vs int8 · **rules-only vs net-only vs hybrid** (§9.4) ·
LLM-traces vs oracle-traces (§9.3) · **oracle-with-ground-truth**, which should collapse and
demonstrates why it was forbidden.

---

## 11. Stack

| Layer | Choice |
|---|---|
| Harness | **ours** — `capstone/harness/`, ~900 lines, no framework |
| LLM gateway | `capstone/gateway/provider.py` — `claude -p`, SHA-256 cache, cost accounting |
| Simulator | ns-3.45, pinned, + our OLSR UB patch; wifi, spectrum, propagation, olsr, mobility, applications, internet, energy, flow-monitor |
| Jammers | `WaveformGenerator` on `MultiModelSpectrumChannel` (correct: non-802.11 emitters share the channel with Wi-Fi PHYs) |
| Mesh app | `sim_ns3/mesh-node-app.cc` — beacons + reciprocal `LinkReport` KPI exchange |
| Bridge | AF_UNIX, newline JSON, ns-3 blocks while sim time is frozen |
| Fast sim | `capstone/sim/refsim.py` — development and bulk only, behind §10.4 |
| Student | numpy forward pass + rule evaluator; fp32 bundle; C header for ESP-IDF |
| Records | JSONL to `data/runs/`; `harness/trace.py` for LLM decisions |
| Tier 2 (stretch) | 4–5 ESP32-S3 on ESP-NOW + 1 interferer |
| Tier 3 (design only) | RTL-SDR on the **sub-GHz fallback band** — it cannot see 2.4 GHz; an nRF24L01+ can |

---

## 12. Plan and gates

Ordered so the thing the brief grades lands first, and so nothing is measured twice. Detail and
day-level sequencing in `AUDIT.md` §6.

| Stage | Work | Exit gate |
|---|---|---|
| **S0 Merge** | `capstone_final.tar.gz` + `student_v7_final.tar.gz` into the repo; delete `Claude outputs/`; rewrite `README.md`; commit | `capstone/harness/` and `capstone/gateway/` are tracked; the repo builds and runs the LLM loop |
| **S1 Evaluation integrity** | one hash split; ground truth out of the controller; cost rule or no cost-rule claim; fix/remove int8 | No test scenario in training. Verifier computes survival. Every headline number re-measured |
| **S2 World** | reactive jammer (channel, arming, burst); shadow-reset race; TDMA enabled; 5 MHz offset; `flow.0.start`; beacon/DATA count; implement or remove the inert calls | Every contract call produces a measurable change (§4.1) |
| **S3 The brief's loop** | `run_llm.py` → ns-3 bridge; golden set; **student distilled from LLM traces**; three-way table | LLM teacher beats the classical baseline with no ground truth; student within N points of the LLM teacher; both reported |
| **S4 Report** | one command regenerates every number; regressions listed; dashboard | `RESULTS.md`, `HARNESS.md`, `RESULTS_LLM.md` mutually consistent and reproducible |
| **S5 Stretch** | ESP32 Tier 2 | only if S0–S4 all passed |

**Never cut:** the ns-3 world with ground truth · the deterministic verifier · the mutation corpus
· records-first scoring · one student under budget · the refusal case at 100% · **an LLM in the
teacher loop with its traces on disk**.

---

## 13. What "done" looks like

1. An ns-3 world that manufactures all 9 families with ground truth, in which **every action in
   the contract does something measurable**.
2. An LLM teacher, inside our own harness, diagnosing live ns-3 episodes with no ground truth —
   every prompt, response and decision on disk and replayable without a model.
3. A student distilled from **those** traces: rules where rules are precise, a net where they are
   not, under 512 KB and 20 ms, running with no LLM.
4. The headline table: baseline ≪ student ≤ LLM teacher, with the oracle shown separately as a
   ceiling, and every student regression listed.
5. The refusal case working *by construction*, with the rogue-agent control proving it is the
   envelope and not the model.
6. Five measured negative results kept in the report, not buried: the ns-3 OLSR crash that
   invalidated our own earlier numbers · S1 being unmeasurable · S10 carrying no information ·
   three separate attempts to add prompt guidance each making the agent worse · refsim-only
   training collapsing 96% → 50% on ns-3.

The honesty is the contribution. The accuracy number is the easy part.
