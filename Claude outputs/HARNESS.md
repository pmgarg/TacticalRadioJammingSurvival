# HARNESS — our own agentic loop, and what it measured

**No LangChain. No LangGraph. No agent framework.** ~900 lines of plain Python in
`capstone/harness/`, plus the existing `capstone/gateway/provider.py`. Everything the
agent does is visible in one call stack, and every decision it ever made is on disk.

---

## 1. What is in it

| File | Lines | Responsibility |
|---|---|---|
| `registry.py` | 110 | Typed tool catalogue, **generated from `contract/agent_contract.json`** so the tools the LLM may call and the tools the simulator implements cannot drift apart |
| `parser.py` | 85 | Strict JSON extraction: brace-balanced scan, **exactly one** repair attempt, then abstain |
| `trace.py` | 100 | Append-only JSONL: prompt, raw response, parse mode, latency, cache hit, tool result |
| `loop.py` | 230 | The loop: observe → render → ask → parse → validate → decide → record, plus the event gate |
| `policy.py` | 150 | A rule DSL small enough that a rule cannot hide a second model in it |
| `induce.py` | 220 | Policy induction: the LLM proposes rules, the held-out corpus decides which survive |
| `mutate.py` | 190 | Mutation corpus along 7 physical axes |
| `sweep.py` | 145 | Detection-rate sweep → the robustness curve |
| `run_llm.py` | 170 | Parallel golden-set episode runner |
| `verify_links.py` | 200 | Executable check of every edge in the architecture DAG |

### Design commitments, and why

* **No hidden retries.** One repair attempt, recorded as `parse_mode: "repaired"`, then
  abstain. An abstention is a scored outcome, not a swallowed error.
* **Every LLM call is replayable.** The provider caches on SHA-256 of the prompt, so a
  re-run of a finished experiment replays byte-identical decisions with no model call.
  Reproducibility of an LLM result is a property you have to build; it is not free.
* **Failure is typed.** `provider error`, `unparseable`, `masked call`, `budget exhausted`
  are four distinct recorded states.
* **The loop is synchronous.** Parallelism lives at the *episode* level (`run_llm.py`),
  never inside a decision. Debuggability beats throughput.

---

## 2. The event gate

A 1 Hz poll over a 60-second episode asks the model the same question sixty times. The
gate re-reasons when the situation **changes**:

* **quiet** — delivery is healthy and no diagnosis is pending → no call at all
* **stable** — the 23-feature panel has moved less than 0.15 (L∞) and the legal-move set is
  unchanged → hold the previous belief, and **downgrade the call to `no_op`**, because
  re-spending budget on an unchanged situation is the exact pathology the gate exists to stop

Measured over 18 episodes: **176 model calls, 37 suppressed by the gate** — a 17 % saving
on top of the cache, and larger on the benign families (`fading` skipped 10 of 23 steps).

---

## 3. Three real defects this found

Building the harness was worth it mainly because running it surfaced bugs the offline
pipeline had hidden for weeks.

### 3.1 A knife-edge threshold in `scan_bad_frac`

`JAM_ENERGY_DBM = -80.0` with a strict `>`. A barrage jammer parks the whole band at
**exactly −80 to −81 dBm**, so `bad_frac` landed on 0.50 — and a 1 dB change in jammer
power flips the feature between "nothing hot" and "everything hot".

The LLM made it visible because it *says what it is thinking*: mid-episode on a barrage
scenario it wrote "Clean spectrum scan (scan_bad_frac=0)" and walked away from the correct
diagnosis. A numeric student makes the same mistake silently.

**Fix:** hot is now measured against **our own pre-onset noise floor**
(`min(JAM_ENERGY_DBM, baseline + 8 dB)`), which is also what an ESP32 has from its
boot-time `noise_floor` reading. After the fix the feature separates cleanly:

| family | scan_bad_frac | scan_noise_spread |
|---|---|---|
| barrage | **+1.00** (all hot) | 0.05 (flat) |
| spot | +0.25 (partial) | 0.64 (peaked) |
| sweep | 0.00 | 0.71 |
| every benign family | −1.00 (nothing hot) | 0.00 |

Effect on the shipped student, measured over 629 mutations: barrage 92.6 % → **96.3 %**,
overall 92.4 % → **92.7 %**.

### 3.2 A normalisation that reads as a lie

Twenty of the 56 features are `unit(x)`: a 0–1 quantity mapped onto [−1, +1]. Rendered
raw, a fraction of **50 % prints as "+0.00"** — which any reader, human or model, calls
"zero". The panel now decodes those twenty back to native percentages before they reach
the prompt, and adds a level word and a bar.

### 3.3 The runner was scoring invalid scenarios

`run_llm.py` called `scenario.make()` directly. `corpus.build()` filters with
`validate()`; `make()` does not. Degenerate scenarios (pre-onset delivery already
collapsed, so there is no clean baseline and the anomaly gate never fires) were being
scored as the agent "failing to diagnose" something that never happened — one episode ran
**zero decision steps** and was counted as a miss. 10 of 28 candidate seeds were affected.

---

## 4. Prompt v1 → v2: a measured ablation

The first prompt told the model the decision rule and the cost asymmetry. It scored
**38.9 %** over 18 episodes — and the failure was a single, legible pattern:

```
MISS node_loss   -> fading      MISS congestion  -> fading
MISS congestion  -> fading      MISS hidden_term -> fading
MISS hidden_term -> fading      MISS reactive    -> fading
MISS spot        -> barrage     MISS spot        -> barrage
```

**Every benign cause collapsed into `fading`, and every narrowband jammer into `barrage`.**
That is not a model failure; it is my prompt. v1 said *"symptoms alone → that is fading or
a dead peer, commit to the benign cause"* — so `fading` became the default answer for "no
attacker", and the other three benign causes were never separated at all.

v2 replaces that with an explicit four-step procedure: **is there an emitter** (noise and
scan only, never symptoms) → **which attacker** (barrage = all hot *and* low spread; spot =
partial *and* high spread; sweep = the hot channel moves; reactive = only while we
transmit) → **which of the four benign causes** (each with its own positive signature, and
"committing to fading when heartbeat_gap is high is WRONG") → **act or test**.

Results are in `RESULTS_LLM.md`.

---

## 5. Policy induction — the LLM writing the student's rules

`induce.py` samples labelled states, shows the model a table, and asks for rules in the
DSL. Then the corpus judges them: **support and precision measured on a held-out split**,
with a stricter bar (0.90 vs 0.75) for any rule concluding a jamming cause, because that is
the expensive error.

From 9 proposed rules: **5 accepted at 1.00 precision on held-out data, 4 dropped.**

```
[barrage_full_band]  IF scan_bad_frac >= 0.9 AND scan_noise_spread <= 0.3
                     THEN barrage          (support=15  precision=1.00)
[spot_fixed_channel] IF scan_bad_frac in [-0.2, 0.9) AND scan_noise_spread >= 0.4
                        AND fade_runlen_mean <= 0.05
                     THEN spot             (support=15  precision=1.00)
[sweep_cycling]      IF ... same scan evidence ... AND fade_runlen_mean > 0.05
                     THEN sweep            (support=15  precision=1.00)
[congestion_self]    IF tx_shadow_loss_delta >= 0.3 AND rssi_pdr_corr >= 0.3
                     THEN congestion       (support=12  precision=1.00)
[hidden_term_asym]   IF link_asymmetry <= -0.3 AND tx_defer_time >= 0.3
                     THEN hidden_term      (support=10  precision=1.00)

dropped: reactive_tx_triggered   support=7   precision=0.57 < 0.90
         fading_multipath        support=20  precision=0.60 < 0.75
         node_loss_clean_dropout support=0   precision=0.00 < 0.75
         node_loss_intermittent  support=22  precision=0.36 < 0.75
```

Two things worth saying out loud:

1. The three spectral rules rediscovered the physics from data — barrage is *flat and
   everywhere*, spot is *peaked and partial*, sweep is *the same evidence but moving*.
   Nobody told the model that; it is in the table.
2. The dropped rules are the honest half. `fading`, `node_loss` and `reactive` **cannot be
   captured by a conjunction of thresholds** on this feature set — precision 0.60, 0.36,
   0.57. That is a real finding, and it is exactly why the student is a network and not a
   rule table: those three need the joint, non-linear structure. The rule set is the
   auditable statement of what the network is *supposed* to be doing on the easy half.

### Does the written policy actually work?

A rule set that reads well and measures badly is a story, not a result. `agent/rule_agent.py`
runs **only** the induced rules, through the same safety envelope, scored by the same
verifier as every other agent. When no rule fires it buys the cheapest unspent evidence and
then abstains — it never guesses, so its errors are omissions rather than false positives.

After a second induction round (two scans per sample, so `scan_periodicity` is live):
**6 of 8 rules accepted**, covering barrage, spot, sweep, reactive, congestion and
node_loss. Measured standalone on 48 held-out episodes:

| family | rule-only agent |
|---|---|
| barrage | 5/6 |
| spot, sweep, reactive, node_loss | 4/6 each |
| **fading, congestion, hidden_term** | **0/6** |
| overall | **43.8 %**, **0 false positives acted** |

And across the 629-mutation corpus: **47.9 %**, again **0 false positives**.

That split is the finding. **The attacker half of the problem is expressible as threshold
conjunctions; the benign half is not.** `fading` and `hidden_term` were rejected by the
held-out gate at 0.59 and 0.41 precision, and no rule for them survives. This is exactly
why the student is a network and not a rule table — and exactly why the rule table is still
worth having: it is the auditable, arguable statement of what the network is supposed to be
doing on the half that can be written down.

Artefacts: `data/policy_v3.json`, `data/policy_v3_audit.json` (every proposal, accepted or
not, with its measured support and precision).

---

## 6. Mutation corpus and the robustness curve

7 axes × 8 families × 6 base scenarios → **629 validated mutations**
(`data/mutations.jsonl`). Each carries its parent, axis and signed magnitude, so a failure
is attributable to a specific deformation of a specific case.

| agent | accuracy under mutation (n=629) | FP acted (fading→hop) |
|---|---|---|
| threshold baseline | 58.2 % | 0 |
| **student (12.8 K params, 49.8 KB)** | **92.7 %** | **0** |

Curves that matter (student):

```
axis          n     acc   robustness curve
power        80   91.2%   -9dB:75%  -6:94%  -3:94%  +3:94%  +6:100%
channel      45   84.4%   +1:100%  +2:75%   +3:75%  +5:89%
geometry    120   83.3%   +10m:83% +25:80%  +50:83% +80:87%
duty         64   93.8%   flat at 94% across 0.25 -> 0.8
load        120   96.7%   flat at 97%
onset       120   98.3%   97-100%
severity     80   97.5%   97-100%
```

**Read the `power` row.** Accuracy degrades monotonically as the jammer gets weaker and
recovers as it gets stronger — a physically correct sensitivity limit, not a cliff. The
agent stops seeing an attacker roughly when the attacker stops being detectable. That is
the right failure mode.

**`geometry` is flat**, which is the load-bearing result: the agent is not keying on the
frozen reference topology.

Per family, under mutation: fading **100 %**, hidden_term 98.4 %, reactive 97.8 %,
congestion 96.9 %, barrage 96.3 %, node_loss 94.6 %, spot 89.5 %, sweep 78.7 %.
**Zero false positives acted on, across all 629.**

---

## 7. Link verification

`python3 harness/verify_links.py [--llm]` turns the architecture DAG's claim table into an
executable test. Current state: **9/9 non-LLM links PASS**, LLM edges verified separately
in `RESULTS_LLM.md`.

```
PASS L1   corpus -> scenarios build                 54 scenarios, 9 families
PASS L3   refsim -> 56-feature vector               56 features, all in [-1,1]
PASS L5   oracle teacher -> controller -> refsim    5/5 correct
PASS L8   student -> controller -> refsim           6/6 correct
PASS L9   safety envelope                           AGENT SAFETY SUITE: 14 passed, 0 failed
PASS L13  episode -> verifier -> scored metrics     declared=fading cost=0.000 fp=False
PASS H1   tool registry generated from contract     16 tools, contract==api
PASS H2   parser: clean / repaired / failed         4/4 modes correct
PASS H3   trace store: append -> reload -> summary   3 records, 1 episode
```
