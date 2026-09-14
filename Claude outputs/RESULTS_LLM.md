# RESULTS — the LLM teacher inside our own harness

**All numbers on this page come from `data/llm_golden_v*.json` and the JSONL traces under
`data/traces/llm_v*/`.** Every decision is replayable offline: the provider caches on
SHA-256 of the prompt, so re-running a finished experiment reproduces it byte-for-byte
without a model call.

Setup: 18 episodes (9 families × 2 validated seeds), closed loop against `refsim` through
the same `Controller` safety envelope the student uses, 9 parallel workers.

---

## 1. The prompt ablation

Four versions of the teacher prompt, identical code, identical seeds, identical envelope.

| version | what changed | accuracy | FP acted | LLM calls | parse failures |
|---|---|---|---|---|---|
| v1 | cost asymmetry + commitment threshold stated | 38.9 % | **0** | 176 | **0** |
| v2 | + four-step procedure; explicit jammer sub-types; benign causes separated | 55.6 % | **0** | 200 | **0** |
| v3 | + evidence-maturity rule (heartbeat_gap accumulates; pdr_spread is the early node_loss signal) | **61.1 %** | **0** | 187 | **0** |
| v4 | + hand-written measured signature table | 55.6 % | **0** | 196 | **0** |
| v5 | + **generated** signature table (units bug fixed) | 50.0 % | **0** | 230 | 1 |
| v6 | v3's reasoning + the corrections the tables were *supposed* to convey, stated causally | 50.0 % | **0** | 209 | **0** |
| v7 | v3 exactly + the sweep two-scan correction only | **pending** (CLI quota) | — | — | — |

**False positives acted on (fading → channel hop): 0 in all 76 episodes.** The refusal gate
passed 100 % in every version. Those two properties are enforced by `controller.py`, not by
the prompt, which is exactly why they did not move while accuracy swung by 22 points.

### What each step actually fixed

**v1 → v2 (+16.7).** v1's failure was a single legible pattern, visible only because an LLM
says what it is thinking:

```
MISS node_loss   -> fading      MISS hidden_term -> fading
MISS congestion  -> fading      MISS reactive    -> fading
MISS spot        -> barrage     MISS spot        -> barrage
```

Every benign cause collapsed into `fading`; every narrowband jammer into `barrage`. That is
not a model failure — v1's text said *"symptoms alone → that is fading or a dead peer,
commit to the benign cause"*, which made `fading` the default answer for "no attacker" and
never separated the other three. v2 replaced it with: is there an emitter (noise and scan
only) → which of four attackers → which of four benign causes, each with its own positive
signature. **spot went 0/2 → 2/2 and congestion 0/2 → 2/2 immediately.**

**v2 → v3 (+5.5).** `node_loss` was still failing, and the trace showed why: at the moment
the agent committed, `heartbeat_gap` read **−0.61 (low)** — the peer had only been dead for
four seconds. The feature was correct; the agent was reading it before it had accumulated.
v3 added an evidence-maturity rule and pointed at `pdr_spread` as the early signal.
node_loss 0/2 → 1/2.

**v4 (−5.5) — a failed change.** Adding a hand-written signature table *lowered* accuracy.
The first suspect was a units bug, and it was real: twenty of the 56 features are `unit(x)`
and the panel decodes them to native percentages, but the hand-written table quoted them on
the [−1,+1] scale. The model was shown `offered_load 35%` and told to look for `−0.30` —
the same number in two notations it had no way to reconcile.

**v5 (−11.1) — the fix did not rescue the idea, which is the more interesting result.**
`harness/prototypes.py` now *generates* the table by running the simulator and formatting
every cell with the panel's own formatter, so the units bug is structurally impossible.
Accuracy fell further, to 50.0 %.

Two reasons, both visible in the generated table itself. Averaged over four seeds it
contains genuinely **colliding rows** — `spot` shows `heartbeat_gap +1.00`, identical to
`node_loss`; `barrage` shows `link_asymmetry −0.56`, identical to `hidden_term` — so
nearest-prototype matching sends the model to the wrong row. And more importantly, a lookup
table invites the model to *stop reasoning*: v5's traces pattern-match rows where v3's
traces argue from physics ("noise is nominal, so there is no emitter; pdr_spread is high,
so one link differs").

**Conclusion, and it is a real finding:** for this task, **explicit causal guidance beats
nearest-prototype matching**. Giving a reasoning model a table of exemplars degrades it.
v6 therefore ships v3's prose, with the corrections the tables were *supposed* to convey
(congestion vs hidden_term separated by `offered_load` and `link_asymmetry`; node_loss has
`retry_ewma` near zero; sweep needs a second scan; `loss_load_corr` is dead) folded in as
**reasons rather than rows**.

---

## 2. Where it stands, per family

| family | v3 | why |
|---|---|---|
| barrage | 100 % | strong positive evidence; `scan_bad_frac` 100 % with flat spread |
| spot | 100 % | partial hot band with high spread — separated once v2 said so |
| congestion | 100 % | `offered_load` at maximum plus high `tx_defer_time` |
| fading | 100 % | and **zero** false-positive hops across every version |
| refusal | 100 % | structural: the envelope's dead-man failsafe, not the model |
| node_loss | 50 % | needs `heartbeat_gap` to accumulate; still commits early half the time |
| hidden_term | 0 % | collides with congestion on `tx_defer_time`; separated only by `offered_load` and `link_asymmetry`, which the model does not weight |
| reactive | 0 % | closed-loop feedback pathology (below) |
| sweep | 0 % | needed a second scan; feature fixed after v3 ran |

---

## 3. Three real defects the LLM teacher surfaced

These are the reason the harness was worth building. All three were invisible to the
offline numeric pipeline and were found because a reasoning teacher **explains itself**.

### 3.1 `scan_bad_frac` was a knife-edge threshold

`JAM_ENERGY_DBM = −80.0` with a strict `>`. A barrage jammer parks the whole band at
**exactly −80 to −81 dBm**, so the feature landed on 0.50 and a 1 dB change in jammer power
flipped it between "nothing hot" and "everything hot". Mid-episode on a barrage scenario
the model wrote *"Clean spectrum scan (scan_bad_frac=0)"* and walked away from the correct
diagnosis. A numeric student makes the same mistake silently.

**Fix:** hot is now measured against our own pre-onset noise floor
(`min(JAM_ENERGY_DBM, baseline + 8 dB)`) — which an ESP32 also has, from its boot-time
`noise_floor` reading.

| family | before | after (bad_frac / spread) |
|---|---|---|
| barrage | 0.50 | **+1.00** / 0.05 |
| spot | 0.38 | +0.62 / **0.78** |
| every benign family | 0.00 | **−1.00** / 0.00 |

Effect on the shipped student over 629 mutations: barrage 92.6 % → **96.3 %**.

### 3.2 `scan_periodicity` could manufacture a sweep out of noise

Periodicity required **three** scans (of a budget of four), and recorded the argmax channel
unconditionally. Under barrage every channel sits at the same level, so the argmax is
decided by noise and jitters between scans — which reads as a moving hot channel, i.e. a
sweep. **Fix:** only record a hot channel when it clears the band median by 6 dB, which
makes the signature specific and lets two scans suffice.

| family | periodicity after fix |
|---|---|
| sweep | **+1.00** |
| spot / barrage / fading | −1.00 |

### 3.3 The runner was scoring invalid scenarios

`run_llm.py` called `scenario.make()` directly. `corpus.build()` filters with `validate()`;
`make()` does not. Degenerate scenarios — pre-onset delivery already collapsed, so there is
no clean baseline and the anomaly gate never fires — were scored as the agent "failing to
diagnose" something that never happened. One episode ran **zero decision steps** and was
counted as a miss. 10 of 28 candidate seeds were affected.

**Also found:** `loss_load_corr` (design statistic S10) measures **0.00 on every family**.
It carries no information on this radio despite its name. It is now excluded from the
teacher's guidance and flagged in the design doc.

---

## 4. Cost, and what makes an LLM teacher affordable

| | v3 |
|---|---|
| model calls | 187 over 18 episodes (~10 per episode) |
| calls suppressed by the event gate | 39 (17 %) |
| mean latency | 15–70 s per call |
| parse failures | **0** |
| repairs needed | **0** |
| provider errors | **0** |

Three mechanisms, all necessary:

1. **The event gate** — re-reason when the situation *changes*, not on a 1 Hz metronome. On
   benign families it skipped 10 of 23 steps.
2. **SHA-256 prompt caching** — a repeated state is free, and a finished experiment
   replays with no model calls at all.
3. **Episode-level parallelism** — 18 episodes across 9 workers, ~15 min wall clock instead
   of ~3 hours.

---

## 5. The honest comparison

| agent | classification | note |
|---|---|---|
| privileged oracle | 100 % | **it is handed `true_cause`.** Not a fair comparator |
| **LLM teacher (v3)** | **61.1 %** | no ground truth; reasons from the panel alone |
| student (12.8 K params) | 93.3 % refsim | distilled from the oracle, trained on 540 scenarios |
| threshold baseline | 54.7 % | control |

Two things must be said plainly:

* The oracle's 100 % is not an achievement, it is a definition — it cheats. The meaningful
  comparison is LLM-teacher vs baseline (61.1 % vs 54.7 %, both with no ground truth), and
  the LLM does it **zero-shot** while the baseline was hand-tuned on this corpus.
* The student's 93.3 % is on a task it was trained for, against a teacher that knew the
  answer. It has not yet been trained on LLM traces — that is edge **L12** in
  `ARCHITECTURE.md`, and it is the next piece of work, not a completed one.

---

## 5a. v6 — more instruction made it worse

v6 added everything the tables were trying to say, as prose: congestion vs hidden_term
separated by `offered_load` and `link_asymmetry`, node_loss has `retry_ewma` near zero,
sweep needs two scans, `loss_load_corr` is dead. It scored **50.0 %**, eleven points below
v3.

The breakdown shows what happened:

| family | v3 | v6 |
|---|---|---|
| **sweep** | 0 % | **100 %** |
| barrage, spot, fading | 100 % | 100 % |
| node_loss | 50 % | **0 %** |
| congestion | 100 % | **0 %** |
| refusal | 100 % | 50 % |

**sweep went 0 → 100 %, and that gain is from the FEATURE fix, not the prompt** — v6 was
the first version to run after `scan_periodicity` stopped being computable only from three
scans. Everything else got *worse*. The prompt grew from ~4.1 KB to ~5.6 KB and the extra
instructions competed for attention with the ones that were already working.

**The lesson, and it is the practical one for anyone building this:** prompt guidance is
not additive. Each addition has to be measured, and "more correct detail" is not the same
as "better decisions". v7 therefore ships **v3 exactly, plus the single sweep sentence** —
one change, so the next measurement is attributable.

The ablation as a whole: **v3 (61.1 %) is the best measured prompt; v4, v5 and v6 all
regressed.** Three different ways of adding information — a hand-written table, a generated
table, and more prose — each made it worse.

---

## 5b. Status of v7

v7 is written and committed (`harness/loop.py`) but **not yet measured** — the Claude CLI
hit its session quota mid-run (twice; it is rate-limited per session window). Nothing is
claimed for it. It is v3's prompt with exactly one change, the sweep two-scan sentence,
chosen because that is the only v6 addition the data supports.

Reproduce with:
```
python3 harness/run_llm.py --per-family 2 --workers 9 \
  --out ../data/llm_golden_v7.json --trace-dir ../data/traces/llm_v7
```

**Confound to state plainly:** v1–v3 ran before the `scan_periodicity` fix and v5–v7 after
it, so the world changed between them, not only the prompt. v3's 61.1 % and v6's 50.0 %
are therefore not a perfectly controlled pair. v7 vs v6 *is* controlled (same features,
one prompt change), which is why it is the measurement worth running.

---

## 6. Open, not hidden

* `reactive` at 0 % in the closed loop. The signal is present (`tx_shadow_loss_delta` is
  the single strongest discriminator in the corpus) but the agent's own probing changes the
  statistic it is reading. The fix under test is a randomised probe schedule so a reactive
  jammer cannot phase-lock, plus a paired-sample estimator over interleaved TX/silent
  windows instead of sequential ones.
* `hidden_term` at 0 %. It shares `tx_defer_time` with congestion and is separated only by
  `offered_load` and `link_asymmetry`. The generated prototype table should address it;
  unverified at time of writing.
* `sweep` was measured before the periodicity fix landed. Its numbers here are stale and
  will be re-run.
* 18 episodes is a small sample. Family-level percentages move in 50-point steps. They
  indicate direction, not precision, and are labelled as such.

---

## 7. The retrain (feature fixes propagated end to end)

The two feature defects in §3 changed 33–37 % of all training rows, so the shipped student
was running on a shifted input distribution. The whole corpus was regenerated and the
student retrained.

**540 ns-3 episodes regenerated, 0 failures.** (First attempt used a stale Sep-10 binary
whose CSV schema predated the telemetry work — it produced 375 traces with every feature at
its default, and a model at 8.7 % on ns-3 with FP 0.64. Caught by checking the per-class
feature means before trusting the model. The correct binary is
`build/scratch/jamming/ns3.45-jamming-sim-optimized`, not the one a level up.)

| metric | shipped (pre-fix) | **retrained** |
|---|---|---|
| **accuracy on ns-3** (authoritative sim) | 86.4 % | **92.7 %** |
| accuracy on refsim | 95.8 % | 95.8 % |
| **mutation-corpus robustness** (n=629) | 92.7 % | **93.8 %** |
| held-out refsim closed loop | 93.3 % | 89.3 % |
| FP fading→jamming (belief and acted) | 0.000 | **0.000** |
| refusal gate | 100 % | **100 %** |
| parameters / size | 12,756 / 49.8 KB | 12,756 / 49.8 KB |

Per family under mutation, retrained: congestion **100 %**, fading **100 %**, node_loss
**100 %**, reactive 98.9 %, barrage 95.4 %, hidden_term 95.3 %, spot 92.4 %, sweep 79.6 %.

**The trade is real and worth naming:** held-out refsim accuracy fell 4 points while ns-3
accuracy rose 6.3 and mutation robustness rose 1.1. ns-3 is the authoritative simulator
(design A2) and the mutation corpus is the robustness test, so this is the better model —
but it is a trade, not a free win, and the refsim number should not be quoted alone.

### A fidelity gap the fix exposed

`scan_periodicity` now separates sweep perfectly in refsim (+1.00 vs −1.00 for everything
else) but reads −1.00 for **every** family in ns-3. The reason is physical: ns-3's spectrum
analyser uses a TX-gated min-hold, which smears a sweeping jammer across the band, so no
single channel clears the 6 dB peak margin at scan time. The fix removed a *false* sweep
signal (barrage's jittering argmax) at the cost of the *true* one in ns-3.

That is the honest state: the feature is correct in refsim, inert in ns-3, and the margin
almost certainly needs to be lower (≈3 dB) or the ns-3 analyser needs a shorter hold. It is
also the cleanest single explanation for why `sweep` remains the weakest family at 79.6 %.
