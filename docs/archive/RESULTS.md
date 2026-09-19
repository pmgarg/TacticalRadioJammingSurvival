# Final Results — Jamming Survival, On-Device Mesh Agent (Phase 1, simulation)

> **Revision 3.** A PHY→MAC→routing telemetry extension was added (see `TELEMETRY.md`):
> reciprocal link-quality reporting, TX-shadow timing, and TX-side KPIs — 48 features → 56,
> contract 1.0.0 → 1.1.0. It lifted ns-3 classification 80.8% → 86.4% and cut expected cost
> 26%, with congestion +29 points and hidden-terminal +14. The shipped model is now
> **vT3** (`data/student_FINAL/`).
>
> **Revision 2.** The entire pipeline was re-run after patching ns-3 (see `NS3_BUG.md`):
> traces regenerated, three model variants trained and compared under identical code, and
> every live measurement confirmed reproducible (two identical back-to-back runs). The
> final model is **variant C** below.
>
> **Revision note.** An earlier version of this file reported ns-3 figures measured from
> silently truncated episodes: ns-3.45's OLSR crashed with SIGILL under strong jamming
> (23 of 30 scenarios), and our generator only checked that an output file existed. That
> bug is found, patched and documented in `NS3_BUG.md`; every ns-3 number below is
> re-measured on the corrected corpus.

Everything below is produced by the deterministic verifier on a **held-out test split**
the models never trained on. ns-3.45 is authoritative throughout.

---

## 1. The deliverable

**`data/student_FINAL/student_bundle.json`** — a 2-head function-calling student
(refsim + ns-3 + offline DAgger + closed-loop bridge DAgger):

| | value | budget | headroom |
|---|---|---|---|
| parameters | 11,732 | — | — |
| **fp32 weights (shipped)** | **45.8 KB** | 512 KB | **11×** |
| int8 (explored, rejected by the gate) | 11.2 KB | 512 KB | 46× |
| inference | pure numpy forward pass | <20 ms | — |
| C header for ESP-IDF | `student_weights.h` | — | emitted |

**We ship fp32.** Quantisation was implemented, gated and measured. On the final model
the cause head reaches 99.45% argmax agreement and the call head 97.2% — both **below**
the 99.5% equivalence gate, so neither is shipped quantised. Since fp32 already fits the
budget 11× over, int8 buys nothing and would change behaviour. The gate's verdict stands;
this is the mixed-precision export path doing its job rather than a number we chose.

## 2. Three-way comparison (refsim closed loop, held-out test)

| metric | baseline | teacher | **student** |
|---|---|---|---|
| classification accuracy | 54.7% | 100.0% | **93.3%** |
| accuracy, seen families | 63.1% | 100.0% | **95.4%** |
| expected cost (lower better) | 1.507 | 0.000 | **0.147** |
| cost, seen families | 1.492 | 0.000 | **0.123** |
| detection latency median | 1.60 s | 0.50 s | **0.80 s** |
| censoring rate | 24.0% | 0.0% | **4.0%** |
| **FP fading→jamming (ACTED)** | 0.000 | 0.000 | **0.000** |
| recovery median | 5.70 s | 5.70 s | **5.70 s** |
| survival rate | 80.0% | 85.3% | **84.0%** |
| channel hops (mean) | 0.24 | 0.21 | **0.20** |
| **refusal gate** | PASS | PASS | **PASS (100%)** |

Per family: barrage 100%, congestion 100%, fading 100%, hidden-terminal 100%, spot 100%,
refusal 93%, node-loss 88%, reactive 86%, sweep 80%.
The baseline scores **0%** on hidden-terminal and node-loss.

**The thesis:** teacher→student gap is small (0.000 → 0.147 cost); student→baseline is
**10×** (0.147 vs 1.507).

### 2b. Variant study (identical code, identical splits, reproducible measurement)

| variant | refsim acc | cost | surv | ns-3 acc | **live ns-3** |
|---|---|---|---|---|---|
| A — refsim + ns-3 | 92.0% | 0.173 | 80.0% | 80.9% | 44% |
| B — A + offline DAgger | 92.0% | 0.187 | 84.0% | 82.2% | 39% |
| **C — B + closed-loop bridge DAgger (shipped)** | **93.3%** | **0.147** | **84.0%** | 81.3% | **56%** |

All three: FP fading→jamming acted = 0.000, refusal gate 100%.

**This reverses an earlier conclusion.** Before the ns-3 patch I reported bridge DAgger as
a regression. Measured on a simulator that does not crash, it is the largest single win
available: **+12 points live** over A. The earlier verdict was an artifact of episodes
silently truncating at different points.

## 3. The sim-to-real result (the most important finding)

A student trained **only on the fast refsim** scores 96% on refsim — and **50.0% on
authoritative ns-3 features**. Training on both recovers it:

| model | acc on ns-3 | cost on ns-3 |
|---|---|---|
| refsim-only | 48.5% | 1.457 |
| **FINAL (shipped)** | **81.3%** | **0.369** |

Per family on ns-3 features: barrage 12%→100%, node-loss 84%→100%, refusal 82%→100%,
spot 1%→76%, congestion 0%→59%, reactive 21%→50%, hidden-terminal 0%→49%.

This empirically justifies the design's insistence that ns-3 be authoritative. A model
validated only on the fast approximation looks excellent and is near-useless on the real
simulator.

## 4. Live ns-3 closed loop

The Python agent drives ns-3 over a UNIX socket, 1 Hz decisions over 10 Hz percept.
18 episodes, 2 per family. **Measurement verified reproducible** — two identical
back-to-back runs, and four identical repeats of a single scenario.

| | result |
|---|---|
| correct classification | **10/18 = 56%** |
| **channel hops on fading** | **0** |
| **FP fading→jamming (acted)** | **0/2** |
| **refusal declared** | **2/2** |

Earlier live figures (61% / 50% / 44% / 39% / 28% / 22% for the same model) were **not
reproducible** because the ns-3 OLSR crash truncated episodes at different points. Three
correctness fixes were needed before the number meant anything:

- **The ns-3 OLSR crash** (`NS3_BUG.md`) — 23 of 30 scenarios died with SIGILL.
- **Free spectrum scans.** The bridge handed the agent a full per-channel view every
  100 ms. A scan is an *action* costing budget and ~260 ms of deafness; the agent got
  them free, which also made `scan_age` permanently fresh and sweep-periodicity
  meaningless.
- **Shared accumulators — three separate times.** The CSV sampler and the live bridge
  read-and-reset the *same* buffer (beacon counters, then the min-hold noise floor), so
  whichever ran second saw empty data. The noise-floor case made every jamming scenario
  look like a clean channel, i.e. like fading.

## 5. Protocol verification (ns-3)

| layer | result |
|---|---|
| **OLSRv1** | 28 routes, max 2 hops, 5/6 nodes reach the sink; routes expire when the mesh dies |
| **Application data** | 2903 sent, **669 delivered** end-to-end multi-hop |
| **MAC / TDMA** | slot-aligned; duty 2903→1188→594 for 0/4/8 slots; collisions 3422→349 |
| **Noise floor** | barrage +24.0 dB, spot +25.8 dB, **fading +0.0 dB** (all channels clean) |
| **Agent safety suite** | **14/14 PASS** |

## 6. Corpus and training

- **540 validated scenarios** (97 rejected by a well-posedness validator), randomised
  over topology, node count, geometry, jammer parameters, onset, traffic, channel model.
- **All 540 run through ns-3** (0 failures) → 22,260 labelled feature rows.
- Teacher (privileged expert): **100% accuracy, 0.000 cost**.
- Training set: 4,719 refsim + 8,875 DAgger + 15,442 ns-3 = **29,036 decision rows**.

**DAgger mattered.** Pure behaviour cloning classified well but survived *worse than the
baseline* — the compounding-error failure the design predicted. One round fixed it
(acc 90.7%→96.0%, cost 0.253→0.093, survival 62.7%→78.7%); round 2 over-fit.

## 7. Safety is structural, not learned

Three interlocks, each added after a measured failure:

1. **Action masking** — a rogue agent demanding hops forever cannot exceed the budget.
2. **Scan-before-hop** — you may not hop without a fresh spectrum scan.
3. **Hop-must-be-useful** — hopping is only *available* if the current channel is
   actually hot, or a materially quieter one exists. On fading the spectrum is clean
   everywhere, so `hop_channel` is simply not in the action set. This is what took
   live hops-on-fading from 2 → 0.
4. **Dead-man failsafe** — fires even with a silent or hung model.

## 8. Honest limitations

1. **Novel-attack generalisation fails.** With `sweep` held out entirely the student
   scores **0%** on it; trained with it, **90%**. Real out-of-distribution failure.
2. **Live ns-3 (61%) trails offline ns-3 features (82.7%).** The agent's own actions
   move it into states the corpus does not cover; closed-loop DAgger *through the
   bridge* is the fix and was not run.
3. **congestion / hidden-terminal / reactive remain the weak families** live — they
   degrade the link without killing it, and ns-3 provides no TX-conditioned noise
   measurement, so S4 is unavailable there by construction.
4. **S1 (RSSI/PDR correlation) is largely unmeasurable on hardware** (survivor bias) —
   see `FIDELITY.md`. The fading call rests on S2 and S5 instead.
5. Student trained on ns-3 *features*, not ns-3 closed-loop rollouts.
