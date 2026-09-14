# Engineering review — Jamming Survival, as of 14 Sep 2026

An outside-eye review of the repo at commit `66c09d1`. Written to be read next to
`AUDIT.md` (what was wrong) and `FIXES.md` (what was changed). This one is about what
I would still not sign off on, and what will happen when this meets real radios.

---

## 1. Where the project actually is

| Layer | State |
|---|---|
| Mode R (refsim) | complete, gated |
| Mode N (ns-3 4.45) | complete, gated — world gate 17/17 |
| Path B→C (LLM teacher → rows → student) | complete, runs stubbed in CI and live against ns-3 |
| Mode H (hardware) | **port only**: 735 lines of HAL + 15/15 host self-test. No agent runs on a board yet. |
| Verification | G0–G5 green: split integrity, contract effect, percept fidelity, safety envelope, rogue control, Mode H self-test |

**The honest headline number.** There are two "overall" figures in the artefacts and
they are not interchangeable:

- `hybrid_v9.txt`: **91.4%** — 70 episodes, one seed per scenario, hybrid agent.
- `v8_vs_v9_fixed_world.txt`: **44.0%** — 5,102 episodes, net only, *including* the
  2,463 held-out `sweep` episodes the net scores 0% on by construction.

Excluding the held-out family the net scores **85.1% (2,247/2,639)**. That is the number
to quote, always with the second half of the sentence: *0% on the held-out family for the
net, 50% once the induced rules are in the loop.* Quoting 91.4% without its denominator
is how the v7 contamination happened in the first place.

---

## 2. What is genuinely good

**The epistemics, not the model.** `scenario/split.py`, the rogue-agent control arm, and
`test_contract_effects.py` are the real contribution. The contract-effect gate found that
`listen_test` and `transmit_probe` changed nothing in any family in any simulator, and
they were *deleted from the contract* rather than defended. A harness that can tell you
your headline result was contamination, and then makes you act on it, is rarer in student
work than a good accuracy number.

**`RawObs` is cut in the right place.** The port is drawn at the telemetry level, not the
PHY level — RSSI, a noise estimate, per-peer delivery — which is why four adapters
(refsim, ns-3 CSV, live bridge, ESP32) sit behind one struct and Mode H is 735 lines
instead of a rewrite. This decision is worth more than anything downstream of it.

**S4′, the TX-shadow statistic.** Bucketing missed beacons by whether our own transmitter
was active within ~3 ms, using only the local clock, never comparing clocks across nodes.
It took reactive-jammer discrimination from 50.7% to 99.0% and it is the one piece of
original signal design in the project. It is also the piece most likely to survive contact
with hardware, precisely because it needs no time sync.

**The hybrid argument is now a measurement, not a claim.** Net 0% → hybrid 50% on a
family the net never saw. Rules carry transferable physics; the net carries structure
that does not. That is a defensible thesis with a number under it.

---

## 3. What I would not sign off on

### 3.1 Two claims in the design were not live in the code — both now fixed

**`abstain_threshold` was 0.0.** The code that computes it was correct; the calibration
SET was wrong. `train_mixed.py` calibrated on refsim-val + ns3-val pooled. refsim is the
easy world (97.6% accurate); ns-3 is the world the agent is scored in (88.4%). Pooled, the
error at q=0 is 9.3% — just under the 10% target — so the smallest valid conformal cut is
0.0 and abstention never fires. Calibrating on the deployment world alone gives **0.68**,
which retains 96.6% of ticks at 9.7% error. Fixed in `train_mixed.py`; `train/
calibrate_abstain.py` re-derives it post-hoc for any existing bundle without a retrain.

Turning it on improved the closed loop: net 84.3% → 85.7%, expected cost 0.200 → 0.171;
hybrid expected cost 0.100 → 0.086. It also made live a path in `hybrid.py` that had never
executed — "net abstained, let a low-precision rule speak".

Two bugs fell out of switching it on, both of which had been masked by the threshold being
zero:
- `verifier.py` scored `row.get("declared") or row["top"]`, so an abstention (declared
  None) silently fell through to argmax and was credited or charged as a claim the agent
  had explicitly refused to make. `Decision` now carries an explicit `abstained` flag; the
  verifier keys on that, not on a null.
- `controller.py` could set `declared_jamming` from the argmax of an abstained tick.

**The harder result, which matters more than the fix.** Abstention was described in the
design as protection against the unknown. It is not, and now there is a number:

| | accuracy | mean posterior | abstains at 0.68 |
|---|---|---|---|
| in-distribution (7 families) | 0.850 | — | 6.4% |
| **held-out `sweep`** | **0.000** | **0.961** | **4.2%** |

The model is **confidently wrong** on the family it has never seen, and abstains on it
*less* than on families it knows. The softmax posterior carries no novelty signal
whatsoever. Conformal abstention controls in-distribution risk — it does correctly pull
back on `hidden_term` (29%) and `congestion` (16%), the two genuinely ambiguous families —
and nothing else. Detecting the unseen family is the induced rules' job. That is the
strongest available argument for the hybrid architecture, and it should be in the writeup.
`train/calibrate_abstain.py --ood` prints this table and refuses to let the claim be
overstated.

**`student_weights.h` was a 162-byte stub.** The int8 export now runs against v9 and emits
a real 54.3 KB C header that compiles. Two changes were needed:

1. **Bias correction.** Weight-only round-to-nearest shifts each layer's output by
   `mean(x) @ (Wq − W)`, and that compounds through three layers. Folding it into the bias
   is free at inference and took the cause head 98.67% → 99.19% argmax agreement, the call
   head 99.25% → 99.77%. It should have been there from the start.
2. **The gate now measures the deployed decision.** `agent/student.py` does not act on
   argmax; it acts on the minimum-expected-cost class, and only above the abstain
   threshold. Gating on unconditional argmax measured a quantity the agent never uses —
   the same class of error as AUDIT F4.3 and F4.4. On the ticks the agent actually acts
   on, fp32 and int8 agree **99.97%**; the disagreements sit at mean confidence 0.53
   against a 0.68 threshold, i.e. on coin flips the agent declines anyway.

   The threshold itself did not move (0.995 / L1 0.02), and the old unconditional number
   is still printed, because moving a goalpost quietly is worse than failing at one.
   Reported honestly alongside: ~1% of ticks flip between abstaining and acting in *each*
   direction (39 unsafe, 33 safe of 3458). It is not a one-sided safety argument.

Result: cause head all-int8, call head int8 except its output layer (a per-layer fallback,
not a relaxed gate — the output layer feeds the softmax with no ReLU to absorb its error,
and is the smallest layer in both heads). **16.0 KB, down from 48.8 KB fp32, 32× headroom
on an ESP32.**

### 3.1b Is the LLM teacher actually teaching? — ablation

The question the design's central claim rests on, and it had never been measured. Identical
pipeline, identical data, identical hyper-parameters; the only difference is whether the 89
rows from the LLM teacher are in the mix. Scored on the ns-3 test split, ex-`sweep`:

| seed | oracle only | + LLM rows | delta |
|---|---|---|---|
| 0 | 84.96% | 88.10% | **+3.14%** |
| 1 | 89.20% | 86.59% | −2.61% |
| 2 | 85.49% | 86.59% | +1.10% |
| 3 | 87.99% | 86.93% | −1.06% |

**Mean +0.14, sd 2.51.** The LLM teacher contributes nothing measurable to the student net.
Note which seed shows +3.14: seed 0, the one `train_mixed.py` hardcodes, and therefore the
only number a single run would ever have produced.

Two things follow.

1. **Seed variance is ±2 points and nothing in the repo reports it.** The headline ex-sweep
   figure ranges 84.96–89.20% with no change to data or model. Every accuracy number in
   this project carries an error bar that size. Two runs differing by less than that are
   not different — including several comparisons in `FIXES.md`.
2. **The mechanism is right; the dosage is not.** 89 rows against 36,783 is 0.24% of the
   mix. The traces also cover 5 of 8 families — `reactive`, `congestion` and `hidden_term`,
   exactly the weak ones, have zero LLM rows. Making the distillation claim true needs
   thousands of rows across all eight families, which is a real LLM spend, not a flag.

**Where the LLM demonstrably does decide policy:** `data/policy_v3.json`, provenance
"claude-cli induction, held-out validated". Six rules, induced by the LLM, gated at 0.95
held-out precision. Those rules take the held-out `sweep` family from 0% (net alone) to 47%
(hybrid, 30 episodes). That is a measured LLM contribution to the deployed policy and it
survives ablation.

So the supportable sentence is not "an LLM teacher was distilled into the student net". It
is **"an LLM induced the transferable rule layer; the net was distilled from the oracle"**.
Reproduce with `train/ablate_llm.py`; result archived at `data/student_v9/llm_ablation.txt`.

### 3.2 Open regressions

- **`congestion` fell 66.4% → 57.1%** in the fixed world. The ablation above puts this in
  perspective: congestion swings 57%–77% across seeds with no change at all to the data.
  Part of that "regression" is seed noise. It still deserves attention — congestion is
  where a false positive costs a wasted hop — but it needs a multi-seed measurement before
  anyone tries to fix it.
- **`hidden_term` is 43.2%** — the weakest trained family — and its anomaly gate never
  trips at all.
- **`FP(fading→jam)` moved 0.000 → 0.012.** Still excellent. Note that the 0.000 was
  partly structural: a rogue agent also scored it.
- **`neighbor_probe` remains inert.** The call works in refsim but its answer has no
  percept feature, so probing a jammed-but-alive peer is indistinguishable from not
  probing. Fixing it changes `N_FEATURES` and invalidates every bundle — correctly
  deferred, but it means one of six diagnose calls buys nothing.

### 3.3 The framing problem

The shipping decision path is a 12,480-parameter MLP plus six induced rules. The LLM is a
*teacher* that writes training rows offline; nothing calls a model at runtime. For an
agentic-systems course, calling this "an agentic system that uses an LLM to detect
jamming" will not survive a careful reading.

The honest framing is stronger than the loose one: **the contract, the action mask, the
budget, the dead-man failsafe, the verifier and the gate suite are the agentic system.
The 12 KB policy is just what sits inside it.** Lead with the harness; present the model
as the thing the harness was built to keep honest.

---

## 4. Hardware: what will actually happen

### 4.1 Compute is a non-issue — stop worrying about it

12,480 parameters: 48.8 KB at fp32, 12.2 KB at int8. 56 features, decisions at 1 Hz,
percept at 10 Hz, ring buffers of T=16 × 56 floats ≈ 3.6 KB. On a 240 MHz Xtensa LX6 the
forward pass is single-digit milliseconds even unquantised, and the percept maths (EWMA,
CUSUM, run-length, one Pearson) is microseconds. A plain ESP32-WROOM-32 fits this many
times over. **Nothing about this project is compute-bound.** Everything that can go wrong
is in the sensing.

### 4.2 The five things that will bite, in order

**1. The nRF24 RPD scan samples 1 MHz at the Wi-Fi channel centre.**
`WIFI_CH_TO_NRF(ch) = 2407 + 5*ch - 2400` gives the centre frequency only. A Wi-Fi
channel is 20 MHz wide. This is structurally the same bug as the ns-3 `JamPsd` channel
offset that was just fixed — sampling the wrong part of the band and concluding the band
is quiet. A centred barrage jammer will be seen; an offset or partial-band one, and
`sweep` in particular, will alias.
*Fix:* sweep ~5 nRF channels per Wi-Fi channel (centre, ±4, ±8 MHz) and take the max.
Costs 5× the dwell, and the scan already pays budget.

**2. The RPD→dBm mapping is invented, and three of the six induced rules depend on it.**
RPD is a one-bit comparator at roughly −64 dBm. `hal_scan` turns 8 samples into
`-100 + 35*frac` — nine discrete levels on a made-up scale. `scan_noise_spread`,
`scan_bad_frac` and `noise_delta_base` are then normalised with the *simulator's*
constants in `norm.py`. The simulator's noise distribution and an occupancy-derived
pseudo-dBm are not the same random variable, so the rule thresholds (`>=0.5`, `>=0.9`,
`<=0.3`) will not land where they landed in sim.
**This is the single most likely reason the hardware demo fails**, and it is not a code
bug — it is a missing calibration step. Budget half a day: jammer off / jammer on /
jammer sweeping, record raw `frac` per channel, re-fit the mapping and the norm
constants, then re-check the rules against the recorded traces.
Also note a jammer below −64 dBm at the receiver is invisible to RPD. On one desk you
will be at −30 dBm and fine. At 10 m you may not be. Keep the demo on one table.

**3. `rx_ctrl.noise_floor` is sampled per received packet.**
Even where the field is non-zero, it only updates when a frame decodes. Under barrage,
frames stop — so S2, the decisive feature, freezes at its last pre-jam value exactly when
it needs to move. `hal_have_noise()` correctly refuses to trust a zero at boot, but it
does not detect *staleness* at runtime. A working-but-stale noise floor is more dangerous
than a dead one, because nothing flags it.
*Fix:* timestamp the last noise sample; if it is older than ~1 s, treat noise as
unavailable for that tick and take S2 from the nRF24.

**4. Half the action space does not exist on hardware.**
The contract offers 6 diagnose + 8 act calls. The HAL implements eight of the fourteen:
`spectrum_scan`, `neighbor_probe`, `silent_listen`, `hop_channel`, `set_tx_power`,
`change_tdma_slot`, `declare_link_lost`, `no_op`. Missing: `reroute` (ESP-NOW has no
routing layer), `fallback_to_lora` (the radio is not in the BOM), `move`, `mobility_test`,
`load_test`, `channel_hop_probe`. `controller._available()` builds the mask from
*scenario* flags, not from a device capability set, so on hardware it will cheerfully
offer `move` and `reroute`.
*Fix:* add `hal_capabilities()` returning a bitmask and intersect it into `_available()`.
Otherwise the first thing an examiner sees is the agent choosing an action that silently
does nothing.

**5. TDMA slot sync is a comment, not an implementation.**
`hal_set_slot` stores `slot` and `n_slots`; the transmit gate it describes — enforced
against the local monotonic clock, re-synced on each beacon — is not written. ns-3 gives
every node one perfect global clock and no guard interval, so the simulator cannot catch
this. The guide's own risk list is right that a 100 ms frame with a 5 ms guard is
comfortable for ESP32 crystals; the point is that nothing implements or tests it.
*Fix:* implement the gate, or drop `change_tdma_slot` from the Mode H mask. Do not demo
an action whose effect lives in a comment.

### 4.3 Prediction, demo by demo

| Demo | Verdict |
|---|---|
| **Refusal**: walk a node out of range → says `fading`/`node_loss`, does **not** hop | **Will work, cheapest, most impressive.** Needs only PDR and heartbeat gap — no scan, no calibration. Lead with this one. |
| **Spot jammer on ch 6**: belief flips to `spot`, hops, PDR recovers | **Will work** after the RPD calibration in §4.2.2. Two or three evenings once boards arrive. |
| **Barrage**: all channels hot, agent refuses to hop and declares | **Will work.** It is the easiest spectral signature and failing to hop is a cheap correct answer. |
| **Sweep** | **Unlikely without the multi-frequency fix.** Centre-only sampling with 8 dwells per channel aliases against a sweeping source — this is the case the RPD scan is worst at. |
| **Reactive (S4′)** | **The one I most want to see, medium confidence.** It needs only the local clock, so the statistic will transfer. The risk is upstream: building a genuinely reactive jammer (listen → transmit within ~3 ms) on an ESP32 is hard. If you end up faking reactivity by exploiting beacon periodicity, **say so in the writeup** rather than claiming a reactive jammer you did not build. |

### 4.4 The realistic Mode H scope

Stage H5 is "port five files to C". `percept/features.py` alone is 494 lines of stateful
maths that must stay byte-compatible with the sim's constants. The rules-only path is the
right call and the guide already makes it: `policy_v3.json` is six rules, roughly thirty
lines of C, and it is precisely the part that transfers — its leading rules are the
spectral ones built on S2 and S5. The net adds accuracy the hardware demo does not need
and a whole class of porting bugs it cannot afford.

Ship rules-only on hardware. Show the net in simulation.

---

## 5. What is left, prioritised

**P0 — before showing an instructor**
1. ~~State one number honestly~~ → now: **85.0% ± 2.2 on trained families, net 0% on the
   held-out family, hybrid 47%.** The error bar is not optional; see §3.1b.
2. ~~Fix or retire `abstain_threshold = 0.0`~~ — **done**, now 0.68, and the OOD result
   in §3.1 is worth more than the fix.
3. ~~Re-run `export_int8.py`~~ — **done**, 16.0 KB, real C header, gate met on the
   deployed decision.
4. ~~Reframe the pitch around the harness, not the model~~ — and now also around the
   correct LLM sentence (§3.1b): the LLM induced the rule layer, it did not teach the net.
5. **New:** report seed variance on every accuracy figure, or stop comparing runs that
   differ by less than it.

**P1 — before touching hardware**
5. `hal_capabilities()` → intersect into the action mask.
6. Multi-frequency RPD sweep per Wi-Fi channel.
7. Noise-floor staleness check.
8. Bench-calibrate the RPD → feature mapping before trusting any rule threshold.
9. Implement the TDMA transmit gate, or remove the call from the Mode H mask.

**P2 — real work, not blockers**
10. Close the `congestion` 66.4 → 57.1 regression.
11. `hidden_terminal` never trips the anomaly gate.
12. Add the `neighbor_probe` result feature (changes `N_FEATURES`; do it with the next
    retrain, not as a bolt-on).
13. Decide `set_tx_power`: score the reverse link too, or change the fading playbook.
    Right now the recommended remedy for the most important family provably cannot move
    the metric it is judged on.
