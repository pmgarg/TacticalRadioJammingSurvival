# Fix log — what was changed, what was tested, what is still open

**Date:** 2026-09-13 · Companion to `AUDIT.md` (the findings) and `DESIGN.md` v2.0 (the target).
Every "verified" line below was actually executed. Every "UNVERIFIED" line says why.

Run everything with `make gates` from the repo root.

---

## 0. Headline: the v7 generalisation claim does not survive a clean split

`RESULTS_LLM.md` §7 says:

> "`sweep` is the **held-out family — never trained on** … it is now **90%**. That is the
> generalisation claim the brief actually cares about."

It was trained on. `data/traces_ns3/train.jsonl` carried **1,742 sweep rows across 43
scenarios, every one of them in `corpus/test`**; val added 7 more. 50 of the 60 held-out
sweep scenarios were seen. The contamination was **exclusively** sweep — train ∩ test for
every other family was zero.

Retrained with sweep genuinely absent (`data/student_v8_clean/`), measured on the same
held-out ns-3 test rows:

| family | n | v7 (leaked) | v8 (clean) | Δ |
|---|---:|---:|---:|---:|
| barrage | 234 | 100.0% | 100.0% | +0.0 |
| congestion | 259 | 95.0% | 91.5% | −3.5 |
| fading | 258 | 100.0% | 100.0% | +0.0 |
| hidden_term | 366 | 66.1% | 56.8% | −9.3 |
| node_loss | 332 | 100.0% | 100.0% | +0.0 |
| reactive | 292 | 54.8% | 59.2% | +4.5 |
| refusal | 607 | 100.0% | 100.0% | +0.0 |
| spot | 291 | 84.2% | 83.8% | −0.3 |
| **sweep** | **405** | **87.9%** | **0.0%** | **−87.9** ← held out |
| **overall** | **3044** | **88.0%** | **75.3%** | −12.7 |

FP(fading→jamming) is 0.000 for both.

Closed loop on `corpus/test`, seen families only: **student 98.5%, baseline 67.7%,
expected cost 0.031 vs 1.431** — a 46× cost improvement, and that number is now honest.
On the novel attack family the student scores **0%**.

**This restores the project's own earlier finding.** `RESULTS.md` §8.1 already said
*"with sweep held out entirely the student scores 0%; trained with it, 90%"* — that was
the true version, and v7 reversed it by accident. A measured 0% on a genuinely novel
attack is a better capstone result than a fabricated 90%, because it is the honest answer
to the question the brief actually asks.

**No ns-3 re-run was needed.** Trace rows carry their `scenario`, so the leak was filtered
out (`python3 -m train.filter_traces`) rather than re-simulated. Six hours saved.

---

## 1. Fixed and verified

### Evaluation integrity

| # | Fix | Verification |
|---|---|---|
| F4.1 | **One canonical split.** New `scenario/split.py` — `split_of(name)` is a pure function of the scenario name, so training and evaluation cannot disagree. `assert_no_leak()` raises rather than reporting a contaminated number. | Reproduces the on-disk `corpus/` split exactly: **540/540 scenarios, 0 disagreements** |
| F4.1 | **`train/filter_traces.py`** removes leaked rows from existing traces without re-simulating. | Dropped 2,274 rows (all sweep); re-run reports 0 |
| F4.2a | **Ground truth out of the deployed failsafe.** `controller.py` no longer reads `sc.lora_jammed`. The escape test is now sensing-derived: no fallback radio, **or** the fallback was already spent and the link did not return. Better behaviour too — it must *try* the escape before declaring there is none. | Refusal gate still 100%; `test_rogue_control.py` scans the function's source (comments stripped) for any truth field |
| F4.2b | **The verifier computes `survived`.** The controller recorded `recovered`; the truth-dependent metric is now computed by the only component allowed to see truth. | Dashboard shows `log.survived = None` → verifier says `False` for the sweep episode |
| F4.3 | **The cost rule is applied.** `Decision.declared` carries the minimum-expected-cost class; `belief` stays the honest posterior; the verifier scores `declared` and falls back to argmax for agents that do not separate them. | `declared` present in every trace row |
| F4.4 | **Conformal abstention calibrated on the statistic inference uses** (min-cost class), not `max(proba)`; the threshold is computed, not hardcoded to 0.0. | Printed at train time; lands at 0.000 on this data (error is already under target — recorded honestly rather than forced) |
| F4.4 | `contract_version` written from the contract file, not hardcoded | bundle carries 1.2.0 |
| F4.6 | **`export_int8.py` no longer crashes** on a head that fails the equivalence gate — it emits fp32 arrays for that head instead of `KeyError`, which is what left the 162-byte stub header the report called "emitted" | code path fixed; re-run `make` target after the next train |
| F4.7a | **DAgger round 2 no longer discards round 1.** It reads the moved `_r{k}/train.jsonl` and raises if a prior round is missing rather than quietly training on less. `os.system`+grep replaced with a checked `subprocess.run`. | — (needs sklearn; logic reviewed, not executed) |
| F4.7b | **`train_mixed` no longer double-counts the base corpus.** `_r1/train.jsonl` is `base + round-1`, so only the genuinely new rows are added now, and the count is printed. | ran clean: `refsim=3478 ns3=13700 bridge=4584×4` |
| F4.8 | **Two vacuous assertions removed.** `ctrl.rung >= 0` (true for every possible run) and a hardcoded `check(..., True)` replaced with real budget and cost-matrix assertions. | safety suite 14/14, all meaningful |
| F4.9 | **`run_episode.py` can run every agent** — `baseline`, `student`, `rule`, `oracle` (labelled PRIVILEGED in the output), with `--bundle` | all four run |
| — | **Gates report N/A, not FAIL, when there is nothing to measure.** One spot episode no longer "fails" the fading false-positive gate. | verified in CLI and dashboard |

### Architecture / honesty

| Fix | Detail |
|---|---|
| `TeacherAgent` → **`OracleLabeller`** | It is handed `true_cause`; calling it the teacher is the error `DESIGN` A5b forbids. Alias kept for compatibility; all call sites migrated; `evaluate.py` prints it as `oracle*` with a ceiling footnote |
| `PLAYBOOK` moved to **`agent/policy_table.py`** | The on-device student was importing, at runtime, the one module that exists only to cheat |
| **Contract 1.1.0 → 1.2.0** | `listen_test` and `transmit_probe` removed: no simulator implemented them, so they cost budget and returned nothing (`DESIGN` A13) |
| **`neighbor_probe` implemented in refsim** | It is the node_loss distinguishing test and it was in the contract, the oracle's confirm table and the LLM prompt, with no implementation anywhere |

### Tarball merge (S0)

`capstone_final.tar.gz` and `student_v7_final.tar.gz` are merged into the tree: `harness/`
(10 modules), `gateway/provider.py`, `agent/llm_teacher.py`, `agent/rule_agent.py`,
`train/reextract_ns3.py`, and `data/{policy_v3,mutations,llm_golden_v7,…}`. Only
`percept/features.py` and `percept/norm.py` differed from the repo copies — the tarball's
are newer (the relative hot-channel threshold, the 3 dB peak margin, two-scan periodicity)
and were taken. `Claude outputs/` is now redundant; delete it when you commit.

### New gates — `make gates`

| Gate | What it proves | Status |
|---|---|---|
| **G0** repo integrity | the agentic code is in the tree and imports | green |
| **G4** leakage | no scenario in both train and test | green (was 50) |
| **G1** contract effects | every call changes something, in some family | green, 2 known-open |
| **G3** safety + rogue | the envelope contains an adversarial agent | 14/14 + 10/10 |
| **G2** fidelity | refsim vs ns-3 | **honestly skipped** — no cached CSVs; it used to print "0/0 pairs agree", which is not agreement |

`tests/test_rogue_control.py` is the one that changes how a number should be *reported*: a
deliberately malicious agent also scores FP(fading→hop) = 0.000, because `hop_channel` is
never in the action set on a clean spectrum. **That is the mask, not the model.** Report it
as a structural guarantee and report the belief-level FP as the model's number.

---

## 2. Changed but NOT verified — the ns-3 C++

ns-3 lives outside the mounted folder, so none of this was compiled or run. Re-read and
brace-checked only. **Compile before trusting.**

| Fix | Where |
|---|---|
| F3a Reactive jammer: PSD pointed at the victim's channel (was stuck on ch 1), armed at its scheduled onset (was armed from t=0 during `main()`), burst length now `period × duty = burst_ms` (was radiating 10 ms instead of 1.5 ms) | `jamming-sim.cc:182-215, 250-252, 959-980, 1005-1016` |
| F3b TX-shadow counters: single sampler with a cached read, so `tick` and `decide` firing at the same instant no longer zero each other | `jamming-sim.cc:524-554, 1266-1269, 1477-1479` |
| F3d `flow.0.start`/`stop` honoured — the hidden-terminal interferer no longer runs from t≈0.55 s against a declared onset of 18 s | `mesh-node-app.{h,cc}`, `jamming-sim.cc:846-853` |
| F3c DATA packets no longer counted as beacons (per-link PDR was pinned at 1.0 for the flow source) | `mesh-node-app.cc:258-272` |
| F3 TDMA actually enabled: `--tdmaSlots=4` passed from both call sites, so `change_tdma_slot` stops being a no-op | `bridge_server.py:111-118`, `gen_ns3_corpus.py:27-32` |
| F3g Spectrum analyzer tracks the agent instead of staying at its initial position | `jamming-sim.cc:498-522, 1107-1110, 1548` |
| — `tx_power` in the bridge state reports the current value, so the agent can observe its own change | `jamming-sim.cc:777-779, 1488, 1527-1529` |
| F3f **not fixed** — the suspected 5 MHz / one-channel analyzer offset needs a measurement first; a `FIXME(F3f)` marks the site and names the experiment | `jamming-sim.cc:298-309` |

> **Consequence, and it is unavoidable:** enabling TDMA changes beacon timing across the
> whole corpus, and the reactive and flow-start fixes change the world those episodes
> describe. **Existing `data/traces_ns3/` is not comparable to anything generated after
> this.** After the C++ compiles clean, run `make ns3-corpus` (~6 h) and retrain. That is
> the one place a full re-run genuinely is required — you cannot fix the simulator and keep
> the old traces.

---

## 2b. The C++ is now COMPILED, RUN and MEASURED

`bash sim_ns3/build_and_check.sh` built clean and ran all 9 families (591 percept rows each,
exit 0). `python3 capstone/tests/test_world_ns3.py` then checked each fix against the measured
CSVs: **14 passed, 2 failed, 1 honestly skipped.**

### Confirmed working

| Fix | Evidence |
|---|---|
| **F3a reactive jammer** | **S4' = +0.390 for reactive**, the highest of any family (congestion +0.200, sweep +0.139, hidden_term +0.088). Loss after our own TX is 56/141 = **40%**; while silent it is 4/1059 = **0.4%**. That is a textbook reactive signature and it was *identically zero* before |
| **F3a(b) arming** | pre-onset floor is quiet at −94.0 dBm — the jammer is no longer armed from t=0 during `main()` |
| **F3b TX-shadow race** | counters non-zero in every family that has them; they used to be zero in every live run |
| **F3 TDMA** | 1,801 tx_attempts recorded with `--tdmaSlots=4`; `change_tdma_slot` is no longer a no-op |
| **S2 intact** | barrage **+24.0 dB**, spot **+25.8 dB**, sweep **+20.4 dB**, **fading +0.0 dB**, dead peer +0.0 dB — reproducing `FIDELITY.md` exactly. The false-positive defence is real and measured |

### Two things the gate found that are still open

**1. F3f is no longer a suspicion — it is measured and quantified.** A spot jammer configured
on channel 6 radiates hottest on **channel 7**. The arithmetic from ns-3's own source
(`ism-spectrum-value-helper.cc`): band index *k* has centre `2389.5 + 5k` MHz, and
`CreateTxPowerSpectralDensity(p, ch)` puts full power in bins `ch+3 … ch+6`, i.e. a 20 MHz
block centred at **2412 + 5·ch** MHz — while Wi-Fi channel *ch* is centred at **2407 + 5·ch**.
**Exactly one channel high, every time.**

Consequence: S5's *hot-channel identity* is shifted by one, so `hop_channel` picks its
"cleanest" channel from a shifted map. The scenarios still work (20 MHz channels overlap
heavily, so a jammer on 7 still crushes 6), which is why this hid.

**Do not fix this before a demo.** Correcting it changes every ns-3 measurement, and the
shipped student was trained on the shifted distribution — fixing the world without
regenerating the corpus makes the demo *worse*. Fix and regenerate together.

**2. `hidden_terminal` per-link PDR is still 95% pinned at 1.0.** The DATA-counted-as-beacons
fix (F3c) landed — congestion improved to 78% — but hidden_terminal has a second cause: the
offline CSV computes PDR over a 100 ms window at 10 Hz beacons, so `expect = 1` and the ratio
quantises to {0, 1} (AUDIT F3e, never fixed). The live bridge works around it with a 1 s
sliding window; the corpus that trains the model does not.

### One check that cannot mean anything yet

`data_tx` is written from `apps[0]->DataSent()` regardless of which node the flow source
actually is, so it reads 0 whenever the source is not node 0 (AUDIT item j). The
`flow.0.start` check is reported as **SKIP**, not PASS — a check that passes because both
numbers are zero is worse than no check.

### Also worth saying

Two of the failures above were found by a gate whose **own first version was wrong**: it read
`noise_dbm` (the nominal floor, which never moves) instead of `meas_floor_dbm`, and it asserted
that a reactive jammer should lift the min-hold noise floor. It cannot: min-hold takes the
*minimum* over the window, so a 1.5 ms burst every ~17 ms is invisible to it by construction —
which is exactly why the design identifies reactive with S4' and not S2. Both gate bugs are
fixed; the episode is a good argument for why a gate has to be read as carefully as the code
it checks.

---

## 3. Still open — decisions, not bugs

**D9 — `set_tx_power` cannot move the metric it is judged on.** G1 found this, and it is
the sharpest open problem. Raising *our* transmit power cannot improve the frames *we
receive*, and `pdr`/recovery are measured on the forward link — so the action is inert by
physics, not by omission. But `PLAYBOOK["fading"] = set_tx_power`, which means **the
recommended remedy for the single most important family provably cannot help**. Either
score the reverse link too (refsim already models it, `refsim.py:401`), or change the
fading action to `reroute`/`move`. Pick one deliberately.

**D10 — `neighbor_probe`'s answer never reaches the agent.** Implemented in refsim now, but
the percept layer has no probe-result feature, so probing a live-but-jammed peer is
indistinguishable from not probing. Fixing it changes `N_FEATURES` and invalidates every
trained bundle — schedule it with the next retrain.

**Still to do from `DESIGN` §12:** point `harness/run_llm.py` at the ns-3 bridge instead of
`RefSim` (S3), and distil the student from **LLM-teacher traces** rather than oracle traces.
That last one is the thesis the brief actually grades, and it has still never been run.

Also: `data/traces_all/val.jsonl` was **48-feature** — two contract versions stale, from
before the telemetry work. Regenerated at 56. Worth asking what else was measured against it.

---

## 4. How to reproduce everything here

```bash
make gates                    # G0, G4, G1, G3, G2
make corpus                   # whole corpus through the clean student -> data/runs
make eval                     # three-way table on the held-out split
make dash                     # http://127.0.0.1:8765
cd capstone && python3 -m train.filter_traces      # leakage check on its own
```

Training needs scikit-learn, which is not installed on every machine here; `make train`
carries the exact command. `data/student_v8_clean/` holds the clean bundle plus the two
comparison reports this document quotes.

---

## 5. Round two — the four remaining gaps

### D9 `set_tx_power` — resolved, not patched

Link health is now the **worse of the two directions**, using the peer's own reciprocal
report when it is fresher than 3 s (`controller.py::_link_health`). An 802.11 data frame
that is not ACKed is a lost frame, and the ACK depends on the peer hearing *us* — so a link
is only usable when traffic flows both ways.

This keeps the physics honest rather than pretending our TX power improves our RX: under
fading both directions degrade and raising power lifts the reverse one, so the action helps;
under jamming at our location the reverse link stays healthy and raising power correctly does
**not** help. **G1: `set_tx_power` now effective in 7/7 families** (was inert in all of them).

### The hybrid student — built, and it closes the held-out gap

`agent/hybrid.py`. Measured over 70 held-out episodes:

| | baseline | net only | **hybrid** |
|---|---:|---:|---:|
| classification, overall | 55.7% | 85.7% | **91.4%** |
| expected cost | 1.600 | 0.171 | **0.086** |
| **sweep — never trained on** | 0% | **0%** | **40%** |

The rule `scan_noise_spread >= 0.5 AND fade_runlen_mean >= 0.2 -> sweep` fires on a family
the net has never seen, because it is a statement about physics rather than about training
rows. A live trace:

```
rule sweep_periodic_fade (held-out precision 1.00) -> sweep; hop ahead of the sweep
                                                     [net said congestion]
```

**And a finding about the arbitration.** Letting every accepted rule lead made things *worse*
— congestion fell 100% → 75%. `reactive_decorrelated_shadow_loss` and
`congestion_selfshadow_zero_silent` share `tx_shadow_loss_delta >= 0.3` and are separated
**only** by `rssi_pdr_corr` — the feature `FIDELITY.md` proved is largely unmeasurable.
Induction had found a discriminator this project had already shown does not exist, and its
1.00 precision was measured on sampled states, not the closed-loop states the agent reaches.

Rules that lean on a measured-unreliable feature (`rssi_pdr_corr` S1, `loss_load_corr` S10)
are now demoted to voting only, and matching rules are tried most-specific-first. The three
rules left leading are **exactly** the spectral ones built on S2 and S5 — the two features
cross-validation found decisive. That was not designed in; it fell out of the rule.

### Path B → Path C — wired and exercised end to end

- `harness/run_llm.py --world ns3` drives the **live ns-3 bridge** (`DESIGN` §9.2). Every
  headline teacher number should come from there: a refsim-only *student* scores 96% on
  refsim and 50% on ns-3, and there is no reason a refsim-only *teacher* is safer.
- `sim/bridge_server.run(..., agent_obj=)` accepts a caller-supplied agent and now returns an
  `episode_log` in the controller's exact shape, so **one verifier scores both worlds**.
- Traces carry `features` and `available`. Without them a trace records what the teacher
  *said* but not what it *saw*, so it could be audited and never distilled from — which is
  the one thing traces exist for.
- `train/llm_traces_to_rows.py` converts traces to training rows. The **cause** label still
  comes from the simulator (free, unlimited, and better than the teacher's guess); the
  **call** target comes from the teacher, which is what the simulator cannot label.
- `train_mixed.py --llm` consumes them, upweighted like bridge rows, and says so loudly when
  they are absent.
- `gateway.provider.StubProvider` is a deterministic offline stand-in, so the whole path is
  testable with no model. `make llm-stub` runs it end to end. **Never report a stub run as a
  teacher result.**

Only the real run remains, and it needs a working `claude` CLI: `make llm`.

### Mode H — the adapter exists and self-tests

`firmware/hal.h` (six functions), `firmware/hal_host.c` (desktop implementation),
`firmware/hal_esp32.c` (ESP-NOW + nRF24 RPD), `firmware/hal_selftest.c`.

**`make hal` → 15 passed, 0 failed**, with no hardware: a quiet world reads quiet; jamming
lifts the floor >10 dB while fading does not; the scan finds the hot channel and leaves the
rest clean; S4′ is strongly positive for reactive while the floor stays quiet; every action
changes something; a dead peer goes silent with the channel still clean.

The ESP32 file probes `rx_ctrl.noise_floor` once at boot and, if it reads 0
([esp-idf#1751](https://github.com/espressif/esp-idf/issues/1751)), reports
`hal_have_noise() == false` so S2 and S5 come from the nRF24 sweep instead. A sensor that
fails to "all clear" is worse than no sensor.

### The channel offset — measured, fixed, proved

`JamPsd()` replaces `SpectrumValue5MhzFactory::CreateTxPowerSpectralDensity`. On that band
model index *k* spans [2387+5k, 2392+5k] MHz, so the stock helper's bins ch+3…ch+6 form a
20 MHz block centred at **2412 + 5·ch** while channel *ch* sits at **2407 + 5·ch** — one
channel high, every time. Bins ch+2…ch+5 land it correctly; verified arithmetically for all
eight channels, with bounds-checked indices so ch=1 no longer writes index −1.

### Still to run (needs your machine)

1. `bash sim_ns3/build_and_check.sh` — rebuild with the channel-offset, PDR-window and
   `data_tx` fixes, then `python3 capstone/tests/test_world_ns3.py`.
2. `make ns3-all` — regenerate the corpus. **Required**: TDMA and the channel fix both change
   the world, so old traces are not comparable to new ones.
3. `make train && make eval && make hybrid` — retrain on the regenerated corpus.
4. `make llm` — the one run that still needs a real model.

---

## 6. Round three — built, regenerated, retrained, re-validated (all of it run, not prescribed)

ns-3 was built **from source in the cloud container** (aarch64 Linux, g++ 13, optimized,
modules `wifi spectrum propagation olsr mobility applications internet energy flow-monitor
stats`). The Mac build is arm64 Mach-O and the device shell is Linux, so the binary there is
genuinely unrunnable from here — building a Linux one removed the dependency on anyone
typing a command.

**Build: 633/633 targets, clean**, including `jamming-sim.cc` and `mesh-node-app.cc` with
every F3 fix. That is the compile validation that was previously only `-fsyntax-only`.

### World gate: 17 passed, 0 failed, 0 skipped

| Fix | Measured |
|---|---|
| **F3f channel offset** | spot jammer configured on ch 6 now radiates **hottest on ch 6** (was ch 7) |
| **F3a reactive** | S4′ **+0.433**, highest of any family; floor stays at −100.9 dBm vs barrage −77.1, so it is no longer a de-facto barrage |
| **F3d flow.0.start** | `data_tx` **0 before onset, 702,400 after** — the check that was previously vacuous now means something |
| **F3c per-link PDR** | hidden_terminal pinned-at-1.0 fraction **95% → 57%**, congestion **78% → 36%** |
| **F3b TX-shadow** | counters non-zero in every family that has them |
| **TDMA** | 1,801 tx_attempts with `--tdmaSlots=4` |
| **S2 intact** | barrage +23.7 dB, spot +27.0 dB, sweep +18.1 dB, **fading +0.0**, dead peer +0.0 |

### Corpus regenerated: 540 episodes, 0 failures

train 332 → 13,700 rows · val 83 → 3,458 · test 125 → 5,102. Leakage check clean: `sweep`
appears **only** in test. Refsim traces regenerated on the same corpus.

### `student_v9` — retrained on the fixed world

The world fixes show up directly in what the model can learn. Same architecture, same split,
same code — only the simulator changed:

| family | v8 (old world) | **v9 (fixed world)** |
|---|---:|---:|
| **reactive** | 50.7% | **99.0%** |
| node_loss | 62.3% | **89.2%** |
| spot | 67.7% | **89.3%** |
| hidden_term | 25.4% | **43.2%** |
| congestion | 66.4% | 57.1% |
| barrage / fading / refusal | 100 / 100 / 99.8% | 100 / 98.8 / 100% |
| **sweep (held out)** | 0.0% | 0.0% |

Excluding the held-out family: **66.9% → 74.2%**. The reactive jump is the headline — that
family was unlearnable while the jammer sat on the wrong channel, armed from t=0, radiating
10 ms bursts instead of 1.5 ms.

### Final three-way, closed loop, seen families

| | baseline | oracle\* | student |
|---|---:|---:|---:|
| classification | 67.7% | 100.0% | **98.5%** |
| expected cost | 1.431 | 0.000 | **0.031** |
| FP gate / refusal gate | PASS / PASS | PASS / PASS | **PASS / PASS** |

\* handed the true cause; a ceiling, not a competitor.

### And with the hybrid, over 70 held-out episodes

| | baseline | net only | **hybrid** |
|---|---:|---:|---:|
| classification | 55.7% | 84.3% | **91.4%** |
| expected cost | 1.600 | 0.200 | **0.100** |
| **sweep — never trained on** | 0% | **0%** | **50%** |

Live trace on the held-out family:
`rule sweep_periodic_fade (held-out precision 1.00) -> sweep; hop ahead of the sweep [net said spot]`

### What is still genuinely yours to run

**One thing:** `make llm`. The `claude` CLI exists in the device VM but is disabled there, so
the real teacher run needs your machine. Everything it depends on is wired and exercised end
to end with a deterministic stub (`make llm-stub`) — the machinery is proven, the teacher is
not.

---

## 6. Round five — the two dead design claims, and what measuring them found

| id | change | evidence |
|---|---|---|
| F5.1 | **`abstain_threshold` was 0.0.** Not a bug in `conformal_threshold` — a bug in the calibration SET. `train_mixed.py` pooled refsim-val (97.6% acc) with ns3-val (88.4%); pooled error at q=0 is 9.3%, just under the 10% target, so the smallest valid cut is 0.0 and abstention never fires. Now calibrated on the deployment world alone | threshold **0.68**, retains 96.6% at 9.7% error. Closed loop: net 84.3→85.7%, expected cost 0.200→0.171; hybrid cost 0.100→0.086 |
| F5.1a | New `train/calibrate_abstain.py` — re-derives the threshold post-hoc for any bundle (calibration changes no weight, so this needs no retrain) and runs the novelty check | `data/student_v9/abstain_calibration.json` |
| F5.1b | `Decision.abstained` added. The verifier scored `declared or top`, so an abstention fell through to argmax and was credited/charged as a claim the agent refused to make. `controller.py` could likewise set `declared_jamming` off an abstained tick. Both masked while the threshold was 0 | `verify/verifier.py`, `agent/controller.py`, `agent/api.py` |
| F5.2 | **`student_weights.h` was a 162-byte stub.** int8 export now runs against v9. Two changes: (a) **bias correction** — fold `mean(x) @ (Wq−W)` into the bias; cause head 98.67→99.19%, call head 99.25→99.77%; (b) the gate now scores the **deployed decision** (min-expected-cost class, on non-abstained ticks) rather than unconditional argmax, because that is what `student.py` acts on | cause all-int8, call int8-except-output. **16.0 KB** vs 48.8 KB fp32. Agreement on acted ticks **0.9997**. Header compiles under `gcc -fsyntax-only` |
| F5.3 | **`train/ablate_llm.py`** — measures whether the LLM teacher changes the student net at all | it does not: **+0.14 ± 2.51 points over 4 seeds**. `data/student_v9/llm_ablation.txt` |

### What measuring them actually found

**Abstention does not detect novelty.** On the held-out `sweep` family the model is 0.000
accurate at mean posterior 0.961 and abstains on only 4.2% of ticks — *less* than the 6.4%
it abstains on in-distribution. The softmax posterior carries no out-of-distribution
signal. Conformal abstention buys in-distribution risk control (it correctly pulls back on
`hidden_term` 29% and `congestion` 16%) and nothing more. Catching the unseen family is the
induced rules' job. Do not advertise the threshold as an OOD guard.

**The LLM teacher does not teach the net.** 89 rows against 36,783 is 0.24% of the mix, and
across four seeds the effect is inside the noise. The +3.14 that a single run would have
shown is at seed 0 — the seed `train_mixed.py` hardcodes. The LLM's real and measurable
contribution is `data/policy_v3.json`: six induced rules, held-out precision gated, which
take `sweep` from 0% to 47%. The supportable claim is "an LLM induced the rule layer", not
"an LLM taught the student".

**Seed variance is ±2 points and no artefact reports it.** Ex-sweep accuracy ranges
84.96–89.20% with identical data and model. Several comparisons elsewhere in this file are
smaller than that and should not be read as differences.
