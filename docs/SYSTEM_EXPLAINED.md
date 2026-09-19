# How the whole thing actually works — ns-3, the teacher, the student, the device

Written to be read top to bottom by someone who has not seen the code. Every claim here is
something you can run. Where something does **not** exist, it says so.

---

## 1. Vocabulary: scenario vs episode

These are different things and the distinction matters everywhere else.

**A SCENARIO is a static description of a world.** A YAML file. Node positions, path-loss
exponent, fading model, which jammer exists, its type/EIRP/duty/channels, when it switches
on, and the ground truth (`truth.cause`, `truth.onset_t`, `truth.recoverable`). It contains
no time series. It is a *recipe*. There are **540** of them; `docs/SCENARIOS.md` lists every
one.

```yaml
name: spot_10071          family: spot           duration: 66.8
truth: {cause: spot, onset_t: 21.1, recoverable: true}
n_nodes: 6                channel: 8             n_channels: 8
jam.0: {type: spot, eirp: 17.3 dBm, duty: 0.7, channels: [8], dwell_ms: 200}
ev.0:  {t: 21.1, type: jammer_on, target: J1}
```

**An EPISODE is one execution of a scenario.** ns-3 runs the recipe for ~60 s of simulated
time, an agent is in the loop making decisions, and the result is a trace plus a verdict.
Same scenario + same agent + same seed = same episode, every time.

**One scenario can produce many episodes** — one per agent. The same `spot_10071` is run by
the baseline, by the LLM teacher, and by the student, and that is exactly how they are
compared.

**A FAMILY is a group of scenarios sharing a true cause.** Nine of them: `barrage`, `spot`,
`sweep`, `reactive` (four jamming types), `fading`, `node_loss`, `congestion`,
`hidden_term` (four benign look-alikes), and `refusal`.

> **`refusal` is not a cause.** A refusal scenario is an *unrecoverable barrage* where the
> correct behaviour is to stop trying and declare the link lost. Its `truth.cause` is
> `barrage`. This cost us a real bug — see §10.

---

## 2. What ns-3 is actually simulating

`sim_ns3/jamming-sim.cc` (~1,700 lines) builds a real 802.11 mesh on a real spectrum model.
It is not a hand-written approximation:

| piece | ns-3 component |
|---|---|
| radio PHY | `SpectrumWifiPhy` on a `MultiModelSpectrumChannel` |
| propagation | log-distance + shadowing + optional Nakagami fading |
| routing | OLSRv1 (real multi-hop, real convergence delay) |
| jammers | `WaveformGenerator` emitting a real PSD onto the shared channel |
| our app | `mesh-node-app.cc` — beacons, data, TDMA slot gate, reverse-link reports |
| measurement | a `SpectrumAnalyzer` node co-located with the agent |

The jammer is not a "set PDR to 0.3" switch. It radiates power into the channel; frames
fail because SINR drops. That is why a jammer 64 m away at 17.3 dBm EIRP reads −64.9 dBm at
the analyzer — and the measured value matches the link budget to 0.1 dB.

**The four jammer types**

- `barrage` — wideband, every channel hot, low spread between channels
- `spot` — narrowband, one channel hot, high spread
- `sweep` — walks across channels over time (**held out of training entirely**)
- `reactive` — fires only when *we* transmit, after `delay_us`, for `burst_ms`

Reactive is the hard one and deserves a note: it destroys **what we send**, not what we
receive. Our own received beacons stay perfect, so `pdr_fast` can sit at 1.00 for an entire
episode while 65% of frames are dying. The evidence lives in the TX-side statistics
(`tx_shadow_loss_delta`, `pdr_reverse`). This is why the harness event gate has to consult
them — see §6.

---

## 3. The bridge: exactly what crosses the wire

`capstone/sim/bridge_server.py` opens a UNIX socket. ns-3 is the client. Two message types,
newline-delimited JSON.

### 3.1 ns-3 → agent, every 100 ms (10 Hz)

```json
{"t": 21.3, "channel": 8, "n_channels": 8, "n_links": 3,
 "pdr":      [0.333, 0.333, 0.333],      "rssi":     [-70.4, -80.2, -80.1],
 "rev_rssi": [-70.1, -80.0, -80.3],      "rev_pdr":  [1.0, 0.0, 0.0],
 "rev_age":  [0.073, 0.048, 0.023],      "hb":       [0.073, 0.048, 0.023],
 "band": [-100.4, -100.0, -96.4, -94.5, -70.9, -67.9, -66.1, -64.9],
 "tx_att": 5, "tx_ack": 5, "tx_defer_ms": 0.007,
 "sh_e": 12, "sh_m": 9, "si_e": 12, "si_m": 0,
 "hops_used": 0, "tx_power": 16.0, "retry": 0.31, "load": 0.107, "tx_duty": 0.35}
```

Reading it: `band[]` is the per-channel energy in dBm — channel 8 at −64.9 with channels 1–2
at −100 *is* the spot-jammer signature. `sh_e/sh_m` are beacons due/missed **inside our own
TX shadow**; `si_e/si_m` are the same **while we were silent**. The difference between those
two ratios is S4′, the reactive discriminator, and it uses only this node's clock — never a
cross-node comparison — which is what makes it portable to cheap hardware.

`rev_*` is what peers report about hearing **us**. That is the only way to tell "my
transmitter is broken" from "my receiver is broken".

### 3.2 agent → ns-3, one reply per state message

```json
{"call": "hop_channel", "channel": 3}
```

Just the call and its arguments. Nothing else crosses.

---

## 4. From wire to the 56-feature vector

```
ns-3 JSON  →  state_to_obs()  →  RawObs  →  FeatureExtractor.update()  →  56 floats
```

`RawObs` (`percept/features.py`) is the **hardware-abstraction port** — the one struct that
sim and hardware both fill. `FeatureExtractor` turns a stream of `RawObs` into 56 features,
all normalised to [−1, +1]: EWMAs at three time constants, a CUSUM change detector, fade
run-lengths, a Pearson correlation between RSSI and PDR, spectrum-scan summaries, budget
counters, and the TX-side statistics.

**The rule that makes this an agentic system:** spectrum data reaches the feature layer
**only after the agent pays for a `spectrum_scan`**. The bridge holds `obs.scan = None` and
calls `ex.note_scan()` on the tick *after* the agent spends a scan from its budget. Before
that, `scan_age` reads "never scanned" and the scan columns are dead. Gate `N3` in
`harness/verify_links.py` enforces this and fails if any tick reports a fresh scan the agent
did not buy.

> This is the single biggest reason the old 85% number was inflated: the CSV corpus handed
> the spectrum over on **100% of ticks**. The same student scores 82.4% per-window there and
> 33.8% per-window on the live bridge where it must buy the evidence.

---

## 5. Decisions happen at 1 Hz, not 10 Hz

Percept runs at 10 Hz so the EWMAs and change detectors have resolution. The **agent is
asked once per second**. Between decisions the controller replies `no_op`.

---

## 6. How a decision is made

```
features(56) → controller builds the ACTION MASK → event gate → render panel
   → LLM → parse → validate against the mask → Decision → apply → record
```

**The action mask** (`agent/controller.py`, mirrored in the bridge) is what makes refusal
structural rather than a prompt instruction. A call the agent may not make is simply not
offered. `hop_channel` appears only when a scan is fresh **and** the scan says hopping would
help. `spectrum_scan` disappears when the budget is spent or the minimum interval has not
elapsed. `fallback_to_lora` is never offered on ns-3 — there is no second radio in that
world, and offering it would let the agent "recover" via something the world cannot do.

**The event gate** (`harness/loop.py`) decides whether this tick is worth a model call. A
60 s episode at 1 Hz would otherwise ask the same question sixty times. It suppresses when:

- **quiet** — delivery healthy, the percept anomaly gate has not fired, *and* the TX-side
  statistics are clean. All three, because for reactive and hidden-terminal the first two
  are healthy while the link is dying.
- **stable** — the panel moved less than 0.15 (L∞) and the legal-move set is unchanged.

With a **heartbeat**: after 10 consecutive suppressed ticks it asks anyway, and a hard cap
of 30 calls per episode. Without the heartbeat, every reactive episode produced *zero*
teacher decisions.

**The panel** is what the LLM actually sees — 23 rows, with units and plain-English meaning,
no raw float dump and **no ground truth**:

```
feature                   value  lvl    meaning
pdr_fast                     0% ▁low   delivery ratio now
noise_delta_base          +1.00 █HIGH  noise floor rise vs nominal (jamming -> high)
energy_no_preamble          95% █HIGH  busy energy with NO decodable packet (jamming -> high)
scan_noise_spread         +1.00 █HIGH  spread across channels (spot -> high, barrage -> low)
scan_bad_frac               50% ▅mid   fraction of channels hot in last scan
tx_shadow_loss_delta      +1.00 █HIGH  extra loss right after OUR transmissions (reactive -> high)
...
```

**The reply** is one JSON object:

```json
{"belief": {"barrage":0.15,"spot":0.75,"reactive":0.05,"sweep":0.05,"fading":0.0,...},
 "call": "hop_channel", "args": {"channel": 3}, "confidence": 0.85,
 "why": "noise floor and busy-energy confirm an emitter, and the scan shows one very hot
         channel among clean ones -- spot, not barrage, so hopping is worth the budget"}
```

Then: brace-balanced JSON extraction → **exactly one** repair attempt → otherwise abstain
(recorded, not swallowed). The call is validated against the mask; an illegal call becomes
`no_op` and is recorded as `masked_call`.

---

## 7. The loop closing: what ns-3 does with the answer

This is the part that makes it a control loop rather than a labelling exercise.
`jamming-sim.cc` reads the reply and **mutates the running simulation**:

| call | what actually changes in ns-3 |
|---|---|
| `hop_channel` | every node's `WifiPhy` `ChannelSettings` is retuned — a mesh-wide coordinated hop. The resynchronisation gap shows up as genuinely lost beacons. |
| `set_tx_power` | the agent's `TxPowerStart/End` change, and `curTxp` is updated so the agent can observe its own effect |
| `change_tdma_slot` | `MeshNodeApp::SetSlot()` — the transmit gate moves |
| `silent_listen` | `SetTxEnabled(false)`, re-enabled after `duration_ms`. A reactive jammer then has nothing to react to — which is exactly how that test works |
| `move` | the agent's position changes and the spectrum analyzer moves with it |
| `spectrum_scan` | the band data is released to the percept layer on the next tick, and budget is charged |
| `declare_link_lost` | the episode ends |

So after the LLM answers, the **next** state message already reflects the consequence.
If it hops onto a clean channel, PDR recovers and `band[]` for the new channel reads quiet.
If it hops onto another jammed channel, it does not. The agent then sees that and decides
again. Nothing is replayed or faked.

---

## 8. Worked example — `spot_10071`, end to end

| t | what ns-3 sends | features | gate | teacher says | ns-3 does |
|---|---|---|---|---|---|
| 1–20 | pdr 1.0, band flat −100.8 | healthy | **quiet**, suppressed | — | nothing |
| 21.1 | *jammer J1 switches on, channel 8, duty 0.7* | | | | |
| 22 | pdr 0.33, band[8] −64.9 | `noise_delta_base` +1.00, `scan_age` "never" | ask | `spectrum_scan` — "emitter confirmed but flavour unresolved" | charges budget, releases band on next tick |
| 23 | scan result now visible | `scan_bad_frac` 0.5, `scan_noise_spread` +1.00 | ask | belief spot 0.75, `hop_channel` ch 3 | **retunes all 6 nodes to ch 3** |
| 24–26 | pdr climbing, band[3] quiet | recovery | **stable**, suppressed | holds | nothing |
| 27 | pdr 0.95 sustained 3 s | `recovery_t` set | | | |

Verdict: `declared_cause = spot`, correct, cost 0.0, survived, one hop spent.

Note what the verifier scores: not the argmax at every tick, but the **claim the agent
committed to when it spent budget** — here, `spot` at t=23 when it hopped.

---

## 9. Policy for the student — what it is and how it is made

The student is **two things plus a controller**, and none of them calls a model.

### 9.1 The rule layer — `data/policy_ns3.json`

Induced by the LLM, then **measured** and kept only if it survives:

```bash
python3 harness/induce.py --from-rows ../data/traces_ns3/train.jsonl \
    --holdout-rows ../data/traces_ns3/val.jsonl --rows 80 --out ../data/policy_ns3.json
```

The LLM proposes candidate rules in a tiny DSL; every candidate is measured on a **held-out**
ns-3 split; only those clearing the precision bar survive, and rejects are recorded in the
audit file. Current output — 8 proposed, 4 accepted:

```
barrage   IF energy_no_preamble>0.3 AND scan_bad_frac>0.0 AND scan_noise_spread<0.2   prec 1.00
spot      IF energy_no_preamble>0.3 AND scan_bad_frac>0.0 AND scan_noise_spread>=0.2  prec 1.00
node_loss IF heartbeat_gap>0.5 AND link_asymmetry>0.5 AND scan_bad_frac<0.0           prec 1.00
fading    IF energy_no_preamble<0.0 AND scan_bad_frac<-0.5 AND tx_shadow_loss_delta>0.5 ... prec 0.82
```

That is the whole rule layer — about 30 lines of C once ported. It is deliberately small
enough that a rule cannot hide a second model inside it.

### 9.2 The neural layer — `data/student_v10/`

Two heads on the 56 features: a **cause head** (8 classes) and a **call head** (12 actions).
12,756 parameters, 49.8 KB fp32, **16.3 KB int8**.

Per DESIGN §9.5 the supervision is split:
- the **cause** label comes from the simulator — free, unlimited, and correct;
- the **call** (which test to buy, when to stop investigating) comes from the **LLM
  teacher's own decisions**. That is the part no free label can supply.

### 9.3 Decision rule — cost, not argmax

`contract/cost_matrix.yaml` is asymmetric: calling jamming when it is really fading costs
**10**, because the response (hopping) destroys a mesh that was still working. So the student
picks the **minimum-expected-cost** class, not the most probable one, and abstains below a
conformal threshold calibrated on the **deployment** world.

### 9.4 Making the on-device policy

```bash
python3 train/llm_traces_to_rows.py --traces ../data/traces/llm_full --split train \
    --out ../data/traces_llm_full/train.jsonl
python3 train/train_mixed.py --out ../data/student_v10 \
    --llm ../data/traces_llm_full/train.jsonl --llm-weight 3
python3 train/calibrate_abstain.py --bundle ../data/student_v10/student_bundle.json \
    --calib ../data/traces_ns3/val.jsonl --ood ../data/traces_ns3/test.jsonl --write
python3 train/export_int8.py --bundle ../data/student_v10/student_bundle.json \
    --traces ../data/traces_ns3/val.jsonl --out ../data/student_v10
```

The last step writes **`student_weights.h`** (55.5 KB) — the C header you compile into the
firmware. int8 ships only if it agrees with fp32 on **99.5%+ of the decisions the agent
actually acts on**; a head that fails keeps its output layer in fp32.

---

## 10. Is the SDR code in this architecture? — **No, and that is deliberate**

Let me be direct, because it matters for what you claim.

**There is no software-defined-radio code in this project.** No GNU Radio, no RTL-SDR, no
UHD/SoapySDR, no IQ sampling, no FFT-based sensing, no modulation or demodulation written by
us. I grepped for all of it; there is none.

What exists instead:

| layer | simulation | hardware |
|---|---|---|
| PHY / spectrum | ns-3 `SpectrumWifiPhy` + `WaveformGenerator` — ns-3's own PHY | ESP32 Wi-Fi radio (ESP-NOW), i.e. vendor silicon |
| energy sensing | ns-3 `SpectrumAnalyzer` | nRF24L01+ **RPD** — a 1-bit "is there energy above ≈ −64 dBm" comparator, swept across channels |
| MAC / mesh | `mesh-node-app.cc` + OLSR | ESP-NOW vendor frames + our beacon/report protocol |

`docs/DESIGN.md` §2.2 ("Why we do not write our own radio stack") states this as a design
commitment. The reasoning holds up: **the agent is drawn at the TELEMETRY level, not the PHY
level.** It never touches a sample. It consumes RSSI, a noise estimate, per-peer delivery and
timing — things every radio already reports. That is precisely why the sim-to-real port is
six functions in `firmware/hal.h` and not a rewrite.

So the honest sentence for your report is: *"an on-device diagnostic agent for a jammed mesh
radio, validated against ns-3 and portable to commodity hardware through a six-function
telemetry port"* — **not** *"an SDR implementation"*. If the brief requires a real SDR
front end, that is a genuine gap and you should say so rather than let the ns-3 PHY stand in
for it.

---

## 11. On-device: files, wiring, compilation

### 11.1 The files

| file | what it is | lines |
|---|---|---|
| `firmware/hal.h` | **the port.** 6 functions + 2 structs. The entire sim-to-real seam | 106 |
| `firmware/hal_host.c` | deterministic desktop stub — lets the whole stack be tested with no hardware | 174 |
| `firmware/hal_esp32.c` | the real one: ESP-NOW mesh + nRF24L01+ RPD sweep | 274 |
| `firmware/hal_selftest.c` | 15 checks over the port's semantics | 150 |
| `data/student_v10/student_weights.h` | the generated int8 model | 55.5 KB |
| `data/policy_ns3.json` | the induced rules — port to a C table + a `for` loop | 4 rules |

The six functions:

```c
int  hal_init(uint8_t my_index, uint8_t channel);
void hal_sense(hal_obs_t *out);                      /* free, continuous            */
bool hal_have_noise(void);                           /* is the noise floor real?    */
int  hal_scan(hal_scan_t *out, uint32_t dwell_ms);   /* COSTS airtime               */
int  hal_neighbor_probe(uint8_t peer, uint32_t timeout_ms);
int  hal_set_channel(uint8_t ch);  int hal_set_tx_power(int8_t dbm);
int  hal_set_slot(uint8_t slot, uint8_t n_slots);  int hal_set_tx(bool enabled);
int  hal_declare_lost(void);
```

### 11.2 Compile and test with no hardware at all

```bash
make hal
# or directly:
cd firmware && cc -std=c11 -Wall -Wextra -O2 -o haltest hal_host.c hal_selftest.c && ./haltest
```

15/15 should pass. This proves the port's *semantics* — a jammer raises the floor, a fade
does not, S4′ separates reactive from barrage, every actuator changes something.

### 11.3 Compile for the ESP32

```bash
cd firmware && cc -std=c11 -Wall -Wextra -fsyntax-only hal_esp32.c   # parse check

# real build: an ESP-IDF v5.x project
idf.py create-project jamsurvive && cd jamsurvive
cp ../firmware/hal.h ../firmware/hal_esp32.c main/
cp ../data/student_v10/student_weights.h main/
# add an RPD-capable nRF24 driver to main/, then
idf.py set-target esp32 && idf.py build && idf.py -p /dev/ttyUSB0 flash monitor
```

Wiring — the only wiring in the project (nRF24L01+ on VSPI, **3.3 V only**, 10 µF across
VCC/GND):

```
CE -> GPIO4    CSN -> GPIO5    SCK -> GPIO18    MOSI -> GPIO23    MISO -> GPIO19
```

### 11.4 What still has to be written for the device

Being straight with you: `firmware/` is the **port**, not the whole agent. Still to port
(`docs/HARDWARE_GUIDE.md` stage H5, in this order):

1. `percept/norm.py` → `norm.h` — constants, must be byte-identical to the sim
2. `percept/ring.py` → ring buffers, EWMA, CUSUM, run-length, Pearson
3. `percept/features.py` → the 56-feature vector (the big one, ~494 lines)
4. `agent/controller.py` → mask + budget + dead-man failsafe
5. `data/policy_ns3.json` → a C table + a `for` loop ← **start here**
6. `student_weights.h` → a 30-line matmul

**If you only have one day, do 1–5 and skip the net.** The rules alone carry the
transferable physics and never make the expensive mistake.

---

## 12. The 29%, and whether the system learns anything

### 12.1 What 29% means

The LLM teacher declared the correct cause in **28.8% of episodes** (517 scoreable). It is a
real number and it means the model frequently reaches the wrong diagnosis from the panel it
is given. Per family:

```
barrage .59   refusal .59   node_loss .54   fading .42   spot .24
reactive .10  sweep .07     hidden_term .02  congestion .00
```

The top confusions, from 9,056 decisions:

```
fading      -> node_loss  1035      hidden_term -> fading   381
sweep       -> reactive    359      node_loss   -> fading   342
reactive    -> fading      331      congestion  -> fading   298
```

The pattern is unmistakable: **everything collapses toward `fading` and `node_loss`** — the
two benign explanations. The teacher is systematically *under-calling* attacks. Given the
prompt tells it that calling jamming on fading is the worst possible error, that is the
prompt working slightly too well.

### 12.2 Your actual question: are we learning what evidence is missing?

**No. Today the system does not do that, and I should not pretend otherwise.**

There is no component that reads the teacher's failures, asks "what would I have needed to
get this right", and feeds that back. Concretely, what is missing:

- nothing compares a wrong decision against the features that *would* have separated the
  classes;
- nothing proposes a new feature, or a new diagnostic call, from observed failures;
- nothing re-runs a failed scenario with extra evidence to test a hypothesis;
- there is **no online learning at all** — the student is frozen at export.

What *does* exist, and is adjacent:

- **`tests/test_contract_effects.py`** measures whether each contract call changes anything
  in any family. It already caused two calls (`listen_test`, `transmit_probe`) to be deleted
  from the contract, and it currently flags `neighbor_probe` as **known-open: implemented,
  but its answer never reaches the agent because the percept layer has no probe-result
  feature**. That is precisely a "which parameter is missing" finding — found by a gate, not
  by the agent, and fixed by a human.
- **DAgger** (`train/dagger.py`, `train/bridge_dagger.py`) retrains on states the agent
  actually visits — on-policy correction, but expert-driven, not failure-diagnosis-driven.
- Every teacher decision stores its `why`. Section 12.1 above was produced by mining those
  strings. **The data to build the loop you are describing already exists and is unused.**

### 12.3 What it would take — and one result that suggests it is worth it

The loop you are imagining is roughly:

1. take every episode the teacher got wrong;
2. for each, compute which features *do* separate the true cause from what it said
   (mutual information between that feature and the pair of classes, on the corpus);
3. ask the LLM: "you said X, it was Y; here are the features that discriminate — was the
   evidence absent from your panel, present but unhelpful, or present and you misread it?";
4. bucket the answers: **missing evidence** → add a percept feature or a diagnostic call;
   **unhelpful** → the statistic is wrong and needs redesign; **misread** → a prompt or
   rule-layer fix.

Step 2 is cheap and needs no LLM. Step 3 is ~500 calls. This is a few hours of work, not a
research programme, and the ingredients — traces with reasoning, a labelled corpus, a
contract-effect gate — are all already here.

And there is already evidence it pays. The one case where a missing-evidence problem *was*
found and fixed — the reactive family, where the gate consulted only `pdr_fast` and never the
TX-side statistics, so the teacher was never even asked — moved that family from **0 teacher
decisions per episode** to 11–14, and moved the student from **0/3 to 2/3** on the live
bridge. That was one missing input, found by hand. A loop that finds them systematically is
the obvious next thing to build.

### 12.4 One more reason not to over-read the 29%

While writing this document I found a labelling bug that had been depressing it: `refusal`
scenarios were being labelled `fading` in the teacher corpus instead of `barrage`, because
the family name was used as the cause and `refusal` is not a cause. 912 of 5,545 rows — 16%
of the corpus — carried the wrong label, and it was the *most expensive* wrong label
available. Fixing it moved teacher agreement 19.9% → 22.0% and took the student from
**54% → 77%** on the live ns-3 bridge, restoring `refusal` from 1/3 back to 3/3.

So: 29% is a real number for a teacher that genuinely struggles, but it was measured through
a corpus with a known defect in it, and the number after the fix has not yet been recomputed
end to end. Treat it as a floor, not a verdict.
