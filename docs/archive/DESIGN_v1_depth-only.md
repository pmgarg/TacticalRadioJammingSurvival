# Jamming Survival — On-Device Mesh Agent
### Complete design and implementation plan

**Team:** Prateek Mohan Garg · Jatin Pahuja
**Capstone:** EAG V3 — Project 9
**Status:** design frozen for build · v1.0

---

## 0. What we are building, in one paragraph

A drone's mesh radio link degrades. Six different physical causes produce nearly the same
symptom. An agent running *on the drone*, with no ground link to ask for help, must work out
which cause it is facing, run the cheapest physical experiment that tells the causes apart,
take a recovery action, verify that the action worked, and — when there is genuinely no
escape — refuse to keep trying and trigger return-to-home instead. We build this in three
pieces: an **ns-3 simulator** that owns the ground truth and can manufacture every failure
mode on demand; a **large teacher model** that solves episodes in simulation and whose
decisions we record; and a **tiny function-calling student** distilled from those traces that
fits in a few hundred kilobytes and runs on a bare ESP32 at 240 MHz. We then measure, honestly,
how much worse the student is than the teacher.

---

## 1. Why this problem is hard (and why it needs an agent, not a classifier)

### 1.1 The observation is a projection of a hidden state

The drone never observes "a jammer". It observes a vector of radio statistics. Several
distinct hidden causes map onto almost the same observation vector:

```
hidden cause  c ∈ C          ──(physics)──▶   observation  o ∈ R^F
```

The map is many-to-one. That is the whole problem. `pdr = 0.1, retries = 0.9` is consistent
with a barrage jammer, with a deep multipath fade, and with a congested channel. No amount
of staring at the passive statistics resolves it, because the information is not there.

### 1.2 Therefore: information-gathering actions are first-class

What *does* resolve it is intervention. If you stop transmitting for 200 ms and the noise
floor drops to baseline, you have proved the interference is reactive to your own
transmissions. If you fly 30 m sideways and the link returns, you have proved it was
geometry, not an attacker. Passive statistics are correlational; these are experiments.

This makes the task a **POMDP with explicit sensing actions**, not a classification problem:

- **Hidden state** `c` — the true cause, plus its parameters (which channel, what duty cycle).
- **Belief** `b(c)` — a 7-way categorical the agent maintains and updates.
- **Actions** split into two families:
  - *epistemic* (change belief, don't fix anything): `spectrum_scan`, `silence_probe`,
    `neighbour_probe`, `move_to` used as a test
  - *pragmatic* (change the world): `hop_channel`, `set_tx_power`, `reroute`,
    `change_tdma_slot`, `fall_back_to_lora`, `declare_link_lost`
- **Costs** — every action costs something real: off-channel time, lost packets, mesh
  resynchronisation, battery, seconds. A spectrum scan of 13 channels at 20 ms dwell is
  260 ms during which you are deaf. Channel hops are not free.
- **Asymmetric loss** — declaring jamming when it was fading is the expensive error, because
  the response (hop channel) destroys the mesh you still had.

A classifier answers "what is it". An agent answers "what is the cheapest thing I can do
right now that most reduces my uncertainty, and when do I stop investigating and act."
That gap is the project.

### 1.3 The failure mode we are designing against

An agent that hops channels forever. It looks busy, it looks like it is trying, and it
flies until the battery is flat. **Not acting is a valid action, and refusing is a required
one.** Section 4.6 makes the refusal structurally guaranteed rather than something we hope
the model learned.

---

## 2. The causal taxonomy

Seven causes. For each: what it looks like passively, what it is confused with, and the
experiment that separates it.

### 2.1 Signature table

| # | Cause | PDR | RSSI of peer | Noise floor | Retries | Spectrum character | Time structure |
|---|---|---|---|---|---|---|---|
| C1 | **Barrage jamming** | collapses, all channels | normal | high on every channel | max | wideband energy, **no decodable preambles** | step onset, persistent |
| C2 | **Spot jamming** | collapses on ch *k* only | normal | high on ch *k* only | max on *k* | narrowband peak | step onset, persistent |
| C3 | **Reactive jamming** | ok-ish but nearly everything is a retransmission | normal | **high only in the window after our own TX** | very high | energy appears µs after our preamble | correlated with *our* TX schedule |
| C4 | **Sweeping jammer** | periodic collapse, rotating across channels | normal | rises and falls per channel with a period | narrowband peak that moves | periodic, period 0.5–5 s |
| C5 | **Fading / multipath** (no attacker) | intermittent, deep nulls | **drops with PDR** | normal | high | clean | Rayleigh-like; fade duration set by Doppler |
| C6 | **Node loss / dead peer** | 0 on that link only, others fine | **no frames at all** from that peer | normal | tx never ACKed | clean | step, one link only |
| C7 | **Congestion / hidden terminal** | degrades with offered load | normal | moderately raised, **802.11-structured** | high (collisions) | **decodable foreign frames** | tracks traffic, bursty |

An eighth, worth carrying because it is cheap to detect and embarrassing to miss:

| C8 | **Own receiver failure** | all links dead inbound, our TX still gets ACKed | nothing received | — | — | — | step, affects everything |

### 2.2 The confusion pairs that actually matter

- **C1 vs C5 (barrage vs fading)** — the false-positive trap named in the brief. Both give
  PDR collapse on every channel. Separated by *noise floor* and by *RSSI–PDR correlation*.
- **C1 vs C7 (jamming vs congestion)** — both raise channel-busy time. Separated by whether
  the busy energy carries a *decodable 802.11 preamble*.
- **C3 vs C7 (reactive vs collisions)** — both look like "everything collides". Separated by
  the silence probe.
- **C6 vs C1 (dead peer vs jammed link)** — both give zero PDR to that peer. Separated by a
  clean channel plus a third-party probe.
- **C2 vs C4 (spot vs sweep)** — same instantaneous picture. Separated only by *time*: the
  sweep comes back. Requires the agent to remember, which is why the model gets a temporal
  window and a persistent belief vector.

### 2.3 The discriminative statistics (this is the intellectual core)

These are the quantities that carry the information. Every one of them is computable on an
ESP32 in fixed point from data the radio already gives us.

**S1 — RSSI/PDR correlation.** Over a 5 s sliding window, `ρ = corr(RSSI_t, PDR_t)`.
Under fading the received power itself drops, so ρ is strongly positive. Under jamming the
peer is still transmitting at the same power and distance, so the frames we *do* receive
arrive at normal RSSI while PDR collapses — ρ ≈ 0. **This single feature is the primary
defence against the expensive false positive.**

**S2 — SINR decomposition.** `SINR ≈ RSSI − noise_floor`. A SINR collapse driven by the
*numerator* falling is path loss or fading. Driven by the *denominator* rising is external
interference. Same SINR, opposite diagnosis, opposite correct action.

**S3 — Energy-without-preamble ratio.** Fraction of CCA-busy time during which no valid
802.11 preamble was decoded. Congestion is other people's *packets*; jamming is *energy*.
Near 1.0 ⇒ jammer. Near 0 with high busy time ⇒ congestion.

**S4 — TX-conditioned noise delta.** `Δ = E[noise | we transmitted within last τ] −
E[noise | silent ≥ τ]`, τ ≈ 2 ms. Large positive Δ ⇒ reactive jammer. This is the feature
that makes C3 detectable *passively*, before we spend an action on the silence probe.

**S5 — Cross-channel PDR variance.** Variance of PDR (or of noise floor) across scanned
channels. High ⇒ spot/sweep. Low with all channels bad ⇒ barrage. Low with all channels
good ⇒ not a channel problem at all — look at the peer or at ourselves.

**S6 — Cross-link agreement.** Fraction of our peers that are degraded. One out of three ⇒
that peer or that geometry. Three out of three ⇒ the channel or our own receiver.

**S7 — Fade-duration statistics.** Under Rayleigh fading with Doppler `f_d`, the average
duration of a fade below threshold is analytically known and short (milliseconds to tens of
ms at drone speeds). Jammer on/off intervals are square and long. The histogram of
below-threshold run-lengths separates them without any action at all.

**S8 — Self-motion coupling.** Correlation of PDR with our own displacement/velocity.
Fading tracks position; an attacker at a fixed point does not (until you move far enough to
change the geometry, which is exactly what `move_to` is for).

**S9 — Heartbeat gap.** Time since last frame of *any* kind from a peer, including its
routing beacons. A peer that is jammed still tries; a dead peer emits nothing, ever.

**S10 — Load coupling.** Correlation of loss with our own offered load. Congestion scales
with load. A barrage jammer does not care what we send.

**Design note.** S1, S2, S3, S4, S7 and S10 are all *passive*. A well-built feature layer
resolves most episodes before the agent ever spends an action. The agent's job is to know
when the passive evidence is ambiguous and which single experiment closes the gap. That
framing is what keeps the action budget low, and action budget is one of the scored metrics.

### 2.4 The experiment catalogue

| Test | Cost | Separates |
|---|---|---|
| `spectrum_scan(all, 20ms)` | ~260 ms deaf, ~15 packets | C1 vs C2 vs C4 vs C5/C6 (clean channel) |
| `spectrum_scan([k], 50ms)` | ~50 ms deaf | confirm a single channel before hopping |
| `silence_probe(200ms)` | 200 ms of no TX | **C3** vs everything else |
| `neighbour_probe(peer, "you_tx_i_listen")` | 1 control frame + 100 ms | C6 (dead peer) and C8 (our RX) vs channel causes |
| `neighbour_probe(peer, "i_tx_you_report")` | 1 frame + report latency | our TX chain vs our RX chain |
| `move_to(+30 m lateral)` | 5–15 s, battery | **C5** vs C1/C2 |
| `set_tx_power(max)` briefly | ~0, battery | weak-signal margin vs interference-limited |
| `reduce own load 50%` | throughput | **C7** |

The agent chooses among these by expected information gain per unit cost. Section 4.5.

---

## 3. System architecture

Four layers. Only one of them is learned, and it is the smallest one.

```
┌──────────────────────────────────────────────────────────────────────┐
│ L3  TEACHER  (offline, ground station, simulation only)              │
│     Claude Opus / Sonnet with the same tool schema and same costs.   │
│     Produces (state → hypothesis, action, rationale) traces.         │
│     NEVER runs on the drone. NEVER sees ground truth.                │
└──────────────────────────────────────────────────────────────────────┘
                                 │  distillation
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ L2  AGENT  (on drone, 2 Hz, event-triggered)          ~150 KB int8   │
│     Tiny function-caller. In: 16×48 feature window + belief + action │
│     history.  Out: belief over 8 causes, one action, its arguments,  │
│     a confidence, and an "unrecoverable" flag.                       │
│     Wrapped in a hard safety envelope (§4.6).                        │
└──────────────────────────────────────────────────────────────────────┘
                                 ▲ features            │ actions
┌──────────────────────────────────────────────────────────────────────┐
│ L1  PERCEPT  (on drone, 10 Hz, fixed point, ~8 KB RAM)               │
│     Ring buffers → EWMAs, correlations, CUSUM, run-length histograms │
│     → the 48-dim normalised feature vector. Plus the anomaly gate.   │
└──────────────────────────────────────────────────────────────────────┘
                                 ▲
┌──────────────────────────────────────────────────────────────────────┐
│ L0  RADIO  (classical, unchanged, µs–ms)                             │
│     PHY · CSMA/TDMA MAC · ARQ · OLSR routing · channel management    │
│     The agent never sits in this loop. It only reconfigures it.      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why the split matters.** Real-time radio behaviour stays deterministic and verifiable.
The model is advisory over a slow control loop, so a slow or wrong inference degrades
recovery quality but cannot break the radio. This is also what makes 50 ms of inference
latency acceptable — we are deciding once or twice a second, not per packet.

### 3.1 The anomaly gate

The agent does not run continuously. A two-sided CUSUM on short-window PDR against a slow
baseline fires the agent:

```
S_t = max(0, S_{t-1} + (pdr_baseline − pdr_t − k)),   fire when S_t > h
```

with `k` = 0.02 (slack) and `h` tuned so that the false-alarm rate under nominal conditions
is < 1/hour and detection latency under a step drop is < 300 ms. Once fired, the agent runs
at 2 Hz until either recovery is verified for 3 consecutive seconds or the episode ends.

Detection latency in the verifier is measured from **attack onset** to **correct
classification**, so the gate's latency is part of the score. Do not set `h` conservatively
to make the false-alarm number look good — it is a direct trade against the headline metric.

### 3.2 The feature vector — 48 dimensions

Grouped, with the discriminative statistic each one serves. `L*` are computed per link and
then reduced (we carry `worst-link` and `mean-over-links`); with 3 peers this stays fixed
size regardless of mesh degree.

**Delivery (8)**
```
 0  pdr_ewma_fast      τ=0.5 s
 1  pdr_ewma_slow      τ=5 s
 2  pdr_delta          f0 − f1
 3  pdr_cusum          gate statistic, normalised
 4  pdr_worst_link
 5  pdr_link_spread    std across links                        → S6
 6  frac_links_degraded                                        → S6
 7  time_since_onset   s, log-scaled
```
**Signal (8)**
```
 8  rssi_mean_fast
 9  rssi_std_fast
10  rssi_slope         dBm/s over 5 s
11  rssi_min_fast      deep-fade indicator
12  rssi_pdr_corr      corr over 5 s        ← S1, the false-positive killer
13  rssi_frames_seen   count (0 ⇒ nothing heard at all)
14  sinr_est           rssi − noise                             → S2
15  sinr_delta_base    vs 60 s baseline                         → S2
```
**Interference (8)**
```
16  noise_now
17  noise_delta_base                                            → S2
18  noise_std
19  cca_busy_frac
20  energy_no_preamble_ratio                                    → S3
21  foreign_frame_rate decodable non-mesh frames/s              → S3
22  noise_tx_conditioned_delta                                  → S4
23  loss_given_own_tx_minus_loss_given_silent                   → S4
```
**MAC / ARQ (6)**
```
24  retry_rate_ewma
25  retries_per_success
26  consecutive_tx_fail_max
27  queue_occupancy
28  offered_load_norm
29  loss_load_corr                                              → S10
```
**Spectral memory (6)**
```
30  scan_age                     s since last scan, log-scaled
31  scan_channels_bad_frac
32  scan_noise_spread            variance across channels       → S5
33  scan_current_ch_rank
34  scan_best_alt_ch_margin      dB better than current
35  scan_periodicity_score       autocorr of per-channel noise  → C4 sweep
```
**Temporal structure (4)**
```
36  fade_runlen_mean             below-threshold run lengths    → S7
37  fade_runlen_p95                                             → S7
38  outage_duty_cycle
39  outage_period_est            0 if aperiodic                 → C4
```
**Self / platform (4)**
```
40  own_speed_norm
41  displacement_since_onset
42  pdr_motion_corr                                             → S8
43  peer_heartbeat_gap_max                                      → S9
```
**Budget / bookkeeping (4)**
```
44  hops_used_frac
45  scans_used_frac
46  actions_used_frac
47  time_in_episode_norm
    (+ appended, not counted in the 48: prev_belief[8], last_action_onehot[10])
```

All features are normalised to roughly `[-1, 1]` with **fixed, hand-set** scaling constants
committed to the repo (`percept/norm.h`), not learned from the training set. This matters:
the same constants must be used in ns-3 and on the ESP32, or sim-to-real transfer fails
silently. Normalisation constants are part of the interface contract.

The model consumes a window of `T = 16` timesteps at 10 Hz — 1.6 s of history — plus the
previous belief and last action. That is enough to see a sweep period only if the period is
short; longer-period structure is carried in features 35 and 39, which are computed over a
much longer horizon in L1. **Put the long memory in the feature layer, not in the model.**
The model stays small that way.

### 3.3 The agent contract (the tool schema)

This JSON schema is the single interface shared by the ns-3 bridge, the teacher, the student
and the ESP32 firmware. It lives at `contract/agent_contract.json` and every component reads
it. Changing it is a breaking change for four subsystems at once, so it is frozen in week 1.

**Sensors** — read-only, some cost time:

```jsonc
rssi()                    // {link_id: dBm}                       cost 0
noise_floor()             // dBm on current channel               cost 0
pdr_per_link()            // {link_id: 0..1}                      cost 0
retry_rate()              // 0..1                                 cost 0
spectrum_scan(channels[], dwell_ms)
                          // [{ch, noise_dbm, busy_frac, decodable_fps}]
                          // cost: len*dwell_ms deaf; default 13*20 = 260 ms
silence_probe(duration_ms) // suppress own TX, return noise trace  cost: duration_ms
neighbour_probe(peer, mode)// mode ∈ {peer_tx_i_listen, i_tx_peer_reports,
                           //         relay_via_third}            cost ~100 ms
link_state()              // per-peer heartbeat age, route table   cost 0
self_state()              // pos, vel, battery, budgets            cost 0
```

**Actions** — all arguments discrete, so every argument is a classification head:

```jsonc
no_op()                                     // explicitly available and often correct
hop_channel(n)          n ∈ 1..13           // cost 200–500 ms resync, risk of mesh split
set_tx_power(p)         p ∈ 8 levels        // -4..20 dBm
reroute(via)            via ∈ peer ids ∪ {direct}
change_tdma_slot(s)     s ∈ 0..7
move_to(dx,dy,dz)       ∈ 7 presets         // ±30 m lateral, ±20 m alt, hold
fall_back_to_lora()                         // 1/1000 the rate, but robust
declare_link_lost()                         // ⇒ failsafe RTH on last known position
```

Each call returns, and each trace row records:

```jsonc
{
  "t": 12.4,
  "belief": {"barrage":0.61,"spot":0.05,"reactive":0.02,"sweep":0.03,
             "fading":0.21,"node_loss":0.03,"congestion":0.04,"rx_fault":0.01},
  "action": "spectrum_scan",
  "args": {"channels":[1,6,11],"dwell_ms":20},
  "why": "pdr collapsed on all links but rssi_pdr_corr=0.05 and noise is +18 dB; "
         "need to know if the rise is one channel or all of them",
  "confidence": 0.62,
  "unrecoverable": 0.03
}
```

`why` is free text **only in the teacher's traces**. The student never generates it — it
emits the belief vector, which is the machine-readable form of the same claim, and the
belief is what the verifier scores. This is how we honour "which function, which arguments,
and why" without putting a language model on the drone.

---

## 4. The decision loop

```
        ┌──────────────┐
        │ nominal 10Hz │◀──────────────────────────────┐
        └──────┬───────┘                               │
               │ CUSUM fires                           │ recovered & stable 3 s
               ▼                                       │
   ┌───────────────────────┐                           │
   │ 1. form belief b(c)   │                           │
   └───────────┬───────────┘                           │
               ▼                                       │
   ┌───────────────────────────────────────┐           │
   │ 2. is b confident enough to act?       │──yes──┐  │
   │    max_c b(c) > θ_act (≈0.75)          │       │  │
   └───────────┬───────────────────────────┘        │  │
               │ no                                  │  │
               ▼                                     ▼  │
   ┌───────────────────────────┐      ┌─────────────────────────┐
   │ 3. cheapest distinguishing│      │ 4. pragmatic action for  │
   │    test: argmax ΔH/cost   │      │    argmax_c b(c)         │
   └───────────┬───────────────┘      └───────────┬─────────────┘
               │ observe                          │
               └────────────▶ update b ◀──────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │ 5. verify: did PDR recover?  │
                    │    budget left? escalate?    │
                    └──────────┬───────────────────┘
                               │ budget exhausted AND b(unrecoverable) > θ_lost
                               ▼
                    ┌──────────────────────────────┐
                    │ 6. declare_link_lost() → RTH │
                    └──────────────────────────────┘
```

### 4.1 Belief update

The student emits a full posterior each tick. It is *not* a Bayes filter with a hand-written
likelihood — that would require a model of the channel we do not have. It is a learned
recurrent posterior: previous belief is an input feature, the network outputs the new one.
The teacher supplies belief targets implicitly (its stated hypothesis), and ns-3 supplies
the true label, so the head is trained on both (§7.3).

### 4.2 Choosing the test — value of information

For each available test `a`, the expected entropy reduction per unit cost:

```
score(a) = ( H(b) − E_o[ H(b | o, a) ] ) / cost(a)
```

We do **not** compute this at run time on the drone — that needs a forward model. Instead:

- In simulation we *can* compute it, because ns-3 gives us the true `c`. We compute the
  oracle test-selection policy offline and use it as (a) an upper bound in the verifier and
  (b) an auxiliary training signal.
- The student learns to *approximate* `argmax_a score(a)` directly from features, because
  that mapping is nearly deterministic: given "all links down, noise +18 dB, no scan in the
  last 8 s", the best test is essentially always the same one.

This is the cleanest instance of the teacher–student idea in the project: an expensive
computation (or an expensive model) done offline, compiled into a cheap feed-forward mapping.

### 4.3 The recovery playbook (what a confident belief maps to)

| Belief | First action | If that fails |
|---|---|---|
| Spot jamming, ch *k* | `hop_channel(best_alt)` from the scan | second-best channel, then treat as barrage |
| Sweeping | `hop_channel` to the channel the sweep just left, then re-hop with the period | `change_tdma_slot` to dodge, else LoRa |
| Barrage | `set_tx_power(max)` + `reroute(via nearest peer)` to shorten hops | `fall_back_to_lora()` | 
| Reactive | `change_tdma_slot` + duty-cycle down + `set_tx_power(min viable)` | LoRa (different band, different trigger) |
| Fading | `move_to` / hold position, `set_tx_power(max)`, **do not hop** | `reroute` via a peer with better geometry |
| Node loss | `reroute` around the dead peer, update topology | if it was the only path: `declare_link_lost` |
| Congestion | back off offered load, `change_tdma_slot`, then `hop_channel` | `hop_channel` to a quiet channel from scan |
| RX fault | `reroute` to TX-only relaying, degrade gracefully | `declare_link_lost` |

Note the row that earns the project: **fading ⇒ do not hop.** The playbook is not
hard-coded into the drone; it is what the trained policy should reproduce, and the table is
the reference we score it against.

### 4.4 Verification of recovery

After a pragmatic action, the agent enters a verification window (2 s). Recovery is
declared when short-window PDR exceeds 0.8 × pre-onset baseline for 3 consecutive 1 s bins.
If not recovered, the belief is updated with the negative evidence — "hopping to ch 6 did
not help" is strong evidence against spot jamming — and the loop re-enters at step 1 with
one unit of budget consumed.

### 4.5 Budgets

Per episode, hard limits enforced outside the model:

```
channel hops                ≤ 3
full spectrum scans         ≤ 4   (and ≥ 2 s apart)
silence probes              ≤ 3
move_to commands            ≤ 2
total actions               ≤ 12
wall-clock before decision  ≤ 30 s
```

These numbers come straight from the "actions consumed" metric and from what a real drone
can afford. They are configurable in `contract/budgets.yaml` and identical in sim and on
hardware.

### 4.6 The safety envelope — how the refusal case is *guaranteed*

The brief's refusal case (broadband jamming on every channel, no line of sight to any peer,
correct answer is declare-lost and RTH) must not depend on the model having learned it. We
enforce it with three mechanisms, in increasing order of authority:

**1. Logit masking at inference.** Before the argmax, the action head's logits for
unavailable actions are set to `-inf`. If `hops_used == 3`, `hop_channel` cannot be selected
*at all*. This is four lines of C on the ESP32 and it makes "hops forever" impossible by
construction, not by training.

**2. A monotone escalation ladder.** Actions are ordered by cost and reversibility. The
agent may move down the ladder but never back up within an episode once a rung is exhausted:
```
no_op → tx_power/tdma → reroute → hop_channel → move_to → lora → declare_link_lost
```

**3. A dead-man rule outside the model entirely.** In `safety/failsafe.c`:
```
if (no_usable_link_for > T_lost)            // T_lost = 20 s
   or (all_channels_scanned_bad && no_peer_reachable && budget_exhausted)
then declare_link_lost(); execute_rth();
```
This runs whether or not the model produces output at all — including if inference hangs or
the model crashes. The model can trigger RTH *early* (that is the skill we want: recognising
hopelessness in 6 s instead of 20). It can never prevent it.

The verifier tests this explicitly: the refusal scenario family is a pass/fail gate, not an
averaged metric. A run that hops more than 3 times or fails to declare within 25 s of onset
is a hard failure regardless of its score elsewhere.

---

## 5. Tier 1 — the ns-3 simulator

The simulator is the only place ground truth exists, so it is the only place the verifier
can be honest. Build it first and build it completely.

### 5.1 Version and modules

Pin **ns-3.44** (or whatever the current release is at kickoff — pin it in `sim/NS3_VERSION`
and never float). Required modules:

| Module | Used for |
|---|---|
| `spectrum` | `MultiModelSpectrumChannel`, `WaveformGenerator`, `SpectrumAnalyzer` — the jammers live here |
| `wifi` | `SpectrumWifiPhy` (**not** `YansWifiPhy`), 802.11n/ax at 2.4 GHz, ad-hoc mode |
| `propagation` | `LogDistance` + `Nakagami` + `Jakes` for the fading scenario |
| `mobility` | waypoint and constant-velocity models for the drone and peers |
| `olsr` | the mesh routing protocol (batman-adv is a Linux kernel module and has no ns-3 port; OLSR is the right ns-3 analogue and is what the brief lists as the alternative) |
| `applications` | `OnOffApplication` / `UdpEchoClient` for the traffic that PDR is measured on |
| `energy` | optional, for the battery/RTH story |

**Why `SpectrumWifiPhy` is non-negotiable:** `YansWifiPhy` models interference only from
other Wi-Fi PHYs. A jammer that is not an 802.11 device is invisible to it. `SpectrumWifiPhy`
shares a `MultiModelSpectrumChannel` with arbitrary spectrum emitters, so a `WaveformGenerator`
transmitting a power spectral density really does raise the noise floor and really does break
reception. The canonical example to start from is
`examples/wireless/adhoc-aloha-ideal-phy-with-microwave-oven.cc`, which does exactly this
with a microwave oven as the interferer. Our jammers are that example, generalised.

### 5.2 Scenario topology

Default: one **agent drone** plus **4 peers** in a mesh, 60–250 m spacing, plus **1 jammer
node** (present or absent depending on family). Traffic: 200 kbps CBR UDP from the agent to
a designated sink two hops away, plus 20 kbps background between peers, so PDR is measured
on something real and routing has a choice to make.

Episode length 60 s. Attack onset uniformly sampled in `[15 s, 30 s]` so the agent has a
clean baseline first and the verifier has an unambiguous `t_onset`.

### 5.3 The jammer library

All in `sim/model/jammers/`, all subclasses of a common `JammerBase : public Application`
that owns a `WaveformGenerator` on the shared spectrum channel.

**Barrage (C1).** One `WaveformGenerator` per channel, or a single generator with a PSD
spanning the whole 2.4 GHz ISM band. Constant power, 100% duty. Parameters: EIRP
(−10…+20 dBm), bandwidth, start time. This is the easy one.

**Spot (C2).** Single generator, PSD confined to one 20 MHz channel. Parameters: channel,
power, duty cycle (a 50% duty spot jammer is much harder to detect and is worth including).

**Sweep (C4).** Spot jammer whose centre frequency steps on a timer. Parameters: dwell per
channel (50–500 ms), sweep order (linear/random), power. Implemented by rescheduling
`SetTxPowerSpectralDensity` with a shifted PSD.

**Reactive (C3).** The interesting one. Needs to detect our transmission and fire within
microseconds. Implementation: attach a `SpectrumAnalyzer` (or hook the
`MonitorSnifferRx`/`PhyRxBegin` trace source on a co-located listener PHY) to the shared
channel; on a rising-edge energy detection above threshold, `Simulator::Schedule` the
waveform generator ON for `t_react` (default 1.5 ms — long enough to corrupt the rest of a
frame) after a reaction delay `t_delay` (default 10 µs). Parameters: threshold, delay,
burst length, and — important for realism — a *probability of firing* below 1.0, so the
agent cannot detect it from a single deterministic correlation.

**Fading, no attacker (C5) — the false-positive trap.** No jammer node at all. Instead:
`NakagamiPropagationLossModel` with `m0 = 1.0` (Rayleigh) on the agent–peer links,
combined with a `JakesPropagationLossModel` for time-correlated fading at a Doppler
frequency matching the drone's speed, plus a mobility pattern that carries the drone through
a deep shadowing region. Tune the parameters until the *marginal distribution of PDR* over
the episode is statistically indistinguishable from the barrage episodes. **If your fading
episodes are visibly easier than your jamming episodes, your false-positive number is a
lie.** We check this in week 3 with a two-sample test on the passive feature marginals.

**Node loss (C6).** `Simulator::Schedule` a `node->GetDevice(0)->SetDown()` — or better,
turn the PHY off entirely — on the peer at onset. No RF trace of any kind afterwards.

**Congestion (C7).** Add 3–6 extra 802.11 nodes on the same channel running heavy
`OnOffApplication` traffic, plus a hidden-terminal geometry: node A and node C both in range
of B but not of each other. Real collisions, real retries, and — crucially — the busy energy
is decodable 802.11, which is what feature 20 keys off.

**RX fault (C8).** Set the agent's PHY RX gain to a large negative value, or drop all
receptions via a callback, while leaving TX intact.

**Composite scenarios.** After the singles work, generate mixtures: fading *and* a spot
jammer; congestion *and* a dead peer. These are the held-out generalisation set.

### 5.4 Ground truth logging

Every episode writes three files into `runs/<episode_id>/`:

```
truth.jsonl     one row per 100 ms:
                {t, cause, cause_params:{...}, jammer_on, jammer_ch,
                 true_sinr_per_link, true_pdr_per_link, peer_alive[]}
percept.jsonl   the exact 48-dim feature vector the agent saw, per tick
trace.jsonl     every agent decision (the schema in §3.3), with the
                pre-action and post-action state
meta.json       seed, family, parameters, ns-3 version, contract hash
```

The agent process **never reads `truth.jsonl`.** Enforce this at the process boundary: the
bridge server does not have the file open. The verifier reads all four.

### 5.5 The ns-3 ↔ agent bridge

The agent (teacher or student) lives in Python; ns-3 is C++. Three options were considered:

| Option | Verdict |
|---|---|
| `ns3-ai` shared memory | Fastest, but the shared-memory struct layout is a second contract to keep in sync, and it fights with the LLM teacher's variable-latency calls |
| `ns3-gym` | Good for RL, opinionated about the Gym interface, more machinery than we need |
| **Custom line-delimited JSON over a UNIX socket** | **Chosen.** Simple, language-agnostic, human-readable traces for free, trivially replayable |

**Design.** A C++ `AgentBridge : public Application` on the drone node. At each decision
epoch (every 100 ms of *simulation* time, or on demand after the anomaly gate fires) it:

1. Assembles the feature vector from L1 (implemented in C++ in the sim, mirroring the ESP32
   code exactly — see §5.6).
2. Writes one JSON line to the socket.
3. **Blocks the simulator** (`Simulator` does not advance) until a reply line arrives.
4. Parses the action and applies it through the appropriate ns-3 API
   (`WifiPhy::SetOperatingChannel`, `WifiRemoteStationManager` power, `Ipv4` route override,
   mobility waypoint, etc.).
5. Charges the action's cost by scheduling the effect at `now + cost_ms` and, for scans and
   silence probes, actually suppressing TX/RX for that duration so the cost is *physically
   real* and not just a scoreboard entry.

Because simulation time is decoupled from wall time, a teacher that takes 4 seconds to answer
costs nothing in fidelity. Determinism is preserved: `RngSeedManager::SetSeed(episode_seed)`
and the agent is the only nondeterminism, which we handle by recording its outputs.

**Replay mode.** The bridge can be fed a recorded `trace.jsonl` instead of a live agent.
This gives byte-identical re-runs for debugging, and lets us re-score old traces after
changing the verifier.

### 5.6 One feature implementation, three call sites

The L1 feature code is written **once**, in plain C99 with no dynamic allocation, in
`percept/`. It is compiled into:

- the ns-3 build (as a C++ shim),
- the ESP-IDF firmware,
- a Python extension (`cffi`) for offline recomputation from raw logs.

This is the single highest-leverage decision in the project. Two implementations of the
feature layer means the model trains on one distribution and is deployed on another, and you
will spend a week not understanding why the ESP32 behaves differently. One implementation,
one set of normalisation constants, three call sites.

### 5.7 Throughput

A 60 s episode with 6 nodes at 802.11n runs in roughly 2–8 s of wall time depending on
traffic, plus agent latency. With the student in the loop that is ~1000 episodes/hour on one
machine, and it parallelises across seeds trivially (`GNU parallel`, one process per core).
With the LLM teacher in the loop it is ~10 min/episode dominated by API latency, which is why
teacher episodes are budgeted carefully in §7.2.

---

## 6. The model ladder

The brief asks a specific question: *run a large model in simulation as a teacher, capture
the situation-to-action traces, distil them into a small model that fits the drone, then
measure the gap.* We build three rungs so that gap is measurable rather than asserted.

| Rung | Model | Where it runs | Size | Latency/decision | Purpose |
|---|---|---|---|---|---|
| **T** Teacher | Claude Opus (golden set) / Sonnet (bulk), tool-calling | ground station, simulation only | — | 2–8 s | generates the training signal; the ceiling |
| **M** Mid student | Qwen2.5-0.5B-Instruct, LoRA fine-tuned on the same traces | laptop / Jetson | ~350 MB int4 | 200–600 ms | measures how much is lost going from a frontier model to a small LLM |
| **S** Drone student | custom multi-head TCN, int8 | **ESP32 / ESP32-S3** | **~46 KB** | **2–15 ms** | what actually flies |
| **B** Baseline | hand-written threshold rules | anywhere | ~2 KB | <1 ms | the floor; if S does not beat B, nothing was learned |
| **O** Oracle | policy given ns-3's true cause | simulation only | — | — | upper bound on achievable score |

Two gaps get reported: **T→S** (the one the brief asks for) and **B→S** (proof the learning
was worth doing). M sits between them and tells you *where* the loss happens — in the
language model shrinking, or in leaving language behind entirely.

### 6.1 Why there is no LLM on the drone — the arithmetic

State this plainly in the report, because it is the constraint that shapes everything:

```
ESP32 (WROOM-32)     4 MB flash   520 KB SRAM   240 MHz LX6   no vector unit
ESP32-S3 N16R8      16 MB flash   512 KB SRAM + 8 MB PSRAM    LX7 + ESP-NN SIMD

SmolLM2-360M   @ int4  ≈ 180 MB   →  11× the S3's entire flash
Qwen2.5-0.5B   @ int4  ≈ 250 MB weights (~350 MB running)  →  15×+
smallest useful LLM ~50M @ int4 ≈ 25 MB → still exceeds flash, and weight-streaming
   from PSRAM at ~50 MB/s effective random-access bandwidth costs >500 ms per token
```

There is no arrangement of a language model that fits. The correct conclusion is not "use a
bigger board" — it is that **the language model was never the right form for this
computation.** The output is not text. It is one action out of ten, a handful of discrete
arguments, and a probability vector over eight causes. Free-form generation is exactly what
the brief tells us not to do. So the student is a *constrained function-caller*: a small
neural network with one classification head per field of the tool call. It performs the same
job as a tool-calling LLM — pick a function, pick its arguments, state why — with the
grammar enforced by the architecture instead of by decoding constraints.

### 6.2 The drone student in detail

```
INPUT
  window   T = 16 timesteps @ 10 Hz  ×  48 features         (§3.2)
  belief   previous posterior, 8 dims, broadcast over T
  history  last action one-hot, 10 dims, broadcast over T
  ────────────────────────────────────────────────────────
  tensor   [16, 66]   int8, scale fixed at compile time

ENCODER   dilated depthwise-separable temporal CNN
  proj      1×1 conv 66 → 64                                    4,288 p
  block ×4  depthwise k=3 dilation {1,2,4,8} + pointwise 64→64
            + ReLU6 + residual                              4×4,400 p
  pool      concat[mean_t, max_t, last_t] → 192                     0 p

TRUNK
  fc        192 → 96 + ReLU6                                   18,528 p

HEADS (all discrete → all classification; softmax at the end)
  hypothesis      96 → 8    barrage|spot|reactive|sweep|fading|node_loss|congestion|rx_fault
  action          96 → 10   no_op|scan|silence|nbr_probe|hop|power|reroute|slot|lora|declare
  arg_channel     96 → 14
  arg_power       96 → 8
  arg_route       96 → 5
  arg_move        96 → 7
  arg_slot        96 → 8
  confidence      96 → 1    (sigmoid)
  unrecoverable   96 → 1    (sigmoid)
                                                               ~6,000 p
  ───────────────────────────────────────────────────────────────────────
  TOTAL ≈ 46,400 parameters  →  46 KB int8 weights
          ≈ 366 K MACs per decision
          arena (activations) ≈ 24 KB
```

**Measured budget vs. available:**

| | needed | ESP32 has | headroom |
|---|---|---|---|
| weights (flash) | 46 KB | 4 MB | 87× |
| arena (SRAM) | 24 KB | ~250 KB free after Wi-Fi + mesh | 10× |
| MACs | 366 K | 240 M cycles/s | ~5–15 ms on LX6, ~2 ms on S3 with ESP-NN |
| decision rate | 2 Hz | — | <3% CPU |

If accuracy is short, widen `d` from 64 to 128 — that is ~170 K params / 170 KB, still
comfortably inside a plain ESP32. **The board is not the constraint. Do not let anyone talk
you into a Jetson.**

Why a TCN and not a GRU: TCNs are pure convolution, so they map cleanly onto ESP-NN's int8
kernels and TFLite Micro's supported ops, they have fixed deterministic latency, and they
have no recurrent state to get out of sync when the agent is gated on and off. Long-horizon
memory (sweep period, fade-run histograms) lives in the *feature layer* where it is cheap and
inspectable, and the recurrent element we do keep — the belief vector fed back as input — is
the one piece of state we actually want to be able to read and log.

### 6.3 Deployment path

```
PyTorch (fp32, QAT-aware)
   └─ torch.onnx.export                     → model.onnx
      └─ onnx2tf / onnx-tf                  → saved_model
         └─ TFLiteConverter, int8 full      → model.tflite   (representative dataset
            └─ xxd -i                       → model_data.cc   = 2000 real windows)
               └─ TFLite Micro + ESP-NN     → firmware
```

Two runtimes are viable and we support both behind one C API (`agent/infer.h`):

- **TFLite Micro** (`esp-tflite-micro` component) — portable, easy, works on plain ESP32.
  This is the default.
- **ESP-DL** — Espressif's own int8 engine with hand-written LX7 assembly. 3–8× faster on
  ESP32-S3. Worth it only if we widen the model or want the power number.

**The int8 equivalence gate.** Before anything ships, run all 2000 validation windows
through fp32 PyTorch and through the ESP32 firmware and assert that the argmax action agrees
on ≥ 99.5% of them and the belief L1 distance is < 0.02. Quantisation bugs are silent and
they look exactly like "the model doesn't generalise to hardware". Catch them here.

---

## 7. Training

### 7.1 Stage map

```
A  scenario corpus + baseline B + verifier              week 3   no learning yet
B  teacher rollouts, filtered                           week 4–5
C  behaviour cloning (BC)  → student S₀                 week 5
D  DAgger, 3 rounds        → S₁ S₂ S₃                   week 6   ← the important one
E  cost-sensitive fine-tune (verifier reward)           week 6   optional
F  QAT → int8 → on-device equivalence                   week 7
G  sim-to-real calibration + small real-data tune       week 7–8
```

### 7.2 Stage B — the teacher, and how to afford it

**Setup.** The teacher gets the tool schema from `contract/agent_contract.json`, the same
action costs, the same budgets, and the same observations the student will get — rendered as
a compact table rather than a raw float vector, with units and a short baseline column so the
model can see what "abnormal" means:

```
t=+3.2s since anomaly | budget: hops 3/3 scans 4/4 actions 11/12

                        now      baseline    Δ
pdr  link A            0.08        0.97    -0.89
pdr  link B            0.11        0.95    -0.84
pdr  link C            0.09        0.96    -0.87
rssi link A          -61 dBm     -60 dBm     -1
rssi/pdr corr           0.04        —          (fading would be > 0.6)
noise floor           -74 dBm     -92 dBm    +18
energy w/o preamble     0.97        0.05
noise | after own TX   +0.4 dB   (reactive would be > 6 dB)
retry rate              0.94        0.03
last spectrum scan   never
peer heartbeat gaps   0.3 / 0.4 / 0.3 s   (all peers alive)
own speed             4.1 m/s   displacement since onset 13 m

history: (none — first decision this episode)
```

**Critical rule: the teacher never sees ground truth.** If it does, its traces encode a
capability the student can never have, and the distillation silently teaches the student to
guess. The teacher's advantage must be *reasoning*, not information.

**Output format.** Forced JSON matching the trace schema in §3.3 — belief, action, args,
`why`, confidence, unrecoverable. Sampled `k = 5` times at temperature 0.7; we keep the
majority action and the mean belief (self-consistency). Disagreement among the 5 is itself a
useful label: it marks the genuinely ambiguous states, and we upweight those in training.

**Rejection sampling.** Run the full episode, score it with the verifier, and **discard any
episode the teacher got wrong.** We distil a *filtered* teacher, which is meaningfully
better than the raw teacher — this is why the student can sometimes beat the teacher's own
average score, and it is worth a paragraph in the report.

**The cost problem, and the trick that solves it.** Naively: 5000 episodes × 40 decision
epochs × 5 samples × ~2.5 K tokens ≈ 2.5 B tokens. Not affordable. Three reductions:

1. **The hypothesis head does not need the teacher at all.** ns-3 already knows the true
   cause, for free, for every tick of every episode. So the belief head is trained by
   ordinary supervised learning on unlimited simulator labels. The teacher is only needed
   for the part ns-3 cannot label: *which experiment to run and when to stop investigating.*
   This alone removes ~70% of the required teacher calls.
2. **Only query at decision points.** The anomaly gate means a typical episode has 8–12
   agent decisions, not 600 ticks.
3. **Tiered teacher.** Sonnet for the bulk corpus, Opus for a 300-episode golden set and for
   every episode in the ambiguous/composite families. Compare the two on the golden set to
   confirm Sonnet's labels are good enough.

Revised budget: 1,500 teacher episodes × 10 decisions × 5 samples = 75,000 calls at
~2 K in / 250 out ≈ **150 M input / 19 M output tokens**, most of it Sonnet, and most of the
input cacheable because the system prompt and tool schema are constant. Add DAgger rounds
(§7.4) at ~15,000 calls each. Budget this in week 1 and check it against your actual
allowance before you commit to the corpus size — this is the single most common way a
distillation capstone dies.

### 7.3 Stage C — behaviour cloning

Multi-task loss over each decision sample:

```
L = λ_h · L_hyp  +  λ_a · L_act  +  λ_g · L_args  +  λ_u · L_unrec  +  λ_c · L_calib

L_hyp   expected-cost loss (not plain cross-entropy):  L_hyp = Σ_c  C(y, c) · p_c
L_act   cross-entropy vs the teacher's majority action, masked by availability
L_args  cross-entropy on the argument head belonging to the chosen action only
L_unrec BCE vs ns-3 ground truth "was this episode actually recoverable"
L_calib Brier score on the hypothesis posterior, so confidence means something
```

`λ = (1.0, 1.0, 0.5, 0.5, 0.2)` to start; sweep later.

**The cost matrix `C(true, declared)`** is where the brief's asymmetry gets encoded. Using
`Σ_c C(y,c)·p_c` rather than weighted cross-entropy means the model minimises *expected
operational cost*, which is exactly the thing we score:

```
                         declared →
true ↓        barrage spot react sweep fading node congest rx
barrage           0     1     2     1     4      4     3     5
spot              1     0     2     1     4      4     3     5
reactive          2     2     0     2     4      4     2     5
sweep             1     1     2     0     4      4     3     5
fading           10    10     8    10     0      5     6     6      ← the expensive row
node_loss         3     3     3     3     4      0     3     4
congestion        3     3     2     3     5      4     0     5
rx_fault          5     5     5     5     5      4     5     0
```

Read the `fading` row: declaring any jamming variety when it was fading costs 8–10, because
the response is a channel hop that splits the mesh. Every other confusion costs less. The
matrix is a config file (`train/cost_matrix.yaml`) and is reported in the paper — it is a
design choice, and reviewers should be able to see it and disagree with it.

**Calibration.** After training, fit a single temperature `T` on the validation set by
minimising NLL, and bake it into the exported model. Without this, `θ_act = 0.75` is a
meaningless number and the agent will act on 0.75 beliefs that are right 40% of the time.

### 7.4 Stage D — DAgger, and why it is not optional

BC alone fails in a specific, predictable way. The student is trained on states the *teacher*
visited. In deployment the student visits states the teacher never did — because it made a
slightly different early choice — and its error compounds. Symptom: the student looks great
on the held-out trace set and falls apart in closed-loop episodes. If you see that gap, this
is why.

DAgger fixes it:

```
for round r in 1..3:
    roll out student S_{r-1} in ns-3 for 500 fresh episodes   (closed loop)
    at every decision point the student reached, ask the teacher what it would do
    add (student's state, teacher's action) to the dataset
    retrain from scratch on the union  →  S_r
```

Expect the largest single improvement in the project here — typically 10–25 points of
closed-loop success rate. Budget one full teacher-call round (~15 K calls) per DAgger round.

A useful refinement: only query the teacher where the student is *uncertain* or where the
episode later went wrong (`uncertainty-based DAgger`). Cuts the teacher cost by ~60% at
almost no loss.

### 7.5 Stage E — cost-sensitive fine-tune (optional, high value if time allows)

BC + DAgger imitate the teacher. They cannot exceed it, and they do not optimise the metrics
we actually care about (recovery time, actions consumed, packets lost). One short RL phase,
KL-anchored to the BC policy so it cannot drift into nonsense:

```
reward per episode  R = + 10 · correct_classification
                       +  8 · survived
                       −  0.4 · detection_latency_s
                       −  0.3 · recovery_time_s
                       −  0.5 · actions_consumed
                       −  0.01 · packets_lost
                       − 25 · false_positive_jamming_when_fading
                       − 20 · failed_to_declare_when_unrecoverable

objective  E[R] − β · KL(π_θ ‖ π_BC),   β ≈ 0.05
```

Use GRPO (group-relative, no value network — simplest thing that works at this scale) or
plain REINFORCE with a moving-average baseline. 46 K parameters and a fast simulator make
this genuinely cheap: ~20 K episodes overnight on one machine.

**Guardrail:** re-run the full verifier after RL. Reward hacking here looks like the agent
learning to `no_op` its way to a low action count. The verifier's `survived` and
`packets_lost` terms should catch it; check anyway.

### 7.6 Stage G — sim-to-real

The gap between ns-3 and five ESP32s on a desk is real and it lands almost entirely in the
feature layer, not the policy.

1. **Domain randomisation during training.** Randomise per-episode: noise-floor bias
   (±6 dB), RSSI bias (±5 dB) and per-frame noise (σ = 2 dB), feature-update jitter
   (±20 ms), a 1–3 tick observation delay, and 2% dropped feature updates. A policy that
   survives this survives hardware.
2. **Measure the real distribution first.** Before any real fine-tuning, run the ESP32 rig
   through all seven attack families and log the raw feature vectors. Plot sim vs real
   marginals for all 48 features. Fix the *simulator* where they disagree badly (usually the
   noise-floor scale and the retry semantics), rather than papering over it in the model.
3. **Recalibrate normalisation, not weights.** Most of the gap closes by re-fitting the
   per-feature normalisation constants on real captures. This is a config change, no
   retraining.
4. **Then, if needed,** fine-tune the last two layers on ~200 hand-labelled real episodes
   (labels are free: you control the jammer, so you know the truth). Freeze the encoder.

---

## 8. The verifier

The verifier is a separate program that reads `truth.jsonl`, `percept.jsonl` and
`trace.jsonl` and produces a scorecard. It shares no code with the agent. Written in week 3,
**before** any learning, so that every model we ever train is scored by the same instrument.

### 8.1 Metric definitions

**Detection latency.** `t_detect − t_onset`, where `t_detect` is the first time the agent's
`argmax(belief)` equals the true cause *and stays there for 3 consecutive decisions*
(single-tick flickers do not count). If it never happens, the episode is **censored**, not
dropped — report latency with a Kaplan–Meier estimator and always report the censoring rate
alongside. Dropping the failures is the easiest way to publish a beautiful, false latency
number.

**Correct classification.** Full 8×8 confusion matrix, plus macro-F1, plus **expected
operational cost** `Σ_i C(y_i, ŷ_i) / N` under the matrix in §7.3. The cost number is the
one to lead with, because it is the only one that reflects that the errors are not equal.

**False positive rate — the headline.**
```
FPR_fade = P( argmax(belief) ∈ {barrage, spot, reactive, sweep}  |  true = fading )
```
computed over the fading family alone. Report a second, stricter version:
```
FPR_fade_acted = P( agent issued hop_channel  |  true = fading )
```
because a wrong belief that does not cause a hop did not actually cost anything. The
difference between these two numbers measures how much the confidence threshold `θ_act` is
buying you.

**Recovery time.** `t_recovered − t_onset`, recovered = short-window PDR ≥ 0.8 × pre-onset
baseline sustained 3 s. Censored where never recovered. For genuinely unrecoverable
episodes, recovery time is not defined and the episode is scored on the refusal gate instead.

**Packets lost in recovery.** `∫ (offered_rate − delivered_rate) dt` from `t_onset` to
`t_recovered` (or episode end). Counts what the outage actually cost the mission.

**Actions consumed.** Raw count, and cost-weighted count using the action costs in §3.3.
Channel hops are not free and the weighted number says so.

**Survived.** Boolean: at episode end, either (a) a usable route to the sink exists, or
(b) the link was genuinely unrecoverable and the agent declared it and initiated RTH. Both
are successes. Hopping forever on an unrecoverable link is a failure even though the drone
"kept trying".

### 8.2 The refusal gate (pass/fail, not averaged)

The refusal family is scored separately and is a hard gate:

```
PASS iff   declared link lost within 25 s of onset
     and   channel hops ≤ 3
     and   RTH initiated
     and   no further hop attempts after declaration
```

Report as a pass rate over ≥ 50 refusal episodes. A model with an excellent average score
and a 60% refusal-gate pass rate has not solved the problem the brief poses.

### 8.3 Evaluation protocol

- **Splits by seed and by family.** Train on seeds 0–4999. Validate on 5000–5499. Test on
  6000–6499, *plus* a **held-out family** never seen in training (sweeping jammer, and the
  fading + spot composite). Generalising to a novel attack is the interesting result.
- **N ≥ 200 episodes per family** at test time, bootstrap 95% CIs on every number.
- **Paired comparison.** Run every model on the *same* seed list so differences are paired;
  use a paired bootstrap, not independent samples. With 200 episodes an unpaired test will
  not resolve a 5-point difference and a paired one will.

### 8.4 The headline table

| | B rules | S drone (int8) | M Qwen-0.5B | T teacher | O oracle |
|---|---|---|---|---|---|
| detection latency (s, median) | | | | | |
| classification macro-F1 | | | | | |
| **expected cost** | | | | | |
| **FPR fading → jamming** | | | | | |
| FPR (acted) | | | | | |
| recovery time (s, median) | | | | | |
| packets lost | | | | | |
| actions consumed (weighted) | | | | | |
| survived (%) | | | | | |
| **refusal gate pass (%)** | | | | | |
| held-out family survived (%) | | | | | |
| model size | 2 KB | **46 KB** | 350 MB | — | — |
| latency/decision | <1 ms | **~8 ms** | ~400 ms | ~4 s | — |

Plus: **S on ESP32 hardware** as an extra column once Tier 2 runs, so the sim-to-real gap is
visible in the same table.

### 8.5 Ablations (each one answers a question a reviewer will ask)

| Ablation | Question it answers |
|---|---|
| remove `rssi_pdr_corr` (feature 12) | is S1 really what prevents the false positive? |
| remove `energy_no_preamble` (20) | how much does jamming-vs-congestion depend on it? |
| passive only — no epistemic actions allowed | how much is the *agentic* part worth vs a classifier? |
| no `silence_probe` | can reactive jamming be caught passively via feature 22 alone? |
| BC only, no DAgger | quantifies compounding error |
| plain CE instead of the cost matrix | quantifies the value of encoding asymmetry |
| fp32 vs int8 | quantisation cost |
| window T = 4 / 8 / 16 / 32 | how much history is needed |
| teacher with ground truth (cheating) | shows why we forbade it — student trained on it should collapse |

The third row is the most important experiment in the project. If passive-only scores nearly
as well as the full agent, the agentic framing was decoration. Expect it to be clearly worse
on reactive jamming and on the fading/barrage pair, and be prepared to report it honestly if
it is not.

---

## 9. Tier 2 — the ESP32 rig

### 9.1 Bill of materials

| Item | Qty | Unit (₹) | Total (₹) | Notes |
|---|---|---|---|---|
| ESP32-WROOM-32 DevKitC | 4 | 400 | 1,600 | peer nodes |
| ESP32-S3-DevKitC-1 N16R8 | 2 | 900 | 1,800 | agent node + spare; ESP-NN acceleration |
| ESP32-WROOM-32 (jammer) | 1 | 400 | 400 | dedicated attacker |
| nRF24L01+ module | 2 | 150 | 300 | **2.4 GHz spectrum monitor** — see note |
| SX1278 LoRa 433 MHz | 2 | 350 | 700 | makes `fall_back_to_lora()` real |
| USB hub (powered, 7-port) | 1 | 900 | 900 | all boards on one laptop for logging |
| Micro-USB / USB-C cables | 7 | 80 | 560 | |
| 2.4 GHz attenuators / RF foil / boxes | — | 500 | 500 | manufacturing fading without a field |
| **Core total** | | | **≈ 6,760** | |
| *optional* RTL-SDR v3 | 1 | 2,200 | 2,200 | see note |

**Important correction on the RTL-SDR.** A stock RTL-SDR (RTL2832U + R820T2) tunes to about
**1.7 GHz**. It **cannot see 2.4 GHz** without a downconverter. The brief's Tier-3 idea is
right but the part is wrong for this band. Three ways to get real spectral confirmation:

1. **nRF24L01+ as a spectrum scanner (₹150, recommended).** Its `RPD` register reports
   received-power-detected per 1 MHz channel across 2.400–2.525 GHz. Sweeping all 126
   channels gives a genuine waterfall of the ISM band at ~20 Hz. This is the classic
   "poor man's 2.4 GHz analyser" and it is entirely adequate for confirming the jammer is
   doing what we think.
2. **A spare ESP32 in promiscuous channel-hopping mode** as an independent monitor — free,
   and it measures with the same instrument the agent uses (which is a limitation as well as
   a convenience).
3. **RTL-SDR for the LoRa band.** At 433 MHz the RTL-SDR works perfectly, so buy it to verify
   the `fall_back_to_lora()` path rather than the 2.4 GHz attack.

Budget either way stays under ₹7,000 for a rig that does everything the brief asks.

### 9.2 Mesh stack choice

**painlessMesh** for the mesh itself (matches the brief, gives self-healing routing and a
watchable topology), plus a **raw ESP-NOW side-channel** for probes and control, because
ESP-NOW gives a per-packet transmit-status callback that painlessMesh does not expose. The
two coexist on the same radio.

### 9.3 Sensing on real silicon — what the ESP32 actually gives us

| Feature group | ESP-IDF mechanism | Fidelity |
|---|---|---|
| RSSI per frame | `esp_wifi_set_promiscuous(true)` + callback → `wifi_promiscuous_pkt_t.rx_ctrl.rssi` | exact |
| **Noise floor** | same struct: `rx_ctrl.noise_floor` (dBm, signed) | **real, per received frame** — this is the key enabler and it is genuinely exposed |
| Channel, rate, sig_mode, sig_len | `rx_ctrl.channel`, `.rate`, `.sig_mode`, `.sig_len` | exact |
| PDR / ACK | `esp_now_register_send_cb` → `ESP_NOW_SEND_SUCCESS` / `_FAIL` (MAC-layer ACK) | exact per unicast |
| Retry rate | count of send-callback failures + application-level sequence gaps | good proxy |
| Channel hop | `esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE)` | exact, ~2 ms |
| Spectrum scan | hop + promiscuous dwell 20–50 ms per channel, accumulate noise_floor and frame count | good |
| TX power | `esp_wifi_set_max_tx_power()` (0.25 dBm units) | exact |
| Peer heartbeat | painlessMesh node list + last-seen timestamps | exact |

**The one honest gap: CCA busy fraction and energy-without-preamble (features 19, 20).**
The ESP32 does not expose raw clear-channel-assessment time. Approximations, in order of
preference:

1. `noise_floor` elevated **and** decoded-frame rate near zero ⇒ energy without preamble.
   This is a coarse but directionally correct substitute and it is what we use.
2. The nRF24L01+ monitor gives true carrier-detect duty cycle per 1 MHz bin; on the rig we
   can feed it in as a *reference* to validate the approximation, though the drone itself
   would not have it.
3. In the simulator, compute feature 20 the same coarse way — **degrade the simulator to
   match the hardware, not the other way round.** If the sim gives the model information the
   ESP32 cannot provide, the sim-to-real gap will be large and mysterious.

Write this constraint into `percept/` from day one. Do not implement the ideal feature in
ns-3 and then discover in week 7 that the hardware cannot compute it.

### 9.4 Jammer firmware — one board, five modes over serial

The jammer is a single ESP32. Modes are selected by a serial command from the host
orchestrator, and the *selected mode plus its start/stop times is written into
`truth.jsonl`*, so on hardware the attacker's own schedule is the ground truth. In rough C:

```c
// BARRAGE (C1): hold every channel with raw energy, round-robin so no channel escapes
case BARRAGE:
  for (ch = 1; ch <= 13; ch++) {
    esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    for (int i = 0; i < BURST; i++)          // junk 802.11 frames at max rate/power
      esp_wifi_80211_tx(WIFI_IF_STA, junk, sizeof(junk), false);
  }
  break;

// SPOT (C2): same, but pinned to one channel; add a duty cycle to make it subtle
case SPOT:
  esp_wifi_set_channel(target_ch, WIFI_SECOND_CHAN_NONE);
  if (duty_gate()) blast(BURST);
  break;

// SWEEP (C4): step the target channel on a timer
case SWEEP:
  if (now - last_hop > dwell_ms) { target_ch = next_ch(target_ch); last_hop = now; }
  esp_wifi_set_channel(target_ch, WIFI_SECOND_CHAN_NONE);
  blast(BURST);
  break;

// REACTIVE (C3): promiscuous RX; on a frame from the agent's MAC, fire a burst
case REACTIVE:
  // installed once: esp_wifi_set_promiscuous_rx_cb(on_rx)
  //   on_rx(): if src_mac == AGENT_MAC && rand() < p_fire  -> blast(REACT_BURST)
  break;

// CONGEST (C7): legitimate-looking heavy traffic, not pure energy -> decodable preambles
case CONGEST:
  esp_now_send(broadcast, payload, len);     // flood, but valid frames
  break;
```

Notes that matter for fidelity:

- **Purest energy comes from RF test mode.** For a barrage/spot jammer that is *energy* and
  not *packets* (so it exercises feature 20, energy-without-preamble, correctly), use
  Espressif's RF certification / continuous-TX path (`esp_wifi` "tx continuous" test build)
  to emit a modulated carrier on the target channel. Keep the `esp_wifi_80211_tx` flood as a
  separate, correctly-labelled **C7** source — it is congestion, and labelling it as jamming
  would poison the training set.
- **Reactive timing is real here.** The promiscuous callback fires microseconds after the
  agent's preamble, which is exactly the effect ns-3 models with a scheduled event. Log the
  reaction-delay histogram on hardware and compare it to the simulator's — that plot is a
  strong slide.
- **Fading with no jammer** needs no firmware at all: turn the jammer off and create fades
  physically (move a node to the edge of range, put RF foil or a body between two boards).
  This is the real false-positive trap and it is the most convincing thing in the demo.

### 9.5 Ground truth on hardware

You control the jammer, so you own the truth. The host orchestrator (`rig/orchestrator.py`)
drives the attack schedule — `t=10s spot ch6`, `t=25s off`, `t=30s sweep`, … — and that
schedule *is* `truth.jsonl`. Peer nodes stream their per-link RSSI, `noise_floor`, and
send-callback outcomes to the host over UART, which becomes `percept.jsonl`; the agent node's
decisions become `trace.jsonl`. **The same verifier that scored ns-3 scores the rig,
unchanged.** One verifier, one feature layer, one model, two channel realities.

### 9.6 What Tier 2 proves that Tier 1 cannot

Real capture effect, real hidden terminals, real clock drift between the reactive jammer and
its target, real timing jitter, real multipath off the walls of the room. If the student —
trained only in ns-3, with domain randomisation — still heals the mesh on the desk, the
sim-to-real story is *demonstrated* rather than argued. The two-minute demo video is three
panes side by side: the nRF24L01+ (or spare-ESP32) spectrum waterfall showing the jammer
energy appear, the mesh topology graph healing itself, and the agent node's live belief
vector resolving to the correct cause and firing the right action.

---

## 10. Repository layout

```
EAG_V3_final_project/
├── docs/
│   ├── DESIGN.md                 ← this document
│   ├── metrics.md                verifier metric definitions, frozen
│   └── report/                   the capstone write-up + figures
├── contract/
│   ├── agent_contract.json       tool + action schema — the shared interface
│   ├── budgets.yaml
│   └── cost_matrix.yaml
├── percept/                      L1 feature layer, C99, no malloc  ← ONE implementation
│   ├── features.c / features.h
│   ├── norm.h                    normalisation constants (sim == hardware)
│   ├── ring.c  cusum.c  corr.c
│   └── py_bindings/              cffi wrapper for offline recompute
├── sim/                          Tier 1
│   ├── NS3_VERSION
│   ├── model/jammers/            barrage spot sweep reactive congestion nodeloss rxfault
│   ├── scenario/                 topology, traffic, mobility, fading
│   ├── bridge/agent_bridge.cc    UNIX-socket JSON bridge (§5.5)
│   ├── run_episode.py            one episode end to end
│   └── gen_corpus.py             parallel scenario generator
├── agent/
│   ├── infer.h / infer_tflm.c    on-device C inference API (student)
│   ├── student/                  PyTorch model, export, quantise
│   ├── teacher/                  prompt, tool loop, self-consistency, rejection sampling
│   ├── baseline/                 threshold rules B
│   └── safety/failsafe.c         dead-man rule + logit masking (§4.6)
├── train/
│   ├── bc.py  dagger.py  rl_grpo.py
│   ├── cost_matrix.yaml
│   └── datasets/
├── verify/
│   ├── verifier.py               reads truth/percept/trace → metrics
│   ├── gates.py                  FP gate + refusal gate
│   └── report.py                 → reports/eval_<sha>.html
├── firmware/                     Tier 2
│   ├── node/  jammer/  observer/
│   └── components/               symlinks to percept/ and agent/
├── rig/
│   ├── orchestrator.py           attack schedule = ground truth
│   └── sdr_capture.py            RTL-SDR spectrum pane
└── runs/  reports/               generated, git-ignored
```

**The three shared-interface files are frozen in week 1:** `contract/agent_contract.json`,
`percept/norm.h`, `verify/metrics.md`. Everything else can churn. Those three are what let
four subsystems and two people work in parallel without integration hell.

---

## 11. Timeline — 8 weeks, split two ways

Work splits along a clean seam: **channel/simulation/verifier** vs **agent/model/training**.
The `contract/` files are the handshake between the two halves.

| Wk | Prateek — sim, RF, verifier | Jatin — agent, model, training | Joint milestone |
|---|---|---|---|
| 1 | ns-3 up; `SpectrumWifiPhy` + one barrage jammer breaks a link | teacher prompt + tool loop returns valid JSON on a hand-built state | **contract + norm.h + metrics.md frozen** |
| 2 | all 7 jammers; ground-truth logging; bridge socket | baseline B; feature layer C99 + cffi; verifier skeleton | one live episode: sim → agent → action → log |
| 3 | fading tuned until indistinguishable from barrage (2-sample test); corpus gen | verifier complete; the two gates; oracle policy | **corpus v1 (5k episodes) + honest verifier** |
| 4 | domain randomisation; composite scenarios | teacher rollouts + rejection sampling; belief head on ns-3 labels | 1.5k filtered teacher episodes banked |
| 5 | throughput tuning; replay mode | BC → S₀; int8 export path working end to end | S₀ scored vs B on held-out set |
| 6 | — support DAgger rollouts | **DAgger ×3 → S₃**; cost-sensitive RL if time | **S₃ beats B on FP + refusal gates** |
| 7 | ESP32 flashed; real feature capture; sim-vs-real marginals | QAT → int8 → ESP32; on-device equivalence gate | student runs on ESP32; equivalence < 0.5% |
| 8 | rig orchestrator; RTL-SDR pane; demo capture | sim-to-real recalibration; final eval sweep | **headline table + demo video + report** |

**Tier-1-complete is the end of week 6** and is a full capstone on its own. Weeks 7–8 are the
hardware payoff. If hardware slips, the project is still complete and defensible at week 6 —
which is exactly the ordering the brief insists on.

---

## 12. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fading episodes are secretly easier than jamming → FP number is a lie | High | Fatal to the thesis | Week-3 two-sample test on feature marginals is a *gate*, not a nicety; if it fails, retune fading before proceeding |
| BC looks great on traces, collapses closed-loop | High | Medium | Expected — that is what DAgger (§7.4) is for; budget it in from the start |
| Teacher token budget blows up | Medium | High | Belief head uses ns-3 labels not the teacher (§7.2); tiered Sonnet/Opus; uncertainty-based DAgger; **cost-check in week 1** |
| int8 quantisation silently changes behaviour | Medium | High | The equivalence gate (§6.3): 99.5% argmax agreement or it doesn't ship |
| Two feature implementations drift | Medium | High | Structurally prevented: one C99 file, three call sites (§5.6) |
| ns-3 reactive jammer can't react fast enough | Medium | Medium | Model reaction as a scheduled event on a PHY trace source, not a polled loop; validate the delay histogram |
| ESP32 "jamming" is really just congestion | Medium | Low | Use RF continuous-carrier test mode for pure energy; keep packet-flood as a separate, correctly-labelled C7 |
| Reward hacking in RL (no_op to save actions) | Medium | Medium | Re-run full verifier after RL; `survived` + `packets_lost` terms penalise it; KL anchor to BC |
| Hardware slips | Medium | Low | Tier 1 is a complete capstone by design; hardware is upside, week 7–8 |
| Scope creep into a real drone / real SDR TX | Low | High | Explicitly out of scope; the desk rig is the deliverable |

---

## 13. What "done" looks like

1. A simulator that manufactures all seven causes with ground truth, and whose fading is
   provably as hard as its jamming.
2. A verifier that reports detection latency, classification accuracy, the false-positive
   rate on fading, recovery time, packets lost, actions consumed, and survival — per family,
   with confidence intervals, tied to a code version.
3. A ~46 KB function-calling student that runs on a bare ESP32 at 2 Hz using <3% CPU, chosen
   over an LLM for reasons the report states with arithmetic.
4. The headline table: Oracle ≥ Teacher ≥ Student ≫ Baseline, with the T→S gap small and the
   S→B gap large, and the ESP32 numbers matching the sim numbers.
5. The refusal case working *by construction*: broadband-everywhere + no peer ⇒ declare lost
   ⇒ RTH, in ≤ 3 hops, guaranteed by the safety envelope and verified as a pass/fail gate.
6. A rig where five ESP32s heal their mesh around a sixth that is jamming them, with the
   RTL-SDR showing the jammer energy on screen — the two-minute video.

The one-sentence version, which is the thing to defend: *most of the intelligence lives in
the feature layer and the verifier's honesty; the learned part is small on purpose, and we
can measure exactly how small we could make it before it broke.*
