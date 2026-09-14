# PHY → MAC → routing telemetry extension

Motivation: the three weakest families on ns-3 were **reactive (50%)**, **congestion
(59%)** and **hidden-terminal (49%)**. All three degrade the link without killing it, and
all three were being starved of the one statistic that identifies them, because we had
declared S4 (TX-conditioned noise) "unavailable in ns-3" — the spectrum analyser is gated
off while the agent transmits, so there is no noise reading during our own TX.

That conclusion was too quick. The information is recoverable **without** measuring noise
during transmission, and every quantity below exists on real ESP32 hardware.

---

## 1. TX-shadow loss (recovers S4 from TIMING, not from noise)

A reactive jammer fires *because we transmitted*. So its damage is not spread uniformly in
time — it is concentrated in a short window after each of our transmissions. We know
exactly when we transmitted, and neighbour beacons are periodic, so we know when each
beacon *should* have arrived:

```
tau = reaction window (~2-3 ms, covers jammer delay + burst)
for each expected beacon slot t_expect from peer p:
    in_shadow = any(our TX interval overlaps [t_expect - tau, t_expect + tau])
    bucket[in_shadow].total += 1
    bucket[in_shadow].miss  += (beacon did not arrive)

S4' = P(miss | in_shadow) - P(miss | silent)
```

- **Reactive jamming** → S4' strongly positive (losses cluster around our own TX).
- **Barrage / spot** → S4' ≈ 0 (losses are uniform in time; the channel is always hot).
- **Congestion** → S4' mildly positive (our TX genuinely collides with others) but paired
  with a *high* decodable-frame rate, which separates it.
- **Fading** → S4' ≈ 0.

**On ESP32:** we own the transmit timestamps (`esp_now_send` call → send-callback), and
beacons are periodic by construction. No extra hardware, no extra radio time.
**In ns-3:** `PhyTxBegin` / `PhyTxEnd` give the TX intervals exactly.

## 2. Reciprocal link quality (how do my neighbours see ME?)

At the PHY, every received frame carries a measured (RSSI, noise, SINR). We attribute it
to the transmitter (802.11 address on ns-3; `rx_ctrl` + source MAC on ESP32) and keep a
per-neighbour table. Each node then **echoes that table in its own beacon**. A receiver
pulls out the entry about itself and learns the *reverse* link — how well it is being
heard.

This is exactly what real mesh protocols do (batman-adv TQ, OLSR ETX, LTE CQI reporting),
and it costs a handful of bytes per beacon.

What it buys, which no amount of local measurement can give:

| observation | conclusion |
|---|---|
| I hear them fine, they hear me badly | **my transmitter**, or a jammer sitting near *them* |
| I hear them badly, they hear me fine | **my receiver**, or a jammer near *me* |
| both directions bad, noise floor clean | geometry / fading |
| both directions bad, noise floor hot | jamming on the shared channel |
| *all* neighbours report me bad at once | my TX chain or my local RF environment, not the link |

Note this turns `listen_test` and `transmit_probe` — which are **actions costing budget** —
into continuous passive telemetry. The tests remain available for confirmation, but the
agent no longer has to spend budget to get the first hint.

## 3. TX-side KPIs from PHY/MAC

Everything here is TX-side, so it is measurable even when reception has collapsed — which
is exactly the regime where our RX-derived features go blind.

| KPI | ns-3 source | ESP32 source | what it separates |
|---|---|---|---|
| `tx_success_ratio` | ACK from `RemoteStationManager` | `esp_now_register_send_cb` → SUCCESS/FAIL | overall link health, TX-side |
| `tx_retry_depth` | retry count before success/drop | successive send-callback failures | interference vs clean loss |
| **`tx_defer_time`** | enqueue → `PhyTxBegin` | `esp_now_send()` → send-callback latency | **channel occupancy before we can talk**: high under congestion/hidden-terminal, low under barrage (the channel is *noisy*, not *busy* with decodable traffic) |
| `tx_attempt_rate` | `PhyTxBegin` count | send count | our own offered load, for S10 |

`tx_defer_time` is the interesting one. CSMA defers to *decodable* carrier. A barrage
jammer raises the noise floor but is not decodable carrier, so deferral stays low while
delivery collapses. Congestion and hidden terminals are other people's real packets, so
deferral rises. That is a clean, TX-only discriminator between "jammed" and "busy" — and
it is the same distinction S3 (energy without preamble) makes from the RX side, so the two
corroborate each other.

---

## New features (48 → 56), contract 1.0.0 → 1.1.0

```
48 rssi_reverse            how well the worst neighbour hears me
49 pdr_reverse             their delivery ratio from me
50 link_asymmetry          pdr_forward - pdr_reverse      (my RX vs my TX fault)
51 frac_peers_report_me_bad                               (my TX chain vs one link)
52 tx_success_ratio        ACKed / attempted
53 tx_defer_time_norm      channel busy before we can transmit
54 tx_shadow_loss_delta    S4' from section 1             (REACTIVE)
55 tx_retry_depth_mean
```

---

# Measured outcome

Implemented in both simulators and verified on **authoritative ns-3**, then used to retrain.

## Do the new signals actually discriminate? (real ns-3, post-onset)

| family | S4′ shadow-Δ | silent-loss | tx_defer | asymmetry |
|---|---|---|---|---|
| **reactive** | **+0.75** | **−1.00** (clean) | −0.94 | −0.03 |
| fading | +0.82 | −0.08 (lossy) | −0.49 | +0.00 |
| **congestion** | +0.00 | −1.00 | **+0.41 ms** | **+0.43** |
| **hidden_term** | +0.00 | −1.00 | **+0.17 ms** | +0.00 |
| barrage | +0.03 | −1.00 | +1.29 ms | −0.80 |

A physical effect had to be accounted for: **a radio is half-duplex, so it cannot receive
while it transmits.** Shadow-window loss is therefore elevated in *every* scenario, and
S4′ alone is not the reactive signature. What identifies reactive jamming is **high
shadow-Δ while the silent-window baseline is clean** — reactive loses 25% of in-shadow
beacons and 0% of silent ones; fading loses 73% and 46%. So the silent-window loss rate is
carried as its own feature (it replaced `tx_retry_depth`, which turned out to be constant
across families and was dead weight).

## Classification gain on ns-3 features (3,044 held-out states)

| family | 48 features | **56 features** | delta |
|---|---|---|---|
| **congestion** | 59% | **88%** | **+29** |
| **hidden_term** | 49% | **63%** | **+14** |
| spot | 76% | 80% | +4 |
| reactive | 50% | 53% | +3 |
| barrage / node_loss / refusal | 100% | 100% | — |
| **overall** | **80.8%** | **86.4%** | **+5.6** |
| expected cost | 0.400 | **0.295** | −26% |

`tx_defer_time` did exactly what it was designed to do: congestion and hidden-terminal are
the two families defined by *decodable* carrier occupying the channel, and they are the two
that improved most. The reciprocal report is what makes `asymmetry` available at all.

## Honest limits

- **Reactive gained only +3 points offline and remains 0/5 live.** The signal is present
  and unambiguous in the features; the failure is in the closed loop, where the agent's own
  actions change its TX pattern and therefore change the very statistic it is reading. This
  is a genuine feedback pathology, not a missing measurement.
- `tx_success_ratio` is ≈1.0 in almost every family in ns-3 and carries little information;
  on real hardware (per-packet ACK from `esp_now_register_send_cb`) it should be far more
  informative, since ns-3's 802.11 retry machinery hides most failures.
