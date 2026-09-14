# Fidelity gate — refsim vs ns-3

Design §10.1 requires that the fast reference simulator agree with authoritative ns-3
before it may be used for bulk training. Same scenarios, same feature extractor, post-onset
marginals compared.

## Result: 37/40 feature-family pairs agree within 0.5 (normalised units)

Fixed during cross-validation (both were real modelling errors):

1. **refsim had no adjacent-channel leakage.** 2.4 GHz channels are 5 MHz apart but 20 MHz
   wide, so a "spot" jammer leaks heavily into neighbours. refsim treated channels as a brick
   wall, so its sweep jammer barely hurt (PDR 0.87) while ns-3's crushed the link. Added an
   802.11-mask leakage table (0/-3/-8/-15/-28/-40 dB by channel offset).
2. **refsim summed barrage power wrong** — took a max over its channels instead of summing
   them, making barrage ~9 dB too weak.
3. **ns-3 never implemented `fade_enter`/`fade_exit`.** The fading scenario had no onset at
   all — it was constant Nakagami. Now schedules extra path loss plus a Rician→Rayleigh
   collapse (m: 3 → 1), matching the refsim model.
4. **refsim reported RSSI for frames it never received.** Real hardware and ns-3 report a
   stale value when nothing decodes. refsim now carries the last measured value.

## Remaining divergences (documented, not fixed)

| family | feature | ns-3 | refsim | note |
|---|---|---|---|---|
| sweep | pdr_fast | -0.75 | -0.17 | ns-3's sweep is more damaging; refsim is **conservative**, so a student trained on refsim is not flattered |
| sweep | scan_bad_frac | 0.35 | -0.31 | same cause |
| fading | rssi_pdr_corr (**S1**) | 0.03 | 0.65 | **see below — this is a finding, not a bug** |

## Finding: S1 is far weaker on real hardware than the design assumed

The design names RSSI/PDR correlation (S1) as "the primary false-positive defence". The
cross-validation shows it is **largely unmeasurable in ns-3**, and by extension on an ESP32,
because of **survivor bias**: RSSI is only observable on frames that successfully decode, and
deeply-faded frames do not decode. The surviving samples are systematically better than the
true mean, which flattens the correlation.

**Consequence for the design:** the fading-vs-jamming decision must rest on **S2 (noise-floor
delta)** and **S5 (per-channel profile)**, which ns-3 shows are decisive and unambiguous:

| scenario | noise floor pre | post | Δ | per-channel |
|---|---|---|---|---|
| barrage | -100.8 | -76.8 | +24.0 | all 8 hot, 4 dB spread |
| spot | -100.8 | -75.1 | +25.8 | ch1-3 clean, 25 dB spread |
| **fading** | -100.8 | **-100.8** | **+0.0** | all clean |
| dead peer | -100.8 | -100.8 | +0.0 | all clean |

Actions taken: (a) refsim's S1 is damped toward the ns-3 value; (b) an ablation without
feature 12 is in the eval plan; (c) domain randomisation perturbs the RSSI channel so the
student cannot over-rely on it.
