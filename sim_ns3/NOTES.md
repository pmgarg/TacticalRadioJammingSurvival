# ns-3 integration notes

## Why SpectrumWifiPhy, not YansWifiPhy
Yans models interference only from other 802.11 PHYs. A WaveformGenerator jammer is not a
Wi-Fi device, so under Yans it would be **completely invisible** and every jamming scenario
would silently do nothing. `SpectrumWifiPhy` + `MultiModelSpectrumChannel` lets arbitrary
spectrum emitters and Wi-Fi PHYs share one channel and actually interfere.

## Jammer PSDs
`SpectrumValue5MhzFactory::CreateTxPowerSpectralDensity(watts, channel)` builds a
2.4 GHz ISM PSD on a 5 MHz-resolution model, including the 802.11 transmit mask
(-28 dB and -40 dB shoulders). Barrage sums per-channel PSDs; sweep re-sets the PSD on a
timer; spot pins one channel.

Note the shoulders are physically real: a "spot" jammer on ch 6 measurably raises ch 4-8,
because 2.4 GHz Wi-Fi channels overlap. Channels 1/6/11 are the only non-overlapping trio.
Our measured spot profile (ch1-3 at -100 dBm, ch4-8 at -75) shows exactly this.

## Measuring the noise floor
`MonitorSnifferRx` gives `SignalNoiseDbm` (signal AND noise) — the same pair an ESP32
exposes via `rx_ctrl.rssi` / `rx_ctrl.noise_floor`. BUT it only fires on **successful**
reception, so under heavy jamming it goes silent exactly when the reading matters most.

Therefore a `SpectrumAnalyzer` is co-located with the agent for a continuous per-channel
band-power reading (this is also what `spectrum_scan()` reads). Caveat: it hears the agent's
own transmissions at ~0 m, so we take a **min-hold** over each 100 ms sample period. Real
radios have the same problem and use the same trick.

## Verified
- ns-3.45, optimized, modules: wifi spectrum propagation olsr mobility applications
  internet energy flow-monitor stats
- Compiles clean; barrage/spot/sweep/fading/dead-peer scenarios all run
- Spot jammer takes PDR 0.65 -> 0.00 at onset; per-channel floor profile separates
  spot (25 dB spread) from barrage (4 dB spread)

## TODO
- AgentBridge: UNIX-socket JSON loop so the Python agent drives the episode at 1 Hz
- Reactive jammer: trigger the waveform off the agent's PhyTxBegin trace
- Congestion / hidden-terminal geometries
- Gate the analyzer sample on own-TX to get a true noise floor
