# Hardware guide — from buying nothing to a working demo

Written assuming you have **never** deployed on hardware. Budget-first: the minimum rig is
about **₹1,100**, the comfortable one about **₹2,400**. Everything here is Tier 2 from the
brief ("4 or 5 ESP32 boards … plus one more ESP32 as the interferer, around ₹2,000").

**Read this first.** Tier 1 (simulation) is already a complete capstone and is what your
numbers come from. Hardware buys you one thing: a demo video where a real mesh really falls
over and really heals. Do not let it put the simulation work at risk. Every step below is
ordered so that you can stop at any point and still have something that works.

---

## 1. What to buy

### Tier MINIMUM — ₹1,100, proves the concept

| # | Item | Why | ~₹ each |
|---|---|---|---|
| 3 | ESP32-WROOM-32 DevKit (38-pin, USB-C or micro-USB) | 1 agent + 1 peer + 1 jammer. Two nodes is a link; you can *see* it break | 280–350 |
| 1 | nRF24L01+ module | **the spectrum scanner.** This is the important one — see §2 | 120–180 |
| 1 | 10 µF electrolytic capacitor | across the nRF24's VCC/GND. Without it the module is famously unreliable | 5 |
| — | Jumper wires (F-F, 20 pcs) + the USB cables you already own | | 60 |

### Tier RECOMMENDED — ₹2,400, makes the video worth watching

Add to the above:

| # | Item | Why | ~₹ each |
|---|---|---|---|
| +2 | ESP32-WROOM-32 DevKit (5 total) | a 4-node mesh has real multi-hop, so `reroute` and `node_loss` mean something | 280–350 |
| 1 | Powered USB hub (4–7 port) | five boards off one laptop port browns out and you will waste an evening on it | 500–700 |
| 1 | nRF24L01+PA+LNA + antenna (optional) | longer range, better scans; needs a stable 3.3 V supply | 250 |

**Do not buy** an RTL-SDR for this. A stock RTL-SDR (R820T2) tunes ~24 MHz–1.7 GHz and
**cannot see the 2.4 GHz band you are jamming**. It is only useful for the sub-GHz LoRa
fallback story, which is not on the critical path. The ₹2,500 is better spent on two more
ESP32s.

**Where to buy (India):** [Robu.in](https://robu.in/product/esp32-38pin-development-board-wifibluetooth-ultra-low-power-consumption-dual-core/),
[Robokits](https://robokits.co.in/wireless-solutions/iot-esp-module/esp32-development-board-wifi-bluetooth) (₹284 at the time of writing),
[Probots](https://probots.co.in/esp32-wroom-32u-chipset-module.html), Quartz Components, or
local Lamington Road / SP Road. Prices move; check two before ordering.

**Board choice:** plain **ESP32-WROOM-32** is fine and cheapest. ESP32-S3 is nicer (more
SRAM for the student) but costs more; the student is 49.6 KB, so a plain ESP32 fits it many
times over. Buy the cheap one.

---

## 2. The one technical decision that matters: how you measure the spectrum

Your whole design rests on **S2 (noise-floor rise)** and **S5 (per-channel profile)**.
`FIDELITY.md` proved those two are what separate jamming from fading — and that S1
(RSSI/PDR correlation) is largely unmeasurable. So the rig must be able to measure channel
energy, or the demo cannot tell a jammer from a fade.

`DESIGN.md` §11.2 assumes the ESP32 gives you this via `rx_ctrl.noise_floor` in promiscuous
mode. **Verify that on day one — do not assume it.** There is a long-standing report that
this field reads **0** on some ESP-IDF versions
([esp-idf#1751](https://github.com/espressif/esp-idf/issues/1751),
[#8022](https://github.com/espressif/esp-idf/issues/8022)), and if it is zero your S2 is
gone.

**The fallback, and honestly the better primary:** the **nRF24L01+ RPD** (Received Power
Detector). Set the radio to channel *n*, listen ~200 µs, read one bit: "was there energy
above roughly −64 dBm?". Sweep 1–125 in 1 MHz steps and you have a real occupancy scan of
2.400–2.525 GHz for ₹150. Counting hits per channel over many sweeps turns that one bit
into a usable energy profile. This is a well-trodden technique
([Hackster](https://www.hackster.io/CiferTech/how-to-make-2-4-ghz-band-scanner-with-nrf24l01),
[ioprog](https://ioprog.com/2015/07/27/nrf24l01-channel-scan/)) and it maps directly onto
your `spectrum_scan()` call.

So: **the nRF24 is your `hal_scan()`.** The ESP32 radio is your mesh. That split is clean
and it is what the wiring below assumes.

---

## 3. OS and toolchain

You are on a Mac, so:

| Need | What | Note |
|---|---|---|
| USB-serial driver | CP2102 (most ESP32 devkits) works natively on macOS 11+. **CH340/CH9102** boards need a driver | This is the classic day-1 blocker. Check which chip your board has before ordering if you can |
| Toolchain | **Arduino IDE 2.x** — fastest to a working blink. ESP-IDF v5.x if you want `esp_now` + promiscuous callbacks properly | Start with Arduino IDE. Move to ESP-IDF only when you need the raw Wi-Fi callbacks |
| Board support | Arduino IDE → Preferences → Additional Board URLs → `https://espressif.github.io/arduino-esp32/package_esp32_index.json`, then Boards Manager → "esp32" | |
| Library | `RF24` by TMRh20 (Library Manager) for the nRF24 | |
| Port check | `ls /dev/cu.*` before and after plugging a board in — the new entry is your port | |

Nothing else. No ns-3 on the boards; the simulator stays on your laptop.

---

## 4. Wiring — nRF24L01+ to ESP32 (the only wiring in the project)

The nRF24 uses SPI. ESP32 default VSPI pins:

```
  nRF24L01+ (8-pin, 2x4 header, notch/antenna facing away from you)

        ┌──────────────┐
   GND  │ 1        2   │ VCC   ──► ESP32 3V3   *** NOT 5V. 5V kills it. ***
   CE   │ 3        4   │ CSN
   SCK  │ 5        6   │ MOSI
   MISO │ 7        8   │ IRQ   ──► leave unconnected
        └──────────────┘

  nRF24 pin      ESP32 GPIO      wire
  ---------      ----------      ----
  GND            GND             black
  VCC            3V3             red      <- 3.3 V ONLY
  CE             GPIO 4          any free GPIO
  CSN  (CS)      GPIO 5          VSPI CS
  SCK            GPIO 18         VSPI CLK
  MOSI           GPIO 23         VSPI MOSI
  MISO           GPIO 19         VSPI MISO
  IRQ            -               not used

  *** Solder or clip a 10 uF capacitor directly across the nRF24's VCC and GND pins. ***
  Nine out of ten "my nRF24 does not work" posts are this. The module draws current in
  short bursts that the devkit's 3.3 V regulator cannot follow.
```

Only **one** board needs the nRF24 — the agent node. The others are plain ESP32s with
nothing wired to them.

---

## 5. Build it in five stages, and test each before the next

This is the same discipline as `docs/CODE_FLOW.md` §5: do not wire stage N+1 until stage N
is proven. On hardware this matters more, because a bad solder joint and a bad algorithm
look identical from the serial monitor.

### Stage H0 — one board blinks (30 min)

Arduino IDE → Board "ESP32 Dev Module" → Port → File ▸ Examples ▸ 01.Basics ▸ Blink →
Upload. If the upload fails, hold **BOOT** while it says "Connecting…".

**Gate:** the LED blinks. You now know the driver, the cable and the toolchain are fine.
More than half of all hardware projects die here; you just cleared it.

### Stage H1 — two boards talk over ESP-NOW (1–2 h)

Flash one as sender, one as receiver, using the stock ESP-NOW example. Print the received
RSSI (`esp_now_register_recv_cb` gives you `esp_now_recv_info_t` with `rx_ctrl`).

**Gate:** board B prints a packet count and an RSSI from board A, and the count rises at a
steady rate. Walk away with board A and watch RSSI fall. **That is your first real
measurement** — and it is the same `rssi` field `RawObs` wants.

### Stage H2 — the beacon + LinkReport protocol (half a day)

Port the wire format you already designed, `MeshPktHdr` + `LinkReport`
(`sim_ns3/mesh-node-app.h:36-54`) — 12 bytes plus 3 bytes per report. Each node broadcasts
at 10 Hz, and each beacon carries "this is how I hear you" for up to 16 peers.

**Gate:** every node prints, for every peer, *forward* PDR (from sequence numbers) and
*reverse* PDR (from the peer's report about us). Unplug one node; its heartbeat gap grows
on all the others. **That is `node_loss`, live.**

This is the stage that carries over most directly — the packet format, the sequence-counted
PDR, and the TX-shadow timing are all pure bookkeeping that needs only a monotonic clock
(`esp_timer_get_time()`), and none of it compares clocks between nodes.

### Stage H3 — the jammer (2 h)

The sixth board. Four modes, matching your corpus:

| mode | how |
|---|---|
| **spot** | sit on one Wi-Fi channel and transmit continuously (ESP-NOW broadcast in a tight loop, or `esp_wifi_80211_tx` raw frames) |
| **barrage** | hop channels every few ms so all of them are hit |
| **sweep** | walk the channel on a `dwell_ms` timer |
| **reactive** | promiscuous RX; when a frame from the agent's MAC is seen, transmit a short burst |

**Gate:** with the jammer on, the mesh's PDR collapses; with it off, PDR recovers. Record
the serial output of both — that is your before/after.

> The reactive mode is the interesting build and it is also the one your simulator got
> wrong (`AUDIT.md` F3a: wrong channel, armed from t=0, 10 ms bursts instead of 1.5 ms).
> Building it on hardware would probably have caught that. Mention that to the examiner —
> it is a good story about why hardware and simulation check each other.

### Stage H4 — the spectrum scan (half a day)

nRF24 on the agent node. Sweep channels, count RPD hits per channel per sweep, map the
nRF24's 1 MHz channels onto your eight logical Wi-Fi channels (Wi-Fi ch *k* ≈ nRF24 channel
`2412 + 5(k−1) − 2400`, i.e. 12, 17, 22 … 47, each ±10 MHz wide).

**Gate — and this is the demo-critical one:** with the jammer on channel 6, the hit count on
the channels around 6 is high and the rest of the band is quiet. With the jammer off, the
whole band is quiet. **Print that as an 8-bar ASCII histogram on the serial monitor.** It is
the single most convincing thing you can show, because it is S5 — the feature your own
cross-validation found decisive — measured on real radio.

### Stage H5 — the student decides (1–2 days)

Port, in this order, all of which are already written and dependency-free:

1. `percept/norm.py` → `norm.h` (constants — must be byte-identical to the sim)
2. `percept/ring.py` → ring buffers, EWMA, CUSUM, run-length, Pearson
3. `percept/features.py` → the 56-feature vector
4. `agent/controller.py` → the mask + budget + dead-man failsafe
5. `data/policy_v3.json` → a C table and a `for` loop (**start here** — the induced rules
   are ~30 lines of C and cover barrage/spot/sweep at 1.00 held-out precision)
6. `student_weights.h` → the net, a 30-line matmul

**Gate:** the agent node prints a belief over the eight causes at 1 Hz, and with the jammer
on channel 6 it says `spot` and hops. With the jammer off but a node walked out of range, it
says `fading` or `node_loss` and does **not** hop.

**If you only have one day, do steps 1–5 and skip the net.** The rule agent alone scores
43.8% standalone with **zero false positives** — and on hardware, "it never made the
expensive mistake" is a better demo than a higher accuracy number.

---

## 6. Observing and testing

**Observe:** serial monitor at 115200 on each board. Have the agent print one line per
second: `t | pdr | noise-ish | top-8 histogram | belief | action`. That is your live view;
you do not need a screen or a web UI on the device.

**Feed the real dashboard:** have the agent print one JSON line per decision, pipe it into a
file, and point `capstone/dash/serve.py` at it. The record format is the same
`classification_trace` the simulator writes, so **the belief timeline you already built
works unchanged on hardware data.** That is a five-line change and it makes the hardware and
simulation demos look like one system.

**The four tests to run on camera:**

| test | setup | what to show |
|---|---|---|
| **spot jamming** | jammer on ch 6, mesh on ch 6 | scan histogram spikes at 6, agent says `spot`, hops, PDR recovers |
| **fading, no attacker** | jammer OFF, walk a peer behind a wall / out of range | PDR collapses, spectrum stays **quiet**, agent says `fading` and **does not hop** ← the money shot |
| **node loss** | jammer OFF, unplug a peer | heartbeat gap grows on that link only, agent says `node_loss`, reroutes |
| **refusal** | jammer in barrage mode, no clean channel | agent declares the link lost instead of hopping forever |

Test 2 is the one to spend time on. Anyone can show a radio breaking. Showing an agent that
*correctly refuses to act* when the spectrum is clean is the actual thesis.

---

## 7. Budget summary

| | items | ₹ |
|---|---|---|
| **Minimum** | 3× ESP32 + nRF24 + cap + wires | **~1,100** |
| **Recommended** | 5× ESP32 + jammer + nRF24 + hub + wires | **~2,400** |
| Skip | RTL-SDR (cannot see 2.4 GHz), ESP32-S3 (you do not need the SRAM), PA+LNA nRF24 (needs a clean supply) | saves ~3,000 |

---

## 8. Honest risks

1. **`rx_ctrl.noise_floor` may read 0.** Test it in stage H1 with a five-line sketch. If it
   is zero, the nRF24 carries S2 and S5 and nothing is lost — which is why §2 puts the
   nRF24 in the minimum BOM rather than treating it as an extra.
2. **TDMA needs cross-node sync, and the simulator cheats.** In ns-3 every node reads one
   perfect global clock with no guard interval and no drift. On hardware, sync slots off the
   beacon; a 100 ms frame with a 5 ms guard tolerates ±2.5 ms, which ESP32 crystals beat
   comfortably between beacons. **Write this down as a known sim-to-real divergence** rather
   than discovering it on the bench.
3. **Nothing else needs tight timing.** Decisions are 1 Hz, and the TX-shadow statistic uses
   only the local node's own clock — it never compares clocks across nodes. That is
   deliberate and it is what makes this portable to ₹300 hardware.
4. **Five boards on one laptop USB port will brown out.** Powered hub, or power the jammer
   from a phone charger.
5. **Budget the time, not just the money.** H0–H2 is one evening. H3–H4 is a weekend.
   H5 is the part that can overrun — which is why the rule table comes before the net.
