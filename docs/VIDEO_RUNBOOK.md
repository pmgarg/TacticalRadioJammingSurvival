# Video runbook — showing the agent detect and fix jamming, live

Ten runs: **5 with the live LLM teacher, 5 with the learned student**, same five
scenarios both times, so the comparison is direct. Everything below is the real scored
pipeline — same ns-3 bridge, same agent, same verifier. Nothing is staged for the camera.

---

## 0. What the viewer will actually see

### Teacher (live LLM), `spot_single_channel` — measured end to end

| t | delivery | belief | call |
|---|---|---|---|
| 5–16 s | 100% | fading | exploring, buys scans |
| 17–18 s | 100% | hidden_term | still wrong, nothing has happened yet |
| **19 s** | **39%** | reactive | **`spectrum_scan`** — pays for evidence the moment it hurts |
| **20 s** | **5%** | **spot** | **`hop_channel`** |
| 21 s | 62% | — | recovering |
| **22 s** | **95%** | — | `change_tdma_slot` |
| 24–34 s | 68–78% | wobbles reactive/node_loss | honest: it does not settle |

**Detection at 19 s, correct diagnosis and the hop at 20 s, 95% delivery by 22 s.** Those
three seconds are your video. The belief wobble afterwards is real — leave it in.

### Student (no LLM), same scenario:

| sim time | on screen |
|---|---|
| 5–18 s | delivery 100%, agent buys `spectrum_scan`s, spectrum flat at −101 dBm |
| **19 s** | **delivery collapses to 39%** · the paid scan shows **ch 5 at −74 dBm (+27 dB)** · belief flips to **spot** · agent calls **`hop_channel`** |
| 24 s on | channel is now **1**, delivery back to **~70%** |

That single transition is your video. Everything else is setup.

**The strongest thing you can say on camera:** the spectrum panel is empty until the agent
*pays* for a scan. It is not the simulator's God-view — it is what the agent bought. Then
point at the jammer appearing the instant it pays.

### The topology panel — what it draws and where the numbers come from

The top-left panel is your ns-3 network, live:

- **Nodes** at their real scenario coordinates, the agent ringed in blue with its current
  channel above it. A peer whose heartbeat gap exceeds 4 s is drawn hollow with a red rim —
  that is `dead_peer` becoming visible without anyone saying so.
- **Edges coloured per link**, not by the aggregate: green ≥ 85% delivery, amber ≥ 35%, red
  below. This is what makes `hidden_terminal` and `node_loss` legible — **one** edge
  degrades while the others stay green. A peer that is not a direct neighbour is drawn faint
  and dashed.
- **Packets in flight** as dots travelling along each edge. Blue dots complete the trip;
  **red dots stop halfway and vanish** — a dropped frame. The loss rate you see is drawn
  from that link's measured `link_pdr`, so when delivery collapses the screen fills with red
  stubs. It is a visualisation of the real telemetry, not a decoration.
- **The jammer** as a violet triangle, inert until its scenario onset time, then filled and
  ringed with expanding arcs.

All of this comes from the scenario YAML (positions, flows, jammer, onset time) plus the
per-link `pdr`/`rssi`/`hb` arrays the bridge already sends. **No ns-3 change was needed** —
which also means it cannot drift away from what the simulator is actually doing.

---

## 0a. FIRST: your ns-3 build is failing. Here is why and the fix.

```
ld64.lld: error: could not load TAPI file at .../MacOSX.sdk/usr/lib/libc++.tbd: malformed file
       error: unknown architecture   arm64e.x1-macos, arm64e.x1-maccatalyst
ld64.lld: error: undefined symbol: std::terminate()   (and ~5000 more)
```

**This is not a problem with the project code.** Nothing in `jamming-sim.cc` is involved.

ns-3 looks for a "fast linker" at configure time and uses it if it finds one:

```cmake
# ns-3.45  build-support/custom-modules/ns3-compiler-and-linker-support.cmake:112
if(${NS3_FAST_LINKERS} AND (NOT ${MSVC}))
  find_program(LLD ld.lld)
  ...
  add_link_options("-fuse-ld=lld")
```

It found `ld.lld` — your Homebrew LLVM — and linked with that instead of Apple's linker.
Your **Command Line Tools SDK is newer than that lld**: the SDK's `.tbd` stub files now
declare an `arm64e.x1` slice, `lld` does not recognise the name, so it refuses to parse
`libc++.tbd` and `libSystem.tbd`. With the C++ standard library unparsed, *every* libc++
symbol comes back undefined — hence thousands of errors that all look like missing
`std::` symbols.

### Fix: stop using lld, use Apple's linker

```bash
cd ~/Documents/NS3/ns-3-dev
./ns3 configure -- -DNS3_FAST_LINKERS=OFF
./ns3 build jamming-sim
```

`NS3_FAST_LINKERS` defaults to `ON`; this turns it off for this build tree, and the option
persists, so you only do it once. Your compiled object files are still valid — only the
link step failed — so this is a **relink, not a full rebuild**: a couple of minutes.

**While you are here, consider building optimized.** Your current tree is `debug`, which
runs the simulator several times slower than it needs to. If you have 30–40 minutes:

```bash
./ns3 configure -d optimized -- -DNS3_FAST_LINKERS=OFF
./ns3 build jamming-sim
export NS3_BIN=$(find $PWD/build -name '*jamming-sim*' -type f -perm +111 | head -1)
```

Two alternatives, if you would rather not change the ns-3 configuration:

- `brew upgrade llvm` — a newer `lld` understands `arm64e.x1`. Fixes the cause rather than
  avoiding it, but it is a large download and it may regress again on the next SDK bump.
- `brew uninstall lld` — ns-3 then cannot find it and falls back on its own. Blunt, and it
  breaks anything else on your machine that wanted lld.

The configure flag is the one to use: it is local to this build tree and changes nothing
else on your system.

---

## 0b. What is actually running — and what to say during the quiet stretches

If you clicked RUN with the teacher and the page showed only `start barrage… teacher`, that
was a real gap in the UI and it is now fixed. Here is the machinery, so you can narrate it.

### Three clocks, not one

| clock | rate | what it is |
|---|---|---|
| **ns-3 simulation time** | 10 Hz state messages | the radio world advancing |
| **decision time** | up to 1 Hz | when the agent is allowed to act |
| **wall clock** | — | how long a real LLM call takes: **30–60 s each** |

Simulation time **freezes** while the teacher thinks. ns-3 sends a state message, the bridge
blocks waiting for a decision, and the simulator does not advance until one arrives. That is
deliberate — it is what lets a slow model drive a fast simulator without distorting the
physics — but it means wall clock and sim time are unrelated. Say that on camera; it is one
of the more interesting properties of the design.

### The status strip

The bar under the controls now reports, continuously:

- **Stage** — `launching ns-3` → `ns-3 running — warm-up (OLSR converging)` → `waiting for
  first decision` → `ns-3 running, agent deciding` → `episode complete`
- **Wall clock** — real seconds since you pressed RUN
- **ns-3 state msgs** — climbs from the first second, so the simulator is *visibly* alive
- **Teacher calls** — decisions taken so far
- **Waiting on** — appears after 2.5 s of quiet and names the cause, e.g.
  *"the LLM teacher is reasoning — 34s (one real model call, typically 30–60s)"*

That last line is what makes the recording explainable. The topology, packets and delivery
plot also animate from the state feed now, so **the network keeps moving while the teacher
thinks** instead of freezing.

### Measured event order, barrage + teacher

```
PHASE building the teacher
PHASE launching ns-3
state#1   t=1.4  warmup=True   pdr=44%     <- page is alive at once
state#2   t=1.9  warmup=True   pdr=80%
state#3   t=2.4  warmup=True   pdr=92%
... 8 more warm-up pulses ...
DECISION #1  t=5.0  fading -> spectrum_scan   <- after the first model call
DECISION #2  t=6.0  fading -> load_test
...
interleaving:  S P P S S S S S S S S D S S D S S D S S D ...
               S = ns-3 state pulse      D = teacher decision
```

### Why nothing happens before t = 5 s

`WARMUP_S = 5.0` in `sim/bridge_server.py`. OLSR has to converge and the beacons have to
ramp before any delivery number means anything, so the agent is not consulted at all during
those five simulated seconds. Narrate it as "the mesh is coming up" — it is not dead time,
it is the routing protocol converging.

### What to say while a teacher call is in flight

> "The status bar says the teacher is reasoning. That is one real model call — it sees the
> evidence panel, thirty-five rows of radio statistics, and nothing else. No ground truth.
> Meanwhile ns-3 is holding simulation time still, so the physics is not distorted by how
> long the model takes. This is also why the student matters: on the drone there is no model
> and no network, and the same decision takes microseconds."

If you would rather not have the pauses on camera at all, **record the student runs for the
flowing footage and the teacher runs for the reasoning**, then cut between them. The student
makes the same class of decision in milliseconds, so its episodes play in real time.

---

## 0c. "It says running but nothing happens" — read this first

### The page shows `STAGE: requesting episode` and never moves

`requesting episode` is set by the **browser**, not the server. If it never changes, the
server never sent a single event — which means **the running server process is old code.**

Python does not reload modules. Editing `live_demo.py` or `bridge_server.py` while the
server is up changes nothing until you restart it. Stop it with Ctrl-C and start it again:

```bash
# Ctrl-C the running server first, then:
cd ~/Documents/Projects/EAG_V3/EAG_V3_final_project/capstone
python3 demo/live_demo.py --port 8099 --ns3 "$NS3_BIN"
```

**Confirm which code is live** before you record anything:

```bash
curl -s http://127.0.0.1:8099/api/version
```

```json
{"features":["phase","state","topology","per-link"],
 "live_demo_py":"d01a5fd067","demo_html":"8a019bec0a","bridge_py":"5d70326be2",
 "ns3_bin":"/.../ns3.45-jamming-sim-optimized","ns3_ok":true}
```

If `/api/version` 404s, you are on old code. If `ns3_ok` is `false`, the binary is missing
or not executable and no episode can start.

### `WAITING ON: ns-3 is starting — 179s` and no error

That was a real bug and it is fixed. The bridge used to launch ns-3 and block on
`accept()`, so a simulator that died on launch left the page waiting forever with nothing
to show. It now:

- **checks the binary first** — missing, empty or non-executable fails instantly with the path;
- **watches the process while waiting** — if ns-3 exits before connecting, you get the error
  in about half a second instead of a minute;
- **prints ns-3's own output**, which is what actually tells you what went wrong;
- **times out a stalled run** (`BRIDGE_READ_TIMEOUT_S`, default 120 s) instead of hanging.

A broken build now looks like this, immediately:

```
✗ ns-3 exited without connecting to the bridge (exit code 1). The binary started but
  never reached the bridge -- usually a broken build. ns-3 said:
  dyld: Library not loaded: @rpath/libns3.45-core-debug.dylib
```

**That message is almost certainly what you will see**, because your build is still
failing — see §0a. Fix the linker first; nothing in the demo can run until `./ns3 build
jamming-sim` succeeds.

### The order to do things in

1. `./ns3 configure -- -DNS3_FAST_LINKERS=OFF && ./ns3 build jamming-sim` → must succeed
2. `export NS3_BIN=$(find $PWD/build -name '*jamming-sim*' -type f -perm +111 | head -1)`
3. `echo $NS3_BIN` → must print a path, and `$NS3_BIN --PrintHelp` must print options
4. restart the demo server
5. `curl -s http://127.0.0.1:8099/api/version` → `ns3_ok: true`
6. run a **student** episode first — it needs no model calls and finishes in ~2 minutes
7. only then try the teacher

---

## 0d. How to read the screen — panel by panel

The page now writes the story itself. **Episode story**, top-right, is the panel to point
at; everything else is supporting evidence.

### `spot_single_channel` + student — the clean win, measured

```
t=  5s ●  mesh is up — delivery 100% on channel 6
t=  6s ?  buys evidence — spectrum_scan          (costs budget)
t= 18s ▲  ATTACK BEGINS — spot jammer, channel 6 only
t= 19s ◆  agent concludes spot (100%)
t= 19s ✦  ACTS: hop_channel {channel: 1}
t= 19s ▼  delivery collapses to 33%
t= 20s ✦  ACTS: change_tdma_slot {slot: 3}
t= 20s ▲  LINK RECOVERED — 84% on channel 1

VERDICT  RECOVERED in 1s
  true cause spot · agent said spot · CORRECT
```

**Lead the video with this one.** Attack at 18 s, diagnosis and hop at 19 s, link back at
20 s. One second of outage, no human in the loop.

### Your barrage screenshot, explained

```
t=  5s ✦ ACTS: set_tx_power {dbm: 20}
t=  5s ● mesh is up — delivery 100% on channel 6
t=  6s ? buys evidence — spectrum_scan          (costs budget)
t= 18s ▲ ATTACK BEGINS — barrage jammer, all channels
t= 19s ◆ agent concludes barrage (100%)
t= 19s ▼ delivery collapses to 18%

VERDICT  LINK NOT RECOVERED
  true cause barrage · agent said barrage · CORRECT
```

**This is a success, not a failure, and that is the interesting part.** The agent named the
cause correctly within one second of the collapse and then deliberately did *not* hop. For
barrage every channel is jammed — the spectrum panel shows all eight bars lifted to about
−77 dBm — so hopping spends a hop from a finite budget and fixes nothing. Restraint is the
right answer. The remaining options are a different waveform or declaring the link lost.

Say exactly that on camera. "The link stayed down" sounds like a defeat until you explain
that the alternative was to thrash between eight jammed channels.

### What each panel is for

| panel | what it shows | what to say |
|---|---|---|
| **Status strip** | stage, wall clock, ns-3 messages, decisions | "the simulator is live — these counters are climbing" |
| **Mesh topology** | nodes at real coordinates, edges coloured per link, packets in flight, jammer | "one edge red, the rest green, means a local problem, not jamming" |
| **Link health** | delivery over time; red dashed = attack onset, amber = agent acted | "the gap between the red line and the amber line is detection latency" |
| **Spectrum** | the scan the agent **paid for** | "empty until it buys a scan — this is bought evidence, not the God-view" |
| **What the agent believes** | probability over the 8 causes | "it commits, and you can watch it commit" |
| **Episode story** | the plot, in order | ← **point here** |
| **Decisions / Tick table** | the raw record | "every one of these is in the trace the verifier scores" |

### The three symbols that matter

- **▲ red** — the attack starts (from the scenario, so it is ground truth)
- **▼** — delivery collapses (what the agent can actually see)
- **◆** — the agent commits to a diagnosis

The distance from ▲ to ◆ is the story of the whole project.

### Beliefs before the attack are not shown, on purpose

Until the anomaly gate fires there is nothing to diagnose, and the belief is just the prior
drifting. Narrating that as a "conclusion" made the agent look like it was guessing at a
healthy link. The story panel stays quiet until the attack begins or delivery drops.

### If the story ever describes the wrong scenario

It cannot any more, and the cause is worth knowing because it looked like an agent bug and
was not.

The event stream was one shared queue read by every `/api/stream` connection — and a queue
**distributes**: each event goes to exactly ONE consumer. With two browser tabs open, or one
stale connection that had not timed out, the events were **split between them**. A page
could receive `phase` and every tick but miss `start`, so the story never reset and narrated
the new episode on top of the old one. That is why a `spot` run showed "barrage jammer, all
channels" and a verdict reading "true cause barrage".

Fixed two ways:

- **the stream now broadcasts** — one queue per client, fanned out. Verified with two
  simultaneous clients receiving byte-identical 34,965-byte streams;
- **the scenario's truth travels with the episode.** `start` now carries
  `{family, onset_t, jammer_type, jammer_channels}` read from the YAML at launch, so the
  story describes the episode that is running rather than whatever the dropdown last
  fetched. The verdict's "true cause" comes from the same place.

Story lines are also sorted by simulation time now, so a late-detected event can never
appear above an earlier one.

**You can safely keep a second tab open** — both will show the same run.

### "It still says running when the episode is over"

Fixed. Three causes, all handled:

- the chip now reads **`✓ complete`** and the wall clock stops on `done`;
- if you **reload the page mid-episode** it joins in progress ("running (joined
  mid-episode)") instead of leaving the clock stuck at 0s — that is what your screenshot
  showed;
- if the event stream goes silent for 45 s the chip flips to **`stream stalled — restart
  the run`** rather than claiming to be running forever.

---

## 0e. The teacher fails instantly with `FileExistsError`

```
✗ FileExistsError: [Errno 17] File exists: '.../capstone/../data/traces/demo'
```

An early version of `live_demo.py` passed a **directory** path to `TraceStore`, which opens
its argument for append — so it created a zero-byte **file** called `demo` where the
directory `demo/` belonged. Every teacher run after that died on it. `exist_ok=True` does
not help: it forgives an existing *directory*, not a *file* in the way.

**One-time cleanup, if you still have the stray file:**

```bash
cd ~/Documents/Projects/EAG_V3/EAG_V3_final_project
ls -la data/traces/demo            # if this is a 0-byte FILE, not a directory:
mv data/traces/demo data/traces/_stray.bak
mkdir -p data/traces/demo
```

The code now does this for itself — `_ensure_dir()` renames anything that is not a
directory to `<name>.not-a-directory.bak` and carries on — so it cannot recur.

---

## 0f. "The student gets most scenarios wrong"

It probably is not the student you think. The demo used to hard-code **student_v10**, whose
`abstain_threshold` is **0.0** — it never abstains, so it answers confidently on every
window including the ones it has no business answering.

**student_v11** is calibrated on live-bridge rows and scores **87.5 %** on seen families
against v10's 77 %:

| | v10 | v11 |
|---|---|---|
| abstain threshold | **0.0** (never abstains) | **0.51** |
| live-bridge accuracy, seen families | 77 % | **87.5 %** |
| barrage / congestion / fading / node_loss | mixed | **100 %** |

Extract it once:

```bash
cd ~/Documents/Projects/EAG_V3/EAG_V3_final_project
mkdir -p data/student_v11 && tar xzf data/student_v11.tgz -C data/student_v11
```

The server now picks the newest student present automatically (v11 → v10 → v9), so after
extracting it and **restarting**, you are on v11. Confirm:

```bash
curl -s http://127.0.0.1:8099/api/version
```

```json
{"student_bundle":".../data/student_v11/student_bundle.json","student_ok":true,"ns3_ok":true}
```

To pin a specific one: `STUDENT_BUNDLE=../data/student_v10/student_bundle.json python3 demo/live_demo.py ...`

### What is still genuinely wrong, so you are not surprised on camera

Even on v11, `sweep` scores **0 %** — it is held out of training entirely and is not in the
train or val split at all. And `hidden_term`, `reactive` and `spot` sit at 67 %. Those are
real limits, explained in `docs/ARCHITECTURE_GAPS.md`: three families are blind because a
telemetry field their discriminator needs is a **constant** in one or both data paths.
Show a failure and explain the cause — it is a stronger submission than five cherry-picked
wins.

---

## 0g. "stream stalled" mid-run, and how to show logs on camera

### The false stall — fixed

If the page flipped to **`stream stalled — restart the run`** while the teacher was thinking,
that was my watchdog, not a real failure. It declared the stream dead after 45 s of silence,
and a single teacher call legitimately takes **30–90 s**. The SSE keepalive was an SSE
*comment*, which does not fire the browser's `onmessage`, so a healthy long call looked
identical to a dead connection.

Fixed two ways: the server now emits a real `ping` **event** every 15 s, and the stall bar is
150 s. A long teacher call no longer trips it, and a genuinely dead stream is still caught.

**How to tell a slow run from a broken one:** the status strip. `WAITING ON` names the cause
and counts up — *"the LLM teacher is reasoning — 34s"* — while `NS-3 STATE MSGS` holds steady
and `TEACHER CALLS` ticks up when the answer lands. All three frozen at 0 is broken; only the
clock moving is normal.

---

### Showing ns-3 and background logs in the video

**Option 1 — the built-in panel (recommended).** There is now an **"ns-3 simulator log — live
stdout"** panel below the topology. It streams the simulator's own output as it runs, tails
automatically, and needs no second window. Measured on a real episode:

```
[sim] congestion_burst nodes=6 jammers=0 truth=congestion tdmaSlots=4
[sim] agent declared link lost at t=35
[sim] done rxOk=62815 rxErr=78946 analyzerClean=116553 analyzerDirty=3446
[diag] node N1 beaconsSent=595 dataSent=1176 dataRx=0 peersHeard=3
[diag] node N5 beaconsSent=594 dataSent=0   dataRx=1098 peersHeard=4
```

Good things to point at: `truth=congestion` is printed **by the simulator**, so it proves the
scenario is what you claim — and the agent never sees that line. `rxOk` vs `rxErr` is the raw
frame count behind the delivery number. The `[diag]` block shows which node actually carried
the traffic.

This also fixed a latent hazard: ns-3's stdout was a pipe nobody read, so a chatty run could
fill the OS buffer and block the simulator with no error anywhere. It is drained on a thread now.

**Option 2 — a terminal pane beside the browser**, if you want the classic two-window shot:

```bash
# pane 1 — the demo server, which prints every request
python3 demo/live_demo.py --port 8099 --ns3 "$NS3_BIN"

# pane 2 — the teacher's decisions as they are written
tail -f data/traces/demo/*.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    try: r=json.loads(l)
    except: continue
    b=r.get('belief') or {}
    top=max(b,key=b.get) if b else '-'
    print(f\"{top:<12} {r.get('call',''):<16} {str(r.get('why',''))[:70]}\")"
```

**Option 3 — ns-3's own verbose logging**, if you want protocol-level detail:

```bash
NS_LOG="JammingSim=level_all|prefix_time" $NS3_BIN --config=... --sock=...
```

Be careful with this one on camera: `level_all` produces thousands of lines per second and
will bury everything else. `level_info` is usually the readable choice.

**For recording:** use option 1 as the main shot. The panel is already in frame, already
styled to match, and scrolls itself. Option 2 is worth ten seconds at the start to establish
that a real process is running, then cut back to the browser.

---

## 0h. "The teacher isn't calling the LLM" and "the verdict is always wrong"

Both were real, both were mine, and neither was the agent.

### The teacher WAS working — those calls came from cache

30 teacher calls in a **6-second** wall clock is impossible for live calls (30–90 s each).
Those were **cache hits**: the provider keys on SHA-256 of (model, system prompt, prompt), so
an identical episode replays the previous run's answers instantly. The answers are real
model output from an earlier run — but nothing was re-reasoned.

The status strip now has a **Model vs cache** counter reading `N live / M cached`, and it
turns amber when every call was a replay. Check it before you record: if it says `0 live`,
you are filming a replay.

To force fresh reasoning:

```bash
rm -rf ~/.cache/jamming-llm      # or wherever LLM_CACHE points
LLM_CACHE=/tmp/fresh-$(date +%s) python3 demo/live_demo.py --port 8099 --ns3 "$NS3_BIN"
```

Cached is *better* for recording — identical every take, no per-take cost — as long as you
know that is what you are showing.

### The verdict said "agent said —" while the belief panel was confident

`congestion_burst` has **no jammer** (`jammers: []`) and its event is `load_spike`, not
`jammer_on`. I only looked for `jammer_on`, so onset was `None`, the story never "engaged",
the conclusion was never recorded, and the verdict printed `agent said — · WRONG`.

Meanwhile the agent had said **congestion**, which is **correct**.

Three fixes:

1. **Onset now comes from `truth.onset_t`**, which every family sets — instead of an event
   type that only the jamming families use. `node_loss` (`node_down`) and `fading` (no event
   at all) were silently broken the same way.
2. **The conclusion is tracked on every tick**, and only the *narration* waits for
   engagement. A confident diagnosis can no longer be lost.
3. **Partial degradation is now a first-class outcome.** Congestion sags 100% → 53% and never
   trips a collapse threshold, so it scored "NO FAULT". There is now a `DEGRADED` verdict.

Verified on `congestion_burst` + student:

```
truth: family=congestion  onset_t=18.0  event_type=load_spike  has_jammer=False
baseline 100%   worst 53%   degraded at t=16.9s (73%)
agent concluded: congestion      true cause: congestion     -> CORRECT
```

The story now reads *"traffic load spikes — no attacker"* rather than inventing a jammer.

**Worth saying on camera:** congestion is the hardest verdict in the set precisely because
the link never dies. An aggressive remedy costs more than it saves; doing nothing leaves
throughput on the floor. That is a genuinely harder call than "hop off the jammed channel".

---

## 0i. ns-3 was CRASHING on every channel hop — fixed, rebuild required

Your last screenshot showed `13 live / 0 cached`, so the LLM was genuinely reasoning. The
verdict was still wrong, and the ns-3 log panel says why:

```
[sim] congestion_burst nodes=6 jammers=0 truth=congestion tdmaSlots=4
aborted. cond="!htCapabilities",
  msg="HT Capabilities element not received for 00:00:00:00:00:05",
  +21.002134784s  src/wifi/model/mpdu-aggregator.cc, line=180
NS_FATAL, terminating
libc++abi: terminating
```

**The simulator died at t = 21.0 s — the exact second the agent called
`hop_channel {channel: 2}`.** SIM TIME froze at 21 s on a 60 s episode, and the verdict was
scored on the truncated run.

### Why

The sim runs `WIFI_STANDARD_80211n`. In 802.11n ad-hoc mode a station learns each peer's HT
Capabilities from frames it receives. After the agent retunes the PHY, that record no longer
applies — and the first attempt to A-MPDU-aggregate to such a peer hits

```cpp
NS_ABORT_MSG_IF(!htCapabilities, "HT Capabilities element not received for " << recipient);
```

which kills the **whole process**, mid-episode, with no result.

`hop_channel` is the agent's single most important remedy, so **this made the most
interesting scenarios the least reliable** — and it would fire unpredictably, whenever a hop
happened to be followed by an aggregation attempt. That is a large part of the flakiness you
have been chasing.

### The fix

A-MPDU aggregation is now disabled in `sim_ns3/jamming-sim.cc`:

```cpp
mac.SetType("ns3::AdhocWifiMac",
            "BE_MaxAmpduSize", UintegerValue(0),
            "BK_MaxAmpduSize", UintegerValue(0),
            "VI_MaxAmpduSize", UintegerValue(0),
            "VO_MaxAmpduSize", UintegerValue(0));
```

Aggregation buys throughput this experiment does not measure — every statistic here is
per-frame delivery, timing and spectrum — so taking the aggregator out of the path costs
nothing and removes the abort entirely.

**Verified:** `congestion_burst` with the student now runs the full 30 decision ticks and
reaches `[sim] done rxOk=64815 rxErr=82966`, where it previously aborted at 21 s.

### You must rebuild

```bash
cp ~/Documents/Projects/EAG_V3/EAG_V3_final_project/sim_ns3/jamming-sim.cc \
   ~/Documents/NS3/ns-3-dev/scratch/jamming/
cd ~/Documents/NS3/ns-3-dev && ./ns3 build jamming-sim
```

### And the UI will no longer hide it

Any `NS_FATAL` / `aborted. cond=` / `terminating` line in the simulator log now raises

> **THE SIMULATOR ABORTED — THIS EPISODE IS NOT A RESULT**

with the ns-3 message verbatim, and it **suppresses the normal verdict**. A truncated run is
never scored as a diagnosis again.

**If you had already recorded takes, re-shoot them.** Any episode where the agent hopped may
have been silently truncated.

---

## 1. One-time setup (~5 minutes)

```bash
cd ~/Documents/NS3/ns-3-dev
cp ~/Documents/Projects/EAG_V3/EAG_V3_final_project/sim_ns3/jamming-sim.cc scratch/jamming/
./ns3 build jamming-sim                     # REQUIRED: your build predates the median fix
export NS3_BIN=$(find $PWD/build -name '*jamming-sim*' -type f -perm +111 | head -1)
echo $NS3_BIN                               # must print a path

cd ~/Documents/Projects/EAG_V3/EAG_V3_final_project/capstone
python3 -c "import numpy,yaml;print('deps ok')"
claude --version                            # only needed for the teacher runs
```

Start the demo server (leave this terminal running):

```bash
python3 demo/live_demo.py --port 8099 --ns3 "$NS3_BIN"
```

Open **http://127.0.0.1:8099**. If the chip top-right says *"ns-3 binary not found"*, your
`NS3_BIN` is wrong — fix it before going further.

### Dry-run everything BEFORE you record

```bash
# each takes 1-3 min; the teacher ones cost real tokens
for s in spot_single_channel barrage_all_channels sweeping_jammer dead_peer fading_no_attacker; do
  echo "=== $s ==="
  curl -s "http://127.0.0.1:8099/api/run?scenario=$s.yaml&agent=student"
  sleep 150
done
```

**Do not skip this.** Some scenarios diagnose correctly and some do not, and you want to
know which before the camera is on — see §4.

---

## 2. The five scenarios, and what to say

Run each one twice: `agent=teacher` first, then `agent=student`.

### 1. `spot_single_channel.yaml` — the clean win. **Lead with this one.**
A jammer parks on one channel. The agent must notice, scan, identify *spot* (not barrage),
and hop to a quiet channel.
> "Delivery is fine. Now the jammer starts. The agent doesn't know what hit it — it buys a
> spectrum scan, sees one channel 27 dB hot and the rest clean, concludes *spot jamming*,
> and hops. Delivery comes back. No human in the loop."

### 2. `barrage_all_channels.yaml` — knowing when NOT to hop
Every channel is jammed. The naive move is to hop; it achieves nothing.
> "Same symptom, different cause. The scan shows *every* channel hot, so hopping is
> pointless. The agent has to reach a different conclusion from the same starting symptom —
> that is the whole reason this needs a reasoning layer rather than a threshold."

### 3. `sweeping_jammer.yaml` — a moving target
The hot channel keeps moving. Watch the spectrum panel walk across the bars.
> "The jammer is sweeping. Watch the hot bar move. A one-shot hop just lands in front of it
> again."

### 4. `dead_peer.yaml` — not an attack at all
A node dies. No RF anomaly. The correct answer is to reroute, not to hop.
> "No jamming here — a neighbour went silent. If the agent hops channel it has wasted a hop
> and fixed nothing. This is the false-positive case."

### 5. `fading_no_attacker.yaml` — restraint
Multipath fading. Nobody is attacking. The correct action is to do *nothing* costly.
> "This is the one that catches naive detectors. Everything looks like interference and
> there is no attacker. Acting here is a false positive, and in a real deployment a hop
> costs you the link for a second."

**Teacher vs student, say this once:** the teacher is Claude reasoning live over the
evidence panel, one call per decision. The student is a 16 KB int8 network plus ~30 lines
of induced rules, distilled from the teacher's decisions, running with **no model and no
network connection** — which is what actually flies on the drone.

---

## 3. Recording layout

- **Browser at 1280×800**, zoom 100%. The page is dark-mode; it films well.
- The topology panel is top-left, so a 16:9 crop of the **left column alone** (topology +
  delivery plot + spectrum) is a clean shot if you want the agent's reasoning off-screen for
  the opening sequence.
- Screen-record the **browser only**, not the whole desktop.
- Keep the terminal running the server in a second window and show it once at the start, so
  it is obvious ns-3 is genuinely executing.
- Optional split-screen: terminal on the left tailing the run, browser on the right.

```bash
# nice-looking live tail for the left pane
tail -f ~/Documents/Projects/EAG_V3/EAG_V3_final_project/data/traces/demo/*.jsonl \
  | python3 -c "import sys,json
for l in sys.stdin:
    try: r=json.loads(l)
    except: continue
    b=r.get('belief') or {}
    top=max(b,key=b.get) if b else '-'
    print(f\"t={r.get('t','?'):>5}  {top:<12} {r.get('call','')}\")"
```

macOS: **Cmd-Shift-5** records a selected region. QuickTime → File → New Screen Recording
also works and is easier to trim.

**Timing — the two agents are NOT the same speed.** A student episode is ~2–3 minutes.
A **teacher episode is 10–15 minutes**, because every decision is a real model call at
~40 s. Five teacher runs is over an hour of wall clock. Start them early, record them
unattended, and trim later. Do not plan to run a teacher episode live in front of anyone.

**Never lower `HARNESS_MAX_CALLS`.** I tested with it set to 6 and the teacher froze on its
first guess (`fading`) and sat on `no_op` while delivery fell to 0% — it looks exactly like
a broken agent. The default is 30; leave it alone.

**Each episode is 1–3 minutes of wall clock for the student.** Ten of them is ~20 minutes of
footage. Record them all, then cut to the 20–30 seconds around each detection. Do not try
to narrate live — record the runs silently, then voice over.

---

## 4. Be honest about what fails

From the measured 27-episode run, the teacher is reliable on `barrage`, `spot` and
`refusal`, mixed on `sweep` and `node_loss`, and currently **wrong** on `fading`,
`congestion`, `hidden_term` and `reactive`.

If a scenario misdiagnoses on camera, **keep it and say so.** A capstone that shows a
0.407 accuracy and explains exactly which confusions remain is a stronger submission than
one that shows five cherry-picked wins. You have the failure analysis to back it:
`hidden_term→fading` and `congestion→node_loss` are the two dominant confusions, and
`docs/TEACHER_AB_RESULT.md` records an A/B that honestly failed to fix them.

---

## 5. If something breaks

| symptom | cause | fix |
|---|---|---|
| chip says "ns-3 binary not found" | `NS3_BIN` unset or wrong | re-run the `find` in §1 |
| spectrum panel stays empty | the agent has not bought a scan yet | correct behaviour — wait, or narrate it |
| "a run is already in progress" | previous episode still going | wait for it, or restart the server |
| teacher run stalls | `claude` CLI not on PATH | `claude --version`; use `agent=student` instead |
| page blank | server not started | check the terminal from §1 |
| delivery never drops | attack onset is after warm-up (5 s) | normal — wait to ~19 s |

---

## Appendix — ns-3's own animator (NetAnim), and why I would not use it here

You asked about ns-3's UI animation. It exists, but be clear about what it is:

**NetAnim is an offline replayer.** `AnimationInterface` writes an XML trace during the
run; you then open that XML in a **separate Qt desktop application** and scrub through it.
It cannot show a live run, and it shows packets and node colours — it has no way to show
the agent's belief, the spectrum it bought, or why it acted. For a video about an *agent's
reasoning*, it shows the least interesting half.

It is also not currently available in your build: the `netanim` module is present in the
ns-3 source tree but was not compiled (`build/lib` has no anim library), and the NetAnim
GUI is a separate program you would have to build against Qt5.

If your instructor specifically wants NetAnim, this is the path:

```bash
# 1. enable the module and rebuild (slow, ~20-40 min)
cd ~/Documents/NS3/ns-3-dev
./ns3 configure --enable-modules=netanim,wifi,spectrum,olsr,internet,applications,mobility
./ns3 build

# 2. build the GUI (needs Qt5: brew install qt@5)
git clone https://gitlab.com/nsnam/netanim.git && cd netanim
qmake NetAnim.pro && make
```

Then in `jamming-sim.cc`, before `Simulator::Run()`:

```cpp
#include "ns3/netanim-module.h"
...
AnimationInterface anim ("jamming.xml");
anim.SetMaxPktsPerTraceFile (500000);
// colour a node red when its delivery collapses, green when it recovers:
//   anim.UpdateNodeColor (nodeId, 255, 0, 0);
//   anim.UpdateNodeDescription (nodeId, "JAMMED");
```

and open `jamming.xml` in the NetAnim app afterwards.

**My recommendation:** use the live dashboard as your main footage, and if you want ns-3's
own visuals as a 10-second establishing shot, record NetAnim separately and cut it in. Do
not make it the backbone of the video — you will spend a day on the Qt build and end up
with a less convincing result than the page you already have working.
