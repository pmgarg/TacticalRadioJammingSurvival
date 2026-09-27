# Why congestion_burst failed, and the capability that was missing

Two separate findings from your screenshot. The first explains that run entirely. The second
is the one you actually asked about, and you were right to expect it — it was not there.

---

## 1. Your teacher run made zero model calls

The belief bars read **13 % for all eight causes**. That is 1/8 — the uniform prior. It is
what the agent returns when it has decided nothing. "barrage" in the tick table is not a
diagnosis; it is `argmax` breaking an eight-way tie by picking the first key.

The trace says why:

```
records   : 23
parse_mode: {'failed': 23}
call      : {'no_op': 23}
error     : 4 attempts failed; last: claude exited 1:
            {"result":"Not logged in · Please ..."}
why       : provider error; abstaining
```

**The Claude CLI on your Mac is not authenticated.** All 23 calls failed, the harness
correctly abstained on every one, and the verdict was scored against an agent that never
thought. Nothing about congestion was actually tested.

```bash
claude login          # in your own Terminal
# then restart:  python3 demo/live_demo.py --port 8099 --ns3 "$NS3_BIN"
```

**This should never have been subtle.** Two changes so it cannot be again:

- a **preflight** probes the model before the episode starts and refuses in ~2 s:
  `"the teacher cannot reach the model, so it would abstain on every tick and produce a
  meaningless run"`;
- any provider error during a run raises a red **THE TEACHER IS NOT REASONING** banner
  saying every belief shown is the 1/8 prior.

The student run is unaffected — it needs no model — so its congestion result stands and is a
genuine miss.

---

## 2. The teacher had no memory of its own decisions. Now it does.

> *"we also have teacher capability that it will learn from wrong decision and try to correct
> in his next decision"*

It did not. The renderers in `harness/loop.py` and `agent/llm_teacher.py` have always been
able to print `tests_run` and `actions_taken` — but the live bridge built its `Context` like
this:

```python
ctx = Context(t=t, channel=..., peers=..., budget=..., available=avail,
              last_scan=last_scan_obj, lora_available=sc.lora_available)
#  tests_run / actions_taken / hypothesis_history / recovery_attempts  -> never set
```

They defaulted to empty, so **every prompt told the teacher it had done nothing**. Thirty
calls per episode, each one a fresh look at the current panel, with no idea that it had
already scanned, already hopped, or that the hop had failed. It could not correct itself
because nothing told it there was anything to correct.

### What it now receives

```
WHAT YOU ALREADY TRIED, AND WHAT HAPPENED
  t=5s   set_tx_power      (you believed barrage)  ->  no change (+0 pts delivery)
  t=21s  hop_channel       (you believed spot)     ->  helped (+91 pts delivery)
  t=28s  change_tdma_slot  (you believed spot)     ->  made it worse (-28 pts delivery)
  A remedy that did not help is evidence AGAINST the cause that motivated it.
  Do not repeat it; revise the diagnosis instead.
  your hypothesis so far: barrage -> reactive -> node_loss -> spot
```

**This is causal evidence, not correlational** — and it is the strongest kind available.
"I hopped and delivery rose 91 points" confirms spot jamming more decisively than any single
statistic in the panel. "I raised TX power believing barrage and nothing changed" is direct
evidence against barrage.

Every remedy is scored 2 s after it is applied against the delivery at the moment it was
taken, and classified `helped` / `no change` / `made it worse`.

**Verified live** on `spot_single_channel`: **28 of 30 ticks** now carry outcome history.
Before this change all 30 were empty.

### What I have not measured

Whether this raises accuracy. That needs a full teacher run, and your CLI cannot make calls
yet. The mechanism is verified; the benefit is not. I am not going to claim a number I have
not measured — run it after `claude login` and compare against the 40.7 % on record.

---

## Why congestion is hard even with this

Congestion's two designated discriminators, `offered_load` and `loss_load_corr`, are
**constant in the training corpus** — ns-3 computes `load` and puts it in the socket message
but the CSV writer never emits it, so `percept/features.py` fills the hole with a default of
0.4 on every training row. The student was never shown what congestion looks like.

The new outcome feedback helps the **teacher** reason around that at run time, but it does
not give the **student** the evidence it was never trained on. That still needs fix #2 in
`docs/GAPS_EXPLAINED_AND_FIXED.md` — emit `load` and `retry` from the CSV writer and
regenerate the corpus.

Your OLSRv2 instinct is the deeper version of the same fix: a neighbour's queue occupancy
and CCA-busy fraction distinguish *our* load from *theirs*, which is exactly the congestion
question.
