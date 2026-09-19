# student_v10 — the first student actually distilled from the LLM teacher

## Why the old 85% and the new numbers are not the same measurement

The single most confusing thing in this project's history. Same student, same simulator:

| measurement | score |
|---|---|
| student, **per-window**, ns-3 CSV corpus (spectrum handed over free) | **82.4%** |
| student, **per-window**, LIVE ns-3 bridge (spectrum must be BOUGHT) | **33.8%** |
| student, **per-episode**, live bridge | **60.9%** |
| LLM teacher, per-episode, live bridge | **28.8%** |

Nothing regressed between them; they are four different questions. The 82% -> 34% collapse
is the same bundle on the same simulator, and it is caused entirely by the CSV corpus
reporting a fresh spectrum scan on 100% of ticks while the live bridge requires the agent
to spend budget on `spectrum_scan` first. Per-episode beats per-window because the early
ticks of an episode are wrong *by design* -- the agent is still gathering evidence, and
what is scored is the claim it commits to.

**Any accuracy figure in this project is meaningless without three qualifiers: which agent,
per-window or per-episode, and which world.**

## The headline result

student_v10 is the first bundle whose action policy comes from the LLM teacher's own
decisions: 5,545 teacher rows (train split only), weighted x3, 23% of the training mix.

| | v9 (oracle-distilled) | **v10 (LLM-distilled)** |
|---|---|---|
| per-window, ns-3 val | 88.4% | **93.0%** |
| per-window, in-distribution test | 85.0% | **91.9%** |
| call head | 92.5% | 93.6% |
| FP fading->jamming | 0.012 | **0.0000** |
| **per-episode, LIVE ns-3 bridge, test split** | **54%** (14/26) | **69%** (18/26) |
| int8 size | 16.0 KB | 16.3 KB |

Per family on the live bridge (test split, 3 episodes each):

| family | v9 | v10 |
|---|---|---|
| barrage | 3/3 | 3/3 |
| spot | 2/3 | 2/3 |
| **reactive** | **0/3** | **2/3** |
| fading | 3/3 | 3/3 |
| node_loss | 2/3 | 3/3 |
| **congestion** | **0/3** | **3/3** |
| hidden_term | 1/2 | 1/2 |
| refusal | 3/3 | **1/3** (regression) |
| sweep (held out) | 0/3 | 0/3 |

The two families the teacher helped most, reactive and congestion, are exactly the two the
oracle-distilled student could not do at all. The refusal regression is real and is the
open item.

Note the student scores 69% per-episode while its teacher scores 28.8%. That is not a
paradox: per DESIGN 9.5 the CAUSE label comes from the simulator (free and unlimited), and
what the teacher contributes is the ACTION and TEST-SELECTION policy -- which test to buy,
and when to stop investigating. The student inherits the teacher's procedure without
inheriting its diagnostic errors.

## Two contaminations found while building this

**1. I repeated the v7 sin.** The first v10 was trained with
`llm_traces_to_rows --split any`, which put 2,263 test-split rows -- including all 1,226
`sweep` rows -- into training. It scored **85.7% on the held-out sweep family**, which
looked like a spectacular generalisation result and was leakage. Retrained with
`--split train`: sweep back to 0.0%, and the in-distribution gain (85.0% -> 91.9%) survives
because it was never the contaminated part.

**2. The old sweep rule was induced on sweep data.** `harness/induce.py` built its table by
calling `make(fam, seed)` for every family in `FAMILIES`, sweep included, at seeds outside
the corpus. So the rule layer was shown the family the net is denied, and the headline
"hybrid recovers the held-out family, 0% -> 47%" rested on that. Induced honestly from
train-split ns-3 rows only (`--from-rows`), no sweep rule survives -- it is dropped at
`support=0`. The four rules that do survive are barrage 1.00, spot 1.00, node_loss 1.00 and
fading 0.82 held-out precision.

`data/policy_ns3.json` is the honest rule set. It is smaller and it does not claim sweep.

## Reproduce

```bash
python3 train/llm_traces_to_rows.py --traces ../data/traces/llm_full --split train \
    --out ../data/traces_llm_full/train.jsonl          # 5,545 rows, 0 sweep
python3 harness/induce.py --from-rows ../data/traces_ns3/train.jsonl \
    --holdout-rows ../data/traces_ns3/val.jsonl --rows 80 --out ../data/policy_ns3.json
python3 train/train_mixed.py --out ../data/student_v10 \
    --llm ../data/traces_llm_full/train.jsonl --llm-weight 3
python3 train/calibrate_abstain.py --bundle ../data/student_v10/student_bundle.json \
    --calib ../data/traces_ns3/val.jsonl --ood ../data/traces_ns3/test.jsonl --write
python3 train/export_int8.py --bundle ../data/student_v10/student_bundle.json \
    --traces ../data/traces_ns3/val.jsonl --out ../data/student_v10
```
