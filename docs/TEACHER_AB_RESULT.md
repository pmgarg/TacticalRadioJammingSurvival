# Did the teacher get better? No. Here is the evidence.

Two full live ns-3 + LLM runs over the **identical 27 seeds** (9 families × 3, seed0=9200).
Only one thing differed: the evidence panel the teacher sees.

| | 29-row panel | 35-row panel |
|---|---|---|
| episodes completed | 27/27 | 27/27 |
| **accuracy** | **0.407** | **0.407** |
| correct | 11/27 | 11/27 |
| model calls | 314 | 316 |
| parse failures | 0 | 0 |
| provider errors | 0 | 0 |

Four episodes flipped — two each way (`reactive_9202` and `sweep_9201` gained,
`spot_9202` and `sweep_9200` lost). **That is noise, not a result.**

---

## What was changed and why

`train/analyse_failures.py` ran on the 27-episode traces. Top confusions:
`hidden_term→fading` (18), `congestion→node_loss` (20), `congestion→fading` (17).

It then found six features that separate exactly those pairs on the 13,700-window
labelled corpus, and which the teacher was **not** being shown:

| feature | separability | splits |
|---|---|---|
| `pdr_cusum` | 1.00 | hidden_term vs fading (step vs drift) |
| `pdr_worst` | 0.99 | one bad link vs all links sagging |
| `queue_occ` | — | self-inflicted congestion vs interference |
| `cca_busy` | — | busy+decodable vs busy+undecodable |
| `scan_cur_rank` | 0.93 | interferer on us vs everywhere |
| `tx_success_ratio` | 0.98 | our fault vs theirs |

All six were added to `agent/llm_teacher.py` PANEL with physical descriptions.
**It made no measurable difference.**

### The A/B was clean — verified, not assumed

All **493/493** decisions in the second run carried the six new rows. The 177 cache-served
decisions were within-run duplicates (316 billed + 177 cached = 493); no prior run had ever
produced a 35-row prompt, so nothing stale could have been served. The negative result is real.

---

## Retraction

I previously reported that the event gate **starved hidden_term to zero teacher calls**.
**That was wrong.** `calls=0` in the run log counts *billed* calls; those decisions were
served from the response cache at 0.0s latency. Measured properly:

| family | decisions/episode | of which cached |
|---|---|---|
| hidden_term | 16.3 | 41 of 49 |
| congestion | 17.0 | 41 of 51 |
| fading | 26.3 | 0 |

hidden_term is **not** starved. It is asked ~16 times per episode and answers `fading`
almost every time. This is a reasoning failure, not a plumbing failure, and my earlier
diagnosis sent the fix in the wrong direction.

---

## The hypothesis that now matters

The separability figures above are computed on **corpus windows, where spectrum data is
free**. In a live episode, information must be *bought*: `scan_*` features are stale or
absent until the agent pays for a `spectrum_scan`, and `scan_age` frequently reads `+1.00`
(never scanned). So a feature that cleanly splits two classes on paper can be unavailable
at the moment the teacher must decide.

That would explain a real 1.00-AUC feature buying zero accuracy — and it is testable:
re-run the separability analysis **restricted to windows where `scan_age` is fresh**, and
compare the ranking. If the top features change, the failure loop has been optimising
against evidence the live agent never has.

Also measured: **36% of decision ticks produce a prompt byte-identical to an earlier tick
in the same run.** A third of the teacher's ticks carry no new information at panel
resolution — which caps how much any panel change can achieve.

---

## Not done

- **540-episode run with the new panel.** Deliberately not run: the 27-seed A/B gives no
  evidence it would help, and it would be a large spend on a null hypothesis.
- **Student re-induction and retraining.** Depends on a teacher corpus that is actually
  better. Retraining from a teacher that did not improve reproduces student_v10.

The six panel rows are left in place — they are physically justified and cost ~60 tokens
per call — but they are **unproven**, and that is how they should be described.
