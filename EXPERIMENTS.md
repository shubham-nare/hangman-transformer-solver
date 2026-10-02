# Experiment log

> Written during the competition and last updated on 2 September 2026, while
> run 5 was still training. It covers the Transformer experiments. The later
> work — run 6, the Kneser-Ney beam posterior and the final 77.58% — is
> summarised in `README.md`, with the raw sweeps in `artifacts/*.log`.

Controlled comparisons only. Every row is measured on the same held-out split of
`train.txt` with the real 6-life engine, so numbers are comparable across rows.

Reference points, all at d256/4L, batch 1024, 20k steps, seed 20260901:

| run | change | held-out win rate | note |
|---|---|---|---|
| baseline: static frequency | — | 12.70% | = `sample_submission.csv` |
| baseline: pattern matching | — | 15.65% | vocabulary lookup; train/test disjoint |
| **run 1** | presence head only | **51.30%** | ablation control |
| **run 2** | + position head (masked LM) | **58.10%** | **+6.80** |

`position_weight` sweep on run 2 (5,000 held-out words) — broad plateau, chose 0.5:

| weight | 0.00 | 0.25 | 0.40 | 0.50 | 0.60 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|
| win rate | 57.74 | 57.84 | 58.14 | **58.24** | 58.26 | 57.94 | 57.66 |

Key finding: `position_weight=0` (presence head alone, but *trained* alongside the
position head) already scores 57.74 vs run 1's 51.30. The gain is from the
multi-task training signal, not from consulting the second head at inference.

## run 3 — scale up, and a decisive negative result on self-play

d384/6L, batch 768, 55k steps, self-play from 25k. **Stopped at 30k.**

| step | 5,000 | 10,000 | 15,000 | 17,500 | 20,000 | 22,500 | **25,000** | 27,500 | 30,000 |
|---|---|---|---|---|---|---|---|---|---|
| win rate | 52.06 | 57.16 | 59.08 | 60.82 | 61.86 | 61.54 | **61.92** | 59.02 | 58.78 |

Scaling worked: **61.92%**, comfortably past run 2's 58.10%, and it passed run 2's
final score by step 15k.

**DAgger self-play at 50% of the batch actively hurts: −3.1 points.** The drop
lands exactly on the step where self-play activates and does not recover over two
further evals. Well outside the ±0.7 noise band on 5,000 eval words.

Why, most likely: the model's own trajectories concentrate on states it already
handles. Giving them half the batch displaces the diverse simulated states that
provide coverage, so the model stops seeing the situations it is bad at. This is
the failure mode the `MixedStateSampler` docstring anticipated — the mistake was
assuming 50% was a safe mix.

Not necessarily a dead idea at a *small* fraction (0.1–0.15) or later in training,
but it is not worth more GPU time while simpler gains are untested.

Best checkpoint kept: `artifacts/run3/best_model.pt` @25k, **61.92%**.

## Head-to-head on 10,000 identical held-out words

Training-time numbers are not comparable across runs — each used whatever eval
slice it was configured with. `scripts/compare_models.py` re-scores every
checkpoint on the same words with the same engine, which is the only way a
difference of a few tenths means anything.

| model | params | win rate | mean wrong |
|---|---|---|---|
| run 2 (d256) | 2.9M | 57.77% | 4.169 |
| exp-ema (d256 + EMA) | 2.9M | 57.91% | 4.168 |
| **run 3 (d384)** | 9.4M | **62.08%** | **3.885** |
| ensemble of all three | — | 61.76% | 3.960 |

**EMA: +0.14 points — noise. Rejected.** Free at inference, but it buys nothing
measurable here, so it is not worth the extra concept in the submitted code.

**Equal-weight ensembling of unequal models makes things worse** (61.76 vs 62.08
for run 3 alone): averaging drags the strong model toward the two weak ones.
Ensembling is only worth revisiting between models of comparable strength — e.g.
run 3 and the Kaggle d448 run.

## Inference was ~30x slower than it needed to be

Self-play collection ran at 269 states/s while *training* managed 2,600 states/s
with a backward pass, which made no sense. Cause: `NeuralPolicy` ran fp32
inference with `chunk_size=8192`; at d384 one activation tensor is ~400 MB
against ~330 MB of free VRAM, so the CUDA allocator thrashed.

Fixed by dropping the default chunk to 2048 and running inference under autocast.
10,000 words now score in 6–11 s versus ~2 min for 5,000 before. Also vectorised
`encode_boards` (3.9x on its own, though it was never the real bottleneck — worth
recording that the first diagnosis was wrong).

## exp-binary — presence objective. Hypothesis wrong, rejected.

Stopped at 10k once the verdict was clear. Matched steps against run 2:

| step | 2,500 | 5,000 | 7,500 | 10,000 |
|---|---|---|---|---|
| run 2 (count) | ~43.3 | ~49.7 | 53.0* | **55.13** |
| exp-binary | 41.66 | 47.36 | 50.62 | **52.52** |

\*interpolated. **Binary presence is −2.6 points. Rejected.**

The reasoning that motivated it was half right. What a guess is *ranked* by really
is presence, not repetition — but that says nothing about which **training signal**
teaches ranking better, and that is where the argument broke down:

* Count-weighted cross-entropy runs a softmax **across the 26 letters**, so the
  letters compete. That comparative pressure is exactly what an argmax policy
  needs.
* Per-letter BCE scores each letter independently. Nothing pushes the right
  letter *above* the others, only toward 1.
* Multiplicity is also a real learning signal about word structure, and the
  binary target discards it.

Worth keeping as a documented negative: the inference objective and the best
training objective are not the same thing.

## run 4 — self-play removed. Best model so far.

d384/6L, batch 768, 55k schedule, no self-play. **Killed at 30k** when the
controlling session ended — not a training failure. Best checkpoint survived.

| step | 10,000 | 15,000 | 20,000 | 25,000 | 27,500 | **30,000** |
|---|---|---|---|---|---|---|
| run 3 (self-play @25k) | 57.16 | 59.08 | 61.86 | 61.92 | 59.02 ↓ | 58.78 ↓ |
| run 4 (no self-play) | 57.20 | 60.12 | 61.38 | 62.46 | 62.32 | **63.68** |

Identical to run 3 through step 25k, as expected — then run 3 degrades and run 4
keeps climbing. That is the self-play result confirmed a second time, from the
other direction.

**63.68% at step 30k, with the LR still at 1.5e-4 and 25k steps of schedule
unspent.** Checkpoint: `artifacts/run4/best_model.pt`.

## Crash resume added

Losing 25k steps of a 4.5 h run to a session ending was avoidable. `train()` now
writes `last_state.pt` (model + optimizer + scaler + step + best score + history)
at every eval, and `--resume` continues from it. An interruption now costs at
most one eval interval. Verified by a resume round-trip in the smoke test.

## In flight

| id | config | status |
|---|---|---|
| **run 5** (flagship) | d384/6L, batch 768, **complete 55k schedule**, `--resume` | local, running |
| kaggle-2 | d448/8L, batch 2048, 34k steps, no self-play | Kaggle T4x2, ~8.5 h elapsed |

Run 5 is run 4's configuration allowed to finish its cosine decay. Run 4 reached
63.68% only halfway through the schedule, and models gain most in the final decay
phase, so 65–66% is the expectation.

Kaggle risk: the 12 h session cap. It does not stream logs and commits no output
on timeout, so if it overruns we lose it entirely and fall back on run 5.

## Settled

| change | verdict |
|---|---|
| position head (masked LM) | **+6.8 — keep**, the single biggest win |
| scale d256→d384 | **+4.3 — keep** |
| DAgger self-play @50% | −3.1 — rejected |
| binary presence objective | −2.6 — rejected |
| EMA weight averaging | +0.14, noise — rejected |
| equal-weight ensemble of unequal models | −0.3 — rejected |
| `position_weight` 0.5 | broad plateau, keep |

## Rejected without testing

- **Risk-aware endgame search.** Greedy argmax P(letter present) is already close
  to optimal: every distinct letter in the word must eventually be guessed, and a
  hit is free, so the only cost is guessing an absent letter. Ordering by
  P(present) *is* the right objective. The "43% of losses were one letter short"
  figure reflects irreducible ambiguity, not bad decisions.
- **Feeding remaining lives to the model.** The optimal guess ordering does not
  depend on lives remaining, so the input carries no signal for this objective.
- **Length-reweighted sampling.** Losses concentrate in lengths 7–11, but those
  already carry proportional weight in the corpus, so reweighting mostly moves
  gradient toward lengths 2–4 where the task is close to information-theoretically
  hopeless (win rate 4–17%).
- **Scaling width/depth further.** Explicitly out of scope: brute force is
  downweighted by the audit and was ruled out by Shubham.
