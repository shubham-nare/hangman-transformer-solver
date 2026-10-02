# Hangman — Brand & Buzzword Hackathon

A character-level Transformer blended with a Kneser-Ney beam-search posterior
that plays Hangman on words it has never seen. It wins **77.58%** of 250,000
games at six lives, against 12.70% for the competition's sample submission.

Everything is trained on the provided `train.txt`. No external corpora, no
pretrained weights, no API calls.

Built for Meltwater's 48-hour solo hackathon (September 2026).

---

## 1. What the data actually is

The competition is framed around brand names, hashtags and internet slang. The
data is not that. It is an **English dictionary word list**:

```
aardvark   aalesund   spectropyrheliometer   bemadded   geogenetic
```

| | `train.txt` | `test.txt` |
|---|---|---|
| words | 225,300 | 250,000 |
| unique | 225,300 | 250,000 |
| length (min / mean / max) | 1 / 9.35 / 29 | 2 / 9.43 / 29 |
| character set | `a-z` | `a-z` |
| digits, spaces, punctuation, uppercase | none | none |

**The two sets share zero words.**

That last line is the whole problem. Memorisation is impossible by construction,
so nothing that depends on having seen a word before can score. The rules also
permit non-letter characters in the vocabulary even though the public data has
none, so the engine handles them anyway — it costs nothing and protects against a
private evaluation set that differs.

## 2. Why a learned model — measured, not asserted

Reference policies, evaluated on held-out training words with the real 6-life
engine (`python -m scripts.evaluate_baselines`):

| policy | win rate | mean wrong |
|---|---|---|
| static letter frequency (= `sample_submission.csv`) | 12.70% | 5.742 |
| pattern matching over the training vocabulary | 15.65% | 5.673 |

The second is the standard statistical approach: filter the training vocabulary
to words consistent with the board, then guess the letter appearing in the most
survivors. It barely beats a fixed letter order — **because the vocabularies are
disjoint**. It can only recognise spellings it has already seen.

So learning orthography rather than a word list is not a stylistic preference
here. It is the only thing that generalises.

For calibration: this is the Trexquant hangman problem, whose reference algorithm
scores ~18%. Absolute win rates in this task are low by nature.

## 3. Approach

### 3.1 Training states — the part that matters most

The obvious way to build training data is to reveal a random subset of each
word's letters. It is also wrong. A real game reveals letters *in the order a
player guessed them*, and carries the scars of that player's misses. A model
fitted to uniform random subsets is answering a different question at inference
than the one it was trained on.

Instead, states come from **simulated trajectories**. Each word is played out by
a stochastic frequency-ranked player and a uniformly random turn along that
trajectory is sampled. Two knobs control the spread:

* `exploration_temperature` perturbs the player's ranking, so the corpus covers
  many plausible orderings rather than one canonical one.
* `uniform_order_probability` plays a quarter of games in a fully random letter
  order, reaching unusual states a competent player rarely visits — which keeps
  the model from over-fitting to the habits of its bootstrap policy.

Whole games are simulated in closed form via cumulative sums over Gumbel-sampled
guess orders, so generating millions of states costs seconds. Benchmarked, state
generation is 2–10 ms against 76–284 ms for the GPU step: **training is
GPU-bound and the sampler is effectively free.**

Tests assert every sampled state is one the engine could actually reach, and that
it is still winnable.

### 3.2 Model

An encoder-only Transformer with two heads. The shipped model is 6 layers,
`d_model` 384, 8 heads, 11.2M parameters; the ablations in section 5 use a
4-layer, 2.8M-parameter version.

* **Presence head** — pools the board (masked mean + max) and asks *which letter
  is somewhere in this word*. This is the decision actually being made.
* **Position head** — a character-level masked language model: for each blank,
  *what letter fills this slot*. This is the denser signal — one gradient per
  hidden character rather than one per board — and it is what teaches English
  orthography: that `q` takes `u`, that `_ing` is a likely ending. Those cues are
  precisely what transfers to unseen words.

The 26-dim guessed-letter vector is projected into the model width and added at
**every position**, so self-attention can reason about eliminated letters at each
character. This is not redundant with the board: a letter guessed and *missed*
never appears on the board, yet knowing it is absent is some of the most
valuable information the player holds.

Per-blank distributions become a word-level score by treating blanks as
independent — `P(absent everywhere) = Π(1 - pᵢ)` — scored as the negative log of
that product. Summing logs avoids underflow, and the result is monotonic in
`P(present)`, which is all a ranking needs.

### 3.3 Beam-search posterior over completions

The independence assumption above is adequate when eleven blanks remain and
nearly worthless when two do — and **69% of the Transformer's losses end with two
or fewer blanks still hidden**. That is the regime the second component targets.

A character n-gram language model with Kneser-Ney smoothing, fitted on
`train.txt`, proposes the most plausible completions consistent with the board
by beam search. Letters are then ranked by an exact posterior over that beam:

```
P(letter hits) = Σ_{w in beam, letter in w} P(w)  /  Σ_{w in beam} P(w)
```

No independence between positions. Constraints follow from the rules: a guess
reveals every occurrence of its letter, so no letter already guessed — hit or
miss — can occupy a remaining blank.

The final guess is the argmax of a convex blend of the two signals. They fail on
different words, which is why the blend beats either alone.

**Making it fast enough to run.** The reference implementation costs 0.27 s per
word — 18 hours over 250,000 games. `hangman/fast_beam.py` flattens the language
model into a log-probability table so a beam step is one gather and one
`argpartition`. The table is the hard part: at order 7 the context space is
27⁶ = 387M rows. Only the contexts that actually occur are stored, with unseen
ones resolved through two levels of back-off baked into a single index at build
time. Because Kneser-Ney's own recursion backs off the same way, that deferral
is exact rather than an approximation. The full evaluation runs in 34 minutes.

### 3.4 Model selection

The loss is a proxy. The competition scores **games won**, and the two diverge:
loss plateaued at 2.65 while win rate climbed from 33% to 39%. So evaluation runs
the real 6-life engine on held-out words and checkpoints on win rate.

## 4. Integrity

`test.txt` ships with its answers. A solver could therefore score 100% by reading
them, which the rules treat as disqualifying.

Rather than rely on discipline, the engine/policy boundary makes it impossible:
policies receive an `Observation` carrying the board, the guess history and the
strike count — and **never the secret word**. A test asserts the observation has
no `word` attribute and that policies are handed observations rather than game
states.

Beyond that:

* Architecture, objective and training decisions were all made on a held-out
  slice of `train.txt` — see `EXPERIMENTS.md`.
* The submission is re-scored by an **independent replay** through the engine, so
  reported numbers are ones a separate simulation produced rather than ones the
  solver claimed about itself.
* A test asserts the notebook makes no network calls and references no external
  model.
* Each submission is sealed with a SHA-256 manifest of the checkpoint, the CSV
  and every source file that produced it (`scripts/seal_submission.py`,
  `FINAL_*/MANIFEST.txt`).

**One caveat on the headline number.** The final inference settings (blend
weight, beam width, position weight) were not chosen blind. Several
configurations were played against the public `test.txt` and the best was
shipped, so 77.58% is mildly optimistic as an estimate for unseen data. The
size of that effect is bounded by the spread across the configurations tried:
the blend weight the held-out sweep preferred (0.26) scores 76.35% on the same
set.

**An approach that was built and thrown away.** An earlier hybrid filtered
external dictionaries (NLTK, SOWPODS) against the board and reached 86.22%. The
rules forbid any external word list, so it was abandoned before being submitted.
The code is kept in `abandoned_experiments/` with a note on why; nothing in the
submitted pipeline imports it. The beam posterior in 3.3 is the legal version of
the same idea — generate the hypothesis set instead of looking it up.

## 5. Results

### Final pipeline — all 250,000 words of `test.txt`

| system | win rate | mean wrong |
|---|---|---|
| static letter frequency (sample submission, held-out estimate) | 12.70% | 5.742 |
| Transformer, presence head only, 2.8M | 53.53% | 4.416 |
| Transformer, dual head, 6 layers / `d_model` 384, 7.6M | 68.80% | 3.512 |
| + order-5 Kneser-Ney beam posterior | 73.87% | 3.273 |
| + order-6 LM, 11.2M Transformer trained 90k steps | 76.61% | 3.058 |
| + order-7 LM | 77.12% | — |
| + retuned blend weight, beam 800 | **77.58%** | **2.937** |

### Controlled ablations — held-out slice of `train.txt`

| change | effect |
|---|---|
| position head (masked LM) | **+6.8** — the gain is from multi-task training, not from consulting the head at inference |
| scale `d_model` 256 → 384 | **+4.3** |
| DAgger self-play at 50% of the batch | −3.1, rejected |
| binary presence objective | −2.6, rejected |
| neural character LM in place of Kneser-Ney | −1.78 (paired test, p < 0.001), rejected |
| EMA weight averaging | +0.14, noise, rejected |
| equal-weight ensemble of unequal models | −0.3, rejected |
| REINFORCE fine-tuning | unstable — 65.8% at step 200, 60.5% at step 400, stopped |

Full detail, including why each negative result came out the way it did, is in
`EXPERIMENTS.md`.

## 6. Reproducing

```bash
pip install numpy pandas pytest
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Data is not included in this repository. Place train.txt and test.txt in data/
kaggle competitions download -c brand-buzzword-hackathon -p data

python -m pytest tests -q                 # 89 tests
python -m scripts.evaluate_baselines      # reference policies

python -u -m scripts.train_model --steps 90000 --batch-size 1024 \
    --d-model 384 --n-layers 6 --n-heads 8 --dim-feedforward 1536 \
    --resume --output-dir artifacts/run6

python -m scripts.generate_submission --checkpoint artifacts/run6/best_model.pt \
    --use-blend --lm-order 7 --beam-width 800 --max-blanks 10 \
    --blend-weight 0.50 --position-weight 0.25

python -m scripts.build_notebook          # emit the Kaggle notebook
```

Model checkpoints, competition data and generated submission CSVs are not
tracked. The order-7 table and its index take about 3.3 GB of RAM.

## 7. Layout

```
hangman/
  game.py          Engine, Observation, batched lockstep play, metrics
  encoding.py      Tensor encoding shared by training and inference
  data.py          Loading and train/validation splitting
  dataset.py       Simulated-trajectory state sampler
  model.py         HangmanTransformer, dual heads, score aggregation
  policy.py        Batched neural guessing policy
  train.py         Training loop, win-rate model selection, crash resume
  char_lm.py       Kneser-Ney character language model
  beam_solver.py   Beam-search posterior, reference implementation
  fast_beam.py     Vectorised beam over dense and sparse back-off tables
  blend_policy.py  Shipped policy: neural marginals + beam posterior
  baselines.py     Static-frequency and pattern-matching references
  submission.py    CSV writing and independent replay validation
  selfplay.py  ema.py  rl.py  neural_lm.py     Measured and rejected
scripts/           Training, sweeps, diagnostics, submission, sealing
tests/             89 tests
gpu_experiment/    Batched GPU beam — isolated, failed verification, not shipped
abandoned_experiments/   External-dictionary hybrid — violates the rules, not shipped
FINAL_*/           SHA-256 manifest for each sealed submission
```

The Kaggle notebook is **generated from these modules** by
`scripts/build_notebook.py` rather than hand-copied, so the submitted code is the
code the test suite covers. A test executes the generated notebook's inlined
modules and plays a game through them.

## 8. Notes on the split

`split_words` shuffles before splitting. `train.txt` is sorted, so a positional
split would put entire prefix families — every `un-` word, say — on one side and
measure the wrong thing.
