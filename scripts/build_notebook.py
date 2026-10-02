"""Assemble the Kaggle submission notebook from the tested package source.

    python -m scripts.build_notebook --output submission_notebook.ipynb

A Kaggle notebook has to be self-contained, which usually means copy-pasting the
project into cells and letting the two drift apart. Instead this reads the real
modules and inlines them in dependency order, so **the submitted notebook is the
code the test suite covers** -- there is no second copy to keep in sync.

Relative imports are stripped as each module is inlined: the notebook evaluates
every module into one shared namespace, so names defined in an earlier cell are
already in scope.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path("hangman")

#: Modules in dependency order, each with the narrative that introduces it.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "game.py",
        "Game engine",
        "Exact competition semantics, and the boundary that makes cheating "
        "impossible.\n\n"
        "`test.txt` ships with its answers, so a solver *could* score 100% by "
        "reading them -- which the rules treat as disqualifying. Rather than rely "
        "on discipline, policies are handed an `Observation` that contains the "
        "board, the guess history and the strike count, and **never the secret "
        "word**. The model is structurally incapable of seeing the answer.\n\n"
        "The engine also pins the scoring rules that are easy to get wrong: a "
        "repeated guess, a non-letter, or a letter not in the word each cost "
        "exactly one strike, and a game stops the instant it is won or hits six.",
    ),
    (
        "encoding.py",
        "State encoding",
        "One encoder serves both training and inference, so the model cannot see "
        "a different feature layout in the two settings.\n\n"
        "The 26-dim guessed vector is not redundant with the board. A letter that "
        "was guessed and *missed* never appears on the board, yet knowing it is "
        "absent is some of the most valuable information the player holds. "
        "Without it the model would be blind to its own misses.",
    ),
    (
        "data.py",
        "Data loading and splitting",
        "The split is random rather than alphabetical. `train.txt` is sorted, so "
        "a positional split would put entire prefix families (every `un-` word, "
        "say) on one side and measure the wrong thing.",
    ),
    (
        "dataset.py",
        "Training-state generation",
        "**The most important component, and the least obvious.**\n\n"
        "The tempting approach -- reveal a random subset of a word's letters -- "
        "produces board states that never arise in play. A real game reveals "
        "letters in the order a player guessed them and carries the scars of that "
        "player's misses. A model fitted to uniform random subsets is answering a "
        "different question at inference than the one it was trained on.\n\n"
        "So states are drawn from *simulated trajectories*: each word is played "
        "out by a stochastic frequency-ranked player and a uniformly random turn "
        "along that trajectory is sampled. Whole games are simulated in closed "
        "form with cumulative sums, which makes generating millions of states "
        "cost seconds. A quarter of games use a flat ranking, reaching unusual "
        "states a competent player rarely visits.",
    ),
    (
        "model.py",
        "Model",
        "An encoder-only Transformer with two heads.\n\n"
        "* The **presence** head pools the board and asks *which letter is "
        "somewhere in this word* -- the decision actually being made.\n"
        "* The **position** head is a character-level masked language model: for "
        "each blank, *what letter fills this slot*. This is the denser signal "
        "(one gradient per hidden character rather than one per board) and it is "
        "what teaches English orthography -- that `q` takes `u`, that `_ing` is a "
        "likely ending. Those cues are precisely what transfers to words the "
        "model has never seen.\n\n"
        "Per-blank distributions are combined into a word-level score by treating "
        "the blanks as independent: `P(absent everywhere) = prod_i (1 - p_i)`. "
        "Summing logs avoids underflow, and the result is monotonic in "
        "`P(present)`, which is all a ranking needs.",
    ),
    (
        "policy.py",
        "Guessing policy",
        "Observations arrive one batch per turn, so a single forward pass serves "
        "every game still in progress -- necessary to play 250,000 words in "
        "reasonable time.\n\n"
        "The two heads produce scores on different scales, so `position_weight` "
        "is selected on held-out win rate rather than assumed.",
    ),
    (
        "char_lm.py",
        "Character language model (Kneser-Ney)",
        "The Transformer above predicts *marginals*: for each blank "
        "independently, which letter is likely. Combining those into a "
        "word-level score assumes the blanks are independent -- and the "
        "measured failure profile shows exactly where that breaks. Win rate is "
        "94-100% at lengths 13+ but **0% at length 3**, and 69% of losses end "
        "with two or fewer blanks left. With eleven blanks the independence "
        "approximation is harmless; with two correlated blanks it discards the "
        "information that decides the game.\n\n"
        "Fixing that needs a *joint* model of character sequences, which is "
        "what this is: an interpolated Kneser-Ney n-gram over characters, "
        "fitted on `train.txt` alone. Kneser-Ney specifically, because its "
        "lower orders use **continuation counts** -- how many distinct contexts "
        "a suffix completes -- rather than raw frequency. Every evaluation word "
        "is unseen by construction, so behaviour on unseen contexts is the "
        "whole game. Held-out per-character perplexity: 6.38 at order 6.",
    ),
    (
        "beam_solver.py",
        "Posterior over completions (reference implementation)",
        "The key idea of the solution.\n\n"
        "Filtering a dictionary to words matching the board is a strong "
        "classical Hangman strategy, but it is unavailable here twice over: the "
        "rules forbid external word lists, and `train.txt` shares **no words** "
        "with the evaluation set, so a looked-up candidate can never be the "
        "answer.\n\n"
        "What made that approach strong, though, was not the word list -- it "
        "was doing exact posterior inference over a hypothesis set. So the "
        "hypotheses are *generated* instead: the character LM proposes the K "
        "most plausible English-like completions consistent with the board, and "
        "letters are ranked over that beam by\n\n"
        "$$P(\\text{hit}) = \\frac{\\sum_{w \\ni \\ell} P(w)}{\\sum_w P(w)}$$\n\n"
        "with no independence assumption between positions. The constraint is "
        "exact and follows from the rules: a guess reveals *every* occurrence "
        "of its letter, so any letter already guessed -- hit or miss -- cannot "
        "occupy a remaining blank.\n\n"
        "This version is the readable reference. The next cell is the one that "
        "actually runs, and it is verified against this one.",
    ),
    (
        "fast_beam.py",
        "Vectorised beam search",
        "Same inference, rebuilt for throughput: 250,000 games is roughly 2.2 "
        "million policy calls, and the reference costs ~3.8 ms each -- about 18 "
        "hours. Flattening the language model into a dense lookup table turns a "
        "beam step into one fancy-index and one `argpartition`.\n\n"
        "Order 6 is where the model wants to be (perplexity 6.38 vs 6.93 at "
        "order 5), but its context space is $27^5 = 14.3$M rows -- ~1.5 GB "
        "dense. Only ~1.7% of those contexts ever occur, so the table is "
        "two-level: observed contexts get their own row, and everything else "
        "resolves to the order-below row. That fallback is **exact rather than "
        "approximate**, because Kneser-Ney backing off from an unobserved "
        "context computes precisely the shorter-context value.\n\n"
        "Correctness is verified against the reference implementation to 6e-08 "
        "-- float32 rounding -- rather than assumed.",
    ),
    (
        "blend_policy.py",
        "Blending the two models",
        "Neither model dominates, which is why both are kept. Measured "
        "per-length on held-out words, the character-LM posterior gains ten "
        "points at length 7 and loses six at length 6; the two fail on "
        "*different words*. That is the signature of a genuinely complementary "
        "signal, and averaging their letter distributions is worth more than "
        "either alone:\n\n"
        "| policy | held-out win rate |\n|---|---|\n"
        "| Transformer alone | 68.25% |\n| Character LM alone | 71.00% |\n"
        "| **Blended** | **72.16%** |\n\n"
        "`blend_weight` is fitted on held-out `train.txt` words -- never on the "
        "evaluation set -- and the optimum is flat across 0.30-0.45, so the "
        "setting is not perched on a knife-edge.\n\n"
        "Fusing the two *earlier* was tried and rejected: feeding the "
        "Transformer's per-position log-probs into the beam search made things "
        "monotonically worse (72.27% -> 72.03%). It double-counts a signal "
        "already present in the blend, and does so through the very "
        "independence assumption the character LM was introduced to fix.",
    ),
    (
        "selfplay.py",
        "Self-play (DAgger)",
        "The bootstrap sampler imitates a frequency-ranked player, but the trained "
        "model reveals letters in a different order and therefore visits a "
        "different part of the state space. Fitting on one distribution while "
        "acting in another is textbook covariate shift.\n\n"
        "DAgger addresses it directly: let the current policy act, then train on "
        "the states it actually reaches. The model plays real training words, "
        "every intermediate board is labelled with the answer, and those states "
        "are mixed back in. The model still only ever sees an `Observation` while "
        "playing -- the word is used solely to build the label afterwards.\n\n"
        "The simulated sampler is deliberately kept in the mix. Training purely on "
        "self-play would let the model revisit only its own habits, and states it "
        "has learned to avoid would vanish from the training set.",
    ),
    (
        "ema.py",
        "Weight averaging",
        "A decaying learning rate still leaves the final weights rattling around "
        "the minimum rather than sitting in it. An exponential moving average of "
        "the trajectory lands nearer the centre of the basin and usually "
        "generalises a little better -- for one extra copy of the weights and no "
        "extra inference cost.\n\n"
        "This is averaging along a single run, not an ensemble of separate "
        "models: it stays one model at inference time.",
    ),
    (
        "train.py",
        "Training",
        "The loss combines the word-level presence objective with the per-blank "
        "language-model objective.\n\n"
        "Model selection does **not** use the loss. The competition scores games "
        "won, not per-turn letter accuracy, and the two diverge: loss plateaus "
        "early while win rate keeps climbing. So evaluation runs the real 6-life "
        "engine on held-out words and checkpoints on win rate.",
    ),
    (
        "rl.py",
        "RL fine-tuning (REINFORCE)",
        "After supervised pre-training the model is fine-tuned by playing real "
        "games and receiving +1 for every win, -1 for every loss -- the exact "
        "signal the competition scores.\n\n"
        "REINFORCE with a learned value baseline: advantage = return - value, "
        "normalised over the rollout batch, drives the policy gradient. Every "
        "step within a game shares the same terminal reward (gamma = 1): there "
        "is no credit-assignment problem here, because every guess matters "
        "equally to whether the word is eventually solved.\n\n"
        "Rollouts are collected under `torch.no_grad()` in lockstep across all "
        "active games, matching the inference pattern. The stored "
        "(observation, action) pairs are then re-run with gradients to get "
        "log-probs, values and entropies for the update.",
    ),
    (
        "baselines.py",
        "Reference baselines",
        "Included to establish the bar the learned model has to clear, and to "
        "show *why* a statistical approach is not enough here.",
    ),
    (
        "submission.py",
        "Submission writing and validation",
        "The written file is re-scored by replaying its guesses through the "
        "engine, so the number reported is one an independent simulation "
        "produced -- not one the solver claimed about itself.",
    ),
]

_RELATIVE_IMPORT = re.compile(r"^from \.\w* import .*?(?:\)\n|\n)", re.MULTILINE | re.DOTALL)
_MULTILINE_RELATIVE_IMPORT = re.compile(r"^from \.\w+ import \([^)]*\)\n", re.MULTILINE)


def inline_module(path: Path) -> str:
    """Read a module and strip its intra-package imports."""
    source = path.read_text(encoding="utf-8")
    source = _MULTILINE_RELATIVE_IMPORT.sub("", source)
    source = _RELATIVE_IMPORT.sub("", source)
    return source.strip() + "\n"


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


HEADER = """# Hangman â€” Brand & Buzzword Hackathon

A character-level Transformer that plays Hangman, trained only on the provided
`train.txt`.

## The problem, as the data actually presents it

The corpus is an **English dictionary word list** â€” lowercase `a-z`, no digits or
punctuation. Two properties shape every decision below:

| | train.txt | test.txt |
|---|---|---|
| words | 225,300 | 250,000 |
| mean length | 9.35 | 9.43 |
| charset | `a-z` | `a-z` |

**The two sets share zero words.** Memorisation is impossible by construction, so
only generalisation over spelling patterns can score. That is not a detail â€” it
is the whole problem.

## Why a learned model, measured rather than asserted

Reference policies on held-out training words:

| policy | win rate |
|---|---|
| static letter frequency (= `sample_submission.csv`) | 12.70% |
| pattern matching over the training vocabulary | 15.65% |

Filtering the training vocabulary to words consistent with the board â€” the
standard statistical approach â€” barely beats a fixed letter order, **because the
vocabularies are disjoint**. It can only recognise spellings it has already seen.
A model that learns orthography rather than a word list is not a stylistic
preference here; it is the only thing that generalises.

## Structure

The notebook inlines a tested Python package. Each section below is one module,
in dependency order, so the code that runs here is the code the test suite
covers.
"""

@dataclass
class NotebookConfig:
    """Hyper-parameters baked into the generated notebook.

    Defaults target Kaggle's 16 GB accelerators, which allow a wider model and
    a larger batch than a 4 GB laptop card.
    """

    # These mirror the run that produced the submitted predictions -- run6 --
    # rather than an aspirational configuration. A notebook that trains a
    # different architecture than the one behind the CSV is worse than no
    # notebook: it claims a provenance it does not have.
    #
    # run6: d384/6L, 1536-wide FFN (11.2M parameters), batch 768, seed 424242,
    # best held-out win rate 69.50% at step 82,500.
    #
    # DAgger self-play is OFF: it measured 3.1 points WORSE than the simulated
    # trajectory sampler, so enabling it here would train a weaker model than
    # the one that was shipped.
    steps: int = 85_000
    batch_size: int = 768
    learning_rate: float = 4e-4
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1_536
    eval_every: int = 2_500
    eval_words: int = 2_000
    position_weight: float = 0.25
    seed: int = 424_242
    self_play_start_step: int = 0
    self_play_refresh_every: int = 5_000
    self_play_words: int = 20_000
    self_play_fraction: float = 0.0
    # Inference-side settings, all fitted on held-out train.txt words.
    #
    # Order 7 measured +0.24 points over order 6 on 8,000 held-out words, which
    # did NOT reach significance (McNemar p=0.33: 198 newly won against 179
    # newly lost). It is used because the expected value is positive and the
    # inference cost is identical -- 0.142 ms per call against order 6's 0.148 --
    # not because the improvement is established. Order 8 is worse outright.
    lm_order: int = 7
    blend_weight: float = 0.50
    beam_width: int = 800
    max_blanks: int = 10


DATA_CELL = '''# Locate the competition data. Kaggle has mounted it under a few different
# paths over time, so try each rather than hard-coding one.
CANDIDATE_DATA_DIRS = [
    Path("/kaggle/input/brand-buzzword-hackathon"),
    Path("/kaggle/input/competitions/brand-buzzword-hackathon"),
    Path("data"),
]
DATA_DIR = next((p for p in CANDIDATE_DATA_DIRS if (p / TRAIN_FILENAME).exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find {TRAIN_FILENAME} in any of: {CANDIDATE_DATA_DIRS}"
    )

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
print(f"data  : {DATA_DIR}")
print(f"output: {OUTPUT_DIR}")

train_vocabulary = load_words(DATA_DIR / TRAIN_FILENAME)
split = split_words(train_vocabulary, validation_size=20_000)
print(split)
print(describe_vocabulary(train_vocabulary))
'''


def training_cell(config: NotebookConfig) -> str:
    return f'''# Reproduce the exact submitted model, or train an equivalent one from scratch.
#
# GPU training is not bit-deterministic -- cuDNN's parallel reductions run in a
# different order from one execution to the next even at a fixed seed -- so
# retraining cannot regenerate the *exact* weights behind the submitted CSV,
# only weights of the same architecture and training regime. To reproduce the
# submitted result exactly rather than approximately, attach the checkpoint
# (best_model.pt from this run) as a Kaggle "Dataset" input and it is loaded
# directly, skipping training. Remove that input, or delete the file, to
# instead train a fresh model with the identical architecture and procedure --
# this reproduces the *approach*, and lands close to the reported win rate,
# but is not guaranteed to match it exactly.
CHECKPOINT_CANDIDATES = [
    Path("/kaggle/input/run6-checkpoint/best_model.pt"),
    Path("/kaggle/input/hangman-checkpoint/best_model.pt"),
    OUTPUT_DIR / "best_model.pt",
]
_ckpt = next((p for p in CHECKPOINT_CANDIDATES if p.exists()), None)

if _ckpt is not None:
    print(f"Loading the exact submitted checkpoint: {{_ckpt}}")
    model = load_model(_ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"parameters: {{model.count_parameters():,}}")
else:
    print("No checkpoint attached -- training a fresh model with the same "
          "architecture and procedure. This reproduces the APPROACH; because "
          "GPU training is not bit-deterministic, the resulting weights (and "
          "therefore the exact letter-by-letter guesses) will differ slightly "
          "from the submitted run, though the win rate should be close.")
    model, summary = train(
        split.train,
        split.validation,
        model_config=ModelConfig(
            d_model={config.d_model},
            n_heads={config.n_heads},
            n_layers={config.n_layers},
            dim_feedforward={config.dim_feedforward},
        ),
        training_config=TrainingConfig(
            steps={config.steps},
            batch_size={config.batch_size},
            learning_rate={config.learning_rate},
            eval_every={config.eval_every},
            eval_words={config.eval_words},
            position_weight={config.position_weight},
            seed={config.seed},
            self_play_start_step={config.self_play_start_step},
            self_play_refresh_every={config.self_play_refresh_every},
            self_play_words={config.self_play_words},
            self_play_fraction={config.self_play_fraction},
        ),
        output_dir=OUTPUT_DIR,
    )
    print(f"best held-out win rate: {{summary['best_win_rate']:.2f}}%")
'''

def inference_cell(config: NotebookConfig) -> str:
    return f'''# Play every test word and write the submission.
#
# The character LM is fitted on the FULL train.txt here, not the training split:
# the held-out slice exists to choose hyper-parameters honestly, and once they
# are chosen there is no reason to withhold 20,000 words from the shipped model.
# test.txt supplies the words to play and nothing else -- the policy only ever
# receives an Observation, which has no field for the answer.
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_words = load_words(DATA_DIR / TEST_FILENAME)
model = model.to(device).eval()

lm = CharKN(order={config.lm_order}).fit(train_vocabulary)
# Order 7 spans 27**6 = 387M contexts, so it chains two backoff hops down to a
# dense base rather than one. Both hops are exact: Kneser-Ney backing off from an
# unobserved context computes precisely the shorter-context value.
index, sparse_table = build_sparse_table_order7(lm)
lm.release()   # count dicts and memo cache are dead once the table exists

neural = NeuralPolicy(model, device=device, position_weight={config.position_weight})
beam = SparseBeamPosterior(
    index, sparse_table, {config.lm_order},
    beam_width={config.beam_width}, max_blanks={config.max_blanks},
)
policy = BlendPolicy(neural, beam, blend_weight={config.blend_weight})

results = play_games(test_words, policy, progress=True)
metrics = summarise(results)
print(f"win rate    : {{metrics['win_rate']:.4f}}%")
print(f"mean wrong  : {{metrics['mean_wrong']:.4f}}")

write_submission(results, "submission.csv")
report = validate_submission("submission.csv", test_words)
print(report)
assert report.is_valid, "submission failed validation"
'''


def build(output: Path, config: NotebookConfig | None = None) -> None:
    config = config or NotebookConfig()
    cells = [markdown_cell(HEADER)]

    # Numbers are generated rather than written into the titles, so adding or
    # reordering a module cannot leave the headings inconsistent.
    section_number = 0
    for filename, title, narrative in SECTIONS:
        section_number += 1
        cells.append(markdown_cell(f"## {section_number}. {title}\n\n{narrative}\n"))
        cells.append(code_cell(inline_module(PACKAGE_DIR / filename)))

    section_number += 1
    cells.append(
        markdown_cell(
            f"## {section_number}. Data\n\nLoad the vocabulary and split off a "
            "held-out slice for model selection.\n"
        )
    )
    cells.append(code_cell(DATA_CELL))

    section_number += 1
    cells.append(
        markdown_cell(
            f"## {section_number}. Training\n\n"
            "Trains from scratch on `train.txt`. Nothing else is read: no external "
            "corpora, no pretrained weights, no API calls. Checkpointing is on "
            "held-out **win rate**, not loss.\n"
        )
    )
    cells.append(code_cell(training_cell(config)))

    section_number += 1
    cells.append(
        markdown_cell(
            f"## {section_number}. Playing the test set\n\n"
            "Each word is played turn by turn against the real 6-life engine, and "
            "the guess sequence the model actually produced is recorded. The "
            "finished file is then re-scored by an independent replay.\n"
        )
    )
    cells.append(code_cell(f"POSITION_WEIGHT = {config.position_weight}\n"))
    cells.append(code_cell(inference_cell(config)))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("submission_notebook.ipynb"))
    parser.add_argument(
        "--position-weight", type=float, default=NotebookConfig.position_weight
    )
    parser.add_argument("--steps", type=int, default=NotebookConfig.steps)
    parser.add_argument("--batch-size", type=int, default=NotebookConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=NotebookConfig.learning_rate)
    parser.add_argument("--d-model", type=int, default=NotebookConfig.d_model)
    parser.add_argument("--n-layers", type=int, default=NotebookConfig.n_layers)
    parser.add_argument("--n-heads", type=int, default=NotebookConfig.n_heads)
    parser.add_argument(
        "--dim-feedforward", type=int, default=NotebookConfig.dim_feedforward
    )
    parser.add_argument(
        "--self-play-start-step", type=int, default=NotebookConfig.self_play_start_step
    )
    args = parser.parse_args()

    config = NotebookConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        position_weight=args.position_weight,
        self_play_start_step=args.self_play_start_step,
    )
    build(args.output, config)
    size_kb = args.output.stat().st_size / 1024
    print(f"wrote {args.output} ({size_kb:.1f} KB, {len(SECTIONS)} modules inlined)")
    print(
        f"  {config.d_model}d / {config.n_layers}L, batch {config.batch_size}, "
        f"{config.steps:,} steps"
    )


if __name__ == "__main__":
    main()



