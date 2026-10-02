"""Where do the losses come from? Anatomy of failure on held-out TRAIN words.

Uses only the held-out slice of train.txt -- never test.txt -- so anything
learned here is a property of the model and of English, not of the evaluation
set. Any hyper-parameter chosen from these numbers still generalises.

Reports:
  1. Win rate by word length            -- which lengths are we losing?
  2. Loss anatomy                       -- blanks still hidden when we died
  3. Miss timing                        -- at which turn the strikes land
  4. First-guess optimality by length   -- turn 1 happens in 100% of games
  5. Calibration                        -- is the score a usable probability?

    python -m scripts.diagnose_losses --checkpoint artifacts/run5/best_model.pt
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.game import ALPHABET, MAX_WRONG_GUESSES, GameState
from hangman.policy import NeuralPolicy
from hangman.train import load_model


def play_and_record(words: list[str], policy, batch: int = 2048) -> list[dict]:
    """Play every word, recording per-game forensics."""
    records: list[dict] = []
    for start in range(0, len(words), batch):
        chunk = words[start : start + batch]
        states = [GameState(word=w) for w in chunk]
        # Per-game: turn index of each miss, and the first guess made.
        miss_turns: list[list[int]] = [[] for _ in chunk]
        first_guess: list[str] = ["" for _ in chunk]
        turn = 0
        active = [i for i, s in enumerate(states) if not s.is_over]
        while active:
            obs = [states[i].observation for i in active]
            guesses = policy.next_guesses(obs)
            for k, i in enumerate(active):
                g = guesses[k]
                if turn == 0:
                    first_guess[i] = g
                hit = states[i].apply_guess(g)
                if not hit:
                    miss_turns[i].append(turn)
            active = [i for i in active if not states[i].is_over]
            turn += 1

        for i, s in enumerate(states):
            blanks_left = s.board.count("_")
            records.append(
                {
                    "word": s.word,
                    "length": len(s.word),
                    "solved": s.is_solved,
                    "wrong": s.wrong_guesses,
                    "blanks_left": blanks_left,
                    "distinct": len(set(s.word)),
                    "n_guesses": len(s.guesses),
                    "miss_turns": miss_turns[i],
                    "first_guess": first_guess[i],
                }
            )
    return records


def optimal_first_guess(train_words: list[str]) -> dict[int, str]:
    """Letter maximising P(letter in word | length), from TRAIN words only."""
    by_len: dict[int, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for w in train_words:
        L = len(w)
        totals[L] += 1
        for ch in set(w):
            by_len[L][ch] += 1
    best: dict[int, str] = {}
    for L, counter in by_len.items():
        if totals[L] >= 50:
            best[L] = counter.most_common(1)[0][0]
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--eval-words", type=int, default=6000)
    ap.add_argument("--validation-size", type=int, default=20_000)
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    model.eval()

    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=args.validation_size, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(split.validation), size=min(args.eval_words, len(split.validation)), replace=False)
    eval_words = [split.validation[i] for i in idx]

    policy = NeuralPolicy(model, device=device, position_weight=args.position_weight)
    print(f"\nplaying {len(eval_words):,} HELD-OUT TRAIN words (test.txt untouched)\n")
    recs = play_and_record(eval_words, policy)

    total = len(recs)
    wins = sum(r["solved"] for r in recs)
    print(f"overall win rate: {100*wins/total:.2f}%  ({wins:,}/{total:,})\n")

    # --- 1. by length ---
    print("1. WIN RATE BY LENGTH")
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_len[r["length"]].append(r)
    print(f"   {'len':>4}{'n':>7}{'win%':>8}{'mean_wrong':>12}   share_of_losses")
    tot_losses = total - wins
    for L in sorted(by_len):
        g = by_len[L]
        if len(g) < 20:
            continue
        w = sum(x["solved"] for x in g)
        losses = len(g) - w
        mw = sum(x["wrong"] for x in g) / len(g)
        share = 100 * losses / max(tot_losses, 1)
        bar = "#" * int(share / 1.5)
        print(f"   {L:>4}{len(g):>7}{100*w/len(g):>7.1f}%{mw:>12.2f}   {share:5.1f}% {bar}")

    # --- 2. loss anatomy ---
    print("\n2. LOSS ANATOMY (how close were we when we died?)")
    losses = [r for r in recs if not r["solved"]]
    if losses:
        bl = Counter(r["blanks_left"] for r in losses)
        print(f"   {'blanks_left':>12}{'count':>8}{'pct':>8}")
        for k in sorted(bl):
            if bl[k] >= 1:
                print(f"   {k:>12}{bl[k]:>8}{100*bl[k]/len(losses):>7.1f}%")
        near = sum(v for k, v in bl.items() if k <= 2)
        print(f"   -> died with <=2 blanks left: {100*near/len(losses):.1f}% of losses (near misses)")

    # --- 3. miss timing ---
    print("\n3. MISS TIMING (turn index of each strike; turn 0 = first guess)")
    mt_win: Counter = Counter()
    mt_loss: Counter = Counter()
    for r in recs:
        for t in r["miss_turns"]:
            (mt_win if r["solved"] else mt_loss)[t] += 1
    print(f"   {'turn':>5}{'misses_in_wins':>16}{'misses_in_losses':>18}")
    for t in range(0, 16):
        if mt_win[t] or mt_loss[t]:
            print(f"   {t:>5}{mt_win[t]:>16,}{mt_loss[t]:>18,}")

    # --- 4. first guess ---
    print("\n4. FIRST GUESS vs TRAIN-OPTIMAL (turn 1 occurs in 100% of games)")
    opt = optimal_first_guess(split.train)
    fg: dict[int, Counter] = defaultdict(Counter)
    for r in recs:
        fg[r["length"]][r["first_guess"]] += 1
    print(f"   {'len':>4}{'model':>8}{'train_opt':>11}   match")
    mismatches = 0
    covered = 0
    for L in sorted(fg):
        if len(by_len[L]) < 20 or L not in opt:
            continue
        m = fg[L].most_common(1)[0][0]
        covered += len(by_len[L])
        ok = m == opt[L]
        if not ok:
            mismatches += len(by_len[L])
        print(f"   {L:>4}{m:>8}{opt[L]:>11}   {'yes' if ok else 'NO'}")
    if covered:
        print(f"   -> games whose first guess differs from train-optimal: {100*mismatches/covered:.1f}%")


if __name__ == "__main__":
    main()
