"""Break a submission's score down by word length and failure mode.

    python -m scripts.analyze_errors --submission submission.csv

Win rate alone does not say where the remaining headroom is. This replays a
finished submission and reports performance per word length, plus how close the
losses came, which is what determines whether more modelling can help.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from hangman.data import DEFAULT_DATA_DIR, TEST_FILENAME, load_words
from hangman.game import MAX_WRONG_GUESSES, GameState
from hangman.submission import GUESSES_COLUMN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    test_words = load_words(args.data_dir / TEST_FILENAME)
    frame = pd.read_csv(args.submission)
    guess_strings = frame[GUESSES_COLUMN].fillna("").tolist()

    by_length: dict[int, list[bool]] = defaultdict(list)
    remaining_when_lost: list[int] = []
    distinct_letters_when_lost: list[int] = []
    solved_total = 0

    for guess_string, word in zip(guess_strings, test_words):
        state = GameState(word=word)
        for guess in guess_string:
            if state.is_over:
                break
            state.apply_guess(guess)

        by_length[len(word)].append(state.is_solved)
        solved_total += state.is_solved
        if not state.is_solved:
            remaining_when_lost.append(state.board.count("_"))
            distinct_letters_when_lost.append(len(set(word)))

    total = len(test_words)
    print(f"overall win rate: {100 * solved_total / total:.2f}%  ({total:,} words)\n")

    print(f"{'length':>7} {'words':>9} {'share':>7} {'win rate':>10} {'losses':>9}")
    print("-" * 46)
    cumulative_loss = 0
    for length in sorted(by_length):
        outcomes = by_length[length]
        wins = sum(outcomes)
        losses = len(outcomes) - wins
        cumulative_loss += losses
        print(
            f"{length:>7} {len(outcomes):>9,} {100 * len(outcomes) / total:>6.1f}% "
            f"{100 * wins / len(outcomes):>9.1f}% {losses:>9,}"
        )

    print(f"\ntotal losses: {cumulative_loss:,}")

    short = [w for length in range(2, 6) for w in by_length.get(length, [])]
    if short:
        short_losses = len(short) - sum(short)
        print(
            f"words of length 2-5: {len(short):,} ({100 * len(short) / total:.1f}% of the set), "
            f"win rate {100 * sum(short) / len(short):.1f}%, "
            f"{short_losses:,} losses ({100 * short_losses / cumulative_loss:.1f}% of all losses)"
        )

    if remaining_when_lost:
        mean_remaining = sum(remaining_when_lost) / len(remaining_when_lost)
        near_misses = sum(1 for r in remaining_when_lost if r <= 1)
        print(
            f"\non a loss: {mean_remaining:.2f} letters still hidden on average; "
            f"{near_misses:,} losses ({100 * near_misses / len(remaining_when_lost):.1f}%) "
            f"were one letter short"
        )
        print(
            f"every loss used all {MAX_WRONG_GUESSES} lives by definition; mean distinct "
            f"letters in a lost word: "
            f"{sum(distinct_letters_when_lost) / len(distinct_letters_when_lost):.2f}"
        )


if __name__ == "__main__":
    main()
