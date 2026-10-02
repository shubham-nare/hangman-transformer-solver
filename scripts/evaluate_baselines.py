"""Measure the reference policies on a held-out slice of the training words.

Run from the project root:

    python -m scripts.evaluate_baselines --sample 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hangman.baselines import PatternMatchingPolicy, StaticFrequencyPolicy, presence_order
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.game import play_games, summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Validation words to evaluate. The pattern matcher is O(vocabulary) "
        "per turn, so a sample keeps the reference run quick.",
    )
    parser.add_argument("--validation-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(
        words, validation_size=args.validation_size, seed=args.seed
    )
    evaluation_words = split.validation[: args.sample]
    print(f"{split}  ->  evaluating on {len(evaluation_words):,} held-out words\n")

    policies = {
        "static frequency": StaticFrequencyPolicy(presence_order(split.train)),
        "pattern matching": PatternMatchingPolicy(split.train),
    }

    print(f"{'policy':<20} {'win rate':>10} {'mean wrong':>12} {'seconds':>9}")
    print("-" * 54)
    for name, policy in policies.items():
        started = time.perf_counter()
        metrics = summarise(play_games(evaluation_words, policy))
        elapsed = time.perf_counter() - started
        print(
            f"{name:<20} {metrics['win_rate']:>9.2f}% "
            f"{metrics['mean_wrong']:>12.3f} {elapsed:>9.1f}"
        )


if __name__ == "__main__":
    main()
