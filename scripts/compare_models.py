"""Score several checkpoints, and their ensemble, on one identical word set.

    python -m scripts.compare_models --checkpoints artifacts/run2/best_model.pt ...

Numbers recorded during training are not directly comparable across runs: they
came from whatever eval slice that run was configured with. This re-scores every
checkpoint on the same held-out words with the same engine, which is the only way
a difference of a few tenths of a point means anything.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.game import play_games, summarise
from hangman.policy import EnsemblePolicy, NeuralPolicy
from hangman.train import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-words", type=int, default=10_000)
    parser.add_argument("--validation-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--position-weight", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Also score the ensemble of all given checkpoints.",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=args.validation_size, seed=args.seed)
    evaluation_words = split.validation[: args.eval_words]
    print(f"scoring on {len(evaluation_words):,} identical held-out words\n")

    policies: list[NeuralPolicy] = []
    print(f"{'checkpoint':<34} {'params':>9} {'win rate':>10} {'mean wrong':>12} {'sec':>6}")
    print("-" * 76)

    for path in args.checkpoints:
        model = load_model(path, device=device)
        policy = NeuralPolicy(
            model,
            device=device,
            chunk_size=args.chunk_size,
            position_weight=args.position_weight,
        )
        policies.append(policy)

        started = time.perf_counter()
        metrics = summarise(play_games(evaluation_words, policy))
        elapsed = time.perf_counter() - started
        name = str(path.parent.name if path.name == "best_model.pt" else path)
        print(
            f"{name:<34} {model.count_parameters() / 1e6:>8.1f}M "
            f"{metrics['win_rate']:>9.2f}% {metrics['mean_wrong']:>12.3f} {elapsed:>6.0f}"
        )

    if args.ensemble and len(policies) > 1:
        started = time.perf_counter()
        metrics = summarise(play_games(evaluation_words, EnsemblePolicy(policies)))
        elapsed = time.perf_counter() - started
        print("-" * 76)
        print(
            f"{'ENSEMBLE of ' + str(len(policies)):<34} {'':>9} "
            f"{metrics['win_rate']:>9.2f}% {metrics['mean_wrong']:>12.3f} {elapsed:>6.0f}"
        )


if __name__ == "__main__":
    main()
