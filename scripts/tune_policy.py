"""Choose how to blend the model's two heads, measured on held-out words.

    python -m scripts.tune_policy --checkpoint artifacts/run2/best_model.pt

The presence head and the aggregated position head produce scores on different
scales, so the right blend is an empirical question rather than something to
assume. This sweeps ``position_weight`` and reports the competition metrics for
each setting on words the model was never trained on.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.game import play_games, summarise
from hangman.policy import NeuralPolicy
from hangman.train import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-words", type=int, default=5_000)
    parser.add_argument("--validation-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="position_weight values to try. 0 = presence head only, 1 = position only.",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=args.validation_size, seed=args.seed)
    evaluation_words = split.validation[: args.eval_words]

    model = load_model(args.checkpoint, device=device)
    has_position_head = model.position_head is not None
    print(f"checkpoint={args.checkpoint}  position head={has_position_head}")
    print(f"evaluating on {len(evaluation_words):,} held-out words\n")

    print(f"{'position_weight':>16} {'win rate':>10} {'mean wrong':>12} {'seconds':>9}")
    print("-" * 50)

    best = (None, -1.0)
    for weight in args.weights:
        if weight > 0.0 and not has_position_head:
            continue
        policy = NeuralPolicy(model, device=device, position_weight=weight)
        started = time.perf_counter()
        metrics = summarise(play_games(evaluation_words, policy))
        elapsed = time.perf_counter() - started
        print(
            f"{weight:>16.2f} {metrics['win_rate']:>9.2f}% "
            f"{metrics['mean_wrong']:>12.3f} {elapsed:>9.1f}"
        )
        if metrics["win_rate"] > best[1]:
            best = (weight, metrics["win_rate"])

    print(f"\nbest position_weight={best[0]} at {best[1]:.2f}% win rate")


if __name__ == "__main__":
    main()
