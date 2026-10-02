"""RL fine-tuning of a pre-trained Hangman transformer via REINFORCE.

    python -m scripts.rl_train --checkpoint artifacts/run5/best_model.pt

Loads a supervised checkpoint, wraps it in an ActorCritic, and fine-tunes it
by playing real Hangman games on the training vocabulary. Rewards are +1 for
every game won and -1 for every game lost. A value baseline reduces variance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.rl import RLConfig, rl_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Supervised best_model.pt to start RL fine-tuning from.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl"),
        help="Directory for best_model.pt, last_state.pt, training_summary.json.",
    )
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument(
        "--games-per-step",
        type=int,
        default=256,
        help="Games played per gradient update.",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.05,
        help="Starting entropy coefficient (linearly decayed to --entropy-end).",
    )
    parser.add_argument(
        "--entropy-end",
        type=float,
        default=0.001,
        help="Final entropy coefficient at the last RL step.",
    )
    parser.add_argument(
        "--progress-reward",
        type=float,
        default=0.02,
        help="Per-position bonus for each letter revealed (added to terminal +-1).",
    )
    parser.add_argument(
        "--value-coeff",
        type=float,
        default=0.5,
        help="Weight on the value (baseline) loss.",
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-words", type=int, default=2_000)
    parser.add_argument("--validation-size", type=int, default=20_000)
    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.5,
        help="Blend of presence and position heads used at evaluation time.",
    )
    parser.add_argument(
        "--hard-word-weight",
        type=float,
        default=0.5,
        help=(
            "0 = uniform word sampling; 1 = weight by 1/length "
            "(focus on hard short words)."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last_state.pt in --output-dir if it exists.",
    )
    args = parser.parse_args()

    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=args.validation_size, seed=args.seed)
    print(f"{split}\n")

    config = RLConfig(
        steps=args.steps,
        games_per_step=args.games_per_step,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        entropy_end=args.entropy_end,
        value_coeff=args.value_coeff,
        eval_every=args.eval_every,
        eval_words=args.eval_words,
        seed=args.seed,
        position_weight=args.position_weight,
        hard_word_weight=args.hard_word_weight,
        progress_reward=args.progress_reward,
    )

    summary = rl_train(
        split.train,
        split.validation,
        args.checkpoint,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
    )

    print(f"\nbest held-out win rate: {summary['best_win_rate']:.2f}%")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"elapsed: {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
