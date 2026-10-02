"""Train the Hangman transformer on the provided training vocabulary.

    python -m scripts.train_model --steps 12000 --batch-size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.model import ModelConfig
from hangman.train import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-words", type=int, default=2_000)
    parser.add_argument("--validation-size", type=int, default=20_000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--position-loss-weight",
        type=float,
        default=1.0,
        help="Weight on the per-blank language-model loss. 0 disables the head.",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.5,
        help="Inference blend of the two heads, used for evaluation during training.",
    )
    parser.add_argument(
        "--self-play-start-step",
        type=int,
        default=0,
        help="Step at which to begin mixing in DAgger self-play states. 0 disables.",
    )
    parser.add_argument("--self-play-refresh-every", type=int, default=4_000)
    parser.add_argument("--self-play-words", type=int, default=20_000)
    parser.add_argument("--self-play-fraction", type=float, default=0.5)
    parser.add_argument(
        "--presence-objective",
        choices=["count", "binary"],
        default="count",
        help="'count' weights letters by repetition; 'binary' scores presence only.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Exponential moving average decay for weight averaging. 0 disables.",
    )
    parser.add_argument("--ema-warmup-steps", type=int, default=1_000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last_state.pt in the output directory, if present.",
    )
    args = parser.parse_args()

    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=args.validation_size, seed=args.seed)
    print(f"{split}\n")

    model_config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_position_head=args.position_loss_weight > 0.0,
    )
    training_config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        eval_words=args.eval_words,
        seed=args.seed,
        position_loss_weight=args.position_loss_weight,
        position_weight=args.position_weight,
        self_play_start_step=args.self_play_start_step,
        self_play_refresh_every=args.self_play_refresh_every,
        self_play_words=args.self_play_words,
        self_play_fraction=args.self_play_fraction,
        presence_objective=args.presence_objective,
        ema_decay=args.ema_decay,
        ema_warmup_steps=args.ema_warmup_steps,
    )

    _, summary = train(
        split.train,
        split.validation,
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
    )

    print(f"\nbest held-out win rate: {summary['best_win_rate']:.2f}%")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"elapsed: {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
