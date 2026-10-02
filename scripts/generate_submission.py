"""Play every test word with a trained model and write ``submission.csv``.

    python -m scripts.generate_submission \\
        --checkpoint artifacts/run6/best_model.pt \\
        --use-blend --lm-order 7 --beam-width 200 \\
        --max-blanks 10 --blend-weight 0.50 --position-weight 0.5

Only competition data is ever read: the transformer and the character language
model are both fitted on ``train.txt``, and ``test.txt`` supplies the words to
play and nothing else.

The games are simulated exactly as the grader describes: the model sees only the
board and its own guess history, and each game stops at the sixth strike or on
completion. The written file is then re-scored by an independent replay, so the
number printed here is the number the leaderboard will compute.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hangman.data import DEFAULT_DATA_DIR, TEST_FILENAME, load_words
from hangman.game import play_games, summarise
from hangman.policy import NeuralPolicy
from hangman.submission import validate_submission, write_submission
from hangman.train import load_model


def build_neural(args, model, device):
    """The neural branch: one model, or several averaged at the score level.

    Averaging log-scores rather than voting on final picks keeps how confident
    each model was, which a vote discards. Returned type is interchangeable --
    both expose ``score`` and ``position_log_probs`` -- so the caller does not
    care how many models are behind it.
    """
    primary = NeuralPolicy(
        model, device=device, chunk_size=args.chunk_size,
        position_weight=args.position_weight,
    )
    if not args.extra_checkpoints:
        return primary

    from hangman.policy import EnsemblePolicy

    members = [primary]
    for path in args.extra_checkpoints:
        members.append(
            NeuralPolicy(
                load_model(path, device=device), device=device,
                chunk_size=args.chunk_size, position_weight=args.position_weight,
            )
        )
    weights = args.ensemble_weights
    if weights and len(weights) != len(members):
        raise SystemExit(
            f"--ensemble-weights has {len(weights)} values for {len(members)} models."
        )
    return EnsemblePolicy(members, weights=weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--extra-checkpoints",
        nargs="*",
        type=Path,
        default=[],
        help="Additional checkpoints to ensemble with --checkpoint. Letter "
             "log-scores are averaged, which keeps each model's confidence "
             "rather than discarding it the way a majority vote would.",
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights for [checkpoint, *extra-checkpoints]. Defaults "
             "to equal weighting.",
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Play only the first N test words. For quick checks; a real "
        "submission needs all of them.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--position-weight", type=float, default=0.5)
    parser.add_argument(
        "--use-blend",
        action="store_true",
        help=(
            "Blend the transformer with a Kneser-Ney character-LM beam posterior. "
            "Both are trained on train.txt only, so this is rules-compliant."
        ),
    )
    parser.add_argument("--lm-order", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=200)
    parser.add_argument("--max-blanks", type=int, default=10)
    parser.add_argument(
        "--exact-below", type=int, default=0,
        help="Enumerate the hypothesis space exhaustively when it holds fewer "
             "completions than this, instead of pruning to --beam-width. At "
             "three blanks the beam keeps 200 of ~5,832 candidates (3.4%%), "
             "which measurably changes 1.6%% of decisions. 6000 covers that "
             "case and costs roughly 4.5x the runtime. 0 disables.",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.5,
        help="Weight on the character-LM branch. Fit on held-out train words.",
    )
    parser.add_argument(
        "--min-word-length", type=int, default=0,
        help="Ignore the character LM for words shorter than this. Measured "
             "per-length, the blend loses ~9.5 points at lengths 3-5 and only "
             "pays from length 8: with train/test disjoint, the LM's posterior "
             "on a short board concentrates on frequent short words that cannot "
             "be the answer. 8 is the measured crossover.",
    )
    parser.add_argument(
        "--nn-weight",
        type=float,
        default=0.0,
        help="Weight on the transformer's per-position evidence inside the beam "
             "search (early fusion). 0 = late fusion only.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_words = load_words(args.data_dir / TEST_FILENAME)
    if args.limit:
        test_words = test_words[: args.limit]

    model = load_model(args.checkpoint, device=device)
    print(f"device={device}  words={len(test_words):,}  checkpoint={args.checkpoint}")
    for extra in args.extra_checkpoints:
        print(f"  + ensemble member: {extra}")

    if args.use_blend:
        from hangman.blend_policy import BlendPolicy
        from hangman.char_lm import CharKN
        from hangman.fast_beam import (
            FastBeamPosterior,
            SparseBeamPosterior,
            build_sparse_table,
            build_sparse_table_order7,
            build_table,
        )

        # Competition data only: the LM is fitted on train.txt, the same file the
        # transformer was trained on. No external word list is involved.
        lm_words = load_words(args.data_dir / "train.txt")
        print(f"fitting CharKN(order={args.lm_order}) on {len(lm_words):,} train words...")
        t0 = time.perf_counter()
        lm = CharKN(order=args.lm_order).fit(lm_words)
        print(f"  fitted in {time.perf_counter()-t0:.0f}s")

        neural = build_neural(args, model, device)
        # Order 6+ has 27**5 possible contexts (~1.5 GB dense) but only ~1.7% of
        # them ever occur, so it uses the two-level sparse table instead.
        t0 = time.perf_counter()
        if args.lm_order >= 6:
            # Order 7's context space is 27**6 = 387M, too large to reach a
            # dense fallback in one hop -- the level below it is itself 14.3M
            # rows. It needs the chunked three-level builder; the two-level one
            # would allocate a 3.1 GB int64 arange and then spend ~20 minutes in
            # a Python loop building the dense base.
            builder = (
                build_sparse_table_order7 if args.lm_order >= 7 else build_sparse_table
            )
            print(f"  building sparse table ({builder.__name__})...")
            index, sparse_table = builder(lm)
            mem = (index.nbytes + sparse_table.nbytes) / 1e6
            print(
                f"  sparse table={sparse_table.shape}  {mem:.0f} MB  "
                f"in {time.perf_counter()-t0:.0f}s"
            )
            beam = SparseBeamPosterior(
                index, sparse_table, args.lm_order,
                beam_width=args.beam_width, max_blanks=args.max_blanks,
                exact_below=args.exact_below,
            )
            lm.release()  # counts + memo cache are dead once the table exists
        else:
            print("  building dense table...")
            table = build_table(lm)
            print(f"  table {table.shape}  {table.nbytes/1e6:.0f} MB  in {time.perf_counter()-t0:.0f}s")
            beam = FastBeamPosterior(
                table, args.lm_order,
                beam_width=args.beam_width, max_blanks=args.max_blanks,
                exact_below=args.exact_below,
            )
            lm.release()  # counts + memo cache are dead once the table exists
        policy = BlendPolicy(
            neural, beam,
            blend_weight=args.blend_weight, nn_weight=args.nn_weight,
            min_word_length=args.min_word_length,
        )
        print(
            f"Using BlendPolicy (neural + char-LM beam; weight={args.blend_weight}, "
            f"order={args.lm_order}, beam={args.beam_width}, max_blanks={args.max_blanks})"
        )
    else:
        policy = build_neural(args, model, device)
        kind = "EnsemblePolicy" if args.extra_checkpoints else "NeuralPolicy"
        print(f"Using {kind} (pure neural, no vocab)")

    started = time.perf_counter()
    results = play_games(test_words, policy, progress=True)
    elapsed = time.perf_counter() - started

    metrics = summarise(results)
    leaderboard = metrics["win_rate"] - metrics["mean_wrong"] / 84
    print(
        f"\nplayed in {elapsed / 60:.1f} min  "
        f"({len(test_words) / max(elapsed, 1e-9):,.0f} words/s)"
    )
    print(f"win rate    : {metrics['win_rate']:.4f}%")
    print(f"total wrong : {int(metrics['total_wrong']):,}")
    print(f"mean wrong  : {metrics['mean_wrong']:.4f}")
    print(f"LB score~   : {leaderboard:.4f}")

    expected_rows = None if args.limit else 250_000
    write_submission(results, args.output, expected_rows=expected_rows)
    print(f"\nwrote {args.output}")

    report = validate_submission(args.output, test_words, expected_rows=expected_rows)
    print(f"\n{report}")
    if not report.is_valid:
        raise SystemExit("Submission failed validation; not safe to upload.")


if __name__ == "__main__":
    main()

