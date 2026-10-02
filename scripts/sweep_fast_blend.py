"""Tune the neural + character-LM blend, using the vectorised beam.

Every number here comes from HELD-OUT TRAIN words. test.txt is not opened. The
held-out slice is disjoint from the words the LM and the transformer were fitted
on, which is the same relationship test.txt has to train.txt -- so a setting
chosen here is chosen on the right kind of evidence and should transfer.

    python -m scripts.sweep_fast_blend --checkpoint artifacts/run5/best_model.pt
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hangman.blend_policy import BlendPolicy
from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.fast_beam import (
    FastBeamPosterior,
    SparseBeamPosterior,
    build_sparse_table,
    build_sparse_table_order7,
    build_table,
)
from hangman.game import play_games, summarise
from hangman.policy import NeuralPolicy
from hangman.train import load_model


def run(policy, words, label):
    t0 = time.perf_counter()
    results = play_games(words, policy)
    dt = time.perf_counter() - t0
    stats = summarise(results)
    buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        buckets[len(r.word)][0] += 1
        buckets[len(r.word)][1] += int(r.solved)
    print(
        f"  {label:<34}win={stats['win_rate']:6.2f}%  "
        f"wrong={stats['mean_wrong']:.3f}  ({dt:.0f}s)",
        flush=True,
    )
    return stats, buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--extra-checkpoints", nargs="*", default=[],
        help="Ensemble these with --checkpoint; letter log-scores are averaged.",
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--eval-words", type=int, default=3000)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument(
        "--lm-train-words", type=int, default=None,
        help="Fit the language model on this many training words instead of the "
             "whole split. The shipped LM sees 225,300 words while this sweep "
             "normally gives it 205,300, and blend weight is exactly the "
             "parameter that trades neural against LM -- so it is measured "
             "under a weaker model than the one that ships. Varying this shows "
             "which way the optimum moves with LM quality.",
    )
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    ap.add_argument("--beam-widths", type=int, nargs="+", default=[200])
    ap.add_argument(
        "--min-word-lengths", type=int, nargs="+", default=[0],
        help="Ignore the character LM for words shorter than this. The blend "
             "measurably HURTS short words: -9.5 points at lengths 3-5, "
             "crossing over to positive only from length 8.",
    )
    ap.add_argument(
        "--exact-below", type=int, nargs="+", default=[0],
        help="Enumerate the hypothesis space exhaustively when it is smaller "
             "than this, instead of pruning to beam width. 0 disables.",
    )
    ap.add_argument(
        "--position-weights", type=float, nargs="+", default=None,
        help="Sweep the presence/position head blend too. Defaults to the "
             "single --position-weight value.",
    )
    ap.add_argument("--max-blanks", type=int, nargs="+", default=[10])
    ap.add_argument(
        "--nn-weights", type=float, nargs="+", default=[0.0],
        help="Weight on the transformer's per-position evidence inside the beam "
             "search. 0 = late fusion (blend only after both models finish).",
    )
    ap.add_argument(
        "--blank-pivots", type=float, nargs="+", default=[None],
        help="Blank count below which the beam gets full weight; above it the "
             "weight decays as pivot/blanks. Omit for a constant weight.",
    )
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=20_000, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(split.validation), size=args.eval_words, replace=False)
    eval_words = [split.validation[i] for i in idx]

    print(f"\nfitting CharKN(order={args.order}) on {len(split.train):,} words...")
    lm_words = split.train
    if args.lm_train_words and args.lm_train_words < len(lm_words):
        pick = np.random.default_rng(args.seed).choice(
            len(lm_words), size=args.lm_train_words, replace=False
        )
        lm_words = [lm_words[i] for i in pick]
        print(f"  LM fitted on a {len(lm_words):,}-word subset "
              f"(deployed LM sees 225,300)")
    lm = CharKN(order=args.order).fit(lm_words)
    print(f"  held-out ppl {lm.perplexity(split.validation[:2000]):.3f}")
    # Order 6+ cannot be densified (27**5 rows), so it uses the two-level table.
    t0 = time.perf_counter()
    if args.order >= 6:
        builder = (
            build_sparse_table_order7 if args.order >= 7 else build_sparse_table
        )
        print(f"building sparse table ({builder.__name__})...")
        index, sparse_table = builder(lm)
        mem = (index.nbytes + sparse_table.nbytes) / 1e6
        print(
            f"  sparse  table={sparse_table.shape}  {mem:.0f} MB  "
            f"in {time.perf_counter()-t0:.0f}s\n"
        )
        make_beam = lambda bw, mb, ex: SparseBeamPosterior(
            index, sparse_table, args.order,
            beam_width=bw, max_blanks=mb, exact_below=ex,
        )
    else:
        print("building dense table...")
        table = build_table(lm)
        print(f"  {table.shape}  {table.nbytes/1e6:.0f} MB  in {time.perf_counter()-t0:.0f}s\n")
        make_beam = lambda bw, mb, ex: FastBeamPosterior(
            table, args.order, beam_width=bw, max_blanks=mb, exact_below=ex,
        )

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)

    def make_neural(pw: float):
        """Neural branch at a given presence/position blend.

        Rebuilt per position_weight because it changes how the two heads are
        combined -- and that setting couples to blend_weight, since both govern
        how much the neural branch contributes. Tuning one without the other is
        how blend_weight ended up stale across two LM upgrades.
        """
        primary = NeuralPolicy(model, device=device, position_weight=pw)
        if not args.extra_checkpoints:
            return primary
        from hangman.policy import EnsemblePolicy

        members = [primary] + [
            NeuralPolicy(load_model(p, device=device), device=device,
                         position_weight=pw)
            for p in args.extra_checkpoints
        ]
        return EnsemblePolicy(members)

    position_weights = args.position_weights or [args.position_weight]

    print(f"playing {len(eval_words):,} HELD-OUT TRAIN words per config\n")
    best = (-1.0, None)
    results_by_len: dict[str, dict] = {}

    for pw in position_weights:
        neural = make_neural(pw)
        for ex in args.exact_below:
            for mb in args.max_blanks:
                for bw in args.beam_widths:
                    beam = make_beam(bw, mb, ex)
                    for pivot in args.blank_pivots:
                        for nn in args.nn_weights:
                            print(
                                f"pos_w={pw}  exact_below={ex}  max_blanks={mb}  "
                                f"beam_width={bw}  blank_pivot={pivot}  nn_weight={nn}:"
                            )
                            for w in args.weights:
                              for mwl in args.min_word_lengths:
                                policy = BlendPolicy(
                                    neural, beam, blend_weight=w,
                                    blank_pivot=pivot, nn_weight=nn,
                                    min_word_length=mwl,
                                )
                                label = f"  weight={w:.2f} min_len={mwl}"
                                stats, buckets = run(policy, eval_words, label)
                                key = (f"pw{pw}_ex{ex}_mb{mb}_bw{bw}"
                                       f"_p{pivot}_nn{nn}_w{w}_mwl{mwl}")
                                results_by_len[key] = buckets
                                if stats["win_rate"] > best[0]:
                                    best = (stats["win_rate"], key)
                            print()

    print(f"BEST: {best[1]}  ->  {best[0]:.2f}%")

    base = (
        f"pw{position_weights[0]}_ex{args.exact_below[0]}"
        f"_mb{args.max_blanks[0]}_bw{args.beam_widths[0]}"
        f"_p{args.blank_pivots[0]}_nn{args.nn_weights[0]}_w{args.weights[0]}"
    )
    if base in results_by_len and best[1] in results_by_len:
        print(f"\nper-length: neural-only vs {best[1]}")
        print(f"{'len':>4}{'n':>7}{'neural':>10}{'best':>9}{'delta':>9}")
        b0, b1 = results_by_len[base], results_by_len[best[1]]
        for L in sorted(b0):
            n0, w0 = b0[L]
            n1, w1 = b1[L]
            if n0 < 25:
                continue
            p0, p1 = 100 * w0 / n0, 100 * w1 / n1
            print(f"{L:>4}{n0:>7}{p0:>9.1f}%{p1:>8.1f}%{p1-p0:>+9.1f}")


if __name__ == "__main__":
    main()
