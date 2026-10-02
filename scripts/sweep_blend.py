"""Fit the neural/beam blend weight on HELD-OUT TRAIN words.

test.txt is never read here. The blend weight is a hyper-parameter, and fitting
it on the evaluation set would be exactly the kind of tuning that fails to
generalise to a different word list.

    python -m scripts.sweep_blend --checkpoint artifacts/run5/best_model.pt
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hangman.beam_solver import BeamPosterior
from hangman.blend_policy import BlendPolicy
from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.game import play_games, summarise
from hangman.policy import NeuralPolicy
from hangman.train import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--eval-words", type=int, default=1000)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--beam-width", type=int, default=200)
    ap.add_argument("--max-blanks", type=int, default=8)
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=20_000, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(split.validation), size=args.eval_words, replace=False)
    eval_words = [split.validation[i] for i in idx]

    print(f"\nfitting CharKN(order={args.order})...")
    lm = CharKN(order=args.order).fit(split.train)
    print(f"  held-out ppl {lm.perplexity(split.validation[:2000]):.3f}")

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    neural = NeuralPolicy(model, device=device, position_weight=args.position_weight)
    beam = BeamPosterior(lm, beam_width=args.beam_width, max_blanks=args.max_blanks)

    print(f"playing {len(eval_words):,} HELD-OUT TRAIN words per weight")
    print(f"beam_width={args.beam_width}  max_blanks={args.max_blanks}\n")
    print(f"{'weight':>8}{'win%':>9}{'mean_wrong':>13}{'secs':>8}")
    best = (-1.0, None)
    per_len: dict[float, dict[int, list[int]]] = {}
    for w in args.weights:
        policy = BlendPolicy(neural, beam, blend_weight=w)
        t0 = time.perf_counter()
        results = play_games(eval_words, policy)
        dt = time.perf_counter() - t0
        stats = summarise(results)
        buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for r in results:
            buckets[len(r.word)][0] += 1
            buckets[len(r.word)][1] += int(r.solved)
        per_len[w] = buckets
        print(f"{w:>8.2f}{stats['win_rate']:>8.2f}%{stats['mean_wrong']:>13.3f}{dt:>8.0f}")
        if stats["win_rate"] > best[0]:
            best = (stats["win_rate"], w)

    print(f"\nbest weight: {best[1]}  ({best[0]:.2f}%)")

    lo, hi = args.weights[0], best[1]
    print(f"\nper-length, weight={lo} vs weight={hi}:")
    print(f"{'len':>4}{'n':>6}{'w=%.2f' % lo:>10}{'w=%.2f' % hi:>10}{'delta':>9}")
    for L in sorted(per_len[lo]):
        n0, w0 = per_len[lo][L]
        n1, w1 = per_len[hi][L]
        if n0 < 12:
            continue
        p0, p1 = 100 * w0 / n0, 100 * w1 / n1
        print(f"{L:>4}{n0:>6}{p0:>9.1f}%{p1:>9.1f}%{p1-p0:>+9.1f}")


if __name__ == "__main__":
    main()
