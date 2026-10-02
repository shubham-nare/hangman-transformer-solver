"""Order 6 vs order 7, judged on win rate with a paired test.

Order 7 has the better perplexity (6.196 vs 6.377) but perplexity is not the
objective -- the beam needs the induced ranking over 26 letters to be right. The
earlier order 5 -> 6 step bought +0.8 to +1.1 points from an 8.6% perplexity
gain; this step is 2.8%, so the expected effect is small and worth measuring
rather than assuming.

Both arms share an identical beam, blend weight, policy and word list. Only the
language-model table differs.

Memory: the two tables are 0.14 GB and 1.68 GB, so the order-6 arm runs first
and is freed before order 7 is built.

    python -m scripts.compare_order67 --checkpoint artifacts/run6/best_model.pt
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from hangman.blend_policy import BlendPolicy
from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.fast_beam import (
    SparseBeamPosterior,
    build_sparse_table,
    build_sparse_table_order7,
)
from hangman.game import play_games, summarise
from hangman.policy import NeuralPolicy
from hangman.train import load_model


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    b_only = int(np.sum(b & ~a))
    a_only = int(np.sum(a & ~b))
    n = b_only + a_only
    if n == 0:
        return b_only, a_only, 1.0
    z = abs(b_only - a_only) / math.sqrt(n)
    return b_only, a_only, math.erfc(z / math.sqrt(2.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/run6/best_model.pt")
    ap.add_argument("--eval-words", type=int, default=8000)
    ap.add_argument("--blend-weight", type=float, default=0.38)
    ap.add_argument("--beam-width", type=int, default=200)
    ap.add_argument("--max-blanks", type=int, default=10)
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    split = split_words(
        load_words(DEFAULT_DATA_DIR / TRAIN_FILENAME),
        validation_size=20_000, seed=args.seed,
    )
    words = split.validation[: args.eval_words]
    print(f"playing {len(words):,} HELD-OUT TRAIN words (test.txt untouched)\n")

    model = load_model(args.checkpoint, device=device)
    neural = NeuralPolicy(model, device=device, position_weight=args.position_weight)

    def play(label, beam):
        policy = BlendPolicy(neural, beam, blend_weight=args.blend_weight)
        t0 = time.perf_counter()
        results = play_games(words, policy)
        stats = summarise(results)
        solved = np.array([r.solved for r in results], dtype=bool)
        print(f"  {label:<22}win={stats['win_rate']:6.2f}%  "
              f"wrong={stats['mean_wrong']:.3f}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        return solved, stats

    print("order 6:")
    lm6 = CharKN(order=6).fit(split.train)
    print(f"  ppl={lm6.perplexity(split.validation[:3000]):.3f}")
    idx6, tab6 = build_sparse_table(lm6)
    lm6.release()
    s6, st6 = play("order 6", SparseBeamPosterior(
        idx6, tab6, 6, beam_width=args.beam_width, max_blanks=args.max_blanks))
    del idx6, tab6

    print("\norder 7:")
    lm7 = CharKN(order=7).fit(split.train)
    print(f"  ppl={lm7.perplexity(split.validation[:3000]):.3f}")
    t0 = time.perf_counter()
    idx7, tab7 = build_sparse_table_order7(lm7)
    lm7.release()
    print(f"  table {(idx7.nbytes+tab7.nbytes)/1e9:.2f} GB in {time.perf_counter()-t0:.0f}s")
    s7, st7 = play("order 7", SparseBeamPosterior(
        idx7, tab7, 7, beam_width=args.beam_width, max_blanks=args.max_blanks))

    wins, losses, p = mcnemar(s6, s7)
    delta = st7["win_rate"] - st6["win_rate"]
    print(f"\norder 7 vs order 6:")
    print(f"  delta={delta:+.2f} pts   newly_won={wins}  newly_lost={losses}  p={p:.4f}")
    if p < 0.05 and delta > 0:
        print("  -> order 7 is SIGNIFICANTLY better; regenerate the submission")
    elif p < 0.05:
        print("  -> order 6 is SIGNIFICANTLY better; keep it")
    else:
        print("  -> indistinguishable; keep order 6 (smaller table, already shipped)")


if __name__ == "__main__":
    main()
