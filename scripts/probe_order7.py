"""Is order 7 worth the engineering? Probe it before building the table.

Order 7 was the perplexity optimum in the order scan (6.261 vs 6.377 at order 6)
but its dense context space is 27**6 = 387M rows, so using it needs a
three-level sparse table that does not exist yet. Rather than build that on the
strength of a perplexity number, this measures win rate directly.

The trick is that the reference :class:`BeamPosterior` queries the language model
through ``logprob_char`` and needs no table at all. It costs ~3.8 ms per call
instead of ~0.15 ms, which is far too slow for a 250k-word submission but
perfectly affordable on a few thousand words -- and it is the same inference,
so the comparison is honest.

Held-out train words only; test.txt is not read.

    python -m scripts.probe_order7 --checkpoint artifacts/run6/best_model.pt
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from hangman.beam_solver import BeamPosterior
from hangman.blend_policy import BlendPolicy
from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
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
    ap.add_argument("--eval-words", type=int, default=2500)
    ap.add_argument("--orders", type=int, nargs="+", default=[6, 7])
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
    print(f"playing {len(words):,} HELD-OUT TRAIN words via the reference "
          f"(table-free) beam\n")

    model = load_model(args.checkpoint, device=device)
    neural = NeuralPolicy(model, device=device, position_weight=args.position_weight)

    outcomes: dict[int, tuple[np.ndarray, dict]] = {}
    for order in args.orders:
        t0 = time.perf_counter()
        lm = CharKN(order=order).fit(split.train)
        ppl = lm.perplexity(split.validation[:3000])
        seen = len(lm._counts[order - 1])
        print(f"order {order}: ppl={ppl:.3f}  observed_contexts={seen:,}  "
              f"dense_rows={27**(order-1):,}  (fit {time.perf_counter()-t0:.0f}s)")

        beam = BeamPosterior(lm, beam_width=args.beam_width,
                             max_blanks=args.max_blanks)
        policy = BlendPolicy(neural, beam, blend_weight=args.blend_weight)
        t0 = time.perf_counter()
        results = play_games(words, policy)
        stats = summarise(results)
        solved = np.array([r.solved for r in results], dtype=bool)
        print(f"  -> win={stats['win_rate']:6.2f}%  wrong={stats['mean_wrong']:.3f}  "
              f"({time.perf_counter()-t0:.0f}s)\n", flush=True)
        outcomes[order] = (solved, stats)

    if len(args.orders) >= 2:
        base, *rest = args.orders
        base_solved, base_stats = outcomes[base]
        print(f"paired comparisons vs order {base}:")
        for order in rest:
            solved, stats = outcomes[order]
            wins, losses, p = mcnemar(base_solved, solved)
            delta = stats["win_rate"] - base_stats["win_rate"]
            verdict = ("SIGNIFICANT" if p < 0.05 else "indistinguishable")
            print(f"  order {order}: delta={delta:+.2f}pts  won={wins} lost={losses}  "
                  f"p={p:.4f}  {verdict}")
        print("\nA positive, significant delta justifies building the "
              "three-level sparse table; anything else does not.")


if __name__ == "__main__":
    main()
