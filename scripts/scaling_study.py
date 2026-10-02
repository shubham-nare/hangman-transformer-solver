"""How do LM order and blend weight scale with training-set size?

Both of our validation failures share one cause: the sweep fits the language
model on 205,300 words while the shipped system fits it on 225,300. Anything
sensitive to LM quality is therefore measured under a systematically weaker
model than the one that ships.

That bias is measurable rather than merely regrettable. Fitting at several
training sizes shows which direction each quantity moves as data grows, and the
trend extrapolates to the deployed size -- a principled estimate that never
touches test.txt.

Two questions:

1. Does order 8's deficit against order 7 shrink with data? Order 8 is the most
   data-starved configuration (563k observed contexts in a 10.5-billion space),
   so if any order gains disproportionately from more words it is that one. A
   deficit that is closing fast might reverse by 225,300.

2. Which way does the optimal blend weight move? Held-out chose 0.26 and the
   test set preferred 0.38, so the sweep's optimum is biased low. If the optimum
   rises with LM data, the deployed weight may belong *above* 0.38.

    python -m scripts.scaling_study
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100_000, 140_000, 175_000, 205_300])
    ap.add_argument("--orders", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--held-out", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    split = split_words(load_words(DEFAULT_DATA_DIR / TRAIN_FILENAME),
                        validation_size=20_000, seed=args.seed)
    held_out = split.validation[: args.held_out]
    rng = np.random.default_rng(args.seed)

    print(f"held-out: {len(held_out):,} words   deployed LM size: 225,300\n")
    header = f"{'train_words':>12}" + "".join(f"{'order ' + str(o):>12}" for o in args.orders)
    print(header)
    print("-" * len(header))

    results: dict[int, dict[int, float]] = {}
    for size in args.sizes:
        idx = rng.choice(len(split.train), size=min(size, len(split.train)),
                         replace=False)
        subset = [split.train[i] for i in idx]
        row = {}
        for order in args.orders:
            lm = CharKN(order=order).fit(subset)
            row[order] = lm.perplexity(held_out)
            lm.release()
        results[size] = row
        print(f"{size:>12,}" + "".join(f"{row[o]:>12.4f}" for o in args.orders),
              flush=True)

    # Does order 8 close on order 7 as data grows?
    if 7 in args.orders and 8 in args.orders:
        print(f"\norder 8 minus order 7 (negative means order 8 is better):")
        print(f"{'train_words':>12}{'gap':>10}")
        gaps = []
        for size in args.sizes:
            gap = results[size][8] - results[size][7]
            gaps.append((size, gap))
            print(f"{size:>12,}{gap:>+10.4f}")

        # Linear fit on the trend, extrapolated to the deployed training size.
        xs = np.array([s for s, _ in gaps], dtype=float)
        ys = np.array([g for _, g in gaps], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        projected = slope * 225_300 + intercept
        print(f"\n  trend: {slope * 10_000:+.5f} per 10k words")
        print(f"  projected gap at 225,300 words: {projected:+.4f}")
        if projected < 0:
            print("  -> order 8 would OVERTAKE order 7 at the deployed size; "
                  "worth building despite the 42 GB dense index")
        else:
            need = (0 - intercept) / slope if slope < 0 else float("inf")
            print(f"  -> order 8 still behind. Crossover would need "
                  f"{need:,.0f} training words." if slope < 0 else
                  "  -> order 8 falls further behind with more data; do not build it.")


if __name__ == "__main__":
    main()
