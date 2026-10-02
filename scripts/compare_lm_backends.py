"""Neural Transformer LM vs Kneser-Ney, judged on win rate rather than perplexity.

Perplexity measures how well a model assigns absolute probabilities. The beam
search does not need that -- it needs the *ranking* of candidate completions to
be right, and then only the induced ranking over 26 letters. A model can be
worse at perplexity and no worse at the decision that actually scores points, so
the two backends are compared by playing games.

Both feed the identical beam search: :class:`FastBeamPosterior` consumes a dense
``(n_contexts, 27)`` log-probability table and does not care what produced it.
Only the table differs, so this isolates the language model.

Everything is measured on held-out train words; test.txt is not read.

    python -m scripts.compare_lm_backends --checkpoint artifacts/run6/best_model.pt
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
from hangman.fast_beam import FastBeamPosterior, build_table
from hangman.game import play_games, summarise
from hangman.neural_lm import NeuralCharLM, NeuralLMConfig, tabulate
from hangman.policy import NeuralPolicy
from hangman.train import load_model


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Paired significance on per-word outcomes; see overnight_finish."""
    import math

    b_only = int(np.sum(b & ~a))
    a_only = int(np.sum(a & ~b))
    n = b_only + a_only
    if n == 0:
        return b_only, a_only, 1.0
    z = abs(b_only - a_only) / np.sqrt(n)
    return b_only, a_only, math.erfc(z / math.sqrt(2.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/run6/best_model.pt")
    ap.add_argument("--neural-lm", type=Path, default=Path("artifacts/neural_lm/neural_lm.pt"))
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

    saved = torch.load(args.neural_lm, map_location=device, weights_only=False)
    config = NeuralLMConfig(**saved["config"])
    neural_lm = NeuralCharLM(config).to(device)
    neural_lm.load_state_dict(saved["model_state"])
    neural_lm.eval()
    print(f"neural LM: context={config.context}  params={neural_lm.count_parameters():,}  "
          f"held-out ppl={saved['held_out_ppl']:.3f}")

    print("tabulating neural LM...")
    t0 = time.perf_counter()
    neural_table = tabulate(neural_lm, device)
    print(f"  {neural_table.shape}  {neural_table.nbytes/1e6:.0f} MB  "
          f"in {time.perf_counter()-t0:.0f}s")

    # Kneser-Ney at the SAME context length, so the comparison isolates the
    # model rather than how much history each one sees.
    kn = CharKN(order=config.context + 1).fit(split.train)
    print(f"CharKN(order={config.context+1}) held-out ppl={kn.perplexity(split.validation[:3000]):.3f}")
    print("tabulating Kneser-Ney...")
    kn_table = build_table(kn)
    kn.release()

    model = load_model(args.checkpoint, device=device)
    policy_neural = NeuralPolicy(model, device=device,
                                 position_weight=args.position_weight)

    def run(label: str, table: np.ndarray) -> tuple[np.ndarray, dict]:
        beam = FastBeamPosterior(
            table, config.context + 1,
            beam_width=args.beam_width, max_blanks=args.max_blanks,
        )
        policy = BlendPolicy(policy_neural, beam, blend_weight=args.blend_weight)
        t0 = time.perf_counter()
        results = play_games(words, policy)
        stats = summarise(results)
        solved = np.array([r.solved for r in results], dtype=bool)
        print(f"  {label:<28}win={stats['win_rate']:6.2f}%  "
              f"wrong={stats['mean_wrong']:.3f}  ({time.perf_counter()-t0:.0f}s)")
        return solved, stats

    print("\nblended win rate, identical beam, only the LM table differs:")
    kn_solved, kn_stats = run("Kneser-Ney backend", kn_table)
    nn_solved, nn_stats = run("Neural Transformer backend", neural_table)

    # A count-based model and a learned one make different errors: Kneser-Ney is
    # near-optimal where a context was observed often, the Transformer degrades
    # gracefully where it was not. Interpolating in log space is a product of
    # experts over the two, renormalised, and costs one weighted sum of arrays
    # already in memory.
    results_by_mix = {}
    for w in (0.3, 0.5, 0.7):
        mixed = (1.0 - w) * kn_table + w * neural_table
        mixed -= np.log(np.exp(mixed).sum(axis=1, keepdims=True))
        solved, stats = run(f"interpolated (neural={w:.1f})", mixed.astype(np.float32))
        results_by_mix[w] = (solved, stats)

    print("\npaired comparisons vs Kneser-Ney:")
    candidates = [("neural", nn_solved, nn_stats)] + [
        (f"mix{w:.1f}", s, st) for w, (s, st) in results_by_mix.items()
    ]
    best = ("kneser-ney", kn_stats["win_rate"])
    for name, solved, stats in candidates:
        wins, losses, p = mcnemar(kn_solved, solved)
        delta = stats["win_rate"] - kn_stats["win_rate"]
        verdict = (
            ("neural-side wins" if delta > 0 else "Kneser-Ney wins")
            if p < 0.05 else "indistinguishable"
        )
        print(f"  {name:<18} delta={delta:+.2f}pts  won={wins} lost={losses}  "
              f"p={p:.4f}  {verdict}")
        if stats["win_rate"] > best[1]:
            best = (name, stats["win_rate"])
    print(f"\nBEST BACKEND: {best[0]} at {best[1]:.2f}%")


if __name__ == "__main__":
    main()
