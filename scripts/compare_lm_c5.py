"""Context-5 showdown: Transformer character LM vs Kneser-Ney, judged on win rate.

At five characters of context only ~1.7% of the 14.3M possible contexts are ever
observed, so Kneser-Ney must back off for the rest while the Transformer predicts
everywhere. This is the regime where a learned model should have the advantage,
and perplexity says the gap has already narrowed from 6.0% to 2.9%. Perplexity is
not the objective though -- the beam needs the induced ranking over 26 letters to
be right, nothing more -- so the two are compared by playing games.

Memory note: a dense context-5 table is 1.4 GB. Kneser-Ney is therefore kept in
its two-level sparse form (141 MB) rather than densified, which also avoids a
~20-minute rebuild. Both posterior classes expose the same interface, so the beam
and the blend are byte-identical across the two arms; only the language model
differs.

    python -m scripts.compare_lm_c5 --checkpoint artifacts/run6/best_model.pt
"""

from __future__ import annotations

import argparse
import math
import time
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
)
from hangman.game import play_games, summarise
from hangman.neural_lm import NeuralCharLM, NeuralLMConfig, tabulate
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
    ap.add_argument("--neural-lm", type=Path,
                    default=Path("artifacts/neural_lm_c5/neural_lm.pt"))
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
    neural_policy = NeuralPolicy(model, device=device,
                                 position_weight=args.position_weight)

    def run(label, beam):
        policy = BlendPolicy(neural_policy, beam, blend_weight=args.blend_weight)
        t0 = time.perf_counter()
        results = play_games(words, policy)
        stats = summarise(results)
        solved = np.array([r.solved for r in results], dtype=bool)
        print(f"  {label:<30}win={stats['win_rate']:6.2f}%  "
              f"wrong={stats['mean_wrong']:.3f}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        return solved, stats

    # --- Kneser-Ney arm, sparse to keep memory in reach -------------------
    print("building Kneser-Ney order-6 (sparse)...")
    t0 = time.perf_counter()
    kn = CharKN(order=6).fit(split.train)
    kn_ppl = kn.perplexity(split.validation[:3000])
    index, kn_table = build_sparse_table(kn)
    kn.release()
    print(f"  ppl={kn_ppl:.3f}  {(index.nbytes+kn_table.nbytes)/1e6:.0f} MB  "
          f"in {time.perf_counter()-t0:.0f}s")
    kn_beam = SparseBeamPosterior(index, kn_table, 6,
                                  beam_width=args.beam_width,
                                  max_blanks=args.max_blanks)
    kn_solved, kn_stats = run("Kneser-Ney order-6", kn_beam)
    del index, kn_table, kn_beam   # free before the 1.4 GB neural table

    # --- Neural arm --------------------------------------------------------
    saved = torch.load(args.neural_lm, map_location=device, weights_only=False)
    config = NeuralLMConfig(**saved["config"])
    lm = NeuralCharLM(config).to(device)
    lm.load_state_dict(saved["model_state"])
    lm.eval()
    print(f"\nneural LM: context={config.context}  "
          f"params={lm.count_parameters():,}  ppl={saved['held_out_ppl']:.3f}")
    print("tabulating...")
    t0 = time.perf_counter()
    nn_table = tabulate(lm, device)
    print(f"  {nn_table.shape}  {nn_table.nbytes/1e9:.2f} GB  "
          f"in {time.perf_counter()-t0:.0f}s")
    nn_beam = FastBeamPosterior(nn_table, config.context + 1,
                                beam_width=args.beam_width,
                                max_blanks=args.max_blanks)
    nn_solved, nn_stats = run("Neural Transformer LM", nn_beam)

    wins, losses, p = mcnemar(kn_solved, nn_solved)
    delta = nn_stats["win_rate"] - kn_stats["win_rate"]
    print(f"\nneural vs Kneser-Ney at context 5:")
    print(f"  delta={delta:+.2f} pts   newly_won={wins}  newly_lost={losses}  p={p:.4f}")
    if p < 0.05:
        print("  -> SIGNIFICANT:", "neural wins" if delta > 0 else "Kneser-Ney wins")
    else:
        print("  -> INDISTINGUISHABLE. The Transformer LM costs nothing "
              "measurable, so the pipeline can be end-to-end neural for free.")


if __name__ == "__main__":
    main()
