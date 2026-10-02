"""Does beam-search posterior inference beat the neural marginals, and where?

Evaluated on HELD-OUT TRAIN words only -- test.txt is never read. The split
mirrors the real setting exactly: the evaluation words are absent from the data
the language model was fitted on, just as test words are absent from train.txt.

    python -m scripts.eval_beam --checkpoint artifacts/run5/best_model.pt
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hangman.beam_solver import BeamPosterior
from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.game import ALPHABET, GameState, play_games, summarise
from hangman.policy import NeuralPolicy
from hangman.train import load_model


class BeamPolicy:
    """Pure character-LM policy: argmax posterior P(letter hits) over the beam."""

    def __init__(self, beam: BeamPosterior) -> None:
        self.beam = beam

    def next_guesses(self, observations):
        out = []
        for obs in observations:
            guessed = obs.guessed_letters
            probs, n = self.beam.hit_probabilities(obs.board, guessed)
            if n == 0:
                for ch in "etaoinshrdlcumwfgypbvkjxqz":
                    if ch not in guessed:
                        out.append(ch)
                        break
                else:
                    out.append("e")
            else:
                out.append(ALPHABET[int(np.argmax(probs))])
        return out


def by_length(words, policy):
    """Play and return {length: (n, wins)} plus overall stats."""
    results = play_games(words, policy)
    buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        L = len(r.word)
        buckets[L][0] += 1
        buckets[L][1] += int(r.solved)
    return buckets, summarise(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--eval-words", type=int, default=1500)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--beam-width", type=int, default=400)
    ap.add_argument("--max-blanks", type=int, default=8)
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    words = load_words(args.data_dir / TRAIN_FILENAME)
    split = split_words(words, validation_size=20_000, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(split.validation), size=args.eval_words, replace=False)
    eval_words = [split.validation[i] for i in idx]

    print(f"\nfitting CharKN(order={args.order}) on {len(split.train):,} train words...")
    t0 = time.perf_counter()
    lm = CharKN(order=args.order).fit(split.train)
    print(f"  fitted in {time.perf_counter()-t0:.1f}s")
    print(f"  held-out perplexity: {lm.perplexity(split.validation[:2000]):.3f}\n")

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    neural = NeuralPolicy(model, device=device, position_weight=args.position_weight)
    beam = BeamPolicy(
        BeamPosterior(lm, beam_width=args.beam_width, max_blanks=args.max_blanks)
    )

    print(f"playing {len(eval_words):,} HELD-OUT TRAIN words\n")
    t0 = time.perf_counter()
    nb, ns = by_length(eval_words, neural)
    t_neural = time.perf_counter() - t0
    t0 = time.perf_counter()
    bb, bs = by_length(eval_words, beam)
    t_beam = time.perf_counter() - t0

    print(f"  NEURAL : win={ns['win_rate']:6.2f}%  wrong={ns['mean_wrong']:.3f}  ({t_neural:.0f}s)")
    print(f"  BEAM   : win={bs['win_rate']:6.2f}%  wrong={bs['mean_wrong']:.3f}  ({t_beam:.0f}s)")
    print(f"\n{'len':>4}{'n':>6}{'neural%':>10}{'beam%':>9}{'delta':>9}")
    for L in sorted(set(nb) | set(bb)):
        n_n, w_n = nb[L]
        n_b, w_b = bb[L]
        if n_n < 15:
            continue
        pn = 100 * w_n / max(n_n, 1)
        pb = 100 * w_b / max(n_b, 1)
        flag = "  <<<" if pb - pn > 5 else ("  >>>" if pn - pb > 5 else "")
        print(f"{L:>4}{n_n:>6}{pn:>9.1f}%{pb:>8.1f}%{pb-pn:>+9.1f}{flag}")


if __name__ == "__main__":
    main()
