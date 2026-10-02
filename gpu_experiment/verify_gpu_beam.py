"""Does the GPU beam reproduce the CPU one, and is it fast enough to matter?

Two questions, in order:

1. **Correctness.** At the same beam width the GPU implementation must give the
   same answers as the CPU one. Anything else means a bug, and a fast wrong
   answer is worse than a slow right one.

2. **Throughput.** The point of moving to the GPU is affording a beam far wider
   than 200. If a wide GPU beam is not much faster than a narrow CPU beam, the
   idea is dead and the existing pipeline stands.

Run from the project root:
    python -m gpu_experiment.verify_gpu_beam
"""

from __future__ import annotations

import time

import numpy as np
import torch

from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.fast_beam import SparseBeamPosterior, build_sparse_table
from hangman.game import GameState
from hangman.policy import NeuralPolicy
from hangman.train import load_model
from hangman.blend_policy import BlendPolicy

from gpu_experiment.gpu_beam import GpuBeamPosterior

ORDER = 6          # order 6 keeps the table at 141 MB while the CPU run is busy


def collect_states(n_words: int = 400) -> tuple[list[str], list[frozenset]]:
    """Real game states, gathered by actually playing with the shipped policy."""
    split = split_words(load_words(DEFAULT_DATA_DIR / TRAIN_FILENAME),
                        validation_size=20_000, seed=20260901)
    words = split.validation[:n_words]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("artifacts/run6/best_model.pt", device=device)
    neural = NeuralPolicy(model, device=device, position_weight=0.5)

    lm = CharKN(order=ORDER).fit(split.train)
    index, table = build_sparse_table(lm)
    lm.release()
    beam = SparseBeamPosterior(index, table, ORDER, beam_width=200, max_blanks=10)
    policy = BlendPolicy(neural, beam, blend_weight=0.38)

    boards, guessed = [], []
    states = [GameState(word=w) for w in words]
    active = [s for s in states if not s.is_over]
    while active:
        obs = [s.observation for s in active]
        for o in obs:
            boards.append(o.board)
            guessed.append(o.guessed_letters)
        for s, g in zip(active, policy.next_guesses(obs)):
            s.apply_guess(g)
        active = [s for s in active if not s.is_over]
    return boards, guessed, index, table


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable; this experiment needs a GPU.")
        return

    print("collecting real game states...")
    boards, guessed, index, table = collect_states()
    print(f"  {len(boards):,} states\n")

    cpu = SparseBeamPosterior(index, table, ORDER, beam_width=200, max_blanks=10)
    gpu = GpuBeamPosterior(index, table, ORDER, beam_width=200, max_blanks=10)

    # --- 1. correctness at matched width -------------------------------
    sample = list(range(0, len(boards), max(1, len(boards) // 600)))[:600]
    sb = [boards[i] for i in sample]
    sg = [guessed[i] for i in sample]

    t0 = time.perf_counter()
    ref = np.stack([cpu.hit_probabilities(b, g)[0] for b, g in zip(sb, sg)])
    t_cpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = gpu.hit_probabilities_batch(sb, sg)
    t_gpu = time.perf_counter() - t0

    active = ref.sum(axis=1) > 0
    diff = np.abs(ref - got)[active]
    agree = (ref[active].argmax(1) == got[active].argmax(1)).mean()
    print(f"1. CORRECTNESS at beam_width=200 on {active.sum()} live states")
    print(f"   max |diff|        {diff.max():.3e}")
    print(f"   mean |diff|       {diff.mean():.3e}")
    print(f"   top-letter agree  {100*agree:.2f}%")
    ok = diff.max() < 1e-4 and agree > 0.999
    print(f"   {'PASS' if ok else '*** FAIL ***'}\n")

    # --- 2. throughput at widths the CPU cannot afford ------------------
    print("2. THROUGHPUT (600 states)")
    print(f"   CPU  width  200 : {t_cpu*1000:>8.0f} ms")
    print(f"   GPU  width  200 : {t_gpu*1000:>8.0f} ms   ({t_cpu/t_gpu:.1f}x)")
    for width in (1000, 5000):
        wide = GpuBeamPosterior(index, table, ORDER, beam_width=width,
                                max_blanks=10)
        wide.hit_probabilities_batch(sb[:32], sg[:32])      # warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wide.hit_probabilities_batch(sb, sg)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        print(f"   GPU  width {width:>4} : {t*1000:>8.0f} ms   "
              f"(projected 250k words: {t/len(sb)*2.2e6/60:.0f} min)")


if __name__ == "__main__":
    main()
