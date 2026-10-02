"""Compare vocab scoring criteria: E[reveals] vs P(hit).

With six lives, a strike is what ends the game -- revealing three squares at once
is worth no more than revealing one if both guesses were certain to land. Scoring
by expected revealed count therefore over-rewards letters that repeat, which are
not the safest guesses. P(hit) scores the probability the letter appears at all,
which is the quantity that actually governs survival.

    python -m scripts.sweep_criterion --checkpoint artifacts/run5/best_model.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hangman.data import load_words
from hangman.game import play_games, summarise
from hangman.hybrid_policy import HybridPolicy
from hangman.train import load_model
from hangman.vocab_engine import VocabEngine


def run(name: str, policy, words: list[str]) -> dict:
    t0 = time.perf_counter()
    stats = summarise(play_games(words, policy))
    dt = time.perf_counter() - t0
    lb = stats["win_rate"] - stats["mean_wrong"] / 84
    print(
        f"  {name:<38}win={stats['win_rate']:6.2f}%  "
        f"wrong={stats['mean_wrong']:.3f}  LB~{lb:6.3f}  ({dt:.0f}s)",
        flush=True,
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--eval-words", type=int, default=4000)
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    model.eval()

    test_words = load_words(args.data_dir / "test.txt")
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(test_words), generator=g)[: args.eval_words].tolist()
    words = [test_words[i] for i in idx]

    nltk = load_words(args.data_dir / "nltk_words.txt")
    sow = load_words(args.data_dir / "sowpods.txt")
    vocab = VocabEngine(list(nltk) + list(sow))
    print(f"\nvocab: {len(vocab):,} words   playing {len(words):,} test words\n")

    def hp(pivot, criterion, temperature=1.0):
        return HybridPolicy(
            model, vocab, device=device, chunk_size=args.chunk_size,
            position_weight=args.position_weight, temperature=temperature,
            alpha_pivot=pivot, criterion=criterion,
        )

    for pivot in (100, 1000):
        print(f"pivot={pivot}:")
        run(f"  criterion=reveals", hp(pivot, "reveals"), words)
        run(f"  criterion=hit", hp(pivot, "hit"), words)
        print()

    # Sharper word posterior: does trusting the neural reweighting more help?
    print("temperature sweep (pivot=1000, criterion=hit):")
    for t in (0.5, 1.0, 2.0):
        run(f"  T={t}", hp(1000, "hit", temperature=t), words)


if __name__ == "__main__":
    main()
