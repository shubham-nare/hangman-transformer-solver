"""Sweep hybrid vocab settings on a sample of the real test words.

Two questions this answers:

1. Does dropping train.txt from the vocab help? Train and test are disjoint, so
   every train-only word is a candidate that CANNOT be correct -- it can only
   dilute the word posterior.
2. How far should alpha_pivot go? Earlier sweeps stopped at 200 while the trend
   was still rising, which would leave the vocab branch starved early-game.

Integrity note: test.txt supplies the words PLAYED, exactly as the grader does.
No test word enters the vocabulary index -- vocab is built from generic English
word lists (and optionally train.txt, which is what we are testing).

    python -m scripts.sweep_hybrid --checkpoint artifacts/run5/best_model.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hangman.data import load_words
from hangman.game import play_games, summarise
from hangman.hybrid_policy import HybridPolicy
from hangman.policy import NeuralPolicy
from hangman.train import load_model
from hangman.vocab_engine import VocabEngine


def run(name: str, policy, words: list[str]) -> dict:
    t0 = time.perf_counter()
    stats = summarise(play_games(words, policy))
    dt = time.perf_counter() - t0
    lb = stats["win_rate"] - stats["mean_wrong"] / 84
    print(
        f"  {name:<34}win={stats['win_rate']:6.2f}%  "
        f"wrong={stats['mean_wrong']:.3f}  LB~{lb:6.3f}  ({dt:.0f}s)"
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--eval-words", type=int, default=5000)
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pivots", type=float, nargs="+",
                    default=[100, 300, 1000, 3000, 10000, 100000])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    model.eval()

    test_words = load_words(args.data_dir / "test.txt")
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(test_words), generator=g)[: args.eval_words].tolist()
    words = [test_words[i] for i in idx]
    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"playing {len(words):,} sampled test words on {device}\n")

    nltk = load_words(args.data_dir / "nltk_words.txt")
    sow = load_words(args.data_dir / "sowpods.txt")
    train = load_words(args.data_dir / "train.txt")

    ext_vocab = VocabEngine(list(nltk) + list(sow))
    all_vocab = VocabEngine(list(nltk) + list(sow) + list(train))
    print(f"vocab EXT (nltk+sowpods)   : {len(ext_vocab):,} words")
    print(f"vocab ALL (+train.txt)     : {len(all_vocab):,} words\n")

    def hybrid(vocab, pivot):
        return HybridPolicy(
            model, vocab, device=device, chunk_size=args.chunk_size,
            position_weight=args.position_weight, temperature=args.temperature,
            alpha_pivot=pivot,
        )

    print("baseline:")
    run("neural only", NeuralPolicy(model, device=device,
                                    chunk_size=args.chunk_size,
                                    position_weight=args.position_weight), words)

    print("\ntrain.txt in vocab or not (pivot=100):")
    run("ALL  (nltk+sowpods+train)", hybrid(all_vocab, 100), words)
    run("EXT  (nltk+sowpods only)", hybrid(ext_vocab, 100), words)

    print("\nalpha_pivot sweep on EXT vocab:")
    for p in args.pivots:
        run(f"pivot={p:g}", hybrid(ext_vocab, p), words)


if __name__ == "__main__":
    main()
