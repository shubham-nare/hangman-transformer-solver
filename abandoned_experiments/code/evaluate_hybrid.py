"""Ablation study: Neural vs Hybrid vs Vocab-only policies.

Usage
-----
python -m scripts.evaluate_hybrid \\
    --checkpoint artifacts/run5/best_model.pt \\
    --train-words data/train.txt \\
    --eval-words 5000 \\
    --alpha-pivot 50 \\
    --temperature 1.0

The script evaluates four policies on the same held-out validation words:
  A. Neural only          (existing NeuralPolicy)
  B. Vocab only           (uniform word prior, no model)
  C. Hybrid (freq)        (blend neural + vocab letter frequency, no neural scoring)
  D. Hybrid (full)        (blend neural + E[reveals] under neural word posterior)

Reports win-rate and mean wrong guesses for each.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hangman.data import load_words, split_words
from hangman.game import play_games, summarise
from hangman.model import HangmanTransformer
from hangman.policy import NeuralPolicy
from hangman.hybrid_policy import HybridPolicy
from hangman.vocab_engine import VocabEngine
from hangman.train import load_model


# --------------------------------------------------------------------------- #
# Vocab-only policy (no neural model)                                          #
# --------------------------------------------------------------------------- #

from hangman.game import ALPHABET, Observation

class VocabOnlyPolicy:
    """Baseline: no model — pure vocabulary candidate filtering + letter frequency."""

    def __init__(self, vocab: VocabEngine) -> None:
        self.vocab = vocab

    def next_guesses(self, observations):
        guesses = []
        for obs in observations:
            wrong = frozenset(g for g in obs.guesses if g.islower() and g not in obs.board)
            Q, n = self.vocab.score_letters(
                pattern=obs.board,
                wrong_letters=wrong,
                guessed_letters=obs.guessed_letters,
                position_log_probs=None,  # uniform prior
            )
            if n == 0:
                # Fall back: guess most common unguessed letter
                for ch in "etaoinshrdlcumwfgypbvkjxqz":
                    if ch not in obs.guessed_letters:
                        guesses.append(ch)
                        break
                else:
                    guesses.append("e")
            else:
                guesses.append(ALPHABET[int(Q.argmax())])
        return guesses


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def run_policy(name, policy, words, verbose=True):
    t0 = time.time()
    results = play_games(words, policy)
    dt = time.time() - t0
    stats = summarise(results)
    leaderboard_score = stats["win_rate"] - stats["mean_wrong"] / 84
    if verbose:
        print(
            f"  {name:<30}  win={stats['win_rate']:6.2f}%  "
            f"mean_wrong={stats['mean_wrong']:.3f}  "
            f"LB~{leaderboard_score:6.3f}  "
            f"({dt:.1f}s)"
        )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-words", default="data/train.txt")
    ap.add_argument("--eval-words", type=int, default=5000)
    ap.add_argument("--alpha-pivot", type=float, default=50.0,
                    help="n_candidates where alpha=0.5 (vocab/neural crossover)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for word posterior")
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    model.eval()

    print(f"Loading training vocabulary: {args.train_words}")
    all_words = load_words(Path(args.train_words))

    # Use the train/val split so val words are NOT in the vocab index.
    # This gives an honest (slightly pessimistic) evaluation of vocab coverage.
    word_split = split_words(all_words, validation_size=20_000)
    train_vocab_words = word_split.train
    val_words = word_split.validation

    print(f"  Train vocab: {len(train_vocab_words):,} words")
    print(f"  Val set:     {len(val_words):,} words")

    # Sample evaluation words
    rng = torch.Generator()
    rng.manual_seed(args.seed)
    idx = torch.randperm(len(val_words), generator=rng)[: args.eval_words].tolist()
    eval_words = [val_words[i] for i in idx]
    print(f"  Evaluating on {len(eval_words):,} held-out words\n")

    # Build vocabulary engine from TRAIN split only (val words are OOV)
    print("Building vocabulary engine...")
    t0 = time.time()
    vocab = VocabEngine(train_vocab_words)
    print(f"  Built in {time.time()-t0:.1f}s  ({len(vocab):,} words indexed)\n")

    neural = NeuralPolicy(
        model, device=device,
        chunk_size=args.chunk_size,
        position_weight=args.position_weight,
    )
    hybrid_full = HybridPolicy(
        model, vocab, device=device,
        chunk_size=args.chunk_size,
        position_weight=args.position_weight,
        temperature=args.temperature,
        alpha_pivot=args.alpha_pivot,
    )
    vocab_only = VocabOnlyPolicy(vocab)

    # Hybrid-frequency: same as hybrid_full but temperature=1e6 (≈uniform prior)
    hybrid_freq = HybridPolicy(
        model, vocab, device=device,
        chunk_size=args.chunk_size,
        position_weight=args.position_weight,
        temperature=1e6,   # effectively uniform word prior → pure letter frequency
        alpha_pivot=args.alpha_pivot,
    )

    print("=" * 70)
    print("Ablation (held-out val words, same games for all policies):")
    print("=" * 70)
    run_policy("A. Neural only", neural, eval_words)
    run_policy("B. Vocab only (freq)", vocab_only, eval_words)
    run_policy("C. Hybrid (freq blend)", hybrid_freq, eval_words)
    run_policy("D. Hybrid (neural posterior)", hybrid_full, eval_words)
    print("=" * 70)

    # Alpha/pivot sweep
    print(f"\nAlpha-pivot sweep (D-style, T={args.temperature}):")
    for pivot in [20, 50, 100, 200]:
        hp = HybridPolicy(
            model, vocab, device=device,
            chunk_size=args.chunk_size,
            position_weight=args.position_weight,
            temperature=args.temperature,
            alpha_pivot=pivot,
        )
        run_policy(f"  pivot={pivot}", hp, eval_words)


if __name__ == "__main__":
    main()
