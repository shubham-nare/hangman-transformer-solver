"""Non-neural reference policies.

These exist to establish the bar, not to be submitted. The competition weights
learned models above "simple N-Gram or statistical frequency baselines", so the
neural solver has to beat these convincingly for the approach to be justified.

Two baselines, in increasing order of strength:

* :class:`StaticFrequencyPolicy` -- a fixed letter order, ignoring the board.
  This is what ``sample_submission.csv`` does.
* :class:`PatternMatchingPolicy` -- filter the training vocabulary down to the
  words consistent with the board, then guess the letter appearing in the most
  survivors. Strong on paper, but it can only ever recognise spellings it has
  already seen, and train/test here are disjoint.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .game import ALPHABET, MASK_CHAR, Observation

# Letters ordered by P(letter appears in a word), measured on train.txt.
# This differs from raw token frequency -- what matters in Hangman is whether a
# letter is present at all, not how often it repeats.
TRAIN_PRESENCE_ORDER: str = "eiarnostlcudpmhgybfvkwzxqj"


def presence_order(words: Sequence[str]) -> str:
    """Rank letters by the fraction of ``words`` containing them at least once."""
    counts = Counter()
    for word in words:
        counts.update(set(word))
    ranked = sorted(ALPHABET, key=lambda letter: -counts[letter])
    return "".join(ranked)


class StaticFrequencyPolicy:
    """Guess a fixed letter order, skipping letters already guessed.

    Ignores the board entirely, so it is the weakest sensible reference point.
    """

    def __init__(self, order: str = TRAIN_PRESENCE_ORDER) -> None:
        self.order = order

    def next_guesses(self, observations: Sequence[Observation]) -> Sequence[str]:
        return [self._guess(obs) for obs in observations]

    def _guess(self, observation: Observation) -> str:
        guessed = observation.guessed_letters
        for letter in self.order:
            if letter not in guessed:
                return letter
        return self.order[-1]


class PatternMatchingPolicy:
    """Constrain the training vocabulary to the board, then vote on letters.

    A word is consistent with the board when, for every letter already guessed,
    its occurrence positions exactly match the board's. That single condition
    covers hits and misses at once: a correct guess reveals *all* of its
    occurrences, so a masked position can never hold an already-guessed letter.

    Words are bucketed by length and stored as integer matrices so each turn is
    a handful of vectorised comparisons rather than a Python loop over the
    vocabulary.
    """

    def __init__(self, words: Sequence[str], *, fallback_order: str | None = None) -> None:
        self.fallback_order = fallback_order or presence_order(words)
        self._codes: dict[int, np.ndarray] = {}
        self._presence: dict[int, np.ndarray] = {}

        by_length: dict[int, list[str]] = {}
        for word in words:
            if set(word) <= set(ALPHABET):
                by_length.setdefault(len(word), []).append(word)

        for length, bucket in by_length.items():
            codes = np.frombuffer("".join(bucket).encode("ascii"), dtype=np.uint8)
            codes = codes.reshape(len(bucket), length)
            self._codes[length] = codes
            # presence[i, j] -> does word i contain letter j at least once
            presence = np.zeros((len(bucket), len(ALPHABET)), dtype=bool)
            for index, letter in enumerate(ALPHABET):
                presence[:, index] = (codes == ord(letter)).any(axis=1)
            self._presence[length] = presence

    def next_guesses(self, observations: Sequence[Observation]) -> Sequence[str]:
        return [self._guess(obs) for obs in observations]

    def _guess(self, observation: Observation) -> str:
        guessed = observation.guessed_letters
        candidates = self._consistent_words(observation)

        if candidates is not None and candidates.size:
            presence = self._presence[observation.length][candidates]
            scores = presence.sum(axis=0).astype(np.float64)
            for letter in guessed:
                if letter in ALPHABET:
                    scores[ALPHABET.index(letter)] = -1.0
            best = int(np.argmax(scores))
            if scores[best] > 0:
                return ALPHABET[best]

        # No consistent word survived: fall back to corpus-wide frequency.
        for letter in self.fallback_order:
            if letter not in guessed:
                return letter
        return self.fallback_order[-1]

    def _consistent_words(self, observation: Observation) -> np.ndarray | None:
        """Indices of vocabulary words compatible with the board, or ``None``."""
        codes = self._codes.get(observation.length)
        if codes is None:
            return None

        board = np.frombuffer(observation.board.encode("ascii"), dtype=np.uint8)
        keep = np.ones(len(codes), dtype=bool)
        for letter in observation.guessed_letters:
            if letter not in ALPHABET:
                continue
            board_positions = board == ord(letter)
            word_positions = codes == ord(letter)
            keep &= ~(word_positions != board_positions).any(axis=1)
            if not keep.any():
                return np.empty(0, dtype=np.int64)

        # Masked positions must not hold a letter that is already revealed.
        revealed = board != ord(MASK_CHAR)
        if revealed.any():
            keep &= (codes[:, revealed] == board[revealed]).all(axis=1)

        return np.flatnonzero(keep)
