"""Loading and splitting the competition word lists.

The corpus is an English dictionary: 225,300 training words and 250,000 test
words, lowercase ``a-z`` only, with **zero overlap** between the two. Nothing can
be memorised, so every design choice here is aimed at measuring generalisation
to unseen spellings.

Model selection uses a held-out slice of ``train.txt``. The public ``test.txt``
ships with its answers, which means a local replay reproduces the public
leaderboard score exactly -- so it is reserved for final reporting and kept out
of the tuning loop, where it would otherwise invite overfitting to the public
split at the expense of the private one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .game import ALPHABET

DEFAULT_DATA_DIR = Path("data")
TRAIN_FILENAME = "train.txt"
TEST_FILENAME = "test.txt"


def load_words(path: str | Path) -> list[str]:
    """Read a newline-delimited word list, preserving file order.

    File order matters for ``test.txt``: ``word_id`` in the submission is the
    zero-based line index, so the list must not be reordered or deduplicated.
    """
    with open(path, encoding="utf-8") as handle:
        return [line.strip().lower() for line in handle if line.strip()]


@dataclass(frozen=True)
class WordSplit:
    """A train/validation partition of the training vocabulary."""

    train: list[str]
    validation: list[str]

    def __str__(self) -> str:
        return (
            f"WordSplit(train={len(self.train):,}, validation={len(self.validation):,})"
        )


def split_words(
    words: Sequence[str],
    *,
    validation_size: int = 20_000,
    seed: int = 20260901,
) -> WordSplit:
    """Randomly partition ``words`` into fitting and model-selection sets.

    The split is random rather than alphabetical: ``train.txt`` is sorted, so a
    positional split would put entire prefix families (every ``un-`` word, say)
    on one side and make validation measure the wrong thing.
    """
    if validation_size >= len(words):
        raise ValueError(
            f"validation_size={validation_size} must be smaller than the "
            f"{len(words)} available words."
        )

    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    return WordSplit(
        train=shuffled[validation_size:],
        validation=shuffled[:validation_size],
    )


def describe_vocabulary(words: Sequence[str]) -> dict[str, object]:
    """Summarise a word list: size, length range, and any out-of-alphabet characters.

    The competition rules allow non-letter characters in the vocabulary even
    though the public corpus contains none. This surfaces that discrepancy
    immediately if the private evaluation set differs.
    """
    lengths = [len(word) for word in words]
    charset = set("".join(words))
    return {
        "count": len(words),
        "unique": len(set(words)),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "charset_size": len(charset),
        "non_alphabet_characters": sorted(charset - set(ALPHABET)),
    }
