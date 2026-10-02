"""Tensor encoding of Hangman game states.

A single encoder serves both training and inference so the model cannot see a
different feature layout in the two settings -- a common and silent source of
train/serve skew.

Each state becomes three tensors:

* ``tokens``   -- ``(batch, max_length)`` board tokens, padded.
* ``guessed``  -- ``(batch, 26)`` binary, letters already submitted.
* ``padding``  -- ``(batch, max_length)`` boolean, ``True`` at padding positions.

The ``guessed`` vector is not redundant with the board. A letter that was
guessed and missed never appears on the board, yet knowing it is absent is some
of the most valuable information the player holds -- it eliminates candidate
spellings. Without this input the model would be blind to its own misses.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .game import ALPHABET, MASK_CHAR, Observation

PAD_TOKEN_ID: int = 0
MASK_TOKEN_ID: int = 1
LETTER_TOKEN_OFFSET: int = 2
VOCABULARY_SIZE: int = LETTER_TOKEN_OFFSET + len(ALPHABET)

# The longest word in the corpus is 29 characters; 32 leaves headroom for a
# private evaluation set with slightly longer entries.
MAX_WORD_LENGTH: int = 32

_LETTER_TO_TOKEN = {letter: LETTER_TOKEN_OFFSET + i for i, letter in enumerate(ALPHABET)}
_LETTER_A = ord("a")


def encode_boards(
    boards: Sequence[str],
    guessed_masks: np.ndarray,
    *,
    max_length: int = MAX_WORD_LENGTH,
) -> dict[str, np.ndarray]:
    """Encode board strings and guessed-letter masks into padded arrays.

    Args:
        boards: Board strings, using ``MASK_CHAR`` for hidden letters.
        guessed_masks: ``(batch, 26)`` binary array of already-guessed letters.
        max_length: Padded sequence length.

    Returns:
        Dict of ``tokens`` (int64), ``guessed`` (float32) and ``padding`` (bool).

    Raises:
        ValueError: If a board is longer than ``max_length``, or if the mask
            shape does not match the number of boards.
    """
    batch = len(boards)
    if guessed_masks.shape != (batch, len(ALPHABET)):
        raise ValueError(
            f"guessed_masks must be ({batch}, {len(ALPHABET)}), "
            f"got {guessed_masks.shape}."
        )

    lengths = np.fromiter((len(board) for board in boards), dtype=np.int64, count=batch)
    if batch and lengths.max() > max_length:
        longest = boards[int(lengths.argmax())]
        raise ValueError(
            f"Board of length {lengths.max()} exceeds max_length={max_length}: {longest!r}"
        )

    # Scatter every board into a padded byte matrix in one shot. Encoding one
    # character at a time in Python dominated inference: this runs on every turn
    # of every game, so 250,000 test words meant tens of millions of iterations.
    occupied = np.arange(max_length)[None, :] < lengths[:, None]
    characters = np.zeros((batch, max_length), dtype=np.uint8)
    if batch:
        characters[occupied] = np.frombuffer(
            "".join(boards).encode("ascii"), dtype=np.uint8
        )

    letter_index = characters.astype(np.int16) - _LETTER_A
    is_letter = (letter_index >= 0) & (letter_index < len(ALPHABET))

    # Anything that is not a revealed a-z letter is a blank from the model's
    # point of view. That covers MASK_CHAR and also any non-letter the private
    # set might contain: those are visible from the start and carry no
    # guessable information.
    tokens = np.where(
        is_letter, letter_index.astype(np.int64) + LETTER_TOKEN_OFFSET, MASK_TOKEN_ID
    )
    tokens[~occupied] = PAD_TOKEN_ID

    return {
        "tokens": tokens,
        "guessed": guessed_masks.astype(np.float32),
        "padding": ~occupied,
    }


def encode_observations(
    observations: Sequence[Observation],
    *,
    max_length: int = MAX_WORD_LENGTH,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Encode live :class:`Observation` values into a batch of tensors."""
    guessed = np.zeros((len(observations), len(ALPHABET)), dtype=np.float32)
    for row, observation in enumerate(observations):
        # Iterate the raw guess tuple rather than the `guessed_letters` property,
        # which builds a fresh frozenset on every access -- and this runs once
        # per game per turn.
        for letter in observation.guesses:
            index = ALPHABET.find(letter)
            if index >= 0:
                guessed[row, index] = 1.0

    arrays = encode_boards(
        [observation.board for observation in observations],
        guessed,
        max_length=max_length,
    )
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
