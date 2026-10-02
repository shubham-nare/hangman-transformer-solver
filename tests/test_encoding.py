"""Tests for state encoding, including equivalence to the reference loop.

The vectorised encoder replaced a straightforward per-character loop. Speed is
worthless if it changed behaviour, so the original implementation is kept here
as an executable specification and the fast path is checked against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hangman.encoding import (
    LETTER_TOKEN_OFFSET,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    encode_boards,
    encode_observations,
)
from hangman.game import ALPHABET, MASK_CHAR, GameState

MAX_LENGTH = 32
_LETTER_TO_TOKEN = {letter: LETTER_TOKEN_OFFSET + i for i, letter in enumerate(ALPHABET)}


def reference_encode(boards, max_length=MAX_LENGTH):
    """The original per-character implementation, kept as the specification."""
    batch = len(boards)
    tokens = np.full((batch, max_length), PAD_TOKEN_ID, dtype=np.int64)
    padding = np.ones((batch, max_length), dtype=bool)
    for row, board in enumerate(boards):
        for column, char in enumerate(board):
            if char == MASK_CHAR:
                tokens[row, column] = MASK_TOKEN_ID
            else:
                tokens[row, column] = _LETTER_TO_TOKEN.get(char, MASK_TOKEN_ID)
        padding[row, : len(board)] = False
    return tokens, padding


BOARD_CASES = [
    ["_____"],
    ["ba_a_a", "____", "z"],
    ["_" * 29],
    ["a" * 32],
    # Non-letters: the rules permit them even though the public data has none.
    ["co_a-co_a", "_1 _", "___!"],
    ["_a_", "b__", "___", "abc"],
]


@pytest.mark.parametrize("boards", BOARD_CASES)
def test_vectorised_encoding_matches_the_reference_loop(boards) -> None:
    guessed = np.zeros((len(boards), len(ALPHABET)), dtype=np.float32)
    result = encode_boards(boards, guessed, max_length=MAX_LENGTH)

    expected_tokens, expected_padding = reference_encode(boards)
    np.testing.assert_array_equal(result["tokens"], expected_tokens)
    np.testing.assert_array_equal(result["padding"], expected_padding)


def test_encoding_handles_an_empty_batch() -> None:
    result = encode_boards([], np.zeros((0, len(ALPHABET)), dtype=np.float32))
    assert result["tokens"].shape == (0, 32)
    assert result["padding"].shape == (0, 32)


def test_board_longer_than_max_length_is_rejected() -> None:
    guessed = np.zeros((1, len(ALPHABET)), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds max_length"):
        encode_boards(["a" * 40], guessed, max_length=32)


def test_guessed_mask_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="guessed_masks"):
        encode_boards(["abc"], np.zeros((2, len(ALPHABET)), dtype=np.float32))


def test_observations_encode_their_guess_history() -> None:
    state = GameState(word="banana")
    state.apply_guess("a")
    state.apply_guess("z")

    batch = encode_observations([state.observation])
    guessed = batch["guessed"][0].numpy()

    assert guessed[ALPHABET.index("a")] == 1.0
    assert guessed[ALPHABET.index("z")] == 1.0
    assert guessed.sum() == 2.0

    tokens = batch["tokens"][0].numpy()
    # "_a_a_a": positions 1, 3, 5 revealed as 'a'.
    assert tokens[0] == MASK_TOKEN_ID
    assert tokens[1] == _LETTER_TO_TOKEN["a"]
    assert bool(batch["padding"][0][6].item()) is True


def test_repeated_guesses_do_not_corrupt_the_mask() -> None:
    """A duplicate guess is recorded twice in history but is still one letter."""
    state = GameState(word="banana")
    state.apply_guess("a")
    state.apply_guess("a")

    guessed = encode_observations([state.observation])["guessed"][0].numpy()
    assert guessed[ALPHABET.index("a")] == 1.0
    assert guessed.sum() == 1.0
