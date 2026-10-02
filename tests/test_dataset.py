"""Invariant tests for simulated training-state generation.

A bug here is invisible: training still converges, the loss still falls, and the
model quietly learns from states that could never occur in a real game. These
tests assert the sampled states are ones the engine could actually produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from hangman.dataset import IGNORE_INDEX, GameStateSampler
from hangman.encoding import LETTER_TOKEN_OFFSET, MASK_TOKEN_ID, PAD_TOKEN_ID
from hangman.game import ALPHABET, MAX_WRONG_GUESSES

WORDS = [
    "banana",
    "hangman",
    "transformer",
    "aardvark",
    "quixotic",
    "spectropyrheliometer",
    "ox",
    "zzz",
]
MAX_LENGTH = 32


@pytest.fixture
def sampler() -> GameStateSampler:
    return GameStateSampler(WORDS, max_length=MAX_LENGTH, seed=7)


def _decode_board(tokens: np.ndarray, padding: np.ndarray) -> str:
    chars = []
    for token, is_pad in zip(tokens, padding):
        if is_pad:
            continue
        chars.append("_" if token == MASK_TOKEN_ID else ALPHABET[token - LETTER_TOKEN_OFFSET])
    return "".join(chars)


def test_board_length_and_padding_match_the_word(sampler: GameStateSampler) -> None:
    indices = np.arange(len(WORDS))
    batch = sampler.sample_for_words(indices)

    for row, word in enumerate(WORDS):
        assert (~batch.padding[row]).sum() == len(word)
        assert (batch.tokens[row][batch.padding[row]] == PAD_TOKEN_ID).all()
        assert len(_decode_board(batch.tokens[row], batch.padding[row])) == len(word)


def test_revealed_positions_agree_with_the_word_and_guessed_mask(
    sampler: GameStateSampler,
) -> None:
    """Every visible character must be a guessed letter, in its true position."""
    indices = np.repeat(np.arange(len(WORDS)), 40)
    batch = sampler.sample_for_words(indices)

    for row, word_index in enumerate(indices):
        word = WORDS[word_index]
        board = _decode_board(batch.tokens[row], batch.padding[row])
        guessed = {ALPHABET[i] for i in np.flatnonzero(batch.guessed[row])}

        for position, (board_char, true_char) in enumerate(zip(board, word)):
            if board_char == "_":
                # A hidden letter can never be one that was already guessed:
                # a correct guess reveals every occurrence at once.
                assert true_char not in guessed, (
                    f"{word!r} position {position}: {true_char!r} hidden despite being guessed"
                )
            else:
                assert board_char == true_char
                assert board_char in guessed


def test_targets_are_a_distribution_over_hidden_letters_only(
    sampler: GameStateSampler,
) -> None:
    indices = np.repeat(np.arange(len(WORDS)), 40)
    batch = sampler.sample_for_words(indices)

    for row, word_index in enumerate(indices):
        word = WORDS[word_index]
        guessed = {ALPHABET[i] for i in np.flatnonzero(batch.guessed[row])}
        hidden = set(word) - guessed

        assert batch.targets[row].sum() == pytest.approx(1.0, abs=1e-5)
        scored = {ALPHABET[i] for i in np.flatnonzero(batch.targets[row])}
        assert scored == hidden, f"{word!r}: scored {scored} but hidden letters are {hidden}"


def test_every_sampled_state_is_still_winnable(sampler: GameStateSampler) -> None:
    """Sampling stops at the terminal pre-guess state, so a state is never dead."""
    indices = np.repeat(np.arange(len(WORDS)), 60)
    batch = sampler.sample_for_words(indices)

    for row, word_index in enumerate(indices):
        word = WORDS[word_index]
        guessed = {ALPHABET[i] for i in np.flatnonzero(batch.guessed[row])}
        misses = len(guessed - set(word))
        assert misses < MAX_WRONG_GUESSES
        assert set(word) - guessed, "a sampled state must have at least one letter left"


def test_targets_weight_letters_by_how_often_they_are_hidden() -> None:
    """'banana' with only 'b' revealed should weight 'a' above 'n' (3 vs 2)."""
    sampler = GameStateSampler(["banana"], max_length=MAX_LENGTH, seed=1)
    batch = sampler.sample_for_words(np.zeros(200, dtype=np.int64))

    a_index, n_index = ALPHABET.index("a"), ALPHABET.index("n")
    both_hidden = (batch.targets[:, a_index] > 0) & (batch.targets[:, n_index] > 0)
    assert both_hidden.any()
    rows = np.flatnonzero(both_hidden)
    assert (batch.targets[rows, a_index] > batch.targets[rows, n_index]).all()


def test_letter_targets_supervise_exactly_the_blanks(sampler: GameStateSampler) -> None:
    """The language-model head must be scored on hidden positions and nowhere else."""
    indices = np.repeat(np.arange(len(WORDS)), 40)
    batch = sampler.sample_for_words(indices)

    for row, word_index in enumerate(indices):
        word = WORDS[word_index]
        board = _decode_board(batch.tokens[row], batch.padding[row])
        targets = batch.letter_targets[row]

        for position in range(len(word)):
            if board[position] == "_":
                assert targets[position] == ALPHABET.index(word[position])
            else:
                assert targets[position] == IGNORE_INDEX

        # Padding is never supervised.
        assert (targets[len(word) :] == IGNORE_INDEX).all()


def test_letter_targets_are_ignored_when_nothing_is_hidden() -> None:
    """A fully revealed board would be a terminal state and is never sampled."""
    sampler = GameStateSampler(["banana"], max_length=MAX_LENGTH, seed=5)
    batch = sampler.sample_for_words(np.zeros(50, dtype=np.int64))
    supervised = (batch.letter_targets != IGNORE_INDEX).sum(axis=1)
    assert (supervised > 0).all()


def test_uniform_order_probability_widens_state_coverage() -> None:
    """A flat ranking must reach states a frequency player would rarely visit."""
    frequency_only = GameStateSampler(
        WORDS, max_length=MAX_LENGTH, uniform_order_probability=0.0, seed=3
    )
    uniform_only = GameStateSampler(
        WORDS, max_length=MAX_LENGTH, uniform_order_probability=1.0, seed=3
    )

    indices = np.repeat(np.arange(len(WORDS)), 50)
    rare = set("qjxz")

    def rare_guess_rate(sampler: GameStateSampler) -> float:
        batch = sampler.sample_for_words(indices)
        rare_indices = [ALPHABET.index(letter) for letter in rare]
        return float(batch.guessed[:, rare_indices].mean())

    assert rare_guess_rate(uniform_only) > rare_guess_rate(frequency_only)
