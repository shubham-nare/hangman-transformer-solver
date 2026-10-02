"""Tests for DAgger self-play state collection and mixing."""

from __future__ import annotations

import numpy as np
import pytest

from hangman.dataset import IGNORE_INDEX, GameStateSampler, encode_supervised_states
from hangman.encoding import LETTER_TOKEN_OFFSET, MASK_TOKEN_ID
from hangman.game import ALPHABET, MAX_WRONG_GUESSES
from hangman.model import HangmanTransformer, ModelConfig
from hangman.selfplay import MixedStateSampler, collect_self_play_states

WORDS = ["banana", "hangman", "transformer", "quixotic", "aardvark"]
MAX_LENGTH = 32
SMALL = ModelConfig(d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, dropout=0.0)


def test_encode_supervised_states_labels_only_the_blanks() -> None:
    guessed = np.zeros((1, len(ALPHABET)), dtype=np.float32)
    guessed[0, ALPHABET.index("a")] = 1.0

    batch = encode_supervised_states(
        ["banana"], ["_a_a_a"], guessed, max_length=MAX_LENGTH
    )

    # Hidden positions are 0 ('b'), 2 ('n'), 4 ('n').
    assert batch.letter_targets[0, 0] == ALPHABET.index("b")
    assert batch.letter_targets[0, 2] == ALPHABET.index("n")
    assert batch.letter_targets[0, 1] == IGNORE_INDEX
    assert (batch.letter_targets[0, 6:] == IGNORE_INDEX).all()

    # Two hidden 'n' against one hidden 'b'.
    assert batch.targets[0, ALPHABET.index("n")] == pytest.approx(2 / 3)
    assert batch.targets[0, ALPHABET.index("b")] == pytest.approx(1 / 3)
    assert batch.targets[0, ALPHABET.index("a")] == 0.0


def test_self_play_states_are_reachable_and_correctly_labelled() -> None:
    """Collected states must satisfy the same invariants as simulated ones."""
    model = HangmanTransformer(SMALL)
    batch = collect_self_play_states(
        model, WORDS, device="cpu", max_length=MAX_LENGTH
    )

    assert len(batch) > 0
    for row in range(len(batch)):
        guessed = {ALPHABET[i] for i in np.flatnonzero(batch.guessed[row])}
        supervised = batch.letter_targets[row]
        hidden = {ALPHABET[i] for i in supervised[supervised != IGNORE_INDEX]}

        # A guessed letter can never still be hidden: a hit reveals all of it.
        assert not (hidden & guessed)
        # Every recorded state is mid-game, so something is still hidden.
        assert hidden
        assert batch.targets[row].sum() == pytest.approx(1.0, abs=1e-5)


def _reconstruct_word_letters(batch, row: int) -> set[str]:
    """Recover the whole word: revealed letters from tokens, hidden from targets."""
    letters: set[str] = set()
    for column in range(batch.tokens.shape[1]):
        if batch.padding[row, column]:
            continue
        token = batch.tokens[row, column]
        if token == MASK_TOKEN_ID:
            letters.add(ALPHABET[batch.letter_targets[row, column]])
        else:
            letters.add(ALPHABET[token - LETTER_TOKEN_OFFSET])
    return letters


def test_self_play_never_records_a_state_past_termination() -> None:
    """Every recorded state must be one a live game could be sitting in."""
    model = HangmanTransformer(SMALL)
    batch = collect_self_play_states(
        model, WORDS, device="cpu", max_length=MAX_LENGTH
    )

    for row in range(len(batch)):
        guessed = {ALPHABET[i] for i in np.flatnonzero(batch.guessed[row])}
        word_letters = _reconstruct_word_letters(batch, row)
        misses = guessed - word_letters
        assert len(misses) < MAX_WRONG_GUESSES, (
            f"state recorded with {len(misses)} misses: the game was already over"
        )


def test_self_play_reconstructs_the_words_it_was_given() -> None:
    """The labels must describe the actual words, not drift from them."""
    model = HangmanTransformer(SMALL)
    batch = collect_self_play_states(
        model, WORDS, device="cpu", max_length=MAX_LENGTH
    )

    expected = {frozenset(word) for word in WORDS}
    seen = {frozenset(_reconstruct_word_letters(batch, row)) for row in range(len(batch))}
    assert seen <= expected


def test_mixed_sampler_falls_back_to_simulation_before_any_refresh() -> None:
    base = GameStateSampler(WORDS, max_length=MAX_LENGTH, seed=1)
    mixed = MixedStateSampler(base, self_play_fraction=0.5, seed=1)

    assert mixed.buffer_size == 0
    batch = mixed.sample(16)
    assert len(batch) == 16


def test_mixed_sampler_blends_both_sources_after_refresh() -> None:
    base = GameStateSampler(WORDS, max_length=MAX_LENGTH, seed=1)
    mixed = MixedStateSampler(base, self_play_fraction=0.5, seed=1)

    model = HangmanTransformer(SMALL)
    collected = collect_self_play_states(
        model, WORDS, device="cpu", max_length=MAX_LENGTH
    )
    mixed.refresh(collected)

    assert mixed.buffer_size == len(collected)
    batch = mixed.sample(32)
    assert len(batch) == 32
    assert batch.targets.sum(axis=1) == pytest.approx(np.ones(32), abs=1e-5)


def test_self_play_fraction_of_zero_disables_the_buffer() -> None:
    base = GameStateSampler(WORDS, max_length=MAX_LENGTH, seed=1)
    mixed = MixedStateSampler(base, self_play_fraction=0.0, seed=1)

    model = HangmanTransformer(SMALL)
    mixed.refresh(
        collect_self_play_states(model, WORDS, device="cpu", max_length=MAX_LENGTH)
    )
    assert len(mixed.sample(8)) == 8


def test_invalid_self_play_fraction_is_rejected() -> None:
    base = GameStateSampler(WORDS, max_length=MAX_LENGTH, seed=1)
    with pytest.raises(ValueError):
        MixedStateSampler(base, self_play_fraction=1.5)
