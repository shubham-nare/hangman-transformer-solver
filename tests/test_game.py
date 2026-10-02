"""Tests for the exact scoring semantics of the Hangman engine.

These pin down the rules that are easy to get subtly wrong and that would
silently inflate a local score relative to the real leaderboard.
"""

from __future__ import annotations

import pytest

from hangman.game import (
    MAX_WRONG_GUESSES,
    GameState,
    play_games,
    summarise,
)


def test_non_letters_are_revealed_from_the_start() -> None:
    state = GameState(word="coca-cola 7")
    assert state.masked_word == "____-____ 7"
    assert not state.is_solved


def test_correct_guess_reveals_every_occurrence() -> None:
    state = GameState(word="banana")
    assert state.apply_guess("a") is True
    assert state.masked_word == "_a_a_a"
    assert state.wrong_guesses == 0


def test_incorrect_guess_costs_one_strike() -> None:
    state = GameState(word="banana")
    assert state.apply_guess("z") is False
    assert state.wrong_guesses == 1
    assert state.masked_word == "______"


def test_repeated_guess_costs_a_strike_even_when_the_letter_is_present() -> None:
    """A duplicate reveals nothing new, so the rules score it as a strike."""
    state = GameState(word="banana")
    state.apply_guess("a")
    assert state.wrong_guesses == 0
    assert state.apply_guess("a") is False
    assert state.wrong_guesses == 1


def test_non_letter_guess_costs_a_strike() -> None:
    """Guessing a character that is already visible still reveals nothing."""
    state = GameState(word="coca-cola")
    assert state.apply_guess("-") is False
    assert state.wrong_guesses == 1


def test_game_ends_after_six_wrong_guesses() -> None:
    state = GameState(word="banana")
    for letter in "zxqwjy":
        state.apply_guess(letter)
    assert state.wrong_guesses == MAX_WRONG_GUESSES
    assert state.is_out_of_lives
    assert state.is_over
    assert not state.is_solved
    with pytest.raises(RuntimeError):
        state.apply_guess("a")


def test_game_ends_immediately_on_completion() -> None:
    state = GameState(word="banana")
    state.apply_guess("a")
    state.apply_guess("n")
    state.apply_guess("b")
    assert state.is_solved
    assert state.is_over
    assert state.wrong_guesses == 0
    assert state.guess_string == "anb"


def test_word_with_no_guessable_letters_is_solved_at_the_start() -> None:
    """An all-symbol word needs no guesses, so its guess string is empty."""
    state = GameState(word="#1 !")
    assert state.is_solved
    assert state.is_over
    assert state.guess_string == ""


def test_guess_must_be_a_single_character() -> None:
    state = GameState(word="banana")
    with pytest.raises(ValueError):
        state.apply_guess("ab")


class ScriptedPolicy:
    """Replays a fixed letter order, ignoring the board. Test scaffolding only."""

    def __init__(self, order: str) -> None:
        self.order = order

    def next_guesses(self, observations):
        return [self.order[len(obs.guesses)] for obs in observations]


def test_observation_does_not_expose_the_secret_word() -> None:
    """Policies must be structurally unable to read the answer."""
    state = GameState(word="banana")
    state.apply_guess("a")
    obs = state.observation

    assert obs.board == "_a_a_a"
    assert not hasattr(obs, "word")
    assert "banana" not in [str(value) for value in vars(obs).values()]
    assert obs.guessed_letters == frozenset("a")
    assert obs.remaining_lives == MAX_WRONG_GUESSES


def test_observation_tracks_lives_as_strikes_accumulate() -> None:
    state = GameState(word="banana")
    state.apply_guess("z")
    state.apply_guess("x")
    assert state.observation.remaining_lives == MAX_WRONG_GUESSES - 2
    assert state.observation.guesses == ("z", "x")


def test_policies_receive_observations_not_states() -> None:
    """Guard the engine/policy boundary against regressions."""
    seen: list[object] = []

    class RecordingPolicy:
        def next_guesses(self, observations):
            seen.extend(observations)
            return ["a"] * len(observations)

    play_games(["banana"], RecordingPolicy())
    assert seen and all(not hasattr(obs, "word") for obs in seen)


def test_play_games_records_the_adaptive_sequence_and_stops_at_termination() -> None:
    results = play_games(["banana", "zzzz"], ScriptedPolicy("anbxqwjy"))

    banana, zzzz = results
    assert banana.solved
    assert banana.guess_string == "anb"
    assert banana.wrong_guesses == 0

    # "zzzz" never matches, so it strikes out after exactly six guesses.
    assert not zzzz.solved
    assert zzzz.wrong_guesses == MAX_WRONG_GUESSES
    assert len(zzzz.guess_string) == MAX_WRONG_GUESSES


def test_summarise_reports_win_rate_and_tie_break_inputs() -> None:
    metrics = summarise(play_games(["banana", "zzzz"], ScriptedPolicy("anbxqwjy")))
    assert metrics["games"] == 2
    assert metrics["win_rate"] == 50.0
    assert metrics["total_wrong"] == MAX_WRONG_GUESSES
    assert metrics["mean_wrong"] == MAX_WRONG_GUESSES / 2
