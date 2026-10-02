"""Tests for ensembling several policies at the score level."""

from __future__ import annotations

import pytest
import torch

from hangman.game import ALPHABET, GameState, play_games
from hangman.model import HangmanTransformer, ModelConfig
from hangman.policy import EnsemblePolicy, NeuralPolicy

SMALL = ModelConfig(d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, dropout=0.0)


def _policies(count: int) -> list[NeuralPolicy]:
    policies = []
    for seed in range(count):
        torch.manual_seed(seed)
        policies.append(NeuralPolicy(HangmanTransformer(SMALL).eval()))
    return policies


def _observations(words):
    return [GameState(word=word).observation for word in words]


def test_single_model_ensemble_matches_that_model() -> None:
    policy = _policies(1)[0]
    observations = _observations(["banana", "hangman", "transformer"])
    assert EnsemblePolicy([policy]).next_guesses(observations) == policy.next_guesses(
        observations
    )


def test_ensemble_returns_one_valid_unguessed_letter_per_game() -> None:
    ensemble = EnsemblePolicy(_policies(3))
    state = GameState(word="transformer")
    seen = set()
    for _ in range(5):
        guess = ensemble.next_guesses([state.observation])[0]
        assert guess in ALPHABET
        assert guess not in seen
        seen.add(guess)
        state.apply_guess(guess)


def test_ensemble_weights_are_normalised() -> None:
    ensemble = EnsemblePolicy(_policies(2), weights=[3.0, 1.0])
    assert sum(ensemble.weights) == pytest.approx(1.0)
    assert ensemble.weights[0] == pytest.approx(0.75)


def test_zero_weight_model_is_ignored() -> None:
    a, b = _policies(2)
    observations = _observations(["quixotic", "aardvark"])
    weighted = EnsemblePolicy([a, b], weights=[1.0, 0.0])
    assert weighted.next_guesses(observations) == a.next_guesses(observations)


def test_ensemble_plays_a_full_game_through_the_engine() -> None:
    ensemble = EnsemblePolicy(_policies(2))
    results = play_games(["banana", "zzzz"], ensemble)
    assert len(results) == 2
    assert all(result.wrong_guesses <= 6 for result in results)


def test_ensemble_rejects_bad_construction() -> None:
    with pytest.raises(ValueError):
        EnsemblePolicy([])
    with pytest.raises(ValueError):
        EnsemblePolicy(_policies(2), weights=[1.0])
    with pytest.raises(ValueError):
        EnsemblePolicy(_policies(2), weights=[0.0, 0.0])
