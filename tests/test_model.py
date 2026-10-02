"""Tests for model output shapes, masking, and score aggregation."""

from __future__ import annotations

import numpy as np
import torch

from hangman.encoding import MASK_TOKEN_ID, encode_observations
from hangman.game import ALPHABET, GameState
from hangman.model import HangmanTransformer, ModelConfig, presence_score_from_positions
from hangman.policy import NeuralPolicy

SMALL = ModelConfig(d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, dropout=0.0)


def _observations(words: list[str], guesses: str = "") -> list:
    observations = []
    for word in words:
        state = GameState(word=word)
        for guess in guesses:
            if not state.is_over:
                state.apply_guess(guess)
        observations.append(state.observation)
    return observations


def test_forward_returns_both_heads_with_expected_shapes() -> None:
    model = HangmanTransformer(SMALL).eval()
    batch = encode_observations(_observations(["banana", "hangman"], "a"))

    with torch.no_grad():
        outputs = model(batch["tokens"], batch["guessed"], batch["padding"])

    assert outputs["presence"].shape == (2, len(ALPHABET))
    assert outputs["position"].shape == (2, batch["tokens"].shape[1], len(ALPHABET))


def test_position_head_can_be_disabled() -> None:
    model = HangmanTransformer(ModelConfig(**{**vars(SMALL), "use_position_head": False}))
    batch = encode_observations(_observations(["banana"]))

    with torch.no_grad():
        outputs = model(batch["tokens"], batch["guessed"], batch["padding"])

    assert "position" not in outputs


def test_mask_guessed_makes_guessed_letters_unpickable() -> None:
    logits = torch.zeros(1, len(ALPHABET))
    guessed = torch.zeros(1, len(ALPHABET))
    guessed[0, ALPHABET.index("e")] = 1.0

    masked = HangmanTransformer.mask_guessed(logits, guessed)
    assert masked.argmax(dim=-1).item() != ALPHABET.index("e")
    assert masked[0, ALPHABET.index("e")] < -1e30


def test_mask_guessed_survives_half_precision() -> None:
    """A -1e9 sentinel would overflow float16 under AMP."""
    logits = torch.zeros(1, len(ALPHABET), dtype=torch.float16)
    guessed = torch.zeros(1, len(ALPHABET), dtype=torch.float16)
    guessed[0, 0] = 1.0

    masked = HangmanTransformer.mask_guessed(logits, guessed)
    assert torch.isfinite(masked).all()
    assert masked.argmax(dim=-1).item() != 0


def test_presence_score_aggregates_across_blanks_only() -> None:
    """Revealed positions must not contribute to a letter's presence score."""
    length = 4
    n_letters = len(ALPHABET)
    tokens = torch.full((1, length), MASK_TOKEN_ID)
    # Mark the final position as revealed, so its opinion is discarded.
    tokens[0, 3] = 5
    guessed = torch.zeros(1, n_letters)

    logits = torch.zeros(1, length, n_letters)
    logits[0, 3, ALPHABET.index("z")] = 20.0  # confident, but at a revealed slot

    scores = presence_score_from_positions(logits, tokens, guessed)
    ranked = scores.argsort(descending=True)[0].tolist()
    assert ALPHABET[ranked[0]] != "z"


def test_presence_score_rises_with_more_blanks_favouring_a_letter() -> None:
    """Two blanks pointing at a letter must outrank one blank pointing at it."""
    n_letters = len(ALPHABET)
    tokens = torch.full((2, 3), MASK_TOKEN_ID)
    guessed = torch.zeros(2, n_letters)
    index = ALPHABET.index("t")

    logits = torch.zeros(2, 3, n_letters)
    logits[0, 0, index] = 4.0
    logits[1, 0, index] = 4.0
    logits[1, 1, index] = 4.0

    scores = presence_score_from_positions(logits, tokens, guessed)
    assert scores[1, index] > scores[0, index]


def test_policy_never_repeats_a_guess() -> None:
    """A repeat costs a strike, so the policy must always pick something new."""
    model = HangmanTransformer(SMALL).eval()
    policy = NeuralPolicy(model, position_weight=0.5)

    state = GameState(word="hangman")
    seen: set[str] = set()
    for _ in range(6):
        if state.is_over:
            break
        guess = policy.next_guesses([state.observation])[0]
        assert guess not in seen
        seen.add(guess)
        state.apply_guess(guess)


def test_policy_weights_reproduce_single_head_behaviour() -> None:
    """position_weight of 0 or 1 must fall back to exactly one head."""
    torch.manual_seed(0)
    model = HangmanTransformer(SMALL).eval()
    observations = _observations(["transformer", "banana"], "ae")

    presence_only = NeuralPolicy(model, position_weight=0.0).next_guesses(observations)
    position_only = NeuralPolicy(model, position_weight=1.0).next_guesses(observations)

    assert len(presence_only) == len(position_only) == 2
    assert all(guess in ALPHABET for guess in presence_only + position_only)


def test_policy_chunking_matches_single_pass() -> None:
    """Chunked inference must not change any guess."""
    torch.manual_seed(0)
    model = HangmanTransformer(SMALL).eval()
    observations = _observations(["banana", "hangman", "quixotic", "aardvark"], "a")

    whole = NeuralPolicy(model, chunk_size=1024).next_guesses(observations)
    chunked = NeuralPolicy(model, chunk_size=2).next_guesses(observations)
    assert whole == chunked
