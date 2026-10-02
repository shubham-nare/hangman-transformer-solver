"""Tests for the presence objectives and weight averaging."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hangman.dataset import GameStateSampler
from hangman.ema import WeightAverager
from hangman.game import ALPHABET
from hangman.model import HangmanTransformer, ModelConfig
from hangman.train import binary_presence_loss, combined_loss, to_tensors

SMALL = ModelConfig(d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, dropout=0.0)
DEVICE = torch.device("cpu")


def test_present_marks_hidden_letters_regardless_of_repetition() -> None:
    """'banana' with 'b' revealed: both 'a' and 'n' are simply present."""
    sampler = GameStateSampler(["banana"], max_length=32, seed=3)
    batch = sampler.sample_for_words(np.zeros(100, dtype=np.int64))

    a, n = ALPHABET.index("a"), ALPHABET.index("n")
    both = (batch.present[:, a] > 0) & (batch.present[:, n] > 0)
    rows = np.flatnonzero(both)
    assert rows.size

    # The count-weighted target ranks 'a' above 'n'; the binary one does not.
    assert (batch.targets[rows, a] > batch.targets[rows, n]).all()
    assert (batch.present[rows, a] == batch.present[rows, n]).all()
    assert set(np.unique(batch.present)) <= {0.0, 1.0}


def test_binary_presence_loss_ignores_already_guessed_letters() -> None:
    """A guessed letter's answer is known, so it must not enter the loss."""
    # Non-zero logits matter: at logit 0 the loss is log(2) whichever way the
    # target points, so a flipped target would be indistinguishable.
    logits = torch.full((1, len(ALPHABET)), 2.0)
    present = torch.zeros(1, len(ALPHABET))
    guessed = torch.zeros(1, len(ALPHABET))

    baseline = binary_presence_loss(logits, present, guessed).item()

    guessed_some = guessed.clone()
    guessed_some[0, :10] = 1.0
    masked = binary_presence_loss(logits, present, guessed_some).item()

    # With uniform logits both average to the same per-letter value, but the
    # masked version is computed over fewer letters.
    assert masked == pytest.approx(baseline, abs=1e-6)

    # Flipping the truth on an already-guessed letter must not move the loss.
    present_guessed = present.clone()
    present_guessed[0, 0] = 1.0
    assert binary_presence_loss(
        logits, present_guessed, guessed_some
    ).item() == pytest.approx(masked, abs=1e-6)

    # ...but flipping it on an unguessed letter must.
    present_unguessed = present.clone()
    present_unguessed[0, 20] = 1.0
    assert binary_presence_loss(
        logits, present_unguessed, guessed_some
    ).item() != pytest.approx(masked, abs=1e-6)


def test_binary_presence_loss_rewards_correct_predictions() -> None:
    present = torch.zeros(1, len(ALPHABET))
    present[0, 5] = 1.0
    guessed = torch.zeros(1, len(ALPHABET))

    confident_right = torch.full((1, len(ALPHABET)), -5.0)
    confident_right[0, 5] = 5.0
    confident_wrong = torch.full((1, len(ALPHABET)), -5.0)
    confident_wrong[0, 5] = -5.0

    assert binary_presence_loss(confident_right, present, guessed) < binary_presence_loss(
        confident_wrong, present, guessed
    )


def test_combined_loss_supports_both_objectives() -> None:
    sampler = GameStateSampler(["banana", "hangman"], max_length=32, seed=1)
    batch = to_tensors(sampler.sample(8), DEVICE)
    model = HangmanTransformer(SMALL)
    outputs = model(batch["tokens"], batch["guessed"], batch["padding"])

    for objective in ("count", "binary"):
        loss, parts = combined_loss(outputs, batch, 1.0, presence_objective=objective)
        assert torch.isfinite(loss)
        assert "presence" in parts and "position" in parts


def test_combined_loss_rejects_an_unknown_objective() -> None:
    sampler = GameStateSampler(["banana"], max_length=32, seed=1)
    batch = to_tensors(sampler.sample(4), DEVICE)
    model = HangmanTransformer(SMALL)
    outputs = model(batch["tokens"], batch["guessed"], batch["padding"])

    with pytest.raises(ValueError, match="presence_objective"):
        combined_loss(outputs, batch, 1.0, presence_objective="nonsense")


def test_weight_averager_tracks_then_lags_the_model() -> None:
    model = HangmanTransformer(SMALL)
    averager = WeightAverager(model, decay=0.9, warmup_steps=1)

    # During warmup the average copies the model exactly.
    averager.update(model)
    for shadow, live in zip(averager.model.parameters(), model.parameters()):
        assert torch.allclose(shadow, live)

    # After warmup it lags: a big weight change moves the average only partway.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    averager.update(model)

    lagging = [
        not torch.allclose(shadow, live)
        for shadow, live in zip(averager.model.parameters(), model.parameters())
    ]
    assert any(lagging)


def test_weight_averager_does_not_track_gradients() -> None:
    model = HangmanTransformer(SMALL)
    averager = WeightAverager(model, decay=0.99)
    assert all(not p.requires_grad for p in averager.model.parameters())


def test_weight_averager_rejects_invalid_decay() -> None:
    model = HangmanTransformer(SMALL)
    with pytest.raises(ValueError):
        WeightAverager(model, decay=1.0)
