"""Tests for the RL fine-tuning module (hangman/rl.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hangman.model import HangmanTransformer, ModelConfig
from hangman.rl import ActorCritic, RLConfig, ValueHead, collect_rollouts, rl_train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORDS = ["cat", "dog", "hello", "world", "python", "hangman"]


def _tiny_transformer() -> HangmanTransformer:
    """A very small transformer that fits in CPU RAM for unit tests."""
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, dim_feedforward=64)
    return HangmanTransformer(cfg)


def _tiny_actor_critic() -> ActorCritic:
    return ActorCritic(_tiny_transformer())


# ---------------------------------------------------------------------------
# return_pooled flag on HangmanTransformer
# ---------------------------------------------------------------------------


def test_return_pooled_adds_key() -> None:
    model = _tiny_transformer()
    tokens = torch.zeros(2, 8, dtype=torch.long)
    guessed = torch.zeros(2, 26)
    padding = torch.zeros(2, 8, dtype=torch.bool)

    out_without = model(tokens, guessed, padding)
    assert "pooled" not in out_without

    out_with = model(tokens, guessed, padding, return_pooled=True)
    assert "pooled" in out_with


def test_return_pooled_shape() -> None:
    model = _tiny_transformer()
    d = model.config.d_model
    n = model.config.n_letters
    batch = 3
    tokens = torch.zeros(batch, 8, dtype=torch.long)
    guessed = torch.zeros(batch, 26)
    padding = torch.zeros(batch, 8, dtype=torch.bool)

    out = model(tokens, guessed, padding, return_pooled=True)
    assert out["pooled"].shape == (batch, 2 * d + n)


def test_return_pooled_backward_compat() -> None:
    """Calling forward without return_pooled must not raise and must omit pooled."""
    model = _tiny_transformer()
    tokens = torch.zeros(1, 4, dtype=torch.long)
    guessed = torch.zeros(1, 26)
    padding = torch.zeros(1, 4, dtype=torch.bool)
    out = model(tokens, guessed, padding)
    assert "presence" in out
    assert "pooled" not in out


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------


def test_actor_critic_output_keys() -> None:
    ac = _tiny_actor_critic()
    tokens = torch.zeros(2, 8, dtype=torch.long)
    guessed = torch.zeros(2, 26)
    padding = torch.zeros(2, 8, dtype=torch.bool)
    out = ac(tokens, guessed, padding)
    assert "presence" in out
    assert "value" in out
    # position head is present on the default tiny config
    assert "position" in out


def test_actor_critic_value_shape() -> None:
    ac = _tiny_actor_critic()
    batch = 5
    tokens = torch.zeros(batch, 8, dtype=torch.long)
    guessed = torch.zeros(batch, 26)
    padding = torch.zeros(batch, 8, dtype=torch.bool)
    out = ac(tokens, guessed, padding)
    assert out["value"].shape == (batch,)


def test_value_head_zero_init() -> None:
    """Last layer of value head must be zero-initialised for stability."""
    model = _tiny_transformer()
    ac = ActorCritic(model)
    last_linear = ac.value_head.net[-1]
    assert torch.all(last_linear.weight == 0)
    assert torch.all(last_linear.bias == 0)


# ---------------------------------------------------------------------------
# collect_rollouts
# ---------------------------------------------------------------------------


def test_collect_rollouts_lengths() -> None:
    """Each entry in obs/acts/returns must have the same length."""
    ac = _tiny_actor_critic()
    device = torch.device("cpu")
    obs, acts, rets = collect_rollouts(
        ac, _WORDS, device, max_length=32, amp=False
    )
    assert len(obs) == len(acts) == len(rets)
    assert len(obs) > 0


def test_collect_rollouts_returns_values() -> None:
    """All returns must be +1 or -1 (terminal reward only)."""
    ac = _tiny_actor_critic()
    device = torch.device("cpu")
    _, _, rets = collect_rollouts(ac, _WORDS, device, max_length=32, amp=False)
    assert all(r in (1.0, -1.0) for r in rets)


def test_collect_rollouts_preserves_train_mode() -> None:
    """collect_rollouts must restore training mode if the model was training."""
    ac = _tiny_actor_critic()
    ac.train()
    collect_rollouts(ac, ["cat", "dog"], torch.device("cpu"), max_length=32, amp=False)
    assert ac.training


def test_collect_rollouts_eval_mode_unchanged() -> None:
    """collect_rollouts must leave eval mode as-is."""
    ac = _tiny_actor_critic()
    ac.eval()
    collect_rollouts(ac, ["cat"], torch.device("cpu"), max_length=32, amp=False)
    assert not ac.training


# ---------------------------------------------------------------------------
# rl_train smoke test
# ---------------------------------------------------------------------------


def test_rl_train_smoke(tmp_path: Path) -> None:
    """RL training must run for 2 steps without error and write a checkpoint."""
    # Save a tiny supervised checkpoint.
    transformer = _tiny_transformer()
    ckpt = tmp_path / "init.pt"
    torch.save(
        {
            "model_state": transformer.state_dict(),
            "model_config": {
                "d_model": 32,
                "n_heads": 2,
                "n_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "max_length": 32,
                "n_letters": 26,
                "use_position_head": True,
            },
            "win_rate": 0.0,
            "step": 0,
        },
        ckpt,
    )

    config = RLConfig(
        steps=2,
        games_per_step=4,
        eval_every=2,
        eval_words=4,
        amp=False,
        seed=42,
    )
    out_dir = tmp_path / "rl_out"
    summary = rl_train(
        train_words=_WORDS,
        validation_words=_WORDS,
        checkpoint_path=ckpt,
        config=config,
        output_dir=out_dir,
        device="cpu",
        resume=False,
    )

    assert (out_dir / "best_model.pt").exists()
    assert (out_dir / "training_summary.json").exists()
    assert "best_win_rate" in summary
    assert len(summary["history"]) >= 1
