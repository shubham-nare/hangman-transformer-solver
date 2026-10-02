"""REINFORCE + value-baseline RL fine-tuning for the Hangman transformer.

Supervised pre-training teaches the model to predict which letters are hidden,
which is a good proxy but not the actual objective. RL fine-tuning closes that
gap by letting the model play real games and receiving +1 for every win, -1 for
every loss -- the exact signal the competition scores.

The algorithm is REINFORCE with a learned value baseline (actor-critic style):

    advantage  = return - value.detach()
    policy_loss = -mean(advantage * log_pi(action))
    value_loss  = MSE(value, return)
    entropy_loss = -mean(entropy)   # exploration bonus
    total = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

Because Hangman is episodic with a terminal reward only (win/loss at game end),
every step within a game shares the same return (gamma = 1, no discounting). This
is deliberate: every guess matters equally to whether the game is won.

Rollouts are collected under ``torch.no_grad()`` in lockstep across all active
games (same pattern as ``play_games`` in game.py), so a single forward pass
serves the whole batch on each turn. After rollout collection the stored
(observation, action) pairs are re-run *with* gradients to produce the log-probs,
values and entropies used in the update.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoding import MAX_WORD_LENGTH, encode_observations
from .game import ALPHABET, GameState, Observation
from .model import HangmanTransformer
from .train import evaluate_win_rate, load_model

N_LETTERS = len(ALPHABET)


@dataclass
class RLConfig:
    """Hyper-parameters for the RL fine-tuning loop."""

    steps: int = 4_000
    #: Games played per gradient update step.
    games_per_step: int = 256
    lr: float = 3e-5
    #: Starting entropy coefficient (linearly decayed to entropy_end over training).
    entropy_coeff: float = 0.05
    #: Final entropy coefficient at the last step.
    entropy_end: float = 0.001
    value_coeff: float = 0.5
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_words: int = 2_000
    log_every: int = 25
    seed: int = 20260901
    amp: bool = True
    #: Blend of presence and position heads used at eval time (not during rollouts).
    position_weight: float = 0.5
    #: 0 = sample words uniformly; 1 = weight fully by 1/length (focus hard short words).
    hard_word_weight: float = 0.5
    #: Per-position reward for each letter revealed correctly (on top of terminal ±1).
    progress_reward: float = 0.02


# ---------------------------------------------------------------------------
# Value head and actor-critic wrapper
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Scalar state-value estimator sitting on top of the transformer pooled rep.

    Takes the same concatenated (mean_pool ‖ max_pool ‖ guessed) tensor the
    presence head sees -- shape ``(batch, 2*d_model + 26)`` -- and outputs a
    scalar value estimate per state.

    The final linear layer is zero-initialised so the value head starts neutral
    and does not destabilise the pre-trained policy in the first RL steps.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Zero-init the output layer for a stable start.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled: Tensor) -> Tensor:
        """Return ``(batch,)`` value estimates."""
        return self.net(pooled).squeeze(-1)


class ActorCritic(nn.Module):
    """Combines the pre-trained transformer (actor) with a fresh value head (critic).

    The transformer's weights are fine-tuned by the RL objective; the value head
    is trained from scratch alongside it. Both share the same optimizer so a
    single ``loss.backward()`` / ``optimizer.step()`` updates everything.
    """

    def __init__(self, transformer: HangmanTransformer) -> None:
        super().__init__()
        self.transformer = transformer
        cfg = transformer.config
        pooled_dim = 2 * cfg.d_model + cfg.n_letters
        self.value_head = ValueHead(pooled_dim, cfg.d_model, cfg.dropout)

    def forward(
        self, tokens: Tensor, guessed: Tensor, padding: Tensor
    ) -> dict[str, Tensor]:
        """Forward pass returning presence logits, position logits, and value.

        Returns the same dict as ``HangmanTransformer.forward`` plus a
        ``"value"`` key with shape ``(batch,)``.
        """
        outputs = self.transformer(tokens, guessed, padding, return_pooled=True)
        outputs["value"] = self.value_head(outputs["pooled"])
        return outputs


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_rollouts(
    actor_critic: ActorCritic,
    words: list[str],
    device: torch.device,
    *,
    max_length: int = MAX_WORD_LENGTH,
    amp: bool = True,
    progress_reward: float = 0.0,
) -> tuple[list[Observation], list[int], list[float]]:
    """Play ``words`` with the current policy, recording every (obs, action, return).

    Games advance in lockstep -- all active games take one turn together before
    any game takes its next. This keeps the batch large on every forward pass,
    which is critical for GPU utilisation when words have different lengths.

    The return assigned to each step is the terminal game outcome (+1 win, -1
    loss), the same for every step in a game. No discounting (gamma = 1): every
    guess contributes equally to whether the word is solved.

    Args:
        actor_critic: The model to play with. Called in eval mode; training mode
            is restored on exit if it was active on entry.
        words: Secret words to play. Each word is one game.
        device: Device for tensor operations.
        max_length: Sequence padding length -- must match the model's config.
        amp: Whether to use automatic mixed precision for the forward pass.

    Returns:
        A triple ``(observations, actions, returns)`` -- one entry per game-turn,
        in the order they were recorded (not sorted by game).
    """
    was_training = actor_critic.training
    actor_critic.eval()

    states = [GameState(word=word) for word in words]
    active_indices = [i for i, s in enumerate(states) if not s.is_over]

    # Per-game storage: list of (obs, action, step_bonus) accumulated turn by turn.
    game_obs: list[list[Observation]] = [[] for _ in words]
    game_acts: list[list[int]] = [[] for _ in words]
    game_bonuses: list[list[float]] = [[] for _ in words]

    while active_indices:
        active_states = [states[i] for i in active_indices]
        observations = [s.observation for s in active_states]

        encoded = encode_observations(observations, max_length=max_length, device=device)
        tokens = encoded["tokens"]
        guessed = encoded["guessed"]
        padding = encoded["padding"]

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            outputs = actor_critic.transformer(tokens, guessed, padding)

        # Mask already-guessed letters and sample from the resulting distribution.
        presence_logits = actor_critic.transformer.mask_guessed(
            outputs["presence"].float(), guessed
        )
        dist = torch.distributions.Categorical(logits=presence_logits)
        sampled = dist.sample()  # (batch,)

        for batch_idx, game_idx in enumerate(active_indices):
            obs = observations[batch_idx]
            action = sampled[batch_idx].item()
            game_obs[game_idx].append(obs)
            game_acts[game_idx].append(action)
            letter = ALPHABET[action]
            board_before = states[game_idx].board.count("_")
            states[game_idx].apply_guess(letter)
            board_after = states[game_idx].board.count("_")
            n_revealed = board_before - board_after
            game_bonuses[game_idx].append(progress_reward * n_revealed)

        active_indices = [i for i in active_indices if not states[i].is_over]

    if was_training:
        actor_critic.train()

    # Assign terminal rewards and flatten into parallel lists.
    all_obs: list[Observation] = []
    all_acts: list[int] = []
    all_returns: list[float] = []

    for game_idx, state in enumerate(states):
        terminal = 1.0 if state.is_solved else -1.0
        for obs, act, bonus in zip(game_obs[game_idx], game_acts[game_idx], game_bonuses[game_idx]):
            all_obs.append(obs)
            all_acts.append(act)
            all_returns.append(terminal + bonus)

    return all_obs, all_acts, all_returns


# ---------------------------------------------------------------------------
# Word sampling with hard-word curriculum
# ---------------------------------------------------------------------------


def _build_word_weights(words: list[str], hard_word_weight: float) -> np.ndarray:
    """Blend uniform and inverse-length sampling weights.

    Short words are harder (fewer constraints per letter guess) and dominate
    the loss tail. Upweighting them means gradient steps see more of the hard
    cases rather than easy long words that the model already handles well.

    At ``hard_word_weight = 0`` words are sampled uniformly; at 1 they are
    sampled in proportion to 1/length (very short words dominate).
    """
    lengths = np.array([max(len(w), 1) for w in words], dtype=np.float64)
    inv_length = 1.0 / lengths
    uniform = np.ones_like(inv_length)
    weights = (1.0 - hard_word_weight) * uniform + hard_word_weight * inv_length
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Main RL training loop
# ---------------------------------------------------------------------------


def rl_train(
    train_words: list[str],
    validation_words: list[str],
    checkpoint_path: str | Path,
    *,
    config: RLConfig | None = None,
    output_dir: Path = Path("artifacts/rl"),
    device: torch.device | str | None = None,
    resume: bool = False,
) -> dict:
    """Fine-tune a pre-trained checkpoint with REINFORCE + value baseline.

    Args:
        train_words: Words to play during rollout collection.
        validation_words: Held-out words for win-rate evaluation.
        checkpoint_path: Path to the supervised ``best_model.pt`` to start from.
        config: RL hyper-parameters; uses defaults if ``None``.
        output_dir: Directory for ``best_model.pt``, ``last_state.pt`` and
            ``training_summary.json``.
        device: Torch device; auto-detected if ``None``.
        resume: If ``True`` and ``last_state.pt`` exists in ``output_dir``,
            resume from that checkpoint rather than the supervised one.

    Returns:
        A summary dict with ``best_win_rate``, ``history``, and ``minutes``.
    """
    config = config or RLConfig()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    transformer = load_model(checkpoint_path, device=device)
    actor_critic = ActorCritic(transformer).to(device)

    optimizer = torch.optim.AdamW(
        actor_critic.parameters(),
        lr=config.lr,
        weight_decay=0.0,  # L2 not appropriate on top of a fine-tuned model
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    eval_words = validation_words[: config.eval_words]
    word_weights = _build_word_weights(train_words, config.hard_word_weight)

    history: list[dict] = []
    best_win_rate = -1.0
    best_path = output_dir / "best_model.pt"
    state_path = output_dir / "last_state.pt"
    start_step = 0

    if resume and state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        actor_critic.load_state_dict(saved["actor_critic_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        start_step = saved["step"]
        best_win_rate = saved["best_win_rate"]
        history = saved.get("history", [])
        print(
            f"resumed from {state_path} at step {start_step:,} "
            f"(best {best_win_rate:.2f}%)"
        )

    n_transformer = sum(p.numel() for p in actor_critic.transformer.parameters())
    n_value = sum(p.numel() for p in actor_critic.value_head.parameters())
    print(f"device={device}  transformer={n_transformer:,}  value_head={n_value:,}")
    print(f"train words={len(train_words):,}  eval words={len(eval_words):,}")
    print(f"rl steps={config.steps:,}  games/step={config.games_per_step}\n")

    actor_critic.train()
    started = time.perf_counter()
    running: dict[str, float] = {}

    for step in range(start_step, config.steps):
        # --- rollout ---
        word_indices = rng.choice(
            len(train_words), size=config.games_per_step, replace=True, p=word_weights
        )
        batch_words = [train_words[i] for i in word_indices]

        # Linear entropy decay: entropy_coeff → entropy_end over all steps
        progress = step / max(config.steps - 1, 1)
        current_entropy_coeff = config.entropy_coeff + (config.entropy_end - config.entropy_coeff) * progress

        obs_list, act_list, ret_list = collect_rollouts(
            actor_critic, batch_words, device, max_length=transformer.config.max_length,
            amp=use_amp, progress_reward=config.progress_reward,
        )

        if not obs_list:
            continue

        returns_t = torch.tensor(ret_list, dtype=torch.float32, device=device)
        actions_t = torch.tensor(act_list, dtype=torch.long, device=device)

        # --- gradient update in mini-chunks to bound activation memory ---
        chunk = 512
        n = len(obs_list)
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0

        optimizer.zero_grad(set_to_none=True)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            chunk_obs = obs_list[start:end]
            chunk_acts = actions_t[start:end]
            chunk_rets = returns_t[start:end]

            encoded = encode_observations(
                chunk_obs, max_length=transformer.config.max_length, device=device
            )
            tokens = encoded["tokens"]
            guessed_t = encoded["guessed"]
            padding = encoded["padding"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = actor_critic(tokens, guessed_t, padding)

                presence_logits = actor_critic.transformer.mask_guessed(
                    outputs["presence"].float(), guessed_t
                )
                dist = torch.distributions.Categorical(logits=presence_logits)
                log_probs = dist.log_prob(chunk_acts)
                entropy = dist.entropy()
                values = outputs["value"]

                advantage = chunk_rets - values.detach()
                # Normalise advantages over the full rollout by rescaling this chunk.
                advantage = (advantage - returns_t.mean()) / (returns_t.std() + 1e-8)

                policy_loss = -(advantage * log_probs).mean()
                value_loss = F.mse_loss(values, chunk_rets)
                entropy_loss = -entropy.mean()

                # Scale by chunk fraction so gradients sum to the full-batch gradient.
                scale = (end - start) / n
                loss = scale * (
                    policy_loss
                    + config.value_coeff * value_loss
                    + current_entropy_coeff * entropy_loss
                )

            scaler.scale(loss).backward()

            total_policy += policy_loss.item() * scale
            total_value += value_loss.item() * scale
            total_entropy += entropy_loss.item() * scale

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(actor_critic.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        win_frac = sum(r > 0 for r in ret_list) / len(ret_list)
        running.setdefault("policy", 0.0)
        running.setdefault("value", 0.0)
        running.setdefault("entropy", 0.0)
        running.setdefault("win_frac", 0.0)
        running["policy"] += total_policy
        running["value"] += total_value
        running["entropy"] += total_entropy
        running["win_frac"] += win_frac

        if (step + 1) % config.log_every == 0:
            elapsed = time.perf_counter() - started
            rate = (step + 1 - start_step) / max(elapsed, 1e-9)
            n_log = config.log_every
            print(
                f"step {step + 1:>5,}/{config.steps:,}  "
                f"policy {running['policy']/n_log:+.4f}  "
                f"value {running['value']/n_log:.4f}  "
                f"entropy {running['entropy']/n_log:.4f}  "
                f"win% {100*running['win_frac']/n_log:.1f}  "
                f"{rate:.2f} steps/s"
            )
            running = {}

        is_last = step + 1 == config.steps
        if (step + 1) % config.eval_every == 0 or is_last:
            metrics = evaluate_win_rate(
                actor_critic.transformer,
                eval_words,
                device,
                position_weight=config.position_weight,
            )
            metrics["step"] = step + 1
            history.append(metrics)
            marker = ""
            if metrics["win_rate"] > best_win_rate:
                best_win_rate = metrics["win_rate"]
                torch.save(
                    {
                        "model_state": actor_critic.transformer.state_dict(),
                        "model_config": asdict(actor_critic.transformer.config),
                        "win_rate": best_win_rate,
                        "step": step + 1,
                    },
                    best_path,
                )
                marker = "  <- best, checkpointed"
            print(
                f"  eval @ {step + 1:,}: win rate {metrics['win_rate']:.2f}%  "
                f"mean wrong {metrics['mean_wrong']:.3f}{marker}"
            )

            torch.save(
                {
                    "actor_critic_state": actor_critic.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "step": step + 1,
                    "best_win_rate": best_win_rate,
                    "history": history,
                },
                state_path,
            )

    summary = {
        "best_win_rate": best_win_rate,
        "checkpoint": str(best_path),
        "history": history,
        "rl_config": asdict(config),
        "minutes": (time.perf_counter() - started) / 60.0,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
