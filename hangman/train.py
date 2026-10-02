"""Training loop for the Hangman transformer.

The loss is a cross-entropy against the distribution of letters still hidden,
with already-guessed letters masked out of the softmax exactly as they are at
inference time. Optimising that distribution is a proxy, though -- the
competition scores *games won*, not per-turn letter accuracy. So model selection
runs the real 6-life engine on held-out words and checkpoints on win rate. The
two can diverge: a model that shaves loss by sharpening already-easy mid-game
states while mishandling the opening turn will win fewer games, and only the
game-level metric notices.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .dataset import IGNORE_INDEX, GameStateSampler, StateBatch
from .ema import WeightAverager
from .game import play_games, summarise
from .model import HangmanTransformer, ModelConfig
from .policy import NeuralPolicy
from .selfplay import MixedStateSampler, collect_self_play_states


@dataclass
class TrainingConfig:
    """Optimisation and evaluation settings."""

    steps: int = 12_000
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 1_000
    eval_words: int = 2_000
    log_every: int = 100
    uniform_order_probability: float = 0.25
    exploration_temperature: float = 1.0
    seed: int = 20260901
    amp: bool = True
    #: Weight on the per-blank language-model loss, relative to the presence loss.
    position_loss_weight: float = 1.0
    #: Blend of the two heads at inference; selected on held-out win rate.
    position_weight: float = 0.5

    # --- DAgger self-play ---
    #: Step at which to start mixing in states from the model's own play. Kept
    #: off early: a barely-trained model's trajectories are not worth imitating.
    self_play_start_step: int = 0
    #: How often to re-collect self-play states, in steps.
    self_play_refresh_every: int = 4_000
    #: Training words to play when refreshing the buffer.
    self_play_words: int = 20_000
    #: Share of each batch drawn from the self-play buffer.
    self_play_fraction: float = 0.5

    #: "count" weights letters by how often they repeat; "binary" scores only
    #: whether a letter is present, which is what a guess is ranked by.
    presence_objective: str = "count"
    #: EMA decay for weight averaging. 0 disables.
    ema_decay: float = 0.0
    ema_warmup_steps: int = 1_000


def cross_entropy_over_hidden_letters(
    logits: Tensor, targets: Tensor, guessed: Tensor
) -> Tensor:
    """Cross-entropy between the model's letter distribution and the hidden one.

    Guessed letters are removed from the softmax so the model never spends
    probability mass on a letter it is not allowed to pick.
    """
    masked = HangmanTransformer.mask_guessed(logits, guessed)
    log_probabilities = torch.log_softmax(masked.float(), dim=-1)
    return -(targets * log_probabilities).sum(dim=-1).mean()


def masked_language_model_loss(
    position_logits: Tensor, letter_targets: Tensor
) -> Tensor:
    """Cross-entropy on each blank's true letter, ignoring revealed positions."""
    return nn.functional.cross_entropy(
        position_logits.float().flatten(0, 1),
        letter_targets.flatten(),
        ignore_index=IGNORE_INDEX,
    )


def binary_presence_loss(
    logits: Tensor, present: Tensor, guessed: Tensor
) -> Tensor:
    """Per-letter binary cross-entropy on "is this letter hidden somewhere?".

    This matches what the policy actually ranks by. The count-weighted
    cross-entropy alternative implicitly rewards letters that repeat, which is
    not what makes a guess good -- guessing ``n`` in ``banana`` hits exactly as
    reliably as guessing ``a``.

    Already-guessed letters are excluded from the loss: their answer is known,
    so scoring them would only teach the model something it is never asked.
    """
    per_letter = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), present, reduction="none"
    )
    unguessed = 1.0 - guessed
    return (per_letter * unguessed).sum() / unguessed.sum().clamp(min=1.0)


def combined_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    position_weight: float,
    presence_objective: str = "count",
) -> tuple[Tensor, dict[str, float]]:
    """Word-level presence loss plus the per-blank language-model loss.

    The two are complementary. Presence supervises the decision actually being
    made -- which letter to call next. The blank-level objective is a denser
    signal: one gradient per hidden character rather than one per board, which
    is what teaches the model English spelling well enough to generalise to
    words it has never seen.
    """
    if presence_objective == "binary":
        presence = binary_presence_loss(
            outputs["presence"], batch["present"], batch["guessed"]
        )
    elif presence_objective == "count":
        presence = cross_entropy_over_hidden_letters(
            outputs["presence"], batch["targets"], batch["guessed"]
        )
    else:
        raise ValueError(
            f"presence_objective must be 'count' or 'binary', got {presence_objective!r}."
        )
    parts = {"presence": presence.item()}

    total = presence
    if "position" in outputs and position_weight > 0.0:
        position = masked_language_model_loss(
            outputs["position"], batch["letter_targets"]
        )
        total = total + position_weight * position
        parts["position"] = position.item()

    return total, parts


def to_tensors(batch: StateBatch, device: torch.device) -> dict[str, Tensor]:
    return {
        "tokens": torch.from_numpy(batch.tokens).to(device, non_blocking=True),
        "guessed": torch.from_numpy(batch.guessed).to(device, non_blocking=True),
        "padding": torch.from_numpy(batch.padding).to(device, non_blocking=True),
        "targets": torch.from_numpy(batch.targets).to(device, non_blocking=True),
        "letter_targets": torch.from_numpy(batch.letter_targets).to(
            device, non_blocking=True
        ),
        "present": torch.from_numpy(batch.present).to(device, non_blocking=True),
    }


def learning_rate_at(step: int, config: TrainingConfig) -> float:
    """Linear warmup into a cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate_win_rate(
    model: HangmanTransformer,
    words: list[str],
    device: torch.device,
    *,
    position_weight: float = 0.5,
) -> dict[str, float]:
    """Play real games on held-out words and report the competition metrics."""
    was_training = model.training
    model.eval()
    policy = NeuralPolicy(model, device=device, position_weight=position_weight)
    metrics = summarise(play_games(words, policy))
    if was_training:
        model.train()
    return metrics


def train(
    train_words: list[str],
    validation_words: list[str],
    *,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    output_dir: Path = Path("artifacts"),
    device: torch.device | str | None = None,
    resume: bool = False,
) -> tuple[HangmanTransformer, dict[str, object]]:
    """Fit the model, checkpointing whenever held-out win rate improves.

    Two checkpoints are written. ``best_model.pt`` holds the highest-scoring
    weights and is what gets shipped. ``last_state.pt`` additionally carries the
    optimizer, scaler and step counter, so ``resume=True`` can pick a run back up
    where it stopped -- a multi-hour run should not have to start over because
    the machine or the session went away.
    """
    model_config = model_config or ModelConfig()
    config = training_config or TrainingConfig()
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = HangmanTransformer(model_config).to(device)
    base_sampler = GameStateSampler(
        train_words,
        max_length=model_config.max_length,
        exploration_temperature=config.exploration_temperature,
        uniform_order_probability=config.uniform_order_probability,
        seed=config.seed,
    )
    self_play_enabled = (
        config.self_play_start_step > 0 and config.self_play_fraction > 0.0
    )
    sampler = MixedStateSampler(
        base_sampler,
        self_play_fraction=config.self_play_fraction if self_play_enabled else 0.0,
        seed=config.seed,
    )
    self_play_rng = np.random.default_rng(config.seed)
    averager = (
        WeightAverager(
            model, decay=config.ema_decay, warmup_steps=config.ema_warmup_steps
        )
        if config.ema_decay > 0.0
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    eval_words = validation_words[: config.eval_words]
    history: list[dict[str, float]] = []
    best_win_rate = -1.0
    best_path = output_dir / "best_model.pt"
    state_path = output_dir / "last_state.pt"

    start_step = 0
    if resume and state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        if averager is not None and saved.get("ema_state") is not None:
            averager.shadow.load_state_dict(saved["ema_state"])
            averager.updates = saved.get("ema_updates", 0)
        start_step = saved["step"]
        best_win_rate = saved["best_win_rate"]
        history = saved.get("history", [])
        print(f"resumed from {state_path} at step {start_step:,} (best {best_win_rate:.2f}%)")

    print(f"device={device}  parameters={model.count_parameters():,}")
    print(f"train words={len(train_words):,}  eval words={len(eval_words):,}")
    print(f"steps={config.steps:,}  batch={config.batch_size}\n")

    model.train()
    running_loss = 0.0
    running_parts: dict[str, float] = {}
    started = time.perf_counter()

    for step in range(start_step, config.steps):
        if self_play_enabled and step >= config.self_play_start_step:
            steps_in = step - config.self_play_start_step
            if steps_in % config.self_play_refresh_every == 0:
                chosen = self_play_rng.choice(
                    len(train_words),
                    size=min(config.self_play_words, len(train_words)),
                    replace=False,
                )
                collected = collect_self_play_states(
                    model,
                    [train_words[i] for i in chosen],
                    device=device,
                    max_length=model_config.max_length,
                    position_weight=config.position_weight,
                )
                sampler.refresh(collected)
                print(
                    f"  self-play @ {step:,}: collected {len(collected):,} states "
                    f"from {len(chosen):,} games"
                )

        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, config)

        batch = to_tensors(sampler.sample(config.batch_size), device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(batch["tokens"], batch["guessed"], batch["padding"])
            loss, loss_parts = combined_loss(
                outputs,
                batch,
                config.position_loss_weight,
                presence_objective=config.presence_objective,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if averager is not None:
            averager.update(model)

        running_loss += loss.item()
        running_parts = {
            key: running_parts.get(key, 0.0) + value
            for key, value in loss_parts.items()
        }

        if (step + 1) % config.log_every == 0:
            elapsed = time.perf_counter() - started
            mean_loss = running_loss / config.log_every
            rate = (step + 1) / elapsed
            breakdown = "  ".join(
                f"{key} {value / config.log_every:.4f}"
                for key, value in running_parts.items()
            )
            print(
                f"step {step + 1:>6,}/{config.steps:,}  loss {mean_loss:.4f}  "
                f"({breakdown})  lr {learning_rate_at(step, config):.2e}  "
                f"{rate:.1f} steps/s"
            )
            running_loss = 0.0
            running_parts = {}

        is_last = step + 1 == config.steps
        if (step + 1) % config.eval_every == 0 or is_last:
            # Once weight averaging is active it is what will be shipped, so it
            # is what gets measured and checkpointed.
            candidate = averager.model if averager is not None else model
            metrics = evaluate_win_rate(
                candidate, eval_words, device, position_weight=config.position_weight
            )
            metrics["step"] = step + 1
            history.append(metrics)
            marker = ""
            if metrics["win_rate"] > best_win_rate:
                best_win_rate = metrics["win_rate"]
                torch.save(
                    {
                        "model_state": candidate.state_dict(),
                        "model_config": asdict(model_config),
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

            # Resumable state, refreshed at every eval so an interruption costs
            # at most one eval interval rather than the whole run.
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "ema_state": averager.shadow.state_dict() if averager else None,
                    "ema_updates": averager.updates if averager else 0,
                    "model_config": asdict(model_config),
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
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "minutes": (time.perf_counter() - started) / 60.0,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    # Restore the best checkpoint rather than returning the final-step weights.
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    return model, summary


def load_model(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> HangmanTransformer:
    """Rebuild a model from a saved checkpoint, config included.

    Checkpoints written before the per-position head existed have no
    ``use_position_head`` entry in their config. Rather than defaulting it and
    failing on a shape mismatch, the flag is recovered from the saved weights
    themselves, so older runs stay loadable for ablation comparisons.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"]
    config_fields = dict(checkpoint["model_config"])
    config_fields.setdefault(
        "use_position_head",
        any(key.startswith("position_head.") for key in state),
    )

    model = HangmanTransformer(ModelConfig(**config_fields))
    model.load_state_dict(state)
    return model.to(device).eval()
