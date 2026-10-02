"""Exponential moving average of model weights.

SGD with a decaying learning rate still leaves the final weights rattling around
the minimum rather than sitting in it. An exponential moving average of the
trajectory lands closer to the centre of the basin, which usually generalises
slightly better than any single iterate -- and unlike an ensemble it costs one
extra copy of the weights and no extra inference.

That distinction matters here: averaging along a single training run is a
better-conditioned estimate of the same model, not a bag of separate models
stacked to buy score. It stays one model at inference time.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class WeightAverager:
    """Maintains an exponential moving average of a model's parameters.

    Args:
        model: The model whose weights are tracked.
        decay: EMA decay. Higher averages over a longer window; 0.999 keeps an
            effective horizon of roughly the last thousand steps.
        warmup_steps: Early weights change fast and are not worth averaging, so
            the average simply tracks the model until this many updates have
            happened.
    """

    def __init__(
        self, model: nn.Module, *, decay: float = 0.999, warmup_steps: int = 1_000
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}.")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.shadow = deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold the current weights into the average."""
        self.updates += 1
        # Ramp the decay in, so the average is not dragged by its initialisation.
        decay = 0.0 if self.updates <= self.warmup_steps else self.decay

        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            shadow_param.mul_(decay).add_(model_param.detach(), alpha=1.0 - decay)

        # Buffers (LayerNorm has none here, but be safe) are copied, not averaged.
        for shadow_buffer, model_buffer in zip(
            self.shadow.buffers(), model.buffers()
        ):
            shadow_buffer.copy_(model_buffer)

    @property
    def model(self) -> nn.Module:
        """The averaged model, ready for evaluation."""
        return self.shadow
