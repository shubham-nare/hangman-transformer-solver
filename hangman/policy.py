"""The neural guessing policy that drives the game engine."""

from __future__ import annotations

from typing import Sequence

import torch

from .encoding import MAX_WORD_LENGTH, encode_observations
from .game import ALPHABET, Observation
from .model import HangmanTransformer, presence_score_from_positions


class NeuralPolicy:
    """Guesses the highest-scoring unguessed letter for each live game.

    Observations arrive one batch per turn, so a single forward pass serves every
    game still in progress. Large batches are split into chunks to bound peak
    memory when playing all 250,000 test words at once.

    The two heads answer the same question differently. The presence head asks
    "which letter is in this word", pooling the whole board. The position head
    asks "what letter fills this specific blank", then those are combined across
    blanks. ``position_weight`` blends their log-scores; it is a validation
    choice, not a guess, and 0.0 or 1.0 recovers either head alone.
    """

    def __init__(
        self,
        model: HangmanTransformer,
        *,
        device: torch.device | str = "cpu",
        chunk_size: int = 2048,
        max_length: int = MAX_WORD_LENGTH,
        position_weight: float = 0.5,
        use_amp: bool | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        # Chunk size trades Python overhead against peak activation memory. It
        # must stay well under free VRAM: at d_model 384 a single 8192-row
        # activation is ~400 MB in fp32, and when that does not fit alongside a
        # training job the allocator thrashes and inference slows by ~30x.
        self.chunk_size = chunk_size
        self.max_length = max_length
        # Half precision halves activation memory and is ample for ranking 26
        # letters; scores are accumulated in fp32 regardless.
        self.use_amp = (
            self.device.type == "cuda" if use_amp is None else use_amp
        )
        if not 0.0 <= position_weight <= 1.0:
            raise ValueError(f"position_weight must be in [0, 1], got {position_weight}.")
        self.position_weight = position_weight

    @torch.inference_mode()
    def next_guesses(self, observations: Sequence[Observation]) -> list[str]:
        guesses: list[str] = []
        for start in range(0, len(observations), self.chunk_size):
            chunk = observations[start : start + self.chunk_size]
            scores = self.score(chunk)
            guesses.extend(ALPHABET[index] for index in scores.argmax(dim=-1).tolist())
        return guesses

    @torch.inference_mode()
    def score(self, observations: Sequence[Observation]) -> torch.Tensor:
        """Return ``(batch, 26)`` log-scores over letters, guessed ones excluded.

        Exposed separately from :meth:`next_guesses` so several policies can be
        combined at the score level rather than by voting on their final picks --
        averaging distributions keeps the confidence information that a vote
        throws away.
        """
        batch = encode_observations(
            observations, max_length=self.max_length, device=self.device
        )
        tokens, guessed, padding = batch["tokens"], batch["guessed"], batch["padding"]
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(tokens, guessed, padding)

        # Already-guessed letters would score as strikes, so they can never win.
        scores = torch.log_softmax(
            self.model.mask_guessed(outputs["presence"].float(), guessed), dim=-1
        )

        if "position" in outputs and self.position_weight > 0.0:
            position_score = presence_score_from_positions(
                outputs["position"], tokens, guessed
            )
            position_scores = torch.log_softmax(
                self.model.mask_guessed(position_score, guessed), dim=-1
            )
            scores = (
                self.position_weight * position_scores
                + (1.0 - self.position_weight) * scores
            )

        return scores

    @torch.inference_mode()
    def position_log_probs(self, observations: Sequence[Observation]) -> torch.Tensor:
        """Per-position letter log-probabilities, ``(batch, length, 26)``.

        Exposed separately from :meth:`score` because the character-LM beam
        consumes per-position evidence directly rather than the pooled letter
        ranking. Already-guessed letters are masked, matching what the policy is
        allowed to play.
        """
        batch = encode_observations(
            observations, max_length=self.max_length, device=self.device
        )
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(batch["tokens"], batch["guessed"], batch["padding"])
        position = outputs["position"].float()
        guessed = batch["guessed"].unsqueeze(1).expand_as(position)
        position = position.masked_fill(guessed > 0, torch.finfo(position.dtype).min)
        return torch.log_softmax(position, dim=-1)


class EnsemblePolicy:
    """Averages the letter distributions of several independently trained models.

    Models trained with different architectures and seeds make different
    mistakes, so averaging their log-scores cancels some of the individual
    error. Combining at the score level rather than by majority vote preserves
    how confident each model was.

    This is the one place where more compute buys score directly, so it is kept
    explicit and optional rather than folded into the single-model path.
    """

    def __init__(self, policies: Sequence[NeuralPolicy], weights: Sequence[float] | None = None) -> None:
        if not policies:
            raise ValueError("An ensemble needs at least one policy.")
        if weights is not None and len(weights) != len(policies):
            raise ValueError(
                f"Got {len(weights)} weights for {len(policies)} policies."
            )
        self.policies = list(policies)
        raw = list(weights) if weights is not None else [1.0] * len(policies)
        total = float(sum(raw))
        if total <= 0.0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        self.weights = [w / total for w in raw]
        self.chunk_size = min(policy.chunk_size for policy in self.policies)

    @torch.inference_mode()
    def next_guesses(self, observations: Sequence[Observation]) -> list[str]:
        guesses: list[str] = []
        for start in range(0, len(observations), self.chunk_size):
            chunk = observations[start : start + self.chunk_size]
            combined = self.score(chunk)
            guesses.extend(
                ALPHABET[index] for index in combined.argmax(dim=-1).tolist()
            )
        return guesses

    @torch.inference_mode()
    def score(self, observations: Sequence[Observation]) -> torch.Tensor:
        """Weighted average of the members' letter log-scores, ``(batch, 26)``.

        Matching :meth:`NeuralPolicy.score` lets an ensemble stand in wherever a
        single model does -- the character-LM blend, in particular, needs no
        knowledge of how many models sit behind the neural branch.
        """
        combined = None
        for policy, weight in zip(self.policies, self.weights):
            scores = policy.score(observations) * weight
            combined = scores if combined is None else combined + scores
        return combined

    @torch.inference_mode()
    def position_log_probs(self, observations: Sequence[Observation]) -> torch.Tensor:
        """Weighted average of the members' per-position log-probabilities.

        Averaged in log space and renormalised, so the result is a proper
        distribution over letters at each position rather than an unnormalised
        geometric mean.
        """
        combined = None
        for policy, weight in zip(self.policies, self.weights):
            lp = policy.position_log_probs(observations) * weight
            combined = lp if combined is None else combined + lp
        return torch.log_softmax(combined, dim=-1)
