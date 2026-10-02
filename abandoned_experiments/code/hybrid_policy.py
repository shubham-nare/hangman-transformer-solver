"""Hybrid neural + vocabulary policy for Hangman.

Architecture:
    Neural branch  -- presence head + position head → log-softmax over 26 letters
    Vocab branch   -- VocabEngine filters candidates, computes Q(letter) = E[reveals]
                      under a neural-reweighted word posterior
    Blend          -- alpha * Q_norm + (1-alpha) * neural_probs
                      where alpha = min(1, pivot / n_candidates)

The vocab branch is free: the position head already ran for the neural branch,
so candidate scoring costs only a numpy gather + matmul, not a second GPU pass.

Alpha behaviour:
    n=1   → alpha=1.0  pure vocab (we know the word)
    n=50  → alpha=1.0  (default pivot=50)
    n=100 → alpha=0.50
    n=500 → alpha=0.10
    n=0   → alpha=0.0  fall back to neural (OOV)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .encoding import MAX_WORD_LENGTH, encode_observations
from .game import ALPHABET, Observation
from .model import HangmanTransformer
from .vocab_engine import VocabEngine

_A = ord("a")
_N = 26


def _alpha(n_cands: int, pivot: float) -> float:
    if n_cands == 0:
        return 0.0
    return min(1.0, pivot / n_cands)


class HybridPolicy:
    """Neural policy blended with vocabulary-guided word posterior.

    Parameters
    ----------
    model:
        Trained HangmanTransformer (position head required for candidate scoring).
    vocab:
        VocabEngine built from the training vocabulary.
    device:
        Torch device for neural inference.
    chunk_size:
        Max observations per forward pass (bounds VRAM usage).
    position_weight:
        Weight on the position head within the neural branch (0=presence only,
        1=position only, 0.5=equal blend — matches training default).
    temperature:
        Softmax temperature for the candidate word posterior.  1.0 = neutral.
        Lower values sharpen the neural reweighting of candidates.
    alpha_pivot:
        Candidate count at which alpha = 0.5 (the vocab/neural crossover).
        Higher → trust vocab more aggressively.
    use_amp:
        Mixed precision for the forward pass.  Defaults to True on CUDA.
    """

    def __init__(
        self,
        model: HangmanTransformer,
        vocab: VocabEngine,
        *,
        device: torch.device | str = "cpu",
        chunk_size: int = 2048,
        position_weight: float = 0.5,
        temperature: float = 1.0,
        alpha_pivot: float = 50.0,
        criterion: str = "reveals",
        use_amp: bool | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.vocab = vocab
        self.chunk_size = chunk_size
        self.position_weight = position_weight
        self.temperature = temperature
        self.alpha_pivot = alpha_pivot
        self.criterion = criterion
        self.use_amp = (
            self.device.type == "cuda" if use_amp is None else use_amp
        )

    @torch.inference_mode()
    def next_guesses(self, observations: Sequence[Observation]) -> list[str]:
        guesses: list[str] = []
        for start in range(0, len(observations), self.chunk_size):
            chunk = list(observations[start : start + self.chunk_size])
            guesses.extend(self._process_chunk(chunk))
        return guesses

    def _process_chunk(self, observations: list[Observation]) -> list[str]:
        batch = encode_observations(
            observations, max_length=MAX_WORD_LENGTH, device=self.device
        )
        tokens, guessed, padding = batch["tokens"], batch["guessed"], batch["padding"]

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(tokens, guessed, padding)

        # --- Neural presence branch: (batch, 26) log-softmax ---
        presence = self.model.mask_guessed(outputs["presence"].float(), guessed)
        neural_log = torch.log_softmax(presence, dim=-1).cpu().numpy()  # (B, 26)

        # --- Neural position branch: (batch, seq_len, 26) log-softmax ---
        # Used to score/reweight vocabulary candidates — no extra GPU pass needed.
        pos_log_np: np.ndarray | None = None
        if "position" in outputs and self.position_weight > 0.0:
            pos_raw = outputs["position"].float()               # (B, L, 26)
            g_exp = guessed.unsqueeze(1).expand_as(pos_raw)
            pos_masked = pos_raw.masked_fill(
                g_exp > 0, torch.finfo(pos_raw.dtype).min
            )
            pos_log_np = torch.log_softmax(pos_masked, dim=-1).cpu().numpy()  # (B, L, 26)

        guesses: list[str] = []
        for i, obs in enumerate(observations):
            pl = pos_log_np[i, : len(obs.board)] if pos_log_np is not None else None
            guesses.append(self._guess_one(obs, neural_log[i], pl))
        return guesses

    def _guess_one(
        self,
        obs: Observation,
        neural_log: np.ndarray,       # (26,) log-softmax from presence head
        pos_log: np.ndarray | None,   # (word_len, 26) from position head
    ) -> str:
        guessed_set: frozenset[str] = obs.guessed_letters
        # Wrong letters: guessed but NOT revealed anywhere on the board
        wrong_set = frozenset(
            g for g in obs.guesses if g.islower() and g not in obs.board
        )

        # Fast path: skip expensive vocab scoring when alpha will be negligible.
        # With pivot=50, alpha < 0.05 when n_cands > 1000. Do a cheap count first.
        fast_ids = self.vocab.get_candidate_ids(obs.board, wrong_set)
        n_fast = len(fast_ids)
        a_fast = _alpha(n_fast, self.alpha_pivot)

        if a_fast < 0.02:
            # Vocab contributes <2% — pure neural is faster and essentially identical
            return self._neural_guess(neural_log, pos_log, obs.board, guessed_set)

        # --- Vocabulary branch (only when alpha is meaningful) ---
        Q, n_cands = self.vocab.score_letters(
            pattern=obs.board,
            wrong_letters=wrong_set,
            guessed_letters=guessed_set,
            position_log_probs=pos_log,
            temperature=self.temperature,
            criterion=self.criterion,
        )
        a = _alpha(n_cands, self.alpha_pivot)

        if a >= 1.0:
            # Only 1-pivot candidates: vocab is authoritative
            best = int(np.argmax(Q))
            return ALPHABET[best]

        # Neural probabilities (from the blended presence+position heads)
        if self.position_weight > 0.0 and pos_log is not None:
            # Replicate the same blend as NeuralPolicy.score()
            # Position head: presence_score_from_positions logic reused here
            # (already masked above, so log-softmax is clean)
            # We use the position head's per-blank marginal as the neural signal
            blank_positions = [j for j, ch in enumerate(obs.board) if ch == "_"]
            if blank_positions:
                blank_lp = pos_log[blank_positions]  # (n_blanks, 26)
                # log P(present) ≈ -log(1 - P(blank)) summed over blanks
                pos_probs = np.exp(blank_lp)         # (n_blanks, 26)
                log_absent = np.log1p(-np.clip(pos_probs, None, 1.0 - 1e-6)).sum(axis=0)
                pos_neural_log = -log_absent          # (26,) — log P(present)
                # Mask guessed
                for ch in guessed_set:
                    if "a" <= ch <= "z":
                        pos_neural_log[ord(ch) - _A] = -1e9
                pos_neural_log -= pos_neural_log.max()
                pos_neural_probs = np.exp(pos_neural_log)
                pos_neural_probs /= pos_neural_probs.sum()
            else:
                pos_neural_probs = None

            # Presence head probs
            presence_probs = np.exp(neural_log - neural_log.max())
            presence_probs /= presence_probs.sum()

            if pos_neural_probs is not None:
                neural_probs = (
                    self.position_weight * pos_neural_probs
                    + (1.0 - self.position_weight) * presence_probs
                )
            else:
                neural_probs = presence_probs
        else:
            neural_probs = np.exp(neural_log - neural_log.max())
            neural_probs /= neural_probs.sum()

        if a <= 0.0:
            return ALPHABET[int(np.argmax(neural_probs))]

        # Normalise Q into a probability distribution
        q_sum = float(Q.sum())
        Q_norm = Q / q_sum if q_sum > 0.0 else np.ones(_N, dtype=np.float32) / _N

        # Final blend
        blended = (1.0 - a) * neural_probs + a * Q_norm

        # Safety: zero guessed letters (already zero in Q_norm, may have fp residue)
        for ch in guessed_set:
            if "a" <= ch <= "z":
                blended[ord(ch) - _A] = 0.0

        return ALPHABET[int(np.argmax(blended))]

    def _neural_guess(
        self,
        neural_log: np.ndarray,
        pos_log: np.ndarray | None,
        board: str,
        guessed_set: frozenset[str],
    ) -> str:
        """Pure neural guess (no vocab) — used on the fast path when alpha < 0.02."""
        if self.position_weight > 0.0 and pos_log is not None:
            blank_positions = [j for j, ch in enumerate(board) if ch == "_"]
            if blank_positions:
                blank_lp = pos_log[blank_positions]
                pos_probs = np.exp(blank_lp)
                log_absent = np.log1p(-np.clip(pos_probs, None, 1.0 - 1e-6)).sum(axis=0)
                pos_neural_log = -log_absent
                for ch in guessed_set:
                    if "a" <= ch <= "z":
                        pos_neural_log[ord(ch) - _A] = -1e9
                pos_neural_log -= pos_neural_log.max()
                pos_neural_probs = np.exp(pos_neural_log)
                pos_neural_probs /= pos_neural_probs.sum()
                presence_probs = np.exp(neural_log - neural_log.max())
                presence_probs /= presence_probs.sum()
                neural_probs = (
                    self.position_weight * pos_neural_probs
                    + (1.0 - self.position_weight) * presence_probs
                )
                return ALPHABET[int(np.argmax(neural_probs))]
        probs = np.exp(neural_log - neural_log.max())
        return ALPHABET[int(np.argmax(probs))]
