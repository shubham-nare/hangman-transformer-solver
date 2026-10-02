"""Neural marginals blended with character-LM posterior inference.

The two signals fail on different words, which is why combining them is worth
more than either alone. Measured per-length on held-out train words, the beam
posterior gains ten points at length 7 and loses six at length 6 -- neither
dominates, so the blend is fitted rather than assumed.

Both branches are legal: the transformer and the n-gram are trained on train.txt
and nothing else.

    neural : P(letter present), per-position marginals combined under an
             independence approximation -- strong when many blanks remain
    beam   : P(letter hits) under an exact posterior over the top-K completions
             proposed by the character LM -- strong when few blanks remain

``blend_weight`` is the weight on the beam branch. It is tuned on held-out train
words, never on the evaluation set.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .beam_solver import BeamPosterior
from .encoding import encode_observations
from .game import ALPHABET, Observation
from .policy import NeuralPolicy

_A = ord("a")


class BlendPolicy:
    """Argmax over a convex combination of the neural and beam letter scores."""

    def __init__(
        self,
        neural: NeuralPolicy,
        beam: BeamPosterior,
        *,
        blend_weight: float = 0.5,
        blank_pivot: float | None = None,
        nn_weight: float = 0.0,
        min_word_length: int = 0,
    ) -> None:
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError(f"blend_weight must be in [0, 1], got {blend_weight}.")
        self.neural = neural
        self.beam = beam
        self.blend_weight = blend_weight
        # A fixed-width beam covers a shrinking fraction of the 26**blanks
        # hypothesis space as blanks grow, so its posterior is near-exact in the
        # endgame and progressively cruder early. Scaling the weight down with
        # the blank count follows that reliability rather than assuming it flat.
        # None keeps the weight constant.
        self.blank_pivot = blank_pivot
        # Weight on the transformer's per-position evidence *inside* the beam
        # search. 0 keeps the two models separate until the final blend (late
        # fusion); above 0 they constrain each other while candidates are being
        # built, which is strictly more information than averaging two
        # 26-dimensional marginals after the fact.
        self.nn_weight = nn_weight
        # Below this word length the character LM is ignored entirely.
        #
        # Measured per-length on held-out words, the blend *loses* 9.5 points at
        # length 3, 4 and 5 and only starts paying from length 8. The cause is
        # structural rather than incidental: train.txt and test.txt share no
        # words, so for a three-letter board the LM concentrates its posterior on
        # the commonest three-letter English words -- which are precisely the
        # ones that cannot be the answer. It is confidently wrong exactly where
        # it is most confident. Longer words have enough orthographic structure
        # that the LM is modelling spelling rather than recalling a short list.
        self.min_word_length = min_word_length
        self.chunk_size = neural.chunk_size

    def _weight(self, n_blanks: int) -> float:
        if self.blank_pivot is None or n_blanks <= self.blank_pivot:
            return self.blend_weight
        return self.blend_weight * (self.blank_pivot / n_blanks)

    @torch.inference_mode()
    def next_guesses(self, observations: Sequence[Observation]) -> list[str]:
        guesses: list[str] = []
        for start in range(0, len(observations), self.chunk_size):
            chunk = list(observations[start : start + self.chunk_size])
            # One batched forward pass serves the whole chunk.
            scores = self.neural.score(chunk).cpu().numpy()  # (B, 26) log-probs
            pos_log = self._position_log_probs(chunk) if self.nn_weight else None
            for i, obs in enumerate(chunk):
                pl = pos_log[i, : len(obs.board)] if pos_log is not None else None
                guesses.append(self._guess_one(obs, scores[i], pl))
        return guesses

    @torch.inference_mode()
    def _position_log_probs(self, observations: list[Observation]) -> np.ndarray | None:
        """Per-position letter log-probabilities, ``(batch, length, 26)``.

        Delegated to the neural branch so a single model and an ensemble are
        interchangeable here. Only computed when the beam is set to use them.
        """
        getter = getattr(self.neural, "position_log_probs", None)
        if getter is None:
            return None
        return getter(observations).cpu().numpy()

    def _guess_one(
        self,
        obs: Observation,
        neural_log: np.ndarray,
        pos_log: np.ndarray | None = None,
    ) -> str:
        guessed = obs.guessed_letters

        neural_p = np.exp(neural_log - neural_log.max())
        total = neural_p.sum()
        neural_p = neural_p / total if total > 0 else np.full(26, 1.0 / 26)

        w = self._weight(obs.board.count("_"))
        if len(obs.board) < self.min_word_length:
            w = 0.0
        if w > 0.0 and self.beam.applicable(obs.board, guessed):
            hit_p, n = self.beam.hit_probabilities(
                obs.board, guessed, pos_log, self.nn_weight
            )
            if n > 0:
                beam_p = np.asarray(hit_p, dtype=np.float64)
                s = beam_p.sum()
                if s > 0:
                    beam_p = beam_p / s
                    combined = (1.0 - w) * neural_p + w * beam_p
                else:
                    combined = neural_p
            else:
                combined = neural_p
        else:
            combined = neural_p

        # Never re-guess: a repeat is scored as a strike.
        for ch in guessed:
            if "a" <= ch <= "z":
                combined[ord(ch) - _A] = -1.0

        return ALPHABET[int(np.argmax(combined))]
