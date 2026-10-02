"""Self-play state collection (DAgger).

The bootstrap sampler in :mod:`hangman.dataset` simulates games with a
stochastic frequency-ranked player. That is a reasonable stand-in, but it is not
the player whose states actually matter -- the trained model reveals letters in
a different order, so it visits a different part of the state space. Fitting on
one distribution and acting in another is exactly the covariate shift that
DAgger addresses: let the current policy act, then train on the states it
actually reaches.

Concretely, the model plays real games on *training* words, every intermediate
board is labelled with the answer, and those states are mixed back into the
training stream. The model still only ever sees an ``Observation`` while
playing; the word is used solely to construct the label afterwards, exactly as
in ordinary supervised learning.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import (
    GameStateSampler,
    StateBatch,
    concatenate_batches,
    encode_supervised_states,
    take_rows,
)
from .game import ALPHABET, GameState
from .model import HangmanTransformer
from .policy import NeuralPolicy

N_LETTERS = len(ALPHABET)


@torch.inference_mode()
def collect_self_play_states(
    model: HangmanTransformer,
    words: list[str],
    *,
    device: torch.device | str,
    max_length: int,
    position_weight: float = 0.5,
    chunk_size: int = 8192,
) -> StateBatch:
    """Play ``words`` with ``model`` and label every state along the way.

    Returns one labelled state per turn per game. Longer games contribute more
    states, which is the desired weighting: those are the words the model finds
    hard, and they are where the remaining wins are.
    """
    was_training = model.training
    model.eval()
    policy = NeuralPolicy(
        model, device=device, chunk_size=chunk_size, position_weight=position_weight
    )

    states = [GameState(word=word) for word in words]
    active = [state for state in states if not state.is_over]

    recorded_words: list[str] = []
    recorded_boards: list[str] = []
    recorded_guessed: list[np.ndarray] = []

    while active:
        observations = [state.observation for state in active]

        for state, observation in zip(active, observations):
            mask = np.zeros(N_LETTERS, dtype=np.float32)
            for letter in observation.guessed_letters:
                index = ALPHABET.find(letter)
                if index >= 0:
                    mask[index] = 1.0
            recorded_words.append(state.word)
            recorded_boards.append(observation.board)
            recorded_guessed.append(mask)

        for state, guess in zip(active, policy.next_guesses(observations)):
            state.apply_guess(guess)
        active = [state for state in active if not state.is_over]

    if was_training:
        model.train()

    return encode_supervised_states(
        recorded_words,
        recorded_boards,
        np.stack(recorded_guessed),
        max_length=max_length,
    )


class MixedStateSampler:
    """Draws each batch from simulated states and self-play states in a fixed ratio.

    Training purely on self-play states would be a mistake: the model would only
    ever revisit its own habits, and states it has learned to avoid -- including
    the ones it reaches after an unlucky opening -- would vanish from the
    training set. Keeping the simulated sampler in the mix preserves that
    coverage while the self-play share corrects the distribution shift.
    """

    def __init__(
        self,
        base: GameStateSampler,
        *,
        self_play_fraction: float = 0.5,
        seed: int = 20260901,
    ) -> None:
        if not 0.0 <= self_play_fraction <= 1.0:
            raise ValueError(
                f"self_play_fraction must be in [0, 1], got {self_play_fraction}."
            )
        self.base = base
        self.self_play_fraction = self_play_fraction
        self.rng = np.random.default_rng(seed)
        self._buffer: StateBatch | None = None

    @property
    def buffer_size(self) -> int:
        return 0 if self._buffer is None else len(self._buffer)

    def refresh(self, batch: StateBatch) -> None:
        """Replace the self-play buffer with freshly collected states."""
        self._buffer = batch

    def sample(self, batch_size: int) -> StateBatch:
        if self._buffer is None or self.self_play_fraction <= 0.0:
            return self.base.sample(batch_size)

        n_self_play = min(
            int(round(batch_size * self.self_play_fraction)), len(self._buffer)
        )
        n_simulated = batch_size - n_self_play
        if n_self_play == 0:
            return self.base.sample(batch_size)

        indices = self.rng.integers(0, len(self._buffer), size=n_self_play)
        parts = [take_rows(self._buffer, indices)]
        if n_simulated > 0:
            parts.insert(0, self.base.sample(n_simulated))
        return concatenate_batches(parts)
