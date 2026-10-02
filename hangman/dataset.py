"""Sampling training states from simulated games.

The subtle part of this problem is not the architecture, it is *which board
states the model is trained on*. The obvious approach -- reveal a random subset
of a word's letters -- produces states that never arise in play. A real game
reveals letters in the order a player guesses them, and carries the scars of
that player's misses. A model fitted to uniform random subsets is being asked a
different question at inference time than the one it was trained on.

States here are drawn from *simulated trajectories* instead. Each word is played
out by a stochastic frequency-ranked player, and a uniformly random turn along
that trajectory is sampled. Two knobs control the spread:

* ``exploration_temperature`` perturbs the player's letter ranking, so the
  corpus covers many plausible orderings rather than one canonical one.
* ``uniform_order_probability`` plays a fraction of games in a fully random
  letter order, which reaches unusual states a competent player rarely visits
  but that still occur -- and keeps the model from over-fitting to the habits of
  its bootstrap policy.

Everything is vectorised across the batch: whole games are simulated in closed
form with cumulative sums, so generating millions of states costs seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import LETTER_TOKEN_OFFSET, MASK_TOKEN_ID, PAD_TOKEN_ID
from .game import ALPHABET, MAX_WRONG_GUESSES

N_LETTERS: int = len(ALPHABET)
_LETTER_A = ord("a")


#: ``letter_targets`` entries at positions that are padding or already revealed.
#: Matches PyTorch's default ``ignore_index`` for cross-entropy.
IGNORE_INDEX: int = -100


@dataclass(frozen=True)
class StateBatch:
    """A batch of encoded training states with their supervision targets.

    Attributes:
        tokens: ``(batch, length)`` board token ids.
        guessed: ``(batch, 26)`` binary already-guessed mask.
        padding: ``(batch, length)`` boolean, ``True`` at padding.
        targets: ``(batch, 26)`` target distribution over hidden letters,
            weighted by occurrence count.
        present: ``(batch, 26)`` binary -- is this letter hidden somewhere?
            This is the quantity a guess is actually ranked by. ``targets``
            answers a subtly different question: in ``banana`` with ``b``
            revealed it assigns ``a`` 2/3 and ``n`` 1/3, because ``a`` occurs
            twice. But guessing ``n`` hits just as surely as guessing ``a`` --
            repetition does not make a letter a better guess. Keeping both lets
            the objective be chosen by measurement rather than assumption.
        letter_targets: ``(batch, length)`` true letter index at each hidden
            position, ``IGNORE_INDEX`` where the position is padding or already
            revealed. This supervises the per-position masked-language-model
            head, which learns orthography directly -- that ``q`` is followed by
            ``u``, that ``_ing`` is a likely ending -- rather than only the
            word-level question of which letters are present somewhere.
    """

    tokens: np.ndarray
    guessed: np.ndarray
    padding: np.ndarray
    targets: np.ndarray
    letter_targets: np.ndarray
    present: np.ndarray

    def __len__(self) -> int:
        return len(self.tokens)


def encode_supervised_states(
    words: Sequence[str],
    boards: Sequence[str],
    guessed_masks: np.ndarray,
    *,
    max_length: int,
) -> StateBatch:
    """Build a supervised batch from explicit (word, board, guessed) triples.

    Used to label states that came from somewhere other than the built-in
    simulator -- self-play, most importantly. The board alone determines which
    positions are hidden, so the targets follow from comparing it to the word.
    """
    batch_size = len(words)
    tokens = np.full((batch_size, max_length), PAD_TOKEN_ID, dtype=np.int64)
    padding = np.ones((batch_size, max_length), dtype=bool)
    letter_targets = np.full((batch_size, max_length), IGNORE_INDEX, dtype=np.int64)
    counts = np.zeros((batch_size, N_LETTERS), dtype=np.float32)

    for row, (word, board) in enumerate(zip(words, boards)):
        padding[row, : len(word)] = False
        for column, (word_char, board_char) in enumerate(zip(word, board)):
            letter = ord(word_char) - _LETTER_A
            if board_char == "_":
                tokens[row, column] = MASK_TOKEN_ID
                if 0 <= letter < N_LETTERS:
                    letter_targets[row, column] = letter
                    counts[row, letter] += 1.0
            else:
                tokens[row, column] = letter + LETTER_TOKEN_OFFSET

    totals = counts.sum(axis=1, keepdims=True).clip(min=1.0)
    return StateBatch(
        tokens=tokens,
        guessed=guessed_masks.astype(np.float32),
        padding=padding,
        targets=counts / totals,
        letter_targets=letter_targets,
        present=(counts > 0).astype(np.float32),
    )


def concatenate_batches(batches: Sequence[StateBatch]) -> StateBatch:
    """Join several batches into one."""
    return StateBatch(
        tokens=np.concatenate([b.tokens for b in batches]),
        guessed=np.concatenate([b.guessed for b in batches]),
        padding=np.concatenate([b.padding for b in batches]),
        targets=np.concatenate([b.targets for b in batches]),
        letter_targets=np.concatenate([b.letter_targets for b in batches]),
        present=np.concatenate([b.present for b in batches]),
    )


def take_rows(batch: StateBatch, indices: np.ndarray) -> StateBatch:
    """Select a subset of rows from a batch."""
    return StateBatch(
        tokens=batch.tokens[indices],
        guessed=batch.guessed[indices],
        padding=batch.padding[indices],
        targets=batch.targets[indices],
        letter_targets=batch.letter_targets[indices],
        present=batch.present[indices],
    )


class GameStateSampler:
    """Generates training states by simulating games over a word list."""

    def __init__(
        self,
        words: list[str],
        *,
        max_length: int,
        exploration_temperature: float = 1.0,
        uniform_order_probability: float = 0.25,
        seed: int = 20260901,
    ) -> None:
        self.max_length = max_length
        self.exploration_temperature = exploration_temperature
        self.uniform_order_probability = uniform_order_probability
        self.rng = np.random.default_rng(seed)

        usable = [w for w in words if 0 < len(w) <= max_length and set(w) <= set(ALPHABET)]
        if not usable:
            raise ValueError("No usable words: check the corpus and max_length.")
        self.words = usable

        self._codes = self._encode_words(usable, max_length)
        self._counts = self._letter_counts(self._codes)
        self._presence = self._counts > 0
        self._n_distinct = self._presence.sum(axis=1)

        # Bootstrap ranking: P(letter appears in a word), in log space.
        presence_rate = self._presence.mean(axis=0).clip(min=1e-6)
        self._log_weights = np.log(presence_rate).astype(np.float32)

    @staticmethod
    def _encode_words(words: list[str], max_length: int) -> np.ndarray:
        """Pack words into a ``(n_words, max_length)`` letter-index matrix, -1 padded."""
        codes = np.full((len(words), max_length), -1, dtype=np.int8)
        for row, word in enumerate(words):
            letters = np.frombuffer(word.encode("ascii"), dtype=np.uint8) - _LETTER_A
            codes[row, : len(word)] = letters
        return codes

    @staticmethod
    def _letter_counts(codes: np.ndarray) -> np.ndarray:
        """Occurrences of each letter per word, ``(n_words, 26)``."""
        counts = np.zeros((len(codes), N_LETTERS), dtype=np.int16)
        for letter in range(N_LETTERS):
            counts[:, letter] = (codes == letter).sum(axis=1)
        return counts

    def sample(self, batch_size: int) -> StateBatch:
        """Draw ``batch_size`` states from freshly simulated games."""
        indices = self.rng.integers(0, len(self.words), size=batch_size)
        return self.sample_for_words(indices)

    def sample_for_words(self, indices: np.ndarray) -> StateBatch:
        """Simulate one game per word index and sample one state from each."""
        presence = self._presence[indices]
        counts = self._counts[indices].astype(np.float32)
        codes = self._codes[indices]
        n_distinct = self._n_distinct[indices]

        order = self._sample_guess_orders(len(indices))
        turn = self._sample_turn(order, presence, n_distinct)

        # rank[b, letter] = the turn at which that letter would be guessed.
        rank = np.argsort(order, axis=1)
        guessed = rank < turn[:, None]

        revealed = guessed & presence
        hidden = presence & ~guessed

        padding = codes < 0
        safe_codes = np.where(padding, 0, codes).astype(np.int64)
        position_revealed = np.take_along_axis(revealed, safe_codes, axis=1)

        tokens = self._build_boards(safe_codes, padding, position_revealed)

        hidden_counts = counts * hidden
        totals = hidden_counts.sum(axis=1, keepdims=True).clip(min=1.0)
        targets = hidden_counts / totals

        # Supervise only the blanks: revealed positions and padding are ignored.
        letter_targets = np.where(
            padding | position_revealed, IGNORE_INDEX, safe_codes
        ).astype(np.int64)

        return StateBatch(
            tokens=tokens,
            guessed=guessed.astype(np.float32),
            padding=padding,
            targets=targets.astype(np.float32),
            letter_targets=letter_targets,
            present=hidden.astype(np.float32),
        )

    def _sample_guess_orders(self, batch_size: int) -> np.ndarray:
        """Sample a guess order per game via the Gumbel-top-k trick.

        Adding Gumbel noise to log-weights and sorting draws an ordering from the
        Plackett-Luce distribution -- sampling letters without replacement in
        proportion to their weight. That is exactly a frequency-ranked player who
        is sometimes wrong about the ranking.
        """
        weights = np.broadcast_to(self._log_weights, (batch_size, N_LETTERS)).copy()

        # A fraction of games use a flat ranking, giving a uniformly random order.
        uniform_rows = self.rng.random(batch_size) < self.uniform_order_probability
        weights[uniform_rows] = 0.0

        uniforms = self.rng.random((batch_size, N_LETTERS)).clip(1e-9, 1 - 1e-9)
        gumbel = -np.log(-np.log(uniforms))
        scores = weights + self.exploration_temperature * gumbel
        return np.argsort(-scores, axis=1)

    def _sample_turn(
        self, order: np.ndarray, presence: np.ndarray, n_distinct: np.ndarray
    ) -> np.ndarray:
        """Pick a uniformly random turn from each game's live trajectory.

        The game is simulated in closed form: a guess is a hit when the letter is
        present, and the game ends at whichever comes first -- the sixth miss, or
        the guess that reveals the final letter. Sampling is inclusive of the
        terminal turn's *pre-guess* state, which is always still winnable and is
        where the hardest decisions live.
        """
        batch_size = len(order)
        rows = np.arange(batch_size)[:, None]
        hits = presence[rows, order]

        hits_cumulative = np.cumsum(hits, axis=1)
        misses_cumulative = np.cumsum(~hits, axis=1)

        solved_turn = np.argmax(hits_cumulative >= n_distinct[:, None], axis=1)

        out_of_lives = misses_cumulative >= MAX_WRONG_GUESSES
        dead_turn = np.where(
            out_of_lives.any(axis=1),
            np.argmax(out_of_lives, axis=1),
            N_LETTERS - 1,
        )

        last_turn = np.minimum(solved_turn, dead_turn)
        return self.rng.integers(0, last_turn + 1)

    @staticmethod
    def _build_boards(
        safe_codes: np.ndarray, padding: np.ndarray, position_revealed: np.ndarray
    ) -> np.ndarray:
        """Render board token ids from letter indices and per-position visibility."""
        tokens = np.full(safe_codes.shape, MASK_TOKEN_ID, dtype=np.int64)
        np.putmask(tokens, position_revealed, safe_codes + LETTER_TOKEN_OFFSET)
        tokens[padding] = PAD_TOKEN_ID
        return tokens
