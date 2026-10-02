"""Vocabulary-guided candidate filtering for Hangman.

Uses numpy boolean-mask filtering grouped by word length. Given a Hangman
board state (revealed pattern + wrong letters), filters down to matching
candidates in O(L * n_words_of_length_L) numpy operations — fast enough that
250 k games with full hybrid inference runs in seconds, not minutes.

Core idea:
  Instead of guessing which letter *exists*, guess the letter that reveals
  the *most positions* in expectation — marginalised over a word posterior
  that uses the neural position head to weight candidates by plausibility.

      Q(letter) = E[revealed positions | letter]
                = sum_w  P(w | state) * count_unrevealed(letter, w)

  P(w | state) proportional to  exp( sum_{blank i} log P(w_i | position_head[i]) / T )

  The model runs ONCE per game state (not once per candidate): position logits
  come from the neural branch at no extra GPU cost.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

_A = ord("a")
_N = 26  # alphabet size


class VocabEngine:
    """Numpy-indexed vocabulary with neural-reweighted word posteriors.

    Parameters
    ----------
    words:
        Sequence of lowercase a-z words.  Duplicates are deduplicated.
    """

    def __init__(self, words: Sequence[str]) -> None:
        # Deduplicate preserving order
        seen: set[str] = set()
        self._words: list[str] = []
        for w in words:
            if w not in seen:
                seen.add(w)
                self._words.append(w)

        n = len(self._words)
        self._max_len = max((len(w) for w in self._words), default=0)

        # Global letter array (n, max_len) int8, -1 for padding
        self._letter_arrays = np.full((n, max(self._max_len, 1)), -1, dtype=np.int8)
        for i, word in enumerate(self._words):
            for j, ch in enumerate(word):
                self._letter_arrays[i, j] = ord(ch) - _A

        # Per-length numpy index structures
        # _ids[L]        : (n_L,) int32 — global word-ids for words of length L
        # _pos[L]        : (n_L, L) int8 — letter index at each position
        # _contains[L]   : (n_L, 26) bool — True if word contains that letter
        self._ids: dict[int, np.ndarray] = {}
        self._pos: dict[int, np.ndarray] = {}
        self._contains: dict[int, np.ndarray] = {}

        groups: dict[int, list[int]] = defaultdict(list)
        for word_id, word in enumerate(self._words):
            groups[len(word)].append(word_id)

        for L, id_list in groups.items():
            ids_arr = np.array(id_list, dtype=np.int32)
            self._ids[L] = ids_arr

            sub = self._letter_arrays[id_list, :L]    # (n_L, L)
            self._pos[L] = sub

            # contains[j, ci] = any(sub[j] == ci)
            contains = np.zeros((len(id_list), _N), dtype=bool)
            for ci in range(_N):
                contains[:, ci] = (sub == ci).any(axis=1)
            self._contains[L] = contains

        # State cache: (pattern, wrong_frozenset) -> (n_cands, letter_arr, global_ids)
        # We cache the NUMPY arrays so score_letters avoids re-filtering.
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        # value: (pos_letters (n_cands, L), global_ids (n_cands,))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_candidates_np(
        self,
        pattern: str,
        wrong_letters: frozenset[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_letters, global_ids) arrays for matching candidates.

        pos_letters: (n_cands, L) int8 — letter at each position
        global_ids:  (n_cands,) int32 — indices into self._words
        Both are empty (length 0) if no candidates found.
        """
        key = (pattern, wrong_letters)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        L = len(pattern)
        empty = (np.empty((0, L), dtype=np.int8), np.empty(0, dtype=np.int32))

        if L not in self._ids:
            self._cache[key] = empty
            return empty

        ids_L = self._ids[L]
        pos_L = self._pos[L]          # (n_L, L)
        cont_L = self._contains[L]    # (n_L, 26)
        mask = np.ones(len(ids_L), dtype=bool)

        # Filter by revealed positions
        for p, ch in enumerate(pattern):
            if ch != "_":
                ci = ord(ch) - _A
                mask &= pos_L[:, p] == ci
                if not mask.any():
                    self._cache[key] = empty
                    return empty

        # Filter out words containing any wrong letter
        for ch in wrong_letters:
            if "a" <= ch <= "z":
                mask &= ~cont_L[:, ord(ch) - _A]
                if not mask.any():
                    self._cache[key] = empty
                    return empty

        result = (pos_L[mask], ids_L[mask])
        self._cache[key] = result
        return result

    def get_candidate_ids(
        self,
        pattern: str,
        wrong_letters: frozenset[str],
    ) -> list[int]:
        """Return list of global word-ids consistent with the current state."""
        _, gids = self._get_candidates_np(pattern, wrong_letters)
        return gids.tolist()

    def score_letters(
        self,
        pattern: str,
        wrong_letters: frozenset[str],
        guessed_letters: frozenset[str],
        position_log_probs: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        criterion: str = "reveals",
    ) -> tuple[np.ndarray, int]:
        """Compute Q(letter) = E[revealed positions | letter] under word posterior.

        The neural model is run once per game state (not once per candidate).
        Position log-probs at blank positions reweight candidates: words whose
        characters match the neural distribution score higher.

        Parameters
        ----------
        pattern:
            Board string, e.g. ``"_a__"`` — ``_`` for blanks.
        wrong_letters:
            Letters guessed that did not reveal any position.
        guessed_letters:
            All guessed letters (hits + misses). Q is zeroed at these indices.
        position_log_probs:
            ``(word_length, 26)`` log-softmax from the position head.  When
            ``None``, a uniform word prior is used (letter-frequency baseline).
        temperature:
            Softmax temperature on candidate scores.  Lower -> sharper reweight.
        criterion:
            ``"reveals"`` scores E[number of positions revealed]; ``"hit"`` scores
            P(letter appears at least once).  With only six lives the binding
            constraint is avoiding a strike, not uncovering many squares at once,
            so ``"hit"`` optimises survival while ``"reveals"`` optimises speed
            and over-rewards letters that happen to repeat.

        Returns
        -------
        Q : np.ndarray
            Shape ``(26,)``. Q[i] = expected revealed positions if letter i is
            guessed next. Zero at already-guessed letter indices.
        n_candidates : int
            Number of vocabulary words consistent with the current board.
        """
        pos_letters, _ = self._get_candidates_np(pattern, wrong_letters)
        n_cands = len(pos_letters)
        Q = np.zeros(_N, dtype=np.float32)
        if n_cands == 0:
            return Q, 0

        L = len(pattern)
        blank_positions = [i for i, ch in enumerate(pattern) if ch == "_"]
        n_blanks = len(blank_positions)
        if n_blanks == 0:
            return Q, n_cands

        blank_arr = np.array(blank_positions, dtype=np.int32)
        # cand_at_blanks: (n_cands, n_blanks) int8 — letter at each blank position
        cand_at_blanks = pos_letters[:, blank_arr]

        # Score each candidate using position log-probs at blank positions
        if position_log_probs is not None:
            blank_lp = position_log_probs[blank_arr].astype(np.float32)  # (n_blanks, 26)
            k_idx = np.arange(n_blanks, dtype=np.int32)
            # scores[j] = sum_k blank_lp[k, cand_at_blanks[j, k]]
            scores = blank_lp[k_idx[None, :], cand_at_blanks].sum(axis=1)  # (n_cands,)
        else:
            scores = np.zeros(n_cands, dtype=np.float32)

        # Numerically stable softmax -> word posterior P(w|state)
        scores = np.where(np.isfinite(scores), scores, -1e9)
        scores = scores - scores.max()
        word_probs = np.exp(np.clip(scores / temperature, -80, 0))
        word_probs /= word_probs.sum()

        # Count matrix: count_matrix[j, ci] = occurrences of ci in blanks of word j
        # Build via one-hot: (n_cands, n_blanks, 26)
        one_hot = np.eye(_N, dtype=np.float32)[cand_at_blanks]  # (n_cands, n_blanks, 26)
        count_matrix = one_hot.sum(axis=1)                       # (n_cands, 26)

        if criterion == "hit":
            # P(letter appears at least once) -- collapses repeats to a single
            # unit of credit, so the score is the chance of avoiding a strike.
            score_matrix = (count_matrix > 0).astype(np.float32)
        elif criterion == "reveals":
            score_matrix = count_matrix
        else:
            raise ValueError(
                f"criterion must be 'reveals' or 'hit', got {criterion!r}."
            )

        Q = (word_probs[:, None] * score_matrix).sum(axis=0)     # (26,)

        # Zero already-guessed letters
        for ch in guessed_letters:
            if "a" <= ch <= "z":
                Q[ord(ch) - _A] = 0.0

        return Q, n_cands

    def __len__(self) -> int:
        return len(self._words)
