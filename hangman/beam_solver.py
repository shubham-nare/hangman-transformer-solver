"""Posterior over word completions by beam search under a character LM.

The banned dictionary approach was strong for one reason: it did exact posterior
inference over a hypothesis set. It filtered a word list to those consistent
with the board, then ranked letters by how many surviving candidates contained
them. The word list is illegal here, and useless anyway -- train and test share
no words, so a looked-up candidate can never be the answer.

This reconstructs the hypothesis set *generatively* instead. A character n-gram
trained on the competition words proposes the K most plausible English-like
strings consistent with the board, and letters are ranked over that beam by the
same posterior rule:

    P(letter hits) = sum_{w in beam : letter in w} P(w)  /  sum_{w in beam} P(w)

No independence assumption between positions -- which is the point. Per-position
marginals are adequate when eleven blanks remain and nearly worthless when two
do, and the measured loss profile puts 69% of defeats in that second regime.

Constraint handling is exact and follows from the rules: a guess reveals *every*
occurrence of its letter, so any letter already guessed -- hit or miss -- cannot
occupy a remaining blank.
"""

from __future__ import annotations

import math
from typing import Sequence

from .char_lm import BOW, EOW, CharKN
from .game import ALPHABET

_A = ord("a")


class BeamPosterior:
    """Ranks letters by posterior hit-probability over beam-searched completions.

    Parameters
    ----------
    lm:
        A fitted :class:`CharKN`.
    beam_width:
        Number of partial completions retained per position. Larger is a closer
        approximation to the full posterior and costs linearly more.
    max_blanks:
        Above this many unknown positions the beam is too thin a slice of a very
        large space to be trustworthy, and the caller should prefer the neural
        marginals. ``None`` disables the guard.
    """

    def __init__(
        self,
        lm: CharKN,
        *,
        beam_width: int = 400,
        max_blanks: int | None = 8,
    ) -> None:
        self.lm = lm
        self.beam_width = beam_width
        self.max_blanks = max_blanks
        self._ctx = lm.order - 1

    # ------------------------------------------------------------------
    def applicable(self, board: str, guessed: frozenset[str]) -> bool:
        """Whether the beam is worth running for this state."""
        blanks = board.count("_")
        if blanks == 0:
            return False
        if self.max_blanks is not None and blanks > self.max_blanks:
            return False
        return True

    def hit_probabilities(
        self,
        board: str,
        guessed: frozenset[str],
        position_log_probs=None,
        nn_weight: float = 0.0,
    ) -> tuple[list[float], int]:
        """Return ``(P(hit) per letter, beam size)`` for the 26 letters.

        Letters already guessed get probability 0 -- re-guessing is a strike.

        ``position_log_probs`` and ``nn_weight`` mirror
        :meth:`FastBeamPosterior.hit_probabilities` so the two are drop-in
        interchangeable, which is what lets the fast implementation be verified
        against this one. They add the transformer's per-position evidence to
        each candidate during the search. Measured at ``nn_weight`` 0.3-1.0 this
        made results monotonically worse -- it double-counts a signal already
        present in the final blend -- so the shipped configuration leaves it at
        0 and this branch is inert.
        """
        allowed = [c for c in ALPHABET if c not in guessed]
        if not allowed:
            return [0.0] * 26, 0
        use_nn = position_log_probs is not None and nn_weight != 0.0

        # Beam entries: (context_tail, logprob, letters_used_bitmask)
        beams: list[tuple[str, float, int]] = [(BOW * self._ctx, 0.0, 0)]

        for pos, ch in enumerate(board):
            nxt: list[tuple[str, float, int]] = []
            if ch != "_":
                # Revealed position: the character is known, no branching.
                for ctx, lp, mask in beams:
                    nxt.append(
                        (
                            (ctx + ch)[-self._ctx :] if self._ctx else "",
                            lp + self.lm.logprob_char(ctx, ch),
                            mask,
                        )
                    )
            else:
                for ctx, lp, mask in beams:
                    for cand in allowed:
                        score = lp + self.lm.logprob_char(ctx, cand)
                        if use_nn:
                            score += nn_weight * float(
                                position_log_probs[pos, ord(cand) - _A]
                            )
                        nxt.append(
                            (
                                (ctx + cand)[-self._ctx :] if self._ctx else "",
                                score,
                                mask | (1 << (ord(cand) - _A)),
                            )
                        )
                if len(nxt) > self.beam_width:
                    nxt.sort(key=lambda t: t[1], reverse=True)
                    del nxt[self.beam_width :]
            beams = nxt
            if not beams:
                return [0.0] * 26, 0

        # End-of-word: a completion that cannot end here is implausible, and this
        # is what makes the model sensitive to word length.
        scored = [(lp + self.lm.logprob_char(ctx, EOW), mask) for ctx, lp, mask in beams]

        top = max(s for s, _ in scored)
        weights = [math.exp(s - top) for s, _ in scored]
        total = sum(weights)
        if total <= 0.0:
            return [0.0] * 26, len(scored)

        hits = [0.0] * 26
        for w, (_, mask) in zip(weights, scored):
            m = mask
            while m:
                low = m & -m
                hits[low.bit_length() - 1] += w
                m ^= low

        return [h / total for h in hits], len(scored)
