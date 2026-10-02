"""Character-level n-gram language model with Kneser-Ney smoothing.

Trained on the competition word list alone. Its job is to assign a probability
to character strings the model has never seen -- which is the whole task here,
since train and test share no words. Kneser-Ney is the right smoothing for that:
its lower-order distributions use *continuation counts* (how many distinct
contexts a suffix completes) rather than raw frequency, so a common character
that appears in only one context is not mistaken for a versatile one.

The model exists to support exact posterior inference. Given a board, the set of
strings consistent with it is enumerable whenever few positions are unknown, and
then

    P(letter hits) = sum_{s : letter in s} P(s)  /  sum_s P(s)

is exact -- no independence assumption between positions. That is precisely the
regime (short words, endgames) where per-position marginals break down.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

BOW = "<"   # beginning-of-word padding symbol
EOW = ">"   # end-of-word symbol: lets the model score word length


class CharKN:
    """Interpolated modified-count Kneser-Ney over characters.

    Parameters
    ----------
    order:
        Maximum n-gram order. 5 means up to 4 characters of context.
    discount:
        Absolute discount subtracted from each count.
    """

    def __init__(self, order: int = 5, discount: float = 0.75) -> None:
        if order < 2:
            raise ValueError(f"order must be >= 2, got {order}.")
        self.order = order
        self.discount = discount
        # counts[k] maps a k-length context -> {next_char: count}
        self._counts: list[dict[str, dict[str, float]]] = [
            defaultdict(lambda: defaultdict(float)) for _ in range(order)
        ]
        # continuation counts for the lower orders
        self._cont: list[dict[str, dict[str, float]]] = [
            defaultdict(lambda: defaultdict(float)) for _ in range(order)
        ]
        self._vocab: set[str] = set()
        self._trained = False
        self._cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    def fit(self, words: Iterable[str]) -> "CharKN":
        """Accumulate n-gram statistics from a word list."""
        pad = BOW * (self.order - 1)
        for word in words:
            self._vocab.update(word)
            padded = pad + word + EOW
            for i in range(self.order - 1, len(padded)):
                nxt = padded[i]
                for k in range(self.order):
                    ctx = padded[i - k : i]
                    self._counts[k][ctx][nxt] += 1.0

        # Continuation counts: for order k, how many distinct characters
        # precede each (context, next) pair. Used by the lower-order terms.
        for k in range(self.order - 1, 0, -1):
            for ctx, nxts in self._counts[k].items():
                lower = ctx[1:]
                for nxt in nxts:
                    self._cont[k - 1][lower][nxt] += 1.0

        self._vocab.add(EOW)
        self._trained = True
        return self

    # ------------------------------------------------------------------
    def _prob(self, ctx: str, nxt: str, k: int, *, top: bool) -> float:
        """Interpolated KN probability of ``nxt`` after k characters of ctx."""
        if k == 0:
            table = self._counts[0].get("", {})
            total = sum(table.values())
            if total <= 0:
                return 1.0 / max(len(self._vocab), 1)
            # Uniform floor so a never-seen character is not impossible.
            v = max(len(self._vocab), 1)
            return (table.get(nxt, 0.0) + 1.0) / (total + v)

        table = (self._counts[k] if top else self._cont[k]).get(ctx, None)
        if not table:
            return self._prob(ctx[1:], nxt, k - 1, top=False)

        total = sum(table.values())
        if total <= 0:
            return self._prob(ctx[1:], nxt, k - 1, top=False)

        count = table.get(nxt, 0.0)
        discounted = max(count - self.discount, 0.0)
        n_types = len(table)
        backoff_weight = self.discount * n_types / total
        lower = self._prob(ctx[1:], nxt, k - 1, top=False)
        return discounted / total + backoff_weight * lower

    def logprob_char(self, context: str, char: str) -> float:
        """log P(char | context), using at most ``order-1`` context characters.

        Memoised: beam search re-queries the same (context, char) pairs millions
        of times, and the backoff recursion underneath is not cheap.
        """
        key = context + "\x00" + char
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        k = min(len(context), self.order - 1)
        ctx = context[len(context) - k :] if k else ""
        value = math.log(max(self._prob(ctx, char, k, top=True), 1e-300))
        self._cache[key] = value
        return value

    def logprob_word(self, word: str, *, with_eow: bool = True) -> float:
        """log P(word) under the model, including the end-of-word symbol."""
        pad = BOW * (self.order - 1)
        padded = pad + word + (EOW if with_eow else "")
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            total += self.logprob_char(padded[i - (self.order - 1) : i], padded[i])
        return total

    # ------------------------------------------------------------------
    def release(self) -> None:
        """Drop everything that is dead once a lookup table has been built.

        The count dictionaries and the memoisation cache exist only to answer
        ``logprob_char`` while a table is being constructed. Afterwards they are
        pure overhead -- and large: building an order-6 table queries roughly
        250k contexts by 27 outputs, so the cache alone accumulates millions of
        string-keyed entries. Left in place they hold on the order of a gigabyte,
        which is enough to push the machine into working-set trimming and slow
        the whole run down by an order of magnitude.

        The model is unusable after this; call it only once the table is built.
        """
        self._counts = []
        self._cont = []
        self._cache = {}
        self._trained = False

    def perplexity(self, words: Sequence[str]) -> float:
        """Per-character perplexity on a held-out word list."""
        total_lp = 0.0
        total_chars = 0
        for w in words:
            total_lp += self.logprob_word(w)
            total_chars += len(w) + 1  # +1 for EOW
        if total_chars == 0:
            return float("inf")
        return math.exp(-total_lp / total_chars)
