"""Vectorised beam-search posterior over word completions.

Same inference as :mod:`hangman.beam_solver`, rebuilt for throughput. The pure
Python version costs ~0.27 s per word, which is 18 hours over the 250k-word
evaluation -- unusable. Here the language model is flattened into a dense
``(n_contexts, 27)`` log-probability table so a beam step becomes one fancy-index
and one ``argpartition``, and the whole search runs in numpy.

Symbol layout: 0-25 are ``a``-``z``, 26 is beginning-of-word padding on the input
side and end-of-word on the output side. Contexts are base-27 integers over the
last ``order-1`` symbols, so the table is ``27**(order-1)`` rows -- 531,441 rows
at order 5, about 57 MB in float32.
"""

from __future__ import annotations

import math

import numpy as np

from .char_lm import BOW, EOW, CharKN

_A = ord("a")
_NSYM = 27          # 26 letters + boundary symbol
_BOUND = 26         # index of BOW (input) / EOW (output)


def build_table(lm: CharKN) -> np.ndarray:
    """Flatten ``lm`` into a dense ``(27**(order-1), 27)`` log-prob table."""
    ctx_len = lm.order - 1
    n_ctx = _NSYM ** ctx_len
    table = np.empty((n_ctx, _NSYM), dtype=np.float32)

    symbols = [chr(_A + i) for i in range(26)] + [BOW]
    out_symbols = [chr(_A + i) for i in range(26)] + [EOW]

    # Decode each context index into its symbol string once.
    for ctx_id in range(n_ctx):
        rem = ctx_id
        chars = []
        for _ in range(ctx_len):
            chars.append(symbols[rem % _NSYM])
            rem //= _NSYM
        ctx = "".join(reversed(chars))
        for out_id, out_ch in enumerate(out_symbols):
            table[ctx_id, out_id] = lm.logprob_char(ctx, out_ch)
    return table


def build_sparse_table(lm: CharKN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-level table for an order the dense form cannot hold.

    At order 6 the dense form needs 27**5 = 14.3M rows (~1.5 GB), but only about
    242k contexts actually occur in 225k words -- 0.06% of the space. So the top
    level stores just the observed contexts, and everything else defers to the
    order-below table.

    That deferral is exact, not an approximation: Kneser-Ney's recursion finds an
    empty table for an unobserved context and backs off to the shorter one, so
    the shorter table already holds the value the model would compute.

    Returns ``(index, table)``: ``table`` stacks the observed-context rows on top
    of the order-below rows, and ``index`` maps any context id straight to its
    row in that single array. Resolving the fallback at build time rather than at
    lookup time matters -- a lookup costs one random gather instead of three, and
    at 250k games these gathers are the dominant cost, being cache-hostile random
    access across ~140 MB.
    """
    ctx_len = lm.order - 1
    n_full = _NSYM ** ctx_len

    observed = sorted(lm._counts[ctx_len].keys())
    out_symbols = [chr(_A + i) for i in range(26)] + [EOW]

    top = np.empty((len(observed), _NSYM), dtype=np.float32)
    observed_ids = np.empty(len(observed), dtype=np.int64)

    for row, ctx in enumerate(observed):
        ctx_id = 0
        for ch in ctx:
            ctx_id = ctx_id * _NSYM + (_BOUND if ch == BOW else ord(ch) - _A)
        observed_ids[row] = ctx_id
        for out_id, out_ch in enumerate(out_symbols):
            top[row, out_id] = lm.logprob_char(ctx, out_ch)

    # One order down, dense -- what unobserved contexts resolve to. This must be
    # built with top=False: Kneser-Ney backs off to the *continuation* counts
    # (how many distinct contexts a suffix completes), not the raw counts. Using
    # the raw-count distribution here is the classic KN mistake and it measurably
    # shifts the probabilities.
    base_len = ctx_len - 1
    base = np.empty((_NSYM ** base_len, _NSYM), dtype=np.float32)
    symbols = [chr(_A + i) for i in range(26)] + [BOW]
    for ctx_id in range(base.shape[0]):
        rem = ctx_id
        chars = []
        for _ in range(base_len):
            chars.append(symbols[rem % _NSYM])
            rem //= _NSYM
        ctx = "".join(reversed(chars))
        for out_id, out_ch in enumerate(out_symbols):
            p = lm._prob(ctx, out_ch, base_len, top=False)
            base[ctx_id, out_id] = math.log(max(p, 1e-300))

    # Stack both levels into one array and bake the fallback into the index, so
    # a lookup is a single gather. Unobserved contexts point at their order-below
    # suffix row, which is the base-27 context id modulo the smaller table size.
    n_top = top.shape[0]
    table = np.concatenate([top, base], axis=0)
    index = (
        np.arange(n_full, dtype=np.int64) % base.shape[0]
    ).astype(np.int32) + np.int32(n_top)
    index[observed_ids] = np.arange(n_top, dtype=np.int32)
    return index, table


def build_sparse_table_order7(lm: CharKN) -> tuple[np.ndarray, np.ndarray]:
    """Three-level table for order 7, whose context space is 27**6 = 387M.

    Order 6 needed two levels: observed contexts, and one dense fallback. Order 7
    cannot reach a dense fallback in one hop -- the level below it is itself
    27**5 = 14.3M rows -- so it chains twice: observed 6-character contexts, then
    observed 5-character contexts, then a dense 4-character table underneath.

    Each hop is exact for the same reason the two-level version was: Kneser-Ney
    backing off from an unobserved context computes precisely the shorter-context
    value, so pointing at that row is the model's own answer rather than an
    approximation.

    The index alone is 387M int32 (1.55 GB), so it is filled in chunks. Building
    it the obvious way -- ``np.arange(27**6)`` in int64, then a modulo -- would
    momentarily need about 7 GB and take the machine down.
    """
    ctx_len = lm.order - 1                      # 6 for order 7
    if ctx_len < 3:
        raise ValueError("use build_sparse_table for orders below 7")
    n_full = _NSYM ** ctx_len
    out_symbols = [chr(_A + i) for i in range(26)] + [EOW]
    symbols = [chr(_A + i) for i in range(26)] + [BOW]

    def context_id(ctx: str) -> int:
        value = 0
        for ch in ctx:
            value = value * _NSYM + (_BOUND if ch == BOW else ord(ch) - _A)
        return value

    def rows_for(observed: list[str], top: bool) -> np.ndarray:
        table = np.empty((len(observed), _NSYM), dtype=np.float32)
        for row, ctx in enumerate(observed):
            k = len(ctx)
            for out_id, out_ch in enumerate(out_symbols):
                p = lm._prob(ctx, out_ch, k, top=top)
                table[row, out_id] = math.log(max(p, 1e-300))
        return table

    # Level A: contexts of full length that were actually observed.
    obs_a = sorted(lm._counts[ctx_len].keys())
    tab_a = rows_for(obs_a, top=True)
    ids_a = np.fromiter((context_id(c) for c in obs_a), dtype=np.int64,
                        count=len(obs_a))

    # Level B: one character shorter, scored with continuation counts because
    # this level is only ever reached by backing off.
    obs_b = sorted(lm._counts[ctx_len - 1].keys())
    tab_b = rows_for(obs_b, top=False)
    ids_b = np.fromiter((context_id(c) for c in obs_b), dtype=np.int64,
                        count=len(obs_b))

    # Level C: dense at two characters shorter -- small enough to enumerate.
    base_len = ctx_len - 2
    n_base = _NSYM ** base_len
    tab_c = np.empty((n_base, _NSYM), dtype=np.float32)
    for ctx_id in range(n_base):
        rem, chars = ctx_id, []
        for _ in range(base_len):
            chars.append(symbols[rem % _NSYM])
            rem //= _NSYM
        ctx = "".join(reversed(chars))
        for out_id, out_ch in enumerate(out_symbols):
            p = lm._prob(ctx, out_ch, base_len, top=False)
            tab_c[ctx_id, out_id] = math.log(max(p, 1e-300))

    table = np.concatenate([tab_a, tab_b, tab_c], axis=0)
    off_b, off_c = len(obs_a), len(obs_a) + len(obs_b)

    # Default every context to its level-C row, in chunks to bound peak memory.
    index = np.empty(n_full, dtype=np.int32)
    chunk = 20_000_000
    for start in range(0, n_full, chunk):
        end = min(start + chunk, n_full)
        ids = np.arange(start, end, dtype=np.int64)
        index[start:end] = (off_c + (ids % n_base)).astype(np.int32)

    # Level B: every full-length context sharing an observed shorter suffix.
    # There are _NSYM of them per suffix, formed by prepending each symbol.
    stride = _NSYM ** (ctx_len - 1)
    prefixes = np.arange(_NSYM, dtype=np.int64)[:, None] * stride
    targets = (prefixes + ids_b[None, :]).ravel()
    index[targets] = np.tile(
        np.arange(off_b, off_b + len(obs_b), dtype=np.int32), _NSYM
    )

    # Level A last, so observed full-length contexts win.
    index[ids_a] = np.arange(len(obs_a), dtype=np.int32)
    return index, table


def _initial_context(ctx_len: int) -> int:
    """Context of all-BOW padding, as a base-27 integer."""
    value = 0
    for _ in range(ctx_len):
        value = value * _NSYM + _BOUND
    return value


class FastBeamPosterior:
    """Numpy beam search returning P(letter hits) over completions.

    Parameters
    ----------
    table:
        Dense log-prob table from :func:`build_table`.
    order:
        The order of the language model the table came from.
    beam_width:
        Partial completions retained per position.
    max_blanks:
        Skip the beam above this many unknown positions -- the search covers too
        thin a slice of the space to be informative, and the neural marginals are
        strong in that regime anyway. ``None`` disables the guard.
    """

    def __init__(
        self,
        table: np.ndarray,
        order: int,
        *,
        beam_width: int = 200,
        max_blanks: int | None = 10,
        exact_below: int = 0,
    ) -> None:
        self.table = table
        self.ctx_len = order - 1
        self.beam_width = beam_width
        self.max_blanks = max_blanks
        # When the whole hypothesis space is smaller than this, keep all of it
        # instead of pruning to beam_width. Two blanks with twenty letters still
        # available is 400 completions against a 200-wide beam -- half the space
        # discarded precisely in the endgame, and 69% of losses end with two or
        # fewer blanks left. Enumerating exhaustively there is cheap and makes
        # the posterior exact rather than truncated. 0 disables.
        self.exact_below = exact_below
        self._pow = _NSYM ** (self.ctx_len - 1) if self.ctx_len > 0 else 0
        self._init_ctx = _initial_context(self.ctx_len)

    def applicable(self, board: str, guessed: frozenset[str]) -> bool:
        blanks = board.count("_")
        if blanks == 0:
            return False
        if self.max_blanks is not None and blanks > self.max_blanks:
            return False
        return True

    def _rows(self, ctx: np.ndarray) -> np.ndarray:
        """Log-prob rows for a batch of context ids."""
        return self.table[ctx]

    def hit_probabilities(
        self,
        board: str,
        guessed: frozenset[str],
        position_log_probs: np.ndarray | None = None,
        nn_weight: float = 0.0,
    ) -> tuple[np.ndarray, int]:
        """Return ``(P(hit) for each of 26 letters, beam size)``.

        When ``position_log_probs`` (shape ``(len(board), 26)``) is supplied, each
        completion is scored by both models jointly:

            score(w) = log P_lm(w) + nn_weight * sum_{blank i} log P_nn(w_i | i)

        The neural term enters *during* the search rather than as a rescoring
        pass afterwards, so pruning is guided by both models and a completion the
        transformer favours is never discarded before it can be scored. It costs
        nothing extra -- the term accumulates one position at a time, exactly as
        the language-model term does.
        """
        allowed = np.array(
            [i for i in range(26) if chr(_A + i) not in guessed], dtype=np.int64
        )
        if allowed.size == 0:
            return np.zeros(26, dtype=np.float64), 0
        use_nn = position_log_probs is not None and nn_weight != 0.0

        # Widen to the full space when it is small enough to hold entirely, so
        # the endgame posterior is exact rather than a truncated approximation.
        width = self.beam_width
        if self.exact_below:
            n_blanks = board.count("_")
            if n_blanks and allowed.size ** n_blanks <= self.exact_below:
                width = int(allowed.size ** n_blanks)

        ctx = np.array([self._init_ctx], dtype=np.int64)
        score = np.zeros(1, dtype=np.float64)
        # Bitmask of letters used by each beam entry, for the presence tally.
        mask = np.zeros(1, dtype=np.int64)

        for pos, ch in enumerate(board):
            if ch != "_":
                sym = ord(ch) - _A
                if not 0 <= sym < 26:
                    # Non-letter positions are revealed from the start and carry
                    # no guessable information; treat as boundary context.
                    sym = _BOUND
                score = score + self._rows(ctx)[:, sym]
                ctx = (ctx % self._pow) * _NSYM + sym if self.ctx_len > 0 else ctx
            else:
                # (beam, n_allowed) expansion
                cand = score[:, None] + self._rows(ctx)[:, allowed]
                if use_nn:
                    # Per-position neural evidence for this blank, broadcast over
                    # the beam. Folded in before pruning, not after.
                    cand = cand + nn_weight * position_log_probs[pos, allowed][None, :]
                flat = cand.ravel()
                k = min(width, flat.size)
                if flat.size > k:
                    top = np.argpartition(flat, -k)[-k:]
                else:
                    top = np.arange(flat.size)
                rows, cols = np.divmod(top, allowed.size)
                syms = allowed[cols]
                score = flat[top]
                ctx = (
                    (ctx[rows] % self._pow) * _NSYM + syms
                    if self.ctx_len > 0
                    else ctx[rows]
                )
                mask = mask[rows] | (np.int64(1) << syms)

        # End-of-word probability: this is what makes the model length-aware.
        score = score + self._rows(ctx)[:, _BOUND]

        score -= score.max()
        weight = np.exp(score)
        total = weight.sum()
        if not np.isfinite(total) or total <= 0.0:
            return np.zeros(26, dtype=np.float64), int(score.size)

        bits = (mask[:, None] >> np.arange(26)[None, :]) & 1
        hits = (weight[:, None] * bits).sum(axis=0) / total
        return hits, int(score.size)


class SparseBeamPosterior(FastBeamPosterior):
    """:class:`FastBeamPosterior` over a two-level table (see build_sparse_table).

    Behaves identically; only the row lookup differs, resolving unobserved
    contexts to the order-below table.
    """

    def __init__(
        self,
        index: np.ndarray,
        table: np.ndarray,
        order: int,
        *,
        beam_width: int = 200,
        max_blanks: int | None = 10,
        exact_below: int = 0,
    ) -> None:
        super().__init__(table, order, beam_width=beam_width,
                         max_blanks=max_blanks, exact_below=exact_below)
        self.index = index

    def _rows(self, ctx: np.ndarray) -> np.ndarray:
        # One gather through the index, one through the table. The fallback was
        # resolved when the index was built.
        return self.table[self.index[ctx]]
