"""Batched beam-search posterior on the GPU.

ISOLATED EXPERIMENT. Nothing here is imported by the shipped pipeline; the
submitted result comes from ``hangman/fast_beam.py`` on the CPU. This directory
exists so the idea can be tried without risk to a validated 77.1204%.

Why bother
----------
The CPU beam keeps 200 candidates. At two blanks that is 200 of 324 (62%), at
three blanks 200 of 5,832 -- **3.4%**. Measured on 15,047 real game states,
that truncation changes the chosen letter on 1.615% of decisions, with
probability shifts up to 0.79. It is a real approximation error, concentrated
one turn before the endgame where most losses are decided.

Widening the beam on the CPU is linear in width and the beam is already the
bottleneck: 250k games take ~14 minutes at width 200, so width 5,000 would take
hours. The GPU changes that arithmetic, but only if the work is batched across
*games* rather than done one game at a time -- a per-state kernel launch on 2.2M
states would be dominated by launch overhead.

So this processes a whole chunk of games in lockstep. Every game advances
through board positions together, with per-game masks handling the fact that
boards differ in length, in which positions are revealed, and in which letters
remain available.

Layout
------
Symbols 0-25 are ``a``-``z``; 26 is the boundary symbol (beginning-of-word on
input, end-of-word on output). Contexts are base-27 integers, exactly as in the
CPU implementation, so the same lookup tables serve both and results can be
compared directly.
"""

from __future__ import annotations

import numpy as np
import torch

_A = ord("a")
NSYM = 27
BOUND = 26


class GpuBeamPosterior:
    """Beam-search posterior over word completions, batched across games.

    Parameters
    ----------
    index, table:
        The sparse two- or three-level lookup produced by
        ``hangman.fast_beam.build_sparse_table*``. ``index`` maps a context id
        to a row of ``table``. Pass ``index=None`` for a dense table.
    order:
        Language-model order the table was built for.
    beam_width:
        Candidates retained per game per position. The whole point of moving to
        the GPU is that this can be far larger than the CPU's 200.
    max_blanks:
        Skip games with more unknown positions than this -- the beam covers too
        thin a slice of the space to be informative there, and the neural
        marginals are strong in that regime.
    device:
        Where the tables live. They are moved once at construction, not per call.
    """

    def __init__(
        self,
        index: np.ndarray | None,
        table: np.ndarray,
        order: int,
        *,
        beam_width: int = 1000,
        max_blanks: int | None = 10,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.ctx_len = order - 1
        self.beam_width = beam_width
        self.max_blanks = max_blanks
        self._pow = NSYM ** (self.ctx_len - 1) if self.ctx_len > 0 else 0

        self.table = torch.from_numpy(table).to(self.device)
        self.index = (
            torch.from_numpy(index).to(self.device) if index is not None else None
        )

        init = 0
        for _ in range(self.ctx_len):
            init = init * NSYM + BOUND
        self._init_ctx = init

    def _rows(self, ctx: torch.Tensor) -> torch.Tensor:
        """Log-probability rows for a tensor of context ids, any shape."""
        if self.index is None:
            return self.table[ctx]
        return self.table[self.index[ctx]]

    @torch.no_grad()
    def hit_probabilities_batch(
        self, boards: list[str], guessed_sets: list[frozenset[str]]
    ) -> np.ndarray:
        """Return ``(batch, 26)`` P(letter appears) for a chunk of game states.

        Games whose blank count exceeds ``max_blanks`` come back as all zeros;
        the caller is expected to fall back to the neural marginals for those,
        matching the CPU implementation's ``applicable`` guard.
        """
        batch = len(boards)
        out = np.zeros((batch, 26), dtype=np.float64)
        if batch == 0:
            return out

        # Which games the beam should actually run for.
        live = [
            i for i, (b, g) in enumerate(zip(boards, guessed_sets))
            if b.count("_") > 0
            and (self.max_blanks is None or b.count("_") <= self.max_blanks)
            and len(g) < 26
        ]
        if not live:
            return out

        max_len = max(len(boards[i]) for i in live)
        n = len(live)
        dev = self.device

        # Per-game board symbols (-1 marks a blank) and padding beyond the word.
        board_sym = torch.full((n, max_len), -2, dtype=torch.long, device=dev)
        allowed = torch.zeros((n, 26), dtype=torch.bool, device=dev)
        for row, i in enumerate(live):
            b = boards[i]
            for p, ch in enumerate(b):
                if ch == "_":
                    board_sym[row, p] = -1
                else:
                    s = ord(ch) - _A
                    board_sym[row, p] = s if 0 <= s < 26 else BOUND
            for c in range(26):
                if chr(_A + c) not in guessed_sets[i]:
                    allowed[row, c] = True

        K = self.beam_width
        ctx = torch.full((n, 1), self._init_ctx, dtype=torch.long, device=dev)
        score = torch.zeros((n, 1), dtype=torch.float32, device=dev)
        # 26-bit membership per candidate, as a bool matrix rather than a bitmask
        # so the final tally is a single matmul.
        used = torch.zeros((n, 1, 26), dtype=torch.bool, device=dev)

        neg_inf = torch.finfo(torch.float32).min

        for p in range(max_len):
            sym = board_sym[:, p]                       # (n,)
            rows = self._rows(ctx)                      # (n, K, 27)

            revealed = sym >= 0                         # known character here
            blank = sym == -1
            beyond = sym == -2                          # past this word's end

            # --- revealed positions: no branching, just accumulate ---
            if revealed.any():
                idx = sym.clamp(min=0).view(n, 1, 1).expand(-1, rows.shape[1], 1)
                add = rows.gather(2, idx).squeeze(2)    # (n, K)
                score = torch.where(revealed.view(n, 1), score + add, score)
                new_ctx = (ctx % self._pow) * NSYM + sym.view(n, 1).clamp(min=0)
                ctx = torch.where(revealed.view(n, 1), new_ctx, ctx)

            # --- blanks: expand over every still-available letter ---
            if blank.any():
                cand = score.unsqueeze(2) + rows[:, :, :26]        # (n, K, 26)
                cand = cand.masked_fill(~allowed.view(n, 1, 26), neg_inf)
                # Games not expanding at this position keep their beam intact:
                # give slot 0 the current score and forbid the rest.
                keep = torch.full_like(cand, neg_inf)
                keep[:, :, 0] = score
                cand = torch.where(blank.view(n, 1, 1), cand, keep)

                flat = cand.reshape(n, -1)                          # (n, K*26)
                k = min(K, flat.shape[1])
                top, pos = torch.topk(flat, k, dim=1)               # (n, k)
                src = pos // 26
                letter = pos % 26

                score = top
                parent_ctx = ctx.gather(1, src)
                stepped = (parent_ctx % self._pow) * NSYM + letter
                ctx = torch.where(blank.view(n, 1), stepped, parent_ctx)

                used = used.gather(
                    1, src.unsqueeze(2).expand(-1, -1, 26)
                ).clone()
                # Only record the letter for games that actually branched here.
                hit = torch.zeros_like(used)
                hit.scatter_(2, letter.unsqueeze(2), True)
                used |= hit & blank.view(n, 1, 1)

            if beyond.all():
                break

        # End-of-word probability: what makes the model length-aware.
        rows = self._rows(ctx)
        score = score + rows[:, :, BOUND]
        # Candidates that were never real (padding fill) sit at -inf and vanish
        # in the softmax below.
        score = score - score.max(dim=1, keepdim=True).values
        weight = torch.exp(score)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-30)

        hits = torch.einsum("nk,nkc->nc", weight, used.to(weight.dtype))
        out[live] = hits.double().cpu().numpy()
        return out
