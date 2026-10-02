"""Neural character language model for the beam-search posterior.

Replaces the Kneser-Ney n-gram with a learned model. Both answer the same
question -- ``P(next character | preceding characters)`` -- but they generalise
differently, and the difference matters here: train and test share no words, so
every context the model meets at evaluation time may be one it never saw during
fitting.

A count-based model can only back off when a context is unseen, discarding the
longer context entirely. A neural model has no such cliff: contexts are
*embedded*, so an unseen one lands near the seen ones it resembles and inherits
a sensible distribution. ``_ough`` and ``_ougl`` are unrelated rows to an n-gram
and neighbouring points to this.

Architecture: character embeddings, concatenated over a fixed context window,
through a residual MLP to a 27-way softmax. The fixed window is deliberate --
it makes the model *tabulatable*. Predictions for all 27**k contexts are
precomputed once into the same dense array the n-gram produced, so the beam
search consumes it unchanged and inference costs a lookup rather than a forward
pass. That keeps a 250k-word evaluation to minutes instead of days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

_A = ord("a")
NSYM = 27          # 26 letters + boundary
BOUND = 26         # beginning-of-word on input, end-of-word on output


@dataclass
class NeuralLMConfig:
    """Sized to train in minutes and to be tabulatable in a single pass."""

    context: int = 4          # characters of history; 27**4 = 531,441 rows
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1


class NeuralCharLM(nn.Module):
    """Causal Transformer over a fixed context window.

    Learned symbol and position embeddings feed a pre-norm Transformer encoder
    with a causal mask, and the representation at the final context position
    predicts the next character. Attention lets the model weigh which part of the
    context matters for a given prediction -- after ``q`` only the immediately
    preceding symbol carries information, whereas after ``un`` the whole prefix
    does -- rather than pushing a flat concatenation through a fixed map.

    The window is fixed on purpose. It is what makes the model *tabulatable*:
    every reachable context can be enumerated once and cached, so beam search
    pays a memory lookup instead of a forward pass. Without that, 250k games
    would need billions of forward passes.
    """

    def __init__(self, config: NeuralLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralLMConfig()
        c = self.config
        self.symbol_embedding = nn.Embedding(NSYM, c.d_model)
        self.position_embedding = nn.Embedding(c.context, c.d_model)
        self.input_norm = nn.LayerNorm(c.d_model)
        self.input_dropout = nn.Dropout(c.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=c.n_layers, norm=nn.LayerNorm(c.d_model),
            enable_nested_tensor=False,
        )
        # Causal mask: position i may attend to positions <= i.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((c.context, c.context), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.output = nn.Linear(c.d_model, NSYM)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, context: Tensor) -> Tensor:
        """``(batch, context)`` symbol ids -> ``(batch, 27)`` next-symbol logits."""
        positions = torch.arange(context.shape[1], device=context.device)
        hidden = self.symbol_embedding(context) + self.position_embedding(positions)
        hidden = self.input_dropout(self.input_norm(hidden))
        encoded = self.encoder(hidden, mask=self.causal_mask)
        return self.output(encoded[:, -1])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def encode_training_pairs(words: list[str], context: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a word list into ``(contexts, next_symbol)`` supervision.

    Each word contributes one example per character plus one for its ending, so
    the model learns where words *stop* -- which is what makes it sensitive to
    length, and length is the strongest prior available on turn one.
    """
    contexts: list[list[int]] = []
    targets: list[int] = []
    for word in words:
        symbols = [BOUND] * context + [ord(ch) - _A for ch in word] + [BOUND]
        for i in range(context, len(symbols)):
            contexts.append(symbols[i - context : i])
            targets.append(symbols[i])
    return (
        np.asarray(contexts, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


@torch.no_grad()
def tabulate(
    model: NeuralCharLM,
    device: torch.device,
    *,
    batch_size: int = 16_384,
) -> np.ndarray:
    """Precompute log-probabilities for every possible context.

    Returns ``(27**context, 27)`` float32 -- the exact shape and meaning the
    n-gram table had, so the beam search needs no modification. Enumerating all
    contexts (rather than only observed ones) is precisely what the neural model
    buys: unseen contexts get a learned distribution instead of a backoff.
    """
    model.eval()
    ctx_len = model.config.context
    n_rows = NSYM ** ctx_len
    table = np.empty((n_rows, NSYM), dtype=np.float32)

    powers = torch.tensor(
        [NSYM ** (ctx_len - 1 - i) for i in range(ctx_len)],
        dtype=torch.long, device=device,
    )
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        ids = torch.arange(start, end, dtype=torch.long, device=device)
        # Decode each row index into its base-27 context digits.
        contexts = (ids.unsqueeze(1) // powers) % NSYM
        logits = model(contexts).float()
        table[start:end] = torch.log_softmax(logits, dim=-1).cpu().numpy()
    return table


def perplexity(
    model: NeuralCharLM, words: list[str], device: torch.device, *, batch_size: int = 8192
) -> float:
    """Per-character perplexity, comparable with :meth:`CharKN.perplexity`."""
    contexts, targets = encode_training_pairs(words, model.config.context)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            end = min(start + batch_size, len(targets))
            x = torch.from_numpy(contexts[start:end]).to(device)
            y = torch.from_numpy(targets[start:end]).to(device)
            logits = model(x).float()
            total += nn.functional.cross_entropy(
                logits, y, reduction="sum"
            ).item()
    return math.exp(total / max(len(targets), 1))
