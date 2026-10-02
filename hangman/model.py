"""Transformer encoder that predicts which letter is hidden behind the board.

The task is masked language modelling over characters, specialised to Hangman.
Given a partially revealed spelling the model scores all 26 letters by how
likely they are to occupy the remaining blanks, and the solver guesses the
highest-scoring letter it has not already tried.

Two design points do the heavy lifting:

* **Guessed-letter conditioning.** The 26-dim guessed vector is projected into
  the model width and added at every position, so self-attention can reason
  about eliminated letters at each character rather than only at the output. A
  board of ``_ a _ _`` means something very different after five failed guesses
  than it does on turn one.
* **Masked pooling.** Mean and max pooling are concatenated over real positions
  only. Mean captures the overall shape of the word, max captures the single
  most diagnostic position -- a distinctive suffix, say -- which mean pooling
  would otherwise dilute on long words.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encoding import MASK_TOKEN_ID, MAX_WORD_LENGTH, VOCABULARY_SIZE
from .game import ALPHABET


def _large_negative(tensor: Tensor) -> float:
    """The most negative finite value representable in ``tensor``'s dtype.

    A hard-coded sentinel like ``-1e9`` overflows under mixed precision, where
    activations are float16 and the largest finite magnitude is 65504.
    """
    return torch.finfo(tensor.dtype).min


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    Defaults are sized for a 4 GB laptop GPU while staying large enough to model
    English orthography: roughly 3.2M parameters.
    """

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 768
    dropout: float = 0.1
    max_length: int = MAX_WORD_LENGTH
    n_letters: int = len(ALPHABET)
    use_position_head: bool = True


class HangmanTransformer(nn.Module):
    """Encoder-only Transformer scoring each letter's presence in the blanks."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        d_model = self.config.d_model

        self.token_embedding = nn.Embedding(VOCABULARY_SIZE, d_model)
        self.position_embedding = nn.Embedding(self.config.max_length, d_model)
        self.guessed_projection = nn.Linear(self.config.n_letters, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(self.config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
            norm=nn.LayerNorm(d_model),
            # Nested tensors are unsupported alongside norm_first=True; naming it
            # explicitly keeps PyTorch from warning on every construction.
            enable_nested_tensor=False,
        )

        # Pooled sequence (mean + max) alongside the raw guessed vector.
        head_input = 2 * d_model + self.config.n_letters
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model, self.config.n_letters),
        )

        # Per-position masked-language-model head: what letter sits in this blank?
        self.position_head = (
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.config.n_letters),
            )
            if self.config.use_position_head
            else None
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        guessed: Tensor,
        padding: Tensor,
        *,
        return_pooled: bool = False,
    ) -> dict[str, Tensor]:
        """Score every letter for a batch of board states.

        Args:
            tokens: ``(batch, length)`` board token ids.
            guessed: ``(batch, 26)`` binary already-guessed mask.
            padding: ``(batch, length)`` boolean, ``True`` at padding positions.
            return_pooled: When ``True``, include the intermediate pooled
                representation ``(batch, 2*d_model + 26)`` under key
                ``"pooled"``. Used by the RL value head, which needs the same
                features the presence head sees. Callers that do not need it pay
                no extra cost when the flag is ``False`` (the default).

        Returns:
            Dict with ``presence`` ``(batch, 26)`` word-level logits and, when the
            position head is enabled, ``position`` ``(batch, length, 26)``
            per-blank logits. Both are raw, before already-guessed letters are
            masked out. Also contains ``"pooled"`` when ``return_pooled=True``.
        """
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)

        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        # Broadcast the guessed-letter context to every character position.
        hidden = hidden + self.guessed_projection(guessed).unsqueeze(1)
        hidden = self.input_dropout(self.input_norm(hidden))

        encoded = self.encoder(hidden, src_key_padding_mask=padding)

        real = (~padding).unsqueeze(-1).to(encoded.dtype)
        summed = (encoded * real).sum(dim=1)
        counts = real.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / counts
        max_pooled = (
            encoded.masked_fill(padding.unsqueeze(-1), _large_negative(encoded))
            .max(dim=1)
            .values
        )

        pooled = torch.cat([mean_pooled, max_pooled, guessed], dim=-1)
        outputs = {"presence": self.head(pooled)}
        if self.position_head is not None:
            outputs["position"] = self.position_head(encoded)
        if return_pooled:
            outputs["pooled"] = pooled
        return outputs

    @staticmethod
    def mask_guessed(logits: Tensor, guessed: Tensor) -> Tensor:
        """Drive already-guessed letters to negligible probability.

        Re-guessing a letter is scored as a strike, so a guessed letter must never
        win the argmax. Applying this at training time too keeps the loss aligned
        with how the model is actually used.
        """
        return logits.masked_fill(guessed > 0, _large_negative(logits))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def presence_score_from_positions(
    position_logits: Tensor, tokens: Tensor, guessed: Tensor
) -> Tensor:
    """Turn per-blank letter distributions into a word-level presence score.

    Hangman rewards guessing a letter that appears *anywhere* in the word, so the
    per-position distributions must be combined across blanks. Treating the
    blanks as independent,

        P(letter absent everywhere) = prod_i (1 - p_i(letter))

    and the returned score is the negative log of that product. It is monotonic
    in P(letter present), which is all a ranking needs, and summing logs avoids
    the underflow that multiplying many small probabilities would cause.

    Independence is an approximation -- the blanks in a real word are strongly
    correlated -- but it is a well-behaved one, and the pooled presence head is
    trained jointly to capture what it misses.
    """
    blanks = tokens == MASK_TOKEN_ID
    masked = position_logits.masked_fill(
        guessed.unsqueeze(1) > 0, _large_negative(position_logits)
    )
    probabilities = torch.softmax(masked.float(), dim=-1)

    log_absent = torch.log1p(-probabilities.clamp(max=1.0 - 1e-6))
    log_absent = log_absent * blanks.unsqueeze(-1).to(log_absent.dtype)
    return -log_absent.sum(dim=1)
