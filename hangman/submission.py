"""Writing and validating the competition submission file.

The submission is a flat CSV with a header and one row per test word:

    word_id,guessed_letters_string
    0,etaonsrhld
    ...

``word_id`` is the zero-based line index of the word in ``test.txt``, and
``guessed_letters_string`` is the chronological sequence of guesses the solver
actually made while playing that word.

The validator here re-scores a written file with the same engine the grader
describes, so that the number reported locally is the number the leaderboard
will compute -- not a number the solver claimed for itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .game import ALPHABET, GameResult, GameState, summarise

WORD_ID_COLUMN: str = "word_id"
GUESSES_COLUMN: str = "guessed_letters_string"
EXPECTED_ROW_COUNT: int = 250_000

#: Divisor relating mean wrong guesses to the efficiency penalty Kaggle folds
#: into the displayed score. Fitted from a real submission: a replay win rate of
#: 53.5292% with 4.4155 mean wrong guesses scored 53.4766 on the public
#: leaderboard, and 4.4155 / 84 = 0.05257 closes that gap to four decimals.
#:
#: The rules describe the tie-break qualitatively but not its constant, so this
#: is an empirical estimate from a single data point. It affects only the
#: predicted display score, never the win rate that decides rank -- more words
#: solved always outranks fewer.
EFFICIENCY_PENALTY_DIVISOR: float = 84.0


def write_submission(
    results: Sequence[GameResult],
    path: str | Path,
    *,
    expected_rows: int | None = EXPECTED_ROW_COUNT,
) -> Path:
    """Write ``results`` to ``path`` in the required CSV schema.

    Args:
        results: Finished games, in test-set order.
        path: Destination file, conventionally ``submission.csv``.
        expected_rows: Row count to assert before writing. Pass ``None`` to skip
            the check (useful when writing a validation-split submission).

    Raises:
        ValueError: If the row count does not match ``expected_rows``.
    """
    if expected_rows is not None and len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} results, got {len(results)}. "
            "The submission must have exactly one row per test word."
        )

    frame = pd.DataFrame(
        {
            WORD_ID_COLUMN: range(len(results)),
            GUESSES_COLUMN: [result.guess_string for result in results],
        }
    )
    path = Path(path)
    frame.to_csv(path, index=False)
    return path


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of re-scoring a submission file against the test words."""

    rows: int
    win_rate: float
    total_wrong: int
    mean_wrong: float
    schema_errors: list[str]
    wasted_guesses: int

    @property
    def is_valid(self) -> bool:
        return not self.schema_errors

    @property
    def estimated_leaderboard_score(self) -> float:
        """Win rate less the estimated efficiency tie-break Kaggle displays."""
        return self.win_rate - self.mean_wrong / EFFICIENCY_PENALTY_DIVISOR

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else f"INVALID ({len(self.schema_errors)} errors)"
        lines = [
            f"Submission: {status}",
            f"  rows            : {self.rows:,}",
            f"  win rate        : {self.win_rate:.4f}%",
            f"  total wrong     : {self.total_wrong:,}",
            f"  mean wrong/word : {self.mean_wrong:.4f}",
            f"  wasted guesses  : {self.wasted_guesses:,} (emitted after termination)",
            f"  est. LB score   : {self.estimated_leaderboard_score:.4f}",
        ]
        lines.extend(f"  ! {error}" for error in self.schema_errors[:10])
        if len(self.schema_errors) > 10:
            lines.append(f"  ! ... and {len(self.schema_errors) - 10} more")
        return "\n".join(lines)


def validate_submission(
    path: str | Path,
    test_words: Sequence[str],
    *,
    expected_rows: int | None = EXPECTED_ROW_COUNT,
) -> ValidationReport:
    """Re-score a submission file by replaying its guesses through the engine.

    Every guess is fed to a fresh :class:`GameState` until the game terminates,
    exactly as the grader's 6-strike lockout describes. Guesses appearing after
    termination are counted as ``wasted_guesses`` -- they are ignored by the
    scorer, so they cost nothing, but a large count signals a solver that does
    not know when to stop.
    """
    frame = pd.read_csv(path, dtype={WORD_ID_COLUMN: "Int64", GUESSES_COLUMN: str})
    frame[GUESSES_COLUMN] = frame[GUESSES_COLUMN].fillna("")

    schema_errors: list[str] = []
    for column in (WORD_ID_COLUMN, GUESSES_COLUMN):
        if column not in frame.columns:
            schema_errors.append(f"missing required column {column!r}")
    if schema_errors:
        return ValidationReport(len(frame), 0.0, 0, 0.0, schema_errors, 0)

    if expected_rows is not None and len(frame) != expected_rows:
        schema_errors.append(f"expected {expected_rows} rows, found {len(frame)}")
    if len(frame) != len(test_words):
        schema_errors.append(
            f"row count {len(frame)} does not match {len(test_words)} test words"
        )
    if list(frame[WORD_ID_COLUMN]) != list(range(len(frame))):
        schema_errors.append(f"{WORD_ID_COLUMN} must be 0..{len(frame) - 1} in order")

    allowed = set(ALPHABET)
    results: list[GameResult] = []
    wasted_guesses = 0

    for row_index, (guess_string, word) in enumerate(
        zip(frame[GUESSES_COLUMN], test_words)
    ):
        invalid = sorted(set(guess_string) - allowed)
        if invalid:
            schema_errors.append(
                f"row {row_index}: non a-z characters in guesses: {invalid}"
            )

        state = GameState(word=word)
        consumed = 0
        for guess in guess_string:
            if state.is_over:
                break
            state.apply_guess(guess)
            consumed += 1
        wasted_guesses += len(guess_string) - consumed

        results.append(
            GameResult(
                word=word,
                guess_string=guess_string,
                solved=state.is_solved,
                wrong_guesses=state.wrong_guesses,
            )
        )

    metrics = summarise(results)
    return ValidationReport(
        rows=len(frame),
        win_rate=metrics["win_rate"],
        total_wrong=int(metrics["total_wrong"]),
        mean_wrong=metrics["mean_wrong"],
        schema_errors=schema_errors,
        wasted_guesses=wasted_guesses,
    )
