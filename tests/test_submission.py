"""Tests for submission writing and independent re-scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from hangman.game import GameResult, play_games
from hangman.submission import (
    GUESSES_COLUMN,
    WORD_ID_COLUMN,
    ValidationReport,
    validate_submission,
    write_submission,
)
from tests.test_game import ScriptedPolicy


def test_write_submission_uses_the_required_schema(tmp_path) -> None:
    results = play_games(["banana", "zzzz"], ScriptedPolicy("anbxqwjy"))
    path = write_submission(results, tmp_path / "submission.csv", expected_rows=None)

    frame = pd.read_csv(path)
    assert list(frame.columns) == [WORD_ID_COLUMN, GUESSES_COLUMN]
    assert list(frame[WORD_ID_COLUMN]) == [0, 1]
    assert frame[GUESSES_COLUMN][0] == "anb"


def test_validation_reproduces_the_engine_score(tmp_path) -> None:
    words = ["banana", "zzzz"]
    results = play_games(words, ScriptedPolicy("anbxqwjy"))
    path = write_submission(results, tmp_path / "submission.csv", expected_rows=None)

    report = validate_submission(path, words, expected_rows=None)
    assert report.is_valid
    assert report.win_rate == 50.0
    assert report.total_wrong == 6
    assert report.wasted_guesses == 0


def test_validation_flags_non_letter_characters(tmp_path) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame({WORD_ID_COLUMN: [0], GUESSES_COLUMN: ["ab c"]}).to_csv(
        path, index=False
    )

    report = validate_submission(path, ["banana"], expected_rows=None)
    assert not report.is_valid
    assert any("non a-z" in error for error in report.schema_errors)


def test_validation_counts_guesses_emitted_after_termination(tmp_path) -> None:
    """The grader locks out trailing guesses; they must not inflate the score."""
    path = tmp_path / "submission.csv"
    # "anb" already solves the word; "xyz" is dead weight the grader ignores.
    pd.DataFrame({WORD_ID_COLUMN: [0], GUESSES_COLUMN: ["anbxyz"]}).to_csv(
        path, index=False
    )

    report = validate_submission(path, ["banana"], expected_rows=None)
    assert report.win_rate == 100.0
    assert report.total_wrong == 0
    assert report.wasted_guesses == 3


def test_validation_flags_wrong_row_count_and_misordered_ids(tmp_path) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame({WORD_ID_COLUMN: [1, 0], GUESSES_COLUMN: ["a", "b"]}).to_csv(
        path, index=False
    )

    report = validate_submission(path, ["banana"], expected_rows=None)
    assert not report.is_valid
    assert any("does not match" in error for error in report.schema_errors)
    assert any("in order" in error for error in report.schema_errors)


def test_estimated_leaderboard_score_reproduces_a_real_submission() -> None:
    """Pin the fitted penalty against the one submission we have ground truth for.

    A replay win rate of 53.5292% with 4.4155 mean wrong guesses scored 53.4766
    on the public leaderboard. If this drifts, the estimate needs refitting.
    """
    report = ValidationReport(
        rows=250_000,
        win_rate=53.5292,
        total_wrong=1_103_865,
        mean_wrong=4.4155,
        schema_errors=[],
        wasted_guesses=0,
    )
    assert report.estimated_leaderboard_score == pytest.approx(53.4766, abs=5e-4)


def test_empty_guess_string_round_trips(tmp_path) -> None:
    """A word with no guessable letters is an instant win with no guesses."""
    words = ["#1 !"]
    results = [GameResult(word="#1 !", guess_string="", solved=True, wrong_guesses=0)]
    path = write_submission(results, tmp_path / "submission.csv", expected_rows=None)

    report = validate_submission(path, words, expected_rows=None)
    assert report.is_valid
    assert report.win_rate == 100.0
