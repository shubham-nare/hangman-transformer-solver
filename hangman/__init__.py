"""Hangman solver for the Meltwater Brand & Buzzword hackathon."""

from .game import (
    ALPHABET,
    MASK_CHAR,
    MAX_WRONG_GUESSES,
    GameResult,
    GameState,
    GuessPolicy,
    Observation,
    play_games,
    summarise,
)
from .submission import validate_submission, write_submission

__all__ = [
    "ALPHABET",
    "MASK_CHAR",
    "MAX_WRONG_GUESSES",
    "GameResult",
    "GameState",
    "GuessPolicy",
    "Observation",
    "play_games",
    "summarise",
    "validate_submission",
    "write_submission",
]
