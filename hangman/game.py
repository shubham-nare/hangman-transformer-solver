"""Hangman game engine implementing exact competition semantics.

Rules (from the competition specification):

* A secret word is hidden as underscores. Non-letter characters (spaces, digits,
  punctuation) are revealed from the start -- only ``a-z`` is ever guessed.
* A guess that reveals at least one new letter is a hit; every other guess is
  exactly one strike. That includes wrong letters, repeated/duplicate guesses,
  spaces, digits and invalid symbols.
* A game ends the instant the word is fully revealed (win) or the strike count
  reaches ``MAX_WRONG_GUESSES`` (loss). Any guesses after that point are ignored.

The engine is deliberately policy-agnostic: it knows how to run a game but not
how to choose a letter. Policies implement :class:`GuessPolicy`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

ALPHABET: str = "abcdefghijklmnopqrstuvwxyz"
LETTER_TO_INDEX: dict[str, int] = {letter: i for i, letter in enumerate(ALPHABET)}
MAX_WRONG_GUESSES: int = 6
MASK_CHAR: str = "_"


def is_guessable(char: str) -> bool:
    """Return whether ``char`` is a letter the player is allowed to guess."""
    return char in LETTER_TO_INDEX


@dataclass(frozen=True)
class Observation:
    """Everything a policy is allowed to know about a game in progress.

    This type deliberately excludes the secret word. The public ``test.txt``
    contains the answers, so a policy that could reach the word would be able to
    cheat -- accidentally or otherwise -- and the competition treats lookup
    leaks as disqualifying. Making the observation the only channel between the
    engine and a policy turns that from a discipline problem into a structural
    guarantee.

    Attributes:
        board: Per-character view, with ``MASK_CHAR`` for hidden letters.
        guesses: Every character submitted so far, in chronological order.
        wrong_guesses: Strikes accumulated so far.
    """

    board: str
    guesses: tuple[str, ...]
    wrong_guesses: int

    @property
    def length(self) -> int:
        return len(self.board)

    @property
    def guessed_letters(self) -> frozenset[str]:
        return frozenset(self.guesses)

    @property
    def remaining_lives(self) -> int:
        return MAX_WRONG_GUESSES - self.wrong_guesses


@dataclass
class GameState:
    """Mutable state of a single Hangman game.

    Attributes:
        word: The secret word, lowercased. Never exposed to policies.
        board: Per-character view. Hidden letters are ``MASK_CHAR``; revealed
            letters and non-letter characters show their true character.
        guesses: Every character submitted, in chronological order.
        wrong_guesses: Number of strikes accumulated so far.
    """

    word: str
    board: list[str] = field(init=False)
    guesses: list[str] = field(default_factory=list)
    wrong_guesses: int = 0

    def __post_init__(self) -> None:
        # Non-letters are visible from the start; letters begin masked.
        self.board = [MASK_CHAR if is_guessable(c) else c for c in self.word]

    @property
    def masked_word(self) -> str:
        """The board as a string, e.g. ``"_p p_ e"``."""
        return "".join(self.board)

    @property
    def guessed_letters(self) -> set[str]:
        """The set of characters already submitted."""
        return set(self.guesses)

    @property
    def is_solved(self) -> bool:
        return MASK_CHAR not in self.board

    @property
    def is_out_of_lives(self) -> bool:
        return self.wrong_guesses >= MAX_WRONG_GUESSES

    @property
    def is_over(self) -> bool:
        return self.is_solved or self.is_out_of_lives

    @property
    def guess_string(self) -> str:
        """The chronological guess sequence, as required by the submission CSV."""
        return "".join(self.guesses)

    @property
    def observation(self) -> Observation:
        """The word-free view handed to policies."""
        return Observation(
            board=self.masked_word,
            guesses=tuple(self.guesses),
            wrong_guesses=self.wrong_guesses,
        )

    def apply_guess(self, guess: str) -> bool:
        """Apply one guess and return whether it revealed a new letter.

        A guess is a hit only if it uncovers at least one previously hidden
        position. Everything else -- a letter not in the word, a repeat of an
        earlier guess, or a non-letter character -- costs exactly one strike.

        Raises:
            RuntimeError: If the game has already terminated.
            ValueError: If ``guess`` is not a single character.
        """
        if self.is_over:
            raise RuntimeError("Cannot guess: this game has already terminated.")
        if len(guess) != 1:
            raise ValueError(f"A guess must be exactly one character, got {guess!r}.")

        revealed_any = False
        if is_guessable(guess) and guess not in self.guessed_letters:
            for i, char in enumerate(self.word):
                if char == guess and self.board[i] == MASK_CHAR:
                    self.board[i] = char
                    revealed_any = True

        self.guesses.append(guess)
        if not revealed_any:
            self.wrong_guesses += 1
        return revealed_any


class GuessPolicy(Protocol):
    """Chooses the next letter for each of a batch of in-progress games.

    Policies receive :class:`Observation` values -- never the secret word -- and
    receive them in a batch so that model-based implementations can run a single
    vectorised forward pass per turn instead of one per game.
    """

    def next_guesses(self, observations: Sequence[Observation]) -> Sequence[str]:
        """Return one guess character per observation, in the same order."""
        ...


@dataclass(frozen=True)
class GameResult:
    """Outcome of a finished game."""

    word: str
    guess_string: str
    solved: bool
    wrong_guesses: int


def play_games(
    words: Iterable[str],
    policy: GuessPolicy,
    *,
    progress: bool = False,
) -> list[GameResult]:
    """Play one game per word, stepping every active game in lockstep.

    All games that are still running are advanced together so that ``policy``
    sees the largest possible batch on each turn. Results are returned in the
    same order as ``words``.

    Args:
        progress: Print a line per turn with the number of games still running.
            A full 250,000-word evaluation takes tens of minutes and is
            otherwise silent until it returns, which makes a slow run
            indistinguishable from a stuck one. Because games advance in
            lockstep the turn count is bounded (a game ends by the sixth strike
            or when solved), so the shrinking active count is a real progress
            signal rather than a guess.
    """
    states = [GameState(word=word) for word in words]
    total = len(states)
    active = [state for state in states if not state.is_over]

    turn = 0
    started = time.perf_counter()
    while active:
        guesses = policy.next_guesses([state.observation for state in active])
        if len(guesses) != len(active):
            raise ValueError(
                f"Policy returned {len(guesses)} guesses for {len(active)} active games."
            )
        for state, guess in zip(active, guesses):
            state.apply_guess(guess)
        active = [state for state in active if not state.is_over]
        turn += 1
        if progress:
            done = total - len(active)
            elapsed = time.perf_counter() - started
            print(
                f"  turn {turn:>2}: {len(active):>7,} active, "
                f"{done:>7,}/{total:,} finished ({100.0*done/max(total,1):5.1f}%)  "
                f"{elapsed/60:.1f} min",
                flush=True,
            )

    return [
        GameResult(
            word=state.word,
            guess_string=state.guess_string,
            solved=state.is_solved,
            wrong_guesses=state.wrong_guesses,
        )
        for state in states
    ]


def summarise(results: Sequence[GameResult]) -> dict[str, float]:
    """Compute the competition metrics for a set of finished games.

    Returns the primary win rate (percentage of words solved) and the
    tie-break inputs: total and mean wrong guesses.
    """
    if not results:
        return {"games": 0, "win_rate": 0.0, "total_wrong": 0, "mean_wrong": 0.0}

    solved = sum(result.solved for result in results)
    total_wrong = sum(result.wrong_guesses for result in results)
    return {
        "games": len(results),
        "win_rate": 100.0 * solved / len(results),
        "total_wrong": total_wrong,
        "mean_wrong": total_wrong / len(results),
    }
