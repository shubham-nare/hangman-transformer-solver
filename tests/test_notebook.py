"""Verify the generated Kaggle notebook is self-contained and executable.

The notebook is assembled by stripping intra-package imports and relying on a
shared namespace. If that stripping ever removes too much -- or the module order
stops matching the dependency order -- the notebook would fail on Kaggle with a
``NameError`` that nothing else in the suite would catch. These tests execute the
inlined modules exactly as the notebook does.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.build_notebook import SECTIONS, NotebookConfig, build

MODULE_MARKER = "from __future__ import annotations"

#: Package modules deliberately kept out of the notebook. Each is a measured,
#: rejected experiment that no inlined module imports, so leaving it out cannot
#: produce the missing-name failure the guard below exists to catch.
#:   neural_lm.py -- neural character LM; lost to Kneser-Ney by 1.78 points.
NOT_SHIPPED = {"neural_lm.py"}


@pytest.fixture(scope="module")
def notebook(tmp_path_factory) -> dict:
    path = tmp_path_factory.mktemp("notebook") / "submission_notebook.ipynb"
    build(path, NotebookConfig())
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _module_cells(notebook: dict) -> list[str]:
    return [source for source in _code_cells(notebook) if MODULE_MARKER in source]


def test_every_module_is_inlined(notebook: dict) -> None:
    assert len(_module_cells(notebook)) == len(SECTIONS)


def test_no_package_module_is_left_out_of_the_notebook() -> None:
    """Every module in the package must appear in SECTIONS or NOT_SHIPPED.

    Regression guard: `selfplay.py` was added to the package but not to
    SECTIONS, so the notebook inlined a `train.py` that called
    `MixedStateSampler` -- a name nothing had defined. Stripping the import
    meant the module still *compiled*, so the failure only surfaced at call
    time, on Kaggle, minutes into a run.
    """
    listed = {filename for filename, _, _ in SECTIONS}
    on_disk = {
        path.name
        for path in Path("hangman").glob("*.py")
        if path.name != "__init__.py"
    }
    assert not listed & NOT_SHIPPED, "a module is both shipped and excluded"
    assert on_disk == listed | NOT_SHIPPED, (
        f"modules missing from the notebook: {on_disk - listed - NOT_SHIPPED}"
    )


def test_notebook_can_actually_train(notebook: dict) -> None:
    """Run the inlined training loop for a few steps.

    Compiling the cells is not enough -- a missing name inside a function body
    only fails when that function runs. This exercises the real entry point the
    notebook calls, which is the check that would have caught the
    `MixedStateSampler` break before it cost a Kaggle run.
    """
    namespace: dict = {}
    for source in _module_cells(notebook):
        exec(compile(source, "<notebook>", "exec"), namespace)

    words = ["banana", "hangman", "transformer", "quixotic", "aardvark", "zebra"]
    _, summary = namespace["train"](
        words,
        words,
        model_config=namespace["ModelConfig"](
            d_model=32, n_heads=4, n_layers=2, dim_feedforward=64
        ),
        training_config=namespace["TrainingConfig"](
            steps=4,
            batch_size=8,
            warmup_steps=1,
            eval_every=4,
            eval_words=3,
            amp=False,
            # Exercise the self-play path too, since that is what broke.
            self_play_start_step=2,
            self_play_refresh_every=2,
            self_play_words=4,
        ),
        output_dir=Path(tempfile.mkdtemp()),
        device="cpu",
    )
    assert "best_win_rate" in summary


def test_no_intra_package_imports_survive(notebook: dict) -> None:
    """A leftover `from .game import ...` would be an ImportError on Kaggle."""
    for source in _code_cells(notebook):
        assert "from ." not in source, f"relative import survived:\n{source[:400]}"


def test_inlined_modules_execute_in_a_shared_namespace(notebook: dict) -> None:
    """This is the check that the notebook will actually run."""
    namespace: dict = {}
    for source in _module_cells(notebook):
        exec(compile(source, "<notebook>", "exec"), namespace)

    for name in (
        "GameState",
        "Observation",
        "play_games",
        "HangmanTransformer",
        "GameStateSampler",
        "NeuralPolicy",
        "train",
        "write_submission",
        "validate_submission",
        "load_words",
    ):
        assert name in namespace, f"{name} missing from the notebook namespace"


def test_notebook_can_play_a_game_end_to_end(notebook: dict) -> None:
    """Exercise the inlined code on a real game, not just on imports."""
    namespace: dict = {}
    for source in _module_cells(notebook):
        exec(compile(source, "<notebook>", "exec"), namespace)

    model = namespace["HangmanTransformer"](
        namespace["ModelConfig"](d_model=32, n_heads=4, n_layers=2, dim_feedforward=64)
    )
    policy = namespace["NeuralPolicy"](model)
    results = namespace["play_games"](["banana", "hangman"], policy)

    assert len(results) == 2
    metrics = namespace["summarise"](results)
    assert 0.0 <= metrics["win_rate"] <= 100.0


def test_notebook_declares_no_external_data_or_network_calls(notebook: dict) -> None:
    """The rules forbid external model APIs; nothing here should reach the network."""
    forbidden = ("requests.", "urllib.request", "http://", "https://", "openai", "anthropic")
    for source in _code_cells(notebook):
        lowered = source.lower()
        for token in forbidden:
            assert token not in lowered, f"notebook references {token!r}"
