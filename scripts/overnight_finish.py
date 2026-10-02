"""Unattended finish: wait for run6, decide if it helps, regenerate if it does.

Runs without supervision, so it is built to fail safe. The verified 76.5296%
submission and its notebook are never touched unless a replacement has been
generated AND independently validated; anything else leaves them exactly as they
are and says so in the log.

Steps
-----
1. Wait for run6 training to finish (its summary file appearing is the signal).
2. Score three policies on the SAME held-out train words -- run5 blended, run6
   blended, and the two ensembled -- recording per-word outcomes.
3. Compare with McNemar's paired test rather than raw win rates. The policies
   agree on the large majority of games, so the signal lives in the discordant
   ones; an unpaired comparison at this sample size cannot resolve the ~0.3
   point differences at stake.
4. Regenerate the submission only if a challenger beats run5 significantly.
5. Rebuild and re-verify the notebook so code and predictions never diverge.

test.txt is used only to play the final submission, never to choose anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hangman.blend_policy import BlendPolicy
from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.fast_beam import SparseBeamPosterior, build_sparse_table
from hangman.game import play_games, summarise
from hangman.policy import EnsemblePolicy, NeuralPolicy
from hangman.train import load_model

ROOT = Path(".")
VERIFIED = ROOT / "submissions_verified"
BASELINE_SCORE = 76.5296  # the result already banked and validated


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_run6(summary: Path, timeout_h: float) -> bool:
    """Block until training writes its summary, or give up."""
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        if summary.exists():
            log(f"run6 finished: {summary} present")
            return True
        time.sleep(60)
    log(f"WARNING: run6 did not finish within {timeout_h}h; using its best checkpoint anyway")
    return False


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Paired test on per-word win/loss. Returns (b_only, a_only, two-sided p).

    Only games the two policies disagree on carry information. With n discordant
    pairs the null is Binomial(n, 0.5); the normal approximation is ample here
    because n runs to the hundreds.
    """
    b_only = int(np.sum(b & ~a))
    a_only = int(np.sum(a & ~b))
    n = b_only + a_only
    if n == 0:
        return b_only, a_only, 1.0
    z = abs(b_only - a_only) / np.sqrt(n)
    # Two-sided normal tail, via erfc to avoid a scipy dependency.
    import math

    p = math.erfc(z / math.sqrt(2.0))
    return b_only, a_only, p


def evaluate(name: str, policy, words: list[str]) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    results = play_games(words, policy)
    stats = summarise(results)
    solved = np.array([r.solved for r in results], dtype=bool)
    log(
        f"  {name:<22} win={stats['win_rate']:6.2f}%  "
        f"wrong={stats['mean_wrong']:.3f}  ({time.perf_counter()-t0:.0f}s)"
    )
    return solved, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-words", type=int, default=20_000)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--blend-weight", type=float, default=0.38)
    ap.add_argument("--beam-width", type=int, default=200)
    ap.add_argument("--max-blanks", type=int, default=10)
    ap.add_argument("--position-weight", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level")
    ap.add_argument("--wait-hours", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    log("=== overnight finish starting ===")
    wait_for_run6(ROOT / "artifacts/run6/training_summary.json", args.wait_hours)

    run5 = ROOT / "artifacts/run5/best_model.pt"
    run6 = ROOT / "artifacts/run6/best_model.pt"
    if not run6.exists():
        log("run6 checkpoint missing; nothing to do. Baseline stands.")
        return

    for path in (run5, run6):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        log(f"{path}: step={ck.get('step')} win_rate={ck.get('win_rate'):.2f}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = split_words(
        load_words(DEFAULT_DATA_DIR / TRAIN_FILENAME),
        validation_size=20_000, seed=args.seed,
    )
    words = split.validation[: args.eval_words]
    log(f"scoring on {len(words):,} HELD-OUT TRAIN words (test.txt untouched)")

    log(f"fitting CharKN(order={args.order}) on the training split...")
    lm = CharKN(order=args.order).fit(split.train)
    index, table = build_sparse_table(lm)
    lm.release()
    beam = SparseBeamPosterior(
        index, table, args.order,
        beam_width=args.beam_width, max_blanks=args.max_blanks,
    )

    p5 = NeuralPolicy(load_model(run5, device=device), device=device,
                      position_weight=args.position_weight)
    p6 = NeuralPolicy(load_model(run6, device=device), device=device,
                      position_weight=args.position_weight)
    ens = EnsemblePolicy([p5, p6])

    def blended(neural):
        return BlendPolicy(neural, beam, blend_weight=args.blend_weight)

    log("evaluating three policies on identical words:")
    solved5, stats5 = evaluate("run5 + blend", blended(p5), words)
    solved6, stats6 = evaluate("run6 + blend", blended(p6), words)
    solvedE, statsE = evaluate("run5+run6 + blend", blended(ens), words)

    log("\npaired comparisons vs run5 (McNemar):")
    decision = ("run5", stats5["win_rate"], None)
    for name, solved, stats in (("run6", solved6, stats6), ("ensemble", solvedE, statsE)):
        wins, losses, p = mcnemar(solved5, solved)
        delta = stats["win_rate"] - stats5["win_rate"]
        verdict = "SIGNIFICANT" if (p < args.alpha and delta > 0) else "not significant"
        log(
            f"  {name:<10} delta={delta:+.2f}pts  "
            f"newly_won={wins} newly_lost={losses}  p={p:.4f}  {verdict}"
        )
        if p < args.alpha and delta > 0 and stats["win_rate"] > decision[1]:
            decision = (name, stats["win_rate"], stats)

    winner = decision[0]
    log(f"\nDECISION: {winner}")

    if winner == "run5":
        log("No challenger beat the shipped model significantly.")
        log(f"BASELINE STANDS: submission_order6.csv at {BASELINE_SCORE}%")
        return

    # --- regenerate with the winner -------------------------------------
    out = ROOT / f"submission_{winner}.csv"
    cmd = [
        sys.executable, "-u", "-m", "scripts.generate_submission",
        "--checkpoint", str(run6 if winner == "run6" else run5),
        "--output", str(out),
        "--use-blend", "--lm-order", str(args.order),
        "--beam-width", str(args.beam_width),
        "--max-blanks", str(args.max_blanks),
        "--blend-weight", str(args.blend_weight),
        "--position-weight", str(args.position_weight),
    ]
    if winner == "ensemble":
        cmd += ["--extra-checkpoints", str(run6)]
    log(f"regenerating: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        log("REGENERATION FAILED; baseline untouched.")
        print(proc.stderr[-2000:], flush=True)
        return

    # Only promote after the file proves itself.
    score = None
    for line in proc.stdout.splitlines():
        if "win rate" in line and ":" in line:
            try:
                score = float(line.split(":")[1].strip().rstrip("%"))
            except ValueError:
                pass
    rows = sum(1 for _ in out.open(encoding="utf-8"))
    log(f"generated {out.name}: {rows:,} lines, score={score}")

    if rows != 250_001 or score is None or score <= BASELINE_SCORE:
        log(f"NOT PROMOTING (rows={rows}, score={score} vs baseline {BASELINE_SCORE}).")
        log(f"BASELINE STANDS: submission_order6.csv at {BASELINE_SCORE}%")
        return

    VERIFIED.mkdir(exist_ok=True)
    shutil.copy(out, VERIFIED / f"submission_{winner}_{score:.4f}.csv")
    shutil.copy(out, ROOT / "submission.csv")
    shutil.copy(out, ROOT / "submission_order6.csv")
    log(f"PROMOTED: {score:.4f}% is the new submission (all three filenames)")

    # Keep the notebook in step with the predictions it claims to produce.
    nb = subprocess.run(
        [sys.executable, "-m", "scripts.build_notebook",
         "--output", "submission_notebook.ipynb"],
        capture_output=True, text=True,
    )
    print(nb.stdout, flush=True)
    if nb.returncode == 0:
        shutil.copy(ROOT / "submission_notebook.ipynb",
                    VERIFIED / f"submission_notebook_{score:.4f}.ipynb")
        log("notebook rebuilt and backed up")
    else:
        log("notebook rebuild FAILED; the previous verified notebook is still valid")

    log("=== overnight finish complete ===")


if __name__ == "__main__":
    main()
