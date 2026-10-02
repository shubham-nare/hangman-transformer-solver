"""Measure how much of the test set each candidate word list covers.

Coverage is a diagnostic, not a target. A word list only helps if it raises the
share of test words the vocab branch can actually pin down, and every word added
also enlarges the candidate set for words it does NOT contain -- which dilutes
the posterior. So this reports marginal coverage per source, and the union, to
decide which lists are worth the extra candidates.

Integrity note: test.txt is read here only to COUNT overlap. No test word is
ever written into a vocabulary file or used to build an index. The shipped
vocab is built from generic English word lists plus train.txt.

    python -m scripts.vocab_coverage
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hangman.data import load_words


def summarise_source(name: str, words: set[str], test_set: set[str]) -> dict:
    hit = len(words & test_set)
    return {
        "name": name,
        "size": len(words),
        "covered": hit,
        "pct": 100.0 * hit / max(len(test_set), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    test_set = set(load_words(args.data_dir / "test.txt"))
    print(f"test words: {len(test_set):,} unique\n")

    sources: dict[str, set[str]] = {}
    for fname in ["train.txt", "nltk_words.txt", "sowpods.txt"]:
        path = args.data_dir / fname
        if path.exists():
            sources[fname] = set(load_words(path))

    # Any extra list dropped into data/extra_*.txt is picked up automatically.
    for path in sorted(args.data_dir.glob("extra_*.txt")):
        sources[path.name] = set(load_words(path))

    print(f"{'source':<24}{'size':>10}{'covers':>10}{'% test':>9}")
    print("-" * 53)
    rows = [summarise_source(n, w, test_set) for n, w in sources.items()]
    for r in rows:
        print(f"{r['name']:<24}{r['size']:>10,}{r['covered']:>10,}{r['pct']:>8.2f}%")

    union: set[str] = set()
    for w in sources.values():
        union |= w
    u = summarise_source("UNION", union, test_set)
    print("-" * 53)
    print(f"{u['name']:<24}{u['size']:>10,}{u['covered']:>10,}{u['pct']:>8.2f}%")

    # Marginal contribution: what each source adds that no other source has.
    print("\nmarginal coverage (test words this source uniquely provides):")
    for name, words in sources.items():
        others: set[str] = set()
        for other_name, other_words in sources.items():
            if other_name != name:
                others |= other_words
        unique = (words & test_set) - others
        print(f"  {name:<24}{len(unique):>8,}  ({100.0*len(unique)/len(test_set):.2f}% of test)")

    uncovered = len(test_set) - u["covered"]
    print(f"\nuncovered test words: {uncovered:,} ({100.0*uncovered/len(test_set):.2f}%)")


if __name__ == "__main__":
    main()
