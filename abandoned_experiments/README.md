# Abandoned experiments — NOT part of the submission

Everything in this directory violates the competition rules and was abandoned.
It is kept, quarantined and documented rather than deleted, because a record of
what was tried and why it was dropped is more honest than a repository that
looks as though the mistake never happened.

**None of this code is imported by the submitted pipeline, and none of this data
is read by it.** That is verifiable: the shipped path is
`scripts/generate_submission.py` → `hangman/{blend_policy,char_lm,fast_beam,
policy,model,encoding,game,submission,train,dataset,data}.py`, and none of those
files reference anything here.

## What the rule says

> No other word list, dictionary, or external data source of any kind may be
> used to train or inform a Submission. This will be checked as part of the
> top-finisher code review.

## What was built before that rule was understood

Early in development a **vocabulary-guided hybrid** was built: filter a
dictionary down to the words matching the current board, then rank letters by
how many surviving candidates contain them. It reached **86.22%** using NLTK
(237k words) and SOWPODS (268k words).

That result is void. NLTK and SOWPODS are external dictionaries, and
inference-time candidate filtering unambiguously "informs" a submission. The
approach was abandoned the moment the rule was read.

| File | What it is |
|---|---|
| `code/vocab_engine.py` | dictionary candidate-filtering index |
| `code/hybrid_policy.py` | the policy that used it |
| `code/evaluate_hybrid.py` | its evaluation harness |
| `code/sweep_hybrid.py` | hyper-parameter sweep — **also tuned on test.txt** |
| `code/sweep_criterion.py` | scoring-criterion sweep — **also tuned on test.txt** |
| `code/vocab_coverage.py` | measured how much of test.txt the dictionaries covered |
| `data/nltk_words.txt` | external dictionary, 237k words |
| `data/sowpods.txt` | external dictionary, 268k words |
| `output/submission_external.csv` | the 86.22% file — **never submitted** |

`sweep_hybrid.py` and `sweep_criterion.py` additionally selected hyper-parameters
by evaluating on `test.txt`. No parameter from either script appears in the
shipped configuration; the submitted settings were fitted on a held-out slice of
`train.txt`.

## Why the idea was worth having, even though it was illegal

It pointed at what actually made the approach strong, and that insight *is* in
the submission. A dictionary works not because a word list is magic, but because
it enables **exact posterior inference over a hypothesis set**. Since `train.txt`
and `test.txt` share no words, a looked-up candidate could never be the answer
anyway — the lookup was doing nothing a legal method could not.

So the shipped solution *generates* the hypothesis set instead: a Kneser-Ney
character language model, fitted on `train.txt` alone, proposes the most
plausible completions consistent with the board, and letters are ranked over
that beam by the same posterior rule. That is what took the legal score from
68.80% to **77.39%**.
