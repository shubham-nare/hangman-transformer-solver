"""Train the neural character LM and compare it against Kneser-Ney.

The comparison is the point. Both models are fitted on the same training split
and scored on the same held-out words, at the *same context length*, so the
difference measures generalisation rather than how much history each one sees.

    python -m scripts.train_neural_lm --steps 6000 --context 4
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hangman.char_lm import CharKN
from hangman.data import DEFAULT_DATA_DIR, TRAIN_FILENAME, load_words, split_words
from hangman.neural_lm import (
    NeuralCharLM,
    NeuralLMConfig,
    encode_training_pairs,
    perplexity,
    tabulate,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--learning-rate", type=float, default=3e-3)
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--output", type=Path, default=Path("artifacts/neural_lm"))
    ap.add_argument(
        "--skip-tabulate", action="store_true",
        help="Train and score only. At context 5 the table is 1.4 GB, so it is "
             "worth deferring until nothing else is holding memory.",
    )
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    split = split_words(
        load_words(DEFAULT_DATA_DIR / TRAIN_FILENAME),
        validation_size=20_000, seed=args.seed,
    )
    held_out = split.validation[:3000]
    print(f"train={len(split.train):,}  held-out={len(held_out):,}")

    contexts, targets = encode_training_pairs(split.train, args.context)
    print(f"training pairs: {len(targets):,}")
    x_all = torch.from_numpy(contexts).to(device)
    y_all = torch.from_numpy(targets).to(device)

    config = NeuralLMConfig(
        context=args.context, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = NeuralCharLM(config).to(device)
    print(f"parameters: {model.count_parameters():,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.01)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model.train()
    started = time.perf_counter()
    running = 0.0

    for step in range(args.steps):
        # Cosine decay; the model is small and converges quickly.
        lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = lr

        idx = torch.randint(0, len(y_all), (args.batch_size,),
                            device=device, generator=generator)
        loss = nn.functional.cross_entropy(model(x_all[idx]), y_all[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item()

        if (step + 1) % 500 == 0:
            model.eval()
            ppl = perplexity(model, held_out, device)
            model.train()
            print(
                f"step {step+1:>5}/{args.steps}  loss {running/500:.4f}  "
                f"held-out ppl {ppl:.3f}  lr {lr:.2e}  "
                f"{(step+1)/(time.perf_counter()-started):.0f} steps/s",
                flush=True,
            )
            running = 0.0

    model.eval()
    neural_ppl = perplexity(model, held_out, device)

    # The comparison that decides whether this replaces the n-gram: same data,
    # same held-out words, same amount of history.
    kn = CharKN(order=args.context + 1).fit(split.train)
    kn_ppl = kn.perplexity(held_out)

    print(f"\n{'model':<34}{'context':>9}{'held-out ppl':>14}")
    print(f"{'NeuralCharLM':<34}{args.context:>9}{neural_ppl:>14.3f}")
    print(f"{f'CharKN (order {args.context+1})':<34}{args.context:>9}{kn_ppl:>14.3f}")
    better = "NEURAL" if neural_ppl < kn_ppl else "KNESER-NEY"
    print(f"-> {better} generalises better on unseen words "
          f"({abs(neural_ppl-kn_ppl)/kn_ppl*100:.1f}% difference)")

    torch.save(
        {"model_state": model.state_dict(), "config": vars(config),
         "held_out_ppl": neural_ppl, "kn_ppl": kn_ppl},
        args.output / "neural_lm.pt",
    )
    print(f"\nsaved {args.output / 'neural_lm.pt'}")

    if args.skip_tabulate:
        print("\nskipping tabulation (--skip-tabulate)")
        return
    print("tabulating all contexts...")
    t0 = time.perf_counter()
    table = tabulate(model, device)
    np.save(args.output / "table.npy", table)
    print(f"  {table.shape}  {table.nbytes/1e6:.0f} MB  in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
