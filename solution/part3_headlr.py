"""Diagnostic for T2: does the structured stream fail to help, or fail to train?

T2 fuses 28 structured features into the head but scores the same as text-only
T1. One explanation is that they carry nothing the encoder lacks; another is
that a randomly-initialised head cannot learn to use them at the encoder's
3e-5. This gives the head its own learning rate and re-runs Part 2's folds.

Run:  uv run python solution/part3_headlr.py           (~13 min per rate, CPU)
      uv run python solution/part3_headlr.py holdout   (~4 min, CPU)
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from common import FIGDIR, fmt, folds, holdout_split, load, score_all
from part3_transformer import fit_predict

full = load().reset_index(drop=True)
tr_i, _ = holdout_split(full)
train = full.iloc[tr_i].reset_index(drop=True)
y = train["human_score"].to_numpy()


def sweep() -> None:
    """Re-run Part 2's folds with the head on its own learning rate."""
    for hl in [1e-3, 1e-2]:
        t0 = time.time()
        oof = np.zeros(len(train))
        for k, (a, b) in enumerate(folds(train, grouped=True)):
            oof[b], _, _ = fit_predict(train.iloc[a], y[a], train.iloc[b], True, head_lr=hl)
            print(f"  head_lr={hl}  fold {k + 1}/5  [{time.time() - t0:.0f}s]", flush=True)
        print(f"RESULT T2 head_lr={hl}: {fmt(score_all(y, oof))}", flush=True)
        np.save(FIGDIR.parent / f"oof_t2_hl{hl}.npy", oof)
    print("ALLDONE", flush=True)


def holdout() -> None:
    """Fit the repaired T2 on all of train, score the held-out rows.

    Run:  uv run python solution/part3_headlr.py holdout
    NOTE: this is the SECOND time Part 3 touches the held-out slice -- the first
    was T1, chosen before this diagnostic existed. Reported with that caveat.
    """
    import time

    import torch
    from part3_transformer import encode, structured_matrix

    te_i = holdout_split(full)[1]
    test = full.iloc[te_i].reset_index(drop=True)
    y_te = test["human_score"].to_numpy()
    pred, model, tok = fit_predict(train, y, test, True, quiet=False, head_lr=1e-2)
    print(f"\nT2 head_lr=1e-2 held-out : {fmt(score_all(y_te, pred))}")
    print("Part 2 RandomForest      : QWK=0.398  MAE=0.784")
    np.save(FIGDIR.parent / "pred_test_t2hl.npy", pred)

    # Latency. The scaler is a serving artefact: fit ONCE, outside the loop.
    _, Ste_all = structured_matrix(train, test)
    model.eval()
    with torch.no_grad():
        for i in range(10):
            e = encode(tok, test.iloc[[i]])
            model(input_ids=e["input_ids"], attention_mask=e["attention_mask"],
                  structured=torch.tensor(Ste_all[i:i + 1]))
        ts = []
        for i in range(100):
            j = i % len(test)
            t0 = time.perf_counter()
            e = encode(tok, test.iloc[[j]])
            model(input_ids=e["input_ids"], attention_mask=e["attention_mask"],
                  structured=torch.tensor(Ste_all[j:j + 1]))
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print(f"latency batch-of-1: mean {ts.mean():.1f} ms  p95 {np.percentile(ts, 95):.1f} ms")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "holdout":
        holdout()
    else:
        sweep()
