"""Part 3 -- fine-tuned transformer, compared against Part 2's classical winner.

Run:  uv run python solution/part3_transformer.py       (~35 min on a laptop CPU)

Three experiments, all on Part 2's exact folds and exact held-out rows:
  T0  frozen MiniLM embeddings -> Ridge      (is fine-tuning worth it?)
  T1  MiniLM fine-tuned on prompt+transcript (does semantics beat TF-IDF?)
  T2  T1 + structured features, late fusion  (fair rival to the RandomForest)
"""
from __future__ import annotations

import sys
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from common import (FIGDIR, LANGUAGES, SEED, feature_blocks, fmt, folds,
                    holdout_split, human_ceiling, load, score_all, to_labels)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 128          # transcripts are short; actual batches come out ~42 tokens
EPOCHS = 4
BATCH = 16
LR = 3e-5
# The head is randomly initialised while the encoder is pretrained. Training both
# at 3e-5 leaves the head unable to use the structured stream at all -- T2 then
# ties T1 exactly. See the AI collaboration log in the report.
HEAD_LR = 1e-2
STRUCTURED = ("length", "asr", "lang", "cefr")
torch.set_num_threads(4)


def h(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def seed_all(s: int = SEED) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


# --------------------------------------------------------------------------- #
class Scorer(nn.Module):
    """MiniLM encoder + optional late-fused structured features -> one scalar.

    Late fusion rather than serialising the numbers into the prompt text: ASR
    confidence is a continuous quantity, and a tokeniser trained on natural
    language has no sensible representation for "0.782". Keeping the two streams
    apart until the head also makes the T1/T2 ablation clean.
    """

    def __init__(self, n_structured: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL)
        hidden = self.encoder.config.hidden_size
        self.n_structured = n_structured
        self.head = nn.Linear(hidden + n_structured, 1)

    def forward(self, input_ids, attention_mask, structured=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pooling over real tokens. Steadier than the [CLS] vector when the
        # head is trained from scratch on 1,278 rows.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.n_structured:
            pooled = torch.cat([pooled, structured], dim=1)
        return self.head(pooled).squeeze(-1)


def encode(tok, df: pd.DataFrame):
    return tok(list(df["prompt"]), list(df["asr_transcript"]), truncation=True,
               max_length=MAX_LEN, padding=True, return_tensors="pt")


def structured_matrix(df_tr, df_te):
    Ntr = pd.concat([feature_blocks(df_tr)[b] for b in STRUCTURED], axis=1).to_numpy()
    Nte = pd.concat([feature_blocks(df_te)[b] for b in STRUCTURED], axis=1).to_numpy()
    sc = StandardScaler().fit(Ntr)          # fit on train fold only
    return sc.transform(Ntr).astype("float32"), sc.transform(Nte).astype("float32")


def fit_predict(df_tr, y_tr, df_te, use_structured: bool, quiet=True, head_lr=None):
    seed_all()
    tok = AutoTokenizer.from_pretrained(MODEL)
    Str, Ste = structured_matrix(df_tr, df_te) if use_structured else (None, None)
    model = Scorer(Str.shape[1] if use_structured else 0)

    enc_tr, enc_te = encode(tok, df_tr), encode(tok, df_te)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    # The head is randomly initialised while the encoder is pretrained, so they do
    # not want the same step size. head_lr=None keeps one rate for both (the
    # reported T1/T2 setting); passing a value gives the head its own.
    if head_lr is None:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(
            [{"params": model.encoder.parameters(), "lr": LR},
             {"params": model.head.parameters(), "lr": head_lr}],
            lr=LR, weight_decay=0.01)
    n = len(df_tr)
    total = EPOCHS * int(np.ceil(n / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR if head_lr is None else [LR, head_lr],
        total_steps=total, pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        order = np.random.permutation(n)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            kw = {"input_ids": enc_tr["input_ids"][b],
                  "attention_mask": enc_tr["attention_mask"][b]}
            if use_structured:
                kw["structured"] = torch.tensor(Str[b])
            loss = nn.functional.mse_loss(model(**kw), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if not quiet:
            print(f"    epoch {ep + 1}/{EPOCHS} loss {loss.item():.3f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(df_te), 64):
            kw = {"input_ids": enc_te["input_ids"][i:i + 64],
                  "attention_mask": enc_te["attention_mask"][i:i + 64]}
            if use_structured:
                kw["structured"] = torch.tensor(Ste[i:i + 64])
            preds.append(model(**kw).numpy())
    return np.concatenate(preds), model, tok


def frozen_embeddings(tok, encoder, df):
    enc = encode(tok, df)
    outs = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            o = encoder(input_ids=enc["input_ids"][i:i + 64],
                        attention_mask=enc["attention_mask"][i:i + 64])
            m = enc["attention_mask"][i:i + 64].unsqueeze(-1).float()
            outs.append(((o.last_hidden_state * m).sum(1) / m.sum(1)).numpy())
    return np.vstack(outs)


# --------------------------------------------------------------------------- #
def main() -> None:
    full = load().reset_index(drop=True)
    tr_i, te_i = holdout_split(full)
    train = full.iloc[tr_i].reset_index(drop=True)
    test = full.iloc[te_i].reset_index(drop=True)
    y_tr, y_te = train["human_score"].to_numpy(), test["human_score"].to_numpy()

    h("0. Setup -- identical protocol to Part 2")
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL)
    n_par = sum(p.numel() for p in enc.parameters())
    n_emb = enc.embeddings.word_embeddings.weight.numel()
    print(f"model: {MODEL}")
    print(f"  {n_par / 1e6:.0f}M parameters, {enc.config.num_hidden_layers} layers, "
          f"hidden {enc.config.hidden_size}, vocab {enc.config.vocab_size}")
    print(f"  {n_emb / n_par:.0%} of parameters are the vocabulary embedding "
          "-- the first thing to attack for on-device")
    print(f"train {len(train)} rows / {train['group_key'].nunique()} groups; "
          f"test {len(test)} rows / {test['group_key'].nunique()} groups "
          "(same rows as Part 2)")
    lens = [len(tok(p, t)["input_ids"]) for p, t in
            zip(train["prompt"][:400], train["asr_transcript"][:400])]
    print(f"token length: median {int(np.median(lens))}, p95 "
          f"{int(np.percentile(lens, 95))}, max {max(lens)} -- max_length={MAX_LEN} "
          "never binds")

    results = {}

    # ---------------------------------------------------------------- T0
    h("T0. Frozen MiniLM embeddings -> Ridge (no fine-tuning)")
    print("Is the pretrained representation already useful, before any training?\n")
    t0 = time.time()
    oof = np.zeros(len(train))
    for a, b in folds(train, grouped=True):
        Etr = frozen_embeddings(tok, enc, train.iloc[a])
        Ete = frozen_embeddings(tok, enc, train.iloc[b])
        oof[b] = Ridge(alpha=10.0).fit(Etr, y_tr[a]).predict(Ete)
    results["T0 frozen MiniLM + Ridge"] = score_all(y_tr, oof)
    print(f"  {fmt(results['T0 frozen MiniLM + Ridge'])}   [{time.time() - t0:.0f}s]")

    # ---------------------------------------------------------------- T1, T2
    for tag, use_struct, hl, label in [
            ("T1", False, None, "T1 MiniLM fine-tuned (text only)"),
            ("T2", True, HEAD_LR, "T2 MiniLM fine-tuned + structured")]:
        h(f"{tag}. {label}")
        t0 = time.time()
        oof = np.zeros(len(train))
        for k, (a, b) in enumerate(folds(train, grouped=True)):
            p, _, _ = fit_predict(train.iloc[a], y_tr[a], train.iloc[b],
                                  use_struct, head_lr=hl)
            oof[b] = p
            print(f"  fold {k + 1}/5 done  [{time.time() - t0:.0f}s]", flush=True)
        results[label] = score_all(y_tr, oof)
        print(f"  {fmt(results[label])}")
        np.save(FIGDIR.parent / f"oof_{tag.lower()}.npy", oof)

    # ---------------------------------------------------------------- compare
    h("Comparison -- cross-validation over the training slice")
    rf_oof = np.load(FIGDIR.parent / "oof_part2.npy")
    results["Part 2 RandomForest (structured)"] = score_all(y_tr, rf_oof)
    tbl = pd.DataFrame(results).T[["MAE", "QWK", "within1", "exact"]].round(3)
    print(tbl.sort_values("MAE").to_string())
    print(f"\nhuman benchmark: {fmt(human_ceiling(full))}")

    # ---------------------------------------------------------------- holdout
    h("Held-out test -- the best transformer, touched once")
    best = min([k for k in results if k.startswith("T")],
               key=lambda k: results[k]["MAE"])
    print(f"best transformer on CV: {best}\n")
    pred_te, model, tok2 = fit_predict(train, y_tr, test, best.startswith("T2"),
                                       quiet=False,
                                       head_lr=HEAD_LR if best.startswith("T2") else None)
    s_te = score_all(y_te, pred_te)
    print(f"\n  {best}")
    print(f"    CV over train : {fmt(results[best])}")
    print(f"    held-out test : {fmt(s_te)}")
    print(f"\n  Part 2 RandomForest held-out: QWK=0.398  MAE=0.784")
    np.save(FIGDIR.parent / "pred_test_t.npy", pred_te)

    # ---------------------------------------------------------------- latency
    h("CPU latency and size")
    model.eval()
    one = test.iloc[[0]]
    e1 = encode(tok2, one)
    kw = {"input_ids": e1["input_ids"], "attention_mask": e1["attention_mask"]}
    if best.startswith("T2"):
        _, S1 = structured_matrix(train, one)
        kw["structured"] = torch.tensor(S1)
    with torch.no_grad():
        for _ in range(10):
            model(**kw)
        ts = []
        for i in range(100):
            r = test.iloc[[i % len(test)]]
            t0 = time.perf_counter()
            e = encode(tok2, r)
            k = {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
            if best.startswith("T2"):
                _, S = structured_matrix(train, r)
                k["structured"] = torch.tensor(S)
            model(**k)
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  tokenise + forward, batch of 1: mean {ts.mean():.1f} ms  "
          f"p95 {np.percentile(ts, 95):.1f} ms  max {ts.max():.1f} ms")
    print(f"  budget 300 ms -> using {ts.mean() / 300:.0%}")
    print(f"  fp32 weights: {size:.0f} MB   vs RandomForest 8.4 ms and a few MB")

    pd.DataFrame(results).T.to_csv(FIGDIR.parent / "part3_results.csv")
    print(f"\nwrote {FIGDIR.parent / 'part3_results.csv'}")


# --------------------------------------------------------------------------- #
def figure() -> None:
    """Redraw the Part 3 figure from the saved OOF arrays.

    Run:  uv run python solution/part3_transformer.py figure
    Kept separate from main() because main() costs ~35 min and this costs none.
    """
    full = load().reset_index(drop=True)
    tr_i, _ = holdout_split(full)
    train = full.iloc[tr_i].reset_index(drop=True)
    y = train["human_score"].to_numpy()
    rf = to_labels(np.load(FIGDIR.parent / "oof_part2.npy"))
    t2_cont = np.load(FIGDIR.parent / "oof_t2.npy")
    res = pd.read_csv(FIGDIR.parent / "part3_results.csv", index_col=0)
    res = res.drop(index="T2 MiniLM fine-tuned + structured")
    res.loc["T2 fine-tuned + structured"] = score_all(y, t2_cont)
    t2 = to_labels(t2_cont)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    # -- 1. who wins, on Part 2's folds
    a = res.sort_values("MAE", ascending=False)
    ax[0].barh(range(len(a)), a["MAE"],
               color=["#4C72B0" if i.startswith("Part 2") else "#8172B2"
                      for i in a.index])
    ax[0].set_yticks(range(len(a)))
    ax[0].set_yticklabels(a.index, fontsize=9)
    hc = human_ceiling(full)["MAE"]
    ax[0].axvline(hc, color="#C44E52", ls="--")
    ax[0].text(hc, len(a) - 1.2, " human rater 2", color="#C44E52",
               fontsize=8, va="center")
    ax[0].set_xlabel("MAE (lower better)")
    ax[0].set_title("The transformer reaches the forest, not past it\n"
                    "(same folds and rows as Part 2)", fontsize=10)

    # -- 2. neither model uses the ends of the scale
    w, xs = 0.27, np.arange(5)
    for off, vals, lab, col in [
            (-w, np.bincount(y, minlength=5), "human", "#C44E52"),
            (0., np.bincount(rf, minlength=5), "RandomForest", "#4C72B0"),
            (w, np.bincount(t2, minlength=5), "MiniLM + structured", "#8172B2")]:
        ax[1].bar(xs + off, vals / len(y), w, label=lab, color=col)
    ax[1].set_xticks(xs)
    ax[1].set_xlabel("score"); ax[1].set_ylabel("share of rows")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Both models collapse to the middle\n"
                    "12% of answers are 0s; both models call 0 on under 0.5%",
                    fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGDIR / "part3_models.png", dpi=130)
    print(f"wrote {FIGDIR / 'part3_models.png'}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figure":
        figure()
    else:
        main()
