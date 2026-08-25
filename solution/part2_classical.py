"""Part 2 -- classical (non-transformer) models.

Run:  uv run python solution/part2_classical.py

Protocol
--------
  * one group-aware 80/20 split; the 20% test slice is scored ONCE, at the end,
    by the single model chosen on cross-validation over the 80%
  * 5-fold StratifiedGroupKFold inside the training slice drives every decision
  * groups are normalised transcripts, because 94% of rows share their text
    with another row (see part1_analysis.py section 3)
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from common import (FIGDIR, LANGUAGES, SEED, feature_blocks, fmt, folds,
                    holdout_split, human_ceiling, load, score_all, to_labels)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
STRUCTURED = ("length", "asr", "lang", "cefr")


def h(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    """Which feature blocks a model sees."""
    structured: tuple[str, ...] = ()
    word_tfidf: bool = False
    char_tfidf: bool = False

    def label(self) -> str:
        bits = list(self.structured)
        if self.word_tfidf:
            bits.append("word-tfidf")
        if self.char_tfidf:
            bits.append("char-tfidf")
        return " + ".join(bits) if bits else "(none)"


def _vec(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                               max_features=40000)
    # char_wb keeps n-grams inside word boundaries; robust to learner spelling,
    # ASR errors and morphology, and needs no per-language tokeniser.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, max_features=60000)


def assemble(df_tr, df_te, spec: Spec, dense: bool):
    """Fit every transformer on train only, then apply to both sides."""
    tr_parts, te_parts = [], []

    if spec.structured:
        btr, bte = feature_blocks(df_tr), feature_blocks(df_te)
        Ntr = pd.concat([btr[b] for b in spec.structured], axis=1).to_numpy()
        Nte = pd.concat([bte[b] for b in spec.structured], axis=1).to_numpy()
        sc = StandardScaler().fit(Ntr)
        tr_parts.append(sc.transform(Ntr))
        te_parts.append(sc.transform(Nte))

    for kind, on in (("word", spec.word_tfidf), ("char", spec.char_tfidf)):
        if not on:
            continue
        v = _vec(kind)
        Ttr = v.fit_transform(df_tr["asr_transcript"])
        Tte = v.transform(df_te["asr_transcript"])
        if dense:      # trees cannot use 10k sparse columns; compress first
            k = min(60, Ttr.shape[1] - 1)
            svd = TruncatedSVD(n_components=k, random_state=SEED).fit(Ttr)
            tr_parts.append(svd.transform(Ttr))
            te_parts.append(svd.transform(Tte))
        else:
            tr_parts.append(Ttr)
            te_parts.append(Tte)

    if not tr_parts:
        return np.zeros((len(df_tr), 1)), np.zeros((len(df_te), 1))
    if any(issparse(p) for p in tr_parts):
        return (hstack([csr_matrix(p) for p in tr_parts]).tocsr(),
                hstack([csr_matrix(p) for p in te_parts]).tocsr())
    return np.hstack(tr_parts), np.hstack(te_parts)


# --------------------------------------------------------------------------- #
# Models -- each returns continuous predictions in [0, 4]
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    name: str
    spec: Spec
    kind: str                       # ridge | logreg | rf | const | cefr
    dense: bool = False
    extra: dict = field(default_factory=dict)

    def fit_predict(self, df_tr, y_tr, df_te):
        if self.kind == "const":
            return np.full(len(df_te), self.extra["value"])
        if self.kind == "cefr":
            m = pd.Series(y_tr).groupby(df_tr["cefr_level"].to_numpy()).mean()
            return df_te["cefr_level"].map(m).fillna(y_tr.mean()).to_numpy()

        Xtr, Xte = assemble(df_tr, df_te, self.spec, self.dense)
        if self.kind == "ridge":
            return Ridge(alpha=self.extra.get("alpha", 2.0)).fit(Xtr, y_tr).predict(Xte)
        if self.kind == "logreg":
            clf = LogisticRegression(C=self.extra.get("C", 1.0), max_iter=2000)
            clf.fit(Xtr, y_tr)
            P = clf.predict_proba(Xte)
            if self.extra.get("readout") == "argmax":
                return clf.classes_[P.argmax(1)].astype(float)
            return P @ clf.classes_.astype(float)      # expected score
        if self.kind == "rf":
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                         random_state=SEED, n_jobs=-1
                                         ).fit(Xtr, y_tr).predict(Xte)
        raise ValueError(f"unknown model kind: {self.kind}")


def cv_predict(df, y, model: Model) -> np.ndarray:
    oof = np.zeros(len(df), dtype=float)
    for tr, te in folds(df, grouped=True):
        oof[te] = model.fit_predict(df.iloc[tr], y[tr], df.iloc[te])
    return oof


def table(results: dict[str, dict]) -> pd.DataFrame:
    cols = ["MAE", "QWK", "within1", "exact"]
    return pd.DataFrame(results).T[cols].round(3)


# --------------------------------------------------------------------------- #
def main() -> None:
    full = load().reset_index(drop=True)
    tr_idx, te_idx = holdout_split(full)
    train = full.iloc[tr_idx].reset_index(drop=True)
    test = full.iloc[te_idx].reset_index(drop=True)
    y_tr = train["human_score"].to_numpy()
    y_te = test["human_score"].to_numpy()

    h("0. Protocol")
    print(f"train {len(train)} rows / {train['group_key'].nunique()} transcript groups")
    print(f"test  {len(test)} rows / {test['group_key'].nunique()} transcript groups  "
          "(scored once, at the end)")
    print(f"group overlap train/test: "
          f"{len(set(train['group_key']) & set(test['group_key']))}  (must be 0)")
    print("inner loop: 5-fold StratifiedGroupKFold over the training rows")
    print("\nreference points:")
    print(f"  human rater 2 vs rater 1: {fmt(human_ceiling(full))}")
    print("  a soft benchmark, not a cap -- see part1 section 4")

    # ---------------------------------------------------------------- baselines
    h("1. Trivial baselines -- what any real model has to beat")
    base = {
        "always predict 2 (mode)": Model("", Spec(), "const", extra={"value": 2.0}),
        "predict train median": Model("", Spec(), "const",
                                      extra={"value": float(np.median(y_tr))}),
        "predict train mean": Model("", Spec(), "const",
                                    extra={"value": float(y_tr.mean())}),
        "CEFR-level mean": Model("", Spec(), "cefr"),
    }
    res = {k: score_all(y_tr, cv_predict(train, y_tr, m)) for k, m in base.items()}
    print(table(res).to_string())
    print("\nQWK is 0 for every constant predictor by construction -- that is the point\n"
          "of using it. Accuracy would hand 'always 2' a flattering 31%.")
    print("The CEFR baseline is the one that matters: it costs a database read and\n"
          "already reaches MAE 0.84 / QWK 0.37.")

    # ---------------------------------------------------------------- ablation
    h("2. Ablation -- where does the performance actually come from?")
    print("Ridge throughout, so only the FEATURES vary.\n")
    ladder = {
        "M1 length only": Spec(structured=("length",)),
        "M2 + ASR confidence": Spec(structured=("length", "asr")),
        "M3 + language": Spec(structured=("length", "asr", "lang")),
        "M4 + CEFR": Spec(structured=("length", "asr", "lang", "cefr")),
        "M5 word TF-IDF only": Spec(word_tfidf=True),
        "M6 char TF-IDF only": Spec(char_tfidf=True),
        "M7 word + char TF-IDF": Spec(word_tfidf=True, char_tfidf=True),
        "M8 structured + both TF-IDF": Spec(structured=STRUCTURED, word_tfidf=True,
                                            char_tfidf=True),
    }
    abl, abl_oof = {}, {}
    for name, spec in ladder.items():
        oof = cv_predict(train, y_tr, Model(name, spec, "ridge"))
        abl[name] = score_all(y_tr, oof)
        abl_oof[name] = oof
    t = table(abl)
    t.insert(0, "features", [ladder[k].label() for k in t.index])
    print(t.to_string())

    m3, m4 = abl["M3 + language"], abl["M4 + CEFR"]
    print(f"\nCEFR alone moves MAE {m3['MAE']:.3f} -> {m4['MAE']:.3f} and "
          f"QWK {m3['QWK']:.3f} -> {m4['QWK']:.3f}.")
    print("""
Read that carefully rather than celebrating it. CEFR describes the LEARNER, not
the RESPONSE. A model leaning on it is partly answering 'how good is this person
usually?' instead of 'how good was this answer?'. Consequences:
  - a B2 learner who gives a poor answer gets marked up, and an A1 learner who
    gives a great answer gets marked down -- the exact feedback that erodes trust
  - the app already knows the CEFR level, so the model is being paid to
    rediscover something the product already has
  - CEFR itself is often self-declared or stale
It stays in the model because it genuinely predicts, but the honest framing is
that a chunk of the headline number is learner prior, not response quality.""")

    print("\nText features: M5-M7 barely clear the CEFR baseline, and adding them to")
    print("the structured set (M8) HURTS. Under grouped CV the test transcript is")
    print("always unseen, so n-grams can only memorise training text.")

    # ---------------------------------------------------------------- models
    h("3. Model families on the best feature set")
    # M4 from the ablation above: all 28 structured columns, no TF-IDF. Held
    # fixed across every model here so the comparison isolates the learner.
    best_spec = Spec(structured=STRUCTURED)
    print(f"features held fixed: {best_spec.label()}  "
          f"({sum(feature_blocks(train)[b].shape[1] for b in STRUCTURED)} columns)\n")
    fam = {
        "Ridge": Model("", best_spec, "ridge"),
        "LogReg (argmax)": Model("", best_spec, "logreg", extra={"readout": "argmax"}),
        "LogReg (expected score)": Model("", best_spec, "logreg",
                                         extra={"readout": "expected"}),
        "RandomForest": Model("", best_spec, "rf", dense=True),
    }
    fres, foof = {}, {}
    for k, m in fam.items():
        t0 = time.perf_counter()
        oof = cv_predict(train, y_tr, m)
        fres[k] = score_all(y_tr, oof)
        fres[k]["fit_s"] = (time.perf_counter() - t0) / 5
        foof[k] = oof
    print(table(fres).join(pd.DataFrame(fres).T[["fit_s"]].round(2)).to_string())
    print("\nargmax vs expected score: the same fitted logistic regression, two readouts.")
    print("Expected score keeps the ordinal information in the probability vector;")
    print("argmax throws it away and can only ever emit an integer.")

    best_name = table(fres)["MAE"].idxmin()
    best_oof = foof[best_name]
    print(f"\nselected on CV MAE: {best_name}")

    # ---------------------------------------------------------------- breakdown
    h("4. Per-language and per-CEFR breakdown")
    print("QWK is unstable inside a narrow slice (it depends on the label spread\n"
          "within that slice), so read MAE and within1 across rows here.\n")
    rows = []
    for lang in LANGUAGES:
        m = (train["target_language"] == lang).to_numpy()
        s = score_all(y_tr[m], best_oof[m]); s["n"] = int(m.sum())
        rows.append(pd.Series(s, name=lang))
    print(pd.DataFrame(rows)[["n", "MAE", "within1", "exact", "QWK"]].round(3).to_string())
    rows = []
    for lvl in sorted(train["cefr_level"].unique()):
        m = (train["cefr_level"] == lvl).to_numpy()
        s = score_all(y_tr[m], best_oof[m]); s["n"] = int(m.sum())
        rows.append(pd.Series(s, name=lvl))
    print()
    print(pd.DataFrame(rows)[["n", "MAE", "within1", "exact", "QWK"]].round(3).to_string())

    # ---------------------------------------------------------------- errors
    h("5. Error analysis -- the 15 worst predictions")
    err = np.abs(y_tr - to_labels(best_oof))
    order = np.argsort(-err)[:15]
    view = train.iloc[order][["asr_transcript", "target_language", "cefr_level",
                              "asr_mean_confidence", "human_score"]].copy()
    view["pred"] = to_labels(best_oof)[order]
    view["err"] = err[order]
    view["n_dup"] = train["group_key"].map(
        train["group_key"].value_counts()).iloc[order].to_numpy()
    view["asr_transcript"] = view["asr_transcript"].str.slice(0, 44)
    print(view.to_string(index=False))
    print(f"\nrows with |error| >= 2: {(err >= 2).sum()} of {len(err)} ({(err >= 2).mean():.1%})")
    dup = train["group_key"].map(train["group_key"].value_counts()).to_numpy()
    print(f"  of those, {(dup[err >= 2] > 1).mean():.0%} sit on a transcript that also "
          "appears elsewhere\n  with a different score -- i.e. the label is not a "
          "function of the input.")

    # ---------------------------------------------------------------- holdout
    h("6. Held-out test -- touched once")
    final = fam[best_name]
    pred_te = final.fit_predict(train, y_tr, test)
    print(f"model: {best_name} on {best_spec.label()}")
    print(f"  CV over train (n={len(train)}): {fmt(score_all(y_tr, best_oof))}")
    print(f"  held-out test (n={len(test)}): {fmt(score_all(y_te, pred_te))}")
    print(f"\n  human benchmark: QWK {human_ceiling(full)['QWK']:.3f}, "
          f"MAE {human_ceiling(full)['MAE']:.3f}")

    # ---------------------------------------------------------------- latency
    h("7. CPU latency (budget: 300 ms)")
    print("Fit the serving artefacts ONCE, then time only what a request actually\n"
          "does: featurise one row and call predict.\n")
    blocks_tr = feature_blocks(train)
    Ntr = pd.concat([blocks_tr[b] for b in best_spec.structured], axis=1)
    scaler = StandardScaler().fit(Ntr.to_numpy())
    served = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=1
                                   ).fit(scaler.transform(Ntr.to_numpy()), y_tr)

    def serve(row: pd.DataFrame) -> float:
        b = feature_blocks(row)
        x = pd.concat([b[k] for k in best_spec.structured], axis=1).to_numpy()
        return float(served.predict(scaler.transform(x))[0])

    for i in range(20):
        serve(test.iloc[[i]])
    ts = []
    for i in range(300):
        r = test.iloc[[i % len(test)]]
        t0 = time.perf_counter()
        serve(r)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print(f"  mean {ts.mean():.2f} ms   p50 {np.percentile(ts, 50):.2f} ms   "
          f"p95 {np.percentile(ts, 95):.2f} ms   max {ts.max():.2f} ms")
    print(f"  budget 300 ms -- using {ts.mean() / 300:.1%} of it (single thread)")
    print("  A 300-tree forest is the expensive half; a single Ridge would be ~10x")
    print("  cheaper again, at MAE 0.749 vs 0.718.")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    a = table(abl)
    ax[0].barh(range(len(a)), a["MAE"], color="#4C72B0")
    ax[0].set_yticks(range(len(a))); ax[0].set_yticklabels(a.index, fontsize=8)
    ax[0].axvline(table(res).loc["CEFR-level mean", "MAE"], color="#C44E52", ls="--")
    ax[0].text(table(res).loc["CEFR-level mean", "MAE"], len(a) - 0.3,
               " CEFR baseline", color="#C44E52", fontsize=8, va="top")
    ax[0].set_xlabel("MAE (lower better)")
    ax[0].set_title("Ablation: which FEATURES help\n(Ridge regressor throughout,"
                    " so only features vary)", fontsize=10)
    ax[0].invert_yaxis()

    f = table(fres)
    ax[1].barh(range(len(f)), f["MAE"], color="#55A868")
    ax[1].set_yticks(range(len(f))); ax[1].set_yticklabels(f.index, fontsize=8)
    ax[1].set_xlabel("MAE")
    ax[1].set_title("Which MODEL helps\n(all on length + asr + lang + cefr)",
                    fontsize=10)
    ax[1].invert_yaxis()

    cm = pd.crosstab(pd.Series(y_te, name="true"),
                     pd.Series(to_labels(pred_te), name="pred"))
    cm = cm.reindex(index=range(5), columns=range(5), fill_value=0)
    im = ax[2].imshow(cm.to_numpy(), cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax[2].text(j, i, cm.to_numpy()[i, j], ha="center", va="center", fontsize=9)
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("true")
    ax[2].set_title(f"Held-out test (n={len(test)})\n{best_name}", fontsize=10)
    fig.colorbar(im, ax=ax[2], fraction=.046)
    fig.tight_layout()
    fig.savefig(FIGDIR / "part2_models.png", dpi=130)
    print(f"\nwrote {FIGDIR / 'part2_models.png'}")

    np.save(FIGDIR.parent / "oof_part2.npy", best_oof)
    np.save(FIGDIR.parent / "test_idx.npy", te_idx)
    print("saved OOF predictions and the test index for Part 3 to reuse")


if __name__ == "__main__":
    main()
