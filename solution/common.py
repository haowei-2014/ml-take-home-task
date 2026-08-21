"""Shared data loading, feature engineering, splitting and metrics.

Imported by part1_analysis.py, part2_classical.py and part3_transformer.py so
that all three parts see exactly the same rows, folds and metric definitions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "candidate" / "dataset.csv"
FIGDIR = Path(__file__).resolve().parent / "figures"

# Full CEFR ladder, not just the levels present in dataset.csv -- C2 is a valid
# production input even though this sample has none (see DATA_DICTIONARY note).
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_ORDINAL = {lvl: i for i, lvl in enumerate(CEFR_LEVELS)}
LANGUAGES = ["de", "en", "es", "fr", "it"]

SEED = 20260821
N_SPLITS = 5

# Tokens an ASR emits when it fails outright rather than when the learner spoke badly.
ASR_FAILURE_RE = re.compile(
    r"(?:xxx|\.\.\.|\?\?\?|mmm|brrr|klk|\*noise\*|\[inaudible\]|\bhmm\b)", re.IGNORECASE
)


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["asr_transcript"] = df["asr_transcript"].fillna("")
    df["word_confs"] = df["asr_word_confidences"].apply(json.loads)
    return df


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _conf_stats(confs: list[float]) -> dict[str, float]:
    if not confs:
        return dict.fromkeys(
            ["c_min", "c_max", "c_std", "c_p10", "c_p25", "c_first", "c_last",
             "c_frac_lt50", "c_frac_lt60", "c_range"], 0.0
        )
    a = np.asarray(confs, dtype=float)
    return {
        "c_min": float(a.min()),
        "c_max": float(a.max()),
        "c_std": float(a.std()),
        "c_p10": float(np.percentile(a, 10)),
        "c_p25": float(np.percentile(a, 25)),
        "c_first": float(a[0]),
        "c_last": float(a[-1]),
        "c_frac_lt50": float((a < 0.5).mean()),
        "c_frac_lt60": float((a < 0.6).mean()),
        "c_range": float(a.max() - a.min()),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Language-agnostic surface features.

    Deliberately does not read transcript *content* word-by-word: the app serves
    five languages today and must not need a new feature set per language.
    """
    rows = []
    for t, confs in zip(df["asr_transcript"], df["word_confs"]):
        words = t.split()
        n = len(words)
        lower = [w.lower() for w in words]
        # "the the the" style ASR stutter
        repeats = sum(1 for i in range(1, n) if lower[i] == lower[i - 1])
        f = {
            "n_words": n,
            "n_chars": len(t),
            "mean_word_len": (len(t.replace(" ", "")) / n) if n else 0.0,
            "type_token_ratio": (len(set(lower)) / n) if n else 0.0,
            "repeat_rate": (repeats / n) if n else 0.0,
            "n_punct": sum(ch in ",.;:!?" for ch in t),
            "asr_failure_flag": float(bool(ASR_FAILURE_RE.search(t))),
            "asr_failure_token_rate": (
                sum(bool(ASR_FAILURE_RE.fullmatch(w)) for w in words) / n if n else 0.0
            ),
        }
        f.update(_conf_stats(confs))
        rows.append(f)

    X = pd.DataFrame(rows, index=df.index)
    X["c_mean"] = df["asr_mean_confidence"].to_numpy()
    X["log_n_words"] = np.log1p(X["n_words"])
    # Length and confidence interact: many words at low confidence is a
    # different situation from few words at low confidence.
    X["words_x_conf"] = X["log_n_words"] * X["c_mean"]

    X["cefr_ordinal"] = df["cefr_level"].map(CEFR_ORDINAL).astype(float)
    # Unseen levels (e.g. C2 in production) must not become NaN.
    X["cefr_ordinal"] = X["cefr_ordinal"].fillna(float(np.mean(list(CEFR_ORDINAL.values()))))
    for lang in LANGUAGES:
        X[f"lang_{lang}"] = (df["target_language"] == lang).astype(float)
    X["lang_other"] = (~df["target_language"].isin(LANGUAGES)).astype(float)
    return X


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def folds(df: pd.DataFrame, grouped: bool = True):
    """Yield (train_idx, test_idx) positional index arrays.

    grouped=True groups by asr_transcript. Only 291 distinct transcripts back
    2,000 rows, so a random split puts near-copies of a test row into training
    and inflates every metric. Grouped is the honest setting; random is kept so
    part2 can quantify the gap.
    """
    if grouped:
        splitter = GroupKFold(n_splits=N_SPLITS)
        return splitter.split(df, groups=df["asr_transcript"])
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return splitter.split(df)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def to_labels(y: np.ndarray) -> np.ndarray:
    """Continuous regression output -> integer 0..4 score."""
    return np.clip(np.rint(np.asarray(y, dtype=float)), 0, 4).astype(int)


def score_all(y_true, y_pred_cont) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_lab = to_labels(y_pred_cont)
    cont = np.asarray(y_pred_cont, dtype=float)
    return {
        "QWK": cohen_kappa_score(y_true, y_lab, weights="quadratic",
                                 labels=[0, 1, 2, 3, 4]),
        "MAE": float(np.abs(y_true - y_lab).mean()),
        "within1": float((np.abs(y_true - y_lab) <= 1).mean()),
        "exact": float((y_true == y_lab).mean()),
        "spearman": float(pd.Series(y_true).corr(pd.Series(cont), method="spearman")),
        "RMSE_cont": float(np.sqrt(((y_true - cont) ** 2).mean())),
    }


def human_ceiling(df: pd.DataFrame) -> dict[str, float]:
    """Rater-2 vs rater-1 on the ~15% doubly-rated rows: the practical ceiling."""
    d = df.dropna(subset=["human_score_2"])
    return score_all(d["human_score"].to_numpy(), d["human_score_2"].to_numpy())


def fmt(d: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items())
