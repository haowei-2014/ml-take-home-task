"""Part 1 -- exploratory analysis.

Run:  uv run python solution/part1_analysis.py
Writes figures to solution/figures/; every number it prints is quoted in README.md.
"""
from __future__ import annotations

print("Running Part 1 -- exploratory analysis.  Needs ~1 min.", flush=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ASR_FAILURE_RE, CEFR_LEVELS, FIGDIR, fmt, human_ceiling, load

pd.set_option("display.width", 200)


def h(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    df = load()
    df["n_words"] = df["asr_transcript"].str.split().str.len()
    df["asr_failed"] = df["asr_transcript"].str.contains(ASR_FAILURE_RE)

    h("1. Shape and integrity")
    print(f"rows={len(df)}  columns={df.shape[1]}")
    raw = df.drop(columns=["word_confs", "n_words", "asr_failed"])
    print(pd.DataFrame({"nulls": raw.isna().sum(), "nunique": raw.nunique()}))
    conf_len = df["word_confs"].apply(len)
    print(f"\nword-confidence list length != word count on "
          f"{(conf_len != df['n_words']).sum()} rows  (0 = internally consistent)")
    print(f"asr_mean_confidence != mean(word_confs) by >0.01 on "
          f"{(abs(df['asr_mean_confidence'] - df['word_confs'].apply(np.mean)) > 0.01).sum()} rows")

    h("2. Target distribution -- unbalanced, and it matters for the metric")
    tgt = df["human_score"].value_counts().sort_index()
    print(pd.DataFrame({"n": tgt, "share": (tgt / len(df)).round(3)}))
    print(f"mean={df['human_score'].mean():.3f}  std={df['human_score'].std():.3f}")
    print("Majority class (2) is 31% -- accuracy is a weak metric here; the score "
          "is ordinal, so distance-aware metrics (QWK / MAE) are the right call.")

    h("3. THE BIG ONE -- transcripts repeat, so a random split leaks")
    vc = df["asr_transcript"].value_counts()
    print(f"distinct transcripts: {len(vc)} across {len(df)} rows")
    print(f"rows whose transcript also appears elsewhere: "
          f"{(df['asr_transcript'].map(vc) > 1).sum()} ({(df['asr_transcript'].map(vc) > 1).mean():.1%})")
    print(f"distinct prompts: {df['prompt'].nunique()}")
    print("\nmost frequent transcripts:")
    print(vc.head(6))
    g = df.groupby("asr_transcript")["human_score"].agg(["count", "mean", "std"])
    multi = g[g["count"] > 1]          # transcripts appearing on 2+ rows
    once = g[g["count"] == 1]

    print(f"\nOf the {len(g)} distinct transcripts: {len(multi)} appear on 2+ rows "
          f"(re-used by different learners), {len(once)} appear exactly once.")
    print(f"\nDo the {len(multi)} re-used transcripts get a consistent human score?")
    print("(one row per bucket; 'transcripts' = how many of those "
          f"{len(multi)} texts fall in it)\n")
    n_distinct = df.groupby("asr_transcript")["human_score"].nunique()[multi.index]
    spread = n_distinct.value_counts().sort_index()
    print(pd.DataFrame({
        "distinct_scores_given": spread.index,
        "transcripts": spread.to_numpy(),
        "share": (spread / len(multi)).round(3).to_numpy(),
        "reading": [
            "identical text -> raters always agreed" if k == 1
            else f"identical text -> raters used {k} different scores"
            for k in spread.index
        ],
    }).to_string(index=False))

    worst = multi.loc[n_distinct[n_distinct == n_distinct.max()].index]
    ex = worst["count"].idxmax()
    ex_rows = df.loc[df["asr_transcript"] == ex, "human_score"]
    print(f'\nworked example -- "{ex}" appears {len(ex_rows)} times, scored:')
    print("   " + "  ".join(f"{k}:{v}" for k, v in
                            ex_rows.value_counts().sort_index().items()))

    print(f"\nmedian within-transcript score std: {multi['std'].median():.3f}  "
          f"vs overall std {df['human_score'].std():.3f}")
    print("=> identical text carries a ~1-point spread of human scores. Raters scored "
          "AUDIO; the transcript is a lossy view of it. This is an irreducible ceiling, "
          "and it means we must split by transcript, not at random.")

    h("4. Human agreement -- what a 'good' number even looks like")
    d2 = df.dropna(subset=["human_score_2"])
    r1 = d2["human_score"].to_numpy()
    r2 = d2["human_score_2"].astype(int).to_numpy()
    qwk_h = human_ceiling(df)["QWK"]
    print(f"doubly-rated rows: {len(d2)} ({len(d2) / len(df):.1%})")
    print("rater-2 vs rater-1:", fmt(human_ceiling(df)))
    rr = np.corrcoef(r1, r2)[0, 1]
    print(f"\nQWK {qwk_h:.3f} is a SOFT benchmark, not a hard cap. Every rating is the")
    print("true quality plus that rater's own noise, so rater 2 handicaps itself in a")
    print("way a model does not -- a model predicting the underlying truth adds no")
    print(f"noise of its own. With rater-rater r={rr:.3f}, such a model would score")
    print(f"~{np.sqrt(rr):.3f} against a single rater, ABOVE the number above.")
    print("The binding constraint is information, not agreement -- see the variance")
    print("split at the end of this section.")
    print("\nconfusion (rater1 rows x rater2 cols):")
    print(pd.crosstab(d2["human_score"], d2["human_score_2"].astype(int)))
    print(f"\nrater means: r1={d2['human_score'].mean():.3f} r2={d2['human_score_2'].mean():.3f} "
          "-> no systematic severity bias between raters")
    print(f"""
Is that benchmark safe to apply to all {len(df)} rows?
  The QWK above is computed on the {len(d2)} rows that carry a second rating --
  they are the only rows where two humans can be compared. But it is quoted as
  THE benchmark for the whole dataset, which only holds if those {len(d2)} rows were
  picked at random. Annotation pipelines often send only the *hard* cases for a
  second opinion; if that happened here, the two raters would disagree more than
  usual on this subset, QWK would come out too low, and every model would look
  closer to human than it really is.

  Check: does the target break down the same way in both populations?
    'all rows'    = all {len(df)} rows in dataset.csv
    'doubly rated'= the {len(d2)} of them where human_score_2 is not empty
  Counts are shown next to shares because the two populations are different
  sizes; only the shares are comparable.""")

    a = df["human_score"].value_counts().sort_index()
    b = d2["human_score"].value_counts().sort_index()
    dist = pd.DataFrame({
        "all rows (n)": a,
        "all rows (share)": (a / len(df)).round(3),
        "doubly rated (n)": b,
        "doubly rated (share)": (b / len(d2)).round(3),
        "gap (pp)": ((b / len(d2) - a / len(df)) * 100).round(1),
    })
    dist.index.name = "human_score"
    print(dist.to_string())
    print(f"\nLargest gap is {abs((b / len(d2) - a / len(df)) * 100).max():.1f} "
          "percentage points. No score band is over-represented among the doubly\n"
          "rated rows -- if hard cases had been cherry-picked for a second opinion,\n"
          "the extreme bands (0 and 4) or the crowded middle (2) would stand out.")

    print("\nThe same check on the variables that actually track difficulty:")
    d2m = df["human_score_2"].notna()
    checks = pd.DataFrame({
        "all rows": [df["asr_mean_confidence"].mean(), df["n_words"].mean(),
                     df["asr_failed"].mean()],
        "doubly rated": [df.loc[d2m, "asr_mean_confidence"].mean(),
                         df.loc[d2m, "n_words"].mean(), df.loc[d2m, "asr_failed"].mean()],
    }, index=["mean ASR confidence", "mean words per transcript", "ASR-failure rate"])
    print(checks.round(4).to_string())
    print(f"=> the second rater looks randomly assigned, so QWK {qwk_h:.3f} is usable "
          f"as a\n   benchmark for all {len(df)} rows. (Matching distributions is "
          "evidence of\n   representativeness, not proof -- n=308 cannot rule out a "
          "small bias.)")

    v_total = df["human_score"].to_numpy().var()
    v_noise = ((r1 - r2) ** 2).mean() / 2      # Var(r1-r2) = 2*Var(noise) if independent
    print(f"""
How much of the target is knowable at all?
  total variance of human_score   {v_total:.3f}  100%
  pure rater noise                {v_noise:.3f}  {v_noise / v_total:.0%}  <- no model can predict this
  everything else                 {v_total - v_noise:.3f}  {1 - v_noise / v_total:.0%}  <- real quality signal
Only a fifth of the variance is annotation noise. The rest is genuine signal --
but most of it lives in the audio (pronunciation, fluency, hesitation) and never
reaches the ASR transcript. A text-only model is capped well below the human
number for reasons of INPUT, not of algorithm.""")

    h("5. Where the signal is")
    print("""Which input columns move the target? For each candidate predictor, split
the rows into its groups and look at the AVERAGE human_score inside each group.
A predictor is useful when those group averages are far apart: knowing the group
then tells you something about the score. A predictor whose group averages are
all the same is useless on its own.

In the next three tables every number describes human_score (the 0-4 target):
  rows            = how many utterances fall in this group
  mean_score      = average human_score of those rows   <- the column that matters
  std_score       = spread of human_score inside the group
  vs_overall      = mean_score minus the overall mean ({:.2f}), in score points
The dataset-wide average is {:.2f} and the scale runs 0-4.""".format(
        df["human_score"].mean(), df["human_score"].mean()))

    def by(col: str, note: str) -> None:
        g = df.groupby(col)["human_score"].agg(
            rows="count", mean_score="mean", std_score="std")
        g["vs_overall"] = g["mean_score"] - df["human_score"].mean()
        spread = g["mean_score"].max() - g["mean_score"].min()
        print(f"\n-- grouped by {col} -- {note}")
        print(g.round(2).to_string())
        print(f"   spread of group means: {spread:.2f} score points "
              f"(from {g['mean_score'].min():.2f} to {g['mean_score'].max():.2f})")

    by("cefr_level", "the learner's proficiency level")
    print("   => STRONG. A1 learners average a full 1.8 points below C1 learners.\n"
          "      This one costs nothing to use in production: the app already stores\n"
          "      each learner's CEFR level, so scoring a request means reading one\n"
          "      field off the user profile -- no extra computation, no extra service\n"
          "      call, no added latency. Contrast a feature like 'grammar errors found\n"
          "      by a parser', which would have to be computed per request inside the\n"
          "      300 ms budget.\n"
          "      Because it is that cheap and that predictive, 'predict the average\n"
          "      score of this CEFR level' is the baseline every model has to beat --\n"
          "      it is model #2 in Part 2 for exactly this reason.\n"
          "      Caveat: a real profile's CEFR can be stale or self-declared, so the\n"
          "      model should not collapse into reading this field alone.")

    by("target_language", "the language being learned")
    print("   => FLAT. All five languages sit within 0.14 points of each other, so\n"
          "      the raters were not systematically harsher on any one language.\n"
          "      Useless as a predictor -- but good news: no label bias to correct.\n"
          "      (Per-language ERROR rates still get reported separately in Part 2:\n"
          "      equal average scores do not guarantee equal model accuracy.)")

    by("prompt", "which speaking prompt the learner answered")
    print("   => FLAT. 10 prompts, ~200 rows each, all within 0.28 points. Prompt\n"
          "      difficulty was evidently balanced during data generation.")

    print("""
-- transcript length --
NOTE the flip: the three tables above grouped BY a predictor and averaged the
score. This one groups BY the score and averages the length, because length is
a number rather than a category. Read it as: utterances a human scored 0 ran
4.2 words on average; those scored 4 ran 12.4.""")
    ln = df.groupby("human_score")["n_words"].agg(
        rows="count", mean_words="mean", median_words="median")
    print(ln.round(2).to_string())

    print("\n-- the two numeric predictors, as correlations --")
    print("Pearson r on all 2000 rows: +1 = perfectly rises together, 0 = unrelated.")
    print(f"  r(n_words, human_score)             = {df['n_words'].corr(df['human_score']):+.3f}"
          "   <- longer answers score higher")
    print(f"  r(asr_mean_confidence, human_score) = "
          f"{df['asr_mean_confidence'].corr(df['human_score']):+.3f}"
          "   <- weaker, and section 6 shows why")

    h("6. ASR confidence does NOT map monotonically to score")
    conf = df["asr_mean_confidence"]
    df["conf_decile"], edges = pd.qcut(conf, 10, labels=False, retbins=True)
    dec = df.groupby("conf_decile").agg(
        n=("human_score", "size"),
        mean_conf=("asr_mean_confidence", "mean"),
        mean_score=("human_score", "mean"),
        asr_fail_rate=("asr_failed", "mean"))
    dec.insert(1, "covers", [f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(edges[:-1], edges[1:])])
    dec.insert(2, "width", edges[1:] - edges[:-1])
    print("Deciles = rows sorted by confidence and cut into 10 groups of ~200. Equal\n"
          "COUNTS, not equal widths: decile 0 spans 0.441 of the scale, decile 4 only\n"
          "0.035. Check `covers`/`width` before comparing rows.\n")
    print(dec.round(3).to_string())
    print("\nDeciles 1-9 rise smoothly 1.17 -> 2.89; decile 0 sits ABOVE decile 1.")
    print("But decile 0 is too wide to trust -- it mixes conf 0.07 with conf 0.50.")

    print("A linear term in confidence must fit one slope, so it predicts LOW at the\n"
          "bottom of the scale where the mean is actually high -- the concrete reason\n"
          "Part 2 prefers trees, which can carve out that region as its own split.")

    h("7. ASR failures: 'the transcript is garbage' != 'the answer was bad'")
    fail = df[df["asr_failed"]]
    print(f"rows with ASR-failure tokens (xxx / ... / ??? / [inaudible] / noise): "
          f"{len(fail)} ({len(fail) / len(df):.1%})")
    print(f"mean human score on those rows: {fail['human_score'].mean():.2f} "
          f"vs {df[~df['asr_failed']]['human_score'].mean():.2f} elsewhere")
    print(fail.groupby("human_score").size().rename("n"))
    print(fail["asr_transcript"].value_counts().head(8))
    print("\n=> these rows are unlearnable from text and are exactly where a confident "
          "wrong answer costs the most trust. Handled in Part 2 as a route-to-abstain "
          "flag rather than something to fit.")

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    tgt.plot(kind="bar", ax=ax[0, 0], color="#4C72B0")
    ax[0, 0].set_title("Target is unbalanced\n(human_score)")
    ax[0, 0].set_xlabel("human_score"); ax[0, 0].set_ylabel("rows")

    order = [c for c in CEFR_LEVELS if c in set(df["cefr_level"])]
    df.boxplot(column="human_score", by="cefr_level", ax=ax[0, 1], grid=False,
               positions=range(len(order)))
    ax[0, 1].set_title("CEFR is the strongest single signal"); ax[0, 1].set_xlabel("cefr_level")

    ax[0, 2].plot(dec["mean_conf"], dec["mean_score"], "o-", color="#C44E52")
    ax[0, 2].annotate("lowest decile breaks the trend\n(and it is not the ASR failures)",
                      xy=(dec["mean_conf"].iloc[0], dec["mean_score"].iloc[0]),
                      xytext=(0.45, 2.6), fontsize=9,
                      arrowprops=dict(arrowstyle="->", color="#555"))
    ax[0, 2].set_title("Confidence -> score is non-monotone")
    ax[0, 2].set_xlabel("mean ASR confidence (decile)"); ax[0, 2].set_ylabel("mean human_score")

    ax[1, 0].scatter(df["n_words"] + np.random.uniform(-.3, .3, len(df)),
                     df["human_score"] + np.random.uniform(-.15, .15, len(df)),
                     s=6, alpha=.25, color="#55A868")
    _r = df["n_words"].corr(df["human_score"])
    ax[1, 0].set_title("Length vs score\n"
                       f"Pearson r = {_r:.2f}  (+1 = perfectly rises together,\n"
                       "0 = unrelated): longer answers do score higher",
                       fontsize=10)
    ax[1, 0].set_xlabel("words in transcript"); ax[1, 0].set_ylabel("human_score")

    ax[1, 1].bar(vc.head(25).index.astype(str).str.slice(0, 22), vc.head(25).values,
                 color="#8172B2")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_title(f"Top-25 transcripts cover {vc.head(25).sum() / len(df):.0%} of rows\n"
                       f"({len(vc)} distinct texts / {len(df)} rows)")
    ax[1, 1].set_ylabel("rows")

    cm = pd.crosstab(d2["human_score"], d2["human_score_2"].astype(int))
    im = ax[1, 2].imshow(cm.to_numpy(), cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[1, 2].text(j, i, cm.to_numpy()[i, j], ha="center", va="center", fontsize=9)
    ax[1, 2].set_title(f"Rater 1 vs rater 2 (n={len(d2)})\n"
                       f"QWK={qwk_h:.3f} -- soft benchmark, not a hard cap", fontsize=10)
    ax[1, 2].set_xlabel("rater 2"); ax[1, 2].set_ylabel("rater 1")
    fig.colorbar(im, ax=ax[1, 2], fraction=.046)

    fig.suptitle("")
    fig.tight_layout()
    out = FIGDIR / "part1_overview.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
