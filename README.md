# Automated learner-response scoring

A RandomForest on 28 length / ASR-confidence / CEFR features scores **MAE 0.784,
QWK 0.398, 85.6% within ±1** on held-out data it was shown once, in 8.4 ms on one
CPU thread. A fine-tuned 118M-parameter multilingual MiniLM, given the same
features, **ties** it — for 100x the memory. The ceiling is in the data, not the
models: only 291 distinct transcripts back 2,000 rows, identical text is scored
anywhere from 0 to 4, and what the raters were judging is the audio.

```bash
uv sync
uv run python solution/part1_analysis.py             # EDA, ~1 min
uv run python solution/part2_classical.py           # classical ML, ~2 min
uv run python solution/part3_transformer.py         # T0 / T1 / T2, ~35 min on CPU
uv run python solution/part3_headlr.py              # T2 head learning-rate fix, ~25 min
uv run python solution/part3_headlr.py holdout      # its held-out run, ~4 min
uv run python solution/part3_transformer.py figure  # redraw the Part 3 figure, seconds
```

`common.py` holds the data, features, splits and metrics shared by all parts, so
Part 3 is scored on the same folds and the same held-out rows as Part 2.

---

## 1. What the data forced me to do

**Only 291 distinct transcripts back the 2,000 rows; 94% of rows share their text
with another row.** To size the risk I built a probe that cannot generalise at
all — it memorises each training transcript's mean score and looks it up.

| split | test rows found in lookup table | QWK |
|---|---|---|
| random 5-fold | 93.0% | 0.417 |
| grouped by transcript | 0.0% | 0.000 |

Grouped, the probe finds nothing, predicts a constant, and scores 0 — correct for
a model that learned nothing. Random, it scores 0.417, in the range a genuine
model reaches here. **A random split pays real-looking scores for pure
memorisation, so every split in this submission is grouped on the transcript.**

**Identical text gets different scores.** Of 173 repeated transcripts, only 14
were scored consistently; 19 span the full 0–4 range. `j'aime le foot` appears 63
times, scored 0 sixteen times and 4 once. Raters heard *audio*; the transcript is
a lossy view of it.

**ASR failure ≠ bad answer.** The 89 rows (4.5%) containing `xxx`, `...`,
`[inaudible]`, `*noise*` average 1.97 vs 1.94 elsewhere. `uh uh brrr klk
[inaudible] the the the` scored **4**: the learner spoke well, the microphone
did not.

I kept them in training. Dropping them gains +0.007 QWK on the clean rows —
noise — but makes the model score the failure rows themselves 0.61 points below
what humans gave, against 0.29 if they stay in. Trained without them, the model
has never met such a row and falls back on the pattern it learned elsewhere,
*short + low confidence = bad answer*, so it marks them down. Trained with them,
it learns the correct response to a broken transcript: predict near the average,
because there is no information here.

**ASR confidence is not monotone in score.** Deciles 1–9 rise 1.17 → 2.89, but
the lowest decile sits *above* decile 1. Mechanism unknown — possibly a
generation artefact. It is the concrete reason to prefer trees: one linear slope
predicts low exactly where the truth is highest.

Also: target unbalanced (12/26/31/19/12); CEFR the strongest single signal (A1
1.37 → C1 3.15); language and prompt flat, so no label bias to correct.

![Part 1 overview](figures/part1_overview.png)

---

## 2. Protocol and metrics

One group-aware **80/20 split**. The 403-row test slice is scored **once**, by
the single model chosen on 5-fold `StratifiedGroupKFold` over the 1,597 training
rows. Zero group overlap, asserted in code. Stratifying tightened the worst
fold-to-fold label imbalance from 7.2pp to 2.2pp — that is an evaluation-variance
fix, not an imbalance fix.

- **MAE** (primary) — average distance from the human rating, on the product's
  own scale.
- **QWK** — ordinal agreement corrected for chance. Any constant predictor scores
  exactly 0; accuracy would award "always predict 2" a flattering 31%.
- **within ±1** — 96.1% of *human–human* disagreement is within ±1, so this is
  the bar for behaving like a second rater.
- **exact accuracy** — for intuition only, never used to choose.

**No resampling or class weights.** This is regression on an ordinal target; the
imbalance is real (most responses genuinely are mediocre); and QWK already
corrects for the marginals.

---

## 3. Results

| | QWK | MAE | within ±1 | exact |
|---|---|---|---|---|
| always predict 2 | 0.000 | 0.925 | 0.763 | 0.312 |
| CEFR-level mean | 0.404 | 0.827 | 0.862 | 0.326 |
| RandomForest — CV over train | 0.522 | 0.718 | 0.885 | 0.404 |
| **RandomForest — held-out test** | **0.398** | **0.784** | **0.856** | **0.372** |
| human rater 2 vs rater 1 | 0.795 | 0.484 | 0.961 | 0.555 |

**The held-out number is materially worse than CV** (0.522 → 0.398). With 48 test
groups it is noisy, but that is what ~15 comparisons on the same folds buys you.
I report the held-out number. CV alone would have said 0.522 and I would have
believed it.

**Latency:** 8.41 ms mean / 8.62 ms p95, single-threaded — 2.8% of the 300 ms
budget.

---

## 4. Why this model

All models draw on five feature blocks, none of which requires reading a
particular language:

- **length** (7) — words, characters, mean word length, type-token ratio,
  repeat rate, punctuation count
- **ASR confidence** (14) — not just the mean but the shape of the per-word
  confidence distribution: min, max, std, percentiles, first/last word, share of
  words below 0.5 and 0.6, plus two ASR-failure flags. A uniform 0.6 across every
  word is a different situation from 0.9 everywhere with one catastrophic 0.1,
  and only the distribution separates them.
- **language** (6) — one-hot, with a spare slot for unseen codes
- **CEFR** (1) — the learner's level as an ordinal. Mapped over the full A1–C2
  ladder, not just the five levels present in this dataset
- **TF-IDF** — the transcript itself, as word 1–2 grams or `char_wb` 3–5 grams.
  Character n-grams because they need no per-language tokeniser and tolerate
  learner spelling and ASR errors

**Ablation — Ridge throughout, so only features vary:**

| | features | MAE | QWK |
|---|---|---|---|
| M1 | length only | 0.798 | 0.414 |
| M2 | + ASR confidence | 0.754 | 0.464 |
| M3 | + language | 0.754 | 0.464 |
| M4 | + CEFR | 0.749 | 0.479 |
| M5–M7 | TF-IDF variants (word / char / both) | 0.866–0.885 | 0.264–0.297 |
| M8 | structured + both TF-IDF | 0.750 | 0.490 |

Nearly all signal is **length + ASR confidence**. Language adds nothing. **Text
does not help**: TF-IDF alone sits below the CEFR baseline, and adding it to the
structured set does not improve MAE — under grouped CV the test transcript is
always unseen, so n-grams can only memorise. This predicts Part 3 will struggle,
since a transformer consumes exactly the input that is not paying here.

CEFR adds less than expected (0.754 → 0.749), being largely redundant with
length. But the model does answer partly *"how good is this person usually?"*
rather than *"how good was this answer?"*. Among responses a human scored 4, it
predicts **1.89** for A1 learners and **3.03** for B2/C1 learners — the same
quality of answer, a 1.1-point difference in what the learner is told. Removing
the CEFR column would not fix this, since length and confidence track proficiency
too.

**Model families, same features:** all use M4 (length + ASR + language + CEFR,
28 columns, no TF-IDF), so only the model varies.

| | MAE | QWK |
|---|---|---|
| Ridge | 0.749 | 0.479 |
| LogReg (argmax) | 0.810 | 0.496 |
| LogReg (expected score) | 0.751 | 0.470 |
| **RandomForest** | **0.718** | **0.522** |

RandomForest is here because Part 1's non-monotone confidence curve is something
no linear model can represent: it averages 300 trees, each splitting the features
at thresholds it chooses.

Ridge is only 0.03 MAE behind the winner and stays the sensible fallback if
operational simplicity matters. Logistic regression is included for calibrated
probabilities; its expected-score readout beats its own argmax, which discards
the ordinal information. **Ruled out:** SVM (O(n²) training), kNN (whole training
set resident at serving time), plain classifiers (discard ordinality).

**Per-language** (CV over train; QWK is unstable in narrow slices, read MAE):

| | n | MAE | within ±1 |
|---|---|---|---|
| de | 317 | 0.744 | 0.886 |
| en | 277 | 0.588 | 0.939 |
| es | 505 | 0.721 | 0.877 |
| fr | 323 | 0.765 | 0.876 |
| it | 175 | 0.783 | 0.834 |

English best, Italian worst — partly sample size (175 rows), but Italian is what
I would watch in production.

![Part 2 models](figures/part2_models.png)

---

## 5. Part 3 — transformer

`paraphrase-multilingual-MiniLM-L12-v2` (118M parameters, 12 layers). Multilingual,
so one model covers all five languages; small enough to fine-tune on a laptop CPU
in ~3 min per fold; and its sentence-similarity pretraining is the closest
available objective to *"does this answer the prompt"*. Input is the **pair**
`(prompt, transcript)` — relevance is a relation between the two, not a property
of either.

Three runs, on Part 2's exact folds and exact held-out rows:

| CV over the 1,597 training rows | MAE | QWK | within ±1 |
|---|---|---|---|
| T0 frozen MiniLM → Ridge (no fine-tuning) | 0.848 | 0.327 | 0.817 |
| T1 MiniLM fine-tuned, text only | 0.795 | 0.418 | 0.854 |
| T2 T1 + the 28 structured features | 0.730 | 0.498 | 0.879 |
| **Part 2 RandomForest** | **0.718** | **0.522** | **0.885** |
| human rater 2 vs rater 1 | 0.484 | 0.795 | 0.961 |

**Held-out: T2 scores MAE 0.759 / QWK 0.414 against the forest's 0.784 / 0.398** —
the two swap places between CV and held-out, and neither margin means anything on
48 groups. (T1 was scored on the held-out slice first, at 0.809 / 0.320; T2 is
therefore this slice's second use in Part 3, which makes it a weaker number than
Part 2's single-shot 0.784.)

**Verdict: a tie on accuracy, so the forest wins on cost.** A 118M-parameter
encoder needs 471 MB of weights and four CPU threads to match 28 hand-built
features in a few MB on one thread.

**Why the encoder adds nothing — Part 2 said it wouldn't.** The ablation there had
TF-IDF alone scoring *below* the CEFR baseline. The transformer eats the same
input, and three things stop it paying:

- **The label is not in the text.** `j'aime le foot` appears 63 times, scored 0
  sixteen times and 4 once. Raters heard audio; a perfect reader of the transcript
  cannot separate those rows.
- **291 distinct transcripts**, so under grouped CV every test transcript is unseen.
- **1,278 rows per fold** to fine-tune 118M parameters.

T0 → T1 (−0.053 MAE) shows fine-tuning does work; T1 → T2 (−0.065) shows the rest
of the gain is the structured columns, not the text. One trap on the way: at first
T2 tied T1 exactly, because the randomly-initialised head was training at the
encoder's 3e-5. Giving the head its own rate (1e-2) is the whole 0.800 → 0.730 —
see AI log #7.

![Part 3 models](figures/part3_models.png)

### Engineering

**Framework.** PyTorch + HuggingFace to fine-tune — the model is on the Hub and
the training loop is 40 lines. For serving, neither: export both candidates to
**ONNX Runtime**, so production runs one C++ runtime with no autograd machinery
and no pickle-versioning risk, and swapping one model for the other later is a
file change rather than a rewrite.

**Latency is not the constraint; memory is.** Measured one request at a time:

| | latency (mean / p95) | threads | resident weights |
|---|---|---|---|
| RandomForest (300 trees), incl. featurising | 8.4 / 8.6 ms | 1 | a few MB |
| T2 MiniLM + structured, incl. tokenising | 10.0 / 11.3 ms | 4 | 471 MB fp32 |

Both fit the 300 ms budget with room to spare — but the encoder needs four
threads to get there, and at millions of learners the 0.5 GB is the real bill:
it decides how many workers fit on a box. **82% of those parameters are the
250k-token multilingual vocabulary embedding**, most of which five languages
never address, so trimming the vocabulary and int8-quantising the linear layers
would plausibly reach ~40 MB. I did not do that work — a 471 MB model that only
ties a few-MB one has not earned it.

**On-device.** The result decides this rather than any platform argument: the
forest is a few MB, so it ships in the app, scores offline, and keeps the
transcript on the phone. I would run the same model both sides — on-device for
the instant "move on / re-prompt" decision, server-side for logging and drift
monitoring, since the standing cost of on-device is that a fix needs an app
release and you stop seeing the inputs you would learn from. A transformer would
make that choice genuinely hard: even quantised it competes with the app's own
memory budget and needs per-platform work (Core ML / ExecuTorch / TFLite).

**The next gain is upstream, not in the model.** ASR already runs in this pipeline
and already emits the per-word confidences that carry half the signal. Acoustic
features — speech rate, pauses, duration — change what the scorer *sees*, and
Part 3 is the evidence that changing the model does not.

---

## 6. Trust: when should the model refuse to score?

**The 89 ASR-failure rows (4.5%) should never reach the model.** Their
transcripts are `xxx`, `...`, `[inaudible]`, `*noise*` — we did not capture what
the learner said, so any score is invented. Their true scores span the whole
range (`{0:10, 1:15, 2:39, 3:18, 4:7}`) and the model, having no information,
predicts near the average for all of them. That is statistically correct and
product-wrong: the learner is shown a number they will read as an assessment of
their answer.

Gating them costs a regex, and unlike a confidence threshold the abstention
*repairs* the problem — "sorry, I didn't catch that" yields usable audio on the
retry.

**The gate sits in front of the model, so the rows still belong in the data:**

| stage | ASR-failure rows | why |
|---|---|---|
| training | keep | teaches "flag on → no information → predict the mean". Dropping them doubles this group's under-scoring (−0.29 → −0.61) |
| evaluation | keep | excluding them flatters MAE 0.718 → 0.711 without changing the model. Production sends every row, so the metric must count every row |
| serving | **gate** | we never heard the learner |

These are three separate decisions, not a contradiction. The regex will miss
novel ASR-failure patterns from a new ASR version or language; when one slips
through, training on these rows is what makes the model degrade to "predict the
mean" instead of confidently marking the learner down, and evaluating on them is
what tells me how bad that degradation is.

**Caveat:** the held-out set holds only 3 such rows against 86 in training —
grouping by transcript makes ~30 failure groups lump unevenly — so these numbers
come from cross-validation and are unconfirmed on held-out data.

---

## 7. Limitations

**The model never predicts 0, and predicts 4 twice in 403 rows.** Of 47 responses
scored 4 by a human, it called 43 a 2 or 3. This is regression-to-the-mean plus
rounding at .5, and it is invisible in MAE because hedging is *optimal* for MAE.
It matters for the product: the app cannot tell "excellent, move on" from "that
didn't work, re-prompt" if the scorer only emits the middle. The transformer does
it too — it never predicts 0 and predicts 4 eleven times in 1,597 — so this is a
property of the objective, not of the model family. Fix is to fit cut points on
inner-CV predictions — not yet done.

**Also:** no ordinal-specific model (ordered logit is the obvious gap); the
ablation is Ridge-only, so its conclusions may not transfer to trees; and no
nested CV — hyperparameters were judged on the same folds I report, which is the
most likely source of the CV-to-holdout gap.

**The transformer got one seed and almost no tuning.** 4 epochs, lr 3e-5, batch
16, mean pooling. The one thing I did search — the head's learning rate — was
worth 0.07 MAE on its own, which is a warning about what else is unexplored
rather than a reassurance.

---

## 8. Next step

**The raters scored three things** — grammar, relevance & completeness,
intelligibility, roughly equally weighted. **grammar and relevance have no
feature at all**. That gap, not model capacity, is what Part 3 ran into.

Two of the three close on the transcripts I already have — what is missing is
external knowledge, not more data:

- **Grammar** — errors per 100 words from LanguageTool (multilingual, offline).
  One column.
- **Relevance** — `cos(prompt, transcript)` from a *frozen* multilingual encoder.

**Then the audio.** The variance decomposition says 46% of the target is real
signal my features cannot see, and intelligibility is a third of the rubric that
no transcript can carry. Even coarse acoustic features — speech rate, pause count,
duration — would likely move the number more than any modelling change on the
current inputs. Part 3 is the evidence: a 118M-parameter multilingual encoder —
the largest modelling change available on this input — only drew level with the
forest. Change the input, not the model.

---

## 9. AI collaboration log

**Claude Code** (Opus) for implementation, **ChatGPT** for planning the Part 2
experiment structure. I worked section by section rather than requesting a
finished solution, and rejected output I could not read.

Things the AI got wrong that I caught:

1 **An overstated ceiling.** Rater 1 vs Rater 2, QWK=0.795. 
It is not a hard cap and now says "soft benchmark".

2 **Wrong oracle performance calculation** When calculating the ceiling of classical ML, 
it uses an oracle averaging scores of same transcript. But the score is not the ceiling of 
classical ML. Pointed out by me.