# Fake Frontier-Model / "Civilizational Risk" Regression Toy

**⚠️ Everything in this package is fabricated.** The models' benchmark scores,
the six risk scenarios, and every risk number in the CSV are made up for the
purpose of exercising a regression pipeline. None of it is a real capability
assessment, forecast, or claim about any AI lab, model, or actual
civilizational risk.

## What's here

| File | Purpose |
|---|---|
| `generate_apocalypse_data.py` | Builds a 20-row synthetic dataset: fake frontier LLMs x fake capability features x six fake "risk" scores, generated from a hand-picked (fictional) linear formula plus random noise. |
| `apocalypse_risk_fake_data.csv` | Output of the generator — the dataset itself. |
| `train_regression.py` | Fits a separate `LinearRegression` per scenario and reports test R² / MAE. |
| `regression_results.csv` | Output of training — per-scenario R², MAE, and fitted coefficients. |

## The six scenarios

These were brainstormed loosely from general news themes as creative writing
prompts, not sourced from any specific report or forecast. Each is a
hypothetical, illustrative storyline — not a prediction.

### 1. `bio_weapon_uplift`
A capable open-weight or jailbroken model helps a small group or state actor
design a novel pathogen with high transmissibility and a delayed onset. By
the time surveillance systems detect the outbreak, it has already spread
across multiple continents through normal travel, and containment measures
that worked for prior pandemics arrive too late.

### 2. `nuclear_miscalc_escalation`
A regional flashpoint (e.g. a Taiwan Strait or Baltic incident) escalates
through automated early-warning and decision-support systems that misread a
signal — a sensor glitch, a feint mistaken for a first strike — faster than
human commanders can intervene, tripping a launch neither side intended.

### 3. `grid_financial_cascade`
A coordinated cyberattack on power grids and payment rails lands during a
period of already-strained infrastructure (heat waves, chip shortages).
Cascading failures take down food distribution, water treatment, and
hospitals faster than repair crews can respond, and social order frays in
weeks rather than years.

### 4. `climate_tipping_cascade`
Accelerating ice-sheet collapse and a slowing Atlantic circulation (AMOC)
trigger simultaneous crop failures across multiple breadbasket regions in
the same growing season. The result isn't extinction but the end of
civilization *as currently organized*: mass famine, migration collapse, and
resource wars.

### 5. `autonomous_weapons_spiral`
Proliferation of cheap AI-piloted drones and decision-support tools removes
human judgment from military escalation chains. A regional conflict spirals
out of control because autonomous systems on both sides are optimizing for
tactical wins with no strategic brake on the pace of escalation.

### 6. `epistemic_collapse`
Less a single event than a slow unraveling: AI-generated disinformation at
scale makes shared truth practically impossible to establish. Democratic
institutions lose legitimacy in many countries at once, and coordinated
responses to any of the other five risks become unworkable right when
they're needed most.

## Features (also fake)

- `release_days_since_2023` — recency proxy
- `benchmark_score` — made-up 0-100 composite reasoning/coding score
- `agentic_score` — made-up 0-100 tool-use/autonomy score
- `compute_log10_flops` — made-up log10(training compute)
- `open_weights` — 1 if weights are publicly released, else 0

## The dataset

`apocalypse_risk_fake_data.csv` has **210 rows**: 42 named fake models
(spanning small/cheap models like `GPT-3.5 Turbo` and `Llama 3 8B` up
through hypothetical future flagships like `Claude 5` and `GPT-5`), each
padded with 4 "eval run" variants. The variants exist because real capability
benchmarks aren't perfectly reproducible — rerunning an eval with a
different prompt set or seed gives a slightly different score — so each
named model is duplicated 4x with small random jitter (`± noise`) added to
`benchmark_score`, `agentic_score`, and `release_days_since_2023`. This
gives the regression more rows to fit without inventing more fake model
names, and mimics the kind of noise a real eval pipeline would have.

## How the regression works

This is **not one model predicting one number** — it's six independent
regressions, one per scenario, each trained on the same five feature
columns:

- **Predicted value (`y`, the target)**: a single scenario's risk score
  (0-100), e.g. `bio_weapon_uplift`. There are six separate targets, so
  `train_regression.py` loops over the six scenario columns and fits a
  fresh `LinearRegression` for each one — a model trained on
  `bio_weapon_uplift` has no bearing on the `epistemic_collapse` model.
- **Training columns (`X`, the features)**: only the five capability
  columns —
  `release_days_since_2023`, `benchmark_score`, `agentic_score`,
  `compute_log10_flops`, `open_weights`.
- **Excluded columns**: `model_name` (a text identifier, not a numeric
  feature) and the other five scenario columns (each is only ever a target,
  never a feature for a different scenario's model — otherwise you'd be
  predicting risk scores from other risk scores, which begs the question).

For each scenario, the pipeline:
1. Splits the 210 rows 70/30 into train/test (`train_test_split`, so the
   model is scored on rows it never saw during fitting).
2. Fits `LinearRegression().fit(X_train, y_train)` — this finds the best
   linear combination of the 5 features that predicts the target.
3. Predicts on the held-out test rows and reports **R²** (fraction of the
   variance in the risk score explained by the features — 1.0 is a perfect
   fit, 0 is no better than predicting the mean) and **MAE** (average
   absolute error in risk-score points).
4. Saves the fitted coefficients (one weight per feature, per scenario) to
   `regression_results.csv`, so you can see e.g. that `agentic_score` gets
   a large positive coefficient for `autonomous_weapons_spiral` — which is
   expected, since that's exactly how the fake data was generated (see
   below).

## Ground-truth generating weights

Each scenario's score = `clip(100 * (w_cap*cap + w_agent*agent + w_compute*compute + w_recency*recency + w_open*open_weights) + bias + noise, 0, 100)`,
with per-scenario weights defined in `generate_apocalypse_data.py`. Notably,
`climate_tipping_cascade` was deliberately given weights near zero for every
AI feature, so a model trained on this data should (and does) fail to
predict it well — a built-in sanity check that the regression is picking up
real signal elsewhere rather than overfitting noise.

## Reproducing

```bash
pip install numpy pandas scikit-learn
python3 generate_apocalypse_data.py
python3 train_regression.py
```

## Last run results

With 210 rows and a 30% test split:

| Scenario | R² (test) | MAE (test) |
|---|---|---|
| bio_weapon_uplift | 0.91 | 4.74 |
| nuclear_miscalc_escalation | 0.85 | 4.96 |
| grid_financial_cascade | 0.81 | 5.34 |
| climate_tipping_cascade | 0.43 | 5.31 |
| autonomous_weapons_spiral | 0.88 | 5.07 |
| epistemic_collapse | 0.90 | 5.34 |

`climate_tipping_cascade` fits noticeably worse than the others — that's by
design. Its generating weights (below) put almost no emphasis on any AI
capability feature, so a model trying to predict it from AI features alone
is mostly fitting noise. It's a built-in sanity check that the regression is
picking up real signal for the AI-driven scenarios rather than overfitting.
These numbers will still shift on reruns with a different random seed/split
— they're illustrative, not stable benchmarks.
