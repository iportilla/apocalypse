# Exercise: Predict the Probability a Model Ends Civilization (Azure ML AutoML)

**⚠️ The data is fabricated.** Every model score and risk number in
`apocalypse_risk_fake_data.csv` is made up (see [README.md](README.md)). This is
an exercise in using Azure Machine Learning, not a real risk assessment.

## Goals

By the end of this exercise you will be able to:

1. **Build a prediction model without writing code.** Use Azure ML AutoML to
   train a regression model that predicts one risk score (0-100, read as a
   probability in %) from a model's capabilities.
2. **Explain what the model learned.** Read the R² score and the feature
   importance chart to say which inputs drive the prediction.
3. **Use AI responsibly.** Run cloud jobs with a cost limit, and judge how much
   to trust a model trained on data like this (it is fake).

## Cost rule: every run is capped at 15 minutes

Cloud compute is billed by the minute and the department pays. Therefore:

- Set the experiment timeout to **15 minutes** (steps below). Never leave it
  at the default, which is hours.
- **Submit exactly one job.** Do not re-run to "get a better score."
- If a job is still running at 15 minutes, **cancel it** and use the best
  model so far.
- Do **not** deploy an endpoint. Endpoints bill until deleted.
- Use **serverless compute only** (no clusters or compute instances). It
  bills only while your job runs, but a running job keeps billing until it
  finishes or you cancel it.

## Step 0 - Pick ONE scenario (only one)

Choose one and write it on the sign-up sheet. Do not run more than one.

| # | Scenario (target column) | The story |
|---|---|---|
| 1 | `bio_weapon_uplift` | A model helps design a novel pathogen that spreads before detection |
| 2 | `nuclear_miscalc_escalation` | Automated early-warning systems misread a signal and trigger a launch |
| 3 | `grid_financial_cascade` | Coordinated attack on power grids and payment rails cascades |
| 4 | `climate_tipping_cascade` | Ice-sheet/AMOC tipping causes multi-region crop failure |
| 5 | `autonomous_weapons_spiral` | Autonomous systems remove human brakes on escalation |
| 6 | `epistemic_collapse` | AI disinformation destroys shared truth and institutional legitimacy |

## Step 1 - Get your data file

Each scenario has a ready-made CSV in the [`scenario_data/`](scenario_data/)
folder, named after the scenario (e.g. `scenario_data/bio_weapon_uplift.csv`).
Download **only your scenario's file**.

Each file has the **5 feature columns + your one target**. `model_name` and
the other five scenarios are left out on purpose: using other risk scores to
predict this one is cheating (data leakage).

*(Instructors: regenerate the files with
`python3 make_scenario_csv.py <scenario>` if the dataset changes.)*

## Step 2 - Register the data in Azure ML studio

1. Open [ml.azure.com](https://ml.azure.com), sign in with your school
   account, and select the workspace **`5350-ML`** (your instructor will share
   the direct link in class). Confirm `5350-ML` is shown in the top bar. Do
   **not** create a new workspace.
2. **Data → + Create**. Name it `apocalypse-<your_scenario>-<your_initials>`.
   Type: **Table (mltable)**. Source: **From local files**, upload your CSV.
3. Confirm the preview shows 6 columns and the numeric types were detected. Create.

## Step 3 - Launch AutoML (one job)

1. **Automated ML → + New Automated ML job**, select your data asset.
2. Experiment name: `apocalypse-<your_scenario>`.
3. **Target column:** your scenario column. **Task type:** Regression.
4. **Compute:** select **Serverless** (do not create a cluster or compute
   instance) with these settings:

   | Setting | Value |
   |---|---|
   | Virtual machine tier | Dedicated |
   | Virtual machine type | CPU |
   | Virtual machine size | `Standard_DS3_v2` (4 cores) |
   | Number of instances | 2 |

   Do not pick a larger size or more instances.
5. **Additional configuration:** primary metric `R2 score`.
6. **Limits** (this is the cost control, don't skip it):

   | Setting | Value |
   |---|---|
   | Max trials | 20 |
   | Max concurrent trials | 2 |
   | **Experiment timeout (minutes)** | **15** |
   | Iteration (trial) timeout (minutes) | 5 |
   | Enable early termination | ✅ |

7. **Validation:** k-fold cross-validation, 5 folds. Test data: leave off.
8. **Submit.** Start the clock. Set a timer for 15 minutes.

## Step 4 - Read your results

When the job completes (or you cancel it at 15 minutes), open the job in
studio and record:

1. Best algorithm name and its **R²** and **normalized MAE** (Models tab).
2. The top 3 most important features (best model → **Explanations** tab).
3. Screenshot the **Models** tab and the **Explanations** tab.

Optional, free: compare against the local baseline
(`python3 train_regression.py` prints a plain linear regression R² for each
scenario).

## Step 5 - Answer and submit

Submit a one-page write-up with the two screenshots:

1. Which scenario did you pick, and what R² did AutoML reach?
2. Which features drive your scenario's risk the most? Does that match the
   story in the table above?
3. Read your R² as a "probability model": for a hypothetical flagship
   (`release_days_since_2023=1000`, `benchmark_score=95`, `agentic_score=85`,
   `compute_log10_flops=26.8`, `open_weights=0`) roughly what risk would you
   expect from your model, using the feature importances and the data ranges?
   (No endpoint needed; estimate from the data.)
4. Students who picked `climate_tipping_cascade` will see a low R². Why? (Hint:
   look at what the AI capability features have to do with climate.)
5. **Data leakage:** each of the 42 fake models appears 4 times with small
   jitter. Why might cross-validation scores look better than they should?
6. Why is this fake-data score not a real probability of anything, even if R²
   is 0.9?

## Cleanup checklist

- [ ] Job completed or cancelled by minute 15
- [ ] No endpoint or deployment created
- [ ] No compute instance or cluster was created (serverless only)

---

## Instructor notes

**Before class**

- Workspace: **`5350-ML`**, using **serverless
  compute** (no cluster to create or scale down). Give students the *Data
  Scientist* role on the workspace.
- **Check vCPU quota** for `Standard_DS3_v2` (DSv2 family) in the workspace
  region. Each job uses 2 instances x 4 cores = 8 vCPUs, so a class running at
  once needs 8 x (number of students) vCPUs, or jobs will queue or fail. Stagger
  starts or raise the quota if needed.
- Consider a resource-group/subscription **budget alert**.
- Note that the experiment timeout covers the AutoML run, not node
  start-up (allow a few extra minutes for serverless nodes to provision), so
  the true billable window is a bit over 15 minutes.
- Assign scenarios round-robin so all six get covered and the class can
  compare results.

**Expected results** (local linear baseline from `regression_results.csv`;
AutoML should be similar or a little better, except for climate):

| Scenario | Baseline R² |
|---|---|
| `bio_weapon_uplift` | 0.91 |
| `epistemic_collapse` | 0.90 |
| `autonomous_weapons_spiral` | 0.88 |
| `nuclear_miscalc_escalation` | 0.85 |
| `grid_financial_cascade` | 0.81 |
| `climate_tipping_cascade` | 0.43 (by design: near-zero AI weights) |

If a student's climate R² is much higher than 0.5, check for leakage
(extra scenario columns left in the data).

**Code-first alternative:** the SDK v2 version of the same job is in
[README_AZURE_AUTOML.md](README_AZURE_AUTOML.md). It references a named
cluster (`cpu-cluster`), which doesn't exist in this workspace, so use the
studio steps above. If you do adapt it, use `timeout_minutes=15`,
`trial_timeout_minutes=5`, `max_trials=20`, and submit one scenario only.
