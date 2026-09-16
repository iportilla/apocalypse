# Running This Toy as an Azure AutoML Regression Experiment

**⚠️ Same fabricated data as the rest of this repo.** This document explains
*how you would wire* `apocalypse_risk_fake_data.csv` up to Azure Machine
Learning's automated ML (AutoML), as an SDK/workflow reference. It is not a
real capability assessment, forecast, or claim about any AI lab, model, or
actual civilizational risk — see [README.md](README.md) for the full
disclaimer and the (fictional) six-scenario dataset this points at.

This is a companion to [`train_regression.py`](train_regression.py), not a
replacement for it. That script fits one plain `LinearRegression` per
scenario, locally, in under a second. This document walks through fitting
the *same six targets* on the *same five features* instead by asking Azure
AutoML to search across several regression algorithms, tune their
hyperparameters, and pick a winner per scenario — the kind of workflow
that's worth the cloud/compute overhead on a real dataset, even though on
this 210-row toy it's overkill. Reference:
[What is automated ML? (Azure Machine Learning docs)](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azureml-api-2).

## What AutoML actually does

Per the Azure docs, AutoML "creates many pipelines in parallel that try
different algorithms and parameters for you." Each iteration (called a
*trial*) produces a candidate model with a score against the metric you
chose; the process stops once it hits your **exit criteria** (a timeout, a
trial budget, or early termination when the score stops improving). It also
builds **voting and stacking ensembles** from the best individual trials by
default — for regression, the stacking meta-model is `ElasticNet`.

The official six-step AutoML workflow, and where this toy dataset lands in
each step:

| Azure's step | This experiment |
|---|---|
| 1. Identify the ML problem | **Regression** — six independent numeric targets, one per scenario |
| 2. Choose code-first or no-code | Code-first, via the **Python SDK v2** (`azure-ai-ml`) |
| 3. Specify training data | `apocalypse_risk_fake_data.csv`, one `MLTable` per scenario (see below) |
| 4. Configure AutoML parameters | `automl.regression(...)`, `set_limits(...)`, `set_training(...)` |
| 5. Submit the job | `ml_client.jobs.create_or_update(...)` |
| 6. Review results | Azure ML studio — metrics, the winning algorithm, optional ONNX export |

## Why six separate jobs, not one

Exactly as in `train_regression.py`, this is **not one model predicting six
numbers** — it's six independent AutoML experiments, one per scenario
column, each searching over the same five feature columns:

- `release_days_since_2023`, `benchmark_score`, `agentic_score`,
  `compute_log10_flops`, `open_weights`

And, just as importantly, each job's training data must **exclude**:

- `model_name` — a free-text identifier, not a numeric feature. Left in,
  AutoML's featurization would try to encode it, and its 4x jittered
  `(eval run N)` suffixes would leak information you don't want a model
  training on.
- The **other five** scenario columns — leaving them in would let AutoML
  "predict" one fake risk score from four other fake risk scores, which
  begs the question. This is the same reasoning in the main README's
  "Excluded columns" section, and it applies just as much to AutoML as it
  does to the plain `LinearRegression` baseline.

## Prerequisites

- An Azure subscription and an Azure Machine Learning workspace.
- A remote **compute cluster or compute instance** in that workspace —
  AutoML jobs submitted via the SDK v2 only run on Azure ML compute, not
  locally.
- Python packages, on top of what `train_regression.py` already needs:

```bash
pip install azure-ai-ml azure-identity mltable
```

## Step 1 — Connect to your workspace

```python
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group_name="<RESOURCE_GROUP>",
    workspace_name="<AZUREML_WORKSPACE_NAME>",
)
```

## Step 2 — Build one `MLTable` per scenario

AutoML v2 expects training data as an `MLTable` data asset, and the target
column has to live inside it. Since each scenario needs its own
feature/target slice of the CSV (see "Why six separate jobs" above), this
loop reuses the exact `FEATURES` / `SCENARIOS` lists from
`train_regression.py` to carve out six scenario-specific tables:

```python
import os
import pandas as pd
import mltable

FEATURES = [
    "release_days_since_2023",
    "benchmark_score",
    "agentic_score",
    "compute_log10_flops",
    "open_weights",
]
SCENARIOS = [
    "bio_weapon_uplift",
    "nuclear_miscalc_escalation",
    "grid_financial_cascade",
    "climate_tipping_cascade",
    "autonomous_weapons_spiral",
    "epistemic_collapse",
]

df = pd.read_csv("apocalypse_risk_fake_data.csv")

for scenario in SCENARIOS:
    data_dir = f"automl_data/{scenario}"
    os.makedirs(data_dir, exist_ok=True)
    df[FEATURES + [scenario]].to_csv(f"{data_dir}/data.csv", index=False)

    table = mltable.from_delimited_files([{"file": f"{data_dir}/data.csv"}])
    table.save(data_dir)   # writes automl_data/<scenario>/MLTable
```

With 210 rows per scenario (well under the 20,000-row threshold), AutoML's
default validation strategy is automatic cross-validation — 10 folds for
datasets under 1,000 rows — so `n_cross_validations` below is optional, but
set explicitly here for reproducibility.

## Step 3 — Configure and submit one AutoML regression job per scenario

```python
from azure.ai.ml import Input, automl
from azure.ai.ml.constants import AssetTypes

submitted_jobs = {}

for scenario in SCENARIOS:
    training_data = Input(
        type=AssetTypes.MLTABLE, path=f"automl_data/{scenario}"
    )

    regression_job = automl.regression(
        compute="cpu-cluster",              # your compute cluster/instance name
        experiment_name=f"apocalypse-{scenario}",
        training_data=training_data,
        target_column_name=scenario,
        primary_metric="r2_score",          # mirrors the R2 already reported
        n_cross_validations=5,
        enable_model_explainability=True,
    )

    # Exit criteria — scaled down from AutoML's defaults (1,000 trials,
    # 6-day timeout) since this is a 210-row toy dataset, not production data.
    regression_job.set_limits(
        timeout_minutes=30,
        trial_timeout_minutes=10,
        max_trials=25,
        max_concurrent_trials=4,
        enable_early_termination=True,
    )

    regression_job.set_training(
        enable_onnx_compatible_models=True,
    )

    submitted_jobs[scenario] = ml_client.jobs.create_or_update(regression_job)
    print(scenario, "->", submitted_jobs[scenario].services["Studio"].endpoint)
```

`primary_metric` is the one setting worth pausing on: AutoML's regression
primary metrics are `r2_score`, `normalized_mean_absolute_error`,
`normalized_root_mean_squared_error`, and `spearman_correlation` — there's
no raw "MAE" option, since primary metrics are normalized so trials are
comparable across folds. `r2_score` was picked here because it's the
headline metric `train_regression.py` already prints; after the job
finishes, `normalized_mean_absolute_error` is the closest analog to the
plain `MAE` column in `regression_results.csv` if you want a second number
to compare against.

## Step 4 — Review results

Each `submitted_jobs[scenario]` prints a Studio URL
(`returned_job.services["Studio"].endpoint`) where you can watch trials run
live, see which algorithm AutoML picked as the winner (candidates include
Elastic Net, LightGBM, Gradient Boosting, Decision Tree, K-Nearest
Neighbors, LARS Lasso, SGD, Random Forest, Extremely Randomized Trees, and
XGBoost, plus the ensembles built on top of them), and inspect the
automatic feature-importance breakdown from `enable_model_explainability`.
From there, a model can be registered and one-click deployed straight from
the studio.

**Worth carrying forward from the sklearn run:** `climate_tipping_cascade`
was deliberately generated with near-zero weight on every AI capability
feature (see `generate_apocalypse_data.py`), which is why the linear
baseline fits it far worse (R² 0.43) than the other five scenarios (R²
0.81–0.91). Expect the same pattern here — AutoML searching harder over
algorithms won't manufacture a real relationship between AI capability
features and a target that was constructed to be mostly independent of
them. If `climate_tipping_cascade`'s AutoML score comes back dramatically
higher than the sklearn baseline, that's a sign to double check for a data
leak, not a sign AutoML found a subtler pattern.

## How this compares to `train_regression.py`

| | `train_regression.py` (this repo, today) | Azure AutoML (this document) |
|---|---|---|
| Algorithm search | Fixed: one `LinearRegression` per scenario | Multiple regression algorithms + voting/stacking ensembles, searched per scenario |
| Hyperparameters | scikit-learn defaults | Tuned automatically per trial |
| Validation | Single 70/30 `train_test_split` | Cross-validation (or train/validation split), chosen automatically or via `n_cross_validations` |
| Featurization | Manual — the 5 feature columns are already clean numerics | Automatic scaling/imputation/encoding, customizable via `set_featurization()` |
| Where it runs | Local Python process, seconds, free | Azure ML remote compute, minutes-to-hours, billed |
| Output | `regression_results.csv` (R², MAE, linear coefficients) | Registered model + full run/metrics history in Azure ML studio, optional ONNX export |
| Explainability | Raw linear coefficients only | Optional automated feature-importance report |

For a 210-row dataset with 5 clean numeric features, the local
`LinearRegression` script remains the right tool — it already recovers the
generating signal well (see the results table in the main
[README.md](README.md)). This document exists so the same problem can be
pointed at Azure AutoML as a workflow reference, not because this toy
dataset needs it.

## Reproducing

```bash
pip install numpy pandas scikit-learn azure-ai-ml azure-identity mltable
python3 generate_apocalypse_data.py   # regenerate the fake dataset, if needed
# then run Steps 1-3 above against your own Azure ML workspace and compute
```
