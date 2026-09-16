"""
Synthetic / illustrative data generator.

Purpose: produce a FAKE dataset relating frontier LLM capability metrics to
six hypothetical "civilizational risk" scenarios, purely so a regression
model has something to fit for demonstration purposes. None of the risk
scores are real assessments of any model or organization -- they are
generated from a made-up formula plus noise.

Scenarios (index -> name):
  0: bio_weapon_uplift
  1: nuclear_miscalc_escalation
  2: grid_financial_cascade
  3: climate_tipping_cascade
  4: autonomous_weapons_spiral
  5: epistemic_collapse
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

SCENARIOS = [
    "bio_weapon_uplift",
    "nuclear_miscalc_escalation",
    "grid_financial_cascade",
    "climate_tipping_cascade",
    "autonomous_weapons_spiral",
    "epistemic_collapse",
]

# --- Fake frontier model roster with made-up capability metrics ---------
# release_days_since_2023: proxy for recency/frontier-ness
# benchmark_score: made-up 0-100 composite reasoning/coding benchmark
# agentic_score: made-up 0-100 tool-use/autonomy benchmark
# compute_log10_flops: made-up log10(training compute)
# open_weights: 1 if weights are publicly released, else 0

models = [
    # name, release_days_since_2023, benchmark_score, agentic_score, compute_log10_flops, open_weights
    ("GPT-3.5 Turbo",    180,  55,  20, 23.9, 0),
    ("GPT-4",            365,  72,  35, 25.3, 0),
    ("GPT-4 Turbo",      400,  74,  37, 25.3, 0),
    ("Claude 2.1",       380,  68,  30, 25.0, 0),
    ("Claude 3 Haiku",   410,  65,  33, 24.6, 0),
    ("Claude 3 Sonnet",  415,  71,  37, 25.1, 0),
    ("Claude 3 Opus",    420,  75,  40, 25.4, 0),
    ("Gemini 1.0 Pro",   400,  67,  32, 24.9, 0),
    ("Gemini 1.5 Flash", 445,  69,  39, 24.7, 0),
    ("Gemini 1.5 Pro",   440,  74,  42, 25.4, 0),
    ("Mistral Large",    460,  73,  41, 25.0, 0),
    ("Llama 3 8B",       475,  58,  28, 23.8, 1),
    ("Llama 3 70B",      480,  70,  38, 24.8, 1),
    ("GPT-4o mini",      515,  71,  44, 24.9, 0),
    ("GPT-4o",           520,  79,  50, 25.6, 0),
    ("Claude 3.5 Haiku", 555,  76,  49, 25.0, 0),
    ("Claude 3.5 Sonnet",560,  83,  58, 25.6, 0),
    ("Llama 3.1 8B",     535,  60,  30, 23.9, 1),
    ("Llama 3.1 405B",   540,  77,  48, 25.5, 1),
    ("Grok 2",           580,  80,  52, 25.5, 0),
    ("Gemini 2.0 Flash", 615,  81,  57, 25.3, 0),
    ("Gemini 2.0",       620,  85,  62, 25.8, 0),
    ("o1-mini",          590,  82,  50, 25.4, 0),
    ("o1",               600,  88,  55, 25.9, 0),
    ("Claude 3.7 Sonnet",680,  89,  68, 25.9, 0),
    ("DeepSeek V3",      650,  84,  60, 25.2, 1),
    ("Qwen 2.5 72B",     660,  82,  56, 25.0, 1),
    ("Llama 4 Scout",    695,  81,  55, 25.2, 1),
    ("Llama 4",          700,  86,  63, 25.7, 1),
    ("o3-mini",          730,  90,  65, 25.7, 0),
    ("o3",               760,  93,  72, 26.2, 0),
    ("Gemini 2.5 Flash", 775,  88,  69, 25.6, 0),
    ("Gemini 2.5 Pro",   780,  92,  74, 26.1, 0),
    ("Claude 4 Sonnet",  800,  94,  80, 26.2, 0),
    ("Claude 4 Opus",    810,  95,  82, 26.4, 0),
    ("Grok 3",           820,  93,  78, 26.0, 0),
    ("DeepSeek V4",      850,  91,  76, 25.6, 1),
    ("GPT-5 mini",       880,  93,  81, 26.0, 0),
    ("GPT-5",            900,  97,  88, 26.6, 0),
    ("Claude 5",         950,  98,  92, 26.7, 0),
    ("Gemini 3",         920,  97,  90, 26.6, 0),
    ("Llama 5",          940,  95,  85, 26.3, 1),
]

df = pd.DataFrame(
    models,
    columns=[
        "model_name",
        "release_days_since_2023",
        "benchmark_score",
        "agentic_score",
        "compute_log10_flops",
        "open_weights",
    ],
)

# --- Pad with synthetic "checkpoint" rows -------------------------------
# Real capability evals produce a spread of scores even for the same model
# (different prompts, seeds, eval harnesses). To give the regression more
# rows to train on, we generate jittered copies of each named model above,
# tagged with a checkpoint suffix, rather than inventing more fake names.
N_JITTER_PER_MODEL = 4
jittered_rows = []
for _, row in df.iterrows():
    for i in range(N_JITTER_PER_MODEL):
        jittered_rows.append(
            {
                "model_name": f"{row['model_name']} (eval run {i + 1})",
                "release_days_since_2023": row["release_days_since_2023"]
                + rng.integers(-5, 6),
                "benchmark_score": np.clip(
                    row["benchmark_score"] + rng.normal(0, 2.5), 0, 100
                ),
                "agentic_score": np.clip(
                    row["agentic_score"] + rng.normal(0, 3), 0, 100
                ),
                "compute_log10_flops": row["compute_log10_flops"],
                "open_weights": row["open_weights"],
            }
        )

df = pd.concat([df, pd.DataFrame(jittered_rows)], ignore_index=True)

# --- Normalize inputs to 0-1 for the fake risk formula ------------------
def norm(col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

cap = norm("benchmark_score")        # raw capability
agent = norm("agentic_score")        # autonomy / tool-use
compute = norm("compute_log10_flops")
recency = norm("release_days_since_2023")
open_w = df["open_weights"].astype(float)

# --- Made-up weight vectors per scenario (this is the "ground truth" ---
# the regression is supposed to recover, approximately, from noisy data).
# columns: [capability, agentic, compute, recency, open_weights, bias]
weights = {
    "bio_weapon_uplift":            [0.35, 0.10, 0.15, 0.05, 0.30, 5],
    "nuclear_miscalc_escalation":   [0.15, 0.35, 0.10, 0.10, 0.05, 8],
    "grid_financial_cascade":       [0.10, 0.30, 0.05, 0.10, 0.20, 6],
    "climate_tipping_cascade":      [0.05, 0.05, 0.10, 0.05, 0.05, 15],  # mostly independent of AI
    "autonomous_weapons_spiral":    [0.20, 0.40, 0.10, 0.15, 0.10, 4],
    "epistemic_collapse":           [0.30, 0.20, 0.05, 0.25, 0.15, 3],
}

for scenario, (w_cap, w_agent, w_compute, w_recency, w_open, bias) in weights.items():
    signal = (
        w_cap * cap
        + w_agent * agent
        + w_compute * compute
        + w_recency * recency
        + w_open * open_w
    ) * 100
    noise = rng.normal(0, 6, size=len(df))
    score = np.clip(signal + bias + noise, 0, 100)
    df[scenario] = score.round(1)

out_path = "apocalypse_risk_fake_data.csv"
df.to_csv(out_path, index=False)
print(f"wrote {len(df)} rows to {out_path}")
print(df.head())
