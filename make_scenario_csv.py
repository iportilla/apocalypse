"""
Build a single-scenario training CSV for the Azure ML exercise.

Usage:
    python3 make_scenario_csv.py bio_weapon_uplift

Writes scenario_data/<scenario>.csv containing the 5 feature columns plus the
chosen scenario as the target. model_name and the other five scenario columns
are dropped so AutoML cannot leak them into the prediction.
"""

import os
import sys

import pandas as pd

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

if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
    sys.exit("usage: python3 make_scenario_csv.py <scenario>\n  scenarios: " + ", ".join(SCENARIOS))

scenario = sys.argv[1]
df = pd.read_csv("apocalypse_risk_fake_data.csv")

os.makedirs("scenario_data", exist_ok=True)
out = f"scenario_data/{scenario}.csv"
df[FEATURES + [scenario]].to_csv(out, index=False)

print(f"wrote {out}: {len(df)} rows, target = {scenario}")
print(f"columns: {FEATURES + [scenario]}")
