"""
Fit a simple regression model per scenario on the fake dataset, just to
demonstrate a working train/evaluate loop. Toy data, toy model.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("apocalypse_risk_fake_data.csv")

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

X = df[FEATURES]

print(f"rows: {len(df)}")
print(f"feature columns (X): {FEATURES}")
print(f"target columns (y), one model fit per scenario: {SCENARIOS}\n")

print(f"{'scenario':<28} {'R2 (test)':>10} {'MAE (test)':>11}")
print("-" * 52)

results = []
for scenario in SCENARIOS:
    y = df[scenario]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{scenario:<28} {r2:>10.2f} {mae:>11.2f}")
    results.append(
        {"scenario": scenario, "r2": r2, "mae": mae, **dict(zip(FEATURES, model.coef_))}
    )

pd.DataFrame(results).to_csv("regression_results.csv", index=False)
print("\nwrote regression_results.csv")
