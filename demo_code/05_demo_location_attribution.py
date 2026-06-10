"""Illustrate the idea of treating geographic location as a model player.

The manuscript uses GeoShapley (Li 2024), which treats the coordinate pair as a
single coalition player and separates a location main effect from location by
predictor interactions. This demo is a much simpler teaching version: it adds
the x and y coordinates as two extra features, computes ordinary SHAP, then sums
the two coordinate contributions into a single "location" share for comparison
with the environmental predictors. The numbers below come from synthetic data
and are only meant to show the workflow, not any manuscript result.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_mrsei_demo.csv"
OUT_PNG = Path(__file__).resolve().parent / "location_attribution_demo.png"
PREDICTORS = ["forest_cover", "soil_organic_carbon", "slope",
              "precipitation", "elevation", "mining_distance"]
TARGET = "mrsei_endpoint"
SEED = 42


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    features = PREDICTORS + ["x", "y"]
    X = df[features]
    y = df[TARGET]

    model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=SEED)
    model.fit(X, y)
    mean_abs = np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)

    contrib = dict(zip(features, mean_abs))
    location = contrib.pop("x") + contrib.pop("y")
    contrib["location"] = location
    total = sum(contrib.values())

    labels = sorted(contrib, key=contrib.get)
    values = [contrib[k] for k in labels]
    colors = ["#C44E52" if k == "location" else "#4A6FA5" for k in labels]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Location as a player vs predictors (synthetic demo)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PNG}")
    print(f"Location share of total attribution: {100 * location / total:.1f}% (synthetic)")


if __name__ == "__main__":
    main()
