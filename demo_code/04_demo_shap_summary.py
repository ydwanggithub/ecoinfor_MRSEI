"""Compute SHAP values and draw a global summary bar plot.

The plot is saved to demo_code/shap_summary_demo.png. The example shows how a
per-cell attribution layer is built once a gradient boosting model is fitted;
real applications use the full predictor set from the manuscript Methods.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_mrsei_demo.csv"
OUT_PNG = Path(__file__).resolve().parent / "shap_summary_demo.png"
PREDICTORS = ["forest_cover", "soil_organic_carbon", "slope",
              "precipitation", "elevation", "mining_distance"]
TARGET = "mrsei_endpoint"
SEED = 42


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    X = df[PREDICTORS]
    y = df[TARGET]

    model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=SEED)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    sorted_features = [PREDICTORS[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(sorted_features, mean_abs[order], color="#4A6FA5")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP importance (synthetic demo)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PNG}")
    print("Top-3 predictors:", sorted_features[-1], sorted_features[-2], sorted_features[-3])


if __name__ == "__main__":
    main()
