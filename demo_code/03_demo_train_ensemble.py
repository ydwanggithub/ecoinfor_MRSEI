"""Train a gradient boosting regressor on synthetic predictors of M-RSEI.

This demo shows the bare structure of an attribution pipeline: load predictors,
split, fit a gradient boosting model, score out of sample, and inspect the
impurity-based importances. The manuscript compares four gradient boosting
regressors; here a single sklearn model stands in for the idea.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_mrsei_demo.csv"
PREDICTORS = ["forest_cover", "soil_organic_carbon", "slope",
              "precipitation", "elevation", "mining_distance"]
TARGET = "mrsei_endpoint"
SEED = 42


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    X = df[PREDICTORS].to_numpy()
    y = df[TARGET].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=SEED)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    print(f"Trained on {len(X_tr)} cells, evaluated on {len(X_te)} cells")
    print(f"R^2: {r2_score(y_te, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_te, y_pred):.3f}")
    print()
    print("Impurity-based importances:")
    for idx in np.argsort(model.feature_importances_)[::-1]:
        print(f"  {PREDICTORS[idx]:<22s} {model.feature_importances_[idx]:.3f}")


if __name__ == "__main__":
    main()
