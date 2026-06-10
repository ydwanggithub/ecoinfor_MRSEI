"""Load the synthetic sample CSV and print a short summary.

The bundled CSV holds 500 synthetic cells with location coordinates, four
RSEI-style spectral components, a few generic environmental predictors, and a
synthetic M-RSEI endpoint. The values are illustrative only.
"""
from pathlib import Path
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_mrsei_demo.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} cells, {df.shape[1]} columns")
    print()
    print("First rows:")
    print(df.head())
    print()
    print("Endpoint summary (mrsei_endpoint):")
    print(df["mrsei_endpoint"].describe().round(4))


if __name__ == "__main__":
    main()
