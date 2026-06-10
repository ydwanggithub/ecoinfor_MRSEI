"""One-shot generator for demo_data/sample_mrsei_demo.csv (synthetic data).

This script is included so that anyone can regenerate the bundled CSV
locally with the same fixed random seed. Run once; the resulting CSV is
also committed to the repository.

The values are fully synthetic and do not correspond to any real location
or measurement. Field names match the columns expected by the demo scripts.
A smooth spatial trend is added to the target so that the location demo has
a location signal to recover; it is illustrative, not a real spatial field.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT_CSV = Path(__file__).resolve().parent.parent / "demo_data" / "sample_mrsei_demo.csv"
N = 500
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 100, N)
    y = rng.uniform(0, 100, N)

    # Four RSEI-style spectral components (greenness and wetness positive,
    # heat and dryness negative), used by the index-construction demo.
    greenness = rng.uniform(0.20, 0.90, N)
    wetness = rng.uniform(0.10, 0.60, N)
    heat = rng.uniform(0.20, 0.80, N)
    dryness = rng.uniform(0.10, 0.70, N)

    # Generic environmental predictors (synthetic ranges only).
    forest_cover = rng.uniform(0.0, 1.0, N)
    soil_organic_carbon = rng.uniform(5.0, 40.0, N)
    slope = rng.uniform(0.0, 35.0, N)
    precipitation = rng.uniform(1200.0, 1900.0, N)
    elevation = rng.uniform(100.0, 1200.0, N)
    mining_distance = rng.uniform(0.0, 8.0, N)

    # Smooth spatial trend = the location signal the location demo recovers.
    location_effect = 0.15 * np.sin(x / 18.0) + 0.12 * np.cos(y / 22.0)

    mrsei = (0.35
             + 0.18 * forest_cover
             + 0.004 * soil_organic_carbon
             - 0.004 * slope
             + 0.02 * mining_distance
             + location_effect
             + rng.normal(0.0, 0.05, N))
    mrsei = np.clip(mrsei, 0.0, 1.0)

    df = pd.DataFrame({
        "cell_id": np.arange(N),
        "x": x.round(2),
        "y": y.round(2),
        "greenness": greenness.round(4),
        "wetness": wetness.round(4),
        "heat": heat.round(4),
        "dryness": dryness.round(4),
        "forest_cover": forest_cover.round(4),
        "soil_organic_carbon": soil_organic_carbon.round(2),
        "slope": slope.round(2),
        "precipitation": precipitation.round(1),
        "elevation": elevation.round(1),
        "mining_distance": mining_distance.round(3),
        "mrsei_endpoint": mrsei.round(4),
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(df)} synthetic cells")


if __name__ == "__main__":
    main()
