"""Scatter the synthetic cells over a square extent, coloured by M-RSEI.

A minimal stand-in for the spatial maps in the manuscript: it shows how the
tabular cells carry coordinates and an endpoint value that can be drawn as a
simple map. The manuscript maps use the real grid and cartographic styling.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_mrsei_demo.csv"
OUT_PNG = Path(__file__).resolve().parent / "map_demo.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    sc = ax.scatter(df["x"], df["y"], c=df["mrsei_endpoint"],
                    cmap="YlGn", s=24, edgecolor="0.3", linewidth=0.2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("M-RSEI (synthetic)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Synthetic cell map")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
