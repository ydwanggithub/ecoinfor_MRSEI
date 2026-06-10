"""Build an RSEI-style index from four synthetic spectral components.

This illustrates the standard remote-sensing ecological index idea: combine
greenness, wetness, heat and dryness through the first principal component,
oriented so that higher values mean better surface ecological quality, then
rescale to [0, 1]. The manuscript M-RSEI tunes each proxy to the mining
landscape; here the components are synthetic and the construction is generic.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_mrsei_demo.csv"
COMPONENTS = ["greenness", "wetness", "heat", "dryness"]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    z = StandardScaler().fit_transform(df[COMPONENTS])

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(z)[:, 0]

    # Orient so greenness loads positive; flip the sign if needed.
    loadings = pca.components_[0]
    if loadings[COMPONENTS.index("greenness")] < 0:
        pc1, loadings = -pc1, -loadings

    index = (pc1 - pc1.min()) / (pc1.max() - pc1.min())

    print(f"PC1 explains {pca.explained_variance_ratio_[0] * 100:.1f}% of component variance")
    print("Signed loadings (greenness/wetness positive, heat/dryness negative):")
    for name, w in zip(COMPONENTS, loadings):
        print(f"  {name:<12s} {w:+.3f}")
    print()
    print(f"RSEI-style index range: [{index.min():.3f}, {index.max():.3f}], mean {index.mean():.3f}")


if __name__ == "__main__":
    main()
