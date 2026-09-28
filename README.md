# ecoinfor_MRSEI

Companion demonstration code and synthetic sample data for the M-RSEI and
GeoShapley location attribution workflow, released alongside the manuscript

> Satellite assessment of ecological condition and associated recovery
> constraints in China's main ion adsorption rare earth mining landscape
> from 2000 to 2025 (under review)

The workflow reads a mining-adapted remote sensing ecological index (M-RSEI),
fits gradient boosting models, and reads the model with GeoShapley so that
geographic location enters the attribution as a coalition player alongside
the environmental predictors.

## What this repository contains

This repository provides a small, self-contained reference implementation so
that other groups can reproduce the index construction, the gradient boosting
attribution, the location-as-player idea, and the basic mapping workflow on
their own data. It does not contain the full source datasets, the GEE-side
preprocessing pipeline, or the manuscript text.

```
demo_data/
  sample_mrsei_demo.csv        synthetic 500-cell tabular sample (random seed fixed)

demo_code/
  01_demo_load_data.py         load the sample CSV and print a head summary
  02_demo_compute_mrsei.py     build an RSEI-style index from four components by PCA
  03_demo_train_ensemble.py    train a gradient boosting regressor on synthetic predictors
  04_demo_shap_summary.py      produce a global SHAP bar plot
  05_demo_location_attribution.py  add coordinates as a player and compare its SHAP share
  06_demo_visualize_map.py     scatter the synthetic cells over a square extent

figures/
  Fig01 ... Fig13 PNG
  Rendered PNG copies of the figures in the revised manuscript, provided for
  quick visual reference. See the manuscript captions for full panel
  descriptions.
```

## Quick start

```bash
conda env create -f environment.yml
conda activate ecoinf-demo
python demo_code/_generate_sample_csv.py   # writes the synthetic CSV (already bundled)
python demo_code/01_demo_load_data.py
python demo_code/02_demo_compute_mrsei.py
python demo_code/03_demo_train_ensemble.py
python demo_code/04_demo_shap_summary.py
python demo_code/05_demo_location_attribution.py
python demo_code/06_demo_visualize_map.py
```

Each script is independent and runs on the bundled synthetic CSV in a few
seconds on a laptop. No GPU or remote service is required.

## On the location attribution

The bundled `05_demo_location_attribution.py` is a simplified teaching version
that adds the coordinate pair as two ordinary features and sums their SHAP
contributions. The manuscript instead uses GeoShapley (Li 2024, Annals of the
American Association of Geographers), which treats the coordinate pair as a
single coalition player and separates a location main effect from location by
predictor interactions. The synthetic numbers printed by the demo are
illustrative and are not the manuscript results.

## Source datasets used in the manuscript

The manuscript itself draws on publicly available remote-sensing products:

- Landsat Collection 2 surface reflectance and surface temperature (USGS, via
  Google Earth Engine)
- TerraClimate monthly climate and water balance (Abatzoglou et al. 2018)
- SoilGrids 2.0 soil properties (Poggio et al. 2021)
- SRTM 30 m elevation
- MODIS MCD12Q1 land cover and NPP-VIIRS-like nighttime lights
- Registry of rare earth mineral sites (Ganzhou Natural Resources Bureau,
  available under the original access conditions of the data provider)

This repository does not redistribute these sources; consult each provider for
licence terms and access.

## Licence

Released under the MIT Licence (see LICENSE).

## Citation

If you use this code in your own work, please cite the manuscript once
published.
