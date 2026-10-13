# Preprocessing

This CPU-only pipeline converts the two immutable raw sources into the external
WVS reference geometry and the benchmark country universe. Run it before
benchmark construction.

## Inputs

- `data/raw/wvs/cross_national_wave_7.csv`
- `data/raw/atlas/atlas_feb_2026.csv`

Do not edit these files. Their raw contents are the provenance anchor for all
downstream artifacts.

## Run order

From the repository root, after installing the project:

```bash
PYTHONPATH=src python -m preprocessing.build_wvs_backbone
PYTHONPATH=src python -m preprocessing.build_countries_config
```

Or run both stages in sequence:

```bash
PYTHONPATH=src bash scripts/preprocessing/build_references.sh
```

Generated files are protected by default. To intentionally rebuild them:

```bash
PYTHONPATH=src OVERWRITE=1 bash scripts/preprocessing/build_references.sh
```

Both stages are deterministic and should be run once per raw-source release.
They require CPU and memory only; no model inference or GPU is involved.

## Stage 1: WVS reference geometry

`preprocessing.build_wvs_backbone` performs the following operations:

1. Validates WVS ISO3 country codes and required columns.
2. Selects 33 questionnaire items and the two Inglehart–Welzel axes.
3. Converts negative WVS questionnaire missing-value codes to missing values.
4. Computes unweighted country means for all 35 dimensions.
5. Z-scores each dimension across countries using the population standard deviation.
6. Computes a cosine-distance matrix over the standardized country vectors.
7. Computes Ward linkage over those same standardized vectors.
8. Runs transparent expected-near and expected-far face-validity checks.

The unweighted aggregation is retained deliberately because it is the method
used to construct the released reference matrix. It should be described as an
external descriptive reference—not as a population-estimating survey analysis.

Machine-readable outputs:

```text
data/processed/references/wvs/
├── audit.json
├── country_vectors.parquet    # 66 countries × 35 standardized dimensions
├── distance_matrix.csv        # 66 × 66 labelled cosine distances
├── distance_matrix.npy        # same matrix as a NumPy array
├── iwm_coords.csv             # unstandardized country-level IWM coordinates
└── linkage.npy                # 65 × 4 Ward-linkage matrix
```

The preprocessing stage does not write reports or figures into `output/`.
Face-validity diagnostics and source provenance are retained in the
machine-readable `audit.json` alongside the reference matrices.

## Stage 2: Country universe

`preprocessing.build_countries_config` validates both sources and computes the
pure set intersection of WVS and Cultural Atlas ISO3 codes. It produces:

```text
configs/countries.yaml
data/processed/references/country_universe.audit.json
```

The released intersection contains 43 countries. No country is manually added
or removed at this stage. The later benchmark split assigns eight of these
countries to the unseen-country test subset; that split policy belongs in
`benchmark.split`, not preprocessing.

## Reproduction contract

- Raw inputs are immutable.
- Numeric WVS outputs must contain 66 countries and 35 dimensions.
- `configs/countries.yaml` must contain 43 sorted, unique ISO3 codes.
- Existing generated artifacts are never silently overwritten.
- Preprocessing writes only the documented machine-readable reference artifacts.
