# GPPyramid Transformer

This repository contains the reference implementation of GPPyramid Transformer, a scale-aware temporal model for estimating daily gross primary productivity (GPP).

The implementation focuses on the main components described in the accompanying manuscript, particularly pyramidal cross-scale attention and Feature-wise Scale Adaptive Fusion (FSAF). 

## Repository structure

```text
GPPyramid_Transformer/
├── gppyramid/
│   ├── __init__.py
│   ├── model.py      # Pyramidal cross-scale attention and FSAF model
│   ├── data.py       # Site-year data interface and normalization utilities
│   ├── train.py      # Training, prediction, and cross-validation utilities
│   ├── metrics.py    # R2, RMSE, MAE, and KGE
│   └── utils.py      # General helper functions
└── README.md
```

## Core modules

- `model.py` contains the main GPPyramid Transformer architecture, including sinusoidal temporal positional encoding, pyramidal cross-scale attention, Feature-wise Scale Adaptive Fusion, and the daily GPP prediction head.
- `data.py` defines a compact site-year data interface. Users can adapt it to their own flux tower and remote-sensing datasets.
- `train.py` provides reference training and prediction utilities, including masked Huber loss and site-level cross-validation.
- `metrics.py` includes the four primary evaluation metrics used for model comparison: R2, RMSE, MAE, and KGE.

## Expected data format

The data interface assumes one CSV file per flux tower site. Each file should contain daily records with at least the following columns:

```text
TIMESTAMP, GPP, NDVI, NIRv, TMP, SWR, VPD, SM, PRE, LAI4
```

where `TIMESTAMP` is formatted as `YYYYMMDD`. The site metadata file should contain at least:

```text
SITE, PFT
```

The default plant functional types are:

```text
CRO, DBF, EBF, ENF, GRA, MF, SAV, SHR, WET
```

Users may modify the variable names and PFT list in their own scripts.

## Notes on data availability

Raw flux tower observations and satellite-derived products are not redistributed in this repository. Users should obtain FLUXNET, AmeriFlux, ICOS, ChinaFLUX, MODIS, ERA5-Land, and other required datasets from their official data portals and process them according to their own data-use agreements.

## Dependencies

The core code requires:

```text
numpy
pandas
torch
scikit-learn
PyYAML
```

`scikit-learn` is not required by the minimal metric implementation, but it may be useful for extending evaluation utilities.

## License

This code is released for academic research and method reference. Please check the repository license file for detailed terms.
