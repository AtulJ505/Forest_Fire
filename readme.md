# Algerian Forest Fire Weather Index Predictor

[![Flask route tests](https://github.com/AtulJ505/Forest_Fire/actions/workflows/ci.yml/badge.svg)](https://github.com/AtulJ505/Forest_Fire/actions/workflows/ci.yml)

An educational regression project using the Algerian Forest Fires dataset. A Flask form applies a saved scaler and Ridge model to estimate the Fire Weather Index (FWI).

## Run locally

```bash
git clone https://github.com/AtulJ505/Forest_Fire.git
cd Forest_Fire
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python application.py
```

Open `http://127.0.0.1:5000`. On Windows, activate with `.venv\Scripts\activate`. The application loads `models/ridge.pkl` and `models/scaler.pkl` relative to its own directory. Only load trusted pickle artifacts; use scikit-learn versions compatible with the saved models or retrain them.

## Inputs and output

The form accepts Temperature, relative humidity (`RH`), wind speed (`Ws`), Rain, FFMC, DMC, ISI, Classes and Region in that order. Consult the notebook for the dataset's units and categorical encoding. Missing, malformed, NaN and infinite values return HTTP 400 before running a model. A successful request displays the predicted FWI.

## Tests

```bash
pip install Flask pytest
python -m pytest -q
```

The route tests use mocked model outputs. They verify form rendering, input validation, feature ordering, and artifact-path resolution without retraining or claiming a model-accuracy result. CI runs these tests on Python 3.11.

## Scope and limitations

This repository demonstrates data preprocessing, regression and a Flask prediction interface. Model quality must be evaluated on held-out data with documented metrics and dependency versions. It is not a validated fire-warning service or an operational safety tool. A live deployment URL is not currently documented.

## Contributing

Include a minimal reproduction for a bug and a regression test when changing application behavior. Keep dataset attribution, preprocessing details and evaluation methodology visible when updating the model.
