# Ames Housing Price Predictor

Streamlit app for the final model of the Ames Housing project: a tuned SVR
(RBF kernel, C=10, gamma=0.001) inside a scikit-learn pipeline that does all the
preprocessing (imputation, log1p on skewed features, ordinal / one-hot encoding,
StandardScaler).

**Held-out test set:** R2 = 0.902, MAE about $15,800, RMSE about $27,600.

## Files

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `train_model.py` | Trains the model (same code as the notebook) and writes the model files |
| `feature_columns.json` | Column order the model expects |
| `defaults.json` | Training-set defaults for inputs the app does not ask for |
| `options.json` | Allowed values for the categorical inputs |
| `metrics.json` | Train / validation / test metrics |
| `requirements.txt` | Pinned dependencies |

The trained model (`svr_pipeline.joblib`) is intentionally not committed.
On the first run `app.py` calls `train_model.py`, which downloads the dataset
(the same public file the notebook uses) and trains the model in a few seconds.
Everything is deterministic (`random_state=42`), so the model you get is
identical every time.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Split is 60/20/20 with `random_state=42` (876 / 292 / 292 rows), exactly as in the notebook.


## Pages

- **Price prediction**: the SVR predictor.
- **Data insights**: the notebook's EDA and Q1-Q12 charts, rebuilt with Plotly (`insights.py`).
- English / Arabic switch in the sidebar (Arabic uses right-to-left layout).
