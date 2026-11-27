"""
Ames Housing price predictor
----------------------------
Streamlit front end for the tuned SVR from the project notebook.

The saved pipeline (model/svr_pipeline.joblib) already contains the whole
preprocessing chain: imputers, log1p transforms, ordinal / one-hot encoders and
the StandardScaler. So the app only has to build one raw row and call predict().
"""
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = pathlib.Path(__file__).parent
MODEL_DIR = HERE

st.set_page_config(page_title="Ames Housing Price Predictor", page_icon="🏠", layout="wide")


if not (MODEL_DIR / "svr_pipeline.joblib").exists():
    # Model not bundled (e.g. deployed from the repo): train it exactly as the
    # notebook does. train_model.py downloads the dataset if needed and writes
    # all files into this directory. Deterministic (random_state=42).
    import subprocess, sys
    subprocess.run([sys.executable, str(HERE / "train_model.py")], check=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "svr_pipeline.joblib")
    features = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    defaults = json.loads((MODEL_DIR / "defaults.json").read_text())
    options = json.loads((MODEL_DIR / "options.json").read_text())
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return model, features, defaults, options, metrics


model, feature_columns, defaults, options, metrics = load_artifacts()
test_metrics = next(m for m in metrics["metrics"] if m["split"] == "Test")


def blank_house():
    """Start from the training-set defaults, then override with what the user types."""
    row = {}
    for col, value in defaults.items():
        row[col] = value if not isinstance(value, float) or not np.isnan(value) else 0.0
    return row


st.title("🏠 Ames Housing Price Predictor")
st.caption(
    "Final model from the project: tuned SVR (RBF kernel, C=10, gamma=0.001), "
    f"trained on {metrics['split_sizes']['train']} houses. "
    f"Test R² = {test_metrics['R2']:.3f}, MAE ≈ ${test_metrics['MAE_usd']:,.0f}."
)

# ------------------------------------------------------------------ inputs
left, middle, right = st.columns(3)

with left:
    st.subheader("Size")
    gr_liv_area = st.number_input("Above-ground living area (GrLivArea, sq ft)", 300, 6000, 1500, 50)
    total_bsmt = st.number_input("Total basement area (TotalBsmtSF, sq ft)", 0, 4000, 1000, 50)
    first_floor = st.number_input("First floor area (1stFlrSF, sq ft)", 300, 4000, 1100, 50)
    second_floor = st.number_input("Second floor area (2ndFlrSF, sq ft)", 0, 2500, 0, 50)
    lot_area = st.number_input("Lot area (LotArea, sq ft)", 1300, 215000, 9500, 500)
    bedrooms = st.number_input("Bedrooms above ground", 0, 8, 3)
    bathrooms = st.number_input("Full bathrooms above ground", 0, 4, 2)
    rooms = st.number_input("Total rooms above ground", 2, 14, 7)

with middle:
    st.subheader("Quality")
    overall_qual = st.slider("Overall quality (1-10)", 1, 10, 6)
    overall_cond = st.slider("Overall condition (1-10)", 1, 10, 5)
    exter_qual = st.selectbox("Exterior quality", ["Fa", "TA", "Gd", "Ex"], index=1)
    kitchen_qual = st.selectbox("Kitchen quality", ["Fa", "TA", "Gd", "Ex"], index=1)
    bsmt_qual = st.selectbox("Basement quality", ["None", "Fa", "TA", "Gd", "Ex"], index=2)
    heating_qc = st.selectbox("Heating quality", ["Po", "Fa", "TA", "Gd", "Ex"], index=2)
    central_air = st.selectbox("Central air", ["Y", "N"], index=0)
    fireplaces = st.number_input("Fireplaces", 0, 4, 0)

with right:
    st.subheader("Garage, age and location")
    garage_cars = st.number_input("Garage capacity (cars)", 0, 4, 2)
    garage_area = st.number_input("Garage area (sq ft)", 0, 1500, 480, 20)
    year_built = st.number_input("Year built", 1872, 2010, 1973, 1)
    year_remod = st.number_input("Year remodelled", 1950, 2010, 1990, 1)
    yr_sold = st.number_input("Year sold", 2006, 2010, 2010, 1)
    neighborhood = st.selectbox(
        "Neighborhood",
        options["Neighborhood"],
        index=options["Neighborhood"].index("NAmes") if "NAmes" in options["Neighborhood"] else 0,
    )
    mszoning = st.selectbox(
        "Zoning", options["MSZoning"],
        index=options["MSZoning"].index("RL") if "RL" in options["MSZoning"] else 0,
    )

# ------------------------------------------------------------------ build the row
house = blank_house()

house.update({
    "GrLivArea": float(gr_liv_area),
    "TotalBsmtSF": float(total_bsmt),
    "1stFlrSF": float(first_floor),
    "2ndFlrSF": float(second_floor),
    "LotArea": float(lot_area),
    "BedroomAbvGr": float(bedrooms),
    "FullBath": float(bathrooms),
    "TotRmsAbvGrd": float(rooms),
    "OverallQual": float(overall_qual),
    "OverallCond": float(overall_cond),
    "ExterQual": exter_qual,
    "KitchenQual": kitchen_qual,
    "BsmtQual": bsmt_qual,
    "HeatingQC": heating_qc,
    "CentralAir": central_air,
    "Fireplaces": float(fireplaces),
    "GarageCars": float(garage_cars),
    "GarageArea": float(garage_area),
    "YrSold": float(yr_sold),
    "Neighborhood": neighborhood,
    "MSZoning": mszoning,
})

# Engineered feature from the notebook: house age at the time of sale
house["HouseAge"] = float(yr_sold - year_built)
# Notebook rule: a missing garage year is filled with the construction year
house["GarageYrBlt"] = float(year_built)
house["YearRemodAdd"] = float(max(year_remod, year_built))

# No garage -> the garage columns describe nothing
if garage_cars == 0:
    house["GarageArea"] = 0.0
    for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        house[col] = "None"

# No basement -> the basement columns describe nothing
if total_bsmt == 0:
    for col in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
        house[col] = "None"

row = pd.DataFrame([house])[feature_columns]

# ------------------------------------------------------------------ prediction
log_price = float(model.predict(row)[0])
price = float(np.expm1(log_price))

st.divider()
result, extra = st.columns([1, 2])
with result:
    st.metric("Estimated sale price", f"${price:,.0f}")
    st.caption(
        f"Typical error on unseen houses: about ±\\${test_metrics['RMSE_usd']:,.0f} (RMSE), "
        f"mean absolute error \\${test_metrics['MAE_usd']:,.0f}."
    )
with extra:
    st.write(
        "**How to read it.** The model was trained on 1,460 houses sold in Ames, Iowa (2006-2010), "
        "on the log of the price and with all preprocessing inside the pipeline. "
        "The price is a prediction, not an appraisal: the further the house is from the training data, "
        "the less reliable it is."
    )
    with st.expander("Model card"):
        st.write(f"Model: SVR, kernel={metrics['params']['kernel']}, C={metrics['params']['C']}, gamma={metrics['params']['gamma']}")
        st.write(f"Input columns: {metrics['n_features']} raw features -> {metrics['n_features_encoded']} encoded features")
        st.dataframe(
            pd.DataFrame(metrics["metrics"]).rename(columns={
                "split": "Split", "R2": "R²", "MAE_log": "MAE (log)",
                "RMSE_log": "RMSE (log)", "MAE_usd": "MAE (USD)", "RMSE_usd": "RMSE (USD)",
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Features you did not change keep the training-set median (numbers) or most frequent value "
            "(categories), so the prediction reflects the inputs shown above."
        )
