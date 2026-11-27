"""
Retrain the final model from the Ames notebook and save it for the Streamlit app.

Everything here mirrors the notebook exactly:
  - same engineered columns (HasGarage, HouseAge, HouseAgeGroup)
  - same drops (Id, HasGarage, HouseAgeGroup, YearBuilt)
  - same missing-value rules, log features, encoders and StandardScaler
  - same 60/20/20 split with random_state=42
  - same tuned SVR: RBF kernel, C=10, gamma=0.001

Run:  python train_model.py
"""
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR

HERE = pathlib.Path(__file__).parent
DATA = HERE / "train.csv"
DATA_URL = "https://raw.githubusercontent.com/vbabashov/house-prices/refs/heads/main/data/raw/train.csv"
MODEL_DIR = HERE

# The dataset is the same public file the notebook loads. Download it once if it
# is not bundled next to this script.
if not DATA.exists():
    import urllib.request
    print("downloading dataset:", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, DATA)

# ---------------------------------------------------------------- data + features
df = pd.read_csv(DATA)

# Engineered columns used in the notebook's analysis section
df["HasGarage"] = df["GarageCars"].apply(lambda x: "With Garage" if x > 0 else "Without Garage")
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["HouseAgeGroup"] = pd.cut(
    df["HouseAge"],
    bins=[-1, 10, 30, 50, 100, 200],
    labels=["0-10 years", "11-30 years", "31-50 years", "51-100 years", "Above 100 years"],
)

X = df.drop(columns=["SalePrice", "Id", "HasGarage", "HouseAgeGroup"])
y = df["SalePrice"]

# Missing values where NA means "the house does not have it"
none_columns = [
    "PoolQC", "MiscFeature", "Alley", "Fence",
    "MasVnrType", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtExposure", "BsmtFinType2",
    "BsmtQual", "BsmtCond", "BsmtFinType1",
]
X[none_columns] = X[none_columns].fillna("None")
X["MasVnrArea"] = X["MasVnrArea"].fillna(0)
X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
# LotFrontage and Electrical stay missing on purpose: the pipeline imputes them
# using the training data only, so there is no leakage.

X = X.drop(columns=["YearBuilt"])

y_log = np.log1p(y)

# ---------------------------------------------------------------- split (60/20/20)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_log, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)
print(f"train {X_train.shape} | val {X_val.shape} | test {X_test.shape}")

# ---------------------------------------------------------------- column groups
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols.remove("MSSubClass")
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
cat_cols.append("MSSubClass")

ordinal_cols = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual",
    "FireplaceQu", "GarageQual", "GarageCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "GarageFinish", "LandSlope", "LotShape",
    "Functional", "PoolQC", "PavedDrive",
]
binary_cols = ["CentralAir", "Street", "Utilities"]
nominal_cols = [c for c in cat_cols if c not in ordinal_cols + binary_cols]

ordinal_categories = [
    ["Po", "Fa", "TA", "Gd", "Ex"],
    ["Po", "Fa", "TA", "Gd", "Ex"],
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    ["Po", "Fa", "TA", "Gd", "Ex"],
    ["Fa", "TA", "Gd", "Ex"],
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    ["None", "No", "Mn", "Av", "Gd"],
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    ["None", "Unf", "RFn", "Fin"],
    ["Gtl", "Mod", "Sev"],
    ["IR3", "IR2", "IR1", "Reg"],
    ["Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    ["None", "Fa", "TA", "Gd", "Ex"],
    ["N", "P", "Y"],
]

log_features = [
    "LotArea", "LotFrontage", "MasVnrArea", "BsmtFinSF1",
    "1stFlrSF", "GrLivArea", "WoodDeckSF", "OpenPorchSF",
]
other_num_cols = [c for c in num_cols if c not in log_features]

# ---------------------------------------------------------------- pipelines
ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=ordinal_categories,
                               handle_unknown="use_encoded_value", unknown_value=-1)),
])
binary_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[["N", "Y"], ["Grvl", "Pave"], ["NoSeWa", "AllPub"]],
                               handle_unknown="use_encoded_value", unknown_value=-1)),
])
nominal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
log_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler()),
])
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(transformers=[
    ("log", log_pipeline, log_features),
    ("num", num_pipeline, other_num_cols),
    ("ordinal", ordinal_pipeline, ordinal_cols),
    ("binary", binary_pipeline, binary_cols),
    ("nominal", nominal_pipeline, nominal_cols),
])

# ---------------------------------------------------------------- final model
best_svr = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", SVR(kernel="rbf", C=10, gamma=0.001)),
]).fit(X_train, y_train)

print("X_train_processed:", best_svr.named_steps["preprocessor"].transform(X_train).shape)

rows = []
for name, Xp, yp in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    pred = best_svr.predict(Xp)
    rows.append({
        "split": name,
        "R2": round(r2_score(yp, pred), 4),
        "MAE_log": round(mean_absolute_error(yp, pred), 4),
        "RMSE_log": round(float(np.sqrt(mean_squared_error(yp, pred))), 4),
        "MAE_usd": round(float(mean_absolute_error(np.expm1(yp), np.expm1(pred))), 0),
        "RMSE_usd": round(float(np.sqrt(mean_squared_error(np.expm1(yp), np.expm1(pred)))), 0),
    })
metrics = pd.DataFrame(rows)
print(metrics.to_string(index=False))

# ---------------------------------------------------------------- save artefacts
joblib.dump(best_svr, MODEL_DIR / "svr_pipeline.joblib")

feature_columns = X_train.columns.tolist()
(MODEL_DIR / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))

# Defaults for every column, so the app can fill whatever the user does not type
defaults, options = {}, {}
for col in feature_columns:
    if col in cat_cols:
        mode = X_train[col].mode(dropna=True)
        defaults[col] = str(mode.iloc[0]) if len(mode) else "None"
        options[col] = sorted(X_train[col].dropna().astype(str).unique().tolist())
    else:
        defaults[col] = float(np.nan if X_train[col].isna().all() else X_train[col].median())

(MODEL_DIR / "defaults.json").write_text(json.dumps(defaults, indent=2))
(MODEL_DIR / "options.json").write_text(json.dumps(options, indent=2))
(MODEL_DIR / "metrics.json").write_text(json.dumps({
    "metrics": rows,
    "n_features": len(feature_columns),
    "n_features_encoded": int(best_svr.named_steps["preprocessor"].transform(X_train).shape[1]),
    "params": {"kernel": "rbf", "C": 10, "gamma": 0.001},
    "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
}, indent=2))

print("saved:", MODEL_DIR / "svr_pipeline.joblib", (MODEL_DIR / "svr_pipeline.joblib").stat().st_size, "bytes")
