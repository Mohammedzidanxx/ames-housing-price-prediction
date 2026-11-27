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

import insights

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


# ------------------------------------------------------------------ language
lang_choice = st.sidebar.radio("Language / اللغة", ["English", "العربية"], horizontal=True, key="lang")
AR = lang_choice == "العربية"


def T(en, ar):
    return ar if AR else en


CSS = """
<style>
h1, h2, h3 { color: #1E3A5F; }
[data-testid="stMetricValue"] { color: #0F766E; }
[data-testid="stSidebar"] { border-inline-end: 1px solid #D5E3E0; }
.stTabs [aria-selected="true"] { color: #0F766E; }
.js-plotly-plot, .plotly, [data-testid="stSlider"], [data-testid="stNumberInput"] input { direction: ltr; }
</style>
"""
RTL = """
<style>
.stApp, [data-testid="stSidebar"], .stMarkdown, [data-testid="stMetric"], [data-testid="stCaptionContainer"],
[data-testid="stAlert"], [data-testid="stExpander"], label, .stTabs { direction: rtl; text-align: right; }
[data-testid="stDataFrame"] { direction: ltr; }
</style>
"""
st.markdown(CSS + (RTL if AR else ""), unsafe_allow_html=True)

QUAL = {"None": T("None", "لا يوجد"), "Po": T("Poor", "ضعيف"), "Fa": T("Fair", "مقبول"), "TA": T("Typical", "عادي"),
        "Gd": T("Good", "جيد"), "Ex": T("Excellent", "ممتاز")}
YN = {"Y": T("Yes", "نعم"), "N": T("No", "لا")}
q = lambda v: QUAL.get(v, v)  # noqa: E731

st.title(T("🏠 Ames Housing Price Predictor", "🏠 توقع أسعار بيوت Ames"))
st.caption(T(
    "Final model from the project: tuned SVR (RBF kernel, C=10, gamma=0.001), "
    f"trained on {metrics['split_sizes']['train']} houses. "
    f"Test R² = {test_metrics['R2']:.3f}, MAE ≈ ${test_metrics['MAE_usd']:,.0f}.",
    "الموديل النهائي للمشروع: SVR بعد الضبط (kernel = RBF, C=10, gamma=0.001)، "
    f"متدرب على {metrics['split_sizes']['train']} بيت. "
    f"Test R² = {test_metrics['R2']:.3f}، متوسط الخطأ المطلق ≈ ${test_metrics['MAE_usd']:,.0f}."))

tab_predict, tab_insights = st.tabs([T("🔮 Price prediction", "🔮 توقع السعر"), T("📊 Data insights", "📊 تحليل البيانات")])

with tab_insights:
    insights.render(AR)

with tab_predict:
    left, middle, right = st.columns(3)

    with left:
        st.subheader(T("Size", "المساحة"))
        gr_liv_area = st.number_input(T("Above-ground living area (GrLivArea, sq ft)", "المساحة المعيشية فوق الأرض (GrLivArea، قدم مربع)"), 300, 6000, 1500, 50)
        total_bsmt = st.number_input(T("Total basement area (TotalBsmtSF, sq ft)", "مساحة البدروم (TotalBsmtSF، قدم مربع)"), 0, 4000, 1000, 50)
        first_floor = st.number_input(T("First floor area (1stFlrSF, sq ft)", "مساحة الدور الأول (1stFlrSF، قدم مربع)"), 300, 4000, 1100, 50)
        second_floor = st.number_input(T("Second floor area (2ndFlrSF, sq ft)", "مساحة الدور التاني (2ndFlrSF، قدم مربع)"), 0, 2500, 0, 50)
        lot_area = st.number_input(T("Lot area (LotArea, sq ft)", "مساحة الأرض (LotArea، قدم مربع)"), 1300, 215000, 9500, 500)
        bedrooms = st.number_input(T("Bedrooms above ground", "غرف النوم فوق الأرض"), 0, 8, 3)
        bathrooms = st.number_input(T("Full bathrooms above ground", "الحمامات الكاملة فوق الأرض"), 0, 4, 2)
        rooms = st.number_input(T("Total rooms above ground", "إجمالي الغرف فوق الأرض"), 2, 14, 7)

    with middle:
        st.subheader(T("Quality", "الجودة"))
        overall_qual = st.slider(T("Overall quality (1-10)", "الجودة الكلية (1-10)"), 1, 10, 6)
        overall_cond = st.slider(T("Overall condition (1-10)", "الحالة الكلية (1-10)"), 1, 10, 5)
        exter_qual = st.selectbox(T("Exterior quality", "جودة الواجهة"), ["Fa", "TA", "Gd", "Ex"], index=1, format_func=q)
        kitchen_qual = st.selectbox(T("Kitchen quality", "جودة المطبخ"), ["Fa", "TA", "Gd", "Ex"], index=1, format_func=q)
        bsmt_qual = st.selectbox(T("Basement quality", "جودة البدروم"), ["None", "Fa", "TA", "Gd", "Ex"], index=2, format_func=q)
        heating_qc = st.selectbox(T("Heating quality", "جودة التدفئة"), ["Po", "Fa", "TA", "Gd", "Ex"], index=2, format_func=q)
        central_air = st.selectbox(T("Central air", "تكييف مركزي"), ["Y", "N"], index=0, format_func=YN.get)
        fireplaces = st.number_input(T("Fireplaces", "عدد المدافئ"), 0, 4, 0)

    with right:
        st.subheader(T("Garage, age and location", "الجراج والعمر والموقع"))
        garage_cars = st.number_input(T("Garage capacity (cars)", "سعة الجراج (عربيات)"), 0, 4, 2)
        garage_area = st.number_input(T("Garage area (sq ft)", "مساحة الجراج (قدم مربع)"), 0, 1500, 480, 20)
        year_built = st.number_input(T("Year built", "سنة البناء"), 1872, 2010, 1973, 1)
        year_remod = st.number_input(T("Year remodelled", "سنة التجديد"), 1950, 2010, 1990, 1)
        yr_sold = st.number_input(T("Year sold", "سنة البيع"), 2006, 2010, 2010, 1)
        neighborhood = st.selectbox(
            T("Neighborhood", "الحي"),
            options["Neighborhood"],
            index=options["Neighborhood"].index("NAmes") if "NAmes" in options["Neighborhood"] else 0,
        )
        mszoning = st.selectbox(
            T("Zoning", "تقسيم المنطقة (Zoning)"), options["MSZoning"],
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

with tab_predict:
    st.divider()
    result, extra = st.columns([1, 2])
    with result:
        st.metric(T("Estimated sale price", "السعر المتوقع للبيع"), f"${price:,.0f}")
        st.caption(T(
            f"Typical error on unseen houses: about ±\\${test_metrics['RMSE_usd']:,.0f} (RMSE), "
            f"mean absolute error \\${test_metrics['MAE_usd']:,.0f}.",
            f"الخطأ المعتاد على بيوت ماشافهاش الموديل: حوالي ±\\${test_metrics['RMSE_usd']:,.0f} (RMSE)، "
            f"ومتوسط الخطأ المطلق \\${test_metrics['MAE_usd']:,.0f}."))
    with extra:
        st.write(T(
            "**How to read it.** The model was trained on 1,460 houses sold in Ames, Iowa (2006-2010), "
            "on the log of the price and with all preprocessing inside the pipeline. "
            "The price is a prediction, not an appraisal: the further the house is from the training data, "
            "the less reliable it is.",
            "**إزاي تقرا النتيجة.** الموديل اتدرب على 1,460 بيت اتباعوا في Ames بولاية أيوا (2006-2010)، "
            "على لوج السعر وكل المعالجة جوه الـ pipeline. "
            "السعر ده توقع مش تقييم رسمي: كل ما البيت يبعد عن بيانات التدريب، التوقع بيبقى أقل موثوقية."))
        with st.expander(T("Model card", "بطاقة الموديل")):
            st.write(T(f"Model: SVR, kernel={metrics['params']['kernel']}, C={metrics['params']['C']}, gamma={metrics['params']['gamma']}",
                       f"الموديل: SVR، kernel={metrics['params']['kernel']}، C={metrics['params']['C']}، gamma={metrics['params']['gamma']}"))
            st.write(T(f"Input columns: {metrics['n_features']} raw features -> {metrics['n_features_encoded']} encoded features",
                       f"أعمدة الإدخال: {metrics['n_features']} عمود خام ← {metrics['n_features_encoded']} عمود بعد الترميز"))
            mdf = pd.DataFrame(metrics["metrics"])
            if AR:
                mdf["split"] = mdf["split"].map({"Train": "تدريب", "Validation": "تحقق", "Val": "تحقق", "Test": "اختبار"}).fillna(mdf["split"])
                ren = {"split": "المجموعة", "R2": "R²", "MAE_log": "MAE (لوج)", "RMSE_log": "RMSE (لوج)", "MAE_usd": "MAE (دولار)", "RMSE_usd": "RMSE (دولار)"}
            else:
                ren = {"split": "Split", "R2": "R²", "MAE_log": "MAE (log)", "RMSE_log": "RMSE (log)", "MAE_usd": "MAE (USD)", "RMSE_usd": "RMSE (USD)"}
            st.dataframe(mdf.rename(columns=ren), hide_index=True, use_container_width=True)
            st.caption(T(
                "Features you did not change keep the training-set median (numbers) or most frequent value "
                "(categories), so the prediction reflects the inputs shown above.",
                "الأعمدة اللي ماغيرتهاش بتاخد وسيط بيانات التدريب (للأرقام) أو القيمة الأكثر تكرارًا (للفئات)، "
                "فالتوقع بيعكس المدخلات الظاهرة فوق."))
