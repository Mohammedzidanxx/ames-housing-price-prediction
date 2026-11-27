"""Data insights page: the notebook's EDA and Q1-Q12 charts, rebuilt with Plotly from data/train.csv."""
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ttest_ind

DATA = pathlib.Path(__file__).parent / "train.csv"
DATA_URL = "https://raw.githubusercontent.com/vbabashov/house-prices/refs/heads/main/data/raw/train.csv"
PALETTE = ["#0F766E", "#F59E0B", "#1E3A5F", "#E76F51", "#6BA292", "#8B5CF6", "#64748B", "#0EA5E9", "#B45309", "#14B8A6"]
INK = "#1F2937"
px.defaults.color_discrete_sequence = PALETTE
px.defaults.color_continuous_scale = ["#E6F4F1", "#0F766E", "#1E3A5F"]
TPL = "plotly_white"


@st.cache_data
def load_df():
    df = pd.read_csv(DATA if DATA.exists() else DATA_URL)
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    return df


def show(fig, height=420):
    fig.update_layout(template=TPL, height=height, margin=dict(l=10, r=10, t=50, b=10), colorway=PALETTE,
                      font=dict(color=INK, size=13), title_font=dict(size=16, color="#1E3A5F"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF")
    fig.update_xaxes(gridcolor="#E5EAF0")
    fig.update_yaxes(gridcolor="#E5EAF0")
    st.plotly_chart(fig, use_container_width=True)


def md(text):
    st.markdown(text.replace("$", "\\$"))


def info(text):
    st.info(text.replace("$", "\\$"))


def render(ar: bool):
    T = lambda en, a: a if ar else en  # noqa: E731
    df = load_df()
    price_lbl = T("Sale price ($)", "سعر البيع ($)")
    avg_lbl = T("Average sale price ($)", "متوسط سعر البيع ($)")
    count_lbl = T("Number of houses", "عدد البيوت")

    md(T(
        "The charts below are the ones from the project notebook (EDA and the 12 questions), "
        "rebuilt interactively from the same `train.csv` (1,460 houses). Hover over a chart to see exact values.",
        "الرسومات دي هي نفس رسومات النوتبوك (التحليل الاستكشافي والـ 12 سؤال)، "
        "معمولة تفاعلي من نفس ملف `train.csv` (1,460 بيت). حرّك الماوس على أي رسمة عشان تشوف القيم بالظبط."))

    names = ["EDA", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12",
             T("Outliers", "القيم الشاذة")]
    tabs = st.tabs(names)

    # ------------------------------------------------------------------ EDA
    with tabs[0]:
        st.subheader(T("Exploratory data analysis", "التحليل الاستكشافي للبيانات"))
        md(T(
            f"**Distributions:** SalePrice is right-skewed (median ${df.SalePrice.median():,.0f} is below the mean of "
            f"${df.SalePrice.mean():,.0f}), with a long tail of expensive houses.",
            f"**التوزيعات:** سعر البيع منحرف لليمين (الوسيط ${df.SalePrice.median():,.0f} أقل من المتوسط "
            f"${df.SalePrice.mean():,.0f}) وفيه ذيل طويل من البيوت الغالية."))
        c1, c2 = st.columns(2)
        with c1:
            show(px.histogram(df, x="SalePrice", nbins=30, marginal=None,
                              title=T("Distribution of SalePrice", "توزيع سعر البيع"),
                              labels={"SalePrice": price_lbl, "count": count_lbl}), 340)
        with c2:
            show(px.box(df, x="SalePrice", title=T("Boxplot of SalePrice", "بوكس بلوت لسعر البيع"),
                        labels={"SalePrice": price_lbl}), 340)

        md("---")
        feat = st.selectbox(T("Feature", "العمود"), ["GrLivArea", "OverallQual", "LotArea"], key="eda_feat")
        c1, c2, c3 = st.columns(3)
        with c1:
            show(px.histogram(df, x=feat, nbins=30, title=T(f"Distribution of {feat}", f"توزيع {feat}"),
                              labels={"count": count_lbl}), 320)
        with c2:
            show(px.box(df, x=feat, title=T(f"Boxplot of {feat}", f"بوكس بلوت لـ {feat}")), 320)
        with c3:
            show(px.scatter(df, x=feat, y="SalePrice", opacity=0.6,
                            title=T(f"{feat} vs SalePrice", f"{feat} مقابل السعر"),
                            labels={"SalePrice": price_lbl}), 320)

        md("---")
        # the notebook ran this before HouseAge was created, so leave it out here too
        num_orig = df.drop(columns=["HouseAge"]).select_dtypes(include=np.number)
        corr = num_orig.corr()["SalePrice"]
        top = corr.drop("SalePrice").abs().nlargest(10).index.tolist()
        cm = num_orig[top + ["SalePrice"]].corr()
        fig = px.imshow(cm.round(2), text_auto=True, color_continuous_scale=["#E76F51", "#FFFFFF", "#0F766E"], zmin=-1, zmax=1,
                        title=T("Correlation heatmap - top 10 features", "هيت ماب الارتباط - أقوى 10 أعمدة"))
        show(fig, 560)

        md("---")
        md(T(
            "**Categorical features:** median price differs clearly across neighborhoods and rises with each level of "
            "ExterQual and KitchenQual.",
            "**الأعمدة الفئوية:** الوسيط بيختلف بوضوح بين الأحياء وبيزيد مع كل مستوى من ExterQual وKitchenQual."))
        cat = st.selectbox(T("Categorical feature", "العمود الفئوي"),
                           ["Neighborhood", "HouseStyle", "BldgType", "KitchenQual", "ExterQual"], key="eda_cat")
        order = df.groupby(cat)["SalePrice"].median().sort_values(ascending=False).index
        show(px.box(df, x=cat, y="SalePrice", category_orders={cat: list(order)},
                    title=T(f"{cat} vs SalePrice", f"{cat} مقابل السعر"), labels={"SalePrice": price_lbl}), 440)
        summ = df.groupby(cat)["SalePrice"].agg(["count", "mean", "median", "min", "max"]).sort_values("median", ascending=False).round(0)
        summ.columns = [T("Count", "العدد"), T("Mean", "المتوسط"), T("Median", "الوسيط"), T("Min", "الأقل"), T("Max", "الأعلى")]
        st.dataframe(summ, use_container_width=True)

    # ------------------------------------------------------------------ Q1
    with tabs[1]:
        st.subheader(T("Q1: Average, median, minimum and maximum prices; cheapest and most expensive houses",
                       "س1: المتوسط والوسيط وأقل وأعلى سعر؛ أرخص وأغلى البيوت"))
        s = df.SalePrice
        m = st.columns(4)
        m[0].metric(T("Mean", "المتوسط"), f"${s.mean():,.0f}")
        m[1].metric(T("Median", "الوسيط"), f"${s.median():,.0f}")
        m[2].metric(T("Min", "الأقل"), f"${s.min():,.0f}")
        m[3].metric(T("Max", "الأعلى"), f"${s.max():,.0f}")
        top10 = df.nlargest(10, "SalePrice")[["Id", "Neighborhood", "GrLivArea", "OverallQual", "BedroomAbvGr", "SalePrice"]]
        top10["Id"] = top10["Id"].astype(str)
        show(px.bar(top10, x="Id", y="SalePrice", color="Neighborhood",
                    title=T("Top 10 most expensive houses", "أغلى 10 بيوت"),
                    labels={"SalePrice": price_lbl, "Id": T("House ID", "رقم البيت")}).update_xaxes(type="category"))
        cols = ["Id", "Neighborhood", "GrLivArea", "OverallQual", "YearBuilt", "SaleCondition", "SalePrice"]
        c1, c2 = st.columns(2)
        c1.markdown(T("**5 cheapest houses**", "**أرخص 5 بيوت**"))
        c1.dataframe(df.nsmallest(5, "SalePrice")[cols].astype({"YearBuilt": str}), hide_index=True)
        c2.markdown(T("**5 most expensive houses**", "**أغلى 5 بيوت**"))
        c2.dataframe(df.nlargest(5, "SalePrice")[cols].astype({"YearBuilt": str}), hide_index=True)
        t10 = df[df.SalePrice >= df.SalePrice.quantile(0.90)]
        pc = ["GrLivArea", "OverallQual", "TotalBsmtSF", "GarageCars", "YearBuilt"]
        prof = pd.DataFrame({T("Top 10%", "أعلى 10%"): t10[pc].median(), T("All houses", "كل البيوت"): df[pc].median()})
        md(T("**Profile of the 10% most expensive houses (medians)**", "**مواصفات أغلى 10% من البيوت (الوسيط)**"))
        st.dataframe(prof)
        info(T(
            "Three of the five cheapest houses are in IDOTRR; the most expensive are in NoRidge and NridgHt. The top 10% have a "
            "median living area of 2,167 sq ft (vs 1,464), quality 8 (vs 6), a 3-car garage (vs 2) and were built around 2004 (vs 1973). "
            "Meaning: expensive houses combine high quality, large space and recent construction.",
            "تلاتة من أرخص 5 بيوت في IDOTRR، وأغلى البيوت في NoRidge وNridgHt. أعلى 10% وسيط مساحتهم 2,167 قدم مربع (مقابل 1,464)، "
            "الجودة 8 (مقابل 6)، جراج 3 عربيات (مقابل 2)، ومبنية حوالي 2004 (مقابل 1973). "
            "المعنى: البيوت الغالية بتجمع جودة عالية ومساحة كبيرة وبناء حديث."))

    # ------------------------------------------------------------------ Q2
    with tabs[2]:
        st.subheader(T("Q2: Which neighborhoods have the highest average sale prices?",
                       "س2: أنهي أحياء متوسط أسعارها أعلى؟"))
        tn = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False).head(10).reset_index()
        show(px.bar(tn, x="Neighborhood", y="SalePrice", title=T("Top 10 neighborhoods by average sale price", "أعلى 10 أحياء في متوسط السعر"),
                    labels={"SalePrice": avg_lbl}))
        info(T(
            "NoRidge has the highest average ($335,295), then NridgHt ($316,271) and StoneBr ($310,499). All three are above $300,000. "
            "Blmngtn is the lowest of the top 10 ($194,871). Location is strongly associated with price.",
            "NoRidge الأعلى في المتوسط ($335,295)، بعده NridgHt ($316,271) وStoneBr ($310,499). التلاتة فوق $300,000. "
            "وBlmngtn أقل واحد في أعلى 10 ($194,871). الموقع مرتبط بقوة بالسعر."))

    # ------------------------------------------------------------------ Q3
    with tabs[3]:
        st.subheader(T("Q3: How does Overall Quality affect Sale Price?", "س3: إزاي جودة البيت الكلية بتأثر على السعر؟"))
        q = df.groupby("OverallQual")["SalePrice"].agg(["count", "mean", "median"]).round(0).reset_index()
        show(px.bar(q, x="OverallQual", y="mean", title=T("Average sale price by overall quality", "متوسط السعر حسب الجودة الكلية"),
                    labels={"mean": avg_lbl, "OverallQual": T("Overall quality", "الجودة الكلية")}).update_xaxes(dtick=1))
        st.dataframe(q.rename(columns={"count": T("Count", "العدد"), "mean": T("Mean", "المتوسط"), "median": T("Median", "الوسيط")}), hide_index=True)
        info(T(
            "Quality 10 averages $438,588 and quality 1 averages $50,150. Price generally rises with quality. Most houses are quality 5, 6 or 7, "
            "and some levels have few houses, so read their averages carefully.",
            "جودة 10 متوسطها $438,588 وجودة 1 متوسطها $50,150. السعر بيزيد عمومًا مع الجودة. أغلب البيوت جودتها 5 أو 6 أو 7، "
            "وفي مستويات عددها قليل فمتوسطاتها لازم تتقرا بحذر."))

    # ------------------------------------------------------------------ Q4
    with tabs[4]:
        st.subheader(T("Q4: Is there a relationship between living area and sale price?", "س4: في علاقة بين المساحة المعيشية والسعر؟"))
        st.metric(T("Correlation GrLivArea vs SalePrice", "الارتباط بين GrLivArea والسعر"), f"{df.GrLivArea.corr(df.SalePrice):.3f}")
        show(px.scatter(df, x="GrLivArea", y="SalePrice", opacity=0.6, title=T("Living area vs sale price", "المساحة المعيشية مقابل السعر"),
                        labels={"GrLivArea": T("Living area (sq ft)", "المساحة المعيشية (قدم مربع)"), "SalePrice": price_lbl}))
        info(T("There is a strong positive linear relationship (0.709): larger houses sell for more.",
                  "في علاقة طردية قوية (0.709): كل ما البيت أكبر سعره بيزيد."))

    # ------------------------------------------------------------------ Q5
    with tabs[5]:
        st.subheader(T("Q5: How does the number of bedrooms affect sale price?", "س5: إزاي عدد الغرف بيأثر على السعر؟"))
        b = df.groupby("BedroomAbvGr")["SalePrice"].agg(["count", "mean", "median"]).round(0).reset_index()
        show(px.bar(b, x="BedroomAbvGr", y="mean", title=T("Average sale price by number of bedrooms", "متوسط السعر حسب عدد الغرف"),
                    labels={"mean": avg_lbl, "BedroomAbvGr": T("Bedrooms", "عدد الغرف")}).update_xaxes(dtick=1))
        st.dataframe(b.rename(columns={"count": T("Count", "العدد"), "mean": T("Mean", "المتوسط"), "median": T("Median", "الوسيط")}), hide_index=True)
        info(T(
            "4 bedrooms average $220,421; 3 bedrooms are the most common (804 houses). Price does not rise steadily with bedrooms, and groups "
            "with 0, 6 or 8 bedrooms are tiny. Bedrooms alone do not set the price: area, quality and neighborhood matter too.",
            "4 غرف متوسطها $220,421؛ و3 غرف هي الأكتر (804 بيت). السعر مش بيزيد بشكل ثابت مع الغرف، ومجموعات 0 و6 و8 غرف صغيرة جدًا. "
            "الغرف لوحدها مش بتحدد السعر: المساحة والجودة والحي كمان مهمين."))

    # ------------------------------------------------------------------ Q6
    with tabs[6]:
        st.subheader(T("Q6: Do houses with garages have higher sale prices?", "س6: البيوت اللي فيها جراج أغلى؟"))
        d = df.copy()
        d["HasGarage"] = np.where(d.GarageCars > 0, T("With garage", "فيه جراج"), T("Without garage", "من غير جراج"))
        g = d.groupby("HasGarage")["SalePrice"].agg(["count", "mean", "median"]).round(0).reset_index()
        show(px.bar(g, x="HasGarage", y="mean", title=T("Average price: with vs without garage", "متوسط السعر: بجراج وبدون جراج"),
                    labels={"mean": avg_lbl, "HasGarage": T("Garage availability", "وجود جراج")}), 360)
        st.dataframe(g.rename(columns={"HasGarage": T("Garage", "الجراج"), "count": T("Count", "العدد"), "mean": T("Mean", "المتوسط"), "median": T("Median", "الوسيط")}), hide_index=True)
        info(T(
            "Houses with a garage average $185,480 vs $103,317 without (a gap of about $82,162). Most houses (1,379 of 1,460) have one. "
            "This may also reflect differences in size, quality and neighborhood.",
            "البيوت اللي فيها جراج متوسطها $185,480 مقابل $103,317 من غير جراج (فرق حوالي $82,162). أغلب البيوت (1,379 من 1,460) فيها جراج. "
            "والفرق ممكن يكون بسبب اختلاف المساحة والجودة والحي كمان."))

    # ------------------------------------------------------------------ Q7
    with tabs[7]:
        st.subheader(T("Q7: How does house age affect sale price?", "س7: إزاي عمر البيت بيأثر على السعر؟"))
        d = df.copy()
        labs = ["0-10", "11-30", "31-50", "51-100", T("Above 100", "أكتر من 100")]
        d["HouseAgeGroup"] = pd.cut(d.HouseAge, bins=[-1, 10, 30, 50, 100, 200], labels=labs)
        a = d.groupby("HouseAgeGroup", observed=False)["SalePrice"].agg(["count", "mean", "median"]).round(0).reset_index()
        show(px.bar(a, x="HouseAgeGroup", y="mean", title=T("Average sale price by house age", "متوسط السعر حسب عمر البيت"),
                    labels={"mean": avg_lbl, "HouseAgeGroup": T("House age at sale (years)", "عمر البيت وقت البيع (سنين)")}))
        st.dataframe(a.rename(columns={"HouseAgeGroup": T("Age group", "فئة العمر"), "count": T("Count", "العدد"), "mean": T("Mean", "المتوسط"), "median": T("Median", "الوسيط")}), hide_index=True)
        info(T(
            "Age = YrSold - YearBuilt. Houses 0-10 years old average $238,952 (highest); 51-100 years old average $132,249 (lowest). "
            "Price generally falls as age rises. The 100+ group has only 28 houses.",
            "العمر = YrSold - YearBuilt. البيوت من 0 لـ 10 سنين متوسطها $238,952 (الأعلى)؛ من 51 لـ 100 سنة متوسطها $132,249 (الأقل). "
            "السعر بيقل عمومًا مع زيادة العمر. مجموعة أكتر من 100 سنة فيها 28 بيت بس."))

    # ------------------------------------------------------------------ Q8
    with tabs[8]:
        st.subheader(T("Q8: Which features have the strongest correlation with SalePrice?", "س8: أنهي أعمدة أقوى ارتباط بالسعر؟"))
        tc = (df.select_dtypes(include=np.number).corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False).head(10).reset_index())
        tc.columns = ["Feature", "Correlation"]
        show(px.bar(tc.iloc[::-1], x="Correlation", y="Feature", orientation="h", text=tc.iloc[::-1]["Correlation"].round(3),
                    title=T("Top 10 features correlated with sale price", "أعلى 10 أعمدة ارتباطًا بالسعر")))
        info(T(
            "OverallQual is strongest (0.791), then GrLivArea (0.709), GarageCars (0.640), GarageArea (0.623), TotalBsmtSF (0.614) and 1stFlrSF (0.606). "
            "Correlation shows association, not causation.",
            "OverallQual الأقوى (0.791)، بعده GrLivArea (0.709) وGarageCars (0.640) وGarageArea (0.623) وTotalBsmtSF (0.614) وَ1stFlrSF (0.606). "
            "الارتباط معناه علاقة مش سببية."))

    # ------------------------------------------------------------------ Q9
    with tabs[9]:
        st.subheader(T("Q9: Lowest-priced neighborhoods, house counts and strength of the neighborhood effect",
                       "س9: أرخص الأحياء وعدد البيوت وقوة تأثير الحي"))
        ns = df.groupby("Neighborhood")["SalePrice"].agg(["count", "mean"]).sort_values("mean")
        fig = px.bar(ns.reset_index(), x="mean", y="Neighborhood", orientation="h",
                     title=T("Average sale price by neighborhood", "متوسط السعر حسب الحي"), labels={"mean": avg_lbl})
        show(fig, 640)
        gm = df.groupby("Neighborhood")["SalePrice"].transform("mean")
        eta = ((gm - df.SalePrice.mean()) ** 2).sum() / ((df.SalePrice - df.SalePrice.mean()) ** 2).sum()
        c = st.columns(3)
        c[0].metric(T("Avg of lowest 3", "متوسط أرخص 3"), f"${ns['mean'].head(3).mean():,.0f}")
        c[1].metric(T("Avg of top 3", "متوسط أعلى 3"), f"${ns['mean'].tail(3).mean():,.0f}")
        c[2].metric("Eta squared", f"{eta:.3f}")
        info(T(
            "Lowest: MeadowV ($98,576), IDOTRR ($100,124), BrDale ($104,494). NAmes has the most houses (225). Neighborhood explains about 55% "
            "of price variance, partly because size and quality also differ between neighborhoods.",
            "الأرخص: MeadowV ($98,576) وIDOTRR ($100,124) وBrDale ($104,494). NAmes فيه أكتر عدد بيوت (225). الحي بيفسر حوالي 55% "
            "من تباين السعر، جزئيًا لأن المساحة والجودة كمان بيختلفوا بين الأحياء."))

    # ------------------------------------------------------------------ Q10
    with tabs[10]:
        st.subheader(T("Q10: How do price and sales volume change by month?", "س10: إزاي السعر وعدد المبيعات بيتغيروا حسب الشهر؟"))
        ms = df.groupby("MoSold")["SalePrice"].agg(["count", "mean", "median"]).round(0).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            show(px.bar(ms, x="MoSold", y="mean", title=T("Average sale price by month", "متوسط السعر حسب الشهر"),
                        labels={"mean": avg_lbl, "MoSold": T("Month", "الشهر")}).update_xaxes(dtick=1), 360)
        with c2:
            show(px.bar(ms, x="MoSold", y="count", title=T("Number of sales by month", "عدد المبيعات حسب الشهر"),
                        labels={"count": T("Sales", "المبيعات"), "MoSold": T("Month", "الشهر")}).update_xaxes(dtick=1), 360)
        info(T(
            "September has the highest average price ($195,683), April the lowest ($171,503), June the most sales (253). Sales volume is "
            "clearly seasonal (peak in early summer); the average price varies much less.",
            "سبتمبر أعلى متوسط سعر ($195,683)، أبريل الأقل ($171,503)، ويونيو أكتر مبيعات (253). عدد المبيعات موسمي بوضوح (ذروة أول الصيف) "
            "بينما متوسط السعر بيتغير أقل بكتير."))

    # ------------------------------------------------------------------ Q11
    with tabs[11]:
        st.subheader(T("Q11: How do bathrooms and garage capacity relate to price?", "س11: إزاي الحمامات وسعة الجراج مرتبطين بالسعر؟"))
        tb = df.FullBath + df.BsmtFullBath + 0.5 * (df.HalfBath + df.BsmtHalfBath)
        bs = df.groupby(tb)["SalePrice"].agg(["count", "mean", "median"]).round(0)
        bs.index.name = "bath"
        bs = bs.reset_index()
        gs = df.groupby("GarageCars")["SalePrice"].agg(["count", "mean", "median"]).round(0).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            show(px.bar(bs, x="bath", y="mean", title=T("Average price by total bathrooms", "متوسط السعر حسب عدد الحمامات"),
                        labels={"mean": avg_lbl, "bath": T("Total bathrooms", "إجمالي الحمامات")}), 360)
        with c2:
            show(px.bar(gs, x="GarageCars", y="mean", title=T("Average price by garage capacity", "متوسط السعر حسب سعة الجراج"),
                        labels={"mean": avg_lbl, "GarageCars": T("Garage capacity (cars)", "سعة الجراج (عربيات)")}).update_xaxes(dtick=1), 360)
        info(T(
            "Average price rises from about $110,870 (1 bathroom) to $319,119 (4 bathrooms), and from $103,317 (no garage) to $309,636 (3-car garage). "
            "Groups above 4 bathrooms and the 4-car garage are very small, so their averages are unreliable.",
            "متوسط السعر بيزيد من حوالي $110,870 (حمام واحد) لـ $319,119 (4 حمامات)، ومن $103,317 (من غير جراج) لـ $309,636 (جراج 3 عربيات). "
            "المجموعات فوق 4 حمامات وجراج 4 عربيات صغيرة جدًا فمتوسطاتها مش موثوقة."))

    # ------------------------------------------------------------------ Q12
    with tabs[12]:
        st.subheader(T("Q12: Does having a basement significantly affect the price?", "س12: وجود بدروم بيأثر على السعر بشكل معنوي؟"))
        hb = df.TotalBsmtSF > 0
        d = df.copy()
        d["Basement"] = np.where(hb, T("Basement", "فيه بدروم"), T("No basement", "من غير بدروم"))
        res = ttest_ind(df.loc[hb, "SalePrice"], df.loc[~hb, "SalePrice"], equal_var=False)
        bsum = d.groupby("Basement")["SalePrice"].agg(["count", "mean", "median"]).round(0).reset_index()
        show(px.box(d, x="Basement", y="SalePrice", title=T("Sale price with and without a basement", "السعر مع وبدون بدروم"),
                    labels={"SalePrice": price_lbl, "Basement": ""}), 400)
        st.dataframe(bsum.rename(columns={"Basement": T("Group", "المجموعة"), "count": T("Count", "العدد"), "mean": T("Mean", "المتوسط"), "median": T("Median", "الوسيط")}), hide_index=True)
        st.metric(T("Welch t-test p-value", "قيمة p لاختبار Welch"), f"{res.pvalue:.2e}")
        info(T(
            "Houses with a basement average $182,878 vs $105,653 without. The difference is statistically significant, but only 37 houses have no "
            "basement and they may differ in other ways, so it is not proof of causation.",
            "البيوت اللي فيها بدروم متوسطها $182,878 مقابل $105,653 من غير بدروم. الفرق معنوي إحصائيًا، لكن 37 بيت بس من غير بدروم "
            "وممكن يختلفوا في حاجات تانية، فده مش إثبات لسببية."))

    # ------------------------------------------------------------------ Outliers
    with tabs[13]:
        st.subheader(T("Outliers and skewness (preprocessing section)", "القيم الشاذة والالتواء (قسم المعالجة)"))
        X = df.drop(columns=["SalePrice", "Id"]).copy()
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)
        X = X.drop(columns=["YearBuilt"])
        num = X.select_dtypes(include="number")
        rows = []
        for col in num.columns:
            q1, q3 = num[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            n = int(((num[col] < q1 - 1.5 * iqr) | (num[col] > q3 + 1.5 * iqr)).sum())
            rows.append({"Feature": col, T("Outliers", "عدد القيم الشاذة"): n, "%": round(n / len(num) * 100, 2)})
        od = pd.DataFrame(rows).sort_values(T("Outliers", "عدد القيم الشاذة"), ascending=False)
        c1, c2 = st.columns([1, 1])
        with c1:
            md(T("**IQR outliers per numeric feature**", "**القيم الشاذة (طريقة IQR) لكل عمود رقمي**"))
            st.dataframe(od, hide_index=True, height=420)
        with c2:
            feat = st.selectbox(T("Feature", "العمود"), list(num.columns), index=list(num.columns).index("LotArea"), key="out_feat")
            show(px.box(num, x=feat, title=T(f"Boxplot of {feat}", f"بوكس بلوت لـ {feat}")), 200)
            h1, h2 = st.columns(2)
            with h1:
                show(px.histogram(num, x=feat, nbins=30, title=T("Original", "الأصلي")), 260)
            with h2:
                show(px.histogram(x=np.log1p(num[feat].clip(lower=0)), nbins=30, title="log1p",
                                  labels={"x": feat}), 260)
        sf = ["LotArea", "LotFrontage", "MasVnrArea", "BsmtFinSF1", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "WoodDeckSF", "OpenPorchSF"]
        sk = pd.DataFrame({"Feature": sf, T("Before", "قبل"): [X[c].skew() for c in sf], T("After log1p", "بعد log1p"): [np.log1p(X[c]).skew() for c in sf]}).round(3)
        md(T("**Skewness before and after log1p**", "**الالتواء قبل وبعد log1p**"))
        st.dataframe(sk, hide_index=True)
        info(T(
            f"SalePrice skewness goes from {df.SalePrice.skew():.2f} to {np.log1p(df.SalePrice).skew():.2f} after log1p, so the model predicts the log of the price. "
            "Outliers were inspected and kept: most are real large or expensive houses, and the log transformation reduces their influence.",
            f"التواء السعر بيتغير من {df.SalePrice.skew():.2f} لـ {np.log1p(df.SalePrice).skew():.2f} بعد log1p، عشان كده الموديل بيتنبأ بلوج السعر. "
            "القيم الشاذة اتفحصت واتساب: أغلبها بيوت حقيقية كبيرة أو غالية، وتحويل log بيقلل تأثيرها."))
