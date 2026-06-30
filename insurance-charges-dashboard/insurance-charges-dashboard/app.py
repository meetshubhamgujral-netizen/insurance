"""
Insurance Charges — ML Model Dashboard (Streamlit)

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then connect the repo at share.streamlit.io
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import pipeline as pl

# ----------------------------------------------------------------------
# Page config + light styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Charges ML Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

FAMILY_COLORS = {"Linear": "#4C72B0", "Tree": "#2E9E5B"}
ACCENT = "#2E7D32"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached loaders  (heavy work runs once, then interactions are instant)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_everything():
    df = pl.load_data()
    X, y = pl.engineer_features(df)
    return df, X, y


@st.cache_resource(show_spinner="Training models (first load only)…")
def train_models(test_size: float, seed: int):
    df, X, y = load_everything()
    X_tr, X_te, y_tr, y_te = pl.split_data(X, y, test_size=test_size, random_state=seed)
    results, fitted, preds = pl.train_and_evaluate(X_tr, X_te, y_tr, y_te)
    return {"results": results, "fitted": fitted, "preds": preds,
            "X_te": X_te, "y_te": y_te}


@st.cache_data(show_spinner="Running 5-fold cross-validation…")
def cross_validate(test_size: float, seed: int):
    _, X, y = load_everything()
    X_tr, _, y_tr, _ = pl.split_data(X, y, test_size=test_size, random_state=seed)
    return pl.cross_validate_models(X_tr, y_tr)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("💸 Insurance ML")
    st.caption("Predicting individual medical insurance charges")

    st.subheader("⚙️ Settings")
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05,
                          help="Fraction of data held out for evaluation.")
    seed = int(st.number_input("Random seed", value=42, step=1))
    show_cv = st.checkbox("Add 5-fold cross-validated R²", value=False,
                          help="More robust but slower (re-fits every model 5×).")

    st.divider()
    st.markdown(
        "**Models compared**\n\n"
        "Linear · Ridge · Lasso · ElasticNet  \n"
        "Decision Tree · Random Forest · Extra Trees · Bagging  \n"
        "AdaBoost · Gradient Boosting · HistGBM"
        + (" · XGBoost" if pl.HAS_XGB else "")
        + (" · LightGBM" if pl.HAS_LGBM else "")
    )
    st.divider()
    st.caption("Data: 1,338 policyholders · target = `charges`")


# ----------------------------------------------------------------------
# Load + train
# ----------------------------------------------------------------------
df, X, y = load_everything()
bundle = train_models(test_size, seed)
results = bundle["results"].copy()
if show_cv:
    cv = cross_validate(test_size, seed)
    results["CV R²"] = results["Model"].map(cv)

best_model = results.iloc[0]["Model"]

st.title("Insurance Charges — ML Model Dashboard")
st.markdown(
    "Predict **annual medical insurance charges** from age, sex, BMI, children, "
    "smoking status and region — comparing **13 regression models** with feature engineering."
)

tab_data, tab_perf, tab_predict, tab_about = st.tabs(
    ["📊 Data Explorer", "🏆 Model Performance", "🔮 Predict", "ℹ️ About"]
)

# ======================================================================
# TAB 1 — DATA EXPLORER
# ======================================================================
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policyholders", f"{len(df):,}")
    c2.metric("Avg. charge", f"${df['charges'].mean():,.0f}")
    c3.metric("Smokers", f"{(df['smoker'].eq('yes')).mean()*100:.1f}%")
    c4.metric("Avg. age", f"{df['age'].mean():.0f}")

    st.markdown("#### How charges are distributed")
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.histogram(df, x="charges", color="smoker", nbins=40, marginal="box",
                           color_discrete_map={"yes": "#C0392B", "no": "#2E9E5B"},
                           labels={"charges": "Annual charge ($)"})
        fig.update_layout(title="Charge distribution by smoking status",
                          legend_title="Smoker", bargap=0.05, height=420)
        st.plotly_chart(fig, width='stretch')
    with cc2:
        fig = px.box(df, x="smoker", y="charges", color="smoker", points="all",
                     color_discrete_map={"yes": "#C0392B", "no": "#2E9E5B"},
                     labels={"charges": "Annual charge ($)", "smoker": "Smoker"})
        fig.update_layout(title="Smoking is the dominant cost driver",
                          showlegend=False, height=420)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Charges vs. continuous drivers")
    cc3, cc4 = st.columns(2)
    with cc3:
        fig = px.scatter(df, x="age", y="charges", color="smoker", trendline="ols",
                         color_discrete_map={"yes": "#C0392B", "no": "#2E9E5B"},
                         labels={"charges": "Annual charge ($)", "age": "Age"})
        fig.update_layout(title="Charges vs. age", legend_title="Smoker", height=420)
        st.plotly_chart(fig, width='stretch')
    with cc4:
        fig = px.scatter(df, x="bmi", y="charges", color="smoker",
                         color_discrete_map={"yes": "#C0392B", "no": "#2E9E5B"},
                         labels={"charges": "Annual charge ($)", "bmi": "BMI"})
        fig.add_vline(x=30, line_dash="dash", line_color="gray",
                      annotation_text="obesity threshold")
        fig.update_layout(title="Charges vs. BMI (note the smoker split above BMI 30)",
                          legend_title="Smoker", height=420)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Correlation (engineered numeric features)")
    corr = pd.concat([X[["age", "bmi", "children", "smoker", "smoker_bmi"]], y], axis=1).corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=420)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Preview raw data"):
        st.dataframe(df, width='stretch', hide_index=True)

# ======================================================================
# TAB 2 — MODEL PERFORMANCE
# ======================================================================
with tab_perf:
    m1, m2, m3 = st.columns(3)
    top = results.iloc[0]
    m1.metric("Best model (R²)", best_model, f"{top['R2']:.4f}")
    m2.metric("Lowest MAE", results.loc[results['MAE'].idxmin(), 'Model'],
              f"${results['MAE'].min():,.0f}")
    m3.metric("Lowest RMSE", results.loc[results['RMSE'].idxmin(), 'Model'],
              f"${results['RMSE'].min():,.0f}")

    st.markdown("#### Results table — test set, sorted by R²")
    disp = results.rename(columns={"R2": "R²"}).copy()
    fmt = {"R²": "{:.4f}", "MAE": "${:,.0f}", "RMSE": "${:,.0f}"}
    if "CV R²" in disp.columns:
        fmt["CV R²"] = "{:.4f}"
    styler = (disp.style.format(fmt)
              .highlight_max(subset=["R²"], color="#C8E6C9")
              .highlight_min(subset=["MAE", "RMSE"], color="#C8E6C9"))
    st.dataframe(styler, width='stretch', hide_index=True)

    st.markdown("#### Metric comparison")
    g1, g2, g3 = st.columns(3)
    with g1:
        d = results.sort_values("R2")
        fig = px.bar(d, x="R2", y="Model", orientation="h", color="Family",
                     color_discrete_map=FAMILY_COLORS, text=d["R2"].map("{:.3f}".format))
        fig.update_layout(title="R² (higher is better)", xaxis_range=[0, 1],
                          showlegend=False, height=460, yaxis_title="")
        st.plotly_chart(fig, width='stretch')
    with g2:
        d = results.sort_values("RMSE", ascending=False)
        fig = px.bar(d, x="RMSE", y="Model", orientation="h", color="Family",
                     color_discrete_map=FAMILY_COLORS, text=d["RMSE"].map("${:,.0f}".format))
        fig.update_layout(title="RMSE (lower is better)", showlegend=False,
                          height=460, yaxis_title="")
        st.plotly_chart(fig, width='stretch')
    with g3:
        d = results.sort_values("MAE", ascending=False)
        fig = px.bar(d, x="MAE", y="Model", orientation="h", color="Family",
                     color_discrete_map=FAMILY_COLORS, text=d["MAE"].map("${:,.0f}".format))
        fig.update_layout(title="MAE (lower is better)", showlegend=False,
                          height=460, yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.markdown("#### Inspect a single model")
    sel = st.selectbox("Choose a model", list(bundle["fitted"].keys()),
                       index=list(bundle["fitted"].keys()).index(best_model))
    y_te = bundle["y_te"]
    p = bundle["preds"][sel]

    s1, s2 = st.columns(2)
    with s1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=p, mode="markers",
                                 marker=dict(color=ACCENT, opacity=0.5, size=7),
                                 name="predictions"))
        lims = [float(y_te.min()), float(y_te.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                 line=dict(color="#C0392B", dash="dash"),
                                 name="perfect"))
        fig.update_layout(title=f"Actual vs. Predicted — {sel}",
                          xaxis_title="Actual ($)", yaxis_title="Predicted ($)",
                          height=430)
        st.plotly_chart(fig, width='stretch')
    with s2:
        resid = np.asarray(y_te) - np.asarray(p)
        fig = px.scatter(x=p, y=resid, labels={"x": "Predicted ($)", "y": "Residual ($)"},
                         color_discrete_sequence=[ACCENT])
        fig.add_hline(y=0, line_dash="dash", line_color="#C0392B")
        fig.update_traces(marker=dict(opacity=0.5, size=7))
        fig.update_layout(title=f"Residuals — {sel}", height=430)
        st.plotly_chart(fig, width='stretch')

    imp, kind = pl.feature_importance(bundle["fitted"][sel])
    if imp is not None:
        fig = px.bar(imp, orientation="h", color_discrete_sequence=[ACCENT],
                     labels={"value": kind, "index": "Feature"})
        fig.update_layout(title=f"Feature {kind} — {sel}", showlegend=False, height=430)
        st.plotly_chart(fig, width='stretch')

# ======================================================================
# TAB 3 — PREDICT
# ======================================================================
with tab_predict:
    st.markdown("#### Estimate a policyholder's annual charge")
    st.caption("Adjust the inputs — predictions update live. Try toggling **Smoker**.")

    i1, i2, i3 = st.columns(3)
    age = i1.slider("Age", 18, 64, 39)
    sex = i1.radio("Sex", ["female", "male"], horizontal=True)
    bmi = i2.slider("BMI", 15.0, 53.0, 30.0, 0.1)
    children = i2.select_slider("Children", options=[0, 1, 2, 3, 4, 5], value=1)
    smoker = i3.radio("Smoker", ["no", "yes"], horizontal=True)
    region = i3.selectbox("Region", pl.REGIONS, index=0)

    model_names = list(bundle["fitted"].keys())
    chosen = st.selectbox("Prediction model", model_names,
                          index=model_names.index(best_model))

    record = {"age": age, "sex": sex, "bmi": bmi,
              "children": children, "smoker": smoker, "region": region}
    row = pl.build_input_features(record)
    pred = float(bundle["fitted"][chosen].predict(row)[0])
    pred = max(pred, 0.0)

    st.markdown("")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric(f"Predicted annual charge", f"${pred:,.0f}",
                  help=f"Model: {chosen}")
        pct = (df["charges"] < pred).mean() * 100
        st.caption(f"Higher than **{pct:.0f}%** of policyholders in the dataset.")
    with r2:
        fig = px.histogram(df, x="charges", nbins=40,
                           color_discrete_sequence=["#B0BEC5"],
                           labels={"charges": "Annual charge ($)"})
        fig.add_vline(x=pred, line_color=ACCENT, line_width=3,
                      annotation_text="this estimate", annotation_position="top")
        fig.update_layout(title="Where this estimate falls", height=300,
                          showlegend=False, bargap=0.05)
        st.plotly_chart(fig, width='stretch')

    st.markdown("##### Agreement across the top models")
    cols = st.columns(min(4, len(model_names)))
    for col, name in zip(cols, results["Model"].head(len(cols))):
        val = max(float(bundle["fitted"][name].predict(row)[0]), 0.0)
        col.metric(name, f"${val:,.0f}")

# ======================================================================
# TAB 4 — ABOUT
# ======================================================================
with tab_about:
    st.markdown(
        f"""
### About this project

A regression study on the classic **medical insurance** dataset
(1,338 policyholders) predicting annual **`charges`** from six attributes.

**Feature engineering**
- Dropped the non-predictive row index.
- Binary-encoded `sex` and `smoker`; one-hot encoded `region` (northeast = baseline).
- Engineered five domain features: `age²`, an obesity flag (BMI ≥ 30),
  and three interactions — **`smoker × bmi`** (the dominant driver),
  `smoker × obese`, and `smoker × age`.

**Models ({len(model_names)} total)**
- *Linear / regularized:* Linear, Ridge, Lasso, ElasticNet (penalties auto-tuned by CV; features standardized).
- *Decision-tree family:* Decision Tree, Random Forest, Extra Trees, Bagging,
  AdaBoost, Gradient Boosting, HistGradientBoosting{', XGBoost' if pl.HAS_XGB else ''}{', LightGBM' if pl.HAS_LGBM else ''}.

**Evaluation** — an {int((1-test_size)*100)}/{int(test_size*100)} train/test split.
- **R²** — variance explained (1.0 = perfect).
- **MAE** — average absolute error, in dollars.
- **RMSE** — root mean squared error; penalizes large misses more than MAE.
- Optional **5-fold cross-validated R²** for stability.

**Key finding** — once the `smoker × bmi` interaction is made explicit,
the simple linear models match the gradient-boosting ensembles (R² ≈ 0.88),
which is the payoff of feature engineering: interpretable models that perform.

---
**Tech stack:** Python · scikit-learn · XGBoost · LightGBM · Plotly · Streamlit  
**Reproducibility:** results depend on the random seed (default 42); the *ranking* is stable.
        """
    )
