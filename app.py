from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_xgboost.json"
MEDIANS_PATH = BASE_DIR / "imputer_medians.csv"
METRICS_PATH = BASE_DIR / "diabetes_model_comparison.csv"
IMPORTANCE_PATH = BASE_DIR / "diabetes_shap_feature_importance.csv"
DATA_PATH = BASE_DIR / "diabetes.csv"  # Pima Indians dataset (with Outcome)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Palette
BG = "#07111f"
CARD = "#0d1b2e"
BORDER = "#1c3553"
TEXT = "#e6eef8"
MUTED = "#8fa6c2"
CYAN = "#4cc9f0"
BLUE = "#2f80ed"
GREEN = "#22c55e"
RED = "#ef4444"
GRID = "rgba(143,166,194,0.15)"

st.set_page_config(
    page_title="Diabetes Risk Analytics",
    page_icon="🩺",
    layout="wide",
)

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background: radial-gradient(1200px 600px at 85% -10%, #12325a 0%, {BG} 55%);
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }}

.hero-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 4rem;
    line-height: 0.95;
    letter-spacing: 1px;
    color: #ffffff;
    text-transform: uppercase;
}}
.hero-title .accent {{ color: {CYAN}; display: block; }}
.hero-sub {{ color: {TEXT}; font-size: 1.1rem; margin-top: 0.8rem; line-height: 1.5; }}

.chip-row {{
    display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end;
    background: rgba(13,27,46,0.8); border: 1px solid {BORDER};
    border-radius: 14px; padding: 14px 18px;
}}
.chip {{ color: {TEXT}; font-weight: 500; font-size: 0.95rem; padding: 2px 10px; }}
.chip b {{ color: {CYAN}; }}

.kpi {{
    background: linear-gradient(160deg, #0f2440, {CARD});
    border: 1px solid {BORDER}; border-radius: 14px;
    padding: 16px 18px; height: 100%;
}}
.kpi .value {{
    font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
    font-size: 2.3rem; color: {CYAN}; line-height: 1.1;
}}
.kpi .label {{ color: {MUTED}; font-size: 0.85rem; margin-top: 2px; }}

.panel-title {{ color: {TEXT}; font-weight: 600; font-size: 1rem; margin-bottom: 4px; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: linear-gradient(160deg, #0d1f36, #0a1626);
    border: 1px solid {BORDER} !important;
    border-radius: 14px;
}}

.insight {{
    background: rgba(13,27,46,0.85); border: 1px solid {BORDER};
    border-left: 4px solid {CYAN}; border-radius: 12px; padding: 14px 18px;
}}
.insight .head {{ color: {CYAN}; font-weight: 600; font-size: 0.95rem; }}
.insight .body {{ color: {TEXT}; font-size: 0.92rem; margin-top: 4px; }}

.footer-bar {{
    background: rgba(13,27,46,0.85); border: 1px solid {BORDER};
    border-radius: 14px; padding: 14px 22px; margin-top: 18px;
    display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;
}}
.footer-bar .left {{ color: {TEXT}; font-weight: 500; }}
.footer-bar .right {{ color: {CYAN}; font-size: 0.9rem; }}

button[data-baseweb="tab"] {{ font-weight: 600; font-size: 1rem; }}
div[data-testid="stForm"] {{
    background: transparent; border: 1px solid {BORDER}; border-radius: 14px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_medians():
    table = pd.read_csv(MEDIANS_PATH)
    if not {"Feature", "Median"}.issubset(table.columns):
        raise ValueError("The imputer_medians.csv schema is invalid.")
    medians = table.set_index("Feature")["Median"].reindex(FEATURES)
    if medians.isna().any():
        raise ValueError("Missing or invalid imputation medians.")
    return medians.astype(float)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH, index_col=0)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)


def validate_dataset(df):
    missing = [c for c in FEATURES + ["Outcome"] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    return df


@st.cache_data
def load_dataset(path):
    return validate_dataset(pd.read_csv(path))


try:
    required_files = [MODEL_PATH, MEDIANS_PATH, METRICS_PATH, IMPORTANCE_PATH]
    missing_files = [p.name for p in required_files if not p.exists()]
    if missing_files:
        raise FileNotFoundError("Missing project files: " + ", ".join(missing_files))

    model = load_model()
    medians = load_medians()
    metrics = load_metrics()
    importance = load_importance()
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.stop()

# Dataset: local diabetes.csv -> auto-download (saved locally) -> upload box
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)


@st.cache_data(show_spinner="Downloading Pima dataset...")
def download_dataset():
    raw = pd.read_csv(DATA_URL, header=None, names=FEATURES + ["Outcome"])
    raw.to_csv(DATA_PATH, index=False)
    return validate_dataset(raw)


df = None
try:
    df = load_dataset(DATA_PATH) if DATA_PATH.exists() else download_dataset()
except Exception:
    df = None

if df is None:
    with st.expander("Upload diabetes.csv to enable the data charts", expanded=True):
        uploaded = st.file_uploader("Pima Indians dataset (with Outcome column)", type="csv")
        if uploaded is not None:
            try:
                df = validate_dataset(pd.read_csv(uploaded))
            except Exception as exc:
                st.error(str(exc))

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def style_fig(fig, height=300):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, sans-serif", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def show(fig):
    st.plotly_chart(fig, config={"displayModeBar": False})


def kpi(col, value, label):
    col.markdown(
        f'<div class="kpi"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def panel_title(text):
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


def insight(col, head, body):
    col.markdown(
        f'<div class="insight"><div class="head">{head}</div>'
        f'<div class="body">{body}</div></div>',
        unsafe_allow_html=True,
    )


AGE_BINS = list(range(20, 100, 10))
AGE_LABELS = [f"{b}-{b + 10}" for b in AGE_BINS[:-1]]
OUTCOME_LABELS = {0: "No Diabetes", 1: "Diabetes"}
OUTCOME_COLORS = {"No Diabetes": GREEN, "Diabetes": RED}

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(
    '<div class="hero-title">Diabetes<span class="accent">Risk</span>Analytics</div>'
    '<div class="hero-sub">Predicting diabetes risk.<br>'
    "Explaining every prediction.</div>",
    unsafe_allow_html=True,
)

st.write("")

# ----------------------------------------------------------------------
# KPI strip
# ----------------------------------------------------------------------
best_name = "XGBoost" if "XGBoost" in metrics.index else metrics.index[0]
best = metrics.loc[best_name]

total_records = f"{len(df):,}" if df is not None else "—"
positive_rate = f"{df['Outcome'].mean():.1%}" if df is not None else "—"

k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, total_records, "Patient records")
kpi(k2, positive_rate, "Diabetes prevalence")
kpi(k3, f"{best['Accuracy']:.1%}" if "Accuracy" in best else "—", "Test accuracy")
kpi(k4, f"{best['Recall']:.1%}" if "Recall" in best else "—", "Test recall")
kpi(k5, f"{best['ROC_AUC']:.3f}" if "ROC_AUC" in best else "—", f"ROC-AUC · {best_name}")

st.write("")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_dash, tab_predict, tab_eval = st.tabs(
    ["Dashboard", "Sample Prediction", "Model Evaluation"]
)

# ======================= DASHBOARD =======================
with tab_dash:
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    if df is not None:
        data = df.copy()
        data["Status"] = data["Outcome"].map(OUTCOME_LABELS)
        data["AgeGroup"] = pd.cut(
            data["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
        )

        # Outcome overview (donut)
        with top_left:
            with st.container(border=True):
                panel_title("Outcome Overview")
                counts = data["Status"].value_counts().reindex(
                    ["No Diabetes", "Diabetes"]
                ).fillna(0)
                fig = go.Figure(
                    go.Pie(
                        labels=counts.index,
                        values=counts.values,
                        hole=0.68,
                        sort=False,
                        marker=dict(colors=[GREEN, RED],
                                    line=dict(color=BG, width=2)),
                        textinfo="percent",
                        textfont=dict(size=14),
                    )
                )
                fig.add_annotation(
                    text=f"<b>{len(data):,}</b><br>Total Patients",
                    showarrow=False, font=dict(size=16, color=TEXT),
                )
                show(style_fig(fig, 300))

        # Age distribution
        with top_right:
            with st.container(border=True):
                panel_title("Age Distribution")
                age_counts = (
                    data["AgeGroup"].value_counts().reindex(AGE_LABELS).fillna(0)
                )
                fig = px.bar(
                    x=age_counts.index.astype(str), y=age_counts.values,
                    labels={"x": "", "y": "Patients"},
                )
                fig.update_traces(marker_color=BLUE)
                show(style_fig(fig, 300))

        # Outcome by age group
        with bottom_left:
            with st.container(border=True):
                panel_title("Outcome by Age Group")
                grouped = (
                    data.groupby(["AgeGroup", "Status"], observed=False)
                    .size().reset_index(name="Patients")
                )
                grouped["AgeGroup"] = grouped["AgeGroup"].astype(str)
                fig = px.bar(
                    grouped, x="AgeGroup", y="Patients", color="Status",
                    color_discrete_map=OUTCOME_COLORS, barmode="stack",
                    category_orders={"AgeGroup": AGE_LABELS,
                                     "Status": ["No Diabetes", "Diabetes"]},
                )
                fig.update_layout(legend_title_text="")
                fig.update_xaxes(title="")
                show(style_fig(fig, 320))
    else:
        for col, name in ((top_left, "Outcome Overview"),
                          (top_right, "Age Distribution"),
                          (bottom_left, "Outcome by Age Group")):
            with col:
                with st.container(border=True):
                    panel_title(name)
                    st.caption("Waiting for diabetes.csv")

    # Top 10 important features
    with bottom_right:
        with st.container(border=True):
            panel_title("Top 10 Important Features")
            required_columns = {"Feature", "Mean_Absolute_SHAP"}
            if required_columns.issubset(importance.columns):
                top10 = (
                    importance.nlargest(10, "Mean_Absolute_SHAP")
                    .sort_values("Mean_Absolute_SHAP")
                )
                fig = px.bar(
                    top10, x="Mean_Absolute_SHAP", y="Feature", orientation="h",
                    labels={"Mean_Absolute_SHAP": "Mean |SHAP value|", "Feature": ""},
                )
                fig.update_traces(marker_color=BLUE)
                show(style_fig(fig, 320))
            else:
                st.error("The SHAP importance CSV has unexpected columns.")

    # Key insights
    st.write("")
    i1, i2, i3 = st.columns(3)
    if required_columns.issubset(importance.columns):
        top_feature = importance.nlargest(1, "Mean_Absolute_SHAP")["Feature"].iloc[0]
        insight(i1, "Strongest driver",
                f"<b>{top_feature}</b> has the largest average influence on the model output.")
    if df is not None:
        rate = data.groupby("AgeGroup", observed=False)["Outcome"].mean().dropna()
        if not rate.empty:
            insight(i2, "Highest-prevalence age group",
                    f"<b>{rate.idxmax()}</b> years, with {rate.max():.0%} diabetes prevalence.")
        glu = data.assign(
            Glucose=data["Glucose"].replace(0, np.nan)
        ).groupby("Outcome")["Glucose"].mean()
        if {0, 1}.issubset(glu.index):
            insight(i3, "Glucose gap",
                    f"Average glucose is <b>{glu[1]:.0f}</b> for the diabetes group "
                    f"versus <b>{glu[0]:.0f}</b> for the rest.")

# ======================= PREDICTION =======================
with tab_predict:
    form_col, result_col = st.columns([1.2, 1])

    with form_col:
        with st.form("prediction_form"):
            left, right = st.columns(2)
            with left:
                pregnancies = st.number_input("Pregnancies", 0, 20, 1, step=1)
                glucose = st.number_input("Glucose", 0.0, 400.0, 120.0)
                blood_pressure = st.number_input(
                    "Diastolic blood pressure", 0.0, 250.0, 70.0
                )
                skin_thickness = st.number_input("Skin thickness", 0.0, 150.0, 20.0)
            with right:
                insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
                bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
                pedigree = st.number_input("Diabetes pedigree function", 0.0, 5.0, 0.5)
                age = st.number_input("Age", 1, 120, 30, step=1)

            submitted = st.form_submit_button("Run prediction", type="primary")

    with result_col:
        if submitted:
            sample = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age,
            }], columns=FEATURES)

            for feature in ZERO_AS_MISSING:
                sample.loc[sample[feature] == 0, feature] = np.nan

            # Apply the training-set medians exported from Colab
            sample = sample.fillna(medians)

            probability = float(model.predict_proba(sample)[0, 1])
            predicted_class = int(probability >= 0.5)

            with st.container(border=True):
                panel_title("Model Output")
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number=dict(suffix="%", valueformat=".1f",
                                    font=dict(color=TEXT)),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor=MUTED),
                            bar=dict(color=RED if predicted_class else GREEN),
                            bgcolor="rgba(0,0,0,0)",
                            borderwidth=0,
                            threshold=dict(line=dict(color=CYAN, width=3),
                                           value=50),
                        ),
                    )
                )
                show(style_fig(gauge, 260))

                if predicted_class == 1:
                    st.markdown(f"**Predicted class: 1** — higher risk "
                                f"(probability {probability:.1%})")
                else:
                    st.markdown(f"**Predicted class: 0** — lower risk "
                                f"(probability {probability:.1%})")
                st.caption(
                    "Classification threshold: 50%. Zero values in glucose, blood "
                    "pressure, skin thickness, insulin and BMI are replaced with "
                    "training-set medians."
                )
        else:
            with st.container(border=True):
                panel_title("Model Output")
                st.caption("Enter measurements and select **Run prediction**.")

# ======================= EVALUATION =======================
with tab_eval:
    numeric_metrics = metrics.select_dtypes("number")

    left, right = st.columns([1, 1.2])
    with left:
        with st.container(border=True):
            panel_title("Held-out test results")
            st.dataframe(numeric_metrics.style.format("{:.4f}"))

    with right:
        with st.container(border=True):
            panel_title("Model comparison")
            long = (
                numeric_metrics.reset_index()
                .melt(id_vars=numeric_metrics.index.name or "index",
                      var_name="Metric", value_name="Score")
            )
            model_col = long.columns[0]
            fig = px.bar(long, x="Metric", y="Score", color=model_col,
                         barmode="group",
                         color_discrete_sequence=[CYAN, BLUE, "#7c3aed", GREEN, RED])
            fig.update_layout(legend_title_text="")
            fig.update_xaxes(title="")
            show(style_fig(fig, 320))

    if required_columns.issubset(importance.columns):
        with st.container(border=True):
            panel_title("SHAP feature importance (all features)")
            st.dataframe(
                importance.sort_values("Mean_Absolute_SHAP", ascending=False),
                hide_index=True,
            )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    '<div class="footer-bar">'
    '<div class="left">💡 Data-driven insights. Better decisions. Healthier outcomes.</div>'
    '<div class="right">#DataAnalytics #HealthcareAnalytics #MachineLearning '
    "#Python #DataScience</div>"
    "</div>",
    unsafe_allow_html=True,
)