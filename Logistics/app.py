"""
Streamlit deployment app for the Logistics Clearance Prediction System.

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PROJECT SETTINGS
# =========================================================

APP_OWNER_NAME = "InsightbyFrank"
LINKEDIN_URL = "http://www.linkedin.com/in/frank-agba"
GITHUB_URL = "https://github.com/InsightByFrank/Logistics"

DASHBOARD_SCREENSHOTS = [
    ("assets/1.png", "Customer & Order Performance"),
    ("assets/2.png", "Shipment & Trade Flow"),
    ("assets/3.png", "Logistics & Carrier Performance"),
]

MODEL_PATH = Path("model/logistics_clearance_models.joblib")

REQUIRED_BATCH_COLUMNS = [
    "shipment_id", "type", "date", "product_category",
    "origin", "O_Country", "destination", "D_Country",
    "value", "freight_cost",
]

NAV_PAGES = [
    "🏠 Overview",
    "📦 Single Shipment",
    "📊 Batch Prediction",
    "🧠 Model Performance",
    "👤 About",
]


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Logistics Clearance Risk",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# THEME
# Matches the existing Power BI dashboard:
#   Navy #12344D · Teal #3F8F9D · Light Teal #DCECEF
#   Orange #F47545 · Red #D9535B · Green #238B5B
#   Background #F3F6F7 · White #FFFFFF
# =========================================================

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background-color: #F3F6F7; }
.stApp p, .stApp li { color: #12344D; }
.stApp label { color: #12344D !important; font-weight: 600; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background-color: #12344D; border-right: 1px solid #0D2739; }
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .sidebar-brand { font-size: 1.15rem; font-weight: 800; padding: 0.25rem 0 0.1rem 0; }
section[data-testid="stSidebar"] .sidebar-subtitle { font-size: 0.78rem; color: #DCECEF !important; margin-bottom: 1.1rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: rgba(255, 255, 255, 0.04);
    border-radius: 8px; padding: 9px 12px; margin-bottom: 6px;
    border: 1px solid transparent; width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background-color: rgba(63, 143, 157, 0.30); }
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background-color: #3F8F9D; border: 1px solid #6FB8C4;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.18); }
.sidebar-footer-link {
    display: block; text-align: center; background-color: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 8px; padding: 8px 10px;
    margin-top: 7px; font-size: 0.83rem; font-weight: 600;
    text-decoration: none !important; color: #FFFFFF !important;
}
.sidebar-footer-link:hover { background-color: #3F8F9D; border-color: #3F8F9D; }

/* ---------- Page header ---------- */
.page-eyebrow { color: #3F8F9D !important; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 3px; }
.main-title { color: #12344D !important; font-size: 2.1rem; font-weight: 800; line-height: 1.2; }
.main-subtitle { color: #526575 !important; font-size: 0.98rem; line-height: 1.55; margin-top: 4px; }

/* ---------- KPI cards ---------- */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE5E8 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 3px 10px rgba(18, 52, 77, 0.08) !important;
    min-height: 105px !important;
    overflow: visible !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #526575 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #12344D !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    letter-spacing: -0.03em !important;
}

@media (max-width: 1200px) {
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] div {
        font-size: 1.7rem !important;
    }
}

/* ---------- Generic cards / badges ---------- */
.card {
    background-color: #FFFFFF; border: 1px solid #DCE5E8; border-radius: 12px;
    padding: 18px 20px; box-shadow: 0 3px 10px rgba(18, 52, 77, 0.06);
    margin-bottom: 14px; color: #12344D !important;
}
.card b, .card strong { color: #12344D !important; }
.card code { color: #12344D !important; background-color: #EAF4F6 !important; }
.badge {
    display: inline-block; background-color: #EAF4F6; color: #226B77 !important;
    border: 1px solid #CFE7EB; border-radius: 999px; padding: 4px 12px;
    font-size: 0.78rem; font-weight: 600; margin: 3px 6px 3px 0;
}

/* ---------- Text / Number / Date inputs — flat fill on EVERY layer, no dark layer can show through ---------- */
div[data-testid="stTextInput"] *,
div[data-testid="stNumberInput"] *,
div[data-testid="stDateInput"] * {
    background-color: #DCECEF !important;
    color: #12344D !important;
    -webkit-text-fill-color: #12344D !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    fill: #12344D !important;
}
div[data-testid="stTextInput"] > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stDateInput"] > div {
    border-radius: 9px !important;
    overflow: hidden;
}
div[data-testid="stTextInput"] [data-testid="stWidgetLabel"],
div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"],
div[data-testid="stDateInput"] [data-testid="stWidgetLabel"],
div[data-testid="stTextInput"] [data-testid="stWidgetLabel"] *,
div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] *,
div[data-testid="stDateInput"] [data-testid="stWidgetLabel"] * {
    background-color: transparent !important;
}
input::placeholder, textarea::placeholder { color: #526575 !important; -webkit-text-fill-color: #526575 !important; opacity: 1 !important; }

/* =========================================================
   SELECTBOX / DROPDOWN
   Power BI Dashboard Theme
   ========================================================= */

/* Main selectbox wrapper */
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #DCECEF !important;
    border: none !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}


/* The actual visible dropdown control */
div[data-testid="stSelectbox"]
div[data-baseweb="select"]
> div {
    background-color: #DCECEF !important;
    border: none !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}


/* Selected value area */
div[data-testid="stSelectbox"]
div[data-baseweb="select"]
[role="combobox"] {
    background-color: #DCECEF !important;
    color: #12344D !important;
    -webkit-text-fill-color: #12344D !important;
    border: none !important;
    box-shadow: none !important;
}


/* Selected text */
div[data-testid="stSelectbox"]
div[data-baseweb="select"]
[role="combobox"] > div {
    background-color: transparent !important;
    color: #12344D !important;
    -webkit-text-fill-color: #12344D !important;
}


/* Text inside the selected option */
div[data-testid="stSelectbox"]
div[data-baseweb="select"]
[role="combobox"] span {
    background-color: transparent !important;
    color: #12344D !important;
    -webkit-text-fill-color: #12344D !important;
}


/* =========================================================
   ARROW AREA
   Keep it transparent
   ========================================================= */

div[data-testid="stSelectbox"]
div[data-baseweb="select"]
svg {
    background: transparent !important;
    fill: #12344D !important;
    color: #12344D !important;
    stroke: #12344D !important;
    opacity: 1 !important;
}


/* Remove any background from the arrow container */
div[data-testid="stSelectbox"]
div[data-baseweb="select"]
[aria-hidden="true"] {
    background-color: transparent !important;
    box-shadow: none !important;
}


/* Dropdown label */
div[data-testid="stSelectbox"]
[data-testid="stWidgetLabel"],
div[data-testid="stSelectbox"]
[data-testid="stWidgetLabel"] * {
    background-color: transparent !important;
    color: #12344D !important;
}


/* =========================================================
   OPEN DROPDOWN MENU
   ========================================================= */

div[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}


div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    box-shadow: 0 8px 20px rgba(18, 52, 77, 0.15) !important;
}


div[data-baseweb="menu"] li {
    background-color: #FFFFFF !important;
    color: #12344D !important;
    border: none !important;
}


div[data-baseweb="menu"] li * {
    background-color: transparent !important;
    color: #12344D !important;
}


/* Hover option */
div[data-baseweb="menu"] li:hover {
    background-color: #DCECEF !important;
    color: #12344D !important;
}


/* Selected option */
div[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #DCECEF !important;
    color: #12344D !important;
}

/* ---------- Widget labels ---------- */
[data-testid="stWidgetLabel"] { background-color: transparent !important; }
[data-testid="stWidgetLabel"] p { color: #12344D !important; font-weight: 600 !important; background-color: transparent !important; }

/* ---------- Section headings ---------- */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #12344D !important;
    font-weight: 700 !important;
}

/* Specifically keep Feature Importance clearly visible */
.stApp h3 {
    color: #12344D !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

.stApp h3 * {
    color: #12344D !important;
    -webkit-text-fill-color: #12344D !important;
    opacity: 1 !important;
}

/* ---------- Buttons ---------- */
.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
    background-color: #3F8F9D !important;
    color: #FFFFFF !important;
    border: 1px solid #327985 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    min-height: 46px;
    box-shadow: none !important;
}

/* Force every button/link text and icon to white + bold */
.stButton > button *,
.stDownloadButton > button *,
.stLinkButton > a * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    fill: #FFFFFF !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > a:hover {
    background-color: #327985 !important;
    border-color: #286773 !important;
    color: #FFFFFF !important;
}

.stButton > button:focus,
.stDownloadButton > button:focus,
.stLinkButton > a:focus {
    background-color: #327985 !important;
    color: #FFFFFF !important;
}

.stButton > button:focus *,
.stDownloadButton > button:focus *,
.stLinkButton > a:focus * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"], button[data-baseweb="tab"] p { color: #526575 !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] p { color: #3F8F9D !important; font-weight: 700; }

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] { border-radius: 10px; }
div[data-testid="stAlert"] p { color: #12344D !important; }

/* ---------- Risk labels & pills ---------- */
.risk-high { color: #D9535B !important; font-weight: 800; font-size: 1.4rem; }
.risk-medium { color: #F47545 !important; font-weight: 800; font-size: 1.4rem; }
.risk-low { color: #238B5B !important; font-weight: 800; font-size: 1.4rem; }
.risk-pill { display: inline-block; border-radius: 999px; padding: 6px 16px; font-weight: 700; font-size: 0.95rem; }
.risk-pill-high { background-color: #FBE4E5; color: #B4232D !important; }
.risk-pill-medium { background-color: #FEE9DE; color: #C45722 !important; }
.risk-pill-low { background-color: #DEF3E7; color: #187447 !important; }

/* ---------- File uploader ---------- */
section[data-testid="stFileUploaderDropzone"] { background-color: #DCECEF !important; border: none !important; border-radius: 12px; }
section[data-testid="stFileUploaderDropzone"] { color: #12344D !important; }
section[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploaderDropzone"] div,
section[data-testid="stFileUploaderDropzone"] p {
    color: #12344D !important;
    font-weight: 600 !important;
}
/* Browse/Upload button: solid teal with white text, matches the rest of the app's buttons */
section[data-testid="stFileUploaderDropzone"] button {
    background-color: #3F8F9D !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}
section[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploaderDropzone"] button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #327985 !important;
}

/* ---------- Dataframe / expanders / dividers ---------- */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
details[data-testid="stExpander"] { background-color: #FFFFFF; border: 1px solid #DCE5E8; border-radius: 10px; }
details[data-testid="stExpander"] summary p { color: #12344D !important; }
hr { border-color: #DCECEF !important; }

/* ---------- Overview dashboard screenshots ---------- */
.section-title {
    color: #12344D !important;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 5px;
}
.section-description {
    color: #526575 !important;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 12px;
}
.dashboard-card {
    background-color: #FFFFFF;
    border: 1px solid #DCE5E8;
    border-radius: 12px 12px 0 0;
    padding: 14px 18px;
    margin-top: 12px;
    margin-bottom: 0;
    box-shadow: 0 3px 10px rgba(18, 52, 77, 0.06);
}
.dashboard-card-title {
    color: #12344D !important;
    font-size: 1.05rem;
    font-weight: 800;
}
</style>
"""


# =========================================================
# SMALL UI HELPERS
# =========================================================

def render_card(html: str) -> None:
    """Render a block of HTML inside the app's standard white card."""
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def render_badges(labels: list[str]) -> str:
    """Build a row of pill-style badges from a list of labels."""
    return "".join(f'<span class="badge">{label}</span>' for label in labels)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def risk_bucket(probability: float) -> tuple[str, str, str]:
    """Map a probability to (label, text class, pill class)."""
    if probability >= 0.70:
        return "HIGH", "risk-high", "risk-pill-high"
    if probability >= 0.40:
        return "MEDIUM", "risk-medium", "risk-pill-medium"
    return "LOW", "risk-low", "risk-pill-low"


def get_feature_importance(model, feature_columns: list[str]) -> pd.DataFrame | None:
    """
    Best-effort extraction of feature importance or coefficients, so the
    Model Performance page works across common scikit-learn estimator types
    (including inside a Pipeline) without needing anything extra saved
    during training.
    """
    estimator = model
    if hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]

    try:
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
        else:
            return None

        if len(values) != len(feature_columns):
            return None

        return (
            pd.DataFrame({"feature": feature_columns, "importance": values})
            .sort_values("importance", ascending=False)
            .set_index("feature")
        )
    except Exception:
        return None


def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Derive month / day-of-week / day-of-month features from a date column."""
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["dayofmonth"] = df[date_col].dt.day
    return df


def classify_risk(probabilities: pd.Series) -> np.ndarray:
    """Vectorized version of risk_bucket for batch scoring."""
    return np.select(
        [probabilities >= 0.70, probabilities >= 0.40],
        ["High", "Medium"],
        default="Low",
    )


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py first.")
    return joblib.load(MODEL_PATH)


# =========================================================
# SIDEBAR
# =========================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🚚 Clearance Risk</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Logistics Prediction System</div>', unsafe_allow_html=True)

        page = st.radio("Navigate", NAV_PAGES, label_visibility="collapsed")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f'<a class="sidebar-footer-link" href="{GITHUB_URL}" target="_blank">💻 View on GitHub</a>', unsafe_allow_html=True)
        st.markdown(f'<a class="sidebar-footer-link" href="{LINKEDIN_URL}" target="_blank">💼 Connect on LinkedIn</a>', unsafe_allow_html=True)
        st.caption(f"Built by {APP_OWNER_NAME}")

    return page


# =========================================================
# PAGE: OVERVIEW
# =========================================================

def render_overview(regression_model, classification_model, feature_columns, risk_threshold: float) -> None:
    page_header(
        "Predictive Logistics Intelligence",
        "From Descriptive Analytics to Predictive Decision Support",
        "This application extends the logistics Power BI analysis with a machine learning "
        "layer that estimates customs clearance time and identifies shipments that may "
        f"exceed the {risk_threshold:.0f} day operational risk threshold.",
    )

    # =========================================================
    # MODEL SUMMARY
    # =========================================================

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Risk Threshold", f"{risk_threshold:.0f} days")
    k2.metric("Model Features", f"{len(feature_columns)}")
    k3.metric("Regression Model", type(regression_model).__name__)
    k4.metric("Classification Model", type(classification_model).__name__)

    st.write("")

    # =========================================================
    # PROJECT PURPOSE
    # =========================================================

    render_card(
        "<b>Why this project matters</b><br><br>"
        "The Power BI analysis provides visibility into historical shipment, carrier, "
        "customer, and trade performance. This application extends that analysis with "
        "a predictive machine learning layer.<br><br>"
        "Instead of only asking what happened, the predictive layer helps answer what "
        "may happen next by estimating customs clearance time and identifying shipments "
        "with a higher probability of exceeding the operational risk threshold.<br><br>"
        "The objective is to support earlier operational decisions, prioritize shipments "
        "that may require attention, and provide an additional layer of intelligence "
        "before potential clearance issues affect the supply chain."
    )

    st.write("")

    # =========================================================
    # POWER BI DASHBOARD SCREENSHOTS
    # =========================================================

    st.markdown(
        '<div class="section-title">Power BI Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'The predictive application is built on the same logistics analysis shown below. '
        'These dashboards provide the descriptive and diagnostic foundation for the '
        'machine learning solution.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    for image_path, caption in DASHBOARD_SCREENSHOTS:
        image_file = Path(image_path)

        if image_file.exists():
            st.markdown(
                f'<div class="dashboard-card"><div class="dashboard-card-title">{caption}</div></div>',
                unsafe_allow_html=True,
            )
            st.image(str(image_file), use_container_width=True)
            st.write("")
        else:
            st.error(
                f"Dashboard image not found: {image_path}. "
                "Make sure the image is inside the assets folder."
            )

    # =========================================================
    # CONNECTION BETWEEN POWER BI AND MACHINE LEARNING
    # =========================================================

    render_card(
        "<b>How the analytics and machine learning layers connect</b><br><br>"
        "<b>Power BI</b> provides descriptive and diagnostic analytics across the "
        "logistics operation.<br><br>"
        "<b>Machine Learning</b> uses historical shipment characteristics to estimate "
        "customs clearance time and calculate the probability of higher clearance risk.<br><br>"
        "<b>Streamlit</b> turns the model into an interactive decision support application "
        "where a user can evaluate an individual shipment or score multiple shipments at once.<br><br>"
        "<b>Operational Outcome</b> is a workflow that moves from understanding historical "
        "performance toward earlier identification of potential shipment risk."
    )

    st.write("")

    render_card(
        "<b>Project Objective</b><br><br>"
        "The goal is not to replace logistics or customs professionals. The goal is to "
        "give them an additional data driven signal that can help prioritize attention "
        "before a potential clearance issue becomes an operational problem."
    )


# =========================================================
# PAGE: SINGLE SHIPMENT
# =========================================================

def render_single_shipment(regression_model, classification_model, feature_columns, risk_threshold: float) -> None:
    page_header(
        "Predictive decision support",
        "Predict Clearance Risk",
        "Estimate expected customs clearance time and identify shipments with a higher "
        f"probability of exceeding the {risk_threshold:.0f}-day operational risk threshold.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        shipment_type = st.selectbox("Shipment Type", ["Export", "Import"])
        product_category = st.selectbox(
            "Product Category",
            ["Electronics", "Textiles", "Consumer Goods", "Industrial Equipment"],
        )
        origin = st.text_input("Origin City", "Mumbai")
        origin_country = st.text_input("Origin Country", "India")

    with c2:
        destination = st.text_input("Destination City", "New York")
        destination_country = st.text_input("Destination Country", "USA")
        shipment_value = st.number_input("Shipment Value (USD)", min_value=0.0, value=85000.0, step=1000.0)
        freight_cost = st.number_input("Freight Cost (USD)", min_value=0.0, value=4250.0, step=100.0)

    with c3:
        shipment_date = st.date_input("Planned Shipment Date")
        shipment_id = st.text_input("Shipment ID", "NEW-SHIPMENT")

    if st.button("Predict Clearance Risk", type="primary"):
        try:
            input_df = pd.DataFrame([{
                "shipment_id": shipment_id,
                "type": shipment_type,
                "date": pd.Timestamp(shipment_date),
                "product_category": product_category,
                "origin": origin,
                "O_Country": origin_country,
                "destination": destination,
                "D_Country": destination_country,
                "value": shipment_value,
                "freight_cost": freight_cost,
            }])
            input_df = add_date_features(input_df)
            model_input = input_df[feature_columns]

            predicted_days = float(regression_model.predict(model_input)[0])
            high_risk_probability = float(classification_model.predict_proba(model_input)[0, 1])
            risk_label, _, pill_class = risk_bucket(high_risk_probability)
            value_at_risk = shipment_value * high_risk_probability

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Clearance", f"{predicted_days:.2f} days")
            m2.metric("High Risk Probability", f"{high_risk_probability:.1%}")
            m3.metric("Shipment Value", f"${shipment_value:,.0f}")
            m4.metric("Estimated Value at Risk", f"${value_at_risk:,.0f}")

            st.markdown(f'<span class="risk-pill {pill_class}">Risk Level: {risk_label}</span>', unsafe_allow_html=True)
            st.write("")
            st.progress(min(max(high_risk_probability, 0.0), 1.0), text="High-risk probability")
            st.caption(
                "This probability is a decision-support signal based on the shipment "
                "characteristics provided. It should be considered alongside customs "
                "documentation, route information, and operational judgment."
            )

            if risk_label == "HIGH":
                st.warning(
                    "Operational recommendation: prioritize this shipment for review. "
                    "Check documentation completeness, customs readiness, and "
                    "route-specific constraints before processing."
                )
            elif risk_label == "MEDIUM":
                st.info(
                    "Operational recommendation: monitor this shipment more closely and "
                    "review customs readiness before processing."
                )
            else:
                st.success(
                    "Operational signal: this shipment is currently classified as low "
                    "clearance risk based on the model."
                )

            st.session_state.prediction_history.insert(0, {
                "shipment_id": shipment_id,
                "destination_country": destination_country,
                "predicted_days": round(predicted_days, 2),
                "risk_probability": round(high_risk_probability, 3),
                "risk_level": risk_label,
                "value_at_risk": round(value_at_risk, 2),
            })

            result_csv = pd.DataFrame([st.session_state.prediction_history[0]]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download This Prediction (CSV)",
                data=result_csv,
                file_name=f"{shipment_id}_prediction.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Couldn't generate a prediction for this input: {exc}")

    if st.session_state.prediction_history:
        st.divider()
        st.subheader("Session Prediction History")
        st.caption("Predictions generated during the current app session.")
        st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)


# =========================================================
# PAGE: BATCH PREDICTION
# =========================================================

def render_batch_prediction(regression_model, classification_model, feature_columns) -> None:
    page_header(
        "Operational scoring",
        "Batch Prediction",
        "Score multiple shipments at once, identify higher-risk movements, estimate "
        "potential value exposure, and export the results back to Power BI for monitoring.",
    )

    st.write(
        "Upload shipment records using the same input fields used during model training. "
        "The historical target `customs_clearance_time_days` is not required because the "
        "model generates the prediction."
    )

    uploaded = st.file_uploader("Upload Shipment CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        batch = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        batch["date"] = pd.to_datetime(batch["date"], dayfirst=True, errors="coerce")
        if batch["date"].isna().any():
            st.error("Some dates could not be parsed.")
            return

        batch = add_date_features(batch)
        model_input = batch[feature_columns]

        batch["predicted_clearance_days"] = regression_model.predict(model_input)
        batch["high_clearance_probability"] = classification_model.predict_proba(model_input)[:, 1]
        batch["clearance_risk"] = classify_risk(batch["high_clearance_probability"])
        batch["value_at_risk_usd"] = batch["value"] * batch["high_clearance_probability"]

        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Shipments Scored", f"{len(batch):,}")
        k2.metric("High Risk", int((batch["clearance_risk"] == "High").sum()))
        k3.metric("Avg Predicted Days", f"{batch['predicted_clearance_days'].mean():.1f}")
        k4.metric("Total Value at Risk", f"${batch['value_at_risk_usd'].sum():,.0f}")

        st.subheader("Shipment Risk Results")
        st.dataframe(
            batch[[
                "shipment_id", "D_Country", "value", "predicted_clearance_days",
                "high_clearance_probability", "clearance_risk", "value_at_risk_usd",
            ]],
            use_container_width=True,
        )

        st.download_button(
            "Download Predictions for Power BI",
            data=batch.to_csv(index=False).encode("utf-8"),
            file_name="shipment_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Couldn't process this file: {exc}")


# =========================================================
# PAGE: MODEL PERFORMANCE
# =========================================================

def render_model_card(title: str, description: str, model) -> None:
    render_card(
        f"<b>{title}</b><br>{description}<br><br>"
        f'<span class="badge">{type(model).__name__}</span>'
    )
    with st.expander("View Hyperparameters"):
        try:
            st.json(model.get_params())
        except Exception:
            st.write("Not available for this model type.")


def render_feature_importance_column(label: str, model, feature_columns) -> None:
    st.write(f"**{label}**")
    importance = get_feature_importance(model, feature_columns)
    if importance is not None:
        st.bar_chart(importance)
    else:
        st.info("Feature importance isn't available for this model type.")


def render_model_performance(regression_model, classification_model, feature_columns) -> None:
    page_header(
        "Model transparency",
        "Model Performance & Explainability",
        "Understand the models behind the predictions and the shipment attributes "
        "that contribute to their decisions.",
    )

    c1, c2 = st.columns(2)
    with c1:
        render_model_card("Regression Model", "Predicts expected customs clearance time", regression_model)
    with c2:
        render_model_card("Classification Model", "Predicts high-clearance-risk probability", classification_model)

    st.divider()
    st.subheader("Feature Importance")
    st.caption("Understanding feature importance helps connect model predictions to the operational characteristics of a shipment.")

    fc1, fc2 = st.columns(2)
    with fc1:
        render_feature_importance_column("Regression Model", regression_model, feature_columns)
    with fc2:
        render_feature_importance_column("Classification Model", classification_model, feature_columns)

    st.divider()
    render_card(
        "<b>How to interpret this page</b><br><br>"
        "Feature importance indicates which available shipment attributes contribute most "
        "to the model's predictions. It should be interpreted as a model explanation rather "
        "than proof that an individual feature causes customs clearance delays."
    )


# =========================================================
# PAGE: ABOUT
# =========================================================

def render_about() -> None:
    page_header(
        "Project story",
        "About This Project",
        "A logistics analytics project that connects business intelligence, machine "
        "learning, and operational decision support.",
    )

    render_card(
        "<b>What this project solves</b><br><br>"
        "Logistics teams can discover shipment problems only after delays have already "
        "affected operations. This project explores how machine learning can move "
        "logistics analytics from retrospective reporting toward proactive decision "
        "support.<br><br>"
        "The Power BI layer provides visibility into shipment, carrier, customer, and "
        "trade performance. The predictive layer then estimates customs clearance time "
        "and identifies shipments with a higher probability of exceeding the operational "
        "risk threshold before processing.<br><br>"
        "The goal is not to replace operational judgment. The goal is to help teams "
        "identify where attention may be needed earlier."
    )

    render_card(
        "<b>Business Value</b><br><br>"
        + render_badges([
            "Early Risk Detection", "Operational Prioritization",
            "Predictive Decision Support", "Value Exposure", "Power BI Integration",
        ])
        + "<br><br>Instead of waiting for a shipment to experience a clearance problem, "
        "the predictive layer provides an early risk signal that can help operational "
        "teams decide which shipments deserve closer review."
    )

    render_card(
        "<b>Project Highlights</b><br><br>"
        "• <b>Analytics to ML:</b> extends an existing Power BI logistics analysis with a "
        "predictive machine-learning layer.<br><br>"
        "• <b>Dual-model approach:</b> regression estimates expected customs clearance "
        "time while classification produces a high-risk probability.<br><br>"
        "• <b>Operational decision support:</b> converts model output into risk levels "
        "and estimated value exposure to help prioritize shipment review.<br><br>"
        "• <b>Batch scoring:</b> allows multiple shipments to be scored at once and the "
        "results exported back into Power BI.<br><br>"
        "• <b>Explainability:</b> surfaces model feature importance to help users "
        "understand which shipment attributes contribute most to predictions.<br><br>"
        "• <b>End-to-end workflow:</b> demonstrates the journey from data analysis and "
        "model development through model persistence, application development, and "
        "deployment."
    )

    render_card(
        "<b>Technology Stack</b><br><br>"
        + render_badges(["Python", "Pandas", "NumPy", "Scikit-learn", "Joblib", "Streamlit", "Power BI"])
    )

    render_card(
        "<b>End-to-End Architecture</b><br><br>"
        "<b>Historical Shipment Data</b> → <b>Power BI Analysis</b> → "
        "<b>Machine Learning Model</b> → <b>Risk Prediction</b> → "
        "<b>Streamlit Application</b> → <b>Power BI Monitoring</b><br><br>"
        "The project demonstrates how descriptive analytics can be extended into "
        "predictive intelligence without separating the machine-learning solution from "
        "the existing business intelligence workflow."
    )

    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        st.link_button("💻 View the Code on GitHub", GITHUB_URL, use_container_width=True)
    with b2:
        st.link_button("💼 Connect on LinkedIn", LINKEDIN_URL, use_container_width=True)

    st.caption(f"Built by {APP_OWNER_NAME}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    load_error = None
    try:
        bundle = load_model_bundle()
    except Exception as exc:
        bundle = None
        load_error = str(exc)

    page = render_sidebar()

    if bundle is None:
        st.error(
            f"Couldn't load the model bundle from `{MODEL_PATH}`. "
            f"Run `train_model.py` first, then reload this app.\n\nDetails: {load_error}"
        )
        st.stop()

    regression_model = bundle["regression_model"]
    classification_model = bundle["classification_model"]
    risk_threshold = bundle["risk_threshold_days"]
    feature_columns = bundle["feature_columns"]

    if page == "🏠 Overview":
        render_overview(regression_model, classification_model, feature_columns, risk_threshold)
    elif page == "📦 Single Shipment":
        render_single_shipment(regression_model, classification_model, feature_columns, risk_threshold)
    elif page == "📊 Batch Prediction":
        render_batch_prediction(regression_model, classification_model, feature_columns)
    elif page == "🧠 Model Performance":
        render_model_performance(regression_model, classification_model, feature_columns)
    elif page == "👤 About":
        render_about()


main()