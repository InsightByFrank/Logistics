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
# EDIT THESE — a few personal touches for the "About" page
# and sidebar footer. Nothing else in the file needs editing
# to get it running.
# =========================================================
APP_OWNER_NAME = "InsightbyFrank"                                   # TODO
LINKEDIN_URL = "http://www.linkedin.com/in/frank-agba"      # TODO
GITHUB_URL = ""      # TODO
POWERBI_EMBED_URL = ""                                          # TODO (optional) — a Power BI "Publish to web" link
DASHBOARD_SCREENSHOTS = [
    ("assets/dashboard_trade_flow.png", "Shipment & Trade Flow"),
    ("assets/dashboard_carrier_performance.png", "Logistics & Carrier Performance"),
    ("assets/dashboard_customer_performance.png", "Customer & Order Performance"),
]  # Drop your 3 exported PNGs into an /assets folder next to this file, using these names
   # (or edit the paths above to match your own filenames).

MODEL_PATH = Path("model/logistics_clearance_models.joblib")

st.set_page_config(
    page_title="Logistics Clearance Risk",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME — same palette as the Power BI dashboard:
# navy #12344D, teal #3F8F9D, cream background #F3F6F7
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ==============================
   APP BACKGROUND
   ============================== */
.stApp {
    background-color: #F3F6F7;
}

/* ==============================
   SIDEBAR — styled like the dark
   navy icon rail in the dashboard
   ============================== */
section[data-testid="stSidebar"] {
    background-color: #12344D;
    border-right: 1px solid #0D2739;
}
section[data-testid="stSidebar"] * {
    color: #E7EEF1 !important;
}
section[data-testid="stSidebar"] .sidebar-brand {
    font-size: 1.15rem;
    font-weight: 800;
    color: #FFFFFF !important;
    padding: 0.25rem 0 0.1rem 0;
}
section[data-testid="stSidebar"] .sidebar-subtitle {
    font-size: 0.78rem;
    color: #9FB4C2 !important;
    margin-bottom: 1.1rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    border: 1px solid transparent;
    transition: background-color 0.15s ease;
    width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(63,143,157,0.25);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background-color: #3F8F9D;
    border: 1px solid #6FB8C4;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}
.sidebar-footer-link {
    display: block;
    text-align: center;
    background-color: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 8px;
    padding: 7px 10px;
    margin-top: 6px;
    font-size: 0.83rem;
    font-weight: 600;
    text-decoration: none !important;
    color: #E7EEF1 !important;
}
.sidebar-footer-link:hover {
    background-color: #3F8F9D;
    border-color: #3F8F9D;
}

/* ==============================
   PAGE HEADER
   ============================== */
.page-eyebrow {
    color: #3F8F9D;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.main-title {
    color: #12344D;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.2;
}
.main-subtitle {
    color: #667780;
    font-size: 0.98rem;
    margin-top: 2px;
}

/* Subtitle and normal text */
p { color: #526575; }

/* ==============================
   KPI CARDS
   ============================== */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E1E8EB;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 3px 10px rgba(18, 52, 77, 0.08);
}
div[data-testid="stMetricLabel"] { color: #667780; }
div[data-testid="stMetricValue"] { color: #12344D; font-weight: 700; }

/* ==============================
   GENERIC CARD / BADGE
   ============================== */
.card {
    background-color: #FFFFFF;
    border: 1px solid #E1E8EB;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 3px 10px rgba(18, 52, 77, 0.06);
    margin-bottom: 14px;
}
.badge {
    display: inline-block;
    background-color: #EAF4F6;
    color: #226B77;
    border: 1px solid #CFE7EB;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 6px 3px 0;
}

/* ==============================
   BUTTON
   ============================== */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    background-color: #3F8F9D;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {
    background-color: #327985;
    color: white;
}

/* ==============================
   INPUT FIELDS
   ============================== */
div[data-baseweb="select"] > div {
    border-radius: 8px;
    border: 1px solid #D6E1E5;
}
input { border-radius: 8px !important; }

/* ==============================
   INFORMATION BOX
   ============================== */
div[data-testid="stAlert"] { border-radius: 10px; }

/* ==============================
   RISK LABELS
   ============================== */
.risk-high { color: #D9535B; font-weight: 800; font-size: 1.4rem; }
.risk-medium { color: #F47545; font-weight: 800; font-size: 1.4rem; }
.risk-low { color: #238B5B; font-weight: 800; font-size: 1.4rem; }

.risk-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 6px 16px;
    font-weight: 700;
    font-size: 0.95rem;
}
.risk-pill-high { background-color: #FBE4E5; color: #D9535B; }
.risk-pill-medium { background-color: #FEE9DE; color: #F47545; }
.risk-pill-low { background-color: #DEF3E7; color: #238B5B; }

/* ==============================
   TABS
   ============================== */
button[data-baseweb="tab"] { color: #526575; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: #3F8F9D; font-weight: 700; }

/* ==============================
   FILE UPLOADER
   ============================== */
section[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF;
    border: 2px dashed #3F8F9D;
    border-radius: 12px;
}

/* ==============================
   DATAFRAME
   ============================== */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ==============================
   DIVIDERS
   ============================== */
hr { border-color: #DCECEF; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def risk_bucket(probability: float):
    if probability >= 0.70:
        return "HIGH", "risk-high", "risk-pill-high"
    elif probability >= 0.40:
        return "MEDIUM", "risk-medium", "risk-pill-medium"
    else:
        return "LOW", "risk-low", "risk-pill-low"


def get_feature_importance(model, feature_columns):
    """Best-effort extraction of feature importance/coefficients so the
    Model Performance page works across common scikit-learn estimator types
    (including inside a Pipeline), without requiring anything extra to be
    saved during training."""
    est = model
    if hasattr(est, "named_steps"):
        est = list(est.named_steps.values())[-1]

    try:
        if hasattr(est, "feature_importances_"):
            values = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            values = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
        else:
            return None
        if len(values) != len(feature_columns):
            return None
        return pd.DataFrame(
            {"feature": feature_columns, "importance": values}
        ).sort_values("importance", ascending=False).set_index("feature")
    except Exception:
        return None


try:
    bundle = load_models()
    MODEL_LOADED = True
    load_error = None
except Exception as exc:
    bundle = None
    MODEL_LOADED = False
    load_error = str(exc)

if MODEL_LOADED:
    regression_model = bundle["regression_model"]
    classification_model = bundle["classification_model"]
    risk_threshold = bundle["risk_threshold_days"]
    feature_columns = bundle["feature_columns"]

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🚚 Clearance Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Logistics Prediction System</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📦 Single Shipment", "📊 Batch Prediction", "🧠 Model Performance", "👤 About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f'<a class="sidebar-footer-link" href="{GITHUB_URL}" target="_blank">💻 View on GitHub</a>', unsafe_allow_html=True)
    st.markdown(f'<a class="sidebar-footer-link" href="{LINKEDIN_URL}" target="_blank">💼 Connect on LinkedIn</a>', unsafe_allow_html=True)
    st.caption(f"Built by {APP_OWNER_NAME}")

if not MODEL_LOADED:
    st.error(
        f"Couldn't load the model bundle from `{MODEL_PATH}`. "
        f"Run `train_model.py` first, then reload this app.\n\nDetails: {load_error}"
    )
    st.stop()


# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "🏠 Overview":
    page_header(
        "End-to-end logistics analytics",
        "From Power BI dashboard to predictive model",
        "This app extends a Power BI logistics dashboard with a machine-learning layer that "
        "predicts customs clearance time and flags high-risk shipments before they're processed."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk Threshold", f"{risk_threshold:.0f} days")
    k2.metric("Model Features", f"{len(feature_columns)}")
    k3.metric("Regression Model", type(regression_model).__name__)
    k4.metric("Classification Model", type(classification_model).__name__)

    st.write("")
    st.markdown(
        '<div class="card">'
        '<b>How it fits together</b><br>'
        'The Power BI dashboards below track historical shipment, carrier, and customer performance. '
        'This app closes the loop: predictions made here can be exported and fed back into Power BI '
        '(via the Batch Prediction CSV export) so at-risk shipments show up in the same reports '
        'stakeholders already use.'
        '</div>',
        unsafe_allow_html=True,
    )

    if POWERBI_EMBED_URL:
        st.components.v1.iframe(POWERBI_EMBED_URL, height=480)
    else:
        cols = st.columns(len(DASHBOARD_SCREENSHOTS))
        for col, (img_path, caption) in zip(cols, DASHBOARD_SCREENSHOTS):
            with col:
                if Path(img_path).exists():
                    st.image(img_path, caption=caption, use_container_width=True)
                else:
                    st.info(f"Add `{img_path}` to show your **{caption}** dashboard here.")

# =========================================================
# PAGE: SINGLE SHIPMENT
# =========================================================
elif page == "📦 Single Shipment":
    page_header(
        "Single shipment",
        "Predict Clearance Risk",
        f"Estimate clearance time and the probability of exceeding {risk_threshold:.0f} days."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        shipment_type = st.selectbox("Shipment Type", ["Export", "Import"])
        product_category = st.selectbox(
            "Product Category",
            ["Electronics", "Textiles", "Consumer Goods", "Industrial Equipment"]
        )
        origin = st.text_input("Origin City", "Mumbai")
        origin_country = st.text_input("Origin Country", "India")

    with c2:
        destination = st.text_input("Destination City", "New York")
        destination_country = st.text_input("Destination Country", "USA")
        shipment_value = st.number_input(
            "Shipment Value (USD)", min_value=0.0, value=85000.0, step=1000.0
        )
        freight_cost = st.number_input(
            "Freight Cost (USD)", min_value=0.0, value=4250.0, step=100.0
        )

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
                "freight_cost": freight_cost
            }])

            input_df["date"] = pd.to_datetime(input_df["date"])
            input_df["month"] = input_df["date"].dt.month
            input_df["dayofweek"] = input_df["date"].dt.dayofweek
            input_df["dayofmonth"] = input_df["date"].dt.day

            model_input = input_df[feature_columns]

            predicted_days = float(regression_model.predict(model_input)[0])
            high_risk_probability = float(classification_model.predict_proba(model_input)[0, 1])
            risk_label, risk_class, pill_class = risk_bucket(high_risk_probability)
            value_at_risk = shipment_value * high_risk_probability

            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Clearance", f"{predicted_days:.2f} days")
            m2.metric("High Risk Probability", f"{high_risk_probability:.1%}")
            m3.metric("Shipment Value", f"${shipment_value:,.0f}")
            m4.metric("Estimated Value at Risk", f"${value_at_risk:,.0f}")

            st.markdown(
                f'<span class="risk-pill {pill_class}">Risk Level: {risk_label}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.progress(min(max(high_risk_probability, 0.0), 1.0), text="High-risk probability")

            if risk_label == "HIGH":
                st.warning(
                    "Prioritize this shipment for operational review. "
                    "Consider checking documentation, customs readiness, "
                    "and route-specific constraints."
                )
            elif risk_label == "MEDIUM":
                st.info("Monitor this shipment more closely than a low-risk shipment.")
            else:
                st.success("The shipment is currently classified as low clearance risk.")

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
        st.subheader("Session prediction history")
        st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)

# =========================================================
# PAGE: BATCH PREDICTION
# =========================================================
elif page == "📊 Batch Prediction":
    page_header(
        "Bulk scoring",
        "Batch Prediction",
        "Upload a shipment CSV to score many shipments at once and export results back to Power BI."
    )

    st.write(
        "The CSV must contain the same shipment fields used during training. "
        "The target column `customs_clearance_time_days` is not required."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)

            required_batch = [
                "shipment_id", "type", "date", "product_category",
                "origin", "O_Country", "destination", "D_Country",
                "value", "freight_cost"
            ]
            missing = [c for c in required_batch if c not in batch.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                batch["date"] = pd.to_datetime(batch["date"], dayfirst=True, errors="coerce")

                if batch["date"].isna().any():
                    st.error("Some dates could not be parsed.")
                else:
                    batch["month"] = batch["date"].dt.month
                    batch["dayofweek"] = batch["date"].dt.dayofweek
                    batch["dayofmonth"] = batch["date"].dt.day

                    model_input = batch[feature_columns]

                    batch["predicted_clearance_days"] = regression_model.predict(model_input)
                    batch["high_clearance_probability"] = classification_model.predict_proba(model_input)[:, 1]
                    batch["clearance_risk"] = np.select(
                        [
                            batch["high_clearance_probability"] >= 0.70,
                            batch["high_clearance_probability"] >= 0.40
                        ],
                        ["High", "Medium"],
                        default="Low"
                    )
                    batch["value_at_risk_usd"] = batch["value"] * batch["high_clearance_probability"]

                    st.divider()
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Shipments Scored", f"{len(batch):,}")
                    k2.metric("High Risk", int((batch["clearance_risk"] == "High").sum()))
                    k3.metric("Avg Predicted Days", f"{batch['predicted_clearance_days'].mean():.1f}")
                    k4.metric("Total Value at Risk", f"${batch['value_at_risk_usd'].sum():,.0f}")

                    st.dataframe(
                        batch[[
                            "shipment_id", "D_Country", "value",
                            "predicted_clearance_days", "high_clearance_probability",
                            "clearance_risk", "value_at_risk_usd"
                        ]],
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Predictions for Power BI",
                        data=batch.to_csv(index=False).encode("utf-8"),
                        file_name="shipment_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as exc:
            st.error(f"Couldn't process this file: {exc}")

# =========================================================
# PAGE: MODEL PERFORMANCE
# =========================================================
elif page == "🧠 Model Performance":
    page_header(
        "Under the hood",
        "Model Performance & Explainability",
        "What the model is, what it was trained on, and which fields drive its predictions."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="card"><b>Regression model</b> (predicts clearance days)<br>'
            f'<span class="badge">{type(regression_model).__name__}</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Hyperparameters"):
            try:
                st.json(regression_model.get_params())
            except Exception:
                st.write("Not available for this model type.")

    with c2:
        st.markdown(
            f'<div class="card"><b>Classification model</b> (predicts high-risk probability)<br>'
            f'<span class="badge">{type(classification_model).__name__}</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Hyperparameters"):
            try:
                st.json(classification_model.get_params())
            except Exception:
                st.write("Not available for this model type.")

    st.divider()
    st.subheader("Feature importance")
    st.caption("Which shipment attributes influence the model's predictions the most.")

    fc1, fc2 = st.columns(2)
    with fc1:
        st.write("**Regression model**")
        importance = get_feature_importance(regression_model, feature_columns)
        if importance is not None:
            st.bar_chart(importance)
        else:
            st.info("Feature importance isn't available for this model type.")

    with fc2:
        st.write("**Classification model**")
        importance = get_feature_importance(classification_model, feature_columns)
        if importance is not None:
            st.bar_chart(importance)
        else:
            st.info("Feature importance isn't available for this model type.")

    st.divider()
    st.markdown(
        '<div class="card"><b>Notes</b><br>'
        'This page reads importance directly from the trained model objects, so it stays accurate '
        'as the model is retrained. If you log held-out metrics (MAE, RMSE, ROC-AUC, etc.) during '
        'training, save them into the joblib bundle (e.g. <code>bundle["metrics"]</code>) and surface '
        'them here for a fuller picture.</div>',
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE: ABOUT
# =========================================================
elif page == "👤 About":
    page_header(
        "Project story",
        "About This Project",
        "Why it exists, how it's built, and how to reach me."
    )

    st.markdown(
        f'''<div class="card">
        <b>What this is</b><br>
        A logistics analytics-to-prediction pipeline: a Power BI dashboard tracks shipment, carrier,
        and customer performance, and this Streamlit app adds a predictive layer that flags shipments
        likely to face long customs clearance — before they're processed, not after.
        </div>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''<div class="card">
        <b>Tech stack</b><br>
        <span class="badge">Python</span>
        <span class="badge">scikit-learn</span>
        <span class="badge">pandas</span>
        <span class="badge">Streamlit</span>
        <span class="badge">Power BI</span>
        <span class="badge">joblib</span>
        </div>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''<div class="card">
        <b>Highlights</b><br>
        • Dual-model setup: a regression model for expected clearance time and a classifier for
        high-risk probability, so users get both a number and a decision signal.<br>
        • Single-shipment and batch scoring, with batch results shaped for re-import into Power BI.<br>
        • Model performance page that reads feature importance straight from the trained models.
        </div>''',
        unsafe_allow_html=True,
    )

    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        st.link_button("💻 View the code on GitHub", GITHUB_URL, use_container_width=True)
    with b2:
        st.link_button("💼 Connect on LinkedIn", LINKEDIN_URL, use_container_width=True)

    st.caption(f"Built by {APP_OWNER_NAME}")