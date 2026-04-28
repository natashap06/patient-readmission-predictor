import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import os
import gdown

# ── Download model from Google Drive if not present ────────────────────────
MODEL_PATH = "rf_model.pkl"
FILE_ID    = "1CXNztkeZqFOhNCJsDOOhxp3OhQxUCUBs"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --bg:           #080d14;
        --bg2:          #0d1520;
        --bg3:          #111c2b;
        --border:       rgba(56,139,253,0.18);
        --border-glow:  rgba(56,139,253,0.45);
        --blue:         #388bfd;
        --blue-dim:     #1e4a8a;
        --cyan:         #39d0f0;
        --green:        #23d18b;
        --amber:        #f5a623;
        --red:          #ff5f5f;
        --text:         #e2eaf4;
        --text-muted:   #6b8099;
        --text-label:   #4a6785;
        --card:         rgba(13,21,32,0.85);
        --glow-blue:    0 0 30px rgba(56,139,253,0.15);
        --glow-cyan:    0 0 30px rgba(57,208,240,0.15);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .main { padding: 0 !important; background: var(--bg) !important; }
    .block-container { padding: 0 2rem 3rem !important; max-width: 1280px !important; }

    #MainMenu, footer, header { visibility: hidden; }

    /* ── Animated grid background ── */
    .bg-grid {
        position: fixed;
        inset: 0;
        z-index: 0;
        background-image:
            linear-gradient(rgba(56,139,253,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56,139,253,0.04) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
    }

    /* ── Top Hero Banner ── */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0a1628 0%, #0d1f38 50%, #091422 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.2rem 2.5rem 2rem;
        margin-bottom: 2rem;
        margin-top: 1.5rem;
        box-shadow: var(--glow-blue), inset 0 1px 0 rgba(56,139,253,0.12);
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(56,139,253,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(57,208,240,0.07) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-inner {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 20px;
    }

    .hero-icon {
        width: 60px; height: 60px;
        background: linear-gradient(135deg, rgba(56,139,253,0.3), rgba(57,208,240,0.2));
        border: 1px solid var(--border-glow);
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 26px;
        box-shadow: 0 0 20px rgba(56,139,253,0.25);
        flex-shrink: 0;
    }

    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: .2em;
        text-transform: uppercase;
        color: var(--cyan);
        margin-bottom: 6px;
        opacity: 0.85;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .hero-sub {
        font-size: 12.5px;
        color: var(--text-muted);
        font-weight: 400;
    }

    .hero-badge {
        margin-left: auto;
        flex-shrink: 0;
        background: rgba(35,209,139,0.1);
        border: 1px solid rgba(35,209,139,0.3);
        border-radius: 20px;
        padding: 6px 14px;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        color: var(--green);
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    /* ── Section headers ── */
    .section-head {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
        margin-top: 4px;
    }

    .section-head-line {
        width: 3px; height: 18px;
        background: linear-gradient(to bottom, var(--blue), var(--cyan));
        border-radius: 2px;
        flex-shrink: 0;
    }

    .section-head-text {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: .15em;
        color: var(--blue);
    }

    /* ── Input group cards ── */
    .input-group {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.4rem 1.3rem 1rem;
        height: 100%;
        backdrop-filter: blur(10px);
        transition: border-color .3s;
    }

    .input-group:hover {
        border-color: var(--border-glow);
    }

    .group-label {
        font-family: 'DM Mono', monospace;
        font-size: 9.5px;
        font-weight: 500;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: var(--text-label);
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 7px;
    }

    /* ── Streamlit widget overrides ── */
    .stNumberInput input,
    .stTextInput input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(56,139,253,0.2) !important;
        border-radius: 9px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        padding: 6px 10px !important;
        transition: border-color .2s !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(56,139,253,0.12) !important;
        outline: none !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(56,139,253,0.2) !important;
        border-radius: 9px !important;
        color: var(--text) !important;
        font-size: 13px !important;
    }

    label, .stNumberInput label, .stSelectbox label {
        color: var(--text-muted) !important;
        font-size: 11.5px !important;
        font-weight: 500 !important;
        letter-spacing: .02em !important;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.8rem 0;
    }

    /* ── Predict Button ── */
    .stButton > button {
        width: 100% !important;
        padding: 0.85rem 1rem !important;
        background: linear-gradient(135deg, #1a56db, #388bfd) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: .03em !important;
        cursor: pointer !important;
        transition: all .25s !important;
        box-shadow: 0 4px 20px rgba(56,139,253,0.35) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1e4ddf, #4a9aff) !important;
        box-shadow: 0 6px 28px rgba(56,139,253,0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Result boxes ── */
    .result-box {
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }

    .result-box::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        pointer-events: none;
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(35,209,139,0.08), rgba(35,209,139,0.04));
        border: 1px solid rgba(35,209,139,0.35);
        box-shadow: 0 0 30px rgba(35,209,139,0.08), inset 0 1px 0 rgba(35,209,139,0.15);
    }

    .result-warning {
        background: linear-gradient(135deg, rgba(245,166,35,0.09), rgba(245,166,35,0.04));
        border: 1px solid rgba(245,166,35,0.35);
        box-shadow: 0 0 30px rgba(245,166,35,0.08), inset 0 1px 0 rgba(245,166,35,0.15);
    }

    .result-danger {
        background: linear-gradient(135deg, rgba(255,95,95,0.09), rgba(255,95,95,0.04));
        border: 1px solid rgba(255,95,95,0.35);
        box-shadow: 0 0 30px rgba(255,95,95,0.10), inset 0 1px 0 rgba(255,95,95,0.15);
    }

    .result-header {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 10px;
    }

    .result-emoji {
        font-size: 28px;
        line-height: 1;
        flex-shrink: 0;
    }

    .result-title-safe    { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: #23d18b; margin-bottom: 4px; }
    .result-title-warning { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: #f5a623; margin-bottom: 4px; }
    .result-title-danger  { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: #ff5f5f; margin-bottom: 4px; }

    .result-tag-safe    { display: inline-block; background: rgba(35,209,139,0.15); color: #23d18b; border-radius: 20px; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .15em; padding: 3px 10px; text-transform: uppercase; margin-bottom: 10px; }
    .result-tag-warning { display: inline-block; background: rgba(245,166,35,0.15); color: #f5a623; border-radius: 20px; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .15em; padding: 3px 10px; text-transform: uppercase; margin-bottom: 10px; }
    .result-tag-danger  { display: inline-block; background: rgba(255,95,95,0.15);  color: #ff5f5f; border-radius: 20px; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .15em; padding: 3px 10px; text-transform: uppercase; margin-bottom: 10px; }

    .result-advice {
        font-size: 13.5px;
        color: rgba(226,234,244,0.75);
        line-height: 1.7;
    }

    /* ── Probability Metric Cards ── */
    .prob-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem 1rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: border-color .3s;
    }

    .prob-card:hover { border-color: var(--border-glow); }

    .prob-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }

    .prob-card-green::before  { background: linear-gradient(90deg, var(--green), #1ea876); }
    .prob-card-amber::before  { background: linear-gradient(90deg, var(--amber), #e07b00); }
    .prob-card-red::before    { background: linear-gradient(90deg, var(--red), #c93030); }

    .prob-number-green  { font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 700; color: var(--green); margin-bottom: 5px; line-height: 1; }
    .prob-number-amber  { font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 700; color: var(--amber); margin-bottom: 5px; line-height: 1; }
    .prob-number-red    { font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 700; color: var(--red);   margin-bottom: 5px; line-height: 1; }

    .prob-label {
        font-family: 'DM Mono', monospace;
        font-size: 9.5px;
        font-weight: 500;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: var(--text-label);
    }

    /* ── Custom progress bars ── */
    .prog-wrap { margin-bottom: 14px; }

    .prog-meta {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .prog-name {
        font-size: 12px;
        font-weight: 500;
        color: var(--text-muted);
    }

    .prog-pct {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: var(--text-muted);
    }

    .prog-track {
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        overflow: hidden;
    }

    .prog-fill-green { height: 100%; background: linear-gradient(90deg, #1ea876, var(--green)); border-radius: 10px; }
    .prog-fill-amber { height: 100%; background: linear-gradient(90deg, #e07b00, var(--amber)); border-radius: 10px; }
    .prog-fill-red   { height: 100%; background: linear-gradient(90deg, #c93030, var(--red));   border-radius: 10px; }

    /* ── Summary Cards ── */
    .summary-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 11px 13px;
        margin-bottom: 9px;
        transition: background .2s, border-color .2s;
    }

    .summary-card:hover {
        background: rgba(56,139,253,0.05);
        border-color: rgba(56,139,253,0.3);
    }

    .summary-key {
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        font-weight: 500;
        letter-spacing: .15em;
        text-transform: uppercase;
        color: var(--text-label);
        margin-bottom: 4px;
    }

    .summary-val {
        font-size: 14px;
        font-weight: 600;
        color: var(--text);
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        color: var(--text-label);
        letter-spacing: .1em;
        padding: 2rem 0 1rem;
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
        text-transform: uppercase;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--blue) !important; }

    /* Progress widget hide (we use custom) */
    div[data-testid="stProgressBar"] { display: none; }

    /* ── Responsive column padding ── */
    [data-testid="column"] { padding: 0 6px !important; }
</style>

<div class="bg-grid"></div>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf   = joblib.load("rf_model.pkl")
    sc   = joblib.load("scaler.pkl")
    with open("feature_names.json") as f:
        features = json.load(f)
    return rf, sc, features

rf_model, scaler, feature_names = load_model()


# ── Hero Banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-icon">🏥</div>
    <div>
      <div class="hero-eyebrow">clinical decision support</div>
      <div class="hero-title">Patient Readmission Predictor</div>
      <div class="hero-sub">Diabetic patient risk assessment powered by Random Forest</div>
    </div>
    <div class="hero-badge">● Model Online</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Section Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-head-line"></div>
  <div class="section-head-text">Patient Clinical Details</div>
</div>
""", unsafe_allow_html=True)


# ── Input Form ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="input-group"><div class="group-label">👤 Patient Info</div>', unsafe_allow_html=True)
    age              = st.number_input("Age (years)",             min_value=0,   max_value=100, value=65,  step=1)
    time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1,   max_value=14,  value=3,   step=1)
    number_diagnoses = st.number_input("Number of Diagnoses",     min_value=1,   max_value=20,  value=5,   step=1)
    gender           = st.selectbox("Gender", ["Female", "Male"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-group"><div class="group-label">🔬 Lab & Procedures</div>', unsafe_allow_html=True)
    num_lab_procedures = st.number_input("Lab Procedures",  min_value=0, max_value=200, value=40,  step=1)
    num_procedures     = st.number_input("Procedures Done", min_value=0, max_value=20,  value=2,   step=1)
    num_medications    = st.number_input("Medications",     min_value=0, max_value=100, value=12,  step=1)
    A1Cresult          = st.selectbox("HbA1c Result", ["Not measured", "Normal", ">7", ">8"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-group"><div class="group-label">🏥 Visit History</div>', unsafe_allow_html=True)
    number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=50, value=1, step=1)
    number_emergency  = st.number_input("Emergency Visits",  min_value=0, max_value=50, value=0, step=1)
    number_inpatient  = st.number_input("Inpatient Visits",  min_value=0, max_value=50, value=1, step=1)
    diag_1            = st.selectbox("Primary Diagnosis", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes",
        "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="input-group"><div class="group-label">💊 Medications</div>', unsafe_allow_html=True)
    insulin     = st.selectbox("Insulin",              ["No", "Steady", "Up", "Down"])
    metformin   = st.selectbox("Metformin",            ["No", "Steady", "Up", "Down"])
    change      = st.selectbox("Medication Change?",   ["No change", "Change made"])
    diabetesMed = st.selectbox("Diabetes Medication?", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Encode Inputs ──────────────────────────────────────────────────────────
def encode_inputs():
    med_map  = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    a1c_map  = {"Not measured": 0, "Normal": 1, ">7": 2, ">8": 3}
    diag_map = {
        "Circulatory": 0, "Respiratory": 1, "Digestive": 2,
        "Diabetes": 3, "Injury": 4, "Musculoskeletal": 5,
        "Genitourinary": 6, "Neoplasms": 7, "Other": 8
    }
    input_data = {feat: 0 for feat in feature_names}
    input_data.update({
        "age": age, "time_in_hospital": time_in_hospital,
        "number_diagnoses": number_diagnoses,
        "gender": 1 if gender == "Male" else 0,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "A1Cresult": a1c_map[A1Cresult],
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_map[diag_1],
        "insulin": med_map[insulin],
        "metformin": med_map[metformin],
        "change": 1 if change == "Change made" else 0,
        "diabetesMed": 1 if diabetesMed == "Yes" else 0,
        "total_visits": number_outpatient + number_emergency + number_inpatient,
        "med_changed": 1 if change == "Change made" else 0,
        "polypharmacy": 1 if num_medications > 10 else 0,
        "lab_proc_ratio": num_lab_procedures / (num_procedures + 1),
    })
    return pd.DataFrame([input_data])


# ── Predict Button ─────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("▶  Predict Readmission Risk")

if predict_clicked:
    with st.spinner("Analysing patient data..."):
        input_df     = encode_inputs()
        input_scaled = scaler.transform(input_df)
        prediction   = rf_model.predict(input_scaled)[0]
        probabilities = rf_model.predict_proba(input_scaled)[0]

    p0 = round(float(probabilities[0]) * 100, 1)
    p1 = round(float(probabilities[1]) * 100, 1)
    p2 = round(float(probabilities[2]) * 100, 1)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
      <div class="section-head-line"></div>
      <div class="section-head-text">Prediction Result</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Result Box ─────────────────────────────────────────────────────────
    if prediction == 0:
        st.markdown("""
        <div class="result-box result-safe">
            <div class="result-header">
                <div class="result-emoji">✅</div>
                <div>
                    <div class="result-title-safe">Not Readmitted</div>
                    <div class="result-tag-safe">Low Risk</div>
                    <div class="result-advice">Patient is unlikely to return soon. Standard discharge protocol recommended. Schedule a routine follow-up in 4–6 weeks.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown("""
        <div class="result-box result-warning">
            <div class="result-header">
                <div class="result-emoji">⚠️</div>
                <div>
                    <div class="result-title-warning">Readmitted After 30 Days</div>
                    <div class="result-tag-warning">Moderate Risk</div>
                    <div class="result-advice">Patient may return after 30 days. Schedule a follow-up appointment within 2–4 weeks. Review medication adherence and lifestyle factors.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-box result-danger">
            <div class="result-header">
                <div class="result-emoji">🚨</div>
                <div>
                    <div class="result-title-danger">Readmitted Within 30 Days</div>
                    <div class="result-tag-danger">High Risk — Immediate Action</div>
                    <div class="result-advice">High-risk patient. Immediate care plan is strongly advised. Coordinate with the clinical team before discharge and arrange intensive monitoring.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability Metrics ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head" style="margin-top:1.4rem">
      <div class="section-head-line"></div>
      <div class="section-head-text">Probability Breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        st.markdown(f"""
        <div class="prob-card prob-card-green">
            <div class="prob-number-green">{p0}%</div>
            <div class="prob-label">Not Readmitted</div>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="prob-card prob-card-amber">
            <div class="prob-number-amber">{p1}%</div>
            <div class="prob-label">After 30 Days</div>
        </div>
        """, unsafe_allow_html=True)

    with mc3:
        st.markdown(f"""
        <div class="prob-card prob-card-red">
            <div class="prob-number-red">{p2}%</div>
            <div class="prob-label">Within 30 Days</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Custom Progress Bars ───────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top: 1.4rem;">
        <div class="prog-wrap">
            <div class="prog-meta">
                <span class="prog-name">Not Readmitted</span>
                <span class="prog-pct">{p0}%</span>
            </div>
            <div class="prog-track"><div class="prog-fill-green" style="width:{p0}%"></div></div>
        </div>
        <div class="prog-wrap">
            <div class="prog-meta">
                <span class="prog-name">Readmitted After 30 Days</span>
                <span class="prog-pct">{p1}%</span>
            </div>
            <div class="prog-track"><div class="prog-fill-amber" style="width:{p1}%"></div></div>
        </div>
        <div class="prog-wrap">
            <div class="prog-meta">
                <span class="prog-name">Readmitted Within 30 Days</span>
                <span class="prog-pct">{p2}%</span>
            </div>
            <div class="prog-track"><div class="prog-fill-red" style="width:{p2}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Patient Summary ────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-head">
      <div class="section-head-line"></div>
      <div class="section-head-text">Patient Summary</div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    def card(key, val):
        return f'<div class="summary-card"><div class="summary-key">{key}</div><div class="summary-val">{val}</div></div>'

    with s1:
        st.markdown(card("Age", f"{age} years") + card("Gender", gender) + card("Time in Hospital", f"{time_in_hospital} days"), unsafe_allow_html=True)

    with s2:
        st.markdown(card("Lab Procedures", num_lab_procedures) + card("Medications", num_medications) + card("HbA1c", A1Cresult), unsafe_allow_html=True)

    with s3:
        st.markdown(card("Emergency Visits", number_emergency) + card("Inpatient Visits", number_inpatient) + card("Total Visits", number_outpatient + number_emergency + number_inpatient), unsafe_allow_html=True)

    with s4:
        st.markdown(card("Insulin", insulin) + card("Primary Diagnosis", diag_1) + card("Med Change", change), unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Diabetic Patient Readmission Predictor &nbsp;·&nbsp;
    Random Forest Model &nbsp;·&nbsp;
    For Clinical Support Use Only
</div>
""", unsafe_allow_html=True)