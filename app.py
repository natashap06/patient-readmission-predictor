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
    with st.spinner("⏳ Setting up... please wait a moment"):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediCheck — Patient Readmission",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Outfit:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #f0f4ff !important;
    color: #1a1a2e !important;
}

.main { background: #f0f4ff !important; }
.block-container { padding: 0 1.5rem 3rem !important; max-width: 1100px !important; }
#MainMenu, footer, header { visibility: hidden; }

.top-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 0 0 32px 32px;
    padding: 2rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    margin-left: -1.5rem;
    margin-right: -1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.top-header::before {
    content: '🏥';
    position: absolute;
    font-size: 180px;
    opacity: 0.06;
    top: -20px;
    right: -20px;
}

.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.header-title {
    font-family: 'Nunito', sans-serif;
    font-size: 36px;
    font-weight: 900;
    color: white;
    margin-bottom: 8px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.15);
}

.header-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.85);
    font-weight: 400;
}

.search-wrapper {
    background: white;
    border-radius: 20px;
    padding: 1.5rem 2rem;
    box-shadow: 0 8px 40px rgba(102,126,234,0.15);
    margin-bottom: 1.5rem;
    border: 2px solid rgba(102,126,234,0.1);
}

.search-title {
    font-family: 'Nunito', sans-serif;
    font-size: 16px;
    font-weight: 800;
    color: #667eea;
    margin-bottom: 4px;
}

.search-hint {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 10px;
}

.quick-result-green {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #34d399;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 14px;
}

.quick-result-yellow {
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    border: 2px solid #fbbf24;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 14px;
}

.quick-result-red {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #f87171;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 14px;
}

.quick-icon { font-size: 36px; flex-shrink: 0; }
.quick-label { font-family: 'Nunito', sans-serif; font-size: 18px; font-weight: 800; margin-bottom: 3px; }
.quick-desc { font-size: 13px; color: #4b5563; line-height: 1.5; }

.section-title {
    font-family: 'Nunito', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: #1a1a2e;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.input-card {
    background: white;
    border-radius: 18px;
    padding: 1.4rem 1.3rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 2px solid transparent;
    transition: border-color .2s;
    margin-bottom: 1rem;
}

.input-card:hover { border-color: rgba(102,126,234,0.3); }

.card-header {
    font-family: 'Nunito', sans-serif;
    font-size: 13px;
    font-weight: 800;
    color: #667eea;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 2px dashed #e0e7ff;
}

label { color: #374151 !important; font-size: 13px !important; font-weight: 600 !important; }

.stNumberInput input {
    background: #f8faff !important;
    border: 2px solid #e0e7ff !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.stSelectbox > div > div {
    background: #f8faff !important;
    border: 2px solid #e0e7ff !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-size: 14px !important;
}

.stTextInput input {
    background: white !important;
    border: 2px solid #e0e7ff !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    color: #1a1a2e !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    box-shadow: 0 6px 24px rgba(102,126,234,0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(102,126,234,0.5) !important;
}

.result-safe {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 3px solid #34d399;
    border-radius: 24px;
    padding: 2rem 2.2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(52,211,153,0.2);
}

.result-warning {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 3px solid #fbbf24;
    border-radius: 24px;
    padding: 2rem 2.2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(251,191,36,0.2);
}

.result-danger {
    background: linear-gradient(135deg, #fff1f2, #fee2e2);
    border: 3px solid #f87171;
    border-radius: 24px;
    padding: 2rem 2.2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(248,113,113,0.2);
}

.result-emoji { font-size: 56px; margin-bottom: 10px; }
.result-title-safe    { font-family: 'Nunito', sans-serif; font-size: 28px; font-weight: 900; color: #065f46; margin-bottom: 8px; }
.result-title-warning { font-family: 'Nunito', sans-serif; font-size: 28px; font-weight: 900; color: #92400e; margin-bottom: 8px; }
.result-title-danger  { font-family: 'Nunito', sans-serif; font-size: 28px; font-weight: 900; color: #991b1b; margin-bottom: 8px; }
.result-tag-safe    { display: inline-block; background: #34d399; color: white; border-radius: 20px; padding: 4px 18px; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 12px; }
.result-tag-warning { display: inline-block; background: #fbbf24; color: white; border-radius: 20px; padding: 4px 18px; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 12px; }
.result-tag-danger  { display: inline-block; background: #f87171; color: white; border-radius: 20px; padding: 4px 18px; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 12px; }
.result-advice { font-size: 14px; color: #374151; line-height: 1.8; max-width: 500px; margin: 0 auto; }

.prob-row { display: flex; gap: 12px; margin-top: 1.5rem; }
.prob-card { flex: 1; background: white; border-radius: 16px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.prob-num-green  { font-family: 'Nunito', sans-serif; font-size: 30px; font-weight: 900; color: #059669; }
.prob-num-yellow { font-family: 'Nunito', sans-serif; font-size: 30px; font-weight: 900; color: #d97706; }
.prob-num-red    { font-family: 'Nunito', sans-serif; font-size: 30px; font-weight: 900; color: #dc2626; }
.prob-label { font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }

.prog-wrap { margin-bottom: 12px; }
.prog-top { display: flex; justify-content: space-between; margin-bottom: 5px; }
.prog-name { font-size: 13px; font-weight: 600; color: #374151; }
.prog-pct  { font-size: 13px; font-weight: 700; color: #667eea; }
.prog-track { height: 10px; background: #e0e7ff; border-radius: 10px; overflow: hidden; }
.prog-fill-green  { height: 100%; background: linear-gradient(90deg, #34d399, #059669); border-radius: 10px; }
.prog-fill-yellow { height: 100%; background: linear-gradient(90deg, #fbbf24, #d97706); border-radius: 10px; }
.prog-fill-red    { height: 100%; background: linear-gradient(90deg, #f87171, #dc2626); border-radius: 10px; }

.sum-card { background: white; border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 4px solid #667eea; }
.sum-key { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 3px; }
.sum-val { font-size: 15px; font-weight: 700; color: #1a1a2e; }

.app-footer { text-align: center; font-size: 12px; color: #94a3b8; padding: 2rem 0 1rem; border-top: 2px dashed #e0e7ff; margin-top: 2rem; font-weight: 600; }
div[data-testid="stProgressBar"] { display: none; }
[data-testid="column"] { padding: 0 6px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf = joblib.load("rf_model.pkl")
    sc = joblib.load("scaler.pkl")
    with open("feature_names.json") as f:
        features = json.load(f)
    return rf, sc, features

rf_model, scaler, feature_names = load_model()


# ── TOP HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <div class="header-badge">🏥 MediCheck System</div>
    <div class="header-title">Patient Readmission Predictor</div>
    <div class="header-sub">Simple · Fast · Accurate — Designed for Receptionists & Clinical Staff</div>
</div>
""", unsafe_allow_html=True)


# ── SMART SEARCH BOX ─────────────────────────────────────────────────────────
st.markdown("""
<div class="search-wrapper">
    <div class="search-title">🔍 Quick Risk Check</div>
    <div class="search-hint">Type a patient condition or feature to instantly see the risk level (e.g. insulin, emergency, diabetes, no medication)</div>
</div>
""", unsafe_allow_html=True)

search_query = st.text_input("", placeholder="🔍  Type here... e.g. insulin, emergency visits, diabetes, high medications")

RISK_KEYWORDS = {
    "high": ["insulin up", "emergency", "inpatient", "many medications", "high medications",
             "polypharmacy", "multiple diagnoses", "circulatory", "long stay", "not measured",
             "medication change", "change made", "diabetes medication", "high risk"],
    "medium": ["insulin steady", "metformin", "outpatient", "respiratory", "digestive",
               "moderate", "normal hba1c", "steady", "medium", "some visits"],
    "low":  ["no insulin", "no medication", "young", "short stay", "no visits",
             "no emergency", "no inpatient", "musculoskeletal", "injury", "normal", "low risk"]
}

if search_query:
    q = search_query.lower().strip()
    matched = "unknown"
    for level, keywords in RISK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matched = level
            break

    if matched == "high":
        st.markdown("""
        <div class="quick-result-red">
            <div class="quick-icon">🚨</div>
            <div>
                <div class="quick-label" style="color:#991b1b">High Risk of Readmission</div>
                <div class="quick-desc">This factor is strongly linked to early readmission within 30 days. Please complete the full assessment below and alert the clinical team.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif matched == "medium":
        st.markdown("""
        <div class="quick-result-yellow">
            <div class="quick-icon">⚠️</div>
            <div>
                <div class="quick-label" style="color:#92400e">Moderate Risk of Readmission</div>
                <div class="quick-desc">This factor may increase readmission risk. A follow-up within 2–4 weeks is recommended. Fill in the full form below for an accurate result.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif matched == "low":
        st.markdown("""
        <div class="quick-result-green">
            <div class="quick-icon">✅</div>
            <div>
                <div class="quick-label" style="color:#065f46">Low Risk of Readmission</div>
                <div class="quick-desc">This factor is associated with lower readmission risk. Standard discharge and routine follow-up should be sufficient.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#f8faff; border:2px solid #e0e7ff; border-radius:14px; padding:1rem 1.4rem; margin-top:1rem;">
            <div style="font-size:14px; color:#667eea; font-weight:700;">💡 Try searching for:</div>
            <div style="font-size:13px; color:#64748b; margin-top:6px; line-height:1.8;">
                <b>🚨 High risk:</b> insulin, emergency, inpatient, medication change, diabetes<br>
                <b>⚠️ Medium risk:</b> metformin, outpatient, respiratory, steady<br>
                <b>✅ Low risk:</b> no insulin, young, short stay, musculoskeletal, injury
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── FULL ASSESSMENT FORM ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Full Patient Assessment</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-card"><div class="card-header">👤 Patient Info</div>', unsafe_allow_html=True)
    age              = st.number_input("Age (years)",         min_value=0,  max_value=100, value=65, step=1)
    gender           = st.selectbox("Gender",                 ["Female", "Male"])
    time_in_hospital = st.number_input("Days in Hospital",    min_value=1,  max_value=14,  value=3,  step=1)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=1,  max_value=20,  value=5,  step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card"><div class="card-header">🏥 Visit History</div>', unsafe_allow_html=True)
    number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=50, value=1, step=1)
    number_emergency  = st.number_input("Emergency Visits",  min_value=0, max_value=50, value=0, step=1)
    number_inpatient  = st.number_input("Inpatient Visits",  min_value=0, max_value=50, value=1, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card"><div class="card-header">🔬 Lab & Procedures</div>', unsafe_allow_html=True)
    num_lab_procedures = st.number_input("Lab Procedures",  min_value=0, max_value=200, value=40, step=1)
    num_procedures     = st.number_input("Procedures Done", min_value=0, max_value=20,  value=2,  step=1)
    num_medications    = st.number_input("Medications",     min_value=0, max_value=100, value=12, step=1)
    A1Cresult          = st.selectbox("HbA1c Result",       ["Not measured", "Normal", ">7", ">8"])
    diag_1             = st.selectbox("Primary Diagnosis",  [
        "Circulatory", "Respiratory", "Digestive", "Diabetes",
        "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card"><div class="card-header">💊 Medications</div>', unsafe_allow_html=True)
    insulin     = st.selectbox("Insulin",              ["No", "Steady", "Up", "Down"])
    metformin   = st.selectbox("Metformin",            ["No", "Steady", "Up", "Down"])
    change      = st.selectbox("Medication Change?",   ["No change", "Change made"])
    diabetesMed = st.selectbox("Diabetes Medication?", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)


# ── Encode Inputs ─────────────────────────────────────────────────────────────
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


# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮  Run Full Prediction")

if predict_clicked:
    with st.spinner("Analysing patient data..."):
        input_df      = encode_inputs()
        input_scaled  = scaler.transform(input_df)
        prediction    = rf_model.predict(input_scaled)[0]
        probabilities = rf_model.predict_proba(input_scaled)[0]

    p0 = round(float(probabilities[0]) * 100, 1)
    p1 = round(float(probabilities[1]) * 100, 1)
    p2 = round(float(probabilities[2]) * 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 0:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-emoji">✅</div>
            <div class="result-title-safe">Not Likely to be Readmitted</div>
            <div class="result-tag-safe">Low Risk</div><br>
            <div class="result-advice">Great news! This patient is unlikely to return within 30 days. Standard discharge is recommended. Schedule a routine follow-up in 4–6 weeks.</div>
        </div>""", unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown(f"""
        <div class="result-warning">
            <div class="result-emoji">⚠️</div>
            <div class="result-title-warning">May Return After 30 Days</div>
            <div class="result-tag-warning">Moderate Risk</div><br>
            <div class="result-advice">This patient may need to return after 30 days. Please schedule a follow-up within 2–4 weeks and review their medication plan.</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-danger">
            <div class="result-emoji">🚨</div>
            <div class="result-title-danger">Likely to Return Within 30 Days!</div>
            <div class="result-tag-danger">High Risk — Act Now</div><br>
            <div class="result-advice">⚠️ This patient is at HIGH risk of early readmission. Please alert the clinical team immediately and create a discharge care plan before the patient leaves.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="prob-row">
        <div class="prob-card">
            <div class="prob-num-green">{p0}%</div>
            <div class="prob-label">✅ Not Readmitted</div>
        </div>
        <div class="prob-card">
            <div class="prob-num-yellow">{p1}%</div>
            <div class="prob-label">⚠️ After 30 Days</div>
        </div>
        <div class="prob-card">
            <div class="prob-num-red">{p2}%</div>
            <div class="prob-label">🚨 Within 30 Days</div>
        </div>
    </div>
    <div style="background:white;border-radius:16px;padding:1.4rem;margin-top:1rem;box-shadow:0 4px 16px rgba(0,0,0,0.06);">
        <div class="prog-wrap">
            <div class="prog-top"><span class="prog-name">✅ Not Readmitted</span><span class="prog-pct">{p0}%</span></div>
            <div class="prog-track"><div class="prog-fill-green" style="width:{p0}%"></div></div>
        </div>
        <div class="prog-wrap">
            <div class="prog-top"><span class="prog-name">⚠️ After 30 Days</span><span class="prog-pct">{p1}%</span></div>
            <div class="prog-track"><div class="prog-fill-yellow" style="width:{p1}%"></div></div>
        </div>
        <div class="prog-wrap">
            <div class="prog-top"><span class="prog-name">🚨 Within 30 Days</span><span class="prog-pct">{p2}%</span></div>
            <div class="prog-track"><div class="prog-fill-red" style="width:{p2}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂️ Patient Summary</div>', unsafe_allow_html=True)

    def card(k, v):
        return f'<div class="sum-card"><div class="sum-key">{k}</div><div class="sum-val">{v}</div></div>'

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(card("Age", f"{age} yrs") + card("Gender", gender) + card("Days in Hospital", f"{time_in_hospital} days"), unsafe_allow_html=True)
    with s2:
        st.markdown(card("Emergency Visits", number_emergency) + card("Medications", num_medications) + card("HbA1c", A1Cresult), unsafe_allow_html=True)
    with s3:
        st.markdown(card("Insulin", insulin) + card("Primary Diagnosis", diag_1) + card("Medication Change", change), unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    🏥 MediCheck — Patient Readmission Predictor &nbsp;·&nbsp; For Clinical & Receptionist Use Only
</div>
""", unsafe_allow_html=True)