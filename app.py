import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from datetime import date

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

.top-banner {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    padding: 1.2rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.banner-icon {
    width: 50px; height: 50px;
    background: rgba(255,255,255,0.15);
    border-radius: 12px;
    display: flex; align-items: center;
    justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.banner-title   { font-size: 20px; font-weight: 700; margin-bottom: 2px; }
.banner-sub     { font-size: 12px; opacity: 0.8; }
.banner-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}

.section-head {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .07em;
    color: #2563eb; margin-bottom: 10px; margin-top: 6px;
    border-bottom: 2px solid #eff6ff; padding-bottom: 6px;
}

.id-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}

.result-safe {
    background: #ecfdf5; border: 1.5px solid #6ee7b7;
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.result-warning {
    background: #fffbeb; border: 1.5px solid #fcd34d;
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.result-danger {
    background: #fef2f2; border: 1.5px solid #fca5a5;
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.result-title-safe    { font-size: 20px; font-weight: 700; color: #065f46; margin-bottom: 4px; }
.result-title-warning { font-size: 20px; font-weight: 700; color: #92400e; margin-bottom: 4px; }
.result-title-danger  { font-size: 20px; font-weight: 700; color: #991b1b; margin-bottom: 4px; }
.result-advice        { font-size: 13px; color: #4b5563; line-height: 1.6; }

.metric-wrap {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem; text-align: center;
}
.metric-number { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
.metric-desc   {
    font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: .05em; color: #94a3b8;
}
.num-green  { color: #059669; }
.num-orange { color: #d97706; }
.num-red    { color: #dc2626; }

.gauge-wrap {
    text-align: center; padding: 1.5rem;
    background: white; border-radius: 14px;
    border: 1.5px solid #e2e8f0; margin-bottom: 1rem;
}
.gauge-label {
    font-size: 12px; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px;
}
.gauge-score { font-size: 52px; font-weight: 700; line-height: 1; }
.gauge-risk  { font-size: 14px; font-weight: 600; margin-top: 6px; }

.summary-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
}
.summary-key {
    font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .05em;
    color: #94a3b8; margin-bottom: 3px;
}
.summary-val { font-size: 14px; font-weight: 600; color: #1e293b; }

.app-footer {
    text-align: center; font-size: 11px; color: #94a3b8;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #e2e8f0; margin-top: 2rem;
}

.stButton > button {
    width: 100%; padding: 0.75rem 1rem;
    background: #2563eb; color: white;
    border: none; border-radius: 10px;
    font-size: 15px; font-weight: 600;
    font-family: 'Inter', sans-serif;
    cursor: pointer; transition: background .2s;
}
.stButton > button:hover { background: #1d4ed8; color: white; border: none; }

.stNumberInput input {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf = joblib.load("rf_model.pkl")
    sc = joblib.load("scaler.pkl")
    with open("feature_names.json") as f:
        features = json.load(f)
    return rf, sc, features

rf_model, scaler, feature_names = load_model()


# ── Session State — Prediction History ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Top Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <div class="banner-icon">🏥</div>
  <div>
    <div class="banner-title">Patient Readmission Predictor</div>
    <div class="banner-sub">Diabetic Patient Early Risk Assessment System</div>
  </div>
  <div class="banner-badge">Random Forest Model</div>
</div>
""", unsafe_allow_html=True)


# ── SECTION 1 — Patient Identity ───────────────────────────────────────────
st.markdown('<div class="section-head">Patient Identity</div>',
            unsafe_allow_html=True)

st.markdown('<div class="id-card">', unsafe_allow_html=True)

id1, id2, id3 = st.columns(3)

with id1:
    patient_name = st.text_input("Patient Full Name",
                                 placeholder="e.g. John Smith")
    patient_id   = st.text_input("Patient ID / MRN",
                                 placeholder="e.g. MRN-00123")

with id2:
    visit_date  = st.date_input("Date of Visit", value=date.today())
    doctor_name = st.text_input("Doctor Name",
                                placeholder="e.g. Dr. Sharma")

with id3:
    ward  = st.text_input("Ward / Department",
                          placeholder="e.g. Diabetic Ward")
    notes = st.text_area("Notes (optional)", height=80,
                         placeholder="Any additional observations...")

st.markdown('</div>', unsafe_allow_html=True)


# ── SECTION 2 — Clinical Details ───────────────────────────────────────────
st.markdown('<div class="section-head">Clinical Details</div>',
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**👤 Patient Info**")
    age              = st.number_input("Age (years)",             min_value=0,   max_value=100, value=65,  step=1)
    time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1,   max_value=14,  value=3,   step=1)
    number_diagnoses = st.number_input("Number of Diagnoses",     min_value=1,   max_value=20,  value=5,   step=1)
    gender           = st.selectbox("Gender", ["Female", "Male"])

with col2:
    st.markdown("**🔬 Lab and Procedures**")
    num_lab_procedures = st.number_input("Lab Procedures",  min_value=0, max_value=200, value=40, step=1)
    num_procedures     = st.number_input("Procedures Done", min_value=0, max_value=20,  value=2,  step=1)
    num_medications    = st.number_input("Medications",     min_value=0, max_value=100, value=12, step=1)
    A1Cresult          = st.selectbox("HbA1c Result",
                                      ["Not measured", "Normal", ">7", ">8"])

with col3:
    st.markdown("**🏥 Visit History**")
    number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=50, value=1, step=1)
    number_emergency  = st.number_input("Emergency Visits",  min_value=0, max_value=50, value=0, step=1)
    number_inpatient  = st.number_input("Inpatient Visits",  min_value=0, max_value=50, value=1, step=1)
    diag_1            = st.selectbox("Primary Diagnosis", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes",
        "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"
    ])

with col4:
    st.markdown("**💊 Medications**")
    insulin     = st.selectbox("Insulin",              ["No", "Steady", "Up", "Down"])
    metformin   = st.selectbox("Metformin",            ["No", "Steady", "Up", "Down"])
    change      = st.selectbox("Medication Change?",   ["No change", "Change made"])
    diabetesMed = st.selectbox("Diabetes Medication?", ["Yes", "No"])

st.markdown("<br>", unsafe_allow_html=True)


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

    input_data["age"]                = age
    input_data["time_in_hospital"]   = time_in_hospital
    input_data["number_diagnoses"]   = number_diagnoses
    input_data["gender"]             = 1 if gender == "Male" else 0
    input_data["num_lab_procedures"] = num_lab_procedures
    input_data["num_procedures"]     = num_procedures
    input_data["num_medications"]    = num_medications
    input_data["A1Cresult"]          = a1c_map[A1Cresult]
    input_data["number_outpatient"]  = number_outpatient
    input_data["number_emergency"]   = number_emergency
    input_data["number_inpatient"]   = number_inpatient
    input_data["diag_1"]             = diag_map[diag_1]
    input_data["insulin"]            = med_map[insulin]
    input_data["metformin"]          = med_map[metformin]
    input_data["change"]             = 1 if change == "Change made" else 0
    input_data["diabetesMed"]        = 1 if diabetesMed == "Yes" else 0
    input_data["total_visits"]       = number_outpatient + number_emergency + number_inpatient
    input_data["med_changed"]        = 1 if change == "Change made" else 0
    input_data["polypharmacy"]       = 1 if num_medications > 10 else 0
    input_data["lab_proc_ratio"]     = num_lab_procedures / (num_procedures + 1)

    return pd.DataFrame([input_data])


# ── Predict Button ─────────────────────────────────────────────────────────
predict_clicked = st.button("▶  Predict Readmission Risk")

if predict_clicked:

    # Validate patient name
    if not patient_name.strip():
        st.warning("Please enter the patient name before predicting.")
        st.stop()

    with st.spinner("Analysing patient data..."):
        input_df      = encode_inputs()
        input_scaled  = scaler.transform(input_df)
        prediction    = rf_model.predict(input_scaled)[0]
        probabilities = rf_model.predict_proba(input_scaled)[0]

    p0 = round(float(probabilities[0]) * 100, 1)
    p1 = round(float(probabilities[1]) * 100, 1)
    p2 = round(float(probabilities[2]) * 100, 1)

    risk_labels = ["Not Readmitted",
                   "Readmitted After 30 Days",
                   "Readmitted Within 30 Days"]
    risk_levels = ["Low", "Medium", "High"]

    # ── Save to history ────────────────────────────────────────────────────
    st.session_state.history.append({
        "Patient Name"   : patient_name.strip() or "Unknown",
        "Patient ID"     : patient_id.strip()   or "N/A",
        "Date"           : str(visit_date),
        "Doctor"         : doctor_name.strip()  or "N/A",
        "Ward"           : ward.strip()         or "N/A",
        "Prediction"     : risk_labels[prediction],
        "Risk Level"     : risk_levels[prediction],
        "Confidence"     : str(max(p0, p1, p2)) + "%"
    })

    st.markdown("---")

    # ── SECTION 3 — Risk Gauge ─────────────────────────────────────────────
    st.markdown('<div class="section-head">Overall Risk Score</div>',
                unsafe_allow_html=True)

    risk_score = p2
    if risk_score >= 50:
        gauge_colour = "#dc2626"
        gauge_label  = "HIGH RISK"
    elif risk_score >= 30:
        gauge_colour = "#d97706"
        gauge_label  = "MEDIUM RISK"
    else:
        gauge_colour = "#059669"
        gauge_label  = "LOW RISK"

    gc1, gc2, gc3 = st.columns([1, 2, 1])
    with gc2:
        st.markdown(f"""
        <div class="gauge-wrap">
            <div class="gauge-label">Readmission Risk Score (within 30 days)</div>
            <div class="gauge-score" style="color:{gauge_colour}">{risk_score}%</div>
            <div class="gauge-risk"  style="color:{gauge_colour}">{gauge_label}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(risk_score / 100)

    # ── SECTION 4 — Prediction Result ─────────────────────────────────────
    st.markdown('<div class="section-head">Prediction Result</div>',
                unsafe_allow_html=True)

    if prediction == 0:
        st.markdown("""
        <div class="result-safe">
            <div class="result-title-safe">✅ Not Readmitted</div>
            <div class="result-advice">
                Patient is unlikely to return soon.
                Standard follow-up recommended. No immediate action required.
            </div>
        </div>""", unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown("""
        <div class="result-warning">
            <div class="result-title-warning">⚠️ Readmitted After 30 Days</div>
            <div class="result-advice">
                Patient may return after 30 days.
                Schedule a follow-up appointment within the next 2 to 4 weeks.
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-danger">
            <div class="result-title-danger">🚨 Readmitted Within 30 Days</div>
            <div class="result-advice">
                High risk patient. Immediate care plan advised.
                Coordinate with the clinical team before discharge.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── SECTION 5 — Probability Breakdown ─────────────────────────────────
    st.markdown('<div class="section-head">Probability Breakdown</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        st.markdown(f"""
        <div class="metric-wrap">
            <div class="metric-number num-green">{p0}%</div>
            <div class="metric-desc">Not Readmitted</div>
        </div>""", unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="metric-wrap">
            <div class="metric-number num-orange">{p1}%</div>
            <div class="metric-desc">After 30 Days</div>
        </div>""", unsafe_allow_html=True)

    with mc3:
        st.markdown(f"""
        <div class="metric-wrap">
            <div class="metric-number num-red">{p2}%</div>
            <div class="metric-desc">Within 30 Days</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Not readmitted**")
    st.progress(p0 / 100)
    st.markdown("**Readmitted after 30 days**")
    st.progress(p1 / 100)
    st.markdown("**Readmitted within 30 days**")
    st.progress(p2 / 100)

    # ── SECTION 6 — Patient Summary ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-head">Patient Summary</div>',
                unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(f'<div class="summary-card"><div class="summary-key">Patient Name</div><div class="summary-val">{patient_name or "N/A"}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">Patient ID</div><div class="summary-val">{patient_id or "N/A"}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">Doctor</div><div class="summary-val">{doctor_name or "N/A"}</div></div>', unsafe_allow_html=True)

    with s2:
        st.markdown(f'<div class="summary-card"><div class="summary-key">Age</div><div class="summary-val">{age} years</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">Gender</div><div class="summary-val">{gender}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">Time in Hospital</div><div class="summary-val">{time_in_hospital} days</div></div>', unsafe_allow_html=True)

    with s3:
        st.markdown(f'<div class="summary-card"><div class="summary-key">Lab Procedures</div><div class="summary-val">{num_lab_procedures}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">Medications</div><div class="summary-val">{num_medications}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">HbA1c</div><div class="summary-val">{A1Cresult}</div></div>', unsafe_allow_html=True)

    with s4:
        st.markdown(f'<div class="summary-card"><div class="summary-key">Emergency Visits</div><div class="summary-val">{number_emergency}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">Primary Diagnosis</div><div class="summary-val">{diag_1}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-key">Insulin</div><div class="summary-val">{insulin}</div></div>', unsafe_allow_html=True)

    # ── SECTION 7 — Download Report ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-head">Download Report</div>',
                unsafe_allow_html=True)

    report = f"""
PATIENT READMISSION RISK REPORT
================================
Patient Name   : {patient_name  or "N/A"}
Patient ID     : {patient_id    or "N/A"}
Date of Visit  : {visit_date}
Doctor         : {doctor_name   or "N/A"}
Ward           : {ward          or "N/A"}

PREDICTION RESULT
-----------------
Outcome        : {risk_labels[prediction]}
Risk Level     : {risk_levels[prediction]}

PROBABILITIES
-------------
Not Readmitted          : {p0}%
Readmitted after 30 days: {p1}%
Readmitted within 30 days: {p2}%

CLINICAL INPUTS
---------------
Age                  : {age}
Gender               : {gender}
Time in Hospital     : {time_in_hospital} days
Number of Diagnoses  : {number_diagnoses}
Lab Procedures       : {num_lab_procedures}
Procedures Done      : {num_procedures}
Medications          : {num_medications}
HbA1c Result         : {A1Cresult}
Outpatient Visits    : {number_outpatient}
Emergency Visits     : {number_emergency}
Inpatient Visits     : {number_inpatient}
Primary Diagnosis    : {diag_1}
Insulin              : {insulin}
Metformin            : {metformin}
Medication Change    : {change}
Diabetes Medication  : {diabetesMed}

Notes: {notes or "None"}

================================
Generated by Patient Readmission Predictor
Date: {date.today()}
    """

    st.download_button(
        label     = "📄 Download Patient Report (.txt)",
        data      = report,
        file_name = f"report_{patient_id or 'patient'}_{visit_date}.txt",
        mime      = "text/plain"
    )


# ── SECTION 8 — Prediction History ────────────────────────────────────────
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown('<div class="section-head">Prediction History — This Session</div>',
                unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history)

    def colour_risk(val):
        if val == "High":
            return "background-color:#fef2f2; color:#991b1b; font-weight:600"
        elif val == "Medium":
            return "background-color:#fffbeb; color:#92400e; font-weight:600"
        else:
            return "background-color:#ecfdf5; color:#065f46; font-weight:600"

    styled_df = history_df.style.map(colour_risk, subset=["Risk Level"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "📥 Download History as CSV",
        data      = csv,
        file_name = f"prediction_history_{date.today()}.csv",
        mime      = "text/csv"
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    🏥 Diabetic Patient Readmission Predictor &mdash;
    Random Forest Model &mdash;
    For clinical support use only
</div>
""", unsafe_allow_html=True)