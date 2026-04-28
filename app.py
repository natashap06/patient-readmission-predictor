import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from datetime import date

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

.banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
    border-radius: 16px; padding: 1.4rem 2rem;
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 1.5rem;
}
.banner-left  { display: flex; align-items: center; gap: 14px; }
.banner-icon  {
    width: 48px; height: 48px; border-radius: 12px;
    background: rgba(255,255,255,0.12);
    display: flex; align-items: center;
    justify-content: center; font-size: 22px;
}
.banner-title { font-size: 19px; font-weight: 700; color: white; margin-bottom: 2px; }
.banner-sub   { font-size: 12px; color: rgba(255,255,255,0.65); }
.banner-right { display: flex; align-items: center; gap: 8px; }
.banner-pill  {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85);
    font-size: 11px; font-weight: 500;
    padding: 5px 12px; border-radius: 20px;
}
.banner-pill-green {
    background: rgba(16,185,129,0.2);
    border: 1px solid rgba(16,185,129,0.4);
    color: #6ee7b7; font-size: 11px; font-weight: 500;
    padding: 5px 12px; border-radius: 20px;
}
.sec-head {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    color: #64748b; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}
.sec-head::after {
    content: ''; flex: 1; height: 1px; background: #f1f5f9;
}
.emg-banner {
    background: #fef2f2; border: 1.5px solid #fca5a5;
    border-radius: 10px; padding: 10px 16px;
    font-size: 13px; color: #991b1b; font-weight: 500;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;
}
.id-strip {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.2rem;
}
.res-strip-safe {
    background: #ecfdf5; border: 1.5px solid #86efac;
    border-radius: 14px; padding: 1.3rem 1.6rem;
    display: flex; align-items: center; gap: 14px; margin-bottom: 1.2rem;
}
.res-strip-warn {
    background: #fffbeb; border: 1.5px solid #fde68a;
    border-radius: 14px; padding: 1.3rem 1.6rem;
    display: flex; align-items: center; gap: 14px; margin-bottom: 1.2rem;
}
.res-strip-danger {
    background: #fef2f2; border: 1.5px solid #fca5a5;
    border-radius: 14px; padding: 1.3rem 1.6rem;
    display: flex; align-items: center; gap: 14px; margin-bottom: 1.2rem;
}
.res-icon   { width:52px; height:52px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0; }
.res-title  { font-size: 18px; font-weight: 700; margin-bottom: 3px; }
.res-advice { font-size: 13px; line-height: 1.6; }
.safe-icon   { background: #d1fae5; }
.warn-icon   { background: #fef3c7; }
.danger-icon { background: #fee2e2; }
.safe-text   { color: #065f46; }
.warn-text   { color: #92400e; }
.danger-text { color: #991b1b; }
.gauge-box {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.5rem; text-align: center;
}
.gauge-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .07em; color: #94a3b8; margin-bottom: 8px;
}
.gauge-num  { font-size: 54px; font-weight: 700; line-height: 1; }
.gauge-risk { font-size: 13px; font-weight: 600; margin-top: 6px; }
.prob-card  {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem; text-align: center;
}
.prob-num { font-size: 28px; font-weight: 700; margin-bottom: 3px; }
.prob-lbl {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .05em; color: #94a3b8;
}
.sum-item {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
}
.sum-key {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .05em; color: #94a3b8; margin-bottom: 2px;
}
.sum-val { font-size: 14px; font-weight: 600; color: #1e293b; }
.stButton > button {
    width: 100%; padding: 0.75rem 1rem;
    background: #2563eb; color: white; border: none;
    border-radius: 10px; font-size: 14px; font-weight: 600;
    font-family: 'Inter', sans-serif; letter-spacing: .02em;
}
.stButton > button:hover { background: #1d4ed8; color: white; border: none; }
.footer {
    text-align: center; font-size: 11px; color: #94a3b8;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #f1f5f9; margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model, Scaler, Features, Medians ──────────────────────────────────
@st.cache_resource
def load_model():
    rf = joblib.load("rf_model.pkl")
    sc = joblib.load("scaler.pkl")
    with open("feature_names.json") as f:
        features = json.load(f)
    with open("feature_medians.json") as f:
        medians = json.load(f)
    return rf, sc, features, medians

rf_model, scaler, feature_names, feature_medians = load_model()

# ── Session History ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── BANNER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div class="banner-left">
    <div class="banner-icon">🏥</div>
    <div>
      <div class="banner-title">Patient Readmission Predictor</div>
      <div class="banner-sub">Diabetic Early Risk Assessment — Clinical Decision Support</div>
    </div>
  </div>
  <div class="banner-right">
    <div class="banner-pill">Random Forest</div>
    <div class="banner-pill-green">● Live</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── EMERGENCY TOGGLE ───────────────────────────────────────────────────────
emergency_mode = st.toggle("🚨 Emergency Mode — minimal input required",
                            value=False)

if emergency_mode:
    st.markdown("""
    <div class="emg-banner">
        🚨 Emergency mode active — only Patient Name required.
        All clinical fields pre-filled with high risk defaults.
    </div>
    """, unsafe_allow_html=True)


# ── SECTION 1 — Patient Identity ───────────────────────────────────────────
st.markdown('<div class="sec-head">Patient Identity</div>',
            unsafe_allow_html=True)

st.markdown('<div class="id-strip">', unsafe_allow_html=True)
id1, id2, id3 = st.columns(3)

with id1:
    patient_name = st.text_input("Patient Full Name *",
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
    notes = st.text_area("Notes (optional)", height=68,
                          placeholder="Any observations...")

st.markdown('</div>', unsafe_allow_html=True)


# ── SECTION 2 — Quick Presets ──────────────────────────────────────────────
st.markdown('<div class="sec-head">Quick Presets</div>',
            unsafe_allow_html=True)

st.markdown("""
<div style="font-size:12px; color:#64748b; margin-bottom:8px;">
    Select a common patient profile to auto-fill fields instantly.
</div>
""", unsafe_allow_html=True)

preset = st.radio(
    "Select patient profile:",
    ["Custom",
     "Elderly High Risk",
     "Young Low Risk",
     "Chronic Diabetic",
     "Post-Surgery"],
    horizontal=True,
    label_visibility="collapsed"
)

presets = {
    "Custom": {
        "age": 65, "time": 3, "diagnoses": 5, "lab": 40,
        "procedures": 2, "meds": 12, "outpatient": 1,
        "emergency": 0, "inpatient": 1
    },
    "Elderly High Risk": {
        "age": 82, "time": 8, "diagnoses": 9, "lab": 65,
        "procedures": 5, "meds": 18, "outpatient": 3,
        "emergency": 3, "inpatient": 4
    },
    "Young Low Risk": {
        "age": 38, "time": 2, "diagnoses": 2, "lab": 22,
        "procedures": 1, "meds": 5, "outpatient": 0,
        "emergency": 0, "inpatient": 0
    },
    "Chronic Diabetic": {
        "age": 60, "time": 5, "diagnoses": 7, "lab": 55,
        "procedures": 3, "meds": 15, "outpatient": 2,
        "emergency": 2, "inpatient": 2
    },
    "Post-Surgery": {
        "age": 55, "time": 7, "diagnoses": 6, "lab": 70,
        "procedures": 6, "meds": 14, "outpatient": 1,
        "emergency": 1, "inpatient": 3
    }
}

pv = presets["Elderly High Risk"] if emergency_mode else presets[preset]


# ── SECTION 3 — Clinical Fields ────────────────────────────────────────────
st.markdown('<div class="sec-head">Clinical Details</div>',
            unsafe_allow_html=True)

cl1, cl2, cl3 = st.columns(3)

with cl1:
    st.markdown("**👤 Patient**")
    age              = st.number_input("Age",
                                        min_value=0, max_value=100,
                                        value=pv["age"], step=1)
    gender           = st.selectbox("Gender", ["Female", "Male"])
    time_in_hospital = st.number_input("Days in Hospital",
                                        min_value=1, max_value=14,
                                        value=pv["time"], step=1)

with cl2:
    st.markdown("**🔬 Clinical**")
    num_lab_procedures = st.number_input("Lab Procedures",
                                          min_value=0, max_value=200,
                                          value=pv["lab"], step=1)
    num_medications    = st.number_input("Medications",
                                          min_value=0, max_value=100,
                                          value=pv["meds"], step=1)
    A1Cresult          = st.selectbox("HbA1c Result",
                                       ["Not measured", "Normal", ">7", ">8"])

with cl3:
    st.markdown("**🏥 Visits and Medication**")
    number_emergency = st.number_input("Emergency Visits",
                                        min_value=0, max_value=50,
                                        value=pv["emergency"], step=1)
    number_inpatient = st.number_input("Inpatient Visits",
                                        min_value=0, max_value=50,
                                        value=pv["inpatient"], step=1)
    insulin          = st.selectbox("Insulin",
                                     ["No", "Steady", "Up", "Down"])

# These are not shown in UI but filled from presets
number_outpatient = pv["outpatient"]
num_procedures    = pv["procedures"]
number_diagnoses  = pv["diagnoses"]

st.markdown("<br>", unsafe_allow_html=True)


# ── Encode Inputs — FIXED WITH MEDIANS ────────────────────────────────────
def encode_inputs():
    med_map  = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    a1c_map  = {"Not measured": 0, "Normal": 1, ">7": 2, ">8": 3}
    diag_map = {
        "Circulatory": 0, "Respiratory": 1, "Digestive": 2,
        "Diabetes": 3,    "Injury": 4,      "Musculoskeletal": 5,
        "Genitourinary": 6, "Neoplasms": 7, "Other": 8
    }

    # ── KEY FIX ────────────────────────────────────────────────────────────
    # Use real median values from training data as defaults
    # This means any feature not filled by the receptionist
    # defaults to a realistic average patient value
    # instead of 0 which was causing always-high-risk predictions
    input_data = {
        feat: float(feature_medians.get(str(feat), 0))
        for feat in feature_names
    }

    # ── Override with receptionist inputs ──────────────────────────────────
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
    input_data["insulin"]            = med_map[insulin]

    # ── Engineered features ────────────────────────────────────────────────
    input_data["total_visits"]   = (number_outpatient +
                                    number_emergency +
                                    number_inpatient)
    input_data["med_changed"]    = float(input_data.get("change", 0))
    input_data["polypharmacy"]   = 1 if num_medications > 10 else 0
    input_data["lab_proc_ratio"] = num_lab_procedures / (num_procedures + 1)

    return pd.DataFrame([input_data])


# ── Predict Button ─────────────────────────────────────────────────────────
predict_clicked = st.button("▶  Predict Readmission Risk")

if predict_clicked:

    if not patient_name.strip():
        st.warning("⚠️ Please enter the patient name before predicting.")
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

    # Save to history
    st.session_state.history.append({
        "Name"       : patient_name.strip() or "Unknown",
        "ID"         : patient_id.strip()   or "N/A",
        "Date"       : str(visit_date),
        "Doctor"     : doctor_name.strip()  or "N/A",
        "Ward"       : ward.strip()         or "N/A",
        "Prediction" : risk_labels[prediction],
        "Risk"       : risk_levels[prediction],
        "Confidence" : f"{max(p0, p1, p2)}%"
    })

    st.markdown("---")

    # ── Risk Gauge ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Risk Score</div>',
                unsafe_allow_html=True)

    risk_score = p2
    if risk_score >= 50:
        g_colour = "#dc2626"; g_label = "HIGH RISK"
    elif risk_score >= 30:
        g_colour = "#d97706"; g_label = "MEDIUM RISK"
    else:
        g_colour = "#059669"; g_label = "LOW RISK"

    ga, gb, gc = st.columns([1, 1.5, 1])
    with gb:
        st.markdown(f"""
        <div class="gauge-box">
            <div class="gauge-label">Readmission within 30 days</div>
            <div class="gauge-num" style="color:{g_colour}">{risk_score}%</div>
            <div class="gauge-risk" style="color:{g_colour}">{g_label}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(risk_score / 100)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Prediction Result</div>',
                unsafe_allow_html=True)

    if prediction == 0:
        st.markdown("""
        <div class="res-strip-safe">
            <div class="res-icon safe-icon">✅</div>
            <div>
                <div class="res-title safe-text">Not Readmitted</div>
                <div class="res-advice safe-text">
                    Patient is unlikely to return soon.
                    Standard follow-up recommended.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown("""
        <div class="res-strip-warn">
            <div class="res-icon warn-icon">⚠️</div>
            <div>
                <div class="res-title warn-text">Readmitted After 30 Days</div>
                <div class="res-advice warn-text">
                    Patient may return after 30 days.
                    Schedule a follow-up within 2 to 4 weeks.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="res-strip-danger">
            <div class="res-icon danger-icon">🚨</div>
            <div>
                <div class="res-title danger-text">Readmitted Within 30 Days</div>
                <div class="res-advice danger-text">
                    High risk patient. Immediate care plan advised.
                    Coordinate with clinical team before discharge.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Probability Cards ──────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Probability Breakdown</div>',
                unsafe_allow_html=True)

    pb1, pb2, pb3 = st.columns(3)

    with pb1:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-num" style="color:#059669">{p0}%</div>
            <div class="prob-lbl">Not readmitted</div>
        </div>""", unsafe_allow_html=True)
        st.progress(p0 / 100)

    with pb2:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-num" style="color:#d97706">{p1}%</div>
            <div class="prob-lbl">After 30 days</div>
        </div>""", unsafe_allow_html=True)
        st.progress(p1 / 100)

    with pb3:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-num" style="color:#dc2626">{p2}%</div>
            <div class="prob-lbl">Within 30 days</div>
        </div>""", unsafe_allow_html=True)
        st.progress(p2 / 100)

    # ── Patient Summary ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-head">Patient Summary</div>',
                unsafe_allow_html=True)

    def sc(key, val):
        return f"""<div class="sum-item">
                     <div class="sum-key">{key}</div>
                     <div class="sum-val">{val}</div>
                   </div>"""

    sm1, sm2, sm3, sm4 = st.columns(4)

    with sm1:
        st.markdown(sc("Patient Name", patient_name  or "N/A"), unsafe_allow_html=True)
        st.markdown(sc("Patient ID",   patient_id    or "N/A"), unsafe_allow_html=True)
        st.markdown(sc("Doctor",       doctor_name   or "N/A"), unsafe_allow_html=True)

    with sm2:
        st.markdown(sc("Age",              f"{age} years"),          unsafe_allow_html=True)
        st.markdown(sc("Gender",           gender),                  unsafe_allow_html=True)
        st.markdown(sc("Days in Hospital", f"{time_in_hospital}"),   unsafe_allow_html=True)

    with sm3:
        st.markdown(sc("Lab Procedures",   num_lab_procedures),      unsafe_allow_html=True)
        st.markdown(sc("Medications",      num_medications),         unsafe_allow_html=True)
        st.markdown(sc("HbA1c",            A1Cresult),               unsafe_allow_html=True)

    with sm4:
        st.markdown(sc("Emergency Visits", number_emergency),        unsafe_allow_html=True)
        st.markdown(sc("Inpatient Visits", number_inpatient),        unsafe_allow_html=True)
        st.markdown(sc("Insulin",          insulin),                 unsafe_allow_html=True)

    # ── Download Report ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-head">Report</div>',
                unsafe_allow_html=True)

    report = f"""
PATIENT READMISSION RISK REPORT
================================
Patient Name    : {patient_name  or "N/A"}
Patient ID      : {patient_id    or "N/A"}
Date of Visit   : {visit_date}
Doctor          : {doctor_name   or "N/A"}
Ward            : {ward          or "N/A"}

PREDICTION RESULT
-----------------
Outcome         : {risk_labels[prediction]}
Risk Level      : {risk_levels[prediction]}

PROBABILITIES
-------------
Not Readmitted           : {p0}%
Readmitted after 30 days : {p1}%
Readmitted within 30 days: {p2}%

CLINICAL INPUTS
---------------
Age                  : {age}
Gender               : {gender}
Days in Hospital     : {time_in_hospital}
Lab Procedures       : {num_lab_procedures}
Medications          : {num_medications}
HbA1c Result         : {A1Cresult}
Emergency Visits     : {number_emergency}
Inpatient Visits     : {number_inpatient}
Insulin              : {insulin}

Notes : {notes or "None"}

================================
Generated by Patient Readmission Predictor
Date  : {date.today()}
    """

    st.download_button(
        label              = "📄 Download Patient Report",
        data               = report,
        file_name          = f"report_{patient_id or 'patient'}_{visit_date}.txt",
        mime               = "text/plain",
        use_container_width = True
    )


# ── Prediction History ─────────────────────────────────────────────────────
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown('<div class="sec-head">Prediction History — This Session</div>',
                unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history)

    def colour_risk(val):
        if val == "High":
            return "background-color:#fef2f2; color:#991b1b; font-weight:600"
        elif val == "Medium":
            return "background-color:#fffbeb; color:#92400e; font-weight:600"
        else:
            return "background-color:#ecfdf5; color:#065f46; font-weight:600"

    styled = history_df.style.map(colour_risk, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv = history_df.to_csv(index=False).encode("utf-8")

    h1, h2 = st.columns(2)
    with h1:
        st.download_button(
            label              = "📥 Download History as CSV",
            data               = csv,
            file_name          = f"history_{date.today()}.csv",
            mime               = "text/csv",
            use_container_width = True
        )
    with h2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🏥 Patient Readmission Predictor &mdash;
    Random Forest Model &mdash; For clinical decision support only
</div>
""", unsafe_allow_html=True)