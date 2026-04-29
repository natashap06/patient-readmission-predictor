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
        "age": 65, "time": 4,  "diagnoses": 7,  "lab": 44,
        "procedures": 1, "meds": 14, "outpatient": 0,
        "emergency": 0,  "inpatient": 0,
        "insulin": "Steady", "A1Cresult": "Not measured"
    },
    "Elderly High Risk": {
        "age": 82, "time": 8,  "diagnoses": 9,  "lab": 65,
        "procedures": 5, "meds": 18, "outpatient": 3,
        "emergency": 3,  "inpatient": 4,
        "insulin": "Up", "A1Cresult": ">8"
    },
    "Young Low Risk": {
        "age": 38, "time": 2,  "diagnoses": 2,  "lab": 22,
        "procedures": 1, "meds": 5,  "outpatient": 0,
        "emergency": 0,  "inpatient": 0,
        "insulin": "No", "A1Cresult": "Normal"
    },
    "Chronic Diabetic": {
        "age": 60, "time": 5,  "diagnoses": 7,  "lab": 55,
        "procedures": 3, "meds": 15, "outpatient": 2,
        "emergency": 2,  "inpatient": 2,
        "insulin": "Steady", "A1Cresult": ">7"
    },
    "Post-Surgery": {
        "age": 55, "time": 7,  "diagnoses": 6,  "lab": 70,
        "procedures": 6, "meds": 14, "outpatient": 1,
        "emergency": 1,  "inpatient": 3,
        "insulin": "Steady", "A1Cresult": "Not measured"
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
                                       ["Not measured", "Normal", ">7", ">8"],
                                       index=["Not measured", "Normal",
                                              ">7", ">8"].index(pv["A1Cresult"]))

with cl3:
    st.markdown("**🏥 Visits and Medication**")
    number_emergency = st.number_input("Emergency Visits",
                                        min_value=0, max_value=50,
                                        value=pv["emergency"], step=1)
    number_inpatient = st.number_input("Inpatient Visits",
                                        min_value=0, max_value=50,
                                        value=pv["inpatient"], step=1)
    insulin          = st.selectbox("Insulin",
                                     ["No", "Steady", "Up", "Down"],
                                     index=["No", "Steady", "Up",
                                            "Down"].index(pv["insulin"]))

# Hidden values filled from presets
number_outpatient = pv["outpatient"]
num_procedures    = pv["procedures"]
number_diagnoses  = pv["diagnoses"]

st.markdown("<br>", unsafe_allow_html=True)


# ── Encode Inputs — uses real medians as defaults ──────────────────────────
def encode_inputs():
    # Maps for categorical fields
    med_map  = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    a1c_map  = {"Not measured": 0, "Normal": 1, ">7": 2, ">8": 3}

    # Start with REAL medians from training data
    # Any feature not explicitly set below gets a realistic average value
    # This is the key fix — prevents always-YES predictions
    input_data = {
        feat: float(feature_medians.get(feat, 0))
        for feat in feature_names
    }

    # Override with what the receptionist entered
    input_data["age"]                = age
    input_data["gender"]             = 1 if gender == "Male" else 0
    input_data["time_in_hospital"]   = time_in_hospital
    input_data["num_lab_procedures"] = num_lab_procedures
    input_data["num_medications"]    = num_medications
    input_data["A1Cresult"]          = a1c_map[A1Cresult]
    input_data["number_emergency"]   = number_emergency
    input_data["number_inpatient"]   = number_inpatient
    input_data["number_outpatient"]  = number_outpatient
    input_data["num_procedures"]     = num_procedures
    input_data["number_diagnoses"]   = number_diagnoses
    input_data["insulin"]            = med_map[insulin]

    # Engineered features — must be recalculated from inputs
    input_data["total_visits"]   = (number_outpatient +
                                    number_emergency +
                                    number_inpatient)
    input_data["polypharmacy"]   = 1 if num_medications > 10 else 0
    input_data["lab_proc_ratio"] = num_lab_procedures / (num_procedures + 1)
    input_data["med_changed"]    = int(input_data.get("change", 0))

    return pd.DataFrame([input_data], columns=feature_names)


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

    p0 = round(float(probabilities[0]) * 100, 1)   # NO
    p1 = round(float(probabilities[1]) * 100, 1)   # >30 days
    p2 = round(float(probabilities[2]) * 100, 1)   # <30 days

    # YES = any readmission (class 1 or 2)
    # NO  = not readmitted  (class 0)
    will_readmit = prediction in [1, 2]
    confidence   = max(p0, p1, p2)

    # Save to session history
    st.session_state.history.append({
        "Name"        : patient_name.strip() or "Unknown",
        "ID"          : patient_id.strip()   or "N/A",
        "Date"        : str(visit_date),
        "Doctor"      : doctor_name.strip()  or "N/A",
        "Ward"        : ward.strip()         or "N/A",
        "Readmitted?" : "Yes" if will_readmit else "No",
        "Confidence"  : f"{confidence}%"
    })

    st.markdown("---")

    # ── OUTPUT — Clean Yes / No + Confidence ──────────────────────────────
    st.markdown('<div class="sec-head">Prediction Result</div>',
                unsafe_allow_html=True)

    ra, rb, rc = st.columns([1, 2, 1])

    with rb:
        if will_readmit:
            st.markdown(f"""
            <div style="
                background: #fef2f2;
                border: 2px solid #fca5a5;
                border-radius: 20px;
                padding: 2.5rem 2rem;
                text-align: center;
            ">
                <div style="font-size:48px; margin-bottom:10px;">⚠️</div>
                <div style="
                    font-size: 12px; font-weight: 700;
                    text-transform: uppercase; letter-spacing:.1em;
                    color: #94a3b8; margin-bottom: 8px;
                ">Will this patient be readmitted?</div>
                <div style="
                    font-size: 60px; font-weight: 800;
                    color: #dc2626; line-height: 1;
                    margin-bottom: 20px;
                ">YES</div>
                <div style="
                    background: white; border-radius: 12px;
                    padding: 14px 24px; display: inline-block;
                    border: 1px solid #fca5a5;
                ">
                    <div style="
                        font-size: 10px; font-weight: 700;
                        text-transform: uppercase; letter-spacing:.06em;
                        color: #94a3b8; margin-bottom: 4px;
                    ">Confidence Level</div>
                    <div style="
                        font-size: 32px; font-weight: 800;
                        color: #dc2626;
                    ">{confidence}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="
                background: #ecfdf5;
                border: 2px solid #86efac;
                border-radius: 20px;
                padding: 2.5rem 2rem;
                text-align: center;
            ">
                <div style="font-size:48px; margin-bottom:10px;">✅</div>
                <div style="
                    font-size: 12px; font-weight: 700;
                    text-transform: uppercase; letter-spacing:.1em;
                    color: #94a3b8; margin-bottom: 8px;
                ">Will this patient be readmitted?</div>
                <div style="
                    font-size: 60px; font-weight: 800;
                    color: #059669; line-height: 1;
                    margin-bottom: 20px;
                ">NO</div>
                <div style="
                    background: white; border-radius: 12px;
                    padding: 14px 24px; display: inline-block;
                    border: 1px solid #86efac;
                ">
                    <div style="
                        font-size: 10px; font-weight: 700;
                        text-transform: uppercase; letter-spacing:.06em;
                        color: #94a3b8; margin-bottom: 4px;
                    ">Confidence Level</div>
                    <div style="
                        font-size: 32px; font-weight: 800;
                        color: #059669;
                    ">{confidence}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
        st.markdown(sc("Age",              f"{age} years"),        unsafe_allow_html=True)
        st.markdown(sc("Gender",           gender),                unsafe_allow_html=True)
        st.markdown(sc("Days in Hospital", f"{time_in_hospital}"), unsafe_allow_html=True)

    with sm3:
        st.markdown(sc("Lab Procedures", num_lab_procedures),      unsafe_allow_html=True)
        st.markdown(sc("Medications",    num_medications),         unsafe_allow_html=True)
        st.markdown(sc("HbA1c",          A1Cresult),               unsafe_allow_html=True)

    with sm4:
        st.markdown(sc("Emergency Visits", number_emergency),      unsafe_allow_html=True)
        st.markdown(sc("Inpatient Visits", number_inpatient),      unsafe_allow_html=True)
        st.markdown(sc("Insulin",          insulin),               unsafe_allow_html=True)

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
Will Be Readmitted : {"YES" if will_readmit else "NO"}
Confidence         : {confidence}%

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
        label               = "📄 Download Patient Report",
        data                = report,
        file_name           = f"report_{patient_id or 'patient'}_{visit_date}.txt",
        mime                = "text/plain",
        use_container_width = True
    )


# ── Prediction History ─────────────────────────────────────────────────────
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown('<div class="sec-head">Prediction History — This Session</div>',
                unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history)

    def colour_readmit(val):
        if val == "Yes":
            return "background-color:#fef2f2; color:#991b1b; font-weight:600"
        else:
            return "background-color:#ecfdf5; color:#065f46; font-weight:600"

    styled = history_df.style.map(colour_readmit, subset=["Readmitted?"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv = history_df.to_csv(index=False).encode("utf-8")

    h1, h2 = st.columns(2)
    with h1:
        st.download_button(
            label               = "📥 Download History as CSV",
            data                = csv,
            file_name           = f"history_{date.today()}.csv",
            mime                = "text/csv",
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