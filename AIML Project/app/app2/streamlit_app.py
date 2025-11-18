import streamlit as st
import json
import os
from datetime import datetime
import subprocess
import re
from typing import Dict, Optional, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompt_template2 import PROMPT

# --------------------------------------------------
# STREAMLIT BASIC CONFIG
# --------------------------------------------------
st.set_page_config(page_title="CareMonitor RAG", layout="wide")
st.set_option("client.showSidebarNavigation", False)

# --------------------------------------------------
# PAGE STATE
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "current_patient" not in st.session_state:
    st.session_state.current_patient = None

if "ollama_model" not in st.session_state:
    # Default to something reasonably capable
    st.session_state.ollama_model = "llama3.2:3b"

# --------------------------------------------------
# SIMPLE NAV HELPERS
# --------------------------------------------------
def go_to(page: str, patient_id: Optional[int] = None):
    st.session_state.page = page
    st.session_state.current_patient = patient_id

# --------------------------------------------------
# FIXED SIDEBAR CSS
# --------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    padding-top: 5px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: none !important;
}
.nav-btn {
    display: block;
    width: 100%;
    padding: 16px 22px;
    margin-bottom: 12px;
    border-radius: 10px;
    font-size: 20px;            /* ⬅ Bigger text */
    font-weight: 700;           /* ⬅ Bold text */
    color: #1e6bff;
    background-color: white;
    border: 2px solid #1e6bff;
    text-align: center;
    transition: 0.2s ease;
}

.nav-btn:hover {
    background-color: #dbe8ff;
}
.nav-btn-active {
    background-color: #1e6bff !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Make the main content (dashboard area) white */
.main, .block-container, .stApp {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

/* Optional: remove Streamlit’s default padding so it matches your design */
.block-container {
    padding-top: 30px !important;
    padding-left: 40px !important;
    padding-right: 40px !important;
}

/* Ensure all normal text switches to dark mode */
h1, h2, h3, h4, p, span, div {
    color: #1a1a1a !important;
}

/* But DON'T override the colored cards */
.card, .card * {
    color: white !important;
}

[data-testid="stSidebar"] {
    border-right: 2px solid #E5E7EB !important;   /* light grey border */
}


/* Make all text inputs and text areas white with a border */
textarea, input[type="text"], .stTextInput > div > input, .stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #D0D5DD !important;   /* Light grey border */
    border-radius: 8px !important;
}

/* Fix the dark background Streamlit sometimes injects */
.stTextInput > div, .stTextArea > div {
    background-color: #ffffff !important;
}

/* Style all Streamlit buttons (Open, Delete, etc.) */
/* White buttons with blue border + blue text */
.stButton > button {
    background-color: white !important;
    color: #1e6bff !important;
    border: 2px solid #1e6bff !important;
    padding: 8px 20px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: 0.2s ease;
}

/* Hover: blue background, white text */
.stButton > button:hover {
    background-color: #1e6bff !important;
    color: white !important;
    border: 2px solid #1e6bff !important;

}

/* Make Streamlit top header white */
header[data-testid="stHeader"] {
    background-color: white !important;
}

/* Make ONLY the 'Add New Patient' heading blue */
h2:contains("Add New Patient") {
    color: #1570EF !important;
    font-weight: 700 !important;
}


</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# DATA I/O
# --------------------------------------------------
DATA_FILE = "patients.json"

def load_db() -> Dict[str, Any]:
    if not os.path.exists(DATA_FILE):
        return {"patients": [], "alerts": []}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"patients": [], "alerts": []}

def save_db(db: Dict[str, Any]):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

db = load_db()

# --------------------------------------------------
# VITALS PARSING & RISK SCORING
# --------------------------------------------------
def parse_vitals(vitals: str) -> Dict[str, Optional[float]]:
    """
    Very simple heuristic parser for vitals in free text.
    Extracts SpO2, HR, SBP, DBP, and Temp when possible.
    """
    v = vitals.lower()
    result: Dict[str, Optional[float]] = {
        "spo2": None,
        "hr": None,
        "sbp": None,
        "dbp": None,
        "temp": None,
    }

    # SpO2 / oxygen saturation
    # e.g. "SpO2 88%", "O2 sat 90", "oxygen 85 %"
    spo2_match = re.search(r"(spo2|o2|oxygen)[^\d]*(\d{2,3})", v)
    if spo2_match:
        try:
            spo2 = float(spo2_match.group(2))
            if 50 <= spo2 <= 100:
                result["spo2"] = spo2
        except ValueError:
            pass

    # Heart rate
    # e.g. "HR 120", "heart rate 110 bpm"
    hr_match = re.search(r"(hr|heart rate)[^\d]*(\d{2,3})", v)
    if hr_match:
        try:
            hr = float(hr_match.group(2))
            if 30 <= hr <= 220:
                result["hr"] = hr
        except ValueError:
            pass

    # Blood pressure
    # e.g. "BP 150/90", "blood pressure 100/60"
    bp_match = re.search(r"(bp|blood pressure)[^\d]*(\d{2,3})\D+(\d{2,3})", v)
    if bp_match:
        try:
            sbp = float(bp_match.group(2))
            dbp = float(bp_match.group(3))
            if 60 <= sbp <= 260:
                result["sbp"] = sbp
            if 30 <= dbp <= 160:
                result["dbp"] = dbp
        except ValueError:
            pass

    # Temperature
    # e.g. "temp 38.5", "temperature 39", "37°C"
    temp_match = re.search(r"(temp|temperature)[^\d]*(\d{2}(?:\.\d)?)", v)
    if temp_match:
        try:
            temp = float(temp_match.group(2))
            if 30 <= temp <= 43:
                result["temp"] = temp
        except ValueError:
            pass

    # Fallback keywords if no numeric oxygen
    if result["spo2"] is None:
        if "very low oxygen" in v or "severe hypoxia" in v:
            result["spo2"] = 85.0
        elif "low oxygen" in v or "hypoxia" in v:
            result["spo2"] = 89.0

    return result


def compute_risk_from_parsed(parsed: Dict[str, Optional[float]], raw_vitals: str) -> int:
    score = 0

    spo2 = parsed.get("spo2")
    if spo2 is not None:
        if spo2 < 88:
            score += 50
        elif spo2 < 92:
            score += 35
        elif spo2 < 95:
            score += 15

    hr = parsed.get("hr")
    if hr is not None:
        if hr >= 130:
            score += 25
        elif hr >= 110:
            score += 15

    sbp = parsed.get("sbp")
    dbp = parsed.get("dbp")
    if sbp is not None:
        if sbp >= 180:
            score += 20
        elif sbp >= 150:
            score += 10
        elif sbp < 90:
            score += 20
    if dbp is not None and dbp >= 110:
        score += 10

    temp = parsed.get("temp")
    if temp is not None:
        if temp >= 39:
            score += 15
        elif temp >= 38:
            score += 10

    vlow = raw_vitals.lower()
    if "confusion" in vlow or "altered mental" in vlow:
        score += 10

    return min(score, 100)


def risk_status(score: int) -> str:
    if score >= 70:
        return "Critical"
    if score >= 40:
        return "Monitor"
    return "Stable"


def compute_total_risk(parsed: Dict[str, Optional[float]],
                       raw_vitals: str,
                       history_text: str,
                       onset_text: str) -> int:
    """
    Combines numeric vitals-based risk with simple text-based
    risk factors like age, hypertension, and deterioration.
    """
    # Start with the existing numeric-based risk
    score = compute_risk_from_parsed(parsed, raw_vitals)

    h = history_text.lower()
    o = onset_text.lower()
    v = raw_vitals.lower()

    # --------------------------
    # Age-based baseline risk
    # --------------------------
    # Look for an age-like number in the history text.
    # Because history shouldn't contain BP numbers, this is fairly safe.
    age_match = re.search(r"\b(\d{2,3})\b", h)
    if age_match:
        try:
            age = int(age_match.group(1))
            if 75 <= age <= 110:
                score += 15      # elderly, higher baseline risk
            elif 65 <= age < 75:
                score += 8       # older adult
        except ValueError:
            pass

    # --------------------------
    # Hypertension / high BP
    # --------------------------
    if ("high blood pressure" in v or
        "hypertension" in v or
        "hypertensive" in v):
        score += 15

    # --------------------------
    # Deterioration / worsening
    # --------------------------
    if any(kw in o for kw in [
        "deteriorat",        # deteriorating / deterioration
        "worsen",            # worsening
        "getting worse",
        "rapid decline",
        "rapidly declining",
        "rapidly deteriorating"
    ]):
        score += 15

    return min(score, 100)




# --------------------------------------------------
# RAG SETUP
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        persist_directory="chroma_index",
        embedding_function=embeddings,
    )
    # Use MMR with a bit larger k to get more diverse similar cases
    return vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )


def retrieve_cases(text: str) -> str:
    retriever = get_retriever()
    docs = retriever.invoke(text)
    if not docs:
        # Empty string => model will correctly say "Insufficient evidence"
        return ""
    cleaned = []
    for d in docs:
        # Escape curly braces so PROMPT.format() never crashes
        text_clean = d.page_content.replace("{", "{{").replace("}", "}}")
        cleaned.append(text_clean)
    return "\n\n---\n\n".join(cleaned)

# --------------------------------------------------
# LLAMA / OLLAMA CALL
# --------------------------------------------------
# NOTE: adjust this path if Ollama is elsewhere on your system
OLLAMA_EXE = r"C:\Users\sofyk\AppData\Local\Programs\Ollama\ollama.exe"

def ask_ollama(prompt: str) -> str:
    model_name = st.session_state.get("ollama_model", "llama3.2:3b")
    try:
        result = subprocess.run(
            [OLLAMA_EXE, "run", model_name],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        output = result.stdout.decode("utf-8", errors="ignore")
        return output
    except Exception as e:
        return f"Error calling Ollama: {e}"

# --------------------------------------------------
# PATIENT SAVE / DB LOGIC
# --------------------------------------------------
def add_patient(name: str, symptom: str, history: str,
                onset: str, vitals: str, ai_summary: str) -> int:
    summary = (
        f"Symptom: {symptom}. "
        f"History: {history}. "
        f"Onset: {onset}. "
        f"Vitals: {vitals}."
    )

    parsed_vitals = parse_vitals(vitals)
    # NEW: use age + deterioration + hypertension text
    risk = compute_total_risk(parsed_vitals, vitals, history, onset)
    status = risk_status(risk)

    record = {
        "id": len(db["patients"]) + 1,
        "name": name,
        "symptom": symptom,
        "history": history,
        "onset": onset,
        "vitals": vitals,
        "parsed_vitals": parsed_vitals,
        "summary": summary,
        "risk": risk,
        "status": status,
        "ai_summary": ai_summary,
        "notes": [],
        "timestamp": datetime.utcnow().isoformat()
    }

    db["patients"].append(record)
    save_db(db)
    return record["id"]

# --------------------------------------------------
# SIDEBAR NAVIGATION + SETTINGS
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <h1 style='color:#144b8b; font-size:50px; font-weight:800; margin-bottom:25px;'>
        🩺 CareMonitor
    </h1>
    """, unsafe_allow_html=True)

    def nav_button(label: str, target: str):
        active = "nav-btn-active" if st.session_state.page == target else ""
        if st.button(label, key=f"nav_{target}", use_container_width=True):
            st.session_state.page = target
        # Apply active style via DOM id of the button
        st.markdown(
            f"<style>div[data-testid='baseButton-secondary'][id='nav_{target}'] {{{active}}}</style>",
            unsafe_allow_html=True
        )

    nav_button("Dashboard", "Dashboard")
    nav_button("Patients", "Patients")
    nav_button("Alerts", "Alerts")

    st.markdown("---")
    st.markdown("**Model Settings**")
    st.session_state.ollama_model = st.selectbox(
        "Ollama model",
        options=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
        index=1,  # default to 3b
        help="Use at least 3B for reasonable clinical reasoning."
    )

# --------------------------------------------------
# PAGE: DASHBOARD
# --------------------------------------------------
if st.session_state.page == "Dashboard":
    st.markdown("""
        <h1 style='color: #ffffff; font-size: 40px;'>Dashboard</h1>
        <p style='color: #9aa0a6; font-size: 18px; margin-top: -10px;'>
            Real-time patient monitoring overview
        </p>
        <br>
        <style>
            .card {
                padding: 25px;
                border-radius: 18px;
                margin-bottom: 20px;
                color: white;
                font-family: sans-serif;
                text-align: left;
            }
            .card h2 {
                font-size: 40px;
                margin-bottom: -5px;
            }
            .card p {
                font-size: 15px;
                margin-top: -10px;
                opacity: 0.8;
            }
        </style>
    """, unsafe_allow_html=True)

    patients = db["patients"]
    total = len(patients)
    high = len([p for p in patients if p["risk"] >= 70])
    moderate = len([p for p in patients if 40 <= p["risk"] < 70])
    low = len([p for p in patients if p["risk"] < 40])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='card' style='background-color:#1e88e5;'>
                <h2>{total}</h2>
                <strong>Total Inpatients</strong>
                <p>Active admissions</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='card' style='background-color:#e53935;'>
                <h2>{high}</h2>
                <strong>High Risk (12h)</strong>
                <p>Requires attention</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class='card' style='background-color:#fb8c00;'>
                <h2>{moderate}</h2>
                <strong>New Alerts (24h)</strong>
                <p>Across all units</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='card' style='background-color:#43a047;'>
                <h2>{low}</h2>
                <strong>Low Risk</strong>
                <p>Stable patients</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3 style='color:white;'>Recent Admissions</h3>", unsafe_allow_html=True)

    # Sort patients by most recent
    patients_sorted = sorted(db["patients"], key=lambda x: x.get("created_at", 0), reverse=True)

    # Limit to last 5 entries (optional)
    recent_patients = patients_sorted[:5]

    for p in recent_patients:
        name = p["name"]
        risk = p.get("risk", 0)

        # Label
        if risk >= 70:
            status_color = "#e53935"  # Red
            status_label = "High Risk"
        elif risk >= 40:
            status_color = "#fb8c00"  # Orange
            status_label = "Monitor"
        else:
            status_color = "#43a047"  # Green
            status_label = "Stable"

        st.markdown(f"""
            <div style="
                padding:15px;
                margin-top:15px;
                border-radius:12px;
                background-color:#ffffff;
                border:2px solid #e5e7eb;    
                border-left:6px solid {status_color};
            ">
                <h4 style="color:white; margin:0;">{name}</h4>
                <p style="color:#9aa0a6; margin-top:2px;">
                    Risk {risk}% — <strong style="color:{status_color};">{status_label}</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)





# --------------------------------------------------
# PAGE: PATIENTS
# --------------------------------------------------
# PAGE: PATIENTS
# --------------------------------------------------
elif st.session_state.page == "Patients":
    st.title("Patients")

    st.subheader("Add New Patient")
    with st.form("new_patient_form"):
        name = st.text_input("Name")
        symptom = st.text_area("Main Symptom")
        history = st.text_area("Patient History (age, sex, comorbidities)")
        onset = st.text_area("Symptom Onset & Evolution")
        vitals = st.text_area("Vitals (HR, SpO₂, BP, Temp, etc)")

        submitted = st.form_submit_button("Generate AI Summary & Save")

    if submitted:
        if not name or not symptom:
            st.error("Name and main symptom are required.")
        else:
            patient_text = (
                f"Symptom: {symptom}\n"
                f"History: {history}\n"
                f"Onset: {onset}\n"
                f"Vitals: {vitals}\n"
            )

            with st.spinner("Retrieving similar cases and calling model..."):
                cases = retrieve_cases(patient_text)
                final_prompt = PROMPT.format(patient=patient_text, cases=cases)
                ai_summary = ask_ollama(final_prompt)

            new_id = add_patient(name, symptom, history, onset, vitals, ai_summary)
            st.success("Patient added successfully!")
            go_to("PatientDetail", new_id)

    st.subheader("Patient List")
    if not db["patients"]:
        st.info("No patients in the system.")
    else:

        # FIX: add enumerate so keys are always unique
        for i, p in enumerate(db["patients"]):

            st.write(f"**{p['name']}** — Risk {p['risk']}% ({p['status']})")

            cols = st.columns([1, 1])

            # FIXED unique button key
            if cols[0].button(
                f"Open → {p['name']}",
                key=f"open_list_{p['id']}_{i}"
            ):
                go_to("PatientDetail", p["id"])

            # FIXED unique delete key
            if cols[1].button(
                "Delete",
                key=f"delete_{p['id']}_{i}"
            ):
                db["patients"] = [q for q in db["patients"] if q["id"] != p["id"]]
                save_db(db)
                st.warning(f"Deleted patient {p['name']}")
                st.rerun()


# --------------------------------------------------
# PAGE: PATIENT DETAIL
# --------------------------------------------------
elif st.session_state.page == "PatientDetail":
    pid = st.session_state.current_patient
    if pid is None:
        st.error("No patient selected.")
    else:
        patient = next((p for p in db["patients"] if p["id"] == pid), None)
        if not patient:
            st.error("Patient not found.")
        else:
            st.title(f"Patient: {patient['name']}")

            parsed_vitals = patient.get("parsed_vitals") or parse_vitals(patient["vitals"])
            risk = patient.get("risk", 0)
            status = patient.get("status", risk_status(risk))

            st.subheader("Vital Signs & Risk")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("SpO₂", f"{parsed_vitals.get('spo2', '—')}")
            col2.metric("HR", f"{parsed_vitals.get('hr', '—')}")
            sbp = parsed_vitals.get("sbp")
            dbp = parsed_vitals.get("dbp")
            bp_str = "—"
            if sbp is not None and dbp is not None:
                bp_str = f"{int(sbp)}/{int(dbp)}"
            col3.metric("BP", bp_str)
            col4.metric("Temp", f"{parsed_vitals.get('temp', '—')}")
            col5.metric("Risk", f"{risk}% ({status})")

            st.caption(f"Raw vitals text: _{patient['vitals']}_")

            st.subheader("AI Diagnosis Summary")
            ai_text = patient["ai_summary"]
            if any(phrase in ai_text.lower() for phrase in [
                "as an ai", "i can't assist", "cannot assist", "outside my guidelines"
            ]):
                st.warning(
                    "The model partially refused or produced meta-text. "
                    "Consider simplifying the input or checking the RAG cases."
                )
            st.write(ai_text)

            st.subheader("Clinical Notes")
            if len(patient["notes"]) == 0:
                st.info("No notes yet.")
            else:
                for n in patient["notes"]:
                    st.write(f"- {n}")

            new_note = st.text_area("Add Note", key="note_input")
            if st.button("Save Note"):
                if new_note.strip():
                    patient["notes"].append(new_note.strip())
                    save_db(db)
                    st.success("Note saved.")
                else:
                    st.error("Note is empty.")

# --------------------------------------------------
# PAGE: ALERTS
# --------------------------------------------------
elif st.session_state.page == "Alerts":
    st.title("Alerts")

    alerts = []
    for p in db["patients"]:
        if p["risk"] >= 70:
            alerts.append(f"⚠ Critical alert for {p['name']} — Risk {p['risk']}%")

    if not alerts:
        st.info("No active alerts.")
    else:
        for a in alerts:
            st.error(a)







