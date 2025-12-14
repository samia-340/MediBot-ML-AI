# streamlit_app.py
import os
import asyncio
import tempfile
from pathlib import Path
from PIL import Image
import streamlit as st
from medibot_a2a import MediBotA2ASystem

# ------------------------------------------------------------------
# Helper to run async code inside Streamlit
# ------------------------------------------------------------------
def run_pipeline(image_path: str):
    system = MediBotA2ASystem()
    result = asyncio.run(system.process_prescription(image_path))
    return result, system


# ------------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="🩺 MediBot – Prescription Analyzer",
    page_icon="💊",
    layout="wide"
)

st.title("🩺 MediBot – AI Prescription Analyzer")
st.caption("Upload a prescription image → YOLO → OCR → ChromaDB → Groq LLM → Full structured output")

# Sidebar
st.sidebar.header("⚙️ Settings")
show_logs = st.sidebar.checkbox("Show Agent Logs", value=False)
show_raw_ocr = st.sidebar.checkbox("Show raw OCR results", value=False)

# Optional Groq API key injection
st.sidebar.subheader("🔑 Groq API Key (optional)")
groq_key = st.sidebar.text_input(
    "Enter GROQ_API_KEY",
    type="password",
    placeholder="sk-xxxxxxxx"
)

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
    st.sidebar.success("API key configured")

st.markdown("---")


# ------------------------------------------------------------------
# File uploader
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload prescription image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Preview
    col_preview, col_info = st.columns([2, 1])
    with col_preview:
        st.image(uploaded_file, caption="Uploaded Prescription", use_column_width=True)

    # Save temporary file
    temp_dir = tempfile.mkdtemp(prefix="medibot_")
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with col_info:
        st.info("Press the button below to start processing.")
        process_button = st.button("🚀 Run MediBot Pipeline", use_container_width=True)

    # ------------------------------------------------------------------
    # PROCESSING PIPELINE
    # ------------------------------------------------------------------
    if process_button:
        with st.spinner("Running MediBot Agents… please wait 5–10s"):
            try:
                result, system = run_pipeline(image_path)
            except Exception as e:
                st.error(f"Pipeline crashed: {e}")
                st.stop()

        if not result.get("success", False):
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            st.json(result)
            st.stop()

        prescription = result["prescription_data"]

        # ------------------------------------------------------------------
        # RESULT TABS
        # ------------------------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "💊 Medicines",
            "📄 Smart Summary",
            "🧾 Raw OCR",
            "🔎 Agent Logs"
        ])

        # ------------------------------------------------------------------
        # TAB 1: VERIFIED MEDICINES
        # ------------------------------------------------------------------
        with tab1:
            st.subheader("💊 Verified Medicines")
            meds = prescription.get("medicines", [])

            if len(meds) == 0:
                st.warning("No medicines detected.")
            else:
                for idx, med in enumerate(meds, 1):
                    name = med.get("name", "")
                    orig = med.get("original_text", "")
                    st.markdown(f"""
                    **{idx}. {name}**  
                    <span style='color:gray'>OCR: {orig}</span>
                    """, unsafe_allow_html=True)

                conf = prescription.get("confidence_score", 0)
                st.progress(conf)
                st.info(f"Confidence Score: **{conf*100:.1f}%**")

                if prescription.get("warnings"):
                    st.warning("⚠️ Verification Warnings:")
                    for w in prescription["warnings"]:
                        st.write(f"- {w}")
                else:
                    st.success("No verification warnings.")

        # ------------------------------------------------------------------
        # TAB 2: SMART SUMMARY
        # ------------------------------------------------------------------
        with tab2:
            st.subheader("🧠 LLM Smart Summary (Groq)")

            smart = prescription.get("smart_summary")
            if not smart:
                st.info("Smart summary is empty (Groq failed or CSV missing).")
            else:
                st.markdown(smart)

            st.markdown("### Diagnosis")
            st.write(prescription.get("diagnosis", "N/A"))

        # ------------------------------------------------------------------
        # TAB 3: RAW OCR AND CSV DETAILS
        # ------------------------------------------------------------------
        with tab3:
            st.subheader("📄 Raw OCR Output")

            ocr_results = getattr(system.state, "ocr_results", {})
            ocr_meds = ocr_results.get("medicines", [])
            ocr_details = ocr_results.get("details", {})

            st.markdown("### 🧾 Medicine OCR (Uncorrected)")
            if not ocr_meds:
                st.info("No raw OCR medicine data.")
            else:
                for i, med in enumerate(ocr_meds, 1):
                    st.write(f"**{i}.** `{med.get('text')}`")
                    crop_path = med.get("crop_path")
                    if crop_path and Path(crop_path).exists():
                        st.image(crop_path, width=200)

            st.markdown("---")
            st.markdown("### 🧾 Patient Details OCR (CSV)")
            if not ocr_details:
                st.info("No CSV OCR data found.")
            else:
                st.json(ocr_details)

        # ------------------------------------------------------------------
        # TAB 4: AGENT LOGS
        # ------------------------------------------------------------------
        with tab4:
            st.subheader("🔎 Internal Agent Logs")
            all_logs = result.get("agent_logs", {})

            for role, logs in all_logs.items():
                with st.expander(f"🧩 {role.upper()}"):
                    for line in logs:
                        st.text(line)

        # ------------------------------------------------------------------
        # END OF RESULT UI
        # ------------------------------------------------------------------

else:
    st.info("Upload a prescription image to begin.")
