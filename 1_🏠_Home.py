import streamlit as st
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="DSR Microplastics Platform",
    page_icon="🌊",
    layout="wide"
)

# --- Title ---
st.title("🌊 DSR Smart Microplastics Analysis Platform")

# --- Platform Introduction ---
st.markdown("""
**DSR Smart Microplastics Analysis Platform** is dedicated to providing researchers with a one-stop solution for data interpretation and risk assessment. The platform supports rapid import of detection data and enables intuitive visualization of the spatiotemporal distribution patterns of microplastics, as well as their physicochemical composition (color, particle size, shape, and polymer type). 

**Key features include:**
* **One-click generation** of input files compliant with the multi-fingerprint PMF source apportionment model and corresponding uncertainty calculations.
* **Intelligent recommendation** of pollution factor quantities based on principal component analysis (PCA).
* **Integration of ecological risk indices** with machine learning algorithms (e.g., LightGBM) for accurate risk quantification and identification of key driving factors.

The DSR platform achieves a full-chain analysis from distribution and source apportionment to risk assessment, and is applicable to microplastics research in various environmental media such as water, soil, and air.
""")

st.markdown("---")
st.subheader("Data Upload")
st.markdown("Please upload your data table (Table 1) below to begin.")

# --- Data Upload ---
uploaded_file = st.file_uploader(
    "Upload your data (Excel or CSV)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        # --- Save to Session State ---
        st.session_state['raw_data'] = df_raw

        st.markdown("### Data Preview (Table 1: Raw Data)")
        st.dataframe(df_raw.head())

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.warning("Please ensure your file format is correct (Single header row expected).")

else:
    if 'raw_data' in st.session_state:
        st.info("Data is loaded. You can navigate to other modules using the sidebar.")