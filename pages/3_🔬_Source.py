import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Source Analysis", layout="wide")
st.title("🔬 Module 2: Source Apportionment Pre-processing")

if 'raw_data' not in st.session_state:
    st.error("Please upload data on the Home page first.")
else:
    df = st.session_state['raw_data'].copy()
    
    # --- 1. Concentration Matrix ---
    st.markdown("### 1. Generate Concentration Matrix (Table 2)")
    st.info("Reconstructing data into the format required for Source Apportionment (Concentration Matrix).")
    
    try:
        df_conc = pd.DataFrame()
        
        # Size Classification
        # 0-1mm = <100um + 100-500um + 500-1000um
        df_conc['Size_0-1mm'] = df['＜100um'] + df['100um-500um'] + df['500um-1000um']
        df_conc['Size_1-5mm'] = df['1000um-5000um']
        
        # Color Classification
        # Colored = Brown + Blue + Red + Yellow
        df_conc['Color_White'] = df['White']
        df_conc['Color_Black'] = df['Black']
        df_conc['Color_Colored'] = df['Brown'] + df['Blue'] + df['Red'] + df['Yellow']
        
        # Polymer (Retain specific types)
        df_conc['Poly_PET'] = df['PET']
        df_conc['Poly_PS'] = df['PS']
        df_conc['Poly_PP'] = df['PP']
        df_conc['Poly_PE'] = df['PE']
        
        # Shape (Retain all)
        df_conc['Shape_Fibre'] = df['Fibre']
        df_conc['Shape_Pellet'] = df['Pellet']
        df_conc['Shape_Film'] = df['Film']
        df_conc['Shape_Fragment'] = df['Fragment']
        
        # Keep Site info
        df_conc['Sites'] = df['Sites']
        
        st.success("Concentration Matrix generated successfully!")
        st.dataframe(df_conc.head())
        
        # Save to session state for other modules if needed
        st.session_state['table2_data'] = df_conc

        st.markdown("---")
        
        # --- 2. Uncertainty Matrix ---
        st.markdown("### 2. Generate Uncertainty Matrix (Table 3)")
        st.caption("Uncertainty Matrix = Concentration Matrix values × 0.2 (Simulating uncertainty calculation for PMF input)")
        
        # Multiply numeric columns by 0.2
        numeric_cols = df_conc.select_dtypes(include=np.number).columns
        df_unc = df_conc.copy()
        df_unc[numeric_cols] = df_unc[numeric_cols] * 0.2
        
        st.success("Uncertainty Matrix generated successfully!")
        st.dataframe(df_unc.head())
        
        st.markdown("---")
        
        # --- 3. PCA ---
        st.markdown("### 3. Principal Component Analysis (PCA)")
        st.info("PCA helps intelligent recommendation of pollution factor quantities based on the Concentration Matrix.")
        
        # Prepare data for PCA (Exclude 'Sites')
        pca_features = df_conc.drop(columns=['Sites'])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(pca_features)

        # User input for variance
        var_target = st.slider("Select Target Explained Variance (%)", 50, 100, 80, 5)
        
        pca = PCA(n_components=var_target / 100.0)
        pca_result = pca.fit_transform(features_scaled)
        
        n_comps = pca.n_components_
        exp_var = pca.explained_variance_ratio_.sum() * 100
        
        st.write(f"To retain **{var_target}%** variance, **{n_comps}** principal components are selected.")
        st.write(f"Actual Cumulative Variance: **{exp_var:.2f}%**")
        
        # PCA Scatter Plot (PC1 vs PC2)
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_comps)])
        pca_df['Sites'] = df['Sites']
        
        if n_comps >= 2:
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Sites', title='PCA Score Plot (PC1 vs PC2)')
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.warning("Less than 2 components selected. Cannot display 2D scatter plot.")

        # --- 4. PMF External Link ---
        st.markdown("---")
        st.subheader("4. PMF Model Resource")
        
        st.success(
            """
            **Data Pre-processing Complete.**
            
            This module has completed the data cleaning and uncertainty calculation steps required for source apportionment.
            The generated **Concentration Matrix** and **Uncertainty Matrix** are intended for use with the **Positive Matrix Factorization (PMF)** model for subsequent source analysis.
            """
        )
        
        st.markdown(
            """
            👉 **[Visit US EPA PMF Model Website](https://www.epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses)**
            """
        )
            
    except Exception as e:
        st.error(f"Source Analysis Error: {e}")
        st.warning("Please check if column names in Table 1 are correct.")
